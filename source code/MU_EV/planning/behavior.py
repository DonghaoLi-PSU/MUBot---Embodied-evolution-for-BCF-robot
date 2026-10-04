# -*- coding: utf-8 -*-
"""
Behavior layer: turns a mission (ordered waypoints) into one primitive at a time.

Closed loop around the hybrid-A* planner, because swimming is far too noisy to
execute a whole plan open-loop:

    NAVIGATE  plan to the current waypoint, execute the first few primitives,
              replan after `replan_every` steps, when the measured pose drifts
              from the expected one, or when the plan runs out.
    RECOVER   entered when the planner finds nothing (start boxed in) or no
              progress is made for `stall_window` steps; executes a few
              clearance-maximising primitives (backing off first if possible),
              then returns to NAVIGATE.
    DONE      all waypoints reached.
    FAILED    too many recoveries on one waypoint, or the step budget ran out.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Sequence

import numpy as np

from MU_EV.planning.planner import Goal, HybridAStarPlanner, Plan, PlannerConfig
from MU_EV.planning.primitives import BACKWARD, PrimitiveLibrary, wrap_angle
from MU_EV.planning.world import Workspace


class Mode(Enum):
    NAVIGATE = "navigate"
    RECOVER = "recover"
    DONE = "done"
    FAILED = "failed"


@dataclass
class BehaviorConfig:
    replan_every: int = 2              # primitives executed per plan before replanning
    deviation_xy: float = 0.03         # [m] replan early beyond this position error
    deviation_theta: float = np.deg2rad(20)
    stall_window: int = 10             # steps without progress before recovering
    stall_eps: float = 0.02            # [m] progress that counts
    recovery_steps: int = 3
    max_recoveries: int = 4            # per waypoint


class Executor(Protocol):
    """Anything that can run one primitive and report the new CoG pose."""
    def execute(self, primitive: str) -> np.ndarray: ...


class BehaviorPlanner:

    def __init__(self, library: PrimitiveLibrary, workspace: Workspace, waypoints: Sequence[Goal],
                 planner_config: Optional[PlannerConfig] = None,
                 config: Optional[BehaviorConfig] = None):
        self.lib = library
        self.ws = workspace
        self.planner = HybridAStarPlanner(library, workspace, planner_config)
        self.cfg = config or BehaviorConfig()
        self.waypoints = list(waypoints)
        self.reset()

    def reset(self):
        self.mode = Mode.NAVIGATE
        self.wp_idx = 0
        self.plan: Optional[Plan] = None
        self.cursor = 0
        self.last: Optional[str] = None
        self.recovery_queue: List[str] = []
        self.recoveries = 0
        self.best_dist = np.inf
        self.since_progress = 0
        self.events: List[str] = []

    @property
    def goal(self) -> Optional[Goal]:
        return self.waypoints[self.wp_idx] if self.wp_idx < len(self.waypoints) else None

    def _log(self, step, msg):
        self.events.append(f"[{step:4d}] {msg}")

    # ---- main entry ---------------------------------------------------------
    def next_primitive(self, pose, step=0) -> Optional[str]:
        if self.mode in (Mode.DONE, Mode.FAILED):
            return None
        pose = np.asarray(pose, dtype=float)

        while self.goal is not None and self.goal.reached(pose):
            self._log(step, f"waypoint {self.wp_idx} reached at ({pose[0]:.3f}, {pose[1]:.3f})")
            self.wp_idx += 1
            self.plan, self.recoveries = None, 0
            self.best_dist, self.since_progress = np.inf, 0
        if self.goal is None:
            self.mode = Mode.DONE
            return None

        # progress monitor
        dist = np.hypot(self.goal.x - pose[0], self.goal.y - pose[1])
        if dist < self.best_dist - self.cfg.stall_eps:
            self.best_dist, self.since_progress = dist, 0
        else:
            self.since_progress += 1
        if self.mode is Mode.NAVIGATE and self.since_progress >= self.cfg.stall_window:
            self._start_recovery(step, "no progress")

        if self.mode is Mode.RECOVER:
            prim = self._recovery_step(pose)
            if prim is not None:
                return self._emit(prim)
            self.mode, self.plan = Mode.NAVIGATE, None
            if self.recoveries > self.cfg.max_recoveries:
                self.mode = Mode.FAILED
                self._log(step, f"FAILED on waypoint {self.wp_idx}")
                return None

        if self._needs_replan(pose):
            self.plan = self.planner.plan(pose, self.goal, last_primitive=self.last)
            self.cursor = 0
            self._log(step, f"plan: {len(self.plan)} prims, reached={self.plan.reached_goal}, "
                            f"exp={self.plan.expansions}, {' '.join(self.plan.primitives[:6])}")
            if len(self.plan) == 0:
                self._start_recovery(step, "planner found no move")
                return self.next_primitive(pose, step)

        prim = self.plan.primitives[self.cursor]
        self.cursor += 1
        return self._emit(prim)

    # ---- internals ----------------------------------------------------------
    def _emit(self, prim):
        self.last = prim
        return prim

    def _needs_replan(self, pose):
        if self.plan is None or self.cursor >= len(self.plan) or self.cursor >= self.cfg.replan_every:
            return True
        exp = self.plan.poses[self.cursor]
        return (np.hypot(*(pose[:2] - exp[:2])) > self.cfg.deviation_xy
                or abs(wrap_angle(pose[2] - exp[2])) > self.cfg.deviation_theta)

    def _start_recovery(self, step, why):
        self.recoveries += 1
        self.since_progress = 0
        self.best_dist = np.inf
        self.mode = Mode.RECOVER
        self.recovery_queue = [None] * self.cfg.recovery_steps   # filled greedily per step
        self._log(step, f"RECOVER ({why}), attempt {self.recoveries}")

    def _recovery_step(self, pose):
        if not self.recovery_queue:
            return None
        self.recovery_queue.pop()
        best, best_score = None, -np.inf
        for prim in self.lib:
            score = self.ws.clearance(self.planner.sweep(pose, prim)).min()
            if prim.name == BACKWARD:
                score += 0.01      # prefer backing straight out when it is about as good
            if score > best_score:
                best, best_score = prim.name, score
        return best


# --------------------------------------------------------------------------- #
@dataclass
class MissionLog:
    poses: List[np.ndarray] = field(default_factory=list)
    primitives: List[str] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)
    collided: bool = False
    final_mode: Mode = Mode.NAVIGATE


def run_mission(behavior: BehaviorPlanner, executor: Executor, start_pose, max_steps=300) -> MissionLog:
    log = MissionLog(poses=[np.asarray(start_pose, dtype=float)])
    pose = log.poses[0]
    for step in range(max_steps):
        prim = behavior.next_primitive(pose, step)
        if prim is None:
            break
        log.modes.append(behavior.mode.value)
        pose = executor.execute(prim)
        log.primitives.append(prim)
        log.poses.append(pose)
        if getattr(executor, "collided", False):
            log.collided = True
            behavior._log(step, "COLLISION reported by executor")
            break
    else:
        behavior.mode = Mode.FAILED
        behavior._log(max_steps, "step budget exhausted")
    log.final_mode = behavior.mode
    return log
