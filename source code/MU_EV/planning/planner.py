# -*- coding: utf-8 -*-
"""
Hybrid-A* search over sequences of motion primitives.

Nodes carry the continuous pose reached by chaining primitive displacements
(no snapping to a lattice, so turn angles need not divide 2*pi), and are
de-duplicated on a discretised key (x, y, heading bin, last primitive).  The
last primitive is part of the key because switching costs depend on it.

Edge cost  = duration
           + switch penalty (primitive change)
           + w_uncertainty * position std of the primitive
           + w_clearance   * mean intrusion into the soft clearance band
Heuristic  = max(distance / v_max, heading error / omega_max)  (admissible in time)

The position covariance is propagated along each branch, and collision checks
inflate the robot by n_sigma standard deviations, so long open-loop chains of
noisy primitives (e.g. backward swimming) get kept away from obstacles.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from MU_EV.planning.primitives import PrimitiveLibrary, compose, wrap_angle
from MU_EV.planning.world import Workspace


@dataclass
class Goal:
    x: float
    y: float
    theta: Optional[float] = None   # None: any final heading
    tol_xy: float = 0.05
    tol_theta: float = np.deg2rad(20)

    def reached(self, pose):
        if np.hypot(pose[0] - self.x, pose[1] - self.y) > self.tol_xy:
            return False
        return self.theta is None or abs(wrap_angle(pose[2] - self.theta)) <= self.tol_theta


@dataclass
class Plan:
    primitives: List[str]
    poses: np.ndarray            # (len+1, 3) expected poses, starting at the query pose
    covs: np.ndarray             # (len+1, 3, 3) open-loop pose covariance
    cost: float
    reached_goal: bool           # False -> best partial plan towards the goal
    expansions: int

    def __len__(self):
        return len(self.primitives)


@dataclass
class PlannerConfig:
    xy_res: float = 0.02
    theta_bins: int = 24
    max_expansions: int = 8000
    heuristic_weight: float = 1.5     # cost <= 1.5x optimal; 1.0 is exact but ~10x slower
    w_uncertainty: float = 5.0        # [s per m of std]
    w_clearance: float = 5.0          # [s per m of intrusion]
    soft_clearance: float = 0.05      # [m] preferred free margin
    n_sigma: float = 2.0              # collision inflation in position std
    min_margin: float = 0.01          # [m] hard margin on top of the inflation
    max_inflation: float = 0.05       # cap so far-horizon branches are not all blocked
    n_waypoints: int = 6              # collision samples per primitive (when no path)


@dataclass(order=True)
class _Node:
    f: float
    idx: int = field(compare=False)


class HybridAStarPlanner:

    def __init__(self, library: PrimitiveLibrary, workspace: Workspace,
                 config: Optional[PlannerConfig] = None):
        self.lib = library
        self.ws = workspace
        self.cfg = config or PlannerConfig()
        self.v_max = max(library.max_speed, 1e-6)
        self.w_max = max(library.max_turn_rate, 1e-6)

    # ---- helpers -----------------------------------------------------------
    def _key(self, pose, last):
        c = self.cfg
        ith = int(np.floor((pose[2] + np.pi) / (2 * np.pi) * c.theta_bins)) % c.theta_bins
        return (int(np.floor(pose[0] / c.xy_res)), int(np.floor(pose[1] / c.xy_res)), ith, last)

    def heuristic(self, pose, goal: Goal):
        d = max(np.hypot(goal.x - pose[0], goal.y - pose[1]) - goal.tol_xy, 0.0)
        h = d / self.v_max
        if goal.theta is not None:
            dth = max(abs(wrap_angle(goal.theta - pose[2])) - goal.tol_theta, 0.0)
            h = max(h, dth / self.w_max)
        return h

    @staticmethod
    def propagate_cov(pose, cov, prim):
        """First-order covariance of compose(pose, delta) with noisy pose and delta."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        dx, dy = prim.delta[0], prim.delta[1]
        Jx = np.array([[1, 0, -s * dx - c * dy], [0, 1, c * dx - s * dy], [0, 0, 1]])
        Ju = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return Jx @ cov @ Jx.T + Ju @ prim.cov @ Ju.T

    def sweep(self, pose, prim):
        """World-frame (K,2) points traversed while executing prim from pose."""
        wp = prim.waypoints(self.cfg.n_waypoints)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return np.column_stack([pose[0] + c * wp[:, 0] - s * wp[:, 1],
                                pose[1] + s * wp[:, 0] + c * wp[:, 1]])

    # ---- search ------------------------------------------------------------
    def plan(self, start, goal: Goal, start_cov=None, last_primitive=None) -> Plan:
        cfg = self.cfg
        start = np.asarray(start, dtype=float)
        poses = [start]
        covs = [np.zeros((3, 3)) if start_cov is None else np.asarray(start_cov)]
        g = [0.0]
        parent = [-1]
        prim_of = [last_primitive]
        best_g = {self._key(start, last_primitive): 0.0}
        h0 = self.heuristic(start, goal)
        best_partial, best_h = 0, h0
        open_ = [_Node(cfg.heuristic_weight * h0, 0)]
        closed = set()
        expansions = 0

        while open_ and expansions < cfg.max_expansions:
            i = heapq.heappop(open_).idx
            key = self._key(poses[i], prim_of[i])
            if key in closed:
                continue
            closed.add(key)
            if goal.reached(poses[i]):
                return self._extract(i, poses, covs, g, parent, prim_of, True, expansions)
            expansions += 1

            for prim in self.lib:
                nxt = compose(poses[i], prim.delta)
                cov = self.propagate_cov(poses[i], covs[i], prim)
                sigma = np.sqrt(max(np.linalg.eigvalsh(cov[:2, :2]).max(), 0.0))
                inflate = cfg.min_margin + min(cfg.n_sigma * sigma, cfg.max_inflation)
                clr = self.ws.clearance(self.sweep(poses[i], prim))
                if np.any(clr <= inflate):
                    continue
                pos_std = np.sqrt(max(np.trace(prim.cov[:2, :2]), 0.0))
                step = (prim.duration
                        + self.lib.switch_cost(prim_of[i], prim.name)
                        + cfg.w_uncertainty * pos_std
                        + cfg.w_clearance * np.mean(np.clip(cfg.soft_clearance - clr, 0, None)))
                gn = g[i] + step
                k = self._key(nxt, prim.name)
                if k in closed or gn >= best_g.get(k, np.inf):
                    continue
                best_g[k] = gn
                poses.append(nxt); covs.append(cov); g.append(gn)
                parent.append(i); prim_of.append(prim.name)
                j = len(poses) - 1
                h = self.heuristic(nxt, goal)
                if h < best_h:
                    best_h, best_partial = h, j
                heapq.heappush(open_, _Node(gn + cfg.heuristic_weight * h, j))

        return self._extract(best_partial, poses, covs, g, parent, prim_of, False, expansions)

    @staticmethod
    def _extract(i, poses, covs, g, parent, prim_of, reached, expansions):
        chain = []
        while i >= 0:
            chain.append(i)
            i = parent[i]
        chain.reverse()
        return Plan(primitives=[prim_of[j] for j in chain[1:]],
                    poses=np.array([poses[j] for j in chain]),
                    covs=np.array([covs[j] for j in chain]),
                    cost=g[chain[-1]], reached_goal=reached, expansions=expansions)
