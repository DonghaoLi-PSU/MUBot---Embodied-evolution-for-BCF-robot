# -*- coding: utf-8 -*-
"""
Motion-primitive model for a fixed-morphology muBot.

A primitive is one CPG parameter set (the vector written to control_signal.csv)
executed for a fixed number of tail-beat cycles.  For planning we only need its
*effect*: the planar rigid-body displacement it produces, expressed in the body
frame at the start of the execution, plus how repeatable that displacement is.

    delta = (dx, dy, dtheta)   body-frame displacement over one execution
    cov   = 3x3 covariance of delta across repeated executions
    path  = optional (K,3) body-frame poses inside the execution (collision checks)

Poses are always sampled at the same oscillation phase (cycle boundaries), so the
body undulation cancels out and only the net swimming motion remains.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

FORWARD    = "forward"
TURN_LEFT  = "turn_left"
TURN_RIGHT = "turn_right"
BACKWARD   = "backward"
PRIMITIVE_NAMES = (FORWARD, TURN_LEFT, TURN_RIGHT, BACKWARD)


# --------------------------------------------------------------------------- #
# SE(2) helpers
# --------------------------------------------------------------------------- #
def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def compose(pose, delta):
    """World pose (x, y, th) followed by body-frame displacement (dx, dy, dth)."""
    x, y, th = pose
    dx, dy, dth = delta
    c, s = np.cos(th), np.sin(th)
    return np.array([x + c * dx - s * dy, y + s * dx + c * dy, wrap_angle(th + dth)])


def relative(p_from, p_to):
    """Body-frame displacement that takes p_from to p_to (inverse of compose)."""
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    c, s = np.cos(p_from[2]), np.sin(p_from[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap_angle(p_to[2] - p_from[2])])


def _V(th):
    if abs(th) < 1e-9:
        return np.eye(2)
    a, b = np.sin(th) / th, (1.0 - np.cos(th)) / th
    return np.array([[a, -b], [b, a]])


def interpolate_arc(delta, n):
    """n body-frame poses along the constant-twist (circular-arc) path to delta."""
    dx, dy, dth = delta
    v = np.linalg.solve(_V(dth), np.array([dx, dy]))
    out = np.zeros((n, 3))
    for k, s in enumerate(np.linspace(1.0 / n, 1.0, n)):
        out[k, :2] = _V(s * dth) @ (s * v)
        out[k, 2] = s * dth
    return out


# --------------------------------------------------------------------------- #
# Primitive + library
# --------------------------------------------------------------------------- #
@dataclass
class MotionPrimitive:
    name: str
    delta: np.ndarray                     # (3,) mean body-frame displacement [m, m, rad]
    duration: float                       # [s] wall time of one execution
    cov: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    path: Optional[np.ndarray] = None     # (K,3) body-frame poses within one execution
    control: Optional[np.ndarray] = None  # CPG vector for control_signal.csv

    def __post_init__(self):
        self.delta = np.asarray(self.delta, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        if self.path is not None:
            self.path = np.asarray(self.path, dtype=float)
        if self.control is not None:
            self.control = np.asarray(self.control, dtype=float)

    def waypoints(self, n=8):
        """Body-frame poses along the execution, ending at delta."""
        if self.path is not None and len(self.path):
            return self.path
        if getattr(self, "_arc", None) is None or len(self._arc) != n:
            self._arc = interpolate_arc(self.delta, n)
        return self._arc

    @property
    def speed(self):
        return float(np.hypot(self.delta[0], self.delta[1]) / self.duration)

    @property
    def turn_rate(self):
        return float(abs(self.delta[2]) / self.duration)


class PrimitiveLibrary:
    """The primitive set of one morphology plus the cost of switching between them.

    switch_penalty is extra time [s] charged whenever the primitive changes; it
    stands in for the transient while the body wave re-establishes itself.
    Per-pair values in `switch_costs[(a, b)]` override the scalar.
    """

    def __init__(self, primitives: Iterable[MotionPrimitive], switch_penalty=0.0,
                 switch_costs: Optional[Dict] = None):
        self.primitives: Dict[str, MotionPrimitive] = {p.name: p for p in primitives}
        self.switch_penalty = float(switch_penalty)
        self.switch_costs = dict(switch_costs or {})

    def __getitem__(self, name):
        return self.primitives[name]

    def __iter__(self):
        return iter(self.primitives.values())

    def names(self):
        return list(self.primitives)

    def switch_cost(self, prev, nxt):
        if prev is None or prev == nxt:
            return 0.0
        return self.switch_costs.get((prev, nxt), self.switch_penalty)

    @property
    def max_speed(self):
        return max(p.speed for p in self)

    @property
    def max_turn_rate(self):
        return max(p.turn_rate for p in self)

    # ---- persistence -------------------------------------------------------
    def to_dict(self):
        def prim(p):
            d = {"delta": p.delta.tolist(), "duration": p.duration, "cov": p.cov.tolist()}
            if p.path is not None:
                d["path"] = p.path.tolist()
            if p.control is not None:
                d["control"] = p.control.tolist()
            return d
        return {
            "switch_penalty": self.switch_penalty,
            "switch_costs": [[a, b, c] for (a, b), c in self.switch_costs.items()],
            "primitives": {p.name: prim(p) for p in self},
        }

    @classmethod
    def from_dict(cls, d):
        prims = [MotionPrimitive(name=k, **v) for k, v in d["primitives"].items()]
        costs = {(a, b): c for a, b, c in d.get("switch_costs", [])}
        return cls(prims, d.get("switch_penalty", 0.0), costs)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# Characterisation from pose logs
# --------------------------------------------------------------------------- #
def characterize(name, trials: Sequence[np.ndarray], period, cycles_per_step=1,
                 skip_cycles=1, n_path=8, control=None) -> MotionPrimitive:
    """Fit a MotionPrimitive from repeated rollouts of one CPG parameter set.

    trials : list of (T,4) arrays [t, x, y, yaw] of the body CoG in the world
             frame (yaw may be wrapped; it is unwrapped here).
    period : oscillation period 1/f of the primitive [s].
    cycles_per_step : tail-beat cycles that make up one planner step.
    skip_cycles : start-up cycles discarded before steady swimming.

    Every window of `cycles_per_step` cycles in every trial gives one sample of
    the body-frame displacement, so a few long rollouts are enough.
    """
    step_T = cycles_per_step * period
    deltas, paths = [], []
    for tr in trials:
        tr = np.asarray(tr, dtype=float)
        t, x, y, yaw = tr[:, 0], tr[:, 1], tr[:, 2], np.unwrap(tr[:, 3])

        def pose_at(tq):
            return np.array([np.interp(tq, t, x), np.interp(tq, t, y), np.interp(tq, t, yaw)])

        t0 = t[0] + skip_cycles * period
        while t0 + step_T <= t[-1] + 1e-9:
            p0 = pose_at(t0)
            deltas.append(relative(p0, pose_at(t0 + step_T)))
            paths.append([relative(p0, pose_at(tq))
                          for tq in t0 + np.linspace(step_T / n_path, step_T, n_path)])
            t0 += step_T
    if not deltas:
        raise ValueError(f"[{name}] no complete step window in the trials")
    deltas = np.array(deltas)
    cov = np.cov(deltas.T) if len(deltas) > 1 else np.zeros((3, 3))
    return MotionPrimitive(name=name, delta=deltas.mean(axis=0), duration=step_T,
                           cov=cov, path=np.mean(paths, axis=0), control=control)


def example_library() -> PrimitiveLibrary:
    """Placeholder numbers for a ~15 cm BCF swimmer (1 s per step).

    Replace with `characterize(...)` results for your morphology.
    """
    d2 = np.deg2rad
    return PrimitiveLibrary([
        MotionPrimitive(FORWARD,    [0.060,  0.000,  0.00],   1.0, np.diag([0.006, 0.006, d2(4)]) ** 2),
        MotionPrimitive(TURN_LEFT,  [0.025,  0.008,  d2(35)], 1.0, np.diag([0.006, 0.006, d2(6)]) ** 2),
        MotionPrimitive(TURN_RIGHT, [0.025, -0.008, -d2(35)], 1.0, np.diag([0.006, 0.006, d2(6)]) ** 2),
        MotionPrimitive(BACKWARD,   [-0.020, 0.000,  0.00],   1.0, np.diag([0.008, 0.008, d2(8)]) ** 2),
    ], switch_penalty=0.3)
