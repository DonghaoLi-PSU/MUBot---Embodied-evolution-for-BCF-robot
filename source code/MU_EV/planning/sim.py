# -*- coding: utf-8 -*-
"""
Stochastic surrogate of the robot for developing the planner without Gazebo.

Each execution samples a displacement from N(delta, noise_scale^2 * cov) and,
right after a primitive change, scales it by `switch_efficiency` to mimic the
transient while the body wave re-forms.  Neither effect is known to the
planner beyond the library's cov and switch_penalty, so this is a fair test of
the closed loop.
"""

import numpy as np

from MU_EV.planning.primitives import PrimitiveLibrary, compose, interpolate_arc
from MU_EV.planning.world import Workspace


class PrimitiveSimulator:

    def __init__(self, library: PrimitiveLibrary, workspace: Workspace, start_pose,
                 noise_scale=1.0, switch_efficiency=0.7, seed=None):
        self.lib = library
        self.ws = workspace
        self.pose = np.asarray(start_pose, dtype=float)
        self.noise_scale = noise_scale
        self.switch_efficiency = switch_efficiency
        self.rng = np.random.default_rng(seed)
        self.last = None
        self.collided = False

    def execute(self, primitive):
        prim = self.lib[primitive]
        delta = self.rng.multivariate_normal(prim.delta, self.noise_scale ** 2 * prim.cov)
        if self.last is not None and self.last != primitive:
            delta = delta * self.switch_efficiency
        self.last = primitive
        for wp in interpolate_arc(delta, 6):
            p = compose(self.pose, wp)
            if self.ws.clearance(p[:2])[0] <= 0.0:
                self.collided = True
                return self.pose
        self.pose = compose(self.pose, delta)
        return self.pose
