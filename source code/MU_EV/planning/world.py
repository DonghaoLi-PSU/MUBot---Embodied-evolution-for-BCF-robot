# -*- coding: utf-8 -*-
"""Planar workspace: rectangular tank walls plus circular obstacles."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Workspace:
    bounds: Tuple[float, float, float, float]               # xmin, xmax, ymin, ymax [m]
    obstacles: List[Tuple[float, float, float]] = field(default_factory=list)  # (cx, cy, r)
    robot_radius: float = 0.05                               # bounding circle of the body [m]

    def clearance(self, xy):
        """Signed free distance of the robot's bounding circle at points xy (N,2)."""
        xy = np.atleast_2d(xy)
        xmin, xmax, ymin, ymax = self.bounds
        d = np.minimum.reduce([xy[:, 0] - xmin, xmax - xy[:, 0], xy[:, 1] - ymin, ymax - xy[:, 1]])
        for cx, cy, r in self.obstacles:
            d = np.minimum(d, np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)
        return d - self.robot_radius

    def is_free(self, xy, margin=0.0):
        return bool(np.all(self.clearance(xy) > margin))
