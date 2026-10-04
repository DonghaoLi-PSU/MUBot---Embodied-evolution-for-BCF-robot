# -*- coding: utf-8 -*-
"""
Waypoint mission in a tank with obstacles, using the surrogate simulator.

    cd "source code" && python3 -m MU_EV.planning.demo [--library lib.json] [--plot out.png]
"""

import argparse

import numpy as np

from MU_EV.planning.behavior import BehaviorPlanner, Mode, run_mission
from MU_EV.planning.planner import Goal
from MU_EV.planning.primitives import PrimitiveLibrary, example_library
from MU_EV.planning.sim import PrimitiveSimulator
from MU_EV.planning.world import Workspace


def default_scenario():
    ws = Workspace(bounds=(0.0, 2.0, 0.0, 1.2),
                   obstacles=[(0.6, 0.45, 0.15), (1.0, 0.85, 0.12), (1.45, 0.4, 0.15)],
                   robot_radius=0.06)
    start = np.array([0.15, 0.2, 0.0])
    waypoints = [Goal(0.9, 0.3), Goal(1.8, 0.95), Goal(0.3, 1.0, theta=np.pi)]
    return ws, start, waypoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", help="primitive library JSON (default: placeholder numbers)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--noise", type=float, default=1.0, help="simulator noise / library cov")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", help="save the first trial's trajectory to this PNG")
    args = ap.parse_args()

    lib = PrimitiveLibrary.load(args.library) if args.library else example_library()
    ws, start, waypoints = default_scenario()

    outcomes, logs = [], []
    for k in range(args.trials):
        bp = BehaviorPlanner(lib, ws, waypoints)
        sim = PrimitiveSimulator(lib, ws, start, noise_scale=args.noise, seed=args.seed + k)
        log = run_mission(bp, sim, start)
        logs.append((log, bp))
        outcomes.append(log.final_mode)
        switches = sum(a != b for a, b in zip(log.primitives, log.primitives[1:]))
        recov = sum(m == Mode.RECOVER.value for m in log.modes)
        print(f"trial {k:2d}: {log.final_mode.value:8s} steps={len(log.primitives):3d} "
              f"switches={switches:3d} recovery_steps={recov:2d} collided={log.collided}")

    ok = sum(o is Mode.DONE for o in outcomes)
    print(f"\nsuccess {ok}/{args.trials}")
    print("\nevents of trial 0:")
    print("\n".join(logs[0][1].events))

    if args.plot:
        plot(ws, waypoints, [l for l, _ in logs], args.plot)


def plot(ws, waypoints, logs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"forward": "#2a6fdb", "turn_left": "#2bb673", "turn_right": "#e8a33d", "backward": "#d64545"}
    fig, ax = plt.subplots(figsize=(9, 5.6))
    xmin, xmax, ymin, ymax = ws.bounds
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, lw=1.5))
    for cx, cy, r in ws.obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color="0.6"))
        ax.add_patch(plt.Circle((cx, cy), r + ws.robot_radius, fill=False, ls="--", color="0.6"))
    for log in logs[1:]:
        P = np.array(log.poses)
        ax.plot(P[:, 0], P[:, 1], color="0.75", lw=0.8, zorder=1)
    P = np.array(logs[0].poses)
    for i, prim in enumerate(logs[0].primitives):
        ax.plot(P[i:i + 2, 0], P[i:i + 2, 1], color=colors.get(prim, "k"), lw=2.2, zorder=2)
    ax.quiver(P[::3, 0], P[::3, 1], np.cos(P[::3, 2]), np.sin(P[::3, 2]),
              scale=40, width=0.003, color="k", zorder=3)
    for i, g in enumerate(waypoints):
        ax.add_patch(plt.Circle((g.x, g.y), g.tol_xy, fill=False, color="purple", lw=1.5))
        ax.annotate(f"WP{i}", (g.x, g.y), xytext=(6, 6), textcoords="offset points", color="purple")
    for name, c in colors.items():
        ax.plot([], [], color=c, lw=2.2, label=name)
    ax.plot([], [], color="0.75", label="other trials")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Closed-loop primitive planning (trial 0 colored by primitive)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
