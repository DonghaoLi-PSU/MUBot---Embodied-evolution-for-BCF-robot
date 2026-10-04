# Behavior planning over motion primitives

Plans for one fixed morphology with four primitives: `forward`, `turn_left`, `turn_right` and `backward`. Each primitive is a CPG parameter set (the vector in `control_signal.csv`) run for a fixed number of tail-beat cycles.

The module only uses numpy. ROS and Gazebo are not needed.

| File | Role |
|---|---|
| `primitives.py` | SE(2) math, `MotionPrimitive` / `PrimitiveLibrary` (JSON), `characterize()` to fit a primitive from pose logs |
| `world.py` | `Workspace`: tank walls + circular obstacles, clearance queries |
| `planner.py` | `HybridAStarPlanner`: search over primitive sequences |
| `behavior.py` | `BehaviorPlanner`: closed-loop mission FSM (navigate / recover / done / failed), `run_mission()` |
| `sim.py` | `PrimitiveSimulator`: stochastic surrogate for testing without Gazebo |
| `demo.py` | 3-waypoint mission in a tank with obstacles |

```bash
cd "source code"
python3 -m MU_EV.planning.demo --trials 20 --plot demo.png
python3 -m MU_EV.planning.demo --library my_morph_primitives.json
```

## Algorithm

**1. Primitive model.** The planner never simulates hydrodynamics. It uses each primitive's measured effect: the mean body-frame displacement `(dx, dy, dθ)` over one execution, its 3×3 covariance across repeats, and the path in between. `characterize()` builds these from CoG pose logs `[t, x, y, yaw]`. It samples the pose at cycle boundaries, so the body undulation cancels and only the net motion remains. It also skips the start-up cycles, and every window of N cycles in a long rollout counts as one sample.

**2. Hybrid A\*.** Nodes keep the exact continuous pose reached by chaining the primitives, so turn angles do not have to divide 360°. Duplicate nodes are merged on `(x, y, heading bin, last primitive)`. The last primitive is part of the key because switching costs depend on it.
- Edge cost: `duration + switch penalty + w_u · std(primitive) + w_c · intrusion into the soft clearance band`. The switch penalty prices in the transient while the body wave rebuilds. Noisy primitives such as backward swimming cost more.
- Heuristic: `max(dist / v_max, heading error / ω_max)`, which is admissible in time. It is weighted by 1.5 by default, so the plan cost is guaranteed to be at most 1.5× optimal. This is about 10× faster on heading-constrained goals.
- Collision check: pose covariance is propagated along each branch to first order, and the robot circle is inflated by `min_margin + 2σ`. Long open-loop chains therefore keep further away from obstacles.
- If the goal is not reached within `max_expansions`, the planner returns the best partial plan, which is enough for receding-horizon control.

**3. Behavior layer (closed loop).** Swimming is too noisy to run a whole plan open-loop.
- `NAVIGATE`: plan to the current waypoint and execute `replan_every` primitives. Replan early if the measured pose drifts from the expected pose by more than `deviation_xy` / `deviation_theta`.
- `RECOVER`: entered when the planner finds no feasible move or there is no progress for `stall_window` steps. It runs a few clearance-maximising primitives, backing off first when that is about as good, then hands back to `NAVIGATE`.
- `DONE` / `FAILED`: `FAILED` means more than `max_recoveries` recoveries on one waypoint, or the step budget ran out.

The surrogate has 1× the library noise plus a 30 % displacement loss after every primitive switch, which the planner does not model. With it, the demo succeeds in 30/30 trials, and in 15/15 at 1.5× noise.

## Connecting to Gazebo

`BehaviorPlanner.next_primitive(pose)` only needs an `Executor` with `execute(name) -> pose`. Two pieces of the current simulation need to change before a Gazebo executor can work:

1. **Characterise each primitive.** Run each primitive's `control` vector for several seconds and log the mass-weighted CoG pose (`Compute_Mass_Weighted_State` in `Hydro_plugin.cc` already computes it). Then call `characterize(...)` and `PrimitiveLibrary.save(...)`. The existing `drift_metrics.csv` is not enough: it stores `|heading drift|`, so the turn direction is lost.
2. **Switch primitives within one episode.** `actuator_plugin.cc` reads `control_signal.csv` once in `Load()`, and `Hydro_plugin.cc` terminates at 6.1 s. A mission needs the actuator to accept a new parameter set at runtime, for example from a ROS topic or by re-reading the file. To avoid torque jumps, it should keep a continuous phase accumulator `φ += 2π f dt` instead of `sin(2π f t + φ_i)`, and blend amplitudes over about one cycle. The episode termination also needs to come from the mission rather than a fixed time.

The CPG is `a_i sin(2π(f t + φ_i))` with no DC term. If your turning primitives do not yet turn well enough, a per-joint bias `b_i` (a static tail deflection) is the usual way to turn in BCF swimming. It would add N parameters to the turning primitives' PGPE search.
