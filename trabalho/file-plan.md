Below is a practical division of responsibilities for the project.

## `geometry/board.py`

Purpose: represent the writing surface and its coordinate frame.

Contents:

* `Board` data structure with width, height, origin, axes, normal, and safety offset.
* Methods to map 2D board coordinates to 3D world coordinates.
* Methods to place and center a digit block inside the writable area.
* Methods to generate “approach” and “retreat” points above the board plane.

Key idea: this file should know nothing about digits or control.

---

## `geometry/transforms.py`

Purpose: small reusable geometric utilities.

Contents:

* Vector normalization.
* Orthonormal frame construction.
* Rotation helpers.
* Plane projection and frame conversion utilities.
* Optional helpers for converting between board-frame and world-frame points.

Key idea: keep this as a low-level utility module with no robot-specific logic.

---

## `trajectory/digits.py`

Purpose: store the stroke definitions for numerals `0`–`9`.

Contents:

* A normalized stroke template for each digit.
* Each digit represented as a list of strokes.
* Each stroke represented as a polyline in 2D board coordinates.
* Optional metadata such as preferred stroke order or entry/exit points.

Suggested representation:

* Coordinates normalized to a unit box, for example `[0,1] × [0,1]`.
* Every digit scaled later by the board layout code.

Key idea: this is the only file that encodes the shapes of the digits.

---

## `trajectory/spline.py`

Purpose: turn piecewise linear digit strokes into smooth paths.

Contents:

* Arc-length parameterization of a polyline.
* Cubic spline or smooth interpolator construction.
* Sampling functions for position and first derivative.
* Optional curvature-aware resampling if needed later.

Key idea: digit strokes are defined simply; this file makes them smooth enough for control.

---

## `trajectory/planner.py`

Purpose: assemble a full 3-digit writing plan from the digit templates.

Contents:

* Digit selection from an integer code, for example `042`.
* Layout logic to split the board into three digit cells.
* Scaling and centering of each digit.
* Construction of the full writing sequence:

  * approach
  * stroke execution
  * retreat
  * free-space transition to next digit
* Conversion of the full plan into a time-parameterized trajectory.

Key idea: this is the “composition layer” that turns digit shapes into a full writing job.

---

## `control/tasks.py`

Purpose: define the task-space error terms used by the QP controller.

Contents:

* Position error computation.
* Orientation error computation relative to the board normal.
* Optional task stacking helpers.
* Construction of the task vector `r`.
* Construction of the task Jacobian `Jr`.

Key idea: isolate all task math here so it can evolve independently of the trajectory code.

---

## `control/qp_controller.py`

Purpose: solve the constrained inverse-kinematics / velocity QP.

Contents:

* Joint-limit handling.
* Secondary objective for joint-limit avoidance.
* QP matrix assembly.
* Solver call.
* Extraction of joint velocity command from the QP solution.

Suggested responsibilities:

* Input: current joint state, desired task state, Jacobian, joint limits.
* Output: joint velocity command.

Key idea: this file should not know anything about digits or board geometry.

---

## `simulation/robot_interface.py`

Purpose: isolate robot-model access behind a small adapter.

Contents:

* Robot instantiation.
* Forward kinematics and geometric Jacobian access.
* Joint-limit access.
* Any UAIBot-specific conversion code.
* Helper methods to get end-effector position and board-facing axis.

Key idea: this is the compatibility layer between your controller and UAIBot.

---

## `simulation/scene.py`

Purpose: construct and manage the simulation scene.

Contents:

* Robot object.
* Board visual object.
* Optional marker/trace objects.
* Animation frame updates.
* Scene initialization and `sim.run()` orchestration.

Key idea: this file handles presentation and simulation objects, not control logic.

---

## `main.py`

Purpose: run the complete pipeline.

Contents:

* Load robot model.
* Define board geometry.
* Select the 3-digit code.
* Build the writing trajectory.
* Run the control loop.
* Update the simulation.

Suggested flow:

1. Initialize robot and board.
2. Parse the code to be written.
3. Generate the full trajectory.
4. Step through trajectory samples.
5. Solve the QP at each step.
6. Animate the scene.

Key idea: this should be a thin orchestration layer only.

---

# Suggested dependency flow

Keep the dependencies one-directional:

```text
main.py
  -> simulation/*
  -> trajectory/*
  -> control/*
  -> geometry/*
```

And more specifically:

```text
trajectory/planner.py -> trajectory/digits.py
trajectory/spline.py  -> geometry/transforms.py
control/tasks.py      -> geometry/board.py, simulation/robot_interface.py
control/qp_controller.py -> control/tasks.py
simulation/scene.py   -> simulation/robot_interface.py, geometry/board.py
```

Avoid circular imports.

---

# Minimal first implementation order

The cleanest implementation sequence is:

1. `trajectory/digits.py`
2. `geometry/board.py`
3. `trajectory/planner.py`
4. `trajectory/spline.py`
5. `control/tasks.py`
6. `control/qp_controller.py`
7. `simulation/robot_interface.py`
8. `simulation/scene.py`
9. `main.py`

That order lets you test path generation before worrying about the full closed-loop simulation.
