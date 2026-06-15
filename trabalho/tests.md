The most important engineering decision now is:

## Do not wait until the full robot-writing pipeline exists before testing.

This project naturally decomposes into:

1. geometric correctness,
2. trajectory correctness,
3. control correctness,
4. simulation correctness.

Each should be tested independently before integration.

Below is a recommended incremental validation plan.

---

# Phase 1 — Geometry Validation

Goal:
Verify that board coordinates and world coordinates are correct.

Files involved:

* `geometry/board.py`
* `geometry/transforms.py`

---

## Test 1.1 — Board frame orthogonality

Verify:

* axes are orthogonal,
* unit-length,
* normal consistency.

Implementation:

```python id="clo1ei"
assert np.isclose(np.linalg.norm(board.x_axis), 1.0)
assert np.isclose(np.linalg.norm(board.y_axis), 1.0)
assert np.isclose(np.dot(board.x_axis, board.y_axis), 0.0)
```

Purpose:
Prevent subtle projection and orientation bugs.

---

## Test 1.2 — Board projection correctness

Input:
Known 2D board coordinates.

Expected:
Correct 3D world positions.

Example:

```python id="gphbb2"
(0,0) → board origin
(1,0) → one board-width direction
(0,1) → one board-height direction
```

Implementation:
Simple numerical assertions.

---

## Test 1.3 — Safety offset correctness

Verify:
Projected writing points lie at:

```text id="i8v6xe"
board_plane + offset * normal
```

Purpose:
Guarantee no board penetration.

---

# Phase 2 — Digit Template Validation

Goal:
Verify digit definitions before introducing smoothing or control.

Files involved:

* `trajectory/digits.py`

---

## Test 2.1 — Digit existence

Verify:
Digits `0–9` all exist.

Implementation:

```python id="50e4wa"
for digit in "0123456789":
    assert digit in DIGITS
```

---

## Test 2.2 — Stroke validity

Verify:

* each stroke contains ≥2 points,
* coordinates remain normalized,
* no NaNs.

Implementation:

```python id="d69v8c"
assert 0.0 <= x <= 1.0
assert 0.0 <= y <= 1.0
```

---

## Test 2.3 — Visualization test

Render:

* raw polylines with matplotlib.

Purpose:
Human visual verification.

This is extremely important.
Most trajectory bugs become visually obvious immediately.

---

# Phase 3 — Spline Validation

Goal:
Verify smoothing quality and continuity.

Files involved:

* `trajectory/spline.py`

---

## Test 3.1 — Spline interpolation correctness

Verify:
Spline approximately follows original control points.

Method:
Plot:

* original polyline,
* smoothed curve.

---

## Test 3.2 — Velocity continuity

Verify:
First derivative continuity.

Method:
Numerically inspect:

```python id="fq48a2"
dx/dt
dy/dt
```

No sharp jumps should exist.

---

## Test 3.3 — Arc-length stability

Verify:
Sampling density does not create:

* clustering,
* gaps,
* unstable velocities.

Method:
Plot inter-sample distance histogram.

---

# Phase 4 — Trajectory Planning Validation

Goal:
Verify full 3-digit composition.

Files involved:

* `trajectory/planner.py`

---

## Test 4.1 — Digit placement

Input:
Example code:

```text id="2sq7s6"
042
```

Verify:

* centered layout,
* proper spacing,
* scaling correctness.

Method:
2D plot of the entire board.

---

## Test 4.2 — Transition generation

Verify:
Approach/retreat motions exist.

Expected sequence:

```text id="mbmjlwm"
free-space
↓
approach
↓
writing
↓
retreat
↓
transition
```

Method:
Color-code trajectory segments in visualization.

---

## Test 4.3 — Board boundary compliance

Verify:
All points remain inside writable area.

Implementation:

```python id="a1hsfh"
assert 0 <= u <= board.width
assert 0 <= v <= board.height
```

---

# Phase 5 — Task-Space Trajectory Validation

Goal:
Verify generated 3D task-space trajectory.

Files involved:

* `trajectory/planner.py`
* `geometry/board.py`

---

## Test 5.1 — Surface consistency

Verify:
Writing samples remain at constant distance from board.

Implementation:

```python id="7evk47"
distance = dot(point - board_origin, board_normal)
```

Expected:
Constant offset.

---

## Test 5.2 — Orientation consistency

Verify:
Desired orientation aligns with board normal.

Method:
Numerical angle check.

---

## Test 5.3 — Velocity smoothness

Plot:

* translational speed over time.

Expected:
No spikes.

This is extremely important for stable QP behavior.

---

# Phase 6 — Controller Unit Validation

Goal:
Verify controller independently from writing.

Files involved:

* `control/tasks.py`
* `control/qp_controller.py`

---

## Test 6.1 — Static point convergence

Task:
Track a fixed point in front of board.

Verify:
Error converges toward zero.

This is the first closed-loop validation.

---

## Test 6.2 — Straight-line tracking

Task:
Move along a single line segment.

Verify:

* stable tracking,
* bounded joint velocities,
* no oscillation.

---

## Test 6.3 — Orientation-only tracking

Task:
Rotate tool toward board normal.

Verify:
Soft orientation behaves properly.

---

## Test 6.4 — Joint-limit avoidance

Task:
Command near-limit configurations.

Verify:
Secondary objective pushes robot away from limits.

---

# Phase 7 — Integrated Writing Validation

Goal:
Validate full system behavior.

Files involved:
All.

---

## Test 7.1 — Single digit

Example:

```text id="s2i70p"
0
```

Purpose:
Simplify debugging.

---

## Test 7.2 — Multi-digit code

Example:

```text id="9spm3l"
123
```

Verify:

* transitions,
* spacing,
* continuity.

---

## Test 7.3 — Worst-case code

Example:

```text id="jhm8a0"
888
```

Purpose:
Stress-test:

* self-intersections,
* curvature,
* repeated motion.

---

## Test 7.4 — Random code batch

Generate:
Many random codes.

Verify:

* no crashes,
* no solver failures,
* bounded tracking error.

---

# Phase 8 — Numerical Robustness Validation

Goal:
Catch stability issues.

---

## Test 8.1 — Solver conditioning

Monitor:

* QP solve success,
* matrix conditioning,
* infeasibility.

---

## Test 8.2 — Time-step sensitivity

Run:
Different `dt`.

Expected:
Behavior remains qualitatively stable.

---

## Test 8.3 — Trajectory density sensitivity

Vary:
Spline sample count.

Verify:
Controller stability remains acceptable.

---

# Phase 9 — Visualization & Debugging Tools

These are not optional.
They dramatically accelerate development.

---

## Tool 9.1 — 2D trajectory plotter

Plot:

* raw strokes,
* splines,
* sampled trajectory.

Use heavily.

---

## Tool 9.2 — 3D board visualization

Render:

* board frame,
* writing points,
* normals.

---

## Tool 9.3 — Tracking-error plots

Plot over time:

* position error norm,
* orientation error,
* joint velocity norm.

---

# Recommended development order

Best practical order:

```text id="j2yw1t"
1. Geometry tests
2. Digit visualization
3. Spline visualization
4. Full trajectory visualization
5. Static point control
6. Straight-line control
7. Single digit writing
8. Three-digit writing
9. Robustness tests
```

This minimizes debugging complexity because:

* every stage becomes visually verifiable,
* errors localize cleanly,
* integration becomes incremental rather than catastrophic.
