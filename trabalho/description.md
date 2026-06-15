The problem is now sufficiently constrained to define a clean architecture.

The simplest robust design is:

1. Represent each digit as one or more normalized 2D strokes
2. Convert strokes into smooth splines
3. Place the splines on the whiteboard plane
4. Generate task-space trajectories
5. Track them with the QP controller
6. Insert free-space transitions between digits

This is substantially simpler than importing fonts or SVGs and is fully sufficient for numeric-only writing.

A good architecture for this project is:

```text
code → stroke generator → spline trajectory
     → board-frame projection
     → task-space samples
     → QP controller
     → robot simulation
```

The important design choices are now mostly settled.

---

# Recommended digit representation

Use normalized vector strokes.

Example for digit `2`:

```python id="c1v13r"
DIGIT_2 = [
    [(0.0, 1.0), (0.8, 1.0), (1.0, 0.8)],
    [(1.0, 0.8), (0.0, 0.0)],
    [(0.0, 0.0), (1.0, 0.0)],
]
```

Advantages:

* trivial to edit
* deterministic
* dependency-free
* easy spline smoothing
* easy scaling
* easy board projection

This is much simpler than:

* fonts
* SVG parsing
* Bézier extraction
* raster tracing

and fully adequate for digits only.

---

# Recommended stroke style

Use segmented “digital handwriting”:

* piecewise-linear
* smoothed afterward with cubic splines

Reasons:

* extremely simple
* stable
* controllable
* easy to debug
* easy to center and scale

Avoid:

* cursive generation
* procedural handwriting models
* font engines

Those add complexity with little benefit here.

---

# Recommended board model

Represent the board as a plane:

```python id="zlk8v5"
board_origin
board_x_axis
board_y_axis
board_normal
```

Then every 2D stroke point `(u, v)` becomes:

```python id="w1wk7z"
p_world = (
    board_origin
    + u * board_x_axis
    + v * board_y_axis
)
```

This is extremely important because:

* it immediately generalizes to cylinders later
* it cleanly separates geometry from control

For the future cylindrical board:

* only the projection layer changes

Everything else remains untouched.

---

# Recommended orientation task

For a vertical board:

```python id="ul13kq"
z_tool ≈ -board_normal
```

Meaning:

* the tool points toward the board
* the effector stays approximately normal to the surface

This maps naturally into your existing orientation task.

---

# Recommended transition strategy

Use explicit pen-up transitions.

Sequence:

```text
approach digit
↓
write stroke
↓
retreat
↓
move laterally
↓
approach next digit
```

Even though the end effector is abstract:

* this prevents board penetration
* simplifies path generation
* improves robustness

This is much simpler than enforcing continuous surface contact across digits.

---

# Recommended writable region

Example:

```python id="4ohzj1"
BOARD_WIDTH  = 0.60
BOARD_HEIGHT = 0.25
```

Then:

* split into 3 equal digit regions
* center each digit automatically
* preserve aspect ratio

Example digit size:

* width: 12 cm
* height: 18 cm

Those are visually balanced for a Franka-sized robot.

---

# Recommended smoothing method

Use cubic spline interpolation.

Dependencies:

* `scipy.interpolate`

Pipeline:

```text
polyline points
→ arc-length parameterization
→ cubic spline
→ dense samples
```

This gives:

* smooth velocity
* smooth curvature
* stable tracking

without complicated trajectory generation.

---

# Recommended trajectory representation

Do not use the old scalar parameter `p`.

Instead use sampled trajectories:

```python id="m3jta5"
TrajectoryPoint:
    position
    velocity
    normal
    timestamp
```

This is the most important architectural change.

The current controller is:

* path-parametric

The new controller should become:

* trajectory-tracking

This is a major improvement.

---

# Recommended controller redesign

Current:

```text
q → path parameter tracking
```

Recommended:

```text
q → time-indexed task-space tracking
```

Meaning:

* desired position is sampled from time
* desired velocity is directly available
* no path phase variable needed

This simplifies arbitrary trajectories enormously.

---

# Recommended control structure

Task vector:

```text
r =
[
 position_error(3)
 orientation_error(2 or 3)
]
```

You may keep:

* soft orientation
* slack variables
* QP structure

Those are already appropriate.

---

# Recommended collision strategy

For now:

* board collision only

Simplest method:

```python id="8l7qpx"
distance_to_board >= safety_offset
```

Example:

```python id="4u1s4n"
offset = 0.01  # 1 cm
```

The writing trajectory itself lies exactly on:

```text
board_plane + offset * normal
```

So the robot never mathematically penetrates the board.

This is much simpler than explicit collision constraints.

---

# Recommended software structure

```text
geometry/
    board.py
    transforms.py

trajectory/
    digits.py
    spline.py
    planner.py

control/
    qp_controller.py
    tasks.py

simulation/
    scene.py
    robot_interface.py
```

Even for a small project, this separation will pay off quickly.

---

# Most important redesign decision

The key insight is:

## The controller should no longer know anything about “digits”.

The controller should only receive:

* desired pose trajectory
* desired velocity trajectory

Everything about:

* numerals
* strokes
* splines
* board layout

must happen before the control layer.

That separation will make the system dramatically cleaner and easier to evolve.
