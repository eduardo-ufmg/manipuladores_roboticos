import pytest
import numpy as np
import matplotlib.pyplot as plt

from geometry.board import Board
from geometry.transforms import create_orthonormal_frame
from trajectory.planner import WritingPlanner

from matplotlib.patches import Rectangle


@pytest.fixture
def test_board():
    origin = np.array([0.5, 0.0, 0.5])
    normal = np.array([-1.0, 0.0, 0.0])
    up_ref = np.array([0.0, 0.0, 1.0])
    x, y, z = create_orthonormal_frame(normal, up_ref)
    return Board(
        width=0.60,
        height=0.25,
        origin=origin,
        x_axis=x,
        y_axis=y,
        normal=z,
        safety_offset=0.01,
    )


@pytest.fixture
def planner(test_board):
    return WritingPlanner(test_board)


def test_digit_placement_and_boundaries(planner):
    """Test 4.1 & Test 4.3 — Digit placement and board boundary compliance"""
    code = "042"
    trajectory = planner.plan(code)

    assert len(trajectory) > 0

    for pt in trajectory:
        # Project world position back to local u, v to verify bounds
        vec = pt.position - planner.board.origin
        u = np.dot(vec, planner.board.x_axis)
        v = np.dot(vec, planner.board.y_axis)

        # Relax boundary constraint slightly by epsilon for floating point margins
        assert (
            -1e-4 <= u <= planner.board.width + 1e-4
        ), f"Trajectory violates width boundary: u={u}"
        assert (
            -1e-4 <= v <= planner.board.height + 1e-4
        ), f"Trajectory violates height boundary: v={v}"


def test_transition_generation_and_surface_consistency(planner):
    """Test 4.2 & Test 5.1 — Transition existence and surface safety offset"""
    trajectory = planner.plan("1")

    z_distances = []
    for pt in trajectory:
        vec = pt.position - planner.board.origin
        dist = np.dot(vec, planner.board.normal)
        z_distances.append(dist)

        # Absolute safety: the robot must never penetrate the offset margin
        assert dist >= planner.board.safety_offset - 1e-6

    z_distances = np.array(z_distances)

    # Assert transitions exist (distance > offset)
    max_dist = np.max(z_distances)
    assert max_dist > planner.board.safety_offset + (planner.approach_distance * 0.9)

    # Assert writing surface is hit exactly
    min_dist = np.min(z_distances)
    assert np.isclose(min_dist, planner.board.safety_offset)


def test_orientation_consistency(planner):
    """Test 5.2 — Orientation consistency"""
    trajectory = planner.plan("8")
    for pt in trajectory:
        np.testing.assert_allclose(pt.normal, planner.board.normal)


def test_velocity_smoothness(planner):
    """Test 5.3 — Velocity smoothness"""
    trajectory = planner.plan("3")

    speeds = [np.linalg.norm(pt.velocity) for pt in trajectory]
    accels = np.diff(speeds) / planner.dt

    # Maximum acceleration should be bounded. Spikes indicate C1 discontinuities.
    max_accel = np.max(np.abs(accels))
    assert (
        max_accel < 15.0
    ), f"Velocity is not smooth. Max longitudinal acceleration: {max_accel} m/s^2"


def test_trajectory_visualization(planner):
    """Visual confirmation of trajectory layout and segmentation."""
    trajectory = planner.plan("042")

    # Separate into writing (on board) and transition (off board)
    write_pts = []
    trans_pts = []

    for pt in trajectory:
        vec = pt.position - planner.board.origin
        dist = np.dot(vec, planner.board.normal)
        u = np.dot(vec, planner.board.x_axis)
        v = np.dot(vec, planner.board.y_axis)

        if np.isclose(dist, planner.board.safety_offset, atol=1e-4):
            write_pts.append((u, v))
        else:
            trans_pts.append((u, v))

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot board boundaries
    rect = Rectangle(
        (0, 0),
        planner.board.width,
        planner.board.height,
        fill=False,
        color="black",
        linewidth=2,
    )
    ax.add_patch(rect)

    # Cell dividers
    for i in range(1, 3):
        ax.axvline(i * planner.board.width / 3.0, color="gray", linestyle="--")

    ax.scatter(
        [p[0] for p in trans_pts],
        [p[1] for p in trans_pts],
        c="lightgray",
        s=1,
        label="Transition Path",
    )
    ax.scatter(
        [p[0] for p in write_pts],
        [p[1] for p in write_pts],
        c="blue",
        s=3,
        label="Writing Path",
    )

    ax.set_aspect("equal")
    ax.set_title("Full Planner Trajectory Output (Projected to 2D)")
    ax.legend()

    output_path = "trajectory/full_trajectory.png"
    plt.savefig(output_path)
    plt.close(fig)

    import os

    assert os.path.exists(output_path)
