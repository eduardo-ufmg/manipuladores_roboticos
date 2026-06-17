import numpy as np
import pytest

from control.qp_controller import ControllerConfig, QPController
from control.tasks import compute_task_error, compute_task_jacobian


@pytest.fixture
def ideal_6dof():
    """Provides limits and an identity Jacobian for a simulated 6-DOF floating joint."""
    q_min = np.full(6, -2.0)
    q_max = np.full(6, 2.0)
    Jg = np.identity(6)
    return q_min, q_max, Jg


@pytest.fixture
def controller():
    config = ControllerConfig(dt=0.01, K_pos=10.0, K_ori=5.0)
    return QPController(config, n_joints=6)


def test_static_point_convergence(ideal_6dof, controller):
    """Test 6.1 — Static point convergence"""
    q_min, q_max, Jg = ideal_6dof

    current_pos = np.array([0.1, 0.0, 0.0])
    desired_pos = np.array([0.0, 0.0, 0.0])
    current_z = np.array([0.0, 0.0, -1.0])
    desired_z = np.array([0.0, 0.0, -1.0])

    r_error = compute_task_error(current_pos, current_z, desired_pos, desired_z)
    Jr = compute_task_jacobian(Jg, current_z, desired_z)

    # Static target
    v_des = np.zeros(3)
    q_current = np.zeros(6)

    q_dot = controller.solve(q_current, q_min, q_max, Jr, r_error, v_des)

    # Since r_error is +0.1 in X, q_dot for X (index 0) must be negative to converge
    assert q_dot[0, 0] < -0.1
    # Y and Z positional velocities should be zero
    assert np.isclose(q_dot[1, 0], 0.0, atol=1e-3)
    assert np.isclose(q_dot[2, 0], 0.0, atol=1e-3)


def test_straight_line_tracking(ideal_6dof, controller):
    """Test 6.2 — Straight-line tracking"""
    q_min, q_max, Jg = ideal_6dof

    current_pos = np.array([0.5, 0.5, 0.5])
    desired_pos = np.array([0.5, 0.5, 0.5])  # Error is currently 0
    current_z = np.array([0.0, 0.0, -1.0])
    desired_z = np.array([0.0, 0.0, -1.0])

    r_error = compute_task_error(current_pos, current_z, desired_pos, desired_z)
    Jr = compute_task_jacobian(Jg, current_z, desired_z)

    # Moving entirely in Y at 0.1 m/s
    v_des = np.array([0.0, 0.1, 0.0])
    q_current = np.zeros(6)

    q_dot = controller.solve(q_current, q_min, q_max, Jr, r_error, v_des)

    # Velocity command must exactly match feedforward when error is zero
    assert np.isclose(q_dot[1, 0], 0.1, atol=1e-3)


def test_orientation_only_tracking(ideal_6dof, controller):
    """Test 6.3 — Orientation-only tracking"""
    q_min, q_max, Jg = ideal_6dof

    current_pos = np.array([0.0, 0.0, 0.0])
    desired_pos = np.array([0.0, 0.0, 0.0])

    # Misaligned orientation (45 degrees off)
    current_z = np.array([0.707, 0.0, -0.707])
    desired_z = np.array([0.0, 0.0, -1.0])

    r_error = compute_task_error(current_pos, current_z, desired_pos, desired_z)
    assert r_error[3, 0] > 0.1  # Soft orientation error > 0

    Jr = compute_task_jacobian(Jg, current_z, desired_z)
    v_des = np.zeros(3)
    q_current = np.zeros(6)

    q_dot = controller.solve(q_current, q_min, q_max, Jr, r_error, v_des)

    # Angular velocity commands must be generated to correct alignment
    omega_command = q_dot[3:6]
    assert np.linalg.norm(omega_command) > 0.1


def test_joint_limit_avoidance(ideal_6dof, controller):
    """Test 6.4 — Joint-limit avoidance"""
    q_min, q_max, Jg = ideal_6dof

    # Move Joint 0 extremely close to its maximum limit (2.0)
    q_current = np.array([1.99, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Zero task error and zero velocity desire
    r_error = np.zeros((4, 1))
    Jr = np.zeros((4, 6))  # Kill the task Jacobian to isolate secondary objective
    v_des = np.zeros(3)

    q_dot = controller.solve(q_current, q_min, q_max, Jr, r_error, v_des)

    # The secondary objective should push Joint 0 negatively away from the 2.0 limit
    assert q_dot[0, 0] < -0.1
