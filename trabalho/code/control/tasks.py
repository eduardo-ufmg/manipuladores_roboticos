import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Constructs the 3x3 skew-symmetric matrix for a 3D vector."""
    v = v.flatten()
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def compute_task_error(
    current_pos: np.ndarray,
    current_z_axis: np.ndarray,
    desired_pos: np.ndarray,
    desired_z_axis: np.ndarray,
) -> np.ndarray:
    """
    Computes the 4x1 task error vector.
    Rows 0:3 correspond to position error.
    Row 3 corresponds to the soft orientation error.
    """
    current_pos = current_pos.reshape(3, 1)
    desired_pos = desired_pos.reshape(3, 1)
    current_z_axis = current_z_axis.reshape(3, 1)
    desired_z_axis = desired_z_axis.reshape(3, 1)

    r = np.zeros((4, 1))

    # 1. Position error
    r[0:3, 0:1] = current_pos - desired_pos

    # 2. Orientation error: 1 - dot(z_d, z_e)
    r[3, 0] = 1.0 - np.dot(desired_z_axis.T, current_z_axis)[0, 0]

    return r


def compute_task_jacobian(
    Jg: np.ndarray, current_z_axis: np.ndarray, desired_z_axis: np.ndarray
) -> np.ndarray:
    """
    Constructs the 4xN task Jacobian from the 6xN geometric Jacobian.
    """
    n_joints = Jg.shape[1]
    Jr = np.zeros((4, n_joints))

    current_z_axis = current_z_axis.reshape(3, 1)
    desired_z_axis = desired_z_axis.reshape(3, 1)

    # 1. Position Jacobian (extract upper 3xN of Jg)
    Jr[0:3, :] = Jg[0:3, :]

    # 2. Orientation Jacobian (Soft projection)
    S_ze = skew_symmetric(current_z_axis)
    Jr[3, :] = desired_z_axis.T @ S_ze @ Jg[3:6, :]

    return Jr
