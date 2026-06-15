import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Returns the unit vector of v."""
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm


def create_orthonormal_frame(
    normal: np.ndarray, up_reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs an orthonormal frame (x, y, z) given a normal vector and a reference 'up' vector.
    The normal defines the z-axis. The y-axis aligns with the 'up' direction.
    """
    z = normalize(normal)
    x = normalize(np.cross(up_reference, z))
    y = np.cross(z, x)
    return x, y, z


def project_to_plane(
    point: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Projects a 3D point orthogonally onto a plane."""
    n = normalize(plane_normal)
    v = point - plane_origin
    distance = np.dot(v, n)
    return point - distance * n
