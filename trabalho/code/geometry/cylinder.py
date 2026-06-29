import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Cylinder:
    radius: float
    width: float
    height: float
    center: np.ndarray
    u_axis: np.ndarray
    v_tangent: np.ndarray
    normal_ref: np.ndarray
    safety_offset: float = 0.01

    def __post_init__(self):
        assert np.isclose(np.linalg.norm(self.u_axis), 1.0)
        assert np.isclose(np.linalg.norm(self.v_tangent), 1.0)
        assert np.isclose(np.linalg.norm(self.normal_ref), 1.0)
        assert np.isclose(np.dot(self.u_axis, self.v_tangent), 0.0)
        assert np.isclose(np.dot(self.u_axis, self.normal_ref), 0.0)
        assert np.isclose(np.dot(self.v_tangent, self.normal_ref), 0.0)

    def evaluate_kinematics(
        self,
        u: float,
        v: float,
        du_dt: float = 0.0,
        dv_dt: float = 0.0,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_c = u - self.width / 2.0
        v_c = v - self.height / 2.0

        theta = v_c / self.radius
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        normal = cos_t * self.normal_ref + sin_t * self.v_tangent
        r_total = self.radius + self.safety_offset + offset

        pos = self.center + u_c * self.u_axis + r_total * normal

        dnormal_dt = (-sin_t * self.normal_ref + cos_t * self.v_tangent) * (
            dv_dt / self.radius
        )
        vel = du_dt * self.u_axis + r_total * dnormal_dt

        return pos, vel, normal

    def compute_distance(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the shortest distance from a 3D point to the cylinder surface,
        returning the scalar distance and the outward unit normal.
        """
        point = point.flatten()
        v_cp = point - self.center

        # Project vector onto the longitudinal axis
        proj_u = np.dot(v_cp, self.u_axis)

        # Rejection vector (strictly radial)
        v_rad = v_cp - proj_u * self.u_axis
        r = np.linalg.norm(v_rad)

        if r < 1e-6:
            # Singularity handling if exact center is queried
            return -self.radius, self.normal_ref

        normal = v_rad / r
        distance = r - self.radius

        return float(distance), normal
