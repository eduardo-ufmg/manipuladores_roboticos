import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Cylinder:
    radius: float
    width: float  # Usable longitudinal length (maps to u)
    height: float  # Usable arc length along circumference (maps to v)
    center: np.ndarray  # Geometric center of the cylinder
    u_axis: np.ndarray  # Longitudinal axis (left-to-right)
    v_tangent: np.ndarray  # Tangent axis at equator (pointing up)
    normal_ref: np.ndarray  # Normal axis at equator (pointing toward robot)
    safety_offset: float = 0.01

    def __post_init__(self):
        """Strict geometric validation for orthogonality and unit length."""
        assert np.isclose(
            np.linalg.norm(self.u_axis), 1.0
        ), "u_axis must be unit length."
        assert np.isclose(
            np.linalg.norm(self.v_tangent), 1.0
        ), "v_tangent must be unit length."
        assert np.isclose(
            np.linalg.norm(self.normal_ref), 1.0
        ), "normal_ref must be unit length."
        assert np.isclose(
            np.dot(self.u_axis, self.v_tangent), 0.0
        ), "u_axis and v_tangent must be orthogonal."
        assert np.isclose(
            np.dot(self.u_axis, self.normal_ref), 0.0
        ), "u_axis and normal_ref must be orthogonal."
        assert np.isclose(
            np.dot(self.v_tangent, self.normal_ref), 0.0
        ), "v_tangent and normal_ref must be orthogonal."

    def evaluate_kinematics(
        self,
        u: float,
        v: float,
        du_dt: float = 0.0,
        dv_dt: float = 0.0,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Maps 2D surface state (u, v, dot{u}, dot{v}) to 3D task-space kinematics.
        """
        # Center the writable domain on the cylinder's geometric center
        u_c = u - self.width / 2.0
        v_c = v - self.height / 2.0

        # Vertical coordinate v maps to arc length; theta is the azimuthal angle
        theta = v_c / self.radius
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Dynamic normal rotating around the longitudinal axis
        normal = cos_t * self.normal_ref + sin_t * self.v_tangent

        # Total radial distance from the longitudinal center axis
        r_total = self.radius + self.safety_offset + offset

        # 3D Position
        pos = self.center + u_c * self.u_axis + r_total * normal

        # 3D Velocity (Analytical Jacobian)
        dnormal_dt = (-sin_t * self.normal_ref + cos_t * self.v_tangent) * (
            dv_dt / self.radius
        )
        vel = du_dt * self.u_axis + r_total * dnormal_dt

        return pos, vel, normal
