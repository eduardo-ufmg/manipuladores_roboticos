from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Board:
    width: float
    height: float
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    normal: np.ndarray
    safety_offset: float = 0.01

    def __post_init__(self):
        """Strict geometric validation for orthogonality and unit length."""
        assert np.isclose(
            np.linalg.norm(self.x_axis), 1.0
        ), "x_axis must be unit length."
        assert np.isclose(
            np.linalg.norm(self.y_axis), 1.0
        ), "y_axis must be unit length."
        assert np.isclose(
            np.linalg.norm(self.normal), 1.0
        ), "normal must be unit length."
        assert np.isclose(
            np.dot(self.x_axis, self.y_axis), 0.0
        ), "Axes must be orthogonal."
        assert np.isclose(
            np.dot(self.normal, self.x_axis), 0.0
        ), "Normal must be orthogonal to x_axis."
        assert np.isclose(
            np.dot(self.normal, self.y_axis), 0.0
        ), "Normal must be orthogonal to y_axis."

    def board_to_world(self, u: float, v: float) -> np.ndarray:
        """
        Maps a 2D board coordinate (u, v) to a 3D world coordinate.
        Includes the strict safety offset along the normal.
        """
        return (
            self.origin
            + u * self.x_axis
            + v * self.y_axis
            + self.safety_offset * self.normal
        )

    def get_approach_pose(
        self, u: float, v: float, approach_distance: float = 0.05
    ) -> np.ndarray:
        """
        Generates a 3D world coordinate retracted from the board for free-space transitions.
        """
        return (
            self.origin
            + u * self.x_axis
            + v * self.y_axis
            + (self.safety_offset + approach_distance) * self.normal
        )

    def evaluate_kinematics(
        self,
        u: float,
        v: float,
        du_dt: float = 0.0,
        dv_dt: float = 0.0,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maps 2D surface state to 3D kinematics for the planar board."""
        pos = (
            self.origin
            + u * self.x_axis
            + v * self.y_axis
            + (self.safety_offset + offset) * self.normal
        )
        vel = du_dt * self.x_axis + dv_dt * self.y_axis
        return pos, vel, self.normal
