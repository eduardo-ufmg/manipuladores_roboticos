from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from geometry.board import Board
from trajectory.digits import DIGITS
from trajectory.spline import StrokeSpline


@dataclass
class TrajectoryPoint:
    position: np.ndarray
    velocity: np.ndarray
    normal: np.ndarray
    timestamp: float


Trajectory: TypeAlias = list[TrajectoryPoint]


class WritingPlanner:
    def __init__(self, board: Board):
        self.board = board

        # Sizing conventions per specification
        self.digit_width = 0.12
        self.digit_height = 0.18
        self.approach_distance = 0.05

        # Kinematic constraints
        self.write_speed = 0.05  # m/s
        self.transition_speed = 0.10  # m/s
        self.dt = 0.01  # Trajectory sample rate (100 Hz)

    def plan(self, code: str, start_time: float = 0.0) -> Trajectory:
        """Assembles the writing trajectory for a 1- to 3-digit code."""
        if not code.isdigit() or not (1 <= len(code) <= 3):
            raise ValueError("Code must be a 1 to 3 digit numeric string.")

        trajectory: Trajectory = []
        current_time = start_time

        cell_width = self.board.width / 3.0
        v_offset = (self.board.height - self.digit_height) / 2.0

        for i, char in enumerate(code):
            strokes = DIGITS[char]

            # Center the 0.12m digit inside the 0.20m cell
            cell_center_u = (i + 0.5) * cell_width
            u_offset = cell_center_u - (self.digit_width / 2.0)

            for stroke in strokes:
                spline = StrokeSpline(stroke)

                # Pre-compute start and end 2D configurations
                pos_start_2d = spline.evaluate_position(np.array([0.0]))[0]
                u_start = u_offset + pos_start_2d[0] * self.digit_width
                v_start = v_offset + pos_start_2d[1] * self.digit_height

                p_approach_start = self.board.get_approach_pose(
                    u_start, v_start, self.approach_distance
                )
                p_write_start = self.board.board_to_world(u_start, v_start)

                # Free-space transition to the new stroke approach point
                if trajectory:
                    p_last = trajectory[-1].position
                    transition_pts, current_time = self.generate_cubic_motion(
                        p_last, p_approach_start, self.transition_speed, current_time
                    )
                    trajectory.extend(transition_pts)

                # Approach motion (drop to board)
                approach_pts, current_time = self.generate_cubic_motion(
                    p_approach_start, p_write_start, self.transition_speed, current_time
                )
                trajectory.extend(approach_pts)

                # Writing motion
                write_pts, current_time = self._generate_writing_motion(
                    spline, u_offset, v_offset, current_time
                )
                trajectory.extend(write_pts)

                # Retreat motion (lift from board)
                p_write_end = write_pts[-1].position
                pos_end_2d = spline.evaluate_position(np.array([spline.length]))[0]
                u_end = u_offset + pos_end_2d[0] * self.digit_width
                v_end = v_offset + pos_end_2d[1] * self.digit_height

                p_retreat_end = self.board.get_approach_pose(
                    u_end, v_end, self.approach_distance
                )

                retreat_pts, current_time = self.generate_cubic_motion(
                    p_write_end, p_retreat_end, self.transition_speed, current_time
                )
                trajectory.extend(retreat_pts)

        return trajectory

    def generate_cubic_motion(
        self, p0: np.ndarray, pf: np.ndarray, speed: float, start_time: float
    ) -> tuple[Trajectory, float]:
        """Generates a rest-to-rest minimum-jerk profile for safe transitions."""
        distance = np.linalg.norm(pf - p0)
        if distance < 1e-6:
            return [], start_time

        duration = distance / speed
        num_steps = max(2, int(duration / self.dt))

        pts = []
        t_current = start_time

        for step in range(1, num_steps + 1):
            t_current += self.dt
            tau = step / num_steps

            # Cubic interpolation: s(tau) = -2*tau^3 + 3*tau^2
            s = -2.0 * (tau**3) + 3.0 * (tau**2)
            ds_dtau = -6.0 * (tau**2) + 6.0 * tau

            pos = p0 + s * (pf - p0)
            vel = (ds_dtau / duration) * (pf - p0)

            # Invert the normal: must strictly oppose the board
            pts.append(TrajectoryPoint(pos, vel, -self.board.normal, t_current))

        return pts, t_current

    def _generate_writing_motion(
        self, spline: StrokeSpline, u_offset: float, v_offset: float, start_time: float
    ) -> tuple[Trajectory, float]:
        """Integrates the 2D spline to produce constant-velocity 3D task-space commands."""
        pts = []
        s_norm = 0.0
        t_current = start_time

        while s_norm < spline.length:
            pos_2d = spline.evaluate_position(np.array([s_norm]))[0]
            vel_2d = spline.evaluate_derivative(np.array([s_norm]))[0]

            v_3d_unscaled = (vel_2d[0] * self.digit_width) * self.board.x_axis + (
                vel_2d[1] * self.digit_height
            ) * self.board.y_axis
            speed_unscaled = np.linalg.norm(v_3d_unscaled)

            if speed_unscaled < 1e-6:
                ds_dt = 1.0
                vel_3d = np.zeros(3)
            else:
                ds_dt = self.write_speed / speed_unscaled
                vel_3d = (v_3d_unscaled / speed_unscaled) * self.write_speed

            u = u_offset + pos_2d[0] * self.digit_width
            v = v_offset + pos_2d[1] * self.digit_height
            pos_3d = self.board.board_to_world(u, v)

            # Invert the normal: must strictly oppose the board
            pts.append(TrajectoryPoint(pos_3d, vel_3d, -self.board.normal, t_current))

            s_norm += ds_dt * self.dt
            t_current += self.dt

        return pts, t_current
