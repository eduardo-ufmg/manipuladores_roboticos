from dataclasses import dataclass
from typing import TypeAlias, Protocol

import numpy as np

from trajectory.digits import DIGITS
from trajectory.spline import StrokeSpline


@dataclass
class TrajectoryPoint:
    position: np.ndarray
    velocity: np.ndarray
    normal: np.ndarray
    timestamp: float
    is_writing: bool = False


Trajectory: TypeAlias = list[TrajectoryPoint]


class WritingSurface(Protocol):
    """Protocol defining the required geometric interface for any target manifold."""

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    def evaluate_kinematics(
        self,
        u: float,
        v: float,
        du_dt: float = 0.0,
        dv_dt: float = 0.0,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class WritingPlanner:
    def __init__(self, surface: WritingSurface):
        self.surface = surface

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

        cell_width = self.surface.width / 3.0
        v_offset = (self.surface.height - self.digit_height) / 2.0

        u_last, v_last = None, None

        for i, char in enumerate(code):
            strokes = DIGITS[char]

            cell_center_u = (i + 0.5) * cell_width
            u_offset = cell_center_u - (self.digit_width / 2.0)

            for stroke in strokes:
                spline = StrokeSpline(stroke)

                pos_start_2d = spline.evaluate_position(np.array([0.0]))[0]
                u_start = u_offset + pos_start_2d[0] * self.digit_width
                v_start = v_offset + pos_start_2d[1] * self.digit_height

                # Free-space transition to the new stroke
                if u_last is not None and v_last is not None:
                    transition_pts, current_time = self._generate_surface_cubic_motion(
                        u_last,
                        v_last,
                        self.approach_distance,
                        u_start,
                        v_start,
                        self.approach_distance,
                        self.transition_speed,
                        current_time,
                    )
                    trajectory.extend(transition_pts)

                # Approach motion (drop to surface)
                approach_pts, current_time = self._generate_surface_cubic_motion(
                    u_start,
                    v_start,
                    self.approach_distance,
                    u_start,
                    v_start,
                    0.0,
                    self.transition_speed,
                    current_time,
                )
                trajectory.extend(approach_pts)

                # Writing motion
                write_pts, current_time = self._generate_writing_motion(
                    spline, u_offset, v_offset, current_time
                )
                trajectory.extend(write_pts)

                # Retreat motion (lift from surface)
                pos_end_2d = spline.evaluate_position(np.array([spline.length]))[0]
                u_end = u_offset + pos_end_2d[0] * self.digit_width
                v_end = v_offset + pos_end_2d[1] * self.digit_height

                retreat_pts, current_time = self._generate_surface_cubic_motion(
                    u_end,
                    v_end,
                    0.0,
                    u_end,
                    v_end,
                    self.approach_distance,
                    self.transition_speed,
                    current_time,
                )
                trajectory.extend(retreat_pts)

                u_last, v_last = u_end, v_end

        return trajectory

    def _generate_surface_cubic_motion(
        self,
        u0: float,
        v0: float,
        d0: float,
        uf: float,
        vf: float,
        df: float,
        speed: float,
        start_time: float,
    ) -> tuple[Trajectory, float]:
        """Generates a minimum-jerk transition safely wrapping around the target manifold."""
        # Cartesian proxy distance for time parameterization
        distance = np.sqrt((uf - u0) ** 2 + (vf - v0) ** 2 + (df - d0) ** 2)
        if distance < 1e-6:
            return [], start_time

        duration = distance / speed
        num_steps = max(2, int(duration / self.dt))

        pts = []
        t_current = start_time

        for step in range(1, num_steps + 1):
            t_current += self.dt
            tau = step / num_steps

            s = -2.0 * (tau**3) + 3.0 * (tau**2)
            ds_dtau = -6.0 * (tau**2) + 6.0 * tau

            u = u0 + s * (uf - u0)
            v = v0 + s * (vf - v0)
            d = d0 + s * (df - d0)

            du_dt = (ds_dtau / duration) * (uf - u0)
            dv_dt = (ds_dtau / duration) * (vf - v0)
            dd_dt = (ds_dtau / duration) * (df - d0)

            pos, base_vel, normal = self.surface.evaluate_kinematics(
                u, v, du_dt, dv_dt, offset=d
            )

            # Incorporate orthogonal velocity to preserve C1 continuity during approach/retreat
            vel = base_vel + dd_dt * normal

            pts.append(TrajectoryPoint(pos, vel, -normal, t_current))

        return pts, t_current

    def _generate_writing_motion(
        self, spline: StrokeSpline, u_offset: float, v_offset: float, start_time: float
    ) -> tuple[Trajectory, float]:
        """Integrates the 2D spline to produce constant-velocity 3D commands on the manifold."""
        pts = []
        s_norm = 0.0
        t_current = start_time

        while s_norm < spline.length:
            pos_2d = spline.evaluate_position(np.array([s_norm]))[0]
            vel_2d = spline.evaluate_derivative(np.array([s_norm]))[0]

            du_ds = vel_2d[0] * self.digit_width
            dv_ds = vel_2d[1] * self.digit_height

            u = u_offset + pos_2d[0] * self.digit_width
            v = v_offset + pos_2d[1] * self.digit_height

            # Temporarily evaluate to extract unscaled spatial speed
            _, vel_3d_unscaled, _ = self.surface.evaluate_kinematics(
                u, v, du_ds, dv_ds, offset=0.0
            )
            speed_unscaled = np.linalg.norm(vel_3d_unscaled)

            if speed_unscaled < 1e-6:
                ds_dt = 1.0
                vel_3d = np.zeros(3)
                du_dt, dv_dt = 0.0, 0.0
            else:
                ds_dt = self.write_speed / speed_unscaled
                du_dt = du_ds * ds_dt
                dv_dt = dv_ds * ds_dt

            pos_3d, vel_3d, normal = self.surface.evaluate_kinematics(
                u, v, du_dt, dv_dt, offset=0.0
            )

            pts.append(
                TrajectoryPoint(pos_3d, vel_3d, -normal, t_current, is_writing=True)
            )

            s_norm += ds_dt * self.dt
            t_current += self.dt

        return pts, t_current

    def generate_cartesian_cubic_motion(
        self,
        p0: np.ndarray,
        pf: np.ndarray,
        target_normal: np.ndarray,
        speed: float,
        start_time: float,
    ) -> tuple[Trajectory, float]:
        """Used strictly by the orchestration layer to transition from unmapped home coordinates."""
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

            s = -2.0 * (tau**3) + 3.0 * (tau**2)
            ds_dtau = -6.0 * (tau**2) + 6.0 * tau

            pos = p0 + s * (pf - p0)
            vel = (ds_dtau / duration) * (pf - p0)

            pts.append(TrajectoryPoint(pos, vel, target_normal, t_current))

        return pts, t_current
