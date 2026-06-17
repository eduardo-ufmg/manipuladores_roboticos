import numpy as np
import uaibot as ub
from geometry.board import Board

# ── Trail configuration ────────────────────────────────────────────────────────
_TRAIL_RADIUS = 0.004  # 4 mm spheres — visible but unobtrusive
_TRAIL_SAMPLE = 4  # place one marker every N control cycles
_TRAIL_SNAP_EPS = 1e-3  # seconds between the two snap keyframes
_TRAIL_DEFAULT_MAX = int(3e3)  # pre-allocated ball count; tune to trajectory length
_TRAIL_COLOR_START = (0.18, 0.52, 1.00)  # blue  — oldest marker
_TRAIL_COLOR_END = (1.00, 0.22, 0.22)  # red   — most recent marker


def _off_screen() -> np.matrix:
    """4×4 HTM with translation far outside the scene (used to hide objects)."""
    htm = np.identity(4)
    htm[0:3, 3] = [100.0, 100.0, 100.0]
    return np.matrix(htm)


def _trail_color(t: float) -> str:
    """
    Linear RGB interpolation between start and end trail colours.
    t ∈ [0, 1]:  0 → oldest (blue),  1 → newest (red).
    Returns a CSS hex string accepted by UAIbot's color argument.
    """
    c0, c1 = _TRAIL_COLOR_START, _TRAIL_COLOR_END
    r = int(255 * (c0[0] + (c1[0] - c0[0]) * t))
    g = int(255 * (c0[1] + (c1[1] - c0[1]) * t))
    b = int(255 * (c0[2] + (c1[2] - c0[2]) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


class Scene:
    def __init__(
        self, robot_interface, board: Board, max_trail_points: int = _TRAIL_DEFAULT_MAX
    ):
        self.robot = robot_interface.robot

        # ── Board ─────────────────────────────────────────────────────────────
        htm_board = np.identity(4)
        htm_board[0:3, 0] = board.x_axis
        htm_board[0:3, 1] = board.y_axis
        htm_board[0:3, 2] = board.normal
        # Shift the box backward along the normal so the surface lies exactly at origin
        thickness = 0.02
        htm_board[0:3, 3] = board.origin - board.normal * (thickness / 2.0)

        self.board_viz = ub.Box(
            name="board_surface",
            width=board.width,
            depth=board.height,
            height=thickness,
            htm=np.matrix(htm_board),
            color="white",
        )

        # ── Moving target marker (existing behaviour) ──────────────────────────
        self.target_marker = ub.Ball(
            name="target_marker",
            radius=0.015,
            color="cyan",
            htm=np.matrix(np.identity(4)),
        )

        # ── Trail markers ──────────────────────────────────────────────────────
        # All balls are pre-allocated and parked off-screen. During record_frame
        # each one is given two keyframes in rapid succession: one off-screen
        # (at t − ε) and one at the target position (at t), so it snaps into
        # place without drifting in from outside the scene.
        # Colours age from blue (oldest) to red (most recent).
        self._trail_balls = [
            ub.Ball(
                name=f"trail_{i}",
                radius=_TRAIL_RADIUS,
                color=_trail_color(i / max(max_trail_points - 1, 1)),
                htm=_off_screen(),
            )
            for i in range(max_trail_points)
        ]
        self._trail_idx = 0  # next ball to place
        self._cycle_count = 0  # cycles since the last marker was dropped

        self.sim = ub.Simulation(
            [self.robot, self.board_viz, self.target_marker, *self._trail_balls]
        )

    def record_frame(self, time: float, target_pos: np.ndarray, q_current: np.ndarray):
        """Saves a single animation step to the simulation."""
        self.robot.add_ani_frame(time=time, q=np.matrix(q_current))

        htm_target = np.identity(4)
        htm_target[0:3, 3] = target_pos.flatten()
        self.target_marker.add_ani_frame(time=time, htm=np.matrix(htm_target))

        # ── Trail: drop a marker every _TRAIL_SAMPLE cycles ───────────────────
        self._cycle_count += 1
        if self._cycle_count >= _TRAIL_SAMPLE and self._trail_idx < len(
            self._trail_balls
        ):
            self._cycle_count = 0
            ball = self._trail_balls[self._trail_idx]

            htm_ball = np.identity(4)
            htm_ball[0:3, 3] = target_pos.flatten()

            # Frame just before: still off-screen (avoids interpolated drift)
            ball.add_ani_frame(
                time=max(0.0, time - _TRAIL_SNAP_EPS),
                htm=_off_screen(),
            )
            # Frame at current time: snap to position and stay there permanently
            ball.add_ani_frame(time=time, htm=np.matrix(htm_ball))

            self._trail_idx += 1

    def render(self):
        """Executes the HTML rendering."""
        self.sim.run()
