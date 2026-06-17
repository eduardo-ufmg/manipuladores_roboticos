import numpy as np
import uaibot as ub
from geometry.board import Board


class Scene:
    def __init__(self, robot_interface, board: Board):
        self.robot = robot_interface.robot

        # Orient the board box to match the orthonormal frame
        htm_board = np.identity(4)
        htm_board[0:3, 0] = board.x_axis
        htm_board[0:3, 1] = board.y_axis
        htm_board[0:3, 2] = board.normal
        # Shift the box backward along the normal so the surface lies exactly at origin
        thickness = 0.02
        htm_board[0:3, 3] = board.origin + board.normal * (thickness / 0.5)

        self.board_viz = ub.Box(
            name="board_surface",
            width=board.width,
            depth=board.height,
            height=thickness,
            htm=np.matrix(htm_board),
            color="white",
        )

        # A small marker to trace the target trajectory
        self.target_marker = ub.Ball(
            name="target_marker",
            radius=0.015,
            color="cyan",
            htm=np.matrix(np.identity(4)),
        )

        self.sim = ub.Simulation([self.robot, self.board_viz, self.target_marker])

    def record_frame(self, time: float, target_pos: np.ndarray, q_current: np.ndarray):
        """Saves a single animation step to the simulation."""
        self.robot.add_ani_frame(time=time, q=np.matrix(q_current))

        htm_target = np.identity(4)
        htm_target[0:3, 3] = target_pos.flatten()
        self.target_marker.add_ani_frame(time=time, htm=np.matrix(htm_target))

    def render(self):
        """Executes the HTML rendering."""
        self.sim.run()
