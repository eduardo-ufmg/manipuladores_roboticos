import numpy as np
import uaibot as ub


class RobotInterface:
    def __init__(self):
        self.robot = ub.Robot.create_kuka_lbr_iiwa()
        self.n_joints = len(self.robot.q)

        limits = np.asarray(self.robot._joint_limit, dtype=float)
        self.q_min = limits[:, 0].reshape(-1, 1)
        self.q_max = limits[:, 1].reshape(-1, 1)

        self.q_current = np.asarray(self.robot.q, dtype=float).reshape(-1, 1)

    def update_joints(self, q_new: np.ndarray):
        """Updates the internal joint state."""
        self.q_current = q_new.reshape(-1, 1)

    def get_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the geometric Jacobian, current end-effector position,
        and current end-effector z-axis.
        """
        Jg, fk = self.robot.jac_geo(self.q_current)
        Jg = np.asarray(Jg, dtype=float)
        fk = np.asarray(fk, dtype=float)

        pos = fk[0:3, 3].reshape((3, 1))
        z_axis = fk[0:3, 2].reshape((3, 1))

        return Jg, pos, z_axis
