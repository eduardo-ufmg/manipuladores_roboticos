import warnings
from dataclasses import dataclass

import numpy as np
import qpsolvers
import scipy.sparse as sp

warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
)


@dataclass(frozen=True)
class ControllerConfig:
    dt: float = 0.01
    k0: float = 1.0  # Joint centering gain
    lambda_d: float = 0.01  # Damping on joint velocities & slacks
    epsilon: float = 1e-3  # Joint limit margin
    K_pos: float = 5.0  # Task position gain
    K_ori: float = 2.0  # Task orientation gain


class QPController:
    def __init__(self, config: ControllerConfig, n_joints: int):
        self.config = config
        self.n = n_joints
        self.m = 4  # 3 positional + 1 orientation constraints

        # Hessian matrix P penalizes joint velocities and slack variables
        self.P_qp = np.block(
            [
                [
                    self.config.lambda_d * np.identity(self.n),
                    np.zeros((self.n, self.m)),
                ],
                [np.zeros((self.m, self.n)), np.identity(self.m)],
            ]
        )

        self.K_matrix = np.diag(
            [self.config.K_pos, self.config.K_pos, self.config.K_pos, self.config.K_ori]
        )

    def solve(
        self,
        q_current: np.ndarray,
        q_min: np.ndarray,
        q_max: np.ndarray,
        Jr: np.ndarray,
        r_error: np.ndarray,
        v_des: np.ndarray,
    ) -> np.ndarray:
        """
        Solves the velocity-level inverse kinematics QP.
        Returns the optimal joint velocity vector dot(q).
        """
        q_current = q_current.reshape(-1, 1)
        q_min = q_min.reshape(-1, 1)
        q_max = q_max.reshape(-1, 1)
        r_error = r_error.reshape(-1, 1)
        v_des = v_des.reshape(3, 1)

        # 1. Joint limit avoidance (Secondary objective)
        q_mean = (q_max + q_min) / 2.0
        q_range_sq = np.square(q_max - q_min)
        q_range_sq[q_range_sq < 1e-6] = 1e-6  # Prevent division by zero

        grad_H = (q_current - q_mean) / q_range_sq
        q_dot_0 = -self.config.k0 * grad_H

        # 2. Task Right-Hand Side constraint formulation
        feedforward = np.vstack([v_des, [[0.0]]])
        task_rhs = -self.K_matrix @ r_error + feedforward

        # 3. Assemble QP matrices
        q_qp = np.concatenate(
            [
                -self.config.lambda_d * q_dot_0.flatten(),
                np.zeros(self.m),
            ]
        )

        # Equality constraint: Jr * dot(q) - w = task_rhs
        A_qp = np.hstack([Jr, -np.identity(self.m)])

        # Inequality constraints: limit bounds translated to velocities
        lb_dq = ((q_min - q_current) / self.config.dt + self.config.epsilon).flatten()
        ub_dq = ((q_max - q_current) / self.config.dt - self.config.epsilon).flatten()

        lb_qp = np.concatenate([lb_dq, np.full(self.m, -np.inf)])
        ub_qp = np.concatenate([ub_dq, np.full(self.m, np.inf)])

        P_qp_sparse = sp.csc_matrix(self.P_qp)
        A_qp_sparse = sp.csc_matrix(A_qp)

        solution = qpsolvers.solve_qp(
            P=P_qp_sparse,
            q=q_qp,
            A=A_qp_sparse,
            b=task_rhs.flatten(),
            lb=lb_qp,
            ub=ub_qp,
            solver="osqp",
        )

        if solution is None:
            raise RuntimeError("QP solver infeasible. Simulation step failed.")

        return solution[: self.n].reshape((self.n, 1))
