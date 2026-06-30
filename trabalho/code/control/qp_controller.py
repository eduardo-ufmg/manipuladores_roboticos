import numpy as np
import qpsolvers
from dataclasses import dataclass

from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class ControllerConfig:
    dt: float = 0.01
    k0: float = 1.0
    lambda_d: float = 0.01
    epsilon: float = 1e-3
    K_pos: float = 5.0
    K_ori: float = 2.0


class QPController:
    def __init__(self, config: ControllerConfig, n_joints: int):
        self.config = config
        self.n = n_joints
        self.m = 4

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
        qd_max: np.ndarray,
        Jr: np.ndarray,
        r_error: np.ndarray,
        v_des: np.ndarray,
        G_ineq: np.ndarray | None = None,
        h_ineq: np.ndarray | None = None,
    ) -> np.ndarray:

        q_current = q_current.reshape(-1, 1)
        q_min = q_min.reshape(-1, 1)
        q_max = q_max.reshape(-1, 1)
        qd_max = qd_max.reshape(-1, 1)
        r_error = r_error.reshape(-1, 1)
        v_des = v_des.reshape(3, 1)

        q_mean = (q_max + q_min) / 2.0
        q_range_sq = np.square(q_max - q_min)
        q_range_sq[q_range_sq < 1e-6] = 1e-6

        grad_H = (q_current - q_mean) / q_range_sq
        q_dot_0 = -self.config.k0 * grad_H

        feedforward = np.vstack([v_des, [[0.0]]])
        task_rhs = -self.K_matrix @ r_error + feedforward

        q_qp = np.concatenate(
            [
                -self.config.lambda_d * q_dot_0.flatten(),
                np.zeros(self.m),
            ]
        )

        A_qp = np.hstack([Jr, -np.identity(self.m)])

        lb_dq = np.maximum(
            -qd_max, (q_min - q_current) / self.config.dt + self.config.epsilon
        ).flatten()

        ub_dq = np.minimum(
            qd_max, (q_max - q_current) / self.config.dt - self.config.epsilon
        ).flatten()

        lb_qp = np.concatenate([lb_dq, np.full(self.m, -np.inf)])
        ub_qp = np.concatenate([ub_dq, np.full(self.m, np.inf)])

        G_qp = None
        h_qp = None
        if G_ineq is not None and h_ineq is not None:
            k_constraints = G_ineq.shape[0]
            G_qp = np.hstack([G_ineq, np.zeros((k_constraints, self.m))])
            h_qp = h_ineq.flatten()

        G_sparse = csc_matrix(G_qp) if G_qp is not None else None

        solution = qpsolvers.solve_qp(
            P=csc_matrix(self.P_qp),
            q=q_qp,
            A=csc_matrix(A_qp),
            b=task_rhs.flatten(),
            G=G_sparse,
            h=h_qp,
            lb=lb_qp,
            ub=ub_qp,
            solver="osqp",
        )

        if solution is None:
            raise RuntimeError(
                "QP solver infeasible. Safety constraint likely conflicts with joint limits."
            )

        return solution[: self.n].reshape((self.n, 1))
