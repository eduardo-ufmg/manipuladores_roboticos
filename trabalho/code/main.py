import numpy as np
from geometry.board import Board
from geometry.transforms import create_orthonormal_frame
from trajectory.planner import WritingPlanner
from control.tasks import compute_task_error, compute_task_jacobian
from control.qp_controller import QPController, ControllerConfig
from simulation_wrapper.robot_interface import RobotInterface
from simulation_wrapper.scene import Scene


def execute_writing_task(code: str, headless: bool = False) -> float:
    """
    Executes the full pipeline for a given code.
    Returns the maximum tracking error observed during the run.
    """
    # 1. Initialize Geometry (Positioned in front of the robot)
    origin = np.array([0.5, 0.0, 0.5])
    normal = np.array([1.0, 0.0, 0.0])
    up_ref = np.array([0.0, 0.0, 1.0])
    x, y, z = create_orthonormal_frame(normal, up_ref)

    board = Board(width=0.60, height=0.25, origin=origin, x_axis=x, y_axis=y, normal=z)

    # 2. Generate Trajectory
    planner = WritingPlanner(board)
    trajectory = planner.plan(code)

    # 3. Initialize Robot & Controller
    robot_if = RobotInterface()
    config = ControllerConfig(dt=planner.dt, K_pos=15.0, K_ori=5.0)
    controller = QPController(config, robot_if.n_joints)

    # 4. Initialize Simulation Scene
    scene = None if headless else Scene(robot_if, board)

    # 5. Closed-Loop Execution
    max_error = 0.0

    for pt in trajectory:
        Jg, current_pos, current_z = robot_if.get_kinematics()

        r_error = compute_task_error(current_pos, current_z, pt.position, pt.normal)
        max_error = max(max_error, np.linalg.norm(r_error[0:3]))

        Jr = compute_task_jacobian(Jg, current_z, pt.normal)

        q_dot = controller.solve(
            q_current=robot_if.q_current,
            q_min=robot_if.q_min,
            q_max=robot_if.q_max,
            Jr=Jr,
            r_error=r_error,
            v_des=pt.velocity,
        )

        q_new = robot_if.q_current + q_dot * config.dt
        robot_if.update_joints(q_new)

        if not headless:
            scene.record_frame(pt.timestamp, pt.position, robot_if.q_current)  # type: ignore - if not headless, scene is not None

    if not headless:
        scene.render()  # type: ignore - if not headless, scene is not None

    return float(max_error)


if __name__ == "__main__":
    # Execute a visual test
    execute_writing_task("666", headless=False)
