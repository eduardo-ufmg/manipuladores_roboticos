import numpy as np

from geometry.cylinder import Cylinder
from trajectory.planner import WritingPlanner
from control.tasks import compute_task_error, compute_task_jacobian
from control.qp_controller import QPController, ControllerConfig
from simulation_wrapper.robot_interface import RobotInterface
from simulation_wrapper.scene import Scene


def execute_writing_task(code: str, headless: bool = False) -> float:
    """
    Executes the full pipeline for a given code on a Horizontal Cylinder.
    Returns the maximum tracking error observed during the run.
    """

    robot_if = RobotInterface()
    Jg, p_home, z_home = robot_if.get_kinematics()

    center = np.array([0.8, 0.0, 0.5])

    u_axis = np.array([0.0, -1.0, 0.0])

    v_tangent = np.array([0.0, 0.0, 1.0])

    normal_ref = np.array([-1.0, 0.0, 0.0])

    surface = Cylinder(
        radius=0.15,
        width=1.00,
        height=0.25,
        center=center,
        u_axis=u_axis,
        v_tangent=v_tangent,
        normal_ref=normal_ref,
        safety_offset=0.01,
    )

    planner = WritingPlanner(surface)

    temp_write_traj = planner.plan(code)
    p_write_start = temp_write_traj[0].position
    write_normal = temp_write_traj[0].normal

    engage_traj, t1 = planner.generate_cartesian_cubic_motion(
        p0=p_home.flatten(),
        pf=p_write_start,
        target_normal=write_normal,
        speed=planner.transition_speed,
        start_time=0.0,
    )

    write_traj = planner.plan(code, start_time=t1)
    t2 = write_traj[-1].timestamp
    p_write_end = write_traj[-1].position

    retract_traj, _ = planner.generate_cartesian_cubic_motion(
        p0=p_write_end,
        pf=p_home.flatten(),
        target_normal=write_normal,
        speed=planner.transition_speed,
        start_time=t2,
    )

    full_trajectory = engage_traj + write_traj + retract_traj

    config = ControllerConfig(dt=planner.dt, K_pos=15.0, K_ori=5.0)
    controller = QPController(config, robot_if.n_joints)

    scene = None if headless else Scene(robot_if, surface)

    max_error = 0.0

    gamma = 15.0
    d_safe = 0.01

    for pt in full_trajectory:
        Jg, current_pos, current_z = robot_if.get_kinematics()

        r_error = compute_task_error(current_pos, current_z, pt.position, pt.normal)
        max_error = max(max_error, np.linalg.norm(r_error[0:3]))

        Jr = compute_task_jacobian(Jg, current_z, pt.normal)

        dist, surf_normal = surface.compute_distance(current_pos)
        Jv = Jg[0:3, :]

        G_ineq = -surf_normal.reshape(1, 3) @ Jv
        h_ineq = np.array([gamma * (dist - d_safe)])

        q_dot = controller.solve(
            q_current=robot_if.q_current,
            q_min=robot_if.q_min,
            q_max=robot_if.q_max,
            Jr=Jr,
            r_error=r_error,
            v_des=pt.velocity,
            G_ineq=G_ineq,
            h_ineq=h_ineq,
        )

        q_new = robot_if.q_current + q_dot * config.dt
        robot_if.update_joints(q_new)

        if not headless:
            scene.record_frame(pt.timestamp, pt.position, robot_if.q_current, is_writing=pt.is_writing)  # type: ignore

    if not headless:
        scene.render()  # type: ignore

    return float(max_error)


if __name__ == "__main__":

    while True:
        try:
            code = input("Enter a 3-digit code (or 'exit' to quit): ")
            if code.lower() == "exit":
                break
            if len(code) != 3 or not code.isdigit():
                print("Invalid input. Please enter a 3-digit code.")
                continue

            max_tracking_error = execute_writing_task(code, headless=False)
            print(
                f"Maximum tracking error for code {code}: {max_tracking_error:.6f} meters"
            )

        except KeyboardInterrupt:
            print("\nExecution interrupted by user.")
            break
