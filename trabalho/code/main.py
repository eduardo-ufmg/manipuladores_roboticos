import time
import math
import random

import numpy as np
from tqdm import tqdm

from geometry.cylinder import Cylinder
from trajectory.planner import WritingPlanner
from control.tasks import compute_task_error, compute_task_jacobian
from control.qp_controller import QPController, ControllerConfig
from simulation_wrapper.robot_interface import RobotInterface
from simulation_wrapper.scene import Scene

#####################################################################################
# MODEL REPO UNREACHABLE WORKAROUND
#####################################################################################

import sys
import uaibot as ub

# Add the /content/code directory to sys.path to ensure local modules can be imported
# This allows importing `RobotInterface` from the `simulation_wrapper` package.
if "/content/code" not in sys.path:
    sys.path.insert(0, "/content/code")

# Import the original RobotInterface class from the local code
from simulation_wrapper.robot_interface import RobotInterface

# Define the OLD and NEW URL prefixes based on the suggested Fix 1
OLD_CDN_PREFIX = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/"
NEW_CDN_PREFIX = "https://raw.githubusercontent.com/UAIbot/uaibot_data/master/"

# Store the original __init__ method of RobotInterface
_original_robot_interface_init = RobotInterface.__init__


# Define a patched __init__ method for RobotInterface
def _patched_robot_interface_init(self, *args, **kwargs):
    # Call the original __init__ method to initialize the robot object and its components
    _original_robot_interface_init(self, *args, **kwargs)

    # After the original initialization, iterate through the robot's links
    # and their 3D models to update the asset URLs. This intervenes before
    # `Model3D.gen_code()` is called, which is where the URL check happens.
    if hasattr(self, "robot") and isinstance(self.robot, ub.robot.Robot):  # type: ignore
        for link in self.robot.links:
            # Each link in UAIbot's robot model typically contains a list of 3D models
            if hasattr(link, "_list_model_3d"):
                for model_3d in link._list_model_3d:  # type: ignore
                    if isinstance(model_3d, ub.graphics.model3d.Model3D):
                        # If the model's URL starts with the old CDN prefix, replace it
                        if model_3d.url and model_3d.url.startswith(OLD_CDN_PREFIX):
                            model_3d.url = model_3d.url.replace(OLD_CDN_PREFIX, NEW_CDN_PREFIX)  # type: ignore


# Apply the monkey-patch: replace the original __init__ with the patched one
# This patch will be active for any subsequent instantiations of RobotInterface
# in the notebook, including the one in the `execute_writing_task` function.
RobotInterface.__init__ = _patched_robot_interface_init

#####################################################################################

CYL_DIM_RADIUS_RANGE = (0.1, 0.6)
CYL_DIM_WIDTH_RANGE = (0.4, 1.0)

IIWA_QD_MAX = np.array(
    [1.7104, 1.7104, 1.7453, 2.2689, 2.4435, 3.1416, 3.1416], dtype=float
)  # from https://xpert.kuka.com/service-express/api/latest/resource/environment/project1_p/documents/kukaid/PB19378/common_PB19378_en.pdf


def generate_random_cylinder() -> Cylinder:
    radius = random.uniform(*CYL_DIM_RADIUS_RANGE)
    width = random.uniform(*CYL_DIM_WIDTH_RANGE)
    height = np.pi * radius

    center = np.array([0.8 + (radius - 0.15), 0.0, 0.5])
    u_axis = np.array([0.0, -1.0, 0.0])
    v_tangent = np.array([0.0, 0.0, 1.0])
    normal_ref = np.array([-1.0, 0.0, 0.0])

    print(f"Generated cylinder: radius={radius:.3f}, width={width:.3f}")

    return Cylinder(
        radius=radius,
        width=width,
        height=height,
        center=center,
        u_axis=u_axis,
        v_tangent=v_tangent,
        normal_ref=normal_ref,
        safety_offset=0.01,
    )


def execute_writing_task(code: str, headless: bool = False) -> float:
    """
    Executes the full pipeline for a given code on a Horizontal Cylinder.
    Returns the maximum tracking error observed during the run.
    """

    robot_if = RobotInterface()
    Jg, p_home, z_home = robot_if.get_kinematics()

    surface = generate_random_cylinder()

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

    n_steps = len(full_trajectory)
    n_engage = len(engage_traj)
    n_write = len(write_traj)
    n_retract = len(retract_traj)

    phase_labels = ["engage"] * n_engage + ["write"] * n_write + ["retract"] * n_retract

    q_exec_min = np.full(robot_if.n_joints, np.inf)
    q_exec_max = np.full(robot_if.n_joints, -np.inf)
    qd_exec_min = np.full(robot_if.n_joints, np.inf)
    qd_exec_max = np.full(robot_if.n_joints, -np.inf)

    t_loop_start = time.perf_counter()

    with tqdm(
        total=n_steps,
        desc="Control loop",
        unit="step",
        dynamic_ncols=True,
        postfix={"phase": "engage", "max_err": 0.0},
    ) as pbar:
        for i, pt in enumerate(full_trajectory):
            Jg, current_pos, current_z = robot_if.get_kinematics()

            r_error = compute_task_error(current_pos, current_z, pt.position, pt.normal)
            step_err = float(np.linalg.norm(r_error[0:3]))
            max_error = max(max_error, step_err)

            Jr = compute_task_jacobian(Jg, current_z, pt.normal)

            dist, surf_normal = surface.compute_distance(current_pos)
            Jv = Jg[0:3, :]

            G_ineq = -surf_normal.reshape(1, 3) @ Jv
            h_ineq = np.array([gamma * (dist - d_safe)])

            q_dot = controller.solve(
                q_current=robot_if.q_current,
                q_min=robot_if.q_min,
                q_max=robot_if.q_max,
                qd_max=IIWA_QD_MAX,
                Jr=Jr,
                r_error=r_error,
                v_des=pt.velocity,
                G_ineq=G_ineq,
                h_ineq=h_ineq,
            )

            q_dot = np.clip(
                q_dot, -IIWA_QD_MAX.reshape(-1, 1), IIWA_QD_MAX.reshape(-1, 1)
            )

            q_exec_min = np.minimum(q_exec_min, robot_if.q_current)
            q_exec_max = np.maximum(q_exec_max, robot_if.q_current)
            qd_exec_min = np.minimum(qd_exec_min, q_dot)
            qd_exec_max = np.maximum(qd_exec_max, q_dot)

            q_new = robot_if.q_current + q_dot * config.dt

            q_new = np.clip(q_new, robot_if.q_min, robot_if.q_max)

            robot_if.update_joints(q_new)

            if not headless:
                scene.record_frame(  # type: ignore
                    pt.timestamp,
                    pt.position,
                    robot_if.q_current,
                    is_writing=pt.is_writing,
                )

            pbar.set_postfix({"phase": phase_labels[i], "max_err": f"{max_error:.4f}"})
            pbar.update(1)

    t_loop_elapsed = time.perf_counter() - t_loop_start
    print(
        f"[timing] Control loop : {t_loop_elapsed:.3f} s  ({n_steps} steps, "
        f"{t_loop_elapsed / n_steps * 1e3:.2f} ms/step)"
    )

    print("\nJoint limit summary")
    print("-" * 120)
    print(
        f"{'Joint':<5} {'Pos Limits':<15} {'Executed Pos':<16} {'Vel Lim':<8} {'Executed Vel':<24}"
    )
    for i in range(robot_if.n_joints):
        print(
            f"J{i+1:<4} [{robot_if.q_min[i][0]:6.3f},{robot_if.q_max[i][0]:6.3f}] "
            f"[{q_exec_min[i][0]:6.3f},{q_exec_max[i][0]:6.3f}] "
            f"{IIWA_QD_MAX[i]:6.3f}\t"
            f"[{qd_exec_min[i][0]:6.3f},{qd_exec_max[i][0]:6.3f}]"
        )
    print("-" * 120)

    if not headless:
        t_render_start = time.perf_counter()
        scene.render()  # type: ignore
        t_render_elapsed = time.perf_counter() - t_render_start
        print(f"[timing] Render       : {t_render_elapsed:.3f} s")

    return float(max_error)


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
        break

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        break
