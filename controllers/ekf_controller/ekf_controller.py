from robot_simple import RobotWrapper
from algorithms import LeftWallHugger, RightWallHugger
import numpy as np

gt_log, od_log = [], []
steps_counter = 0
terminate = False
enable_ekf_updates = True
left_wall_hugger = False

# initialise the robot wrapper
robot_wrapper = RobotWrapper()

# slower, consistent motion improves ekf stability and makes corrections smoother
if left_wall_hugger:
    nav = LeftWallHugger(
        base_velocity=0.3 * robot_wrapper.max_velocity,
        distance_threshold=150)
else:
    nav = RightWallHugger(
        base_velocity=0.3 * robot_wrapper.max_velocity, 
        distance_threshold=150)

while robot_wrapper.supervisor.step(robot_wrapper.timestep) != -1 and not terminate:

    # uncalibrated proximity sensors
    ps = robot_wrapper.get_prox_sensors()
    lwv, rwv = nav.get_wheel_velocities(ps)
    robot_wrapper.set_wheel_velocity(lwv, rwv)

    # ground truth pose
    gt_pose = robot_wrapper.robot_node.getField("translation").getSFVec3f()
    gt_theta = robot_wrapper.robot_node.getField("rotation").getSFRotation()[3]
    gt_log.append(np.array([gt_pose[0], gt_pose[1], gt_theta]))
    print(f"Ground Truth: {gt_pose[0]}, {gt_pose[1]}, {gt_theta}")

    # odometry + ekf predict step
    robot_wrapper.update_odometry()
    pose = robot_wrapper.xhat
    print(f"Pose After EKF Predict: {pose}")

    if enable_ekf_updates:
        # ekf update step
        robot_wrapper.ekf_update()
        pose = robot_wrapper.xhat
        print(f"Pose After EKF Update: {pose}")

    # record corrected odometry
    od_log.append(pose)

    # terminate if reached centre, or more than 50000 steps
    steps_counter += 1
    if (-0.18 < pose[0] and pose[0] < 0.18) and (-0.18 < pose[1] and pose[1] < 0.18) or steps_counter == 50000:
        robot_wrapper.set_wheel_velocity(0.0, 0.0)
        terminate = True

# save the odometry data for plotting
lm_log = [l for v in robot_wrapper.lm_log.values() for l in v]
np.savez_compressed(
    "odometry_logs.npz",
    gt=np.array(gt_log),
    od=np.array(od_log),
    lm=np.array(lm_log)
)
print("Logs saved to odometry_logs.npz")