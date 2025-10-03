from robot_simple import RobotWrapper
from algorithms import LeftWallHugger, RightWallHugger
import numpy as np

# initialise the robot wrapper
robot_wrapper = RobotWrapper()

gt_log, od_log = [], []
steps_counter = 0
terminate = False

# slower, consistent motion improves ekf stability and smoothes corrections
nav = LeftWallHugger(
    base_velocity=0.3 * robot_wrapper.max_velocity,
    distance_threshold=150)

# nav = RightWallHugger(
#     base_velocity=0.3 * robot_wrapper.max_velocity, 
#     distance_threshold=150)

while robot_wrapper.supervisor.step(robot_wrapper.timestep) != -1 and not terminate:

    # uncalibrated proximity sensors
    ps = robot_wrapper.get_prox_sensors()
    lwv, rwv = nav.get_wheel_velocities(ps)
    robot_wrapper.set_wheel_velocity(lwv, rwv)

    # ground truth pose
    gt_pose = robot_wrapper.robot_node.getField("translation").getSFVec3f()
    gt_theta = robot_wrapper.robot_node.getField("rotation").getSFRotation()[3]
    gt_log.append([gt_pose[0], gt_pose[1], gt_theta])
    print(f"Ground Truth: {gt_pose[0]}, {gt_pose[1]}, {gt_theta}")

    # odometry + ekf predict step
    robot_wrapper.update_odometry()
    pose = robot_wrapper.get_pose()
    print(f"Pose After EKF Predict: {pose}")

    # ekf update step
    robot_wrapper.ekf_update()
    pose = robot_wrapper.get_pose()
    print(f"Pose After EKF Update: {pose}")

    # record corrected odometry
    od_log.append(pose)

    # terminate if reached centre, or more than 50000 steps
    steps_counter += 1
    if (-0.18 < pose[0] and pose[0] < 0.18) and (-0.18 < pose[1] and pose[1] < 0.18) or steps_counter == 50000:
        robot_wrapper.set_wheel_velocity(0.0, 0.0)
        terminate = True

# save the odometry data for plotting
np.savez_compressed(
    "odometry_logs.npz",
    gt=np.array(gt_log),
    od=np.array(od_log)
)
print("Logs saved to odometry_logs.npz")