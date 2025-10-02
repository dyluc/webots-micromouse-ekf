"""
Conclusions:

Avoid integrating measurements when object distance is < ~0.4m, otherwise we risk distorting the 
odometry with noisy readings. Above ~0.4m with landmark sizes 0.5x0.5x0.5, field of view 90 degrees
keeps the estimates stable and variance low.

The distance bias is roughly ~0.027 m

Distance offset: -0.027 m, variance: 0.000009 m^2
Bearing offset:  -0.01 deg, variance: 0.04 deg^2
"""

from controller import Supervisor
import numpy as np
import math

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = supervisor.getFromDef("epuck")

# initialise the camera and enable recognition
camera = supervisor.getDevice("camera1")
camera.enable(timestep)
camera.recognitionEnable(timestep)

landmark1 = np.array([-1.35, -0.09, 0.0])

# test positions in x, y, z
test_positions = [
    # [-1.35, -0.20, 0.0],
    # [-1.35, -0.22, 0.0],
    # [-1.35, -0.24, 0.0],
    # [-1.35, -0.26, 0.0],
    # [-1.35, -0.28, 0.0],
    [-1.35, -0.30, 0.0], # above produce larger errors, > ~0.3 metres produces more reliable enough bearing measurements
    [-1.35, -0.32, 0.0],
    [-1.35, -0.34, 0.0],
    [-1.35, -0.36, 0.0],
    [-1.35, -0.38, 0.0],
    [-1.35, -0.40, 0.0],
    [-1.35, -0.42, 0.0],
    [-1.35, -0.46, 0.0],
    [-1.35, -0.48, 0.0],
    [-1.35, -0.50, 0.0],
    [-1.35, -0.52, 0.0],
    [-1.35, -0.54, 0.0],
    [-1.35, -0.63, 0.0],
    [-1.35, -0.72, 0.0],
    [-1.35, -0.81, 0.0],
    [-1.35, -0.90, 0.0],
    [-1.35, -0.99, 0.0],
]

translation_field = robot.getField("translation")

# initialise errors lists
distance_errors = []
bearing_errors  = []

print("\n#######################################################")
print("CALIBRATING CAMERA DISTANCES AND BEARINGS")
print("#######################################################\n")

rotation_field = robot.getField("rotation")
robot_yaw = 1.57 # start
angle_range = np.deg2rad(40) # +- 40 degs

# create 9 test yaws from -40 to +40 degs around current heading
test_yaws = np.linspace(robot_yaw - angle_range, robot_yaw + angle_range, 9)

for pos in test_positions:
    pos = np.array(pos)
    print(f"Placing robot at {pos}")
    translation_field.setSFVec3f(pos.tolist())
    
    for yaw in test_yaws:
        # rotate to correct yaw
        rotation_field.setSFRotation([0, 0, 1, yaw])
        
        for _ in range(20):
            supervisor.step(timestep) # a few timesteps to let the camera stabilise
        
        recognition_objects = camera.getRecognitionObjects()
        obj = recognition_objects[0] # landmark1 is the only one in view
        obj_pos = np.array(obj.getPosition())

        # ground truth distances and bearings
        delta = landmark1[:2] - pos[:2]
        gt_distance = np.linalg.norm(delta)
        gt_bearing = math.atan2(delta[1], delta[0]) - yaw
        gt_bearing = (gt_bearing + np.pi) % (2*np.pi) - np.pi # wrapped to [-pi, pi]

        # compute the measured distance and bearing (same as in robot_simple)
        measured_distance = np.hypot(obj_pos[0], obj_pos[1]) # x, y (pos[0]**2 + pos[1]**2)**0.5)
        measured_bearing = math.atan2(obj_pos[1], obj_pos[0]) # y, x

        # append errors to lists
        distance_errors.append(measured_distance - gt_distance)
        bearing_errors.append(measured_bearing - gt_bearing)

        print(f"Yaw {math.degrees(yaw):.1f}°: Measured {measured_distance:.3f} m, Measured bearing: {measured_bearing}, "
              f"Ground truth {gt_distance:.3f} m, Ground truth bearing {gt_bearing}, Bearing error {math.degrees(measured_bearing - gt_bearing):.2f}°")

# now compute the means (offsets) and variances from errors
d_errors = np.array(distance_errors)
b_errors = np.array(bearing_errors)

d_offset = np.mean(d_errors)
d_variance = np.var(d_errors)

b_offset = np.mean(b_errors)
b_variance = np.var(b_errors)

print(f"\nDistance offset: {d_offset:.3f} m, variance: {d_variance:.6f} m^2")
print(f"Bearing offset:  {math.degrees(b_offset):.2f} deg, variance: {math.degrees(b_variance):.2f} deg^2")

# save the calibration data for plotting
np.savez("camera_calibration_data.npz",
         test_positions=test_positions,
         distance_errors=distance_errors,
         bearing_errors=bearing_errors)

print("\n#######################################################")
print("CAMERA CALIBRATION COMPLETE")
print("#######################################################\n")
