"""
Compute the estimated offset of lidar wall distance readings from 0, 90, 180, and 270 degrees.
All estimated offsets are ~0.040 m (likely just something I haven't accounted for - epuck radius, lidar translation, etc). 

Conclusions:

Very little variance in observations, which is great—lidar readings are very consistent.

Lidar translation vector: (0.01, 0.0, 0.01)
Rotation vector: (0, 0, 1, pi)

Front ray:
    Estimated offset: 0.040 m
    Measurement variance: 0.000142 m^2
Left ray:
    Estimated offset: 0.040 m
    Measurement variance: 0.000161 m^2
Back ray:
    Estimated offset: 0.041 m
    Measurement variance: 0.000134 m^2
Right ray:
    Estimated offset: 0.039 m
    Measurement variance: 0.000137 m^2
"""

from controller import Supervisor
import numpy as np
import math

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = supervisor.getFromDef("epuck")

# initialise the sensors
lidar = supervisor.getDevice("lidar")
lidar.enable(timestep)
lidar_max_range = lidar.getMaxRange()

print("\n#######################################################")
print("CALIBRATING LIDAR")
print("#######################################################\n")

# initialise test distances in metres
calibration_distances = [round(d, 2) for d in np.arange(0.9, 0, -0.01)] # 0.9 ~ below lidar_max_range - estimated offset

# robot fields
translation_field = robot.getField("translation")
rotation_field = robot.getField("rotation")

# angles for both the robot heading to face the north wall in rads, and the related ray to activate in degs
angles = {
    "Front": (math.pi / 2.0, 0), # robot heading, lidar index
    "Left":  (0.0, 270),
    "Back": (-math.pi / 2.0, 180),
    "Right": (math.pi, 90)
}

# create dictionary to store readings per ray
readings_per_ray = {name: [] for name in angles.keys()}

# 1.44 (map edge) - 0.01 (wall thickness) - 0.037 (~epuck radius) = 1.393 (starting y)
for d in calibration_distances:
    print("-------------------------------------------------------")
    print(f"Placing robot at {d:.3f} m from the north wall")
    translation_field.setSFVec3f([-1.35, 1.393 - d, 0.0])

    # thetas = angles[ray_name][0]
    # degrees = angles[ray_name][1]
    for ray_name, a in angles.items():
        theta = angles[ray_name][0]
        rd = angles[ray_name][1] # ray degrees
        print(f"Rotating to {theta:.3f} rad, reading from {rd} degrees")
        rotation_field.setSFRotation([0, 0, 1, theta])

        # a few timesteps to let the sensor stabilise
        for _ in range(20):
            supervisor.step(timestep)

        image = lidar.getRangeImage()
        reading = image[rd]
        print(f"{ray_name} ray reading: {reading}")   
        readings_per_ray[ray_name].append(reading)
  
# now compute the means (offsets) and variances from each ray's errors
for ray_name, readings in readings_per_ray.items():
    readings = np.array(readings)
    valid_mask = np.isfinite(readings) # drop inf readings from closer distances
    readings = readings[valid_mask]

    if len(readings) == 0:
        print(f"{ray_name} ray has no valid readings.")
        continue

    # associate each reading with its ground-truth distance
    ground_truths = np.array(calibration_distances[:len(readings)])
    offset = np.mean(readings - ground_truths)
    variance = np.var(readings - ground_truths)

    print(f"\n{ray_name} ray:")
    print(f"  Estimated offset: {offset:.3f} m")
    print(f"  Measurement variance: {variance:.6f} m^2")


# save the calibration data for plotting
np.savez("lidar_calibration_data.npz",
         calibration_distances=calibration_distances,
         readings_per_ray=readings_per_ray,
         ray_names=list(readings_per_ray.keys()))

print("\n#######################################################")
print("LIDAR CALIBRATION COMPLETE")
print("#######################################################\n")
