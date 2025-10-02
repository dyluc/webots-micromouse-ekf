"""
Compute the estimated offset of IR sensor wall distance readings from
Front, Left, Back, and Right sensors.

Conclusions:

Findings contrast those of lidar. Low mean - very small offset required, higher variance

Front sensor:
  Estimated offset: 0.003 m
  Measurement variance: 0.001671 m^2
Left sensor:
  Estimated offset: 0.001 m
  Measurement variance: 0.001742 m^2
Back sensor:
  Estimated offset: -0.012 m
  Measurement variance: 0.001745 m^2
Right sensor:
  Estimated offset: -0.001 m
  Measurement variance: 0.000722 m^2
"""

from controller import Supervisor
import numpy as np
import math

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = supervisor.getFromDef("epuck")

# initialise the sensors
frontSensor = supervisor.getDevice('front distance sensor')
leftSensor  = supervisor.getDevice('left distance sensor')
rightSensor = supervisor.getDevice('right distance sensor')
rearSensor  = supervisor.getDevice('rear distance sensor')

ir_sensors = {
    "Front": frontSensor,
    "Left": leftSensor,
    "Back": rearSensor,
    "Right": rightSensor
}

for s in ir_sensors.values():
    s.enable(timestep)

print("\n#######################################################")
print("CALIBRATING IR SENSORS")
print("#######################################################\n")

# initialise test distances in metres (match lidar distances for consistency)
calibration_distances = [round(d, 2) for d in np.arange(0.9, 0, -0.01)]

translation_field = robot.getField("translation")
rotation_field = robot.getField("rotation")

# headings for each sensor to face the north wall
headings = {
    "Front": math.pi / 2.0,
    "Left":  0.0,
    "Back": -math.pi / 2.0,
    "Right": math.pi
}

# create dictionary to store readings per sensor
readings_per_sensor = {name: [] for name in ir_sensors.keys()}

for d in calibration_distances:
    print("-------------------------------------------------------")
    print(f"Placing robot at {d:.3f} m from the north wall")
    translation_field.setSFVec3f([-1.35, 1.393 - d, 0.0])

    for sensor_name, sensor in ir_sensors.items():
        theta = headings[sensor_name]
        print(f"Rotating to {theta:.3f} rad, reading from {sensor_name} sensor")
        rotation_field.setSFRotation([0, 0, 1, theta])

        # a few timesteps to let the sensor stabilise
        for _ in range(20):
            supervisor.step(timestep)

        reading = sensor.getValue()
        print(f"{sensor_name} sensor reading: {reading}")
        readings_per_sensor[sensor_name].append(reading)

# now compute the means (offsets) and variances from each sensor's errors
for sensor_name, readings in readings_per_sensor.items():
    readings = np.array(readings)
    valid_mask = np.isfinite(readings) # drop inf readings from closer distances
    readings = readings[valid_mask]

    if len(readings) == 0:
        print(f"{sensor_name} sensor has no valid readings.")
        continue

    # associate each reading with its ground-truth distance
    ground_truths = np.array(calibration_distances[:len(readings)])
    offset = np.mean(readings - ground_truths)
    variance = np.var(readings - ground_truths)

    print(f"\n{sensor_name} sensor:")
    print(f"  Estimated offset: {offset:.3f} m")
    print(f"  Measurement variance: {variance:.6f} m^2")

# save the calibration data for plotting
np.savez("ir_calibration_data.npz",
         calibration_distances=calibration_distances,
         readings_per_sensor=readings_per_sensor,
         sensor_names=list(readings_per_sensor.keys()))

print("\n#######################################################")
print("IR CALIBRATION COMPLETE")
print("#######################################################\n")
