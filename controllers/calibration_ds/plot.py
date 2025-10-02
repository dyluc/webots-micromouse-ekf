"""
Generate a plot of measured IR distances vs. ground-truth distances
for each sensor direction, showing how consistent the readings are.
"""

import numpy as np
import matplotlib.pyplot as plt

# load the saved calibration data
data = np.load("ir_calibration_data.npz", allow_pickle=True)

calibration_distances = data["calibration_distances"]
readings_per_sensor = data["readings_per_sensor"].item() # dict
sensor_names = data["sensor_names"]

plt.figure(figsize=(8, 6))

for sensor in sensor_names:
    readings = np.array(readings_per_sensor[sensor])
    valid_mask = np.isfinite(readings) # drop the infs again
    readings = readings[valid_mask]
    ground_truths = np.array(calibration_distances[:len(readings)])
    plt.scatter(ground_truths, readings, label=sensor, alpha=0.7)

plt.xlabel("Ground-Truth Distance (m)")
plt.ylabel("Measured Distance (m)")
plt.title("IR Calibration: Measured vs Ground-Truth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ir_calibration_plot.png", dpi=300)
plt.show()
