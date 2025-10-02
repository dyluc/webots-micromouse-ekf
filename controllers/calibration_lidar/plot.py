"""
Generate a scatter plot of measured lidar distances vs. ground-truth distances
for each ray direction, showing how consistent the readings are.
"""

import numpy as np
import matplotlib.pyplot as plt

# load the saved calibration data
data = np.load("lidar_calibration_data.npz", allow_pickle=True)

calibration_distances = data["calibration_distances"]
readings_per_ray = data["readings_per_ray"].item() # dict
ray_names = data["ray_names"]

plt.figure(figsize=(8, 6))

for ray_name in ray_names:
    readings = np.array(readings_per_ray[ray_name])
    valid_mask = np.isfinite(readings) # drop the infs again
    readings = readings[valid_mask]
    ground_truths = np.array(calibration_distances[:len(readings)])
    plt.scatter(ground_truths, readings, label=ray_name, alpha=0.7)

plt.xlabel("Ground-Truth Distance (m)")
plt.ylabel("Measured Distance (m)")
plt.title("Lidar Calibration: Measured vs Ground-Truth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lidar_calibration_plot.png", dpi=300)
plt.show()
