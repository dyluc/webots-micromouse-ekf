import numpy as np
import matplotlib.pyplot as plt

# load the saved calibration data
data = np.load("turn_errors.npz")
error = data['error']
turn_angle = data['turn_angle']
num_turns = len(error)

plt.figure(figsize=(8,4))
plt.plot(range(1, num_turns+1), error, marker='o', linestyle='-', color='r', label="Rotation error (deg)")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Turn number")
plt.ylabel("Error (deg)")
plt.title(f"Per-turn Rotation Error (Target {turn_angle}°/turn)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("post_calibrated_epuck_configurations.png", dpi=300)
plt.show()
