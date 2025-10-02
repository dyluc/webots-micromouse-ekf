import numpy as np
import matplotlib.pyplot as plt

# load the saved odometry data
data = np.load("odometry_logs.npz")
gt = data['gt']
od = data['od']

# label = "Odometry - Dead-Reckoning (EKF Predict Only)"
# fig_name = "robot_trajectory_comparison_dead_reckoning.png"

label = "Odometry - EKF Corrected Trajectory (Camera Landmark Observations)"
fig_name = "robot_trajectory_comparison_ekf.png"

# take ground truth and odometry x and y
gt_x, gt_y = gt[:, 0], gt[:, 1]
od_x, od_y = od[:, 0], od[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(gt_x, gt_y, c='black', s=2, label='Ground Truth')
plt.scatter(od_x, od_y, c='red', s=2, label=label)

plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('Robot Trajectory Comparison')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig(fig_name, dpi=300)
plt.show()
