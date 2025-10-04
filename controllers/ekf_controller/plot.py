import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.image as mpimg

# load the saved odometry data
data = np.load("odometry_logs.npz")
gt = data["gt"]
od = data["od"]
lm = data["lm"]

# title = "Dead-Reckoning (EKF Predict Only): Right Wall Navigation"
# fig_name = "robot_trajectory_comparison_dead_reckoning_rightwall.png"

title = "EKF Corrected Trajectory (Camera Landmark Observations): Right Wall Navigation"
fig_name = "robot_trajectory_comparison_ekf_rightwall.png"

gt_x, gt_y = gt[:, 0], gt[:, 1]
od_x, od_y = od[:, 0], od[:, 1]

plt.figure(figsize=(10, 7))

# customise the legend a little
gt_handle = mlines.Line2D([], [], color="black", linestyle="-", linewidth=2, label="Ground Truth")
od_handle = mlines.Line2D([], [], color="orange", linestyle="-", linewidth=2, label="Odometry")

handles = [gt_handle, od_handle]

# ground truth plot as black lines
plt.scatter(gt_x, gt_y, c="black", s=5.0, label="Ground Truth")

# odometry plot as dashed red lines
plt.scatter(od_x, od_y, c="orange", s=1.0, label="Odometry")

# landmark observations plot as green lines
if lm.size > 0:
    lm_x, lm_y = lm[:, 0], lm[:, 1]
    lm_label = "Landmark Corrections \n(EKF Update Points \n< 0.4m Distance)"
    plt.scatter(lm_x, lm_y, c="lime", s=0.5, label=lm_label)
    lm_handle = mlines.Line2D([], [], color="lime", linestyle="-", linewidth=2, label=lm_label)
    handles.append(lm_handle)

    # plot landmark locations
    lm_loc = np.array([
        [-1.35, -0.09],
        [-0.09, -1.17],
        [0.09, 0.27],
        [-0.09, 0.45],
        [0.99, 0.45],
        [0.81, -0.45],
        [0.09, -1.35]
    ])
    plt.scatter(lm_loc[:, 0], lm_loc[:, 1], c="red", marker="s", s=100)

    lm_loc_handle = mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=10, label="Landmarks")
    handles.append(lm_loc_handle)

# overlay transparent image of map
img = mpimg.imread("maze_uk2015f_orthographic.png")
plt.imshow(img, extent=[-1.44, 1.44, -1.44, 1.44], alpha=0.6)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(title, fontsize=14)
plt.axis("equal")
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(handles=handles, loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(fig_name, dpi=300)
plt.show()
