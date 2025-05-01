import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dotenv import load_dotenv

from calibration import KITTICalibration
from labels_handler import KITTILabelHandler

load_dotenv(dotenv_path=".env")
    
KITTI_PATH = os.environ.get("KITTI_PATH")
file = "000000"

# Paths
calib_path = Path(KITTI_PATH)/"calib"/f"{file}.txt"
vel_path   = Path(KITTI_PATH)/"velodyne"/f"{file}.bin"
img_path   = Path(KITTI_PATH)/"image_2"/f"{file}.png"
label_file = Path(KITTI_PATH)/"label_2"/f"{file}.txt"

# Read image and LiDAR
image = cv2.imread(str(img_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
lidar = np.fromfile(vel_path, dtype=np.float32).reshape(-1,4)

# Calibration
calib = KITTICalibration(calib_path)
label = KITTILabelHandler(label_file)


def main_plot():
    # Angles to visualize
    angles = [-15, 0, 15]
    fig, axes = plt.subplots(1, len(angles), figsize=(20, 6))
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for ax, angle in zip(axes, angles):
        # Show base image
        ax.imshow(np.zeros(image.shape[:2], dtype=np.uint8))
        ax.axis('off')

        # Scatter LiDAR points with yaw rotation
        # img_pts, depth = calib.rotate_camera_and_project_with_yaw(lidar, angle)
        img_pts, depth = calib.rotate_camera_and_project_with_pitch(lidar, angle)
        
        mask = (
            (depth > 0) &
            (img_pts[:,0] >= 0) & (img_pts[:,1] >= 0) &
            (img_pts[:,0] < image.shape[1]) & (img_pts[:,1] < image.shape[0])
        )
        pts, dp = img_pts[mask], depth[mask]
        ax.scatter(pts[:,0], pts[:,1], c=dp, cmap='jet', s=1, alpha=0.3)

        # Overlay rotated 3D boxes
        # label.plot_boxes_with_yaw(calib, ax, angle=angle)
        label.plot_boxes_with_pitch(calib, ax, angle)
        
        ax.set_title(f"Yaw: {angle}°", fontsize=14)
        # ax.set_title(f"Pitch: {angle}°", fontsize=14)

    fig.suptitle("LiDAR & 3D Boxes at Multiple Yaw Angles", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

if __name__ == '__main__':
    main_plot()
