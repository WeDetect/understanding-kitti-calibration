import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dotenv import load_dotenv

from point_cloud_handlers.calibration import KITTICalibration
from point_cloud_handlers.labels_handler import KITTILabelHandler
from plot_utils import draw_box_edges_on_plot, draw_points_on_plot

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
    yaw_angles = [0, 0, 0]
    pitch_angles = [-15, 0, 15]
    fig, axes = plt.subplots(1, len(yaw_angles), figsize=(20, 6))
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for ax, yaw_angle, pitch_angle in zip(axes, yaw_angles, pitch_angles):
        empty_array = np.zeros(image.shape[:2], dtype=np.uint8)
        ax.imshow(empty_array)
        ax.axis('off')

        img_pts, depth = calib.rotate_camera_and_project(lidar, yaw_deg=yaw_angle, pitch_deg=pitch_angle )
        draw_points_on_plot(ax, img_pts, depth, image.shape)

        boxes = label.get_3d_boxes_rotated(calib, pitch_deg=pitch_angle, yaw_deg=yaw_angle)
        draw_box_edges_on_plot(ax, boxes, color="red")
        
        ax.set_title("Rotated Images with 3D Object Boxes")

    fig.suptitle("LiDAR & 3D Boxes at Multiple Angles", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

if __name__ == '__main__':
    main_plot()
