import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

from clibration import KITTICalibration

load_dotenv(dotenv_path=".env")  

# if __name__ == "__main__": 
#     KITTI_PATH = os.environ.get("KITTI_PATH")
    
#     file = "000000"
#     clibration_file_path = Path(KITTI_PATH) / "calib" / f"{file}.txt"
#     velodyne_path = Path(KITTI_PATH) / "velodyne" / f"{file}.bin"
#     image_path = Path(KITTI_PATH) / "image_2" / f"{file}.png"
    
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     lidar = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

#     calibration_obj = KITTICalibration(clibration_file_path)
#     img_points, depth  = calibration_obj.project_lidar_to_image(lidar_points=lidar)
    
    
#     # todo: understand mask:
#     mask = (depth > 0) & (img_points[:, 0] >= 0) & (img_points[:, 1] >= 0) & \
#        (img_points[:, 0] < image.shape[1]) & (img_points[:, 1] < image.shape[0])
       
#     img_points = img_points[mask]
#     depth = depth[mask]

#     # === Visualize ===
#     empty_image = np.zeros(image.shape[:2], dtype=np.uint8)
    
#     plt.figure(figsize=(12, 6))
#     plt.imshow(empty_image)
#     plt.scatter(img_points[:, 0], img_points[:, 1], c=depth, cmap='jet', s=1)
#     plt.title("Projected LiDAR Points on Image")
#     plt.axis('off')
#     plt.colorbar(label='Depth (m)')
#     plt.show()


if __name__ == "__main__": 
    KITTI_PATH = os.environ.get("KITTI_PATH")

    file = "000000"
    clibration_file_path = Path(KITTI_PATH) / "calib" / f"{file}.txt"
    velodyne_path = Path(KITTI_PATH) / "velodyne" / f"{file}.bin"
    image_path = Path(KITTI_PATH) / "image_2" / f"{file}.png"

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lidar = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

    calibration_obj = KITTICalibration(clibration_file_path)

    # Rotations to visualize
    angles = [-30, -15, 0, 15, 30]
    fig, axes = plt.subplots(1, len(angles), figsize=(20, 5))

    for ax, angle_deg in zip(axes, angles):
        img_points, depth = calibration_obj.rotate_camera_and_project(lidar_points=lidar, angle_deg=angle_deg)
        # img_points, depth = calibration_obj.rotate_camera_vertically_and_project(lidar_points=lidar, angle_deg=angle_deg)

        # Apply mask to keep points inside image boundaries and in front of the camera
        mask = (depth > 0) & (img_points[:, 0] >= 0) & (img_points[:, 1] >= 0) & \
               (img_points[:, 0] < image.shape[1]) & (img_points[:, 1] < image.shape[0])

        img_points_masked = img_points[mask]
        depth_masked = depth[mask]

        ax.imshow(np.zeros(image.shape[:2], dtype=np.uint8))
        scatter = ax.scatter(img_points_masked[:, 0], img_points_masked[:, 1], c=depth_masked, cmap='jet', s=1)
        ax.set_title(f"Rotation: {angle_deg}°")
        ax.axis('off')

    fig.suptitle("Projected LiDAR Points on Different Rotated Camera Views", fontsize=16)
    plt.tight_layout()
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.6, label='Depth (m)')
    plt.show()