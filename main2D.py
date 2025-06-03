import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dotenv import load_dotenv

from point_cloud_handlers.calibration import KITTICalibration
from point_cloud_handlers.labels_handler import KITTILabelHandler
from point_cloud_handlers.plot_utils import draw_points_on_plot, draw_rect_on_plot

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
    yaw_angles =      [-15        ,0      ,15]
    pitch_angles =    [0        ,0      ,0]    
    roll_angles =     [0        ,0      ,0]   
    # Translation
    tx_s =            [5        ,0      ,-5]   
    ty_s =            [0        ,0      ,0]   
    tz_s =            [0        ,0      ,0]   
     
    fig, axes = plt.subplots(1, len(yaw_angles), figsize=(20, 6))
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for ax, yaw_angle, pitch_angle, roll_angle, tx, ty, tz in zip(
        axes, yaw_angles, pitch_angles, roll_angles, tx_s, ty_s, tz_s
        ):
        empty_array = np.zeros(image.shape[:2], dtype=np.uint8)
        ax.imshow(empty_array)
        ax.axis('off')

        R = calib.get_camera_extrinsic(
            yaw=yaw_angle, pitch=pitch_angle, roll=roll_angle,
            tx=tx, ty=ty, tz=tz)
        
        img_pts, depth = calib.rotate_camera_and_project(lidar, R)
        draw_points_on_plot(ax, img_pts, depth, image.shape)

        objects_type, objects_rect = label.get_2d_boxes_rotated(calib, R)
        draw_rect_on_plot(ax, objects_rect)
        
        ax.set_title(f"Yaw: {yaw_angle}°, Pitch: {pitch_angle}°, Roll: {roll_angle}° \n" \
            f"tx: {tx}, ty: {ty}, tz: {tz}")        

    fig.suptitle("LiDAR & 2D Rects at Multiple  Angles", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()



if __name__ == '__main__':
    main_plot()
