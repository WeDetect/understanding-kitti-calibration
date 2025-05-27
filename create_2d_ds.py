import numpy as np
import cv2
from pathlib import Path
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from calibration import KITTICalibration
from labels_handler import KITTILabelHandler
from plot_utils import draw_points_on_plot
from yolo_adapter import rects_to_yolo, save_yolo_label

from tqdm import tqdm

load_dotenv(dotenv_path=".env")

KITTI_PATH = os.environ.get("KITTI_PATH")
print(KITTI_PATH)
file = "000000"

# Paths
calib_dir_path = Path(KITTI_PATH)/"calib"
vel_dir_path   = Path(KITTI_PATH)/"velodyne"
img_dir_path   = Path(KITTI_PATH)/"image_2"
label_dir_path = Path(KITTI_PATH)/"label_2"

# Angles
yaw_angles =      [-45        ,0      ,45]
pitch_angles =    [0        ,0      ,0]    
roll_angles =     [0        ,0      ,0]   
# Transform
tx_s =            [0        ,0      ,0]   
ty_s =            [0        ,0      ,0]   
tz_s =            [0        ,0      ,0]   
     
def get_file_data(file):
    img_path = img_dir_path / f"{file}.png"
    vel_path = vel_dir_path / f"{file}.bin"
    calib_path = calib_dir_path / f"{file}.txt"
    label_file = label_dir_path / f"{file}.txt"

    # Read image and LiDAR
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lidar = np.fromfile(vel_path, dtype=np.float32).reshape(-1,4)

    # Calibration and Labels
    calib = KITTICalibration(calib_path)
    label_handler = KITTILabelHandler(label_file)
    return image,lidar,calib,label_handler

def create_file_variants(file):
    image, lidar, calib, label_handler = get_file_data(file)
    
    for i, (yaw_angle, pitch_angle, roll_angle, tx, ty, tz) in enumerate(
        zip(yaw_angles, pitch_angles, roll_angles, tx_s, ty_s, tz_s)
    ):
        dataset_path = Path("datasets") / f"dataset_{i}"
        image_dir = dataset_path / "images"
        label_dir = dataset_path / "labels"
        readme_file = dataset_path / "DatasetInfo.md"
        
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        if not readme_file.exists(): 
            create_readme_file(yaw_angle, pitch_angle, roll_angle, tx, ty, tz, readme_file)
            
        # Project LiDAR and get 3D boxes
        R = calib.get_camera_extrinsic(
            yaw=yaw_angle, pitch=pitch_angle, roll=roll_angle,
            tx=tx, ty=ty, tz=tz)
        
        img_pts, depth = calib.rotate_camera_and_project(lidar, R)
        fig = draw_image(image, img_pts, depth)
        save_image(file, image_dir, fig)
        
        objects_type, objects_rect = label_handler.get_2d_boxes_rotated(calib, R)
        save_labels(file, image, label_dir, objects_type, objects_rect) 
        
def draw_image(image, img_pts, depth):
    fig, ax = plt.subplots()
    empty_array = np.zeros(image.shape[:2], dtype=np.uint8)
    ax.imshow(empty_array, cmap='gray')
    draw_points_on_plot(ax, img_pts, depth, image.shape)
    ax.axis("off")
    return fig

def save_labels(file, image, label_dir, objects_type, objects_rect):
    label_filename = label_dir / f"{file}.txt"
    yolo_lines = rects_to_yolo(objects_rect, image.shape, class_names=objects_type)
    save_yolo_label(output=label_filename, yolo_lines=yolo_lines)

def save_image(file, image_dir, fig):
    image_filename = image_dir / f"{file}.png"
    fig.savefig(image_filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def create_readme_file(yaw_angle, pitch_angle, roll_angle, tx, ty, tz, readme_file):
    with open(readme_file, 'w') as f:
        f.write(
            "# Dataset {i} \n" \
            f"yaw_angle: {yaw_angle}, pitch_angle: {pitch_angle}, roll_angle: {roll_angle} \n" \
            f"tx: {tx}, ty: {ty}, tz: {tz} \n"
        )

if __name__ == "__main__":
    for i in tqdm(range(7481)):
        file_name = f"{i:06d}"
        create_file_variants(file=file_name)
    print("Datasets saved successfully.")
