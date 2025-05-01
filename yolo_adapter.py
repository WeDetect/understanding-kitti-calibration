from pathlib import Path  
from typing import Sequence

CLASS_NAME_TO_CLASS_ID = {
    "Car": 0, 
    "Pedestrian": 1,
    "Cyclist": 2, 
    "Truck": 3,
    "Van": 4,
    "Tram": 5,
    "Misc": 6
}


def rects_to_yolo(rects: list, image_shape: tuple, class_names: list):
    """
    Converts 2D bounding boxes to YOLO format.
    
    Args:
        rects (list of tuples): Each tuple is (xmin, ymin, xmax, ymax)
        image_shape (tuple): (H, W, C) or (H, W)
        class_names (list of str): class name per rectangle (same length as rects)
        
    Returns:
        yolo_labels (list of lists): each sublist is [class_id, x_center, y_center, width, height]
    """
    H, W = image_shape[:2]
    yolo_labels = []

    for i, (xmin, ymin, xmax, ymax) in enumerate(rects):
        x_center = (xmin + xmax) / 2.0 / W
        y_center = (ymin + ymax) / 2.0 / H
        width    = (xmax - xmin) / W
        height   = (ymax - ymin) / H

        class_id = 0  # default class
        if class_names and i < len(class_names):
            class_name = class_names[i]
            class_id = CLASS_NAME_TO_CLASS_ID.get(class_name, 0)
        yolo_labels.append([class_id, x_center, y_center, width, height])

    return yolo_labels


def save_yolo_label(output: Path, yolo_lines: list[tuple]):
    with open(output, "w") as f:
        for line in yolo_lines:
            f.write(" ".join(map(str, line)) + "\n")