import numpy as np

from calibration import KITTICalibration

class KITTILabelHandler:
    def __init__(self, label_file):
        self.label_file = label_file
        self.labels = self._read_labels()

    def _read_labels(self):
        labels = []
        with open(self.label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                obj_type = parts[0]
                if obj_type == 'DontCare':
                    continue
                label = {
                    'type': obj_type,
                    'h': float(parts[8]),
                    'w': float(parts[9]),
                    'l': float(parts[10]),
                    'x': float(parts[11]),
                    'y': float(parts[12]),
                    'z': float(parts[13]),
                    'ry': float(parts[14])
                }
                labels.append(label)
        return labels

    def get_3d_boxes(self):
        boxes = []
        for label in self.labels:
            box = self.compute_box_3d(label)
            boxes.append(box)
        return boxes

    def compute_box_3d(self, label):
        h, w, l = label['h'], label['w'], label['l']
        x, y, z = label['x'], label['y'], label['z']
        ry = label['ry']

        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])

        R = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        corners_3d = R @ corners_3d
        corners_3d += np.array([[x], [y], [z]])
        return corners_3d.T
    
    def plot_boxes_on_ax(self, calib: KITTICalibration, ax, angle=0, color="#F00"):
        """
        Projects 3D bounding boxes into the rotated camera view and draws them on a Matplotlib Axes.

        Args:
            label_file: Path to KITTI labels (.txt)
            calib:      KITTICalibration object
            ax:         Matplotlib Axes
            angle:      yaw rotation in degrees
            color:      Optional
        """
        # Compute yaw rotation matrix around Y axis
        theta = np.radians(angle)
        R_y = np.array([
            [ np.cos(theta), 0, np.sin(theta), 0],
            [             0, 1,             0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [             0, 0,             0, 1]
        ])

        boxes = self.get_3d_boxes()

        for corners_3d in boxes:
            # Homogenize corners (8,4)
            corners_h = np.hstack([corners_3d, np.ones((8,1))])

            # Rotate in camera frame
            rotated = (R_y @ corners_h.T).T        # (8,4)
            # Rectify if needed (assuming labels in camera0 coords)
            rectified = (calib.R0_rect @ rotated.T).T  # (8,4)
            # Project to image
            img_pts_h = (calib.P2 @ rectified.T).T      # (8,3)
            img_pts = img_pts_h[:, :2] / img_pts_h[:, 2:3]

            # Define 12 edges of the box
            edges = [
                (0,1),(1,2),(2,3),(3,0),  # bottom
                (4,5),(5,6),(6,7),(7,4),  # top
                (0,4),(1,5),(2,6),(3,7)   # sides
            ]
            for i, j in edges:
                ax.plot(
                    [img_pts[i,0], img_pts[j,0]],
                    [img_pts[i,1], img_pts[j,1]],
                    color=color, linewidth=1.5
                )

