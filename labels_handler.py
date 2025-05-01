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
    
    def plot_boxes_with_yaw(self, calib: KITTICalibration, ax, angle=0, color="#F00"):
        """
        Projects and draws rotated 3D bounding boxes onto a Matplotlib Axes.
        """
        yaw_matrix = self._get_yaw_rotation_matrix(angle)
        boxes = self.get_3d_boxes()

        for corners_3d in boxes:
            img_pts = self._project_box_to_image(corners_3d, yaw_matrix, calib)
            self._draw_box_edges(ax, img_pts, color)


    def _get_yaw_rotation_matrix(self, angle_degrees: float) -> np.ndarray:
        """
        Returns a 4x4 homogeneous yaw rotation matrix (Y-axis) in camera coordinates.
        """
        theta = np.radians(angle_degrees)
        return np.array([
            [ np.cos(theta), 0, np.sin(theta), 0],
            [             0, 1,             0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [             0, 0,             0, 1]
        ])
        
    def _get_pitch_rotation_matrix(self, pitch_degrees: float) -> np.ndarray:
        """
        Returns a 4x4 homogeneous pitch rotation matrix (X-axis) in camera coordinates.
        """
        theta = np.radians(pitch_degrees)
        return np.array([
            [1,             0,              0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta),  np.cos(theta), 0],
            [0,             0,              0, 1]
        ])


    def _project_box_to_image(self, corners_3d: np.ndarray, yaw_matrix: np.ndarray, calib: KITTICalibration) -> np.ndarray:
        """
        Applies rotation, rectification, and projection of 3D box corners to image coordinates.
        Returns:
            img_pts: (8, 2) array of 2D image points
        """
        corners_h = np.hstack([corners_3d, np.ones((8, 1))])        # (8, 4)
        rotated = (yaw_matrix @ corners_h.T).T                      # (8, 4)
        rectified = (calib.R0_rect @ rotated.T).T                   # (8, 4)
        img_pts_h = (calib.P2 @ rectified.T).T                      # (8, 3)
        img_pts = img_pts_h[:, :2] / img_pts_h[:, 2:3]              # normalize
        return img_pts


    def _draw_box_edges(self, ax, img_pts: np.ndarray, color: str):
        """
        Draws the 12 edges of a projected 3D bounding box on a Matplotlib Axes.
        """
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

    def plot_boxes_with_pitch(self, calib: KITTICalibration, ax, pitch=0, color="#0F0"):
        """
        Projects and draws 3D bounding boxes from a pitched camera view onto a Matplotlib Axes.
        """
        pitch_matrix = self._get_pitch_rotation_matrix(pitch)
        boxes = self.get_3d_boxes()

        for corners_3d in boxes:
            img_pts = self._project_box_to_image(corners_3d, pitch_matrix, calib)
            self._draw_box_edges(ax, img_pts, color)
