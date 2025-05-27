import numpy as np


class KITTIHandlerBase:
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
        
    def _get_roll_rotation_matrix(self, roll_degrees: float) -> np.ndarray:
        """
        Returns a 4x4 homogeneous roll rotation matrix (Z-axis) in camera coordinates.
        """
        theta = np.radians(roll_degrees)
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [            0,              0, 1, 0],
            [            0,              0, 0, 1]
        ])

    def _get_rotation_matrix(self, yaw_deg=0, pitch_deg=0, roll_deg=0):
        """Build and return a single 4×4 yaw * pitch * roll homogeneous rotation matrix."""
        Ry = self._get_yaw_rotation_matrix(yaw_deg) if yaw_deg else np.eye(4)
        Rx = self._get_pitch_rotation_matrix(pitch_deg) if pitch_deg else np.eye(4)
        Rz = self._get_roll_rotation_matrix(roll_deg) if roll_deg else np.eye(4)

        return Rz @ Ry @ Rx  # Order: roll → yaw → pitch


    def _get_translation_matrix(self, tx: float, ty: float, tz: float) -> np.ndarray:
        """
        Returns a 4x4 homogeneous translation matrix to move the camera in 3D space.
        tx, ty, tz specify translation along the X, Y, Z axes respectively.
        """
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0,  1]
        ])
        
    def get_camera_extrinsic(self, yaw=0, pitch=0, roll=0, tx=0, ty=0, tz=0):
        """
        Returns a 4x4 extrinsic matrix representing the camera's pose
        (i.e., rotation and translation) in world space.
        """
        R = self._get_rotation_matrix(yaw, pitch, roll)
        T = self._get_translation_matrix(tx, ty, tz)
        return T @ R  # Apply rotation first, then move the camera

