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
        
    def _get_rotation_matrix(self, yaw_deg=0, pitch_deg=0):
        """Build and return a single 4×4 yaw*∘*pitch homogeneous rotation matrix."""
        Ry = np.eye(4)
        if yaw_deg:
            Ry = self._get_yaw_rotation_matrix(yaw_deg)

        Rx = np.eye(4)
        if pitch_deg:
            Rx = self._get_pitch_rotation_matrix(pitch_deg)

        return Ry @ Rx