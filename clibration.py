import numpy as np

class KITTICalibration:
    def __init__(self, calib_file):
        self.calib = self._read_calib_file(calib_file)
        self.P2 = self._get_matrix('P2', (3, 4))
        self.R0_rect = self._get_matrix('R0_rect', (3, 3))
        self.Tr_velo_to_cam = self._get_matrix('Tr_velo_to_cam', (3, 4))
        
        # Convert Tr_velo_to_cam to 4x4
        self.Tr_velo_to_cam = self._to_homogeneous(self.Tr_velo_to_cam)
        self.R0_rect = self._to_homogeneous(self.R0_rect, is_rect=True)

    def _read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value =line.split(':', 1)
                data[key] = np.array([float(x) for x in value.strip().split()])
        return data

    def _get_matrix(self, key, shape):
        return self.calib[key].reshape(shape)

    def _to_homogeneous(self, mat, is_rect=False):
        if is_rect:
            mat_h = np.eye(4)
            mat_h[:3, :3] = mat
        else:
            mat_h = np.vstack((mat, [0, 0, 0, 1]))
        return mat_h

    def project_lidar_to_image(self, lidar_points):
        lidar_hom = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
        cam_points = (self.Tr_velo_to_cam @ lidar_hom.T).T
        cam_points = (self.R0_rect @ cam_points.T).T
        image_points = (self.P2 @ cam_points.T).T
        image_points = image_points[:, :2] / image_points[:, 2:3]
        return image_points, cam_points[:, 2]  # image coords, depth
    
    def rotate_camera_and_project(self, lidar_points, angle_deg):
        # Rotation matrix around Y axis (yaw)
        theta = np.radians(angle_deg)
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

        # Homogenize lidar points
        lidar_hom = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))

        # Transform to camera space
        cam_points = (self.Tr_velo_to_cam @ lidar_hom.T).T
        cam_points = (R_y @ cam_points.T).T
        cam_points = (self.R0_rect @ cam_points.T).T

        # Project to image
        image_points = (self.P2 @ cam_points.T).T
        image_points = image_points[:, :2] / image_points[:, 2:3]

        return image_points, cam_points[:, 2]  # image coords, depth

    def rotate_camera_vertically_and_project(self, lidar_points, angle_deg):
        theta = np.radians(angle_deg)
        R_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

        lidar_hom = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
        cam_points = (self.Tr_velo_to_cam @ lidar_hom.T).T
        cam_points = (R_x @ cam_points.T).T
        cam_points = (self.R0_rect @ cam_points.T).T
        image_points = (self.P2 @ cam_points.T).T
        image_points = image_points[:, :2] / image_points[:, 2:3]

        return image_points, cam_points[:, 2]