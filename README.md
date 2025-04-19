# KITTI CALIBRATION FILES UNDERSTANDING
https://chatgpt.com/share/68023537-8e64-8006-829f-82cef9646bc6
## KITTI Camera Setup

Matrix | Camera Type | Position
---|---|---
P0 | Gray-scale | Left
P1 | Gray-scale | Right
P2 | Color (RGB) | Left
P3 | Color (RGB) | Right
### What is a Projection Matrix?
Each P matrix is a 3×4 projection matrix of the form:
P=K⋅[R ∣ T]
Where:

K is the intrinsic matrix (camera lens model),

[R | T] is the extrinsic matrix (rotation and translation from world or rectified camera coordinates).
## File Structure
```
P0: [3x4 projection matrix for camera 0]
P1: [3x4 projection matrix for camera 1]
P2: [3x4 projection matrix for camera 2]
P3: [3x4 projection matrix for camera 3]
R0_rect: [3x3 rectifying rotation matrix for camera 0]
Tr_velo_to_cam: [3x4 matrix to transform from Velodyne (LiDAR) to camera coordinates]
Tr_imu_to_velo: [3x4 matrix to transform from IMU to Velodyne]

```
Parameter | Description
---|---
P0, P1, P2, P3 | Projection matrices for each camera. These project 3D camera points into 2D image coordinates.
R0_rect | Rectification matrix to align the camera images to a common frame.
Tr_velo_to_cam | Transformation matrix from LiDAR to camera coordinates (rotation + translation).
Tr_imu_to_velo | Transformation matrix from IMU to LiDAR coordinates.

## LiDAR to Image Projection Flow:
To project a LiDAR point into an image:

1. **LiDAR ➝ Camera coordinates** Apply Tr_velo_to_cam to transform the 3D point from Velodyne to the camera frame.

2. **Rectification** Apply R0_rect to align the camera frame with the image plane.

3. **Projection to 2D** Use the appropriate projection matrix P2 (for example, for the left color camera) to map 3D points into image coordinates.

exampel flow (in np matrix form)
```python
image_point = P2 @ R0_rect @ Tr_velo_to_cam @ lidar_point
```

## Camera Angle
### Key Intrinsic Parameters (from P or K matrix)
A camera’s intrinsic matrix (often denoted K) looks like this:
```
K = [
    fx  0   cx 
    0   fy  cy
    0   0   1]
```

Where:

f_x, f_y: focal lengths in pixels (related to the real focal length and pixel size),

c_x, c_y: the optical center (usually image center),

f_x = f * (image_width / sensor_width)

If you look at P2:
```
P2: [
    fx, 0,  cx, tx,
    0,  fy, cy, ty,
    0,  0,  1,  0
    ]
```


### You want to simulate a wider or narrower field of view (zoom in/out)
* Make view wider (zoom out) → Decrease focal lengths f_x, f_y
    * Decrease fx/fy = more of the world fits in the image (wider FOV).
* Make view narrower (zoom in) → Increase focal lengths f_x, f_y
    * Increase fx/fy = zoomed-in view (narrower FOV).

### You want to rotate the camera to face a different direction
([more information about camera extrinsic matrix](https://medium.com/data-science/camera-extrinsic-matrix-with-example-in-python-cfe80acab8dd))

* The rotation is handled in Tr_velo_to_cam and R0_rect

* You can modify the rotation matrix to simulate the camera turning

**For example**
Rotate the camera 20° to the left:
```python 
theta = np.radians(20)
R_y = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])

```

### Summary
You want to... | Modify... | Effect
---|---|---
Zoom in / out (change FOV) | fx, fy in K or P | Optical zoom
Shift view left/right/up/down | cx, cy in K or P | Pan/tilt effect
Rotate camera in 3D space | Rotation matrix in Tr_velo_to_cam | Viewpoint changes
Move camera to a different position | Translation in Tr_velo_to_cam | Change origin