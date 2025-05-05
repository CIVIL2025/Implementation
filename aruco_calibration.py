import os 
import cv2
import numpy as np


save_fd = 'aruco_camera_calibration/static_camera'
img_fd = f'{save_fd}/calibration_images'


image_files = [os.path.join(img_fd, f) for f in os.listdir(img_fd) if f.endswith(".jpg")]
image_files.sort()  # Ensure files are in order


all_charuco_corners = []
all_charuco_ids = []

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
charuco = cv2.aruco.CharucoBoard((9, 7), 0.05, 0.035, d)

for image_file in image_files:
    image = cv2.imread(image_file)
    image_copy = image.copy()

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, d, parameters=parameters)
    
    # If at least one marker is detected
    if len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, charuco)
        if charuco_retval:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

# Calibrate camera
retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, charuco, image.shape[:2], None, None)

# Save calibration data
np.save(f'{save_fd}/camera_matrix.npy', camera_matrix)
np.save(f'{save_fd}/dist_coeffs.npy', dist_coeffs)
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]
distortion = dist_coeffs.flatten()
print(fx)
print(fy)
print(cx)
print(cy)
print(distortion)