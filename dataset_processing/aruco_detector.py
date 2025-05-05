import cv2
import pickle
import numpy as np 
from tqdm import tqdm


class VisionDetector:
    def __init__(
            self,
            marker_size,
            calibration_dir,
            inpainting_factor=0.1
    ):  
        
        camera_matrix = np.load(f'{calibration_dir}/camera_matrix.npy')
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        self.camera_intrinsics = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]])
        
        self.distortion = np.load(f'{calibration_dir}/dist_coeffs.npy').flatten()
        
        self.marker_size = marker_size
        self.inpainting_factor = inpainting_factor
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


    def estimatePoseSingleMarkers(self, corners):
        """
        Description: Aruco Marker pose estimation.
        Author: M lab
        Date: Jul 31, 2023 at 10:49
        URL: https://github.com/Menginventor/aruco_example_cv_4.8.0
        """
        '''
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        '''
        marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, -self.marker_size / 2, 0],
                                [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(marker_points, c, self.camera_intrinsics, self.distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash



    def plot_aruco(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)

        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)
        frame_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners)

        rvecs = []
        tvecs = [] 
        for i in range(len(corners)):
            rvec, tvec, _ = self.estimatePoseSingleMarkers(corners[i])

            frame_markers = cv2.drawFrameAxes(
                frame_markers.copy(),
                self.camera_intrinsics,
                np.zeros(5),
                np.array(rvec),
                np.array(tvec),
                0.025
            )
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs, corners, ids, frame_markers

    def aruco_inpainting(self, image, base_image):
        # Get the base image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)

        rvecs = []
        tvecs = []
        for i in range(len(corners)): 
            new_corner = self.mask_extension(corners[i][0], 0.3, 0.3)
            curr_points = np.array(new_corner, dtype = np.int32)
            mask = np.zeros_like(image)
            cv2.fillConvexPoly(mask, curr_points, (255, 255, 255))
            base_region = cv2.bitwise_and(base_image, mask)
            mask_inv = cv2.bitwise_not(mask)
            image_bg = cv2.bitwise_and(image, mask_inv)
            image = cv2.add(image_bg, base_region)

            rvec, tvec, _ = self.estimatePoseSingleMarkers(corners[i])
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs, corners, ids, image

    def cv2_aruco_inpainting(self, image, inpainting_radius):
        # This method uses the inpainting method provided by the cv2 library
        # Get the base image
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(image)

        rvecs = []
        tvecs = []
        for i in range(len(corners)):
            new_corner = self.mask_extension(corners[i][0], self.inpainting_factor, self.inpainting_factor)
            curr_points = np.array(new_corner, dtype = np.int32)
            mask = np.zeros_like(image)
            cv2.fillConvexPoly(mask, curr_points, (255, 255, 255))
            mask = np.transpose(mask, (2, 0, 1))
            mask = mask[0]
            image = cv2.inpaint(image, mask, inpainting_radius, cv2.INPAINT_NS)

            rvec, tvec, _ = self.estimatePoseSingleMarkers(corners[i])
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs, corners, ids, image


    def mask_extension(self, corner, extention_x, extention_y):
        vector_x = corner[0] - corner[1]
        vector_y = corner[0] - corner[3]
        new_corner = np.zeros_like(corner)
        new_corner[0] = corner[0] + vector_x* extention_x + vector_y* extention_y
        new_corner[1] = corner[1] - vector_x* extention_x + vector_y* extention_y
        new_corner[2] = corner[2] - vector_x* extention_x - vector_y* extention_y
        new_corner[3] = corner[3] + vector_x* extention_x - vector_y* extention_y

        return new_corner


if __name__ == '__main__':

    marker_size = 0.02
    calibration_dir = 'aruco_camera_calibration/static_camera'
    detector = VisionDetector(marker_size, calibration_dir)

    data_file = ""
    with open(data_file, 'rb') as file:
        data = pickle.load(file)
    
    video_list = []
    process_bar = tqdm(total=len(data['img']), desc="Processing images")
    for image in data['img']:
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        rvecs, tvecs, corners, ids, frame_markers = detector.plot_aruco(image)
        video_list.append(frame_markers)
        process_bar.update(1)
    process_bar.close()
    
    print("saving video")
    # dump into video
    video_path = f""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    height, width, _ = video_list[0].shape
    frame_size = (width, height)
    vid_w = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    steps = len(video_list)

    process_bar = tqdm(total=steps, desc="Writing video")
    for step in range(steps):
        img = video_list[step]
        process_bar.update(1)

        vid_w.write(img)
    vid_w.release()