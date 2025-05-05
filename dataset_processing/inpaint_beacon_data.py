import os
import cv2
import pickle
from tqdm import tqdm
from natsort import os_sorted
from aruco_detector import VisionDetector


marker_detector = VisionDetector(marker_size=0.02,
                                 calibration_dir="aruco_camera_calibration/static_camera",
                                 inpainting_factor=0.8)

gripper_marker_detector = VisionDetector(marker_size=0.02,
                                 calibration_dir="aruco_camera_calibration/gripper_camera",
                                 inpainting_factor=0.8)

data_fd = 'expert_demos/push_button'
demo_list = os_sorted(os.listdir(data_fd))

for demo_path in demo_list:
    file_path = f"{data_fd}/{demo_path}"
    with open(file_path, "rb") as file:
        demo = pickle.load(file)

    steps = len(demo['img'])
    
    img_inpainted = []
    img_gripper_inpainted = []
    
    print(f"[*] Inpainting demo: {file_path}")

    for step in tqdm(range(steps)):
        img = demo['img'][step]
        _, _, _, ids, inpainted_img = marker_detector.cv2_aruco_inpainting(img, 50)
        
        img_gripper = demo['img_gripper'][step]
        _, _, _, _, inpainted_img_gripper = gripper_marker_detector.cv2_aruco_inpainting(img_gripper, 50)

        # Also changing the image format from bgr to rgb
        img_inpainted.append(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
        img_gripper_inpainted.append(cv2.cvtColor(inpainted_img_gripper, cv2.COLOR_BGR2RGB))

    demo['img_inpainted'] = img_inpainted
    demo['img_gripper_inpainted'] = img_gripper_inpainted


    for key, value in demo.items():
        print(key, len(value))

    with open(file_path, "wb") as file:
        pickle.dump(demo, file)