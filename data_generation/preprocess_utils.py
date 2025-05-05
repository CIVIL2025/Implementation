import torch
import numpy as np
import pybullet as p
from termcolor import colored
from scipy.spatial.distance import cdist
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


# NOTE: default information on the calvin dataset this includes the joint state, and logical states
# for the fix objects on the table, and the pos and rot for the blocks
DEFAULT_BEACON_MAPPING = {'sliding_door': slice(0,1),
                'drawer': slice(1,2),
                'button': slice(2,3),
                'switch': slice(3,4),
                'light': slice(4,5),
                'green_light': slice(5,6),
                'red_block_pos': slice(6,9),
                'red_block_rot': slice(9,12),
                'blue_block_pos': slice(12, 15),
                'blue_block_rot': slice(15, 18),
                'pink_block_pos': slice(18, 21),
                'pink_block_rot': slice(21, 24)}

# NOTE: added new function in the environment to return the pos and ori of each object in the environment 
BEACON_MAPPING = {'slider_pos': slice(0, 3),
                  'slider_rot': slice(3, 6),
                  'drawer_pos': slice(6, 9),
                  'drawer_rot': slice(9, 12),
                  'button_pos': slice(12, 15),
                  'button_rot': slice(15, 18),
                  'switch_pos': slice(18, 21),
                  'switch_rot': slice(21, 24),
                  'light_pos': slice(24,27),
                  'light_rot': slice(27,30),
                  'green_light_pos': slice(30, 33),
                  'green_light_rot': slice(33, 36),
                  'red_block_pos': slice(36,39),
                  'red_block_rot': slice(39,42),
                  'blue_block_pos': slice(42, 45),
                  'blue_block_rot': slice(45, 48),
                  'pink_block_pos': slice(48, 51),
                  'pink_block_rot': slice(51, 54)}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_egocam_info(env):
    gripper_camera = env.cameras[1]
    camera_ls = p.getLinkState(
        bodyUniqueId=gripper_camera.robot_uid, linkIndex=gripper_camera.gripper_cam_link, physicsClientId=gripper_camera.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = p.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
    projection_matrix = p.computeProjectionMatrixFOV(
                fov=gripper_camera.fov, aspect=gripper_camera.aspect, nearVal=gripper_camera.nearval, farVal=gripper_camera.farval
    )
    return view_matrix, projection_matrix


def world_to_pixel(world_point, view_matrix, projection_matrix, img_width, img_height):
    """
    Transforms a 3D point from world coordinates to pixel coordinates.

    world_point: (x, y, z) coordinates in world frame
    view_matrix: 4x4 view matrix from PyBullet
    projection_matrix: 4x4 projection matrix from PyBullet
    img_width: width of the camera image
    img_height: height of the camera image
    """
    world_point = np.array([*world_point, 1.0])  # Convert to homogeneous coordinates
    
    view_matrix = np.array(view_matrix).reshape(4, 4).T  
    projection_matrix = np.array(projection_matrix).reshape(4, 4).T

    camera_coords = view_matrix @ world_point # world to camera
    image_coords = projection_matrix @ camera_coords # camera to 2d
    image_coords /= image_coords[3]  # Divide by w to normalize

    x = (image_coords[0] * 0.5 + 0.5) * img_width
    y = (1.0 - (image_coords[1] * 0.5 + 0.5)) * img_height  # Flip y-axis


    return int(x), int(y)

def get_pixel_coords(env, obs):
    w, h = obs['rgb_obs']['rgb_gripper'].shape[:2]
    view_matrix, projection_matrix = get_egocam_info(env)

    pixel_coord = {}
    adjusted_beacon = np.array([])
    
    for key, beacon_idx in BEACON_MAPPING.items():
        beacon = obs['scene_beacon'][beacon_idx]
        if key == 'slider_pos':
            beacon = np.add(beacon, np.array([0.28, 0.0, 0.0]))
        elif key == 'drawer_pos':
            beacon = np.add(beacon, np.array([0.0, -0.16, 0.0]))
        adjusted_beacon = np.concatenate((adjusted_beacon, beacon), axis = 0)

        if key in ['red_block_pos', 'blue_block_pos', 'pink_block_pos', 'slider_pos', 'drawer_pos', 'switch_pos']:
            x,y = world_to_pixel(beacon, view_matrix, projection_matrix, w, h)
            pixel_coord[key] = (x,y)

    return adjusted_beacon, pixel_coord


class BeaconMaskGenerator():
    def __init__(self):
        sam_checkpoint = '/projects/recon/RT1/data_generation/SAM_Model_Checkpoint_H.pth'
        model_type = 'vit_h'

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def _find_best_mask(self, masks, x, y):
        
        best_mask = None
        min_distance = float('inf')
        h, w = masks[0].shape 
        if x < 0 or x >= w or y < 0 or y >= h:
            return np.zeros((h, w), dtype=bool)

        for mask in masks:
            if 0 <= x < w and 0 <= y < h:
                if mask[y, x]:  
                    return mask
            true_pixels = np.argwhere(mask)
            if true_pixels.size == 0:
                continue  

            distances = cdist([(y, x)], true_pixels, metric='euclidean')
            min_dist = distances.min()
            if min_dist < min_distance:
                min_distance = min_dist
                best_mask = mask

        return best_mask if best_mask is not None else np.zeros((h, w), dtype=bool)
    
    
    def generate_all_masks(self, image, pixel_coords):
        result = self.mask_generator.generate(image)
        masks = [segmen_dict['segmentation'] for segmen_dict in result]
        masks_dict = {}
        for key, (x, y) in pixel_coords.items():
            mask = self._find_best_mask(masks, x, y)
            mask_uint8 = (mask * 255).astype(np.uint8)
            masks_dict[key] = mask_uint8
        return masks_dict
    
    def generate_one_mask(self, image, pixel_coords, masked_beacon):
        result = self.mask_generator.generate(image)
        masks = [segmen_dict['segmentation'] for segmen_dict in result]
        
        beacon_masks = []
        for beacon in masked_beacon:
            assert beacon in pixel_coords.keys(), colored("[ERROR] ", 'red')+f"Targted masked beacon {beacon} does not exist!"
            (x,y) = pixel_coords[beacon]
            mask = self._find_best_mask(masks, x, y)
            beacon_masks.append(mask)
        segmentation = np.zeros_like(beacon_masks[0], dtype=np.uint8) 
        for i, mask in enumerate(beacon_masks, start=1):
            segmentation[mask > 0] = i 
        return segmentation



