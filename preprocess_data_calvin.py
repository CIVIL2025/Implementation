import pickle
import os
from termcolor import colored
import torch
import torchvision
from torchvision.transforms import v2, InterpolationMode
import numpy as np
import h5py

"""
keys inside the calvin demo dict:
observations: pixels, pixels_egocentric, joint_states, eef_states, gripper_states
            + pixels_segmen, pixels_egocentric_segmen if segmen
robot_obs 
scene_obs:
    (dtype=np.float32, shape=(24,))
    sliding door (1): joint state
    drawer (1): joint state
    button (1): joint state
    switch (1): joint state
    lightbulb (1): on=1, off=0
    green light (1): on=1, off=0
    red block (6): (x, y, z, euler_x, euler_y, euler_z)
    blue block (6): (x, y, z, euler_x, euler_y, euler_z)
    pink block (6): (x, y, z, euler_x, euler_y, euler_z)
actions: "ee absolute actions" 
rel_Actions: "ee relative actions"
task_emb: "language task embedding
"""

DEFAULT_MASK_MAPPING = {'robot': 0,
                'robot_base': 1,
                'red_block': 2,
                'blue_block': 3,
                'purple_block': 4,
                'table': 5,
                'background': 6}

MASK_MAPPING = {'robot': {'type': 'body', 'bodyId': 0},
                'robot_base': {'type': 'body', 'bodyId': 1},
                'red_block': {'type': 'body', 'bodyId': 2},
                'blue_block': {'type': 'body', 'bodyId': 3},
                'purple_block': {'type': 'body', 'bodyId': 4},
                'table': {'type': 'body', 'bodyId': 5},
                'background': {'type': 'body', 'bodyId': 6},
                'slider': {'type': 'link', 'bodyId': 5, 'linkId': [2]},
                'drawer': {'type': 'link', 'bodyId': 5, 'linkId': [3]},
                'button': {'type': 'link', 'bodyId': 5, 'linkId': [0]},
                'switch': {'type': 'link', 'bodyId': 5, 'linkId': [1]},
                'light': {'type': 'link', 'bodyId': 5, 'linkId': [5]},
                'green_light': {'type': 'link', 'bodyId': 5, 'linkId': [4]},
                'gripper_fingers': {'type': 'link', 'bodyId': 0, 'linkId': [9, 10, 11, 12]}}


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



def process_image_rgb(images, target_img_size):

    if target_img_size != images.shape[1:3]:
        resize_img = v2.Compose([
            v2.Resize(target_img_size, antialias=True)
        ])
        images = resize_img(torch.from_numpy(images).permute(0, 3, 1, 2))
        images = images.permute(0, 2, 3, 1).detach().cpu().numpy()

    return images

def resize_mask(mask, target_mask_size):
    transform = v2.Compose([
        v2.Resize(target_mask_size, interpolation=InterpolationMode.NEAREST)
    ])
    return transform(torch.from_numpy(mask)).detach().cpu().numpy()
    

def process_binary_mask(segmentation, target_masks, target_mask_size, separate_channel_mask=False):

    mask = []

    for target in target_masks:
        if np.max(segmentation[0]) <= 6 or np.max(segmentation[0])==255:
            mask.append(np.where(segmentation == DEFAULT_MASK_MAPPING[target], 1, 0))
        else:
            if MASK_MAPPING[target]['type'] == 'body':
                merged = np.where((segmentation & ((1 << 24) - 1)) == MASK_MAPPING[target]['bodyId'], 1, 0)
            elif MASK_MAPPING[target]['type'] == 'link':
                bodyId = MASK_MAPPING[target]['bodyId']
                linkId = MASK_MAPPING[target]['linkId']
                merged = np.where(((segmentation & ((1 << 24) - 1)) == bodyId) & (np.isin((segmentation >> 24) - 1, linkId)), 1, 0)
            else:
                print("Unknown target type")
                continue

            mask.append(merged)
            
    if separate_channel_mask:
        mask = np.stack(mask, axis=1)
    else:
        mask = np.sum(mask, axis=0)
        mask = np.expand_dims(mask, axis=1)

    mask = mask.astype(np.float32)

    if target_mask_size != mask.shape[-2:]:
        mask = resize_mask(mask, target_mask_size)

    return mask

def mask2pixelcount(segmentation, target_masks, target_mask_size):
    mask = process_binary_mask(segmentation, target_masks, target_mask_size, separate_channel_mask=True)

    mask = np.moveaxis(mask, [0, 1], [1, 0])
    pixel_count = np.count_nonzero(mask, axis=(2,3))

    return pixel_count.transpose()

def preprocess_data(data_fd, save_name, cfg):

    data_paths = [dir for dir in os.listdir(data_fd) if not 'augmented' in dir]

    # Save all folder demos into one file
    all_path = f'{cfg.output_dir}/{save_name}'
    all_f = h5py.File(all_path, 'w')
    all_data_grp = all_f.create_group("data")
    all_total_len = 0

    for confg_idx, path in enumerate(data_paths):
        with_segmen = True if '_segmen' in path else False
        print(path)
        print(colored("[mask]", 'green') + f'Dataset contains mask: {with_segmen}')
        print(colored("[action_type]", "green") + f'Using {cfg.model.action_type}')


        task_demos = pickle.load(open(f"{data_fd}/{path}", 'rb'))
        n_demos = len(task_demos['observations'])

        # # Save data in hdf5 file to use viola's custom data loader
        # file_name = path.replace(".pkl", "_augmented.hd5f")
        # h5py_path = f"{cfg.output_dir}/{file_name}"
        # h5py_f = h5py.File(h5py_path, 'w')
        # data_grp = h5py_f.create_group("data")

        print(f"[*] Saving {n_demos} demos to {all_path}")

        total_len = 0

        for i in range(n_demos):
            
            static_view_rgb = task_demos["observations"][i]["pixels"]
            static_view_rgb = process_image_rgb(static_view_rgb, cfg.model.img_size)
            
            egocentric_view_rgb = task_demos["observations"][i]["pixels_egocentric"]
            if hasattr(cfg.model, "img_ego_size"):
                egocentric_view_rgb = process_image_rgb(egocentric_view_rgb, cfg.model.img_ego_size)
            else:
                egocentric_view_rgb = process_image_rgb(egocentric_view_rgb, cfg.model.img_size)

            # joint_states = task_demos["observations"][i]["joint_states"]
            # gripper_states = np.expand_dims(task_demos["observations"][i]["gripper_states"], axis=1)
            # ee_states = task_demos["observations"][i]["ee_states"]
            joint_states = task_demos["observations"][i]["robot_obs"][:, 7:-1]
            gripper_states = np.expand_dims(task_demos["observations"][i]["robot_obs"][:, -1], axis=1)
            ee_states = task_demos["observations"][i]["robot_obs"][:, :6]

            actions = task_demos[cfg.model.action_type][i]

            beacons = []
            for target in cfg.dataset.target_beacons:
                if 'scene_beacon' in task_demos['observations'][0].keys():
                    target_beacon = task_demos['observations'][i]['scene_beacon'][:, BEACON_MAPPING[target]]
                elif 'scene_obs' in task_demos['observations'][0].keys():
                    target_beacon = task_demos['observations'][i]['scene_obs'][:, DEFAULT_BEACON_MAPPING[target]]
                beacons.append(target_beacon)
            beacons = np.concatenate(beacons, axis=1)

            if with_segmen:
                if hasattr(cfg.model, "bbox_type") and cfg.model.bbox_type == 'viola':
                    print(colored("[bbox]", "green") + "Using viola's bbox")
                    target_masks = cfg.dataset.viola_masks
                else: target_masks = cfg.dataset.target_masks
                segmentation = task_demos["observations"][i]["pixels_segmen"]
                ego_segmentation = task_demos["observations"][i]["pixels_egocentric_segmen"]

                static_view_mask = process_binary_mask(segmentation,
                                                       target_masks, 
                                                       cfg.model.mask_size, 
                                                       cfg.model.separate_channel_mask)

                if hasattr(cfg.model, "mask_ego_size"):
                    ego_mask_size = cfg.model.mask_ego_size
                else:
                    ego_mask_size = cfg.model.mask_size
                
                ego_view_bool = mask2pixelcount(ego_segmentation,
                                                target_masks,
                                                ego_mask_size)
                ego_view_mask = process_binary_mask(ego_segmentation,
                                                    target_masks,
                                                    ego_mask_size,
                                                    cfg.model.separate_channel_mask)

                if cfg.model.separate_channel_mask:
                    steps, objects, height, width = static_view_mask.shape
                    temp_masks = torch.FloatTensor(static_view_mask).view(steps*objects, height, width)
                    static_view_bboxes = torch.zeros(steps*objects, 4)
                    available_bboxes = torch.any(temp_masks, dim=[1, 2])
                    static_view_bboxes[available_bboxes] = torchvision.ops.masks_to_boxes(temp_masks[available_bboxes])
                    static_view_bboxes = static_view_bboxes.view(steps, objects, 4).detach().cpu().numpy()

                    steps, objects, height, width = ego_view_mask.shape
                    temp_masks = torch.FloatTensor(ego_view_mask).view(steps*objects, height, width)
                    ego_view_bboxes = torch.zeros(steps*objects, 4)
                    available_bboxes = torch.any(temp_masks, dim=[1, 2])
                    ego_view_bboxes[available_bboxes] = torchvision.ops.masks_to_boxes(temp_masks[available_bboxes])
                    ego_view_bboxes = ego_view_bboxes.view(steps, objects, 4).detach().cpu().numpy()

                if 'static_lang_segmen' in task_demos["observations"][i].keys():
                    static_lang_mask = np.array([np.expand_dims(mask, axis=0) for mask in task_demos["observations"][i]["static_lang_segmen"]])
                if 'ego_lang_segmen' in task_demos["observations"][i].keys():
                    ego_lang_mask = np.array([np.expand_dims(mask, axis=0) for mask in task_demos["observations"][i]["ego_lang_segmen"]])
                    if ego_lang_mask.shape[-2:] != ego_mask_size:
                        ego_lang_mask = resize_mask(ego_lang_mask, ego_mask_size)
            # # Data for individual files
            # demo_grp = data_grp.create_group(f"demo_{i}")

            # # NOTE: Save images in [..., H, W, Channels] format 
            # obs_group = demo_grp.create_group("obs")
            # obs_group.create_dataset("image_rgb", data=static_view_rgb)
            # obs_group.create_dataset("joint_states", data=joint_states)
            # obs_group.create_dataset("ee_states", data=ee_states)
            # obs_group.create_dataset("gripper_states", data=gripper_states)
            # obs_group.create_dataset("beacons", data=beacons)
            # if with_segmen:
            #     obs_group.create_dataset("image_mask", data=static_view_mask)

            # Data to merged file
            all_demo_grp = all_data_grp.create_group(f"demo_{i + n_demos*confg_idx}")

            # NOTE: Save images in [..., H, W, Channels] format 
            all_obs_group = all_demo_grp.create_group("obs")
            all_obs_group.create_dataset("image_rgb", data=static_view_rgb)
            all_obs_group.create_dataset("image_ego", data=egocentric_view_rgb)
            all_obs_group.create_dataset("joint_states", data=joint_states)
            all_obs_group.create_dataset("ee_states", data=ee_states)
            all_obs_group.create_dataset("gripper_states", data=gripper_states)
            all_obs_group.create_dataset("beacons", data=beacons)
            if with_segmen:
                if hasattr(cfg.model, "mask_type") and cfg.model.mask_type == 'langsam':
                    if 'static_lang_segmen' in task_demos["observations"][i].keys():
                        all_obs_group.create_dataset("image_mask", data=static_lang_mask)
                    if 'ego_lang_segmen' in task_demos["observations"][i].keys():
                        all_obs_group.create_dataset("image_ego_mask", data=ego_lang_mask)
                else: 
                    all_obs_group.create_dataset("image_mask", data=static_view_mask)
                    all_obs_group.create_dataset("ego_bool", data=ego_view_bool)
                    all_obs_group.create_dataset("image_ego_mask", data=ego_view_mask)

                if cfg.model.separate_channel_mask:
                    all_obs_group.create_dataset("image_bbox", data=static_view_bboxes)
                    all_obs_group.create_dataset("image_ego_bbox", data=ego_view_bboxes)
   

            all_demo_grp.create_dataset("actions", data=actions)
            all_demo_grp.attrs["num_samples"] = len(static_view_rgb)
            all_total_len += len(static_view_rgb)


        # data_grp.attrs["total"] = total_len
        # data_grp.attrs["num_demos"] = n_demos
        # h5py_f.close()

    all_data_grp.attrs["total"] = all_total_len
    all_data_grp.attrs["num_demos"] = n_demos * len(data_paths)
    all_f.close()
    print(f'[*] Total n demos: {n_demos * len(data_paths)}, saved to: {all_path}')