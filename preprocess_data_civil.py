import os
import h5py
import torch
import pickle
import numpy as np

from natsort import os_sorted
from termcolor import colored
from torchvision.transforms import v2, InterpolationMode


MASK_MAPPING = {'red_cup': 1,
                'red_plate': 2,
                'red_button': 1,
                'pan': 1,
                'all': 0,
                }

BEACON_MAPPING = {'red_plate_pos': slice(0, 3),
                  'red_plate_rot': slice(3, 6),
                  'red_cup_pos': slice(6, 9),
                  'red_cup_rot': slice(9, 12),
                  'red_button_pos': slice(0, 3),
                  'red_button_rot': slice(0, 6),
                  'pan_pos': slice(0, 3),
                  'pan_rot': slice(3, 6),
                  }

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
        v2.Resize(target_mask_size, interpolation=InterpolationMode.BILINEAR)
    ])
    return transform(torch.from_numpy(mask)).detach().cpu().numpy()
    

def process_binary_mask(segmentation, target_masks, target_mask_size, mask_mapping, separate_channel_mask=False):

    mask = []

    for target in target_masks:
        if target == 'all':
            merged = np.where(segmentation != mask_mapping[target], 1, 0)
        else:
            merged = np.where(segmentation == mask_mapping[target], 1, 0)
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


def preprocess_data(data_fd, save_name, cfg, play_data=False):

    data_paths = [dir for dir in os_sorted(os.listdir(data_fd))]

    # Save all demos in the folder to one file
    all_path = f'{cfg.output_dir}/{save_name}'
    all_f = h5py.File(all_path, 'w')
    all_data_grp = all_f.create_group('data')
    all_total_len = 0

    if cfg.params.use_color_jitter:
        color_transform = v2.Compose([v2.ColorJitter(**cfg.params.color_jitter)])

    for file_idx, file_path in enumerate(data_paths):
        print(colored(f'[*] Processing {file_path}', 'green'))
        
        with open(f"{data_fd}/{file_path}", "rb") as file:
            file_demo = pickle.load(file)

        if not play_data:
            # Robot states and actions
            joint_states = np.asarray(file_demo['joint_state'])
            if type(file_demo['gripper_state'][0]) == list:
                file_demo['gripper_state'][0] = file_demo['gripper_state'][0][0] 
            gripper_states = np.expand_dims(np.asarray(file_demo['gripper_state']), axis=1)
            ee_states = np.asarray(file_demo['ee_state'])

            actions = np.asarray(file_demo[cfg.action_type])
            if type(file_demo['gripper_action'][0]) == list:
                file_demo['gripper_action'][0] = file_demo['gripper_action'][0][0] 
            gripper_actions = np.expand_dims(np.asarray(file_demo['gripper_action']), axis=1)
            actions = np.concatenate((actions, gripper_actions), axis=1)
            if 'bboxes' in file_demo.keys():
                img_bboxes = np.asarray(file_demo['bboxes'])
            if 'language' in file_demo.keys():
                language = np.asarray(file_demo['language'])

        beacon_mapping = BEACON_MAPPING
        mask_mapping = MASK_MAPPING

        # Beacon readings
        # NOTE: note the beacons are not always available so the model loss calculation needs to filter out the NaN cases
        beacons = []
        all_beacons = np.asarray(file_demo['beacons'])
        for target in cfg.dataset.target_beacons:
            target_beacon = all_beacons[:, beacon_mapping[target]]
            beacons.append(target_beacon)
        beacons = np.concatenate(beacons, axis=1)

        # Images processing
        # NOTE: original images is bgr format, inpainted images were changed to rgb
        static_view_rgb = np.asarray(file_demo['img_inpainted'])
        static_view_rgb = process_image_rgb(static_view_rgb, cfg.img_size)
        egocentric_view_rgb = np.asarray(file_demo['img_gripper_inpainted'])

        if hasattr(cfg.model, "img_ego_size"):
            egocentric_view_rgb = process_image_rgb(egocentric_view_rgb, cfg.img_ego_size)
        else:
            egocentric_view_rgb = process_image_rgb(egocentric_view_rgb,
            cfg.img_size)

        
        if cfg.params.use_color_jitter and file_idx % cfg.params.keep_percent != 0:
            static_view_tensor = torch.from_numpy(static_view_rgb).permute(0, 3, 1, 2)
            egocentric_view_tensor = torch.from_numpy(egocentric_view_rgb).permute(0, 3, 1, 2)
            input_tensor = torch.cat((static_view_tensor, egocentric_view_tensor), dim=0)
            out_tensor = color_transform(input_tensor)
            out_tensor = torch.split(out_tensor, [len(static_view_tensor), len(egocentric_view_tensor)], dim=0)

            static_view_rgb = out_tensor[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            egocentric_view_rgb = out_tensor[1].permute(0, 2, 3, 1).detach().cpu().numpy()
            print(colored(f'[*] Applying color jitting to {file_path}', 'red'))
        
        # Images segmentation
        segmentation = np.asarray(file_demo['img_segmen'])
        static_view_mask = process_binary_mask(segmentation,
                                               cfg.dataset.target_masks,
                                               cfg.mask_size,
                                               mask_mapping,
                                               cfg.separate_channel_mask)

        
        if hasattr(cfg.model, "mask_ego_size"):
            ego_mask_size = cfg.mask_ego_size
        else:
            ego_mask_size = cfg.mask_size

        ego_segmentation = np.asarray(file_demo['img_gripper_segmen'])
        ego_view_mask = process_binary_mask(ego_segmentation,
                                            cfg.dataset.target_masks,
                                            ego_mask_size,
                                            mask_mapping,
                                            cfg.separate_channel_mask)  
        
        # Data to merged file
        all_demo_grp = all_data_grp.create_group(f"demo_{file_idx}")

        # Observations
        # NOTE: Save images in [..., H, W, Channels] format 
        all_obs_group = all_demo_grp.create_group("obs")
        all_obs_group.create_dataset("image_rgb", data=static_view_rgb)
        all_obs_group.create_dataset("image_ego", data=egocentric_view_rgb)
        all_obs_group.create_dataset("image_mask", data=static_view_mask)
        all_obs_group.create_dataset("image_ego_mask", data=ego_view_mask)
        all_obs_group.create_dataset("beacons", data=beacons)

        if not play_data:
            all_obs_group.create_dataset("joint_states", data=joint_states)
            all_obs_group.create_dataset("ee_states", data=ee_states)
            all_obs_group.create_dataset("gripper_states", data=gripper_states)
            if 'bboxes' in file_demo.keys():
                all_obs_group.create_dataset("image_bbox", data=img_bboxes)
            else:
                all_obs_group.create_dataset("image_bbox", data=np.zeros((len(static_view_rgb), 4)))
            if 'language' in file_demo.keys():
                all_obs_group.create_dataset("language", data=language)

            # Actions
            all_demo_grp.create_dataset("actions", data=actions)
        
        # Attributes for sequence dataset
        all_demo_grp.attrs["num_samples"] = len(static_view_rgb)
        all_total_len += len(static_view_rgb)

    all_data_grp.attrs["total"] = all_total_len
    all_data_grp.attrs["num_demos"] = len(data_paths)
    all_f.close()

    print(f'[*] Total n demos {len(data_paths)}, saved to: {all_path}')
    