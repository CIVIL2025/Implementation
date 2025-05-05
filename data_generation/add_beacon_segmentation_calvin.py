
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path

from calvin_env.envs.play_table_env import get_env
from preprocess_utils import get_pixel_coords, BeaconMaskGenerator

def visualize_masks(obs, pixel_coord, masks):
    img_viz = obs['rgb_obs']['rgb_gripper'].copy()
    for label, (x, y) in pixel_coord.items():
        cv2.circle(img_viz, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for key, mask in masks.items():
        overlay = cv2.bitwise_and(img_viz, img_viz, mask=mask)
        output_path = f"/projects/recon/new_recon_transformer/training_outputs/mask_{key}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.png"
        success = cv2.imwrite(output_path, overlay)
    if success:
        input(f"Images Saved") # to prevent a multitude of images from being saved
    else:
        input("Error saving image")


DATA_CFG = Path('../../calvin/dataset')
SAVE_DATA_PATH = Path("../expert_demos/calvin")
BENCHMARKS = ['task_D_D']
TASKS = ['mdt_red_two_pos_train_segmen_50']
datasets = ["training"] 
get_beacon_segmen = True
masked_beacon = ['red_block_pos'] #'red_block_pos', 'blue_block_pos', 'pink_block_pos', 'slider_pos', 'drawer_pos', 'switch_pos'

masker = BeaconMaskGenerator()

for benchmark in BENCHMARKS:
    for task in TASKS:
        for dataset in datasets:

            cfg_folder = DATA_CFG / benchmark / dataset
            env = get_env(cfg_folder, show_gui=False)

            dataset_path = SAVE_DATA_PATH / task / dataset
            task_list = os.listdir(dataset_path)

            for subtask in task_list:

                task_path = dataset_path / subtask
                task_demos = pickle.load(open(task_path, 'rb'))
                n_demos = len(task_demos['observations'])

                print(f'Adding segmentation map to {task_path}, with {n_demos} demos')

                for i in tqdm(range(n_demos)):

                    init_robot_obs = task_demos['observations'][i]['robot_obs'][0]
                    init_scene_obs = task_demos['observations'][i]['scene_obs'][0]
                    actions = task_demos['actions'][i]

                    env.reset(init_robot_obs, init_scene_obs)
                    obs = env.get_obs()

                    # ID list: 0: robot, 1: robot base, 2: red block, 3: blue block, 4: purple block, 5: table, 6: background
                    # NOTE: reseting the environment and applying the same action does not lead to the same state
                    # thus we are updating the all states to and saving to a new file
                    pixels = []
                    pixels_egocentric = []
                    pixels_segmen = []
                    pixels_egocentric_segmen = []
                    robot_obs = []
                    scene_obs = []
                    joint_states = []
                    ee_states = []
                    gripper_states = []
                    pixel_coords = []
                    adjusted_beacons = []
                    beacon_masks = []


                    for j in range(actions.shape[0]):

                        pixels.append(obs['rgb_obs']['rgb_static'])
                        pixels_egocentric.append(obs['rgb_obs']['rgb_gripper'])
                        pixels_segmen.append(obs['segmen']['segmen_static'])
                        pixels_egocentric_segmen.append(obs['segmen']['segmen_gripper'])
                        robot_obs.append(obs['robot_obs'])
                        scene_obs.append(obs['scene_obs'])
                        joint_states.append(obs['robot_obs'][7:-1])
                        ee_states.append(obs['robot_obs'][:6])
                        gripper_states.append(obs['robot_obs'][-1])
                        
                        # pixel coordinates
                        adjusted_beacon, pixel_coord = get_pixel_coords(env, obs)
                        adjusted_beacons.append(adjusted_beacon)
                            
                        if get_beacon_segmen:
                            # masks = masker.generate_all_masks(obs['rgb_obs']['rgb_gripper'], pixel_coord)
                            beacon_segmen = masker.generate_one_mask(obs['rgb_obs']['rgb_gripper'], pixel_coord, masked_beacon)
                            beacon_masks.append(beacon_segmen)

                            # visualize_masks(obs, pixel_coord, masks)

                        action = actions[j, :]
                        action = ((action[:3]), (action[3:6]), ([action[-1]]))
                        action = tuple(map(tuple, action))

                        obs, _, _, _ = env.step(action)

                    task_demos['observations'][i]['pixels'] = np.array(pixels, dtype=np.uint8)
                    task_demos['observations'][i]['pixels_egocentric'] = np.array(pixels_egocentric, dtype=np.uint8)
                    task_demos['observations'][i]['pixels_segmen'] = np.array(pixels_segmen, dtype=np.uint8)
                    task_demos['observations'][i]['pixels_egocentric_segmen'] = np.array(pixels_egocentric_segmen, dtype=np.uint8)
                    task_demos['observations'][i]['robot_obs'] = np.array(robot_obs, dtype=np.float32)
                    task_demos['observations'][i]['scene_obs'] = np.array(scene_obs, dtype=np.float32)
                    task_demos['observations'][i]['joint_states'] = np.array(joint_states, dtype=np.float32)
                    task_demos['observations'][i]['ee_states'] = np.array(ee_states, dtype=np.float32)
                    task_demos['observations'][i]['gripper_states'] = np.array(gripper_states, dtype=np.float32)
                    task_demos['states'][i] = np.array(adjusted_beacons, dtype=np.float32)
                    if get_beacon_segmen:
                        print("Getting segmentation form beacon...")
                        task_demos['observations'][i]['beacon_segmen'] = np.array(beacon_masks, dtype=np.uint8)
                    
                save_dir = SAVE_DATA_PATH / Path(f"{'_'.join(task.split('_')[:-1]) + '_beacon_mask_' + str(n_demos)}/{dataset}")
                os.makedirs(save_dir, exist_ok=True)
                task_segmen_path = f"{save_dir}/{subtask}"
                with open(task_segmen_path, "wb") as f:
                    pickle.dump(task_demos, f)