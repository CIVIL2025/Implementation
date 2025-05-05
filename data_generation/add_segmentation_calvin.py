
import os


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

from calvin_env.envs.play_table_env import get_env


DATA_CFG = Path('../../calvin/dataset')
SAVE_DATA_PATH = Path("../expert_demos/calvin")
BENCHMARKS = ['task_D_D']
TASKS = ['mdt_test_stack_on_pink_150']
datasets = ["training"]



for benchmark in BENCHMARKS:
    for task in TASKS:
        benchmark_path = SAVE_DATA_PATH / benchmark

        for dataset in datasets:

            cfg_folder = DATA_CFG / benchmark / dataset
            env = get_env(cfg_folder, show_gui=False)

            dataset_path = SAVE_DATA_PATH / task / dataset
            file_list = os.listdir(dataset_path)

            for file in file_list:
                if '_segmen' in task:
                    continue

                task_path = dataset_path / file
                task_demos = pickle.load(open(task_path, 'rb'))
                n_demos = len(task_demos['observations'])

                print(f'Adding segmentation map to {task_path}, with {n_demos} demos')

                for i in tqdm(range(n_demos)):

                    init_robot_obs = task_demos['observations'][i]['robot_obs'][0]
                    init_scene_obs = task_demos['observations'][i]['scene_obs'][0]
                    actions = task_demos['actions'][i]
                    rel_actions = task_demos['rel_actions'][i]

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
                    scene_beacon = []


                    for j in range(rel_actions.shape[0]):

                        pixels.append(obs['rgb_obs']['rgb_static'])
                        pixels_egocentric.append(obs['rgb_obs']['rgb_gripper'])
                        pixels_segmen.append(obs['segmen']['segmen_static'])
                        pixels_egocentric_segmen.append(obs['segmen']['segmen_gripper'])
                        robot_obs.append(obs['robot_obs'])
                        scene_obs.append(obs['scene_obs'])
                        scene_beacon.append(obs['scene_beacon'])
                        joint_states.append(obs['robot_obs'][7:-1])
                        ee_states.append(obs['robot_obs'][:6])
                        gripper_states.append(obs['robot_obs'][-1])

                        action = tuple(rel_actions[j, :])
                        obs, _, _, _ = env.step(action)

                    # print(np.unique(pixels_segmen))
                    # print(np.unique(pixels_egocentric_segmen))
                    task_demos['observations'][i]['pixels'] = np.array(pixels, dtype=np.uint8)
                    task_demos['observations'][i]['pixels_egocentric'] = np.array(pixels_egocentric, dtype=np.uint8)
                    task_demos['observations'][i]['pixels_segmen'] = np.array(pixels_segmen)
                    task_demos['observations'][i]['pixels_egocentric_segmen'] = np.array(pixels_egocentric_segmen)
                    task_demos['observations'][i]['robot_obs'] = np.array(robot_obs, dtype=np.float32)
                    task_demos['observations'][i]['scene_obs'] = np.array(scene_obs, dtype=np.float32)
                    task_demos['observations'][i]['scene_beacon'] = np.array(scene_beacon, dtype=np.float32)
                    task_demos['observations'][i]['joint_states'] = np.array(joint_states, dtype=np.float32)
                    task_demos['observations'][i]['ee_states'] = np.array(ee_states, dtype=np.float32)
                    task_demos['observations'][i]['gripper_states'] = np.array(gripper_states, dtype=np.float32)

                
                task_segmen_path = SAVE_DATA_PATH / f"{task}_segmen" / dataset
                save_name = file.replace(".pkl", "_segmen.pkl")
                os.makedirs(task_segmen_path, exist_ok=True)
                with open(task_segmen_path / save_name, "wb") as f:
                    pickle.dump(task_demos, f)
            


                



