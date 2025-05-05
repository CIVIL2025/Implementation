import glob
import h5py
import tqdm
import pickle as pkl
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

def get_annotation(data_dir, dataset_name):
    annotation = np.load(f"{data_dir}/{dataset_name}/lang_annotations/auto_lang_ann.npy", allow_pickle = True)
    annotation = dict(enumerate(annotation.flatten(), 1))
    anno_ranges = annotation[1]["info"]["indx"]
    tasks = annotation[1]["language"]["task"]
    texts = annotation[1]["language"]["ann"]
    assert len(anno_ranges) == len(tasks), "Tasks and anno_ranges have different lengths: {} vs {}".format(len(tasks), len(anno_ranges))
    assert len(texts) == len(tasks), "Tasks and texts have different lengths: {} vs {}".format(len(tasks), len(texts))
    return anno_ranges, tasks, texts

DATASET_PATH = Path("/projects/recon/calvin/dataset")
SAVE_DATA_PATH = Path("../expert_demos/calvin")
# BENCHMARKS = ['calvin_debug_dataset']
BENCHMARKS = []
img_size = (128, 128)

# create save directory
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# load sentence transformer
lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Total number of tasks
datasets = ["training", "validation"]

for benchmark in BENCHMARKS: 
    benchmark_path = DATASET_PATH / benchmark
    # Find the list of unique tasks:
    annotation = np.load(f"{benchmark_path}/training/lang_annotations/auto_lang_ann.npy", allow_pickle = True)
    annotation = dict(enumerate(annotation.flatten(), 1))
    unique_tasks = list(set(annotation[1]["language"]["task"]))

    save_benchmark_path = SAVE_DATA_PATH / benchmark
    

    for dataset in datasets:
        # We are saving demos related to one task into a same file
        print(f"Unique tasks from the dataset: {unique_tasks}")
        data_dict = {
            task: {
                "observations": [],
                "states": [],
                "actions": [],
                "task_emb": []
            }
            for task in unique_tasks
        }
        timesteps = sorted(glob.glob(f"{benchmark_path}/{dataset}/episode_*.npz"))
        anno_ranges, tasks, texts = get_annotation(benchmark_path, dataset)
        
        for (start, end), task, text in zip(anno_ranges, tasks, texts): 
            pixels = []
            pixels_ego = []
            joint_states = []
            gripper_states = []
            eef_states = []
            observation = {"pixels": [], 
                        "pixels_egocentric": [], 
                        "joint_states": [],
                        "gripper_states": [],
                        "eef_states": []}
            beacon_states = []
            actions = []


            pbar = tqdm.tqdm(total=end - start +1)
            timestep_range = [(start + n) for n in range(0, end-start+1)]
            for timestep in timestep_range:
                file_path = f"{benchmark_path}/{dataset}/episode_{str(timestep).zfill(7)}.npz"
                data = np.load(file_path)

                pixels.append(data["rgb_static"]) # 200 * 200 * 3
                pixels_ego.append(data["rgb_gripper"]) # 84 * 84 * 3
                joint_states.append(data["robot_obs"][7:-1]) # joint angles
                eef_states.append(data["robot_obs"][:6])
                gripper_states.append([data["robot_obs"][-1]])

                beacon_states.append(data["scene_obs"]) 
                actions.append(data["actions"])
                pbar.update(1)
            pbar.close()
        

            observation["pixels"] = np.array(pixels, dtype=np.uint8)
            observation["pixels_egocentric"] = np.array(pixels_ego, dtype=np.uint8)
            observation["joint_states"] = np.array(joint_states, dtype=np.float32)
            observation["eef_states"] = np.array(eef_states, dtype=np.float32)
            observation["gripper_states"] = np.array(gripper_states, dtype=np.float32)

            beacon_states = np.array(beacon_states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)

            data_dict[task]['observations'].append(observation)
            data_dict[task]['states'].append(beacon_states)
            data_dict[task]['actions'].append(actions)
            data_dict[task]['task_emb'].append(lang_model.encode(text))
            # data_dict[task]['task_emb'].append(text)
        
        # save the converted dataset
        save_path = save_benchmark_path / dataset
        save_path.mkdir(parents=True, exist_ok=True)
        for task, data in data_dict.items():
            save_data_path = save_path / (
                task + ".pkl"
            )
            with open(save_data_path, "wb") as f:
                pkl.dump(data, f)
            print(f"Saved to {str(save_data_path)}")

    # Test code generated
    # for task in unique_tasks:
    #     print(f"For task {task}")
    #     print(f"num of demos from observations: {len(data_dict[task]['observations'])}")
    #     print(f"num of demos from beacon_states: {len(data_dict[task]['beacon_states'])}")
    #     print(f"num of demos from task_emb: {len(data_dict[task]['task_emb'])}")
    #     print(f"length of a task embedding: {len(data_dict[task]['task_emb'][0])}")