import os
import hydra
import torch
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd 

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
from tqdm import tqdm
from termcolor import colored
from omegaconf import OmegaConf, DictConfig

from calvin_agent.evaluation.utils import join_vis_lang
from calvin_env.envs.play_table_env import get_env, PlayTableSimEnv
from calvin_env.envs.tasks import Tasks
from calvin_evaluation_utils import get_env_state_for_initial_condition
from calvin_evaluation_sequences import initilize_task_list, get_sequences

from utils import ModelObservationBuffer, AttentionRecorder, VideoRecorder
from policy_models.network_modules import BaseTransformerPolicy


NUM_SUBTASK=1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model: BaseTransformerPolicy,
                   save_dir: str,
                   model_name: str,
                   cfg: DictConfig,
                   env: PlayTableSimEnv,
                   eval_sequences: tuple[tuple, tuple],
                   val_annotations: DictConfig, 
                   task_oracle: Tasks, 
                   video_save_mode = "none",
                   attn_save_mode = "none",
                   step_length = 200,
                   render = False) -> tuple[float, np.float32]:

    total_success = 0
    rollout_distances = []
    video_save_dir = f"{save_dir}/eval_video_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    attn_save_dir =  f"{save_dir}/eval_attention_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    video_recorder = VideoRecorder(video_save_dir, save_mode = video_save_mode)
    obs_buffer = ModelObservationBuffer(cfg.model, cfg.params.seq_len, cfg.dataset.target_beacons, cfg.dataset.target_masks) ###

    if hasattr(model, 'token_labels'):
        n_layers = cfg.model.model.num_layers if hasattr(cfg.model.model, 'num_layers') else 2

        attn_recorder = AttentionRecorder(attn_save_dir, 
                                          save_mode=attn_save_mode, 
                                          seq_len=cfg.params.seq_len, 
                                          n_layers=n_layers,
                                          token_labels=model.token_labels)
    else:
        attn_recorder = AttentionRecorder(attn_save_dir,
                                          save_mode=attn_save_mode,
                                          seq_len=cfg.params.seq_len)
        

    for total_trial, (initial_state, eval_sequence) in enumerate(eval_sequences):
        robot_obs, scene_obs = initial_state
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        if "stacking_block" in cfg.dataset.dataset_dir:
            print(colored("Close gripper for red block in hand task", "green"))
            _, _, _, _ = env.step(np.array([0.0,0.0,0.0,0.0,0.0,0.0,-1.0]))
        # Initialize obs buffer
        obs = env.get_obs()
        obs_buffer.reset_buffer(obs)

        success_subtask = 0
        distance2block = []

        for subtask in eval_sequence:
            if subtask in ["stack_on_pink_block", "stack_on_blue_block"]:
                lang_annotation = val_annotations['stack_block'][0]
            else:
                lang_annotation = val_annotations[subtask][0]
            start_info = env.get_info()

            success = False
            for step in range(step_length):
                step_seq = obs_buffer.get_sequence()
                for key, value in step_seq.items():
                    step_seq[key] = value.to(DEVICE)

                test_data, action = model.get_action(step_seq)        

                if action[-1] <= 0:
                    action[-1] = -1
                else:
                    action[-1] = 1

                if cfg.model.action_type == 'abs':
                    action = ((action[:3]), (action[3:6]), ([action[-1]]))
                    action = tuple(map(tuple, action)) 
                elif cfg.model.action_type == 'rel_actions':
                    pass

                obs, _, _, current_info = env.step(action)

                # TODO: this class cannot handle variable beacon dimension
                video_recorder.record(env, obs, test_data, lang_annotation, cfg.dataset.target_beacons)
                attn_recorder.record(step, test_data)


                # Update buffer
                obs_buffer.apppend_obs(obs)
                
                # Calculate the l2 distance between block and robot ee
                ee_pos = obs['robot_obs'][:3]
                red_block = obs['scene_beacon'][36:39]
                distance2block.append(np.linalg.norm(ee_pos - red_block))

                if render:
                    attn_recorder.visualize(total_trial, step, test_data["attn_weights"])

                    img = env.render(mode="rgb_array")
                    join_vis_lang(img, lang_annotation)
                
                # Check if current step solves a task
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    print(colored("success", "green"), end=" ")
                    success = True
                    break
            if success:
                success_subtask += 1
            else:
                print("Fail")
                break
        
        if render:
            attn_recorder.close_vis()

        rollout_sum_distance = np.sum(distance2block / distance2block[0])
        rollout_distances.append(rollout_sum_distance)
        if NUM_SUBTASK == success_subtask:
            total_success += 1
            print(colored('[Rollout results]', 'green') + f'success rate : {total_success / (total_trial + 1)}, sum_distance: {rollout_sum_distance}, n_rollout: {len(rollout_distances)}')
        video_recorder.save(model_name, NUM_SUBTASK == success_subtask)

        attn_recorder.save(f"attn_map_rollout_{total_trial}", total_trial, NUM_SUBTASK == success_subtask)
    success_rate = total_success / (total_trial + 1)
    avg_sum_distance = np.mean(rollout_distances)
    print(colored('[Eval results]', 'green') + f'success rate: {success_rate}, avg_distance: {avg_sum_distance}, n_rollouts: {len(rollout_distances)}')

    return success_rate, avg_sum_distance

def save_results(save_dir, snapshot_name, success_rate, avg_distance, round, init_condition, evaluation):
    if init_condition is not None:
        file_path = f"{save_dir}/results_{evaluation}_{str(round).zfill(2)}_extended.csv"
    else:
        file_path = f"{save_dir}/results_{evaluation}_{str(round).zfill(2)}.csv"
    os.makedirs(save_dir , exist_ok = True)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()
    df[snapshot_name] = [success_rate, avg_distance]
    df.to_csv(file_path, index=False)
    print(colored('[Eval] ', 'green') + f'Saving evaluation results to {file_path}')
 
@hydra.main(version_base=None, config_path="conf", config_name="test_calvin")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # import pdb
    # pdb.set_trace()
    env = get_env(cfg.env.val_folder, show_gui=cfg.env.show_gui)
    task_oracle = hydra.utils.instantiate(OmegaConf.load(cfg.env.task_cfg))
    val_annotations = OmegaConf.load(cfg.env.val_annotations)


    for round in range(cfg.test_round):

        # Get eval sequences
        if cfg.init_condition_path is not None:
            eval_sequences = []
            for path in cfg.init_condition_path:
                with open(path, "rb") as file:
                    eval_sequences.extend(pickle.load(file))
            # random.shuffle(eval_sequences)
            print(colored('NOT SHUFFLING', 'red'))
            eval_sequences = eval_sequences[: cfg.num_sequences]
        else:
            initilize_task_list(cfg.evaluation.task_categories,
                                cfg.evaluation.tasks)
            eval_sequences_temp = get_sequences(cfg.evaluation.possible_conditions,
                                                num_sequences=cfg.num_sequences)
            eval_sequences = []
            progress_bar = tqdm(total = len(eval_sequences_temp), desc="Generating initial conditions...")
            for (initial_state, eval_sequence) in eval_sequences_temp:
                eval_sequences.append((get_env_state_for_initial_condition(initial_state, cfg.evaluation, env), eval_sequence))
                progress_bar.update(1)

        for dir in cfg.model_list.models_dir:

            snapshots_test = []
            for snapshot in os.listdir(f"{dir}/calvin_models"):
                _, checkpoint = snapshot.replace(".pt", "").rsplit("_", 1)
                checkpoint = checkpoint if checkpoint == 'best' or checkpoint =='trainbest' else int(checkpoint)

                if checkpoint in cfg.model_list.tested_checkpoint:
                    snapshots_test.append(snapshot)
            snapshots_test = sorted(snapshots_test)
            print(colored('[Eval]', 'green') + f'Snapshots to be evaluated: {snapshots_test}')

            save_dir = f'{dir}/eval_results'
            training_cfg = OmegaConf.load(f"{dir}/.hydra/config.yaml")
            
            for snapshot in snapshots_test:
                snapshot_name = snapshot.replace(".pt", "")
                snapshot_path = f"{dir}/calvin_models/{snapshot}"

                model = hydra.utils.instantiate(training_cfg.model.model)
                if hasattr(model, 'register_intermediate_attention_hooks'):
                    model.register_intermediate_attention_hooks()

                print(colored('[Eval] ', 'green') + f"Evaluating: " + colored(model.name, 'green') + f" from {snapshot_path}")
                print(colored('[Eval] ', 'green') +  f'Saving evaluation results to: {save_dir}')

                try:
                    model.load_state_dict(torch.load(snapshot_path, weights_only=True, map_location=DEVICE), strict=True)
                except:
                    print(colored("[**********************************ALERT**********************************]\n", "red") + 
                          colored("Some part of the model is missing or have changed! It might be fine if this is a CLIP model, otherwise it is a SERIOUS ISSUE.", "red"))
                    model.load_state_dict(torch.load(snapshot_path, weights_only=True, map_location=DEVICE), strict=False)
                model.eval().to(DEVICE)

                with torch.no_grad():
                    success_rate, avg_distance = evaluate_model(model,
                                                                save_dir,
                                                                snapshot_name,
                                                                training_cfg,
                                                                env,
                                                                eval_sequences,
                                                                val_annotations,
                                                                task_oracle,
                                                                cfg.video_save_mode,
                                                                cfg.attn_save_mode,
                                                                cfg.step_len,
                                                                cfg.debug)
                    save_results(save_dir, snapshot_name, success_rate, avg_distance, round + 1, cfg.init_condition_path, cfg.evaluation.name)

if __name__ == '__main__':
    main()
