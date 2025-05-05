
import os
import cv2
import time
import copy
import torch
import hydra
import pickle
import torchvision
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils

from omegaconf import OmegaConf
from argparse import ArgumentParser
from torchvision.transforms import v2
from transformers import BertModel, BertTokenizer
from utils import FR3, Joystick, camera, ModelObservationBuffer, VideoRecorder, GDonThread


from deva.inference.eval_args import add_common_eval_args
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.inference.inference_core import DEVAInferenceCore
from deva.model.network import DEVA
from deva.ext.grounding_dino import get_grounding_dino_model

try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    # not sure why this happens sometimes
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel


def grounding_dino_args(parser: ArgumentParser):
    # Grounded Segment Anything
    parser.add_argument('--GROUNDING_DINO_CONFIG_PATH',
                        default='../Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py')

    parser.add_argument('--GROUNDING_DINO_CHECKPOINT_PATH',
                        default='../Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth')

    parser.add_argument('--DINO_THRESHOLD', default=0.40, type=float)
    parser.add_argument('--DINO_NMS_THRESHOLD', default=0.8, type=float)

    parser.add_argument('--prompt', type=str, help='Separate classes with a single fullstop', default='cup for drinking.red empty plate')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lang_model = BertModel.from_pretrained('bert-base-uncased')
def get_bert_embedding(text):
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding])
    outputs = lang_model(input_ids)
    last_hidden_states = outputs.last_hidden_state
    cls_token = last_hidden_states[0][0]
    return cls_token


def main():
    robot = FR3()
    control_mode = "v"

    print('[*] Connecting to robot...')
    conn = robot.connect(8080)
    print('[*] Connection complete')

    print('[*] Connecting to gripper')
    conn_gripper = robot.connect(8081)
    print('[*] Connection complete')

    interface = Joystick()

    # Home position
    # START = [-1.37002, 0.900083, 1.41625, -1.88775, -0.17674, 2.05773, -0.722304] # stirring task
    # START = [-1.45233, 0.91503, 1.39972, -1.92329, -0.310326, 2.04672, 2.38064] # coffee making
    # START = [-0.0413756, -0.08899825, 0.0123554, -1.80487, -0.000971178, 2.08393, 0.71632] # push button
    # START = [-0.11452, 0.0462251, 0.205407, -1.31481, -0.0074332, 1.37608, 0.882886] # red_plate red_cup
    # START = [-0.0832917, 0.0534155, 0.19788, -1.47245, -0.061943, 1.88297, 0.866635] # bowl pulling
    # START = [-0.123849, 0.173138, 0.407359, -2.13113, -0.110262, 2.30485, 1.13426] #placing
    START = [-0.177101, -0.0234967, 0.208954, -1.5112, -0.10339, 1.91946, 0.846834] # Doh

    robot.go2position(conn, START)
    robot.send2gripper(conn_gripper, "o")
    gripper_state = [1]
    prev_gripper_state = 1
    # robot.send2gripper(conn_gripper, "c")
    # gripper_state = [-1]
    # prev_gripper_state = -1

    # text_input = "Scooping up the meat in the pan."
    # text_input = "Stir the vegetables in the pan."
    # text_input = "Push down the red button."
    # text_input = "Pick up the cup and place it in the red empty plate."
    # text_input = "Place the cup for drinking on the red plate."
    # text_input = "Pick up the red cup for drinking."
    text_input = "Pull the white empty cooking pan to the center of the table."
    # text_input = "Put the red cup underneath the coffee machine."
    text_input = np.expand_dims(get_bert_embedding(text_input).detach().numpy(), axis=0)

    # # Bounding box prompts
    # prompts = 'cup for drinking.red empty plate' # red cup
    # prompts = 'black pan'
    prompts = "black and red button"
    # prompts = "empty white cooking pan"
    # prompts = "cup for drinking"
    # prompts = "cup for drinking.red flat plate"
    prompts = prompts.split('.')

    encoder = 'img'
    # task = 'stirring'
    # task = 'red_cup'
    task = 'push_button'
    # task = 'user_study'

    position = "middle"
    
    # CIVIL
    # train_cfg_path = "training_outputs/CIVIL/push_button/civil_button_21-51-03" # model_3
    # train_cfg_path = "training_outputs/CIVIL/push_button/14080_11-55-44" # model 1
    # train_cfg_path = "training_outputs/CIVIL/push_button/14080_13-27-17" # model 2

    # train_cfg_path = "training_outputs/CIVIL/pick_red_cup/3017653_15-13-17"
    # train_cfg_path = "training_outputs/CIVIL/pick_red_cup/3020718_01-09-32"
    # train_cfg_path = "training_outputs/CIVIL/pick_red_cup/civil_pick_14-26-58"

    # train_cfg_path = "training_outputs/CIVIL/bowl_pulling/3006726_21-53-25"
    # train_cfg_path = "training_outputs/CIVIL/bowl_pulling/3006726_14-33-50"
    # train_cfg_path = "training_outputs/CIVIL/bowl_pulling/3006726_19-05-25"

    # train_cfg_path = "training_outputs/CIVIL/stirring_scooping/16-07-46"  # CIVIL Scooping
    # train_cfg_path = "training_outputs/CIVIL/stirring_scooping/16410_14-16-56"
    # train_cfg_path = "training_outputs/CIVIL/stirring_scooping/3027780_10-52-44"

    # train_cfg_path = "training_outputs/CIVIL/user_study/3033914_16-52-50" 
    # train_cfg_path = "training_outputs/CIVIL/user_study/3035781_19-06-46"
    # train_cfg_path = "training_outputs/CIVIL/user_study/3035785_19-41-07"


    # Object oriented
    # train_cfg_path = "training_outputs/object_oriented/push_button/object_push_11-52-44"
    train_cfg_path = "training_outputs/object_oriented/push_button/object_push_12-51-31"
    # train_cfg_path = "training_outputs/object_oriented/push_button/object_push_paper_14-50-24"

    # train_cfg_path = "training_outputs/object_oriented/bowl/object_bowl_13-40-25"
    # train_cfg_path = "training_outputs/object_oriented/bowl/object_bowl_16-16-30"
    # train_cfg_path = "training_outputs/object_oriented/bowl/15990_16-39-02"

    # train_cfg_path = "training_outputs/object_oriented/pick_cup/16185_11-46-00"
    # train_cfg_path = "training_outputs/object_oriented/pick_cup/16186_11-45-58"
    # train_cfg_path = "training_outputs/object_oriented/pick_cup/16463_09-19-41"
    # train_cfg_path = "training_outputs/object_oriented/pick_cup/16465_09-19-16"
    
    # train_cfg_path = "training_outputs/object_oriented/stirring_scooping/16188_11-54-31"
    # train_cfg_path = "training_outputs/object_oriented/stirring_scooping/16189_11-54-36"
    # train_cfg_path = "training_outputs/object_oriented/stirring_scooping/object_stirring_10-38-38"

    # language
    # train_cfg_path = "training_outputs/film_conditioned/push_button/2996675_11-16-18"
    # train_cfg_path = "training_outputs/film_conditioned/push_button/14080_10-35-01"
    # train_cfg_path = "training_outputs/film_conditioned/push_button/14080_10-53-35"

    # train_cfg_path = "training_outputs/film_conditioned/stirring_scooping/3017612_14-56-36"
    # train_cfg_path = "training_outputs/film_conditioned/stirring_scooping/3017613_15-08-55"
    # train_cfg_path = "training_outputs/film_conditioned/stirring_scooping/3017434_13-07-26"

    # train_cfg_path = "training_outputs/film_conditioned/bowl_pulling/3006726_12-14-53"
    # train_cfg_path = "training_outputs/film_conditioned/bowl_pulling/3006726_13-23-47"
    # train_cfg_path = "training_outputs/film_conditioned/bowl_pulling/3006726_10-45-01"

    # train_cfg_path = "training_outputs/film_conditioned/pick_red_cup/16410_15-49-30"
    # train_cfg_path = "training_outputs/film_conditioned/pick_red_cup/3024098_19-13-41"
    # train_cfg_path = "training_outputs/film_conditioned/pick_red_cup/3024098_19-46-25"

    # bc 
    # train_cfg_path = "training_outputs/CIVIL/user_study/3034517_01-37-17"
    # train_cfg_path = "training_outputs/CIVIL/user_study/3034518_01-37-22"
    # train_cfg_path = "training_outputs/CIVIL/user_study/3034519_01-37-17"


    #################################### Grounding Dino ###########################################
    # For Task-VIOLA only
    parser = ArgumentParser()
    grounding_dino_args(parser)

    args = parser.parse_args()
    gd_cfg = vars(args)

    cfg = OmegaConf.load(f"{train_cfg_path}/.hydra/config.yaml")

    GROUNDING_DINO_CONFIG_PATH = gd_cfg['GROUNDING_DINO_CONFIG_PATH']
    GROUNDING_DINO_CHECKPOINT_PATH = gd_cfg['GROUNDING_DINO_CHECKPOINT_PATH']

    gd_model = GroundingDINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                  model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
                                  device='cuda')
    
    gd_thread = GDonThread(gd_model,
                           gd_cfg['DINO_THRESHOLD'],
                           gd_cfg['DINO_NMS_THRESHOLD'],
                           prompts,
                           cfg.img_size)
    #################################### Grounding Dino ###########################################

    static_cam = camera(cam_id=0, visualize=False, queue=gd_thread.queue)
    time.sleep(1)

    while gd_thread.bboxes is None:
        continue

    # static_cam = camera(cam_id=0, visualize=False)
    gripper_cam = camera(cam_id=2, visualize=False)
    time.sleep(1)

    resize_img = v2.Compose([v2.Resize(cfg.img_size, antialias=True)])
    img = copy.deepcopy(static_cam.frame)
    img = resize_img(torch.from_numpy(img).permute(2, 0, 1))
    img = img.permute(1, 2, 0).detach().cpu().numpy()

    with torch.no_grad():
        detections = gd_model.predict_with_classes(img,
                                                classes=prompts,
                                                box_threshold=gd_cfg['DINO_THRESHOLD'],
                                                text_threshold=gd_cfg['DINO_THRESHOLD'])
        
    img_bboxes = np.zeros((len(prompts), 4))

    for i, id in enumerate(range(len(prompts))):
        class_id_idx = np.where(detections.class_id == id, True, False)

        if np.any(class_id_idx):
            highest_confidence_idx = np.argmax(detections.confidence[class_id_idx])
            bbox = detections.xyxy[class_id_idx][highest_confidence_idx]

            img_bboxes[i, :] = bbox

    img_tensor = torch.from_numpy(cv2.cvtColor(static_cam.frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    img_tensor = resize_img(img_tensor)
    img_tensor = torchvision.utils.draw_bounding_boxes(img_tensor, torch.from_numpy(img_bboxes), colors="red")
    bbox_img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()

    cv2.imshow('static', cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    os.environ['HYDRA_FULL_ERROR'] = '1'

    model = hydra.utils.instantiate(cfg.model)
    # CIVIL
    model.load_state_dict(torch.load(f"{train_cfg_path}/calvin_models/civil_020_best.pt", weights_only=True, map_location=cfg.params.device), strict=True) 
    
    # FiLM
    # model.load_state_dict(torch.load(f"{train_cfg_path}/calvin_models/film_conditioned_020_mask_best.pt", weights_only=True, map_location=cfg.params.device), strict=True)
    
    # Object Oriented
    # model.load_state_dict(torch.load(f"{train_cfg_path}/calvin_models/object_oriented_ego_model_010_best.pt", weights_only=True, map_location=cfg.params.device), strict=True)

    # bc 
    # model.load_state_dict(torch.load(f"{train_cfg_path}/calvin_models/simple_bc_015_mask_best.pt", weights_only=True, map_location=cfg.params.device), strict=True)

    save_dir = f"{train_cfg_path}/rollouts"
    os.makedirs(save_dir, exist_ok=True)
    model.eval().to(cfg.params.device)

    obs_buffer = ModelObservationBuffer(cfg,
                                        cfg.params.seq_len,
                                        cfg.dataset.target_beacons,
                                        cfg.dataset.target_masks)
    recorder = VideoRecorder('test_rollouts')

    run = True
    # robot_control = True
    robot_control = False
    trajectory = []
    while run:

        z, (a_button, b_button, x_button, y_button, back_button), start_button = interface.getInput()

        if start_button:
            run = False
            print("!!!")

        if a_button:
            robot_control = True
            time.sleep(0.1)
        if b_button and robot_control:
            robot_control = False
            time.sleep(0.1)
        if back_button:
            robot.go2position(conn, START)
            state = robot.readState(conn)
            obs_buffer.reset_buffer()


        # Update sequence
        state = robot.readState(conn)
        obs = {
                "image_rgb": cv2.cvtColor(static_cam.frame, cv2.COLOR_BGR2RGB),
               "image_ego": cv2.cvtColor(gripper_cam.frame, cv2.COLOR_BGR2RGB),
               "joint_states": state["q"],
               "ee_states": state["x"],
               "gripper_states": gripper_state,
               "language": text_input,
               "image_bbox": np.array(gd_thread.bboxes)
            #    "image_bbox": img_bboxes
               }
        obs_buffer.apppend_obs(obs)

        if robot_control:
            print("saving state")
            trajectory.append(state["q"])

        # Get model action
        step_seq = obs_buffer.get_sequence()
        step_seq = TensorUtils.to_device(step_seq, cfg.params.device)
        _, action = model.get_action(step_seq, encoder=encoder)

        if robot_control:
            if action[-1] <= 0.0:
                gripper_action = "c"
                gripper_state = [-1]
            else:
                gripper_action = "o"
                gripper_state = [1]


            if cfg.action_type == "joint_vel":
                qdot = action[:7] * 1.
                qdot = robot.constraint2workspace(qdot, state, action_type="qdot", task = task)
            elif cfg.action_type == "ee_vel":
                xdot = action[:6] * 3
                xdot = robot.constraint2workspace(xdot, state, action_type="xdot", task = task)
                qdot = robot.xdot2qdot(xdot, state)

            if task != 'stirring':
                robot.send2gripper(conn_gripper, gripper_action)

            if prev_gripper_state != gripper_state[0]:
                print("actuating gripper")
                time.sleep(1)
                prev_gripper_state = gripper_state[0]

            robot.send2robot(conn, qdot, control_mode)
        else:
            z = interface.getAction(z)
            # Convert joystick input to joint velocity
            state = robot.readState(conn)
            xdot = list(z)
            xdot = robot.constraint2workspace(xdot, state, action_type="xdot", task=task)
            qdot = robot.xdot2qdot(xdot, state)


            if x_button:
                print("[*] Closing gripper...")
                robot.send2gripper(conn_gripper, "c")
                gripper_action = -1
                gripper_state = [-1]
            if y_button:
                print("[*] Opening gripper...")
                robot.send2gripper(conn_gripper, "o")
                gripper_action = 1
                gripper_state = [1]

            robot.send2robot(conn, qdot, control_mode)


        cv2.imshow("img_gripper", gripper_cam.frame)
        cv2.imshow("img", static_cam.frame)

        # print(gd_thread.detections)
        # labels = [prompts[idx] for idx in gd_thread.detections.class_id]
        # img_tensor = torch.from_numpy(cv2.cvtColor(static_cam.frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        # img_tensor = resize_img(img_tensor)
        # bboxes = gd_thread.bboxes
        # img_tensor = torchvision.utils.draw_bounding_boxes(img_tensor, torch.from_numpy(bboxes), colors="red")
        # bbox_img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()

        # cv2.imshow('static', cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        recorder.record([static_cam.frame, gripper_cam.frame])

    # Kill camera threads
    static_cam.done = True
    gripper_cam.done = True
    static_cam.frame_thread.join()
    gripper_cam.frame_thread.join()

    # Return robot home
    robot.go2position(conn, START)
    robot.send2gripper(conn_gripper, "c")

    # save results
    # recorder.save()
    num_demo = len(os.listdir(save_dir))
    save_data = {"position": position, "trajectory": trajectory}
    save_path = f"{save_dir}/trajectory_{num_demo}.pkl"
    print(f"[*] Saving file: {save_path}")
    with open(save_path, "wb") as file:
        pickle.dump(save_data, file)


if __name__ == '__main__':
    main()