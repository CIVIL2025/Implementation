from torchvision.transforms import Compose, Resize, InterpolationMode, v2, ToTensor
from collections import deque
from preprocess_data_calvin import BEACON_MAPPING, MASK_MAPPING
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from calvin_agent.utils.utils import add_text

import os
import cv2
import time
import copy
import tqdm
import wave
import torch 
import socket
import pygame
import whisper
import pyaudio
import argparse
import datetime
import torchvision
import numpy as np
from queue import Queue
from copy import deepcopy
from threading import Thread
from termcolor import colored
from dataclasses import dataclass
from torch.utils.data import Dataset, ConcatDataset, DataLoader 

@dataclass
class ResultArgs:
    img: np.ndarray

STEP_SIZE_L = 0.1
STEP_SIZE_A = 0.15 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
FORMAT = pyaudio.paInt16


class ModelObservationBuffer():
    def __init__(self, model_config, seq_len, target_beacons, target_masks):
        
        self.target_beacons = target_beacons
        self.target_masks = target_masks
        self.separate_channel_mask = model_config['separate_channel_mask']
        self.seq_len = seq_len

        obs_specs = model_config['obs_specs'][0]['obs']

        self.rgb_keys = []
        self.state_keys = []
        self.mask_keys = []

        if 'rgb' in obs_specs.keys():
            self.rgb_keys = obs_specs['rgb']

            self.img_size = {}
            self.resize_img = {}

            for key in self.rgb_keys:
                if 'ego' in key and hasattr(model_config, "img_ego_size"):
                    self.img_size[key] = model_config.img_ego_size
                    self.resize_img[key] = v2.Compose([v2.Resize(model_config.img_ego_size, antialias=True)])
                else:
                    self.img_size[key] = model_config.img_size
                    self.resize_img[key] = v2.Compose([v2.Resize(model_config.img_size, antialias=True)])

                    
        for key in obs_specs['low_dim']:
            if 'mask' in key or 'bool' in key or 'bbox' in key:
                self.mask_keys.append(key)

            else:
                self.state_keys.append(key)

        if len(self.mask_keys) != 0:
            self.mask_size = {}
            self.resize_mask = {}

            for key in self.mask_keys:
                if 'ego' in key and hasattr(model_config, "mask_ego_size"):
                    self.mask_size[key] = model_config.mask_ego_size
                    self.resize_mask[key] = v2.Compose([v2.Resize(model_config.mask_ego_size,
                                                                 interpolation=InterpolationMode.NEAREST)])
                else:
                    self.mask_size[key] = model_config.mask_size
                    self.resize_mask[key] = v2.Compose([v2.Resize(model_config.mask_size,
                                                                 interpolation=InterpolationMode.NEAREST)])
                    

        self.obs_keys = [key for key_list in obs_specs.values() for key in key_list]
        self.obs_buffer = {obs_key:deque(maxlen=seq_len) for obs_key in self.obs_keys}

        self.img_mapping = {'image_rgb': 'rgb_static',
                            'image_ego': 'rgb_gripper'}
        
        self.mask_obs_mapping = {'image_mask': 'segmen_static',
                                 'image_bbox': 'segmen_static',
                                 'image_bool': 'segmen_static',
                                 'image_ego_mask': 'segmen_gripper',
                                 'image_ego_bbox': 'segmen_gripper',
                                 'ego_bool': 'segmen_gripper'}
        
        # NOTE: we have used the binary state of the gripper as the state, the actual gripper width is saved under idx 6
        self.robot_obs_mapping = {'ee_states': slice(0, 6),
                                  'joint_states': slice(7, 14),
                                  'gripper_states': slice(14,None)} 
                                                        
    def apppend_obs(self, obs):

        for key in self.rgb_keys:
            self.obs_buffer[key].append(obs['rgb_obs'][self.img_mapping[key]])

        for key in self.state_keys:
            if key == 'beacons':
                # self.obs_buffer[key].append(obs['scene_obs'])
                self.obs_buffer[key].append(obs['scene_beacon'])
            else:
                self.obs_buffer[key].append(obs['robot_obs'][self.robot_obs_mapping[key]])

        for key in self.mask_keys:
            self.obs_buffer[key].append(obs['segmen'][self.mask_obs_mapping[key]])

    def reset_buffer(self, obs):
        for key in self.obs_keys:
            self.obs_buffer[key].clear()

        self.apppend_obs(obs)

        for key in self.obs_keys:
            self.obs_buffer[key].extend(self.obs_buffer[key]*self.seq_len)

    def get_sequence(self):

        obs_sequence = {}

        for key in self.rgb_keys:
            image_seq = self.process_image_rgb(np.asarray(self.obs_buffer[key]),
                                               key)
            obs_sequence[key] = image_seq.unsqueeze(0)
                
        for key in self.state_keys:
            if key == 'beacons':
                state_seq = [np.asarray(self.obs_buffer[key])[:, BEACON_MAPPING[target]] for target in self.target_beacons]
                state_seq = torch.from_numpy(np.concatenate(state_seq, axis=1)).unsqueeze(0).float()
            else:    
                state_seq = torch.from_numpy(np.asarray(self.obs_buffer[key])).unsqueeze(0).float()

            obs_sequence[key] = state_seq

        for key in self.mask_keys:
            if 'mask' in key:
                mask_seq = self.process_binary_mask(np.asarray(self.obs_buffer[key]),
                                                    self.target_masks,
                                                    key,
                                                    self.separate_channel_mask)
                obs_sequence[key] = mask_seq.unsqueeze(0)
            elif 'bbox' in key:
                # pdb.set_trace()
                mask_seq = self.process_binary_mask(np.asarray(self.obs_buffer[key]),
                                                    self.target_masks,
                                                    key,
                                                    self.separate_channel_mask)
                seq, objects, height, width = mask_seq.shape
                mask_seq = mask_seq.view(seq*objects, height, width)
                bboxes = torch.zeros(seq*objects, 4)
                available_bboxes = torch.any(mask_seq, dim=[1, 2])
                bboxes[available_bboxes] = torchvision.ops.masks_to_boxes(mask_seq[available_bboxes])
                obs_sequence[key] = bboxes.view(seq, objects, 4).unsqueeze(0)

            elif 'bool' in key:           
                pixel_count = self.mask2pixelcount(np.asarray(self.obs_buffer[key]),
                                                    self.target_masks,
                                                    key)
                obs_sequence[key] = pixel_count.unsqueeze(0)

        return obs_sequence
    
    def process_image_rgb(self, images, target_img):
        image_seq = torch.from_numpy(images).permute(0, 3, 1, 2)
        if self.img_size[target_img] != image_seq.shape[-2]:
            image_seq = self.resize_img[target_img](image_seq)
        
        return image_seq.float() / 255.

    def process_binary_mask(self, segmentation, target_masks, mask_key, separate_channel_mask=False):

        mask = []
        # pdb.set_trace()
        for target in target_masks:
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

        mask = torch.from_numpy(mask.astype(np.float32))

        if self.mask_size[mask_key] != mask.shape[-2:]:
            mask = self.resize_mask[mask_key](mask)

        return mask

    def mask2pixelcount(self, segmentation, target_masks, mask_key):
        mask = self.process_binary_mask(segmentation, target_masks, mask_key, separate_channel_mask=True)
        mask = mask.permute(1, 0, 2, 3)
        pixel_count = torch.count_nonzero(mask, dim=(2,3))

        return pixel_count.permute(1, 0)


class VideoRecorder():
    def __init__(self, 
                save_fd, 
                save_mode = None
    ):
        self.roll_out_frames = [] 
        self.save_mode = save_mode
        self.save_fd = save_fd
        print(colored('[Recorder] ', 'green') + f'Video recorder mode is {self.save_mode}')

    def record(self, env, obs, test_data, lang_annotation, target_beacon):
        if self.save_mode == "none":
            pass
        else:
            img = env.render(mode="rgb_array")
            rollout_img = deepcopy(img)

            rollout_img = self._write_text(rollout_img, obs, test_data["b_hat"], target_beacon)
            if test_data["mask"] is not None: 
                rollout_img = self._add_segmen(rollout_img, test_data["mask"])
            rollout_img = np.ascontiguousarray(rollout_img, dtype=np.uint8)
            # add_text(rollout_img, lang_annotation)
            self.roll_out_frames.append(cv2.cvtColor(np.ascontiguousarray(rollout_img, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    def _write_text(self, img, obs, b_hat, target_beacon):
        if len(target_beacon) > 0:
            beacon = []
            for b in target_beacon:
                beacon.extend([round(num,4) for num in obs['scene_beacon'][BEACON_MAPPING[b]]])
            b_info = f"b: {beacon}"
            img = np.ascontiguousarray(img, dtype=np.uint8)
            cv2.putText(img, text=b_info, org=(1, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale= (0.7 / 500) * 150,
                    color=(0, 0, 0),
                    thickness= 1,
                    lineType=cv2.LINE_AA,)

            if b_hat is not None: 
                if b_hat.shape[-1] == 6:
                    b_hat = b_hat[0]
                b_error = beacon - b_hat[-1].cpu().numpy()

                b_info = f"b: {beacon}"
                b_hat_info = f"b_hat: {b_hat}"
                b_error_info = f"b_error: {b_error}"
                
                cv2.putText(img, text=b_hat_info, org=(1, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= (0.7 / 500) * 150,
                            color=(0, 0, 0),
                            thickness= 1,
                            lineType=cv2.LINE_AA,)
                
                cv2.putText(img, text=b_error_info, org=(1, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= (0.7 / 500) * 150,
                            color=(0, 0, 0),
                            thickness= 1,
                            lineType=cv2.LINE_AA,)
            return img
        return img
    
    def _add_segmen(self, img, img_segmen):
        img_segmen = img_segmen.squeeze(0)
        transform = Compose([ToTensor()])
        if len(img_segmen.shape) == 3: # is bbox
            img = torchvision.utils.draw_bounding_boxes(transform(img), img_segmen[-1])
        elif len(img_segmen.shape) == 4: # is mask
            segmen = img_segmen[-1]
            image_mask = torch.repeat_interleave(segmen, 3, dim=0)
            img = transform(img)
            try:
                assert image_mask.shape == img.shape
            except AssertionError:
                n_px = img.shape[1]
                adjust_size = Compose([Resize(n_px, interpolation=InterpolationMode.BICUBIC)])
                image_mask = adjust_size(image_mask).cpu()
            img = img * image_mask
        return np.moveaxis(img.numpy()*255, 0, -1)
        

    def save(self, model_name, iter_num, success = False):
        if self.save_mode == 'all' or (self.save_mode == 'success' and success):
            save_path = f"{self.save_fd}"
            os.makedirs(save_path, exist_ok = True)
            save_path = f"{save_path}/{model_name}_{str(iter_num).zfill(3)}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (200, 200)
            vid_w = cv2.VideoWriter(save_path, fourcc, fps, frame_size)

            for frame in self.roll_out_frames:
                vid_w.write(frame)
            vid_w.release()
            print(colored('[video]', 'cyan') + f'save video at {save_path}')
        elif self.save_mode == 'none':
            pass

        self.roll_out_frames = []


class AttentionRecorder():
    def __init__(self,
                 save_fd, 
                 save_mode, 
                 seq_len,
                 n_layers = 2, 
                 token_labels=['state', 'beacon', 'ego', 'action_token']):
        
        self.save_fd = save_fd
        self.save_mode = save_mode
        self.seq_len = seq_len
        self.token_labels = token_labels
        self.rollout_attention_weights = []

        self.n_cols = np.ceil(np.sqrt(n_layers)).astype(int)
        self.n_rows = np.ceil(self.n_cols / n_layers).astype(int)

    def visualize(self, n_rollout, step, attn_weights):

        if step == self.seq_len - 1:
            self.vis_fig, self.vis_axes = plt.subplots(1, 2, figsize=(15, 15))
            plt.ion()
            self.im_list, self.cbar_list = self._initialize_figure(attn_weights, self.vis_axes, n_rollout)
        elif step >= self.seq_len:
            self._update_vis(n_rollout, step, attn_weights)

    def close_vis(self):
        plt.ioff()
        plt.close(self.vis_fig)

    def record(self, step, test_data):
        if self.save_mode == 'none':
            pass
        else:
            if test_data["attn_weights"] is not None:
                self.rollout_attention_weights.append([step] + test_data["attn_weights"])


    def save(self, filename, n_rollout, success=False):
    
        if self.save_mode == 'all' or (self.save_mode == 'success' and success):
            if len(self.rollout_attention_weights) != 0:
                fig, axes = plt.subplots(self.n_rows, self.n_cols, figsize=(15, 15))
                rollout_attention_weights = self.rollout_attention_weights
                im_list, cbar_list = self._initialize_figure(rollout_attention_weights[0][1:], axes, n_rollout)

                ani = animation.FuncAnimation(fig=fig,
                                            func=partial(self._update_animation,
                                                        im_list=im_list,
                                                        cbar_list=cbar_list,
                                                        n_rollout=n_rollout),
                                            frames=rollout_attention_weights,
                                            interval=20)
                ani_writer = animation.FFMpegWriter(fps=10)

                os.makedirs(self.save_fd, exist_ok=True)
                save_path = f"{self.save_fd}/{filename}.mp4"
                ani.save(save_path, writer=ani_writer)

                plt.close(fig)
            
        elif self.save_mode == 'none':
            pass

        self.rollout_attention_weights = []

    def _initialize_figure(self, attn_weights, axes, n_rollout):

        im_list = []
        cbar_list = []
        
        for i, (ax, attn_weight) in enumerate(zip(axes, attn_weights)):
            im = ax.imshow(attn_weight)

            labels = []
            for j in range(self.seq_len):
                step_labels = [f"{token}_T-{self.seq_len-1-j}" for token in self.token_labels]
                labels.extend(step_labels)

            ax.set_yticks(range(len(labels)), labels=labels)
            ax.set_xticks(range(len(labels)), labels=labels, rotation=90, ha="right", rotation_mode="anchor")
            ax.set_title(f"Attention Matrix layer {i+1}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
            cbar.ax.set_ylabel("Attention Weights", rotation=-90, va="bottom")

            im_list.append(im)
            cbar_list.append(cbar)

        plt.tight_layout()
        plt.suptitle(f"Rollout: {n_rollout}, Step: 0")

        return im_list, cbar_list        


    def _update_vis(self, n_rollout, step, attn_weights):

        for i, attn_weight in enumerate(attn_weights):
            self.im_list[i].set_data(attn_weight)
            self.cbar_list[i].update_normal(self.im_list[i])

        plt.draw()
        plt.suptitle(f"Rollout: {n_rollout}, Step: {step}")
        plt.pause(0.01)

    @staticmethod
    def _update_animation(frame, im_list, cbar_list, n_rollout):

        step = frame[0]
        attn_weights = frame[1:]

        for i, attn_weight in enumerate(attn_weights):
            im_list[i].set_data(attn_weight)
            cbar_list[i].update_normal(im_list[i])

        plt.suptitle(f"Rollout: {n_rollout}, Step: {step}")

##################################################### Real Robot Utils #####################################################

class GDonThread():
    def __init__(self, gd_model, dino_threshold, dino_nms_threshold, prompts, img_size):

        self.gd_model = gd_model
        self.dino_threshold = dino_threshold
        self.dino_nms_theshold = dino_nms_threshold
        self.prompts = prompts
        self.resize_img = v2.Compose([v2.Resize(img_size, antialias=True)])
        self.bboxes = None
        self.detections = None
    
        self.queue = Queue(maxsize=1)
        self.thread = Thread(target=self.save_result, args=(self.queue,))
        self.thread.daemon = True
        self.thread.start()

    def save_result(self, queue:Queue):
        while True:
            try:
                args: ResultArgs = queue.get(block=False)
            except:
                continue

            if args is None:
                queue.task_done()
                break

            img = args.img
            img = self.resize_img(torch.from_numpy(img).permute(2, 0, 1))
            img = img.permute(1, 2, 0).detach().cpu().numpy()

            with torch.no_grad():
                detections = self.gd_model.predict_with_classes(img,
                                                                classes=self.prompts,
                                                                box_threshold=self.dino_threshold,
                                                                text_threshold=self.dino_threshold)
            img_bboxes = np.zeros((len(self.prompts), 4))

            for i, id in enumerate(range(len(self.prompts))):
                class_id_idx = np.where(detections.class_id == id, True, False)

                if np.any(class_id_idx):
                    highest_confidence_idx = np.argmax(detections.confidence[class_id_idx])
                    bbox = detections.xyxy[class_id_idx][highest_confidence_idx]
                    img_bboxes[i, :] = bbox

            self.bboxes = img_bboxes


class Joystick(object):
    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)

        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        self.X_pressed = self.gamepad.get_button(2)
        self.Y_pressed = self.gamepad.get_button(3)
        self.Back_pressed = self.gamepad.get_button(6)

        Buttons = (self.A_pressed, self.B_pressed, self.X_pressed, self.Y_pressed, self.Back_pressed)
        start = self.gamepad.get_button(7)
        
        z1 = self.gamepad.get_axis(0) # Left stick (left-right) : (-1 to 1)
        z2 = self.gamepad.get_axis(1) # Left stick (up-down) : (-1 to 1)
        z3 = self.gamepad.get_axis(4) # Right stick (up-down) : (-1 to 1)
        z = [z1, z2, z3]

        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return tuple(z), Buttons, start
    
    def getAction(self, z):
        if self.toggle:
            action = (0, 0, 0, STEP_SIZE_A * z[0], STEP_SIZE_A * z[1], -STEP_SIZE_A * z[2])
        else:
            action = (STEP_SIZE_L * z[0], -STEP_SIZE_L * z[1],
            -STEP_SIZE_L * z[2], 0, 0, 0)

        return action

class camera():
    def __init__(self, cam_id, aruco_detector=None, roi=None, visualize=False, queue=None):

        self.cap = cv2.VideoCapture(cam_id)
        time.sleep(0.5)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.detector = aruco_detector

        self.done = False
        self.frame = None
        self.rvecs = None
        self.tvecs = None
        self.corners = None
        self.ids = None
        self.roi = roi
        self.cam_id = cam_id
        self.visualize = visualize
        self.queue = queue

        self.frame_thread = Thread(target=self._run)
        self.frame_thread.daemon = True
        self.frame_thread.start()

    def get_frame(self):
        _, frame = self.cap.read()
        clone = frame.copy()

        if self.roi is not None:
            clone = clone[int(self.roi[1]):int(self.roi[1] + self.roi [3]), \
                        int(self.roi[0]):int(self.roi[0] + self.roi[2])]

        return clone
    
    def _run(self):
        if self.detector is not None:
            display = pygame.display.set_mode((640, 480))

        while not self.done:
            self.frame = self.get_frame()

            if self.queue is not None:
                self.queue.put(ResultArgs(copy.deepcopy(self.frame)))

            if self.visualize:
                cv2.imshow(str(self.cam_id), self.frame)
                cv2.waitKey(1)
            
            if self.detector is not None:
                rvecs, tvecs, corners, ids, frame_markers = self.detector.plot_aruco(self.frame)
                self.rvecs = rvecs
                self.tvecs = tvecs
                self.corners = corners
                self.ids = ids
                
                image = np.rot90(frame_markers)
                image = np.flipud(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                surf = pygame.surfarray.make_surface(image)
                display.blit(surf, (0,0))
                pygame.display.flip()
                self.key_pressed = None
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a:
                            self.key_pressed = 'a'
                        elif event.key == pygame.K_s:
                            self.key_pressed = 's'
                        elif event.key == pygame.K_v:
                            self.key_pressed = 'v'
                

class FR3(object):

    def __init__(self):
        self.home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

    def connect(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('172.16.0.3', port))
        s.listen()
        conn, addr = s.accept()
        return conn

    def send2gripper(self, conn, command):
        send_msg = "s," + command + ","
        conn.send(send_msg.encode())

    def send2robot(self, conn, qdot, control_mode, limit=1.0):
        qdot = np.asarray(qdot)
        scale = np.linalg.norm(qdot)
        if scale > limit:
            qdot *= limit/scale
        # print(qdot)
        send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
        if send_msg == '0.,0.,0.,0.,0.,0.,0.':
            send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
        send_msg = "s," + send_msg + "," + control_mode + ","
        conn.send(send_msg.encode())

    def listen2robot(self, conn):
        state_length = 7 + 6 + 42
        message = str(conn.recv(2048))[2:-2]
        state_str = list(message.split(","))
        for idx in range(len(state_str)):
            if state_str[idx] == "s":
                state_str = state_str[idx+1:idx+1+state_length]
                break
        try:
            state_vector = [float(item) for item in state_str]
        except ValueError:
            return None
        if len(state_vector) is not state_length:
            return None
        state_vector = np.asarray(state_vector)
        states = {}
        states["q"] = state_vector[0:7]
        states["O_F"] = state_vector[7:13]
        states["J"] = state_vector[13:].reshape((7,6)).T

        # get cartesian pose
        xyz_lin, R = self.joint2pose(state_vector[0:7])
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
        states["x"] = np.array(xyz)
        return states

    def readState(self, conn):
        while True:
            states = self.listen2robot(conn)
            if states is not None:
                break
        return states

    def xdot2qdot(self, xdot, states):
        J_inv = np.linalg.pinv(states["J"])
        return J_inv @ np.asarray(xdot)
    
    def qdot2xdot(self, qdot, states):
        return states["J"] @ np.asarray(qdot)

    def joint2pose(self, q):
        def RotX(q):
            return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
        def RotZ(q):
            return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        def TransX(q, x, y, z):
            return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
        def TransZ(q, x, y, z):
            return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        H1 = TransZ(q[0], 0, 0, 0.333)
        H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
        H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
        H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
        H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
        H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
        H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
        H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
        T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
        R = T[:,:3][:3]
        xyz = T[:,3][:3]
        return xyz, R

    def go2position(self, conn, goal=False):
        if not goal:
            goal = self.home
        total_time = 15.0
        start_time = time.time()
        states = self.readState(conn)
        dist = np.linalg.norm(states["q"] - goal)
        elapsed_time = time.time() - start_time
        while dist > 0.05 and elapsed_time < total_time:
            qdot = np.clip(goal - states["q"], -0.1, 0.1)
            self.send2robot(conn, qdot, "v")
            states = self.readState(conn)
            dist = np.linalg.norm(states["q"] - goal)
            elapsed_time = time.time() - start_time

    def wrap_angles(self, theta):
        if theta < -np.pi:
            theta += 2*np.pi
        elif theta > np.pi:
            theta -= 2*np.pi
        else:
            theta = theta
        return theta
    
    def constraint2workspace(self, action, state, action_type='qdot', task = 'red_cup'):
        y_min, y_max = -0.48103476, 0.51927134
        x_min, x_max = 0.29575483, 0.78672391
    
        if task == 'red_cup':
            z_min, z_max = 0.06320518, 0.45028534
        elif task == 'stirring':
            z_min, z_max = 0.1720848, 0.45028534
        elif task == 'push_button':
            z_min, z_max =  0.05520486, 0.58732426
        if task == 'user_study':
            y_max = 0.28668681
            z_min, z_max = 0.04320518, 0.45028534
            
        if action_type == 'qdot':
            xdot = self.qdot2xdot(action, state)
        else:
            xdot = action

        # X
        if (state['x'][0] <= x_min and xdot[0] < 0.) or (state['x'][0] >= x_max and xdot[0] > 0.):
            print("constrain")
            xdot[0] = 0.
        # Y
        if (state['x'][1] <= y_min and xdot[1] < 0.) or (state['x'][1] >= y_max and xdot[1] > 0.):
            print("constrain")
            xdot[1] = 0.
        # Z
        if (state['x'][2] <= z_min and xdot[2] < 0.) or (state['x'][2] >= z_max and xdot[2] > 0.):
            print("constrain")
            xdot[2] = 0.
        
        if action_type == 'qdot':
            constrained_action = self.xdot2qdot(xdot, state)
        else:
            constrained_action = xdot

        return constrained_action 


class ModelObservationBufferReal():
    def __init__(self, model_config, seq_len, target_beacons, target_masks):
        
        self.target_beacons = target_beacons
        self.target_masks = target_masks
        self.separate_channel_mask = model_config['separate_channel_mask']
        self.seq_len = seq_len

        obs_specs = model_config['obs_specs'][0]['obs']

        self.rgb_keys = []
        self.state_keys = []
        self.mask_keys = []

        if 'rgb' in obs_specs.keys():
            self.rgb_keys = obs_specs['rgb']

            self.img_size = {}
            self.resize_img = {}

            for key in self.rgb_keys:
                if 'ego' in key and hasattr(model_config, "img_ego_size"):
                    self.img_size[key] = model_config.img_ego_size
                    self.resize_img[key] = v2.Compose([v2.Resize(model_config.img_ego_size, antialias=True)])
                else:
                    self.img_size[key] = model_config.img_size
                    self.resize_img[key] = v2.Compose([v2.Resize(model_config.img_size, antialias=True)])

                    
        for key in obs_specs['low_dim']:
            self.state_keys.append(key)
            

        if len(self.mask_keys) != 0:
            self.mask_size = {}
            self.resize_mask = {}

            for key in self.mask_keys:
                if 'ego' in key and hasattr(model_config, "mask_ego_size"):
                    self.mask_size[key] = model_config.mask_ego_size
                    self.resize_mask[key] = v2.Compose([v2.Resize(model_config.mask_ego_size,
                                                                 interpolation=InterpolationMode.BILINEAR)])
                else:
                    self.mask_size[key] = model_config.mask_size
                    self.resize_mask[key] = v2.Compose([v2.Resize(model_config.mask_size,
                                                                 interpolation=InterpolationMode.BILINEAR)])
                    

        self.obs_keys = [key for key_list in obs_specs.values() for key in key_list]
        self.obs_buffer = {obs_key:deque(maxlen=seq_len) for obs_key in self.obs_keys}

                                                        
    def apppend_obs(self, obs):
        for key, value in obs.items():
            if key in self.rgb_keys:
                img = self.process_image_rgb(np.asarray(value), key)
                self.obs_buffer[key].append(img)

            elif key in self.state_keys:
                state = torch.from_numpy(np.asarray(value)).float()
                self.obs_buffer[key].append(state)

            elif key in self.mask_keys:
                mask = torch.from_numpy(np.asarray(value)).float()
                self.obs_buffer[key].append(mask)
        
    def reset_buffer(self):
        for key in self.obs_keys:
            self.obs_buffer[key].clear()

    def get_sequence(self):
        
        obs_sequence = {}

        for key in self.obs_buffer.keys():
            if len(self.obs_buffer[key]) != 0:
                key_seq = torch.stack(list(self.obs_buffer[key])).unsqueeze(0)
                obs_sequence[key] = key_seq
            

        return obs_sequence
    
    def process_image_rgb(self, images, target_img):
        # image_seq = torch.from_numpy(images).permute(0, 3, 1, 2)
        image_seq = torch.from_numpy(images).permute(2, 0, 1)
        if self.img_size[target_img] != image_seq.shape[-2]:
            image_seq = self.resize_img[target_img](image_seq)
        
        return image_seq.float() / 255.

    def process_binary_mask(self, segmentation, target_masks, mask_key, separate_channel_mask=False):

        mask = []
        # pdb.set_trace()
        for target in target_masks:
            merged = np.where(segmentation == MASK_MAPPING[target], 1, 0)
            mask.append(merged)
            
        if separate_channel_mask:
            mask = np.stack(mask, axis=1)
        else:
            mask = np.sum(mask, axis=0)
            mask = np.expand_dims(mask, axis=1)

        mask = torch.from_numpy(mask.astype(np.float32))

        if self.mask_size[mask_key] != mask.shape[-2:]:
            mask = self.resize_mask[mask_key](mask)

        return mask

    def mask2pixelcount(self, segmentation, target_masks, mask_key):
        mask = self.process_binary_mask(segmentation, target_masks, mask_key, separate_channel_mask=True)
        mask = mask.permute(1, 0, 2, 3)
        pixel_count = torch.count_nonzero(mask, dim=(2,3))

        return pixel_count.permute(1, 0)
    


class VideoRecorderReal():
    def __init__(self,
                 save_fd):
        self.roll_out_frames = {}
        self.save_fd = save_fd
    
    def record(self, obs_list):
        for i, obs in enumerate(obs_list):
            if i not in self.roll_out_frames.keys():
                self.roll_out_frames[i] = []
            self.roll_out_frames[i].append(obs)
    
    def save(self):
        print('Saving test rollouts...')
        save_dir = f'{self.save_fd}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print(name for name in os.listdir(save_dir) if os.path.isdir(name))
        test_num = len(os.listdir(save_dir))
        save_dir = f'{save_dir}/test_{str(test_num)}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_size = (640, 480)

        
        for cam_id, obs in self.roll_out_frames.items():
            save_path = f"{save_dir}/cam_{str(cam_id)}.mp4"
            vid_w = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
            print(len(obs))

            for frame in obs:
                vid_w.write(frame)
            vid_w.release()
        return
    

######################################################### audio recorder #########################################################

class VoiceRecorder(Thread):
    def __init__(
        self,
        save_dir,
        demo_num
    ):
        Thread.__init__(self)
        self.daemon = True
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if 'Razer Seiren Mini' in info["name"]:
            # if 'HD Pro Webcam C920' in info["name"]:
            # if 'HDA Intel PCH: ALC233 Analog' in info["name"]: 
                self.mic_index = info["index"]
                break

        self.save_dir = save_dir
        self.demo_num = demo_num
        
        self.rate = 16000
        # self.rate = 48000
        self.chunk = 1024
        self.keep_recording = False
        self.voice_info = []
        # self.start()

    def start_recording(self):
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.rate,
                        input=True,
                        input_device_index=self.mic_index,
                        frames_per_buffer=self.chunk)
        self.start_time = time.time()
        self.end_time = 0

    def run(self, ):
        print("\n[*] Voice recording started.")
        self.frames = []
        self.start_recording()
        while self.keep_recording:      
            for i in range(0, int(self.rate / self.chunk)):
                data = self.stream.read(self.chunk)
                self.frames.append(data)
                        
        print("[*] Voice recording stopped.")
                            
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        self.end_time = time.time()

        # Save the recording as a WAV file
        self.save_path = f'{self.save_dir}/recording_{self.start_time}_{self.end_time}_{self.demo_num}.wav'
        wave_file = wave.open(self.save_path, 'wb')
        wave_file.setnchannels(1)
        wave_file.setsampwidth(self.p.get_sample_size(FORMAT))
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        timestamps = [self.start_time, self.end_time]
        self.voice_info.append(timestamps)
        # Save the start time and ending time
        # pickle_path = f'{self.save_dir}/recording_{self.start_time}_{self.end_time}.pkl'
        # with open(pickle_path, 'ab') as file:
        #     pickle.dump(timestamps, file)
        time.sleep(1)
    
    def save_recording(self, trajectory):
        # convert voice recording to text
        model = whisper.load_model("medium")
        # model = whisper.load_model("tiny")
        result = model.transcribe(self.save_path)
        print("***************************************\n\n")
        print(result['text'])
        print("\n\n***************************************")
        trajectory['language'] = result['text']
        os.remove(self.save_path)
        return trajectory