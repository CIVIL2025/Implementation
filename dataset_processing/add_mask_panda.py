import os
import cv2
import json
import torch
import pickle
import numpy as np
from os import path
from tqdm import tqdm
from typing import Dict, List
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args#, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
# from deva.ext.with_text_processor import process_frame_with_text as process_frame
from deva.inference.object_info import ObjectInfo
from deva.inference.frame_utils import FrameInfo
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.ext.grounding_dino import segment_with_text
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    # not sure why this happens sometimes
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import SamPredictor


from copy import deepcopy
from natsort import os_sorted
from results_utils_panda import ResultSaver
from deva.model.network import DEVA


def make_segmentation_with_text(cfg: Dict, image_np: np.ndarray, gd_model: GroundingDINOModel,
                                sam_model: SamPredictor, prompts: List[str],
                                min_side: int) -> tuple[torch.Tensor, List[ObjectInfo]]:
    mask, segments_info = segment_with_text(cfg, gd_model, sam_model, image_np, prompts, min_side)
    return mask, segments_info


@torch.inference_mode()
def process_frame_with_text(deva: DEVAInferenceCore,
                            gd_model: GroundingDINOModel,
                            sam_model: SamPredictor,
                            frame_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else: 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config
    raw_prompt = cfg['prompt']
    prompts = raw_prompt.split('.')

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    # frame_name = path.basename(frame_path)
    frame_name = frame_path
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })


    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
                                                              prompts, new_min_side)
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')

                
                prob = deva.incorporate_detection(this_image, mask, new_segments_info)
                deva.next_voting_frame += cfg['detection_every']

                
                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       prompts=prompts)

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np,
                                           prompts=prompts)
                    deva.clear_buffer()

        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np,
                                   prompts=prompts)

    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
                                                              prompts, new_min_side)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                frame_name,
                                need_resize=need_resize,
                                shape=(h, w),
                                image_np=image_np,
                                prompts=prompts)

def get_model_and_config(parser: ArgumentParser):
    args = parser.parse_args()
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']    

    # Change relative paths to abs paths
    cwd = os.getcwd()
    model_root = os.path.join(cwd, '../Tracking-Anything-with-DEVA')
    for key, value in config.items():
        if 'path' in key.lower() or key == 'model':
            config[key] = os.path.abspath(os.path.join(model_root, value))

    # Load our checkpoint
    network = DEVA(config).cuda().eval()
    if config['model'] is not None:
        model_weights = torch.load(config['model'])
        network.load_weights(model_weights)
    else:
        print('No model loaded.')

    return network, config, args


if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument('--demo_pth')

    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    deva_model, cfg, args = get_model_and_config(parser)

    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')
    """
    Temporal setting
    """
    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    demo_path = cfg['demo_pth']
    demo_fd = f'/projects/recon/human-placed-markers/{demo_path}'
    demo_list = os_sorted(os.listdir(demo_fd))

    for rel_path in demo_list:
        demo_path = f"{demo_fd}/{rel_path}"

        with open(demo_path, "rb") as file:
            demo_data = pickle.load(file)


        for key in ['img', 'img_gripper']:
            loader = np.asarray(demo_data[key])

            # Start eval
            vid_length = len(loader)
            # no need to count usage for LT if the video is not that long anyway
            cfg['enable_long_term_count_usage'] = (
                cfg['enable_long_term']
                and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
                    cfg['num_prototypes']) >= cfg['max_long_term_elements'])

            # print('Configuration:', cfg)

            deva = DEVAInferenceCore(deva_model, config=cfg)
            deva.next_voting_frame = cfg['num_voting_frames'] - 1
            deva.enabled_long_id()
            
            new_rel = rel_path.replace(".pkl", "")
            out_path = f"{demo_fd}_vis/{new_rel}_{key}"
        
            result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

            with torch.cuda.amp.autocast(enabled=cfg['amp']):
                for ti, frame in enumerate(tqdm(loader)):
                    im_path = f'{key}_{ti}.png'
                    process_frame_with_text(deva, gd_model, sam_model, im_path, result_saver, ti, image_np=frame)
                flush_buffer(deva, result_saver)
            result_saver.end()

            # save this as a video-level json
            with open(path.join(out_path, 'pred.json'), 'w') as f:
                json.dump(result_saver.video_json, f, indent=4)  # prettier json

            # Add segmentation to dataset
            demo_data[f"{key}_segmen"] = result_saver.all_segmentations            

        # Save segmentation to original_file
        with open(demo_path, "wb") as file:
            pickle.dump(demo_data, file)
