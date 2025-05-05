import os
import torch
import pickle
import numpy as np
import torchvision

from tqdm import tqdm
from natsort import os_sorted
from argparse import ArgumentParser
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    # not sure why this happens sometimes
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel



def main():

    parser = ArgumentParser()

    # Grounded Segment Anything
    parser.add_argument('--GROUNDING_DINO_CONFIG_PATH',
                        default='../Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py')

    parser.add_argument('--GROUNDING_DINO_CHECKPOINT_PATH',
                        default='../Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth')

    parser.add_argument('--DINO_THRESHOLD', default=0.35, type=float)
    parser.add_argument('--DINO_NMS_THRESHOLD', default=0.8, type=float)

    parser.add_argument('--prompt', type=str, help='Separate classes with a single fullstop')

    parser.add_argument('--demo_pth')


    args = parser.parse_args()
    cfg = vars(args)

    prompts = cfg['prompt']
    prompts = prompts.split('.')

    GROUNDING_DINO_CONFIG_PATH = cfg['GROUNDING_DINO_CONFIG_PATH']
    GROUNDING_DINO_CHECKPOINT_PATH = cfg['GROUNDING_DINO_CHECKPOINT_PATH']

    BOX_THRESHOLD = TEXT_THRESHOLD = cfg['DINO_THRESHOLD']
    NMS_THRESHOLD = cfg['DINO_NMS_THRESHOLD']

    gd_model = GroundingDINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                  model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
                                  device='cuda')
    

    demo_path = cfg['demo_pth']
    demo_fd = f'/projects/recon/human-placed-markers/{demo_path}'
    demo_list = os_sorted(os.listdir(demo_fd))

    for rel_path in demo_list:
        demo_path = f"{demo_fd}/{rel_path}"

        with open(demo_path, "rb") as file:
            demo_data = pickle.load(file)

        images = np.asarray(demo_data['img'])
        steps = len(images)
        bboxes = []

        print(f"Adding bboxes to: {demo_path}")
        for step in tqdm(range(steps)):
            img = images[step]
            detections = gd_model.predict_with_classes(img,
                                                       classes=prompts,
                                                       box_threshold=BOX_THRESHOLD,
                                                       text_threshold=TEXT_THRESHOLD)
            
            # nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy),
            #                               torch.from_numpy(detections.confidence),
            #                               NMS_THRESHOLD).numpy().tolist()

            # detections.xyxy = detections.xyxy[nms_idx]
            # detections.confidence = detections.confidence[nms_idx]
            # detections.class_id = detections.class_id[nms_idx]

            img_bboxes = np.zeros((len(prompts), 4))

            # temp_bboxes = {id:{} for id in range(len(prompts))}
            # for idx, id in enumerate(detections.class_id):
            #     temp_bboxes[id].append(detections.xyxy[idx])


            for i, id in enumerate(range(len(prompts))):
                class_id_idx = np.where(detections.class_id == id, True, False)

                if np.any(class_id_idx):
                    highest_confidence_idx = np.argmax(detections.confidence[class_id_idx])
                    bbox = detections.xyxy[class_id_idx][highest_confidence_idx]

                    img_bboxes[i, :] = bbox

            bboxes.append(img_bboxes)

        bboxes = np.asarray(bboxes)
        demo_data['bboxes'] = bboxes
        demo_data['prompt'] = prompts
        
        print(f"bboxes shape: {bboxes.shape}")
        # Save bbox to original_file
        with open(demo_path, "wb") as file:
            pickle.dump(demo_data, file)
    


if __name__ == '__main__':
    main()