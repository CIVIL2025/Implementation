from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import os
import yaml
import hydra
import torch
import wandb
import pickle
import datetime
import hydra.utils
import collections, functools, operator

from tqdm import tqdm
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf
from preprocess_data_calvin import preprocess_data
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Subset


now = datetime.datetime.now()
curr_time = now.strftime("%Y-%m-%d_%H:%M:%S")

def dataset_loader(file_path, obs_specs, seq_len):
    all_obs_keys = [key for _, obs_list in obs_specs['obs'].items() for key in obs_list]
    dataset = SequenceDataset(
                hdf5_path=file_path,
                obs_keys=all_obs_keys,
                dataset_keys=["actions"],
                load_next_obs=False,
                frame_stack=1,
                seq_length=seq_len,                  # length temporal sequences
                pad_frame_stack=True,
                pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
                get_pad_mask=False,
                goal_mode=None,
                hdf5_cache_mode='all',          # cache dataset in memory to avoid repeated file i/o
                hdf5_use_swmr=False,
                hdf5_normalize_obs=None,
                filter_by_attribute=None,       # can optionally provide a filter key here
            )
    
    return dataset

def train(cfg):

    dataset = dataset_loader(f"{cfg.output_dir}/{cfg.dataset.dataset_name}", cfg.model.obs_specs[0], cfg.params.seq_len)

    train_size = int(0.9 * len(dataset))  
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.params.batch_size, shuffle=False)

    if cfg.params.use_play:
        play_dataset = dataset_loader(f"{cfg.output_dir}/{cfg.dataset.playset_name}", cfg.model.obs_specs[1], cfg.params.seq_len)
        subset_indices = list(range(len(train_dataset)))  
        sub_play_dataset = Subset(play_dataset, subset_indices)
        play_loader = DataLoader(sub_play_dataset, batch_size=cfg.params.batch_size, shuffle=True)

    model = hydra.utils.instantiate(cfg.model.model)
    model.to(cfg.params.device)
    if 'encoder_pretrained' in cfg.model.model.keys():
        if not cfg.model.model.encoder_pretrained:
            pretrain_train_loader = DataLoader(train_dataset, batch_size=cfg.model.pretrain_batch_size, shuffle=True)
            if cfg.params.use_play:
                pretrain_play_loader = DataLoader(sub_play_dataset, batch_size=cfg.model.pretrain_batch_size, shuffle=True)
            else:
                pretrain_play_loader = []
                sub_play_dataset = []
            model.pretrain_encoder(cfg.params.device, pretrain_train_loader,
                             pretrain_play_loader,
                             len(train_dataset),
                             len(sub_play_dataset))

    model.get_optimizer(cfg.params.lr)

    losses = []
    best_loss = float('inf')
    best_train_loss = float('inf')
    save_path = f"{cfg.output_dir}/calvin_models"
    os.makedirs(save_path, exist_ok=True)

    wandb.init(project = model.name)
    for epoch in tqdm(range(cfg.params.epoch)):
        play_data = None
        epoch_loss = []
        model.train()

        if cfg.params.use_play:
            play_iterator = iter(play_loader)

        for (idx, data) in enumerate(train_loader):
            data = TensorUtils.to_device(data, cfg.params.device)

            if cfg.params.use_play:
                try:
                    play_data = next(play_iterator)
                except StopIteration:
                    play_iterator = iter(play_loader)
                    play_data = next(play_iterator)

                play_data = TensorUtils.to_device(play_data, cfg.params.device)

            iter_loss = model.calculate_loss(data, play_data)

            model.step_optimizer(iter_loss)
            iter_loss["train_loss"] = iter_loss["train_loss"].item()
            epoch_loss.append(iter_loss)

        epoch_loss = dict(functools.reduce(operator.add, map(collections.Counter, epoch_loss)))
        for loss_type in epoch_loss.keys():
            epoch_loss[loss_type] /= len(train_loader)
        
        if epoch_loss["train_loss"] < best_train_loss:
            best_train_loss = epoch_loss["train_loss"]
            torch.save(model.state_dict(), f"{save_path}/{model.name}_{str(cfg.dataset.num_demos).zfill(3)}_trainbest.pt")

        if (epoch + 1) % 5 == 0:
            print(epoch_loss)
        losses.append(epoch_loss)
        wandb.log(epoch_loss)
        model.step_scheduler()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                eval_loss = []

                for (idx, data) in enumerate(test_loader):

                    data = TensorUtils.to_device(data, cfg.params.device)
                    iter_loss = model.calculate_loss(data)
                    eval_loss.append(iter_loss)

            eval_loss = dict(functools.reduce(operator.add, map(collections.Counter, eval_loss)))
            for loss_type in eval_loss.keys():
                eval_loss[loss_type] /= len(test_loader)
            print(eval_loss)
            wandb.log(eval_loss)
            if eval_loss["test_loss"].item() < best_loss:
                best_loss = eval_loss["test_loss"].item()
                torch.save(model.state_dict(), f"{save_path}/{model.name}_{str(cfg.dataset.num_demos).zfill(3)}_best.pt")

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"{save_path}/{model.name}_{str(cfg.dataset.num_demos).zfill(3)}_{str(epoch).zfill(4)}.pt")
        
    with open(f"{cfg.output_dir}/loss.pkl", "wb") as f:
        pickle.dump(losses, f)

@hydra.main(version_base=None, config_path='conf', config_name = 'train_calvin')
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    yaml_config = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    ObsUtils.initialize_obs_utils_with_obs_specs(cfg.model.obs_specs[0])

    preprocess_data(cfg.dataset.dataset_dir, cfg.dataset.dataset_name, cfg)

    if cfg.params.use_play:
        preprocess_data(cfg.dataset.playset_dir,cfg.dataset.playset_name, cfg)

    train(cfg)

if __name__ == '__main__':
    main()




        