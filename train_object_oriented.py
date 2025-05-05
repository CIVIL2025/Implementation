import os
import yaml
import hydra
import torch
import wandb
import pickle
import hydra.utils
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf


import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.utils.dataset import SequenceDataset

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Subset

from preprocess_data_civil import preprocess_data


class NLLLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

    def forward(self, prediction, target):
        # Make sure prediction is in the form of distributions
        log_probs = prediction.log_prob(target)
        loss = -log_probs
        return loss.mean()
    

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

    dataset = dataset_loader(f"{cfg.output_dir}/{cfg.dataset.dataset_name}", cfg.obs_specs[0], cfg.params.seq_len)

    train_size = int(0.9 * len(dataset))  
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.params.batch_size, shuffle=False)

    model = hydra.utils.instantiate(cfg.model)
    model.to(cfg.params.device)

    # optimizer = torch.optim.Adam(model.paramaters(), lr=cfg.params.lr)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    optimizer = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer(params=model.parameters())

    scheduler = hydra.utils.instantiate(cfg.scheduler)
    scheduler = scheduler(optimizer=optimizer)

    losses = []
    best_loss = float('inf')
    save_path = f"{cfg.output_dir}/calvin_models"
    os.makedirs(save_path, exist_ok=True)

    wandb.init(project = model.name)

    # First training loop
    for epoch in tqdm(range(cfg.params.epoch)):
        epoch_loss = []
        model.train()

        num_iters = len(train_loader)
        for (idx, data) in enumerate(train_loader):
            data = TensorUtils.to_device(data, cfg.params.device)

            output = model(data['obs'])
            loss = loss_fn(output, data['actions'])

            optimizer.zero_grad()
            loss.backward()
            if hasattr(cfg.params, "grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.params.grad_clip)
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + idx / num_iters)
            epoch_loss.append(loss.item())

        epoch_loss = np.mean(epoch_loss)

        if (epoch + 1) % 5 == 0:
            print(epoch_loss)
        losses.append(epoch_loss)
        wandb.log({'train_loss': epoch_loss})

        if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        if (epoch) % 20 == 0:
            model.eval()
            with torch.no_grad():
                eval_loss = []
                for (idx, data) in enumerate(test_loader):

                    data = TensorUtils.to_device(data, cfg.params.device)
                    output = model(data['obs'])
                    loss = loss_fn(output, data['actions'])

                    eval_loss.append(loss.item())

            eval_loss = np.mean(eval_loss)
            print(eval_loss)
            wandb.log({'test_loss': eval_loss})
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), f"{save_path}/{model.name}_{str(cfg.dataset.num_demos).zfill(3)}_best.pt")

        if (epoch) % 100 == 0:
            torch.save(model.state_dict(), f"{save_path}/{model.name}_{str(cfg.dataset.num_demos).zfill(3)}_{str(epoch).zfill(4)}.pt")


@hydra.main(version_base=None, config_path='conf', config_name = 'train_civil')
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    yaml_config = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    ObsUtils.initialize_obs_utils_with_obs_specs(cfg.obs_specs[0])

    preprocess_data(cfg.dataset.dataset_dir, cfg.dataset.dataset_name, cfg)

    if cfg.params.use_play:
        preprocess_data(cfg.dataset.playset_dir,cfg.dataset.playset_name, cfg)

    train(cfg)


if __name__ == '__main__':
    main()




        