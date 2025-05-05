import hydra
import torch
import torch.nn as nn
from termcolor import colored
from omegaconf import OmegaConf
import robomimic.utils.tensor_utils as TensorUtils
from .network_modules import (PositionalEncoding, 
                              SequentialImageEncoder, 
                              apply_batch_mask, 
                              BaseTransformerPolicy,
                              visualize_tokens)

class BeaconEncoderLatentTransformer(BaseTransformerPolicy):
    def __init__(self, latent_model_dir = None):
        super(BeaconEncoderLatentTransformer, self).__init__()

        latent_model_cfg = OmegaConf.load(f"{'/'.join(latent_model_dir.split('/')[:-2])}/.hydra/config.yaml")
        self.latent_model = hydra.utils.instantiate(latent_model_cfg.model.model)
        try:
            self.latent_model.load_state_dict(torch.load(latent_model_dir, weights_only=True))
        except:
            print(colored("[**********************************ALERT**********************************]\n", "red") + 
                    colored("Some part of the model is missing or have changed! It might be fine if this is a CLIP model, otherwise it is a SERIOUS ISSUE.", "red"))
            self.latent_model.load_state_dict(torch.load(latent_model_dir, weights_only=True), strict=False)
        self.latent_model.eval()
        for name, param in self.latent_model.named_parameters():
            param.requires_grad=False
        
        self.state_key = self.latent_model.state_key

        if 'static_encoder' in dict(self.latent_model.named_children()).keys(): 
            self.static_encoder = SequentialImageEncoder(output_dim=self.latent_model.fc1.in_features)
        self.ego_encoder = SequentialImageEncoder(output_dim=self.latent_model.fc1.in_features)

        # Loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.name = 'beacon_encoder_latent'
        

    def feature_encoder(self, data):
        batch = data['joint_states'].shape[0]

        states = torch.cat((data[self.state_key], data['gripper_states']), dim=-1)
        states = self.latent_model.state_proj(states).unsqueeze(dim=-2)

        b_token = self.latent_model.beacon_proj(data['beacons']).unsqueeze(dim=-2)

        mask_phi = None
        mask_ego_phi = None
        if torch.is_grad_enabled():
            ego_segmen = data['image_ego'] * data['image_ego_mask']
            mask_ego_phi = self.latent_model.ego_encoder(ego_segmen).unsqueeze(dim=-2)
            if hasattr(self, 'static_encoder'):
                static_segmen = data['image_rgb'] * data['image_mask']
                mask_phi = self.latent_model.static_encoder(static_segmen).unsqueeze(dim=-2)

        action_tokens = self.latent_model.action_token.repeat(batch, 1, 1)

        ego_phi = self.ego_encoder(data['image_ego']).unsqueeze(dim=-2)
        static_phi = None
        if hasattr(self, 'static_encoder'):
            static_phi = self.static_encoder(data['image_rgb']).unsqueeze(dim=-2)
            tokens = torch.cat((states, b_token, static_phi, ego_phi, action_tokens), dim=-2)
        else:
            tokens = torch.cat((states, b_token, ego_phi, action_tokens), dim=-2)
        
        if hasattr(self.latent_model, 'layer_norm'):
            tokens = self.latent_model.layer_norm(tokens)
        # visualize_tokens(tokens)

        return tokens, static_phi, ego_phi, mask_phi, mask_ego_phi
    
    def forward(self, data):
        
        tokens, static_phi, ego_phi, mask_phi, mask_ego_phi = TensorUtils.time_distributed(data, self.feature_encoder)

        if not torch.is_grad_enabled():
            self.input_tokens = []
            pos_encoding = self.latent_model.pos_encoder(tokens)
            tokens = tokens + pos_encoding.unsqueeze(0).unsqueeze(2)

            encoded_action_tokens = self.latent_model.forward_transformer(tokens)
            action = self.latent_model.forward_policy(encoded_action_tokens)

            batch, seq, n_tokens, token_dim = tokens.shape
            input_tokens = tokens.view(batch, seq * n_tokens, token_dim)
            self.input_tokens.insert(0, input_tokens)
        else:
            action = None



        return action, static_phi, ego_phi, mask_phi, mask_ego_phi
    
    def calculate_loss(self, data, play_data=None):

        action, static_phi, ego_phi, mask_phi, mask_ego_phi = self.forward(data['obs'])

        loss = {}
        if not torch.is_grad_enabled():
            ego_segmen = data['obs']['image_ego'] * data['obs']['image_ego_mask']
            mask_ego_phi = TensorUtils.time_distributed(ego_segmen, self.latent_model.ego_encoder).unsqueeze(dim=-2) 
            loss1 = self.mse_loss(ego_phi, mask_ego_phi)
            loss['test_loss'] = loss1
            loss['latent_test_loss'] = loss1.item() 
            if hasattr(self, 'static_encoder'):
                static_segmen = data['obs']['image_rgb'] * data['obs']['image_mask']
                mask_phi = TensorUtils.time_distributed(static_segmen, self.latent_model.static_encoder).unsqueeze(dim=-2) 
                loss2 = self.mse_loss(static_phi, mask_phi)
                loss['test_loss'] += loss2
                loss['latent_static_test_loss'] = loss2.item()

        else:
            loss1 = self.mse_loss(ego_phi, mask_ego_phi)
            loss['train_loss'] = loss1
            loss['latent_loss'] = loss1.item()
            if hasattr(self, 'static_encoder'):
                loss2 = self.mse_loss(static_phi, mask_phi)
                loss['train_loss'] += loss2
                loss['latent_static_loss'] = loss2.item()

        if play_data is not None:
            ego_phi = TensorUtils.time_distributed(play_data['obs']['image_ego'], self.ego_encoder).unsqueeze(dim=-2)
            ego_segmen = play_data['obs']['image_ego'] * play_data['obs']['image_ego_mask'] 
            masked_ego_phi = TensorUtils.time_distributed(ego_segmen, self.latent_model.ego_encoder).unsqueeze(dim=-2)  
            loss3 = self.mse_loss(ego_phi, masked_ego_phi)
            loss['train_loss'] += loss3 
            loss['play_loss'] = loss3.item()
            if hasattr(self, 'static_encoder'):
                static_phi = TensorUtils.time_distributed(play_data['obs']['image_rgb'], self.static_encoder).unsqueeze(dim=-2)
                static_segmen = play_data['obs']['image_rgb'] * play_data['obs']['image_mask']
                mask_phi = TensorUtils.time_distributed(static_segmen, self.latent_model.static_encoder).unsqueeze(dim=-2)  
                loss4 = self.mse_loss(static_phi, mask_phi)
                loss['train_loss'] += loss4
                loss['play_static_loss'] = loss4.item()


        return loss

    def get_action(self, buffer):
        test_data = {"mask": None, "b_hat": None, "attn_weights": None}
        with torch.no_grad():
            action, _, _, _, _ = self.forward(buffer)
            test_data["attn_weights"] = self.get_attention_weights()
        return test_data, action[:, -1, :].squeeze(0).detach().cpu().numpy()
    