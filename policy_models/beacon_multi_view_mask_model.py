import torch
import torch.nn as nn
import torch.nn.functional as F
import robomimic.utils.tensor_utils as TensorUtils
from .network_modules import (SequentialImageEncoder,
                             Image2BeaconPredictor,
                             PositionalEncoding,
                             BaseTransformerPolicy)

class BeaconMultiViewMaskTransformer(BaseTransformerPolicy):
    def __init__(self,
                 token_dim=128, 
                 state_dim=8, 
                 action_dim=7, 
                 beacon_dim=3, 
                 num_heads=4, 
                 num_layers=2, 
                 sequence_length=5,
                 use_ee=False):
        super(BeaconMultiViewMaskTransformer, self).__init__()

        if use_ee:
            self.state_key = 'ee_states'
        else:
            self.state_key = 'joint_states'
        

        # Unsupervised image features
        self.static_encoder = SequentialImageEncoder(output_dim=token_dim)
        self.ego_encoder = SequentialImageEncoder(output_dim=token_dim)

        # Beacon prediction network
        self.beacon_predictor = Image2BeaconPredictor(beacon_dim=beacon_dim)
        if token_dim == beacon_dim:
            self.beacon_proj = nn.Identity()
        else:
            self.beacon_proj = nn.Linear(beacon_dim, token_dim)

        # States
        if token_dim == state_dim:
            self.state_proj = nn.Identity()
        else:    
            self.state_proj = nn.Linear(state_dim, token_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim,
                                                   nhead=num_heads,
                                                   activation="relu",
                                                   batch_first=True,
                                                   norm_first=False)
        self.tranformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                        num_layers=num_layers)

        # Position encoding
        self.pos_encoder = PositionalEncoding(d_model=token_dim, max_len=sequence_length)

        # Action token
        action_token = torch.rand(1, 1, token_dim)
        self.action_token = nn.Parameter(action_token, 
                                         requires_grad=True)

        # MLP policy
        self.fc1 = nn.Linear(token_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.pi_mean = nn.Linear(32, action_dim)
        self.pi_std = nn.Linear(32, action_dim)

        # Loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.name = 'pred_beacon_staticegomask_transformer'
        self.token_labels = ['states', 'beacon', 'static', 'ego', 'action_tokens']
        
    def feature_encoder(self, data):        
        batch = data['joint_states'].shape[0]
        states = torch.cat((data[self.state_key], data['gripper_states']), dim=-1)
        states = self.state_proj(states).unsqueeze(dim=-2)
        b_hat = self.beacon_predictor(data['image_rgb'])
        b_token = self.beacon_proj(b_hat).unsqueeze(dim=-2)

        ego_segmen = data['image_ego'] * data['image_ego_mask']
        static_segmen = data['image_rgb'] * data['image_mask']
        
        static_phi = self.static_encoder(static_segmen).unsqueeze(dim=-2) 
        ego_phi = self.ego_encoder(ego_segmen).unsqueeze(dim=-2)

        action_tokens = self.action_token.repeat(batch, 1, 1)

        tokens = torch.cat((states, b_token, static_phi, ego_phi, action_tokens), dim=-2)

        return b_hat, tokens
    
    def forward_transformer(self, tokens):
        batch, seq, n_tokens, token_dim = tokens.shape
        tokens = tokens.view(batch, seq*n_tokens, token_dim)

        encoded_tokens = self.tranformer_encoder(tokens)
        encoded_tokens = encoded_tokens.view(batch, seq, n_tokens, token_dim)

        return encoded_tokens[:, :, -1, :] # the action token is appended last in the sequence

    def forward_policy(self, action_tokens):
        batch, seq, token_dim = action_tokens.shape
        action_tokens = action_tokens.reshape(batch*seq, token_dim)
        x = F.relu(self.fc1(action_tokens))
        x = F.relu(self.fc2(x))
        x_mean = self.pi_mean(x)
        x_std = torch.exp(0.5 * self.pi_std(x))
        eps = torch.rand_like(x_std)
        x = x_mean + x_std * eps
        return x.view(batch, seq, -1)

    def forward(self, data):
        b_hat, tokens = TensorUtils.time_distributed(data, self.feature_encoder)

        pos_encoding = self.pos_encoder(tokens)
        tokens = tokens + pos_encoding.unsqueeze(0).unsqueeze(2)

        encoded_action_tokens = self.forward_transformer(tokens)
        action = self.forward_policy(encoded_action_tokens)

        return b_hat, action, tokens
    
    def calculate_loss(self, data, play_data=None):

        b_hat, action, _ = self.forward(data['obs'])

        loss = {}
        loss1 = self.mse_loss(action, data['actions'])
        loss2 = self.mse_loss(b_hat, data['obs']['beacons'])

        if not torch.is_grad_enabled():
            loss['test_loss'] = loss1 + loss2
            loss['policy_test_loss'] = loss1.item()
            loss['beacon_test_loss'] = loss2.item()
        else:
            loss['train_loss'] = loss1 + loss2
            loss['policy_loss'] = loss1.item()
            loss['beacon_loss'] = loss2.item()

        if play_data is not None:
            b_hat = TensorUtils.time_distributed(play_data['obs']['image_rgb'], self.beacon_predictor)
            loss3 = self.mse_loss(b_hat, play_data['obs']['beacons']) 
            loss['train_loss'] += loss3
            loss['play_loss'] = loss3.item()
        
        return loss
    
    def get_action(self, buffer):
        test_data = {"mask": None, "b_hat": None, "attn_weights": None}
        self.input_tokens = []
        with torch.no_grad():
            b_hat, action, tokens = self.forward(buffer)
            
            batch, seq, n_tokens, token_dim = tokens.shape
            tokens = tokens.view(batch, seq*n_tokens, token_dim)
            self.input_tokens.insert(0, tokens)
            test_data["b_hat"] = b_hat
            test_data["attn_weights"] = self.get_attention_weights()
        return test_data, action[:, -1, :].squeeze(0).detach().cpu().numpy()