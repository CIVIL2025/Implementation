import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
import robomimic.utils.tensor_utils as TensorUtils
from torch.optim.lr_scheduler import StepLR
from .network_modules import (CLIPAdapterImageEncoder,
                              SequentialImageEncoder, 
                             PositionalEncoding)


class CLIPAdapterTransformer(nn.Module):
    def __init__(self, token_dim=128, state_dim=8, action_dim=7, beacon_dim=3, num_heads=4, num_layers=2, sequence_length=5):
        super(CLIPAdapterTransformer, self).__init__()

        # Unsupervised image features
        self.image_encoder = CLIPAdapterImageEncoder(output_dim=token_dim)

        self.ego_encoder = SequentialImageEncoder(output_dim=token_dim)

        # States
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

        self.name = 'clip_adapter_transformer'

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())
        
    def feature_encoder(self, data):        
        batch = data['joint_states'].shape[0]

        states = torch.cat((data['joint_states'], data['gripper_states']), dim=-1)
        states = self.state_proj(states).unsqueeze(dim=-2)

        phi = self.image_encoder(data['image_rgb']).unsqueeze(dim=-2).float()
        ego_phi = self.ego_encoder(data['image_ego']).unsqueeze(dim=-2).float()

        action_tokens = self.action_token.repeat(batch, 1, 1)

        tokens = torch.cat((states, phi, ego_phi, action_tokens), dim=-2)

        return tokens
    
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
        tokens = TensorUtils.time_distributed(data, self.feature_encoder)

        pos_encoding = self.pos_encoder(tokens)
        tokens = tokens + pos_encoding.unsqueeze(0).unsqueeze(2)

        encoded_action_tokens = self.forward_transformer(tokens)
        action = self.forward_policy(encoded_action_tokens)

        return action
    
    def calculate_loss(self, data, play_data=None):

        action = self.forward(data['obs'])


        loss = {}
        loss1 = self.mse_loss(action, data['actions'])


        if not torch.is_grad_enabled():
            loss['test_loss'] = loss1 
            loss['policy_test_loss'] = loss1.item()

        else:
            loss['train_loss'] = loss1 
            loss['policy_loss'] = loss1.item()

        return loss

    def get_action(self, buffer):
        test_data = {"mask": None, "b_hat": None}
        with torch.no_grad():
            action = self.forward(buffer)[:, -1, :].squeeze(0).detach().cpu().numpy()
        
        return test_data, action
    
    def get_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)

    def step_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss['train_loss'].backward()
        self.optimizer.step()

    def step_scheduler(self):
        self.scheduler.step()