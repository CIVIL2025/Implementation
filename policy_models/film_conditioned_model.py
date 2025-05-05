import torch
import torch.nn as nn

import robomimic.utils.tensor_utils as TensorUtils
from .network_modules import BaseTransformerPolicy


class FilmConditioned(BaseTransformerPolicy):
    def __init__(self,
                 state_encoder,
                 static_view_encoder,
                 ego_view_encoder,
                 transformer_encoder,
                 action_head,
                 pos_encoder,
                 token_dim,
                 action_dim,
                 state_dim
                 ):
        super(FilmConditioned, self).__init__()
    
        # Modules used during inferance
        self.state_encoder = state_encoder(state_dim=state_dim,
                                           token_dim=token_dim)
        
        self.static_view_encoder = static_view_encoder(token_dim=token_dim)
        
        self.ego_view_encoder = ego_view_encoder(token_dim=token_dim)

        self.transformer_encoder = transformer_encoder(token_dim=token_dim)

        self.action_head = action_head(token_dim=token_dim,
                                       action_dim=action_dim)
        
        self.pos_encoder = pos_encoder(token_dim)

        action_token = torch.rand(1, 1, token_dim)
        self.action_token = nn.Parameter(action_token,
                                         requires_grad=True)

        # Loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.name = "film_conditioned"
        self.token_labels = ["state", "static", "ego", "action"]

    def feature_encoder(self, data, encoder="mask_img"):        
        batch = data['joint_states'].shape[0]

        states = torch.cat((data['joint_states'], data['gripper_states']), dim=-1)
        state_token = self.state_encoder(states).unsqueeze(dim=-2)

        static_token = self.static_view_encoder(data['image_rgb'], data['language'])
        ego_token = self.ego_view_encoder(data['image_ego'])

        action_token = self.action_token.repeat(batch, 1, 1)

        static_token = static_token.unsqueeze(dim=-2)
        ego_token = ego_token.unsqueeze(dim=-2)

        tokens = torch.cat((state_token, static_token, ego_token, action_token), dim=-2)

        return tokens

    def forward_transformer(self, tokens):
        batch, seq, n_tokens, token_dim = tokens.shape
        tokens = tokens.view(batch, seq*n_tokens, token_dim)

        # Mask to prevent the transformer from attending to future observations
        self.src_mask = self.compute_mask(seq, n_tokens).to(tokens.device)
        encoded_tokens = self.transformer_encoder(tokens,
                                                  mask=self.src_mask)
        encoded_tokens = encoded_tokens.view(batch, seq, n_tokens, token_dim)

        return encoded_tokens[:, :, -1, :] # the action token is appended last in the sequence
    
    def forward(self, data, encoder="mask_img"):

        tokens = TensorUtils.time_distributed(data, self.feature_encoder, **{'encoder': encoder})

        pos_encoding = self.pos_encoder(tokens)
        tokens = tokens + pos_encoding.unsqueeze(0).unsqueeze(2)

        encoded_action_tokens = self.forward_transformer(tokens)
        action = self.action_head(encoded_action_tokens)

        return action, tokens
    

    def calculate_loss(self, data, play_data=None):

        action, _ = self.forward(data['obs'])

        loss = {}

        loss1 = self.mse_loss(action, data['actions'])


        if not torch.is_grad_enabled():
            loss['test_loss'] = loss1
            loss['policy_test_loss'] = loss1.item()
        else:
            loss['train_loss'] = loss1 
            loss['policy_loss'] = loss1.item()

        return loss
    
    def get_action(self, buffer, encoder="img", get_attn_mat=False):
        test_data = {'mask': None, 'b_hat': None, "attn_weights": None}
        self.input_tokens = []
        with torch.no_grad():
            action, tokens = self.forward(buffer, encoder=encoder)
            if get_attn_mat:
                batch, seq, n_tokens, token_dim = tokens.shape
                tokens = tokens.view(batch, seq*n_tokens, token_dim)
                self.input_tokens.insert(0, tokens)
                test_data["attn_weights"] = self.get_attention_weights(self.src_mask)
        return test_data, action[:, -1, :].squeeze(0).detach().cpu().numpy()
    
    def compute_mask(self, seq_len, n_tokens):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.repeat_interleave(n_tokens, dim=-1).repeat_interleave(n_tokens, dim=-2).to(torch.bool)