
import torch
import torch.nn as nn
import robomimic.utils.tensor_utils as TensorUtils
from .network_modules import BaseTransformerPolicy


class CIVIL(BaseTransformerPolicy):
    def __init__(self,
                 state_encoder,
                 beacon_encoder,
                 static_view_encoder,
                 ego_view_encoder,
                 transformer_encoder,
                 action_head,
                 pos_encoder,
                 token_dim,
                 action_dim,
                 state_dim,
                 beacon_dim,
                 white_mask=False,
                 ):
        super(CIVIL, self).__init__()
    
        # Modules used during inferance
        self.state_encoder = state_encoder(state_dim=state_dim,
                                           token_dim=token_dim)
        
        self.beacon_encoder =  beacon_encoder(beacon_dim=beacon_dim,
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

        # Images encoder trained with mask
        self.static_mask_encoder = static_view_encoder(token_dim=token_dim)
        self.ego_mask_encoder = ego_view_encoder(token_dim=token_dim)

        assert self.static_view_encoder is not self.static_mask_encoder, "Static mask encoder, and normal encoder should not have the same parameters"
        assert self.ego_mask_encoder is not self.ego_view_encoder, "Ego mask encoder, and normal encoder should not have the same parameters"

        # Loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.name = "civil"
        self.token_labels = ["state", "beacon", "static", "ego", "action"]
        self.white_mask = white_mask

    def feature_encoder(self, data, encoder="mask_img"):        
        batch = data['joint_states'].shape[0]

        states = torch.cat((data['joint_states'], data['gripper_states']), dim=-1)
        state_token = self.state_encoder(states).unsqueeze(dim=-2)

        b_hat, beacon_token = self.beacon_encoder(data['image_rgb'])

        if encoder == 'img':
            # b_hat, beacon_token = self.static_view_encoder(data['image_rgb'])
            static_token = self.static_view_encoder(data['image_rgb'])
            ego_token = self.ego_view_encoder(data['image_ego'])
        elif encoder == 'mask_img':
            mask_img = data['image_rgb'] * data['image_mask']
            mask_ego = data['image_ego'] * data['image_ego_mask']
            if self.white_mask:
                mask_img = torch.where(mask_img == 0, torch.ones_like(mask_img), mask_img)
                mask_ego = torch.where(mask_ego == 0, torch.ones_like(mask_ego), mask_ego)
            static_token = self.static_mask_encoder(mask_img)
            ego_token = self.ego_mask_encoder(mask_ego)
        

        action_token = self.action_token.repeat(batch, 1, 1)

        beacon_token = beacon_token.unsqueeze(dim=-2)
        static_token = static_token.unsqueeze(dim=-2)
        ego_token = ego_token.unsqueeze(dim=-2)

        tokens = torch.cat((state_token, beacon_token, static_token, ego_token, action_token), dim=-2)

        return b_hat, tokens

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

        b_hat, tokens = TensorUtils.time_distributed(data, self.feature_encoder, **{'encoder': encoder})

        pos_encoding = self.pos_encoder(tokens)
        tokens = tokens + pos_encoding.unsqueeze(0).unsqueeze(2)

        encoded_action_tokens = self.forward_transformer(tokens)
        action = self.action_head(encoded_action_tokens)

        return b_hat, action, tokens
    

    def calculate_loss(self, data, play_data=None):

        b_hat, action, _ = self.forward(data['obs'])

        loss = {}

        loss1 = self.mse_loss(action, data['actions'])
        # Calculate beacon loss filtering out nan cases
        loss2 = 0.0
        for i, beacon in enumerate(torch.split(data['obs']['beacons'], 3, dim=-1)):
            nan_index = torch.isnan(beacon).any(dim=-1)
            loss2 += self.mse_loss(b_hat[:, :,i*3:i*3+3][~nan_index], beacon[~nan_index])


        if not torch.is_grad_enabled():
            loss['test_loss'] = loss1 + loss2
            loss['policy_test_loss'] = loss1.item()
            loss['beacon_test_loss'] = loss2.item()
        else:
            loss['train_loss'] = loss1 + loss2
            loss['policy_loss'] = loss1.item()
            loss['beacon_loss'] = loss2.item()

        if play_data is not None:
            b_hat, _ = TensorUtils.time_distributed(play_data['obs']['image_rgb'], self.beacon_encoder)
            # Calculate beacon loss filtering out nan cases
            loss3 = 0.0
            for i, beacon in enumerate(torch.split(play_data['obs']['beacons'], 3, dim=-1)):
                nan_index = torch.isnan(beacon).any(dim=-1)
                loss3 += self.mse_loss(b_hat[:, :, i*3:i*3+3][~nan_index], beacon[~nan_index])
            loss['train_loss'] += loss3
            loss['play_loss'] = loss3.item()

        return loss
    
    def forward_encoders(self, data):

        mask_img = data['image_rgb'] * data['image_mask']
        mask_ego = data['image_ego'] * data['image_ego_mask']
        if self.white_mask:
            mask_img = torch.where(mask_img == 0, torch.ones_like(mask_img), mask_img)
            mask_ego = torch.where(mask_ego == 0, torch.ones_like(mask_ego), mask_ego)
        static_phi_mask = self.static_mask_encoder(mask_img)
        ego_phi_mask = self.ego_mask_encoder(mask_ego)

        static_phi= self.static_view_encoder(data['image_rgb'])
        ego_phi= self.ego_view_encoder(data['image_ego'])


        return static_phi_mask, ego_phi_mask, static_phi, ego_phi

    def encoder_loss(self, data, play_data=None):
        
        static_phi_mask, ego_phi_mask, static_phi, ego_phi = TensorUtils.time_distributed(data['obs'], self.forward_encoders)

        loss = {}

        loss1 = self.mse_loss(static_phi_mask, static_phi)
        loss2 = self.mse_loss(ego_phi_mask, ego_phi)

        if not torch.is_grad_enabled():
            loss['test_encoder_loss'] = loss1 + loss2
            loss['test_static_encoder_loss'] = loss1.item()
            loss['test_ego_encoder_loss'] = loss2.item()
        else:
            loss['train_encoder_loss'] = loss1 + loss2
            loss['static_encoder_loss'] = loss1.item()
            loss['ego_encoder_loss'] = loss2.item()

        if play_data is not None:
            static_phi_mask, ego_phi_mask, static_phi, ego_phi = TensorUtils.time_distributed(play_data['obs'], self.forward_encoders)
            loss3 = self.mse_loss(static_phi_mask, static_phi)
            loss4 = self.mse_loss(ego_phi_mask, ego_phi)
            loss['train_encoder_loss'] += loss3 + loss4
            loss['play_encoder_loss'] = loss3.item() + loss4.item()

        return loss
    
    def get_action(self, buffer, encoder="img", get_attn_mat=False):
        test_data = {'mask': None, 'b_hat': None, "attn_weights": None}
        self.input_tokens = []
        with torch.no_grad():
            b_hat, action, tokens = self.forward(buffer, encoder=encoder)
            test_data["b_hat"] = b_hat
            if get_attn_mat:
                batch, seq, n_tokens, token_dim = tokens.shape
                tokens = tokens.view(batch, seq*n_tokens, token_dim)
                self.input_tokens.insert(0, tokens)
                test_data["attn_weights"] = self.get_attention_weights(self.src_mask)
        return test_data, action[:, -1, :].squeeze(0).detach().cpu().numpy()
    
    def compute_mask(self, seq_len, n_tokens):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.repeat_interleave(n_tokens, dim=-1).repeat_interleave(n_tokens, dim=-2).to(torch.bool)
        