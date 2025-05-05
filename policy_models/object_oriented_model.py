import torch
import torch.nn as nn
import robomimic.utils.tensor_utils as TensorUtils
from .network_modules import (BaseTransformerPolicy,
                              ResnetConv,
                              RoIAlignWrapper,
                              FlattenProjection,
                              BBoxPositionEncoding,
                              SpatialProjection,
                              EyeInHandKeypointNet,
                              TemporalSinusoidalPositionEncoding,
                              StateEncoder,
                              MLPPolicy,
                              DataAugGroup,
                              IdentityAug,
                              TranslationAugGroup,
                              TemporalGMMPolicyMLPLayer)


class ObjectOriented(BaseTransformerPolicy):
    def __init__(self,
                 data_aug: DataAugGroup,
                 img_aug,
                 encoder: ResnetConv,
                 pooling: RoIAlignWrapper,
                 projection: FlattenProjection,
                 bbox_norm: BBoxPositionEncoding,
                 spatial_projection: SpatialProjection,
                 ego_encoder: EyeInHandKeypointNet,
                 pos_encoder: TemporalSinusoidalPositionEncoding,
                 transformer_encoder: nn.TransformerEncoder,
                 state_encoder: StateEncoder,
                 action_head: MLPPolicy,
                 img_size = [200, 200],
                 ego_size = [200, 200],
                 k_obj=4,
                 token_dim=128,
                 state_dim=8,
                 action_dim=8):
        super(ObjectOriented, self).__init__()

        # Data Augmentation
        self.data_aug = data_aug
        self.img_aug = img_aug(input_shapes=([3]+img_size, [3]+ego_size))

        # Workspace image feature map 
        self.encoder = encoder
        spatial_map_shape = self.encoder.output_shape([3] + img_size)
        
        # Region features
        spatial_scale = spatial_map_shape[-1] / img_size[-1]
        self.pooling = pooling(spatial_scale=spatial_scale)
        pooling_shape = self.pooling.output_shape(spatial_map_shape)

        # Feature projection
        projection_shape = [k_obj] + list(pooling_shape)
        self.projection = projection(projection_shape,
                                     out_dim=token_dim)
        
        # Bbox sinusoidal encoding
        self.bbox_norm = bbox_norm(input_shape=(k_obj, token_dim))

        # Global spatial map
        self.spatial_projection = spatial_projection(out_dim=token_dim,
                                                     input_shape=spatial_map_shape)
        
        # Ego image encoder
        self.ego_global_encoder = ego_encoder(img_h=ego_size[0],
                                              img_w=ego_size[1],
                                              visual_feature_dimension=token_dim)

        # Temporal sinusoidal encoding for all tokens
        self.pos_encoder = pos_encoder(input_shape=[token_dim])

        # Transformer encoder
        self.transformer_encoder = transformer_encoder 

        # State projection
        self.state_encoder = state_encoder(state_dim, token_dim)

        # Policy
        self.action_head = action_head(token_dim=token_dim,
                                       action_dim=action_dim)
        
        # Action token
        action_token = torch.rand(1, 1, token_dim)
        self.action_token = nn.Parameter(action_token,
                                         requires_grad=True)
        
        # Loss function
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.name = 'object_oriented_ego_model'


    def feature_encoder(self, data):
        batch = data['joint_states'].shape[0]

        data['image_rgb'], data['image_ego'] = self.img_aug((data['image_rgb'], data['image_ego']))
        data['image_rgb'], data['image_ego'] = self.data_aug((data['image_rgb'], data['image_ego']))

        # Region features (Bboxes)
        spatial_map = self.encoder(data['image_rgb'])
        
        x = self.pooling(spatial_map, [bbox for bbox in data['image_bbox']])
        x = self.projection(x)

        bbox_norm = self.bbox_norm(data['image_bbox'])
        
        bbox_tokens = x + bbox_norm
    
        # Global features
        # Static view
        static_token = self.spatial_projection(spatial_map).unsqueeze(dim=-2)

        # Ego view
        ego_token = self.ego_global_encoder(data['image_ego']).unsqueeze(dim=-2)

        # States
        states = torch.cat((data['joint_states'], data['gripper_states']), dim=-1)
        states = self.state_encoder(states).unsqueeze(dim=-2)

        # Action tokens
        action_tokens = self.action_token.repeat(batch, 1, 1)

        # Group all tokens
        tokens = torch.cat((bbox_tokens, states, static_token, ego_token, action_tokens), dim=-2)
        
        # Mask to ignore missing bboxes
        n_tokens = tokens.shape[1]
        n_bbox = bbox_tokens.shape[1]
        src_key_padding_mask = torch.zeros(batch, n_tokens).to(torch.bool).to(tokens.device)
        src_key_padding_mask[:, :n_bbox] = ~torch.any(data['image_bbox'], dim=-1)
        
        return tokens, src_key_padding_mask 
    
    def forward_transformer(self, tokens, src_key_padding_mask):
        batch, seq, n_tokens, token_dim = tokens.shape
        tokens = tokens.view(batch, seq*n_tokens, token_dim)

        # Mask to ignore missing bboxes
        src_key_padding_mask = src_key_padding_mask.view(batch, seq*n_tokens)

        # Mask to prevent the transformer from attending to future observations
        self.src_mask = self.compute_mask(seq, n_tokens).to(tokens.device)

        encoded_tokens = self.transformer_encoder(tokens,
                                                  mask=self.src_mask,
                                                  src_key_padding_mask=src_key_padding_mask)
        encoded_tokens = encoded_tokens.view(batch, seq, n_tokens, token_dim)

        return encoded_tokens[:, :, -1, :] # the action token is appended last in the sequence
    
    def forward(self, data, last_step=False):
        tokens, src_key_padding_mask = TensorUtils.time_distributed(data, self.feature_encoder)

        pos_encoding = self.pos_encoder(tokens)
        tokens = tokens + pos_encoding.unsqueeze(0).unsqueeze(2)

        encoded_action_tokens = self.forward_transformer(tokens, src_key_padding_mask)
        if last_step:
            encoded_action_tokens = encoded_action_tokens[:, -1:, :]
        action = self.action_head(encoded_action_tokens)

        return action
    
    def get_action(self, buffer, encoder=None):
        test_data = {}
        with torch.no_grad():
            action = self.forward(buffer, last_step=True)

            if isinstance(self.action_head, TemporalGMMPolicyMLPLayer):
                action.sample()

        return test_data, action[:, -1, :].squeeze(0).detach().cpu().numpy()
    
    def compute_mask(self, seq_len, n_tokens):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.repeat_interleave(n_tokens, dim=-1).repeat_interleave(n_tokens, dim=-2).to(torch.bool)        