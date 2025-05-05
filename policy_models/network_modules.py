import abc
import tqdm
import clip
import math
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn 
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributions as D
from termcolor import colored
from byol_pytorch import BYOL
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.base_nets import CropRandomizer
from torchvision.models import resnet18, resnet50
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2, Compose, Resize, CenterCrop, Normalize, InterpolationMode, ColorJitter, GaussianBlur, RandomGrayscale, Resize, ToTensor


class SequentialImageEncoder(nn.Module):
    def __init__(self, feature_dim=512, output_dim=64, pretrained = True):
        super(SequentialImageEncoder, self).__init__()
        
        self.cnn = resnet18(pretrained=pretrained)
        self.cnn.fc = nn.Identity() 
        self.output_dim = output_dim
    
        self.fc = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)  

        return x
    

class Image2BeaconPredictor(nn.Module):
    def __init__(self, feature_dim=512, beacon_dim=4):
        super(Image2BeaconPredictor, self).__init__()

        self.beacon_dim = beacon_dim

        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity() 

        self.beacon_pred = nn.Linear(feature_dim, beacon_dim)

    def forward(self, x):
        x = self.cnn(x)  
        x = self.beacon_pred(x)

        return x
    

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class RandomResize(nn.Module):
    def __init__(self, sizes, p=0.2):
        super().__init__()
        self.sizes = sizes
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            chosen_size = random.choice(self.sizes)

            resize_transform = Resize(chosen_size)
            return resize_transform(x)
        return x

class Image2BeaconPredictorAugmented(nn.Module):
    def __init__(self, feature_dim=512, beacon_dim=4, augment_fn = True):
        super(Image2BeaconPredictorAugmented, self).__init__()

        self.beacon_dim = beacon_dim

        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity() 

        self.beacon_pred = nn.Linear(feature_dim, beacon_dim)
        self.sizes = (30, 50, 100)

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            RandomGrayscale(p=0.2),
            RandomApply(
                GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            RandomResize(self.sizes, p=0.2)
        )
        self.aug1 = augment_fn if not augment_fn else DEFAULT_AUG

    def forward(self, x):
        x = self.aug1(x)
        x = self.cnn(x)  
        x = self.beacon_pred(x)

        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, freq = 10000.0):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(freq)) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :]
    

def apply_batch_mask(token, mask_ratio):
        if torch.is_grad_enabled:
            print("masking out things")
            batch_size, seq_len = token.shape[:2]
            token_shape = token.shape
            mask = torch.rand(batch_size, device = token.device) < mask_ratio  
            mask = mask.view(batch_size, *([1] * (len(token_shape)-1)))
            return token * (~mask)
        else:
            return token
        
def generate_attn_mask(batch, seq, n_tokens, target_token_idx, mask_ratio):
    S = seq * n_tokens
    B = batch

    mask = torch.zeros((B, S))

    for b in range(B):
        if torch.rand(1) < mask_ratio and torch.is_grad_enabled():
            for s in range(seq):
                idx = s * -n_tokens + target_token_idx
                mask[b, idx] = float('-inf')
    
    return mask

def visualize_tokens(tokens):
    tokens = np.array(tokens[0].detach().cpu())
    plt.figure(figsize=(8, 6))
    sns.heatmap(tokens, center=0, annot=True, fmt=".2f", linewidths=0.5, cbar=True)

    plt.title("Token Matrix Visualization")
    plt.xlabel("Token Index")
    plt.ylabel("Token Index")
    plt.savefig("token_entries_visualization.png")
    input("************************************************")
    return


######################################## Base Transformer ##########################################

class BaseTransformerPolicy(nn.Module):
    @abc.abstractmethod
    def feature_encoder():
        raise NotImplementedError

    @abc.abstractmethod
    def forward_transformer():
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward_policy():
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward():
        raise NotImplementedError
    
    @abc.abstractmethod
    def calculate_loss():
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_action():
        raise NotImplementedError
    
    def get_attention_weights(self, mask=None):

        transformer_attention_weights = []
        for src, self_attn in zip(self.input_tokens, self._transformer_attn_layers):
            src_key_padding_mask = None
            src_mask = None

            if mask is not None:
                src_mask = mask

            src_key_padding_mask = F._canonical_mask(
                mask=src_key_padding_mask,
                mask_name="src_key_padding_mask",
                other_type=F._none_or_dtype(src_mask),
                other_name="src_mask",
                target_type=src.dtype,
            )

            src_mask = F._canonical_mask(
                mask=src_mask,
                mask_name="src_mask",
                other_type=None,
                other_name="",
                target_type=src.dtype,
                check_other=False,
            )

            _, attention_weights = self_attn(src,
                                                src,
                                                src,
                                                attn_mask=src_mask,
                                                key_padding_mask=src_key_padding_mask,
                                                need_weights=True,
                                                is_causal=False)
            
            transformer_attention_weights.append(attention_weights.detach().cpu().numpy()[0])

        return transformer_attention_weights
    

    def hook_attn_weights(self, module, input, output):
        """
        Hook to store the output of intermediate transformer encoder layers
        this will be used to calculate the attention weights later on
        """
        self.input_tokens.append(output)


    def register_intermediate_attention_hooks(self):

        self._transformer_attn_layers = []
        transformer_encoder_layers = []
        if hasattr(self, "latent_model"): 
            for name, module in self.latent_model.named_modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    transformer_encoder_layers.append(module)
                    self._transformer_attn_layers.append(module.self_attn)

        else: 
            for name, module in self.named_modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    transformer_encoder_layers.append(module)
                    self._transformer_attn_layers.append(module.self_attn)

                
        self.model_hooks = []
        # We only need the outputs of intermediate encoder layers
        for module in transformer_encoder_layers[:-1]:
            hook_handle = module.register_forward_hook(self.hook_attn_weights)
            self.model_hooks.append(hook_handle)

        self.input_tokens = []


    def remove_model_hooks(self):

        if len(self.model_hooks) != 0:
            for handle in self.model_hooks:
                handle.remove()

        else:
            print("There are not hooks to remove")

    
    def get_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)

    def step_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss['train_loss'].backward()
        self.optimizer.step()

    def step_scheduler(self):
        self.scheduler.step()


######################################## CIVIL ##########################################


class StateEncoder(nn.Module):
    def __init__(self, state_dim, token_dim):
        super(StateEncoder, self).__init__()   

        if token_dim == state_dim:
            self.state_proj = nn.Identity()
        else:
            self.state_proj = nn.Linear(state_dim, token_dim) 

    def forward(self, states):   
        return self.state_proj(states)


class BeaconEncoder(nn.Module):
    def __init__(self, beacon_dim, token_dim, pretrained=True):
        super(BeaconEncoder, self).__init__()

        self.cnn = resnet18(pretrained=pretrained)
        self.cnn.fc = nn.Identity()
        self.beacon_pred = nn.Linear(512, beacon_dim)

        if beacon_dim == token_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(beacon_dim, token_dim)

    def forward(self, img):
        x = self.cnn(img)
        b_hat = self.beacon_pred(x)
        b_token = self.proj(b_hat)
        return b_hat, b_token


class ImageEncoder(nn.Module):
    def __init__(self, token_dim, pretrained=True):
        super(ImageEncoder, self).__init__()
        
        self.cnn = resnet18(pretrained=pretrained)
        self.cnn.fc = nn.Identity()
        
        self.proj = nn.Linear(512, token_dim)

    def forward(self, img):
        x = self.cnn(img)
        return self.proj(x)


class TransformerEncoder(nn.Module):
    def __init__(self, token_dim, num_head, activation="relu", num_layers=2):
        super(TransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim,
                                                   nhead=num_head,
                                                   activation=activation,
                                                   batch_first=True,
                                                   norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

    def forward(self, tokens, mask):
        return self.encoder(tokens, mask=mask)


class MLPPolicy(nn.Module):
    def __init__(self, token_dim, action_dim):
        super(MLPPolicy, self).__init__()

        self.fc1 = nn.Linear(token_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.pi_mean = nn.Linear(32, action_dim)
        self.pi_std = nn.Linear(32, action_dim)

    def forward(self, action_tokens):
        batch, seq, token_dim = action_tokens.shape
        action_tokens = action_tokens.reshape(batch*seq, token_dim)
        x = F.relu(self.fc1(action_tokens))
        x = F.relu(self.fc2(x))
        x_mean = self.pi_mean(x)
        x_std = torch.exp(0.5 * self.pi_std(x))
        eps = torch.rand_like(x_std)
        x = x_mean + x_std * eps
        return x.view(batch, seq, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, freq = 10000.0):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(freq)) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :]

    
########################################### BYOL Model #############################################
'''
BYOL implementation: https://github.com/lucidrains/byol-pytorch
@misc{grill2020bootstrap,
    title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
    author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
    year = {2020},
    eprint = {2006.07733},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
'''

class BYOLImageEncoder(nn.Module):
    def __init__(self, feature_dim=512, pretrained = False):
        super(BYOLImageEncoder, self).__init__()
        self.resnet = resnet50(pretrained = True)
        self.resnet.fc = nn.Linear(in_features = self.resnet.fc.in_features, out_features = feature_dim)

    def pretrain(self, device, dataloaders, dataset_length, num_epochs = 200, view = "static"): # or "ego"
        byol = BYOL(
            self.resnet,
            image_size = 32,
            hidden_layer = 'avgpool'
            )
        byol.train()
        opt = torch.optim.Adam(byol.parameters(), lr=1e-6)
        
        for epoch in tqdm.tqdm(range(num_epochs)):
            running_loss = 0.0
            for dataloader, length in zip(dataloaders, dataset_length):
                for data in dataloader:
                    if view  == "static":
                        image = data['obs']['image_rgb'] * 255
                    elif view == "ego":
                        image = data['obs']['image_ego'] * 255
                    batch, sequence, channel, height, width = image.shape
                    image = image.view(batch * sequence, channel, height, width)
                    loss = byol(image.to(device))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    byol.update_moving_average()
                    running_loss += loss.item() * image.shape[0]
            epoch_loss = running_loss / length
            print( f'byol epoch loss: {epoch_loss}')
        byol.eval()
        self.resnet.eval()
        for name, param in self.resnet.named_parameters():
            param.requires_grad=False

    def forward(self, x):
        if len(x.shape) == 5:
            batch_size, sequence_length, channels, height, width = x.size()
            x = x.view(batch_size * sequence_length, channels, height, width)
            x = self.resnet(x)
            x = x.view(batch_size, sequence_length, -1)
        else:
            x = self.resnet(x)

        return x
    
########################################### Object Oriented Model #############################################

'''
Taken from https://github.com/UT-Austin-RPL/VIOLA
@article{zhu2022viola,
  title={VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors},
  author={Zhu, Yifeng and Joshi, Abhishek and Stone, Peter and Zhu, Yuke},
  journal={6th Annual Conference on Robot Learning (CoRL)},
  year={2022}
}
'''

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = torch.nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
                                      torch.from_numpy(np.linspace(-1, 1, in_h)).float())

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h
        
    def forward(self, x):
        assert(x.shape[1] == self._in_c), colored("[Spatial Softmax]", "red") + f'_in_c should be {self._in_c}, got {x.shape[1]} instead'
        assert(x.shape[2] == self._in_h), colored("[Spatial Softmax]", "red") + f'_in_c should be {self._in_h}, got {x.shape[2]} instead'
        assert(x.shape[3] == self._in_w), colored("[Spatial Softmax]", "red") + f'_in_c should be {self._in_w}, got {x.shape[3]} instead'

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)
        attention = F.softmax(h, dim=-1)

        keypoint_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True).view(-1, self._num_kp)
        keypoint_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True).view(-1, self._num_kp)

        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class RoIAlignWrapper(nn.Module):
    def __init__(self,  
                output_size = (6,6),
                spatial_scale = 1.0,
                sampling_ratio = -1,
                aligned = True,
                bbox_size = 4
                ):
        super(RoIAlignWrapper, self).__init__()
        assert(aligned==True)
        self.output_size = output_size
        self.bbox_size = bbox_size
        self.roi_align = torchvision.ops.RoIAlign(output_size=output_size,
                                                  spatial_scale=spatial_scale,
                                                  sampling_ratio=sampling_ratio,
                                                  aligned = aligned)
        
    def forward(self, x, bbox_list):
        batch_size, channel_size, h, w = x.shape
        bbox_num = bbox_list[0].shape[0]
        out = self.roi_align(x, bbox_list)
        out = out.reshape(batch_size, bbox_num, channel_size, *self.output_size)

        return out
    
    def output_shape(self, input_shape):
        """Return a batch of input sequences"""
        return (input_shape[0], self.output_size[0], self.output_size[1])
    

class BBoxPositionEncoding(nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=1,            
            dim=4,
            factor_ratio=1.,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / (dim * 2)) * dim)
        self.channels = channels

        inv_freq = 1.0 / (10 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var
        factor = nn.Parameter(torch.ones(1) * factor_ratio)
        self.register_parameter("factor", factor)

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)
        pos_embed = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_embed_sin = pos_embed.sin()
        pos_embed_cos = pos_embed.cos()
        spatial_pos_embedding = torch.cat([pos_embed_sin, pos_embed_cos], dim=-1)
        self.spatial_pos_embedding = torch.flatten(spatial_pos_embedding, start_dim=-2)
        return self.spatial_pos_embedding * self.factor


class SpatialProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim):
        super().__init__()

        assert(len(input_shape) == 3), "You should not use FlattenProjection if not having a tensor with 3 dimensions (excluding batch dimension)"

        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

class EyeInHandKeypointNet(nn.Module):
    def __init__(self,
                 img_h=128,
                 img_w=128,
                 num_kp=16,
                 visual_feature_dimension=64,
                 remove_layer_num=4):
        super().__init__()
        
        self.encoder = ResnetConv(remove_layer_num=remove_layer_num, img_c=3)
        encoder_output_shape = self.encoder.output_shape(input_shape=(3, img_h, img_w))
        self.spatial_softmax = SpatialSoftmax(in_c=encoder_output_shape[0], in_h=encoder_output_shape[1], in_w=encoder_output_shape[2], num_kp=num_kp)
        self.fc = torch.nn.Sequential(torch.nn.Linear(num_kp * 2, visual_feature_dimension))
        self.visual_feature_dimension = visual_feature_dimension
        
    def forward(self, x):
        if self.training:
            x = self.encoder(x) 
            x = self.spatial_softmax(x)
            x = self.fc(x)
        else:
            with torch.no_grad():
                x = self.encoder(x)
                x = self.spatial_softmax(x)
                x = self.fc(x)            
        return x

class ResnetConv(torch.nn.Module):
    def __init__(self,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 img_c=3,
                 last_c=None,
                 no_stride=False):

        super().__init__()

        assert(remove_layer_num <= 5)
        # For training policy
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[:-remove_layer_num]
       
        if img_c != 3:
            # If use eye_in_hand, we need to increase the channel size
            conv0 = torch.nn.Conv2d(img_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1
        self.resnet18_embeddings = torch.nn.Sequential(*layers)

        if no_training:
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        self.remove_layer_num = remove_layer_num

    def forward(self, x):
        h = self.resnet18_embeddings(x)
        return h

    def output_shape(self, input_shape):
        assert(len(input_shape) == 3)
        
        if self.remove_layer_num == 2:
            out_c = 512
            scale = 32.
        elif self.remove_layer_num == 3:
            out_c = 256
            scale = 16.
        elif self.remove_layer_num == 4:
            out_c = 128
            scale = 8.
        elif self.remove_layer_num == 5:
            out_c = 64
            scale = 4.

        if self.no_stride:
            scale = scale / 4.
        out_h = int(math.ceil(input_shape[1] / scale))
        out_w = int(math.ceil(input_shape[2] / scale))
        return (out_c, out_h, out_w)
    
class FlattenProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim):
        super().__init__()

        assert(len(input_shape) == 4), "You should not use FlattenProjection if not having a tensor with 4 dimensions (excluding batch dimension)"
        in_dim = input_shape[-3] * input_shape[-2] * input_shape[-1]
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = torch.flatten(x, start_dim=2)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)
    
class TemporalSinusoidalPositionEncoding(nn.Module):
    def __init__(self,
                 input_shape,
                 inv_freq_factor=10,
                 factor_ratio=None):
        super().__init__()
        self.input_shape = input_shape
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_shape[-1]
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels))
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)
        
    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape


class DataAugGroup(torch.nn.Module):
    """
    Add augmentation to multiple inputs
    """
    def __init__(
            self,
            use_color_jitter=False,
            use_random_erasing=False,
            **aug_kwargs
    ):
        super().__init__()
        
        transforms = []

        self.use_color_jitter = use_color_jitter
        self.use_random_erasing = use_random_erasing

        if self.use_color_jitter:
            color_jitter = torchvision.transforms.ColorJitter(**aug_kwargs["color_jitter"])
            transforms.append(color_jitter)
        if self.use_random_erasing:
            random_erasing = torchvision.transforms.RandomErasing(**aug_kwargs["random_erasing"])
            transforms.append(random_erasing)

        self.transforms = torchvision.transforms.Compose(transforms)

    def forward(self, x_groups):
        split_channels = []
        for i in range(len(x_groups)):
            split_channels.append(x_groups[i].shape[0])
        if self.training:
            x = torch.cat(x_groups, dim=0)
            out = self.transforms(x)
            out = torch.split(out, split_channels, dim=0)
            return out
        else:
            out = x_groups
        return out
    

class IdentityAug(nn.Module):
    def __init__(self,
                 input_shapes=None,
                 *args,
                 **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def output_shape(self, input_shape):
        return input_shape


class TranslationAugGroup(nn.Module):
    """
    Add translation augmentation to a group of images, applying the same translation)
    """
    def __init__(
            self,
            input_shapes,
            translation,
    ):
        super().__init__()

        self.pad = nn.ReplicationPad2d(translation)

        self.channels = []
        for input_shape in input_shapes:
            self.channels.append(input_shape[0])
        pad_output_shape = (sum(self.channels), input_shape[1] + translation, input_shape[2] + translation)
        self.crop_randomizer = CropRandomizer(input_shape=pad_output_shape,
                                              crop_height=input_shape[1],
                                              crop_width=input_shape[2])

    def forward(self, x_groups):
        if self.training:
            x = torch.cat(x_groups, dim=1)
            out = self.pad(x)
            out = self.crop_randomizer.forward_in(out)
            out = torch.split(out, self.channels, dim=1)
            return out
        else:
            out = x_groups
        return out
    

class TemporalGMMPolicyMLPLayer(nn.Module):
    """This is a mlp layer that handles temporal sequence. (because of of restricted usage from robomimic)
    """
    def __init__(self, 
                 token_dim,
                 action_dim,
                 num_modes=5, 
                 min_std=0.0001,
                 num_layers=2,
                 num_dim=1024,              
                 mlp_activation="relu",
                 std_activation="softplus", 
                 low_noise_eval=True, 
                 use_tanh=False):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = action_dim
        input_dim = token_dim
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.std_activation = std_activation
        self.use_tanh = use_tanh
        
        if mlp_activation == 'relu':
            mlp_activate_fn = torch.nn.ReLU
        elif mlp_activation == 'leaky-relu':
            mlp_activate_fn = torch.nn.LeakyReLU
        
        out_dim = self.num_modes * self.output_dim
        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            mlp_activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                mlp_activate_fn()]

        else:
            self._layers += [torch.nn.Linear(input_dim, num_dim)]
        self.mlp_layers = torch.nn.Sequential(*self._layers)
        
        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        self.mean_layer = nn.Linear(num_dim, out_dim)
        self.scale_layer = nn.Linear(num_dim, out_dim)
        self.logits_layer = nn.Linear(num_dim, self.num_modes)

    def forward_fn(self, x):
        out = self.mlp_layers(x)
        means = self.mean_layer(out).view(-1, self.num_modes, self.output_dim)
        scales = self.scale_layer(out).view(-1, self.num_modes, self.output_dim)
        logits = self.logits_layer(out)

        means = torch.tanh(means)
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std
        
        return means, scales, logits
        
    def forward(self, x):

        means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        self.means = means
        self.scales = scales
        self.logits = logits

        return dist


########################################### CLIP Model #############################################

class CLIPImageEncoder(nn.Module):
    def __init__(self, feature_dim = 512, output_dim = 64, device = 'cuda'):
        super(CLIPImageEncoder, self).__init__()
        # self.model, _ = clip.load("ViT-B/32", device=device)
        self.model, _ = clip.load("RN50", device=device)
        self.to_pil_image = torchvision.transforms.ToPILImage()
        print(colored('[CLIP]', 'red') + f'CLIP model has initialized.')
        print(self.model.__dict__['_modules']['visual'])

        for name, param in self.model.named_parameters():
            param.requires_grad=False
        self.model.eval()
        self.device = device
        self.fc = nn.Linear(feature_dim, output_dim)
        self.preprocess = self._transform(self.model.visual.input_resolution)
        
    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            # _convert_image_to_rgb,
            # ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def forward(self, x):
        with torch.no_grad():
            # image_tensor = []
            # for batch in x:
            #     image = self.to_pil_image(batch)
            #     image_tensor.append(self.preprocess(image))
            # image_tensor = torch.stack(image_tensor, dim=0)
            image_tensor = self.preprocess(x)
            image_features = self.model.encode_image(image_tensor.to(self.device))
        
        phi = self.fc(image_features.float())
        return phi
    
    def state_dict(self, *args, **kwargs):
        # Only return the state_dict of fully connected layer
        return self.fc.state_dict(*args, **kwargs)


    
########################################### CLIP Model with Adapters #############################################

class CLIPAdapterImageEncoder(nn.Module):
    def __init__(self, feature_dim = 1024, output_dim = 64, device = 'cuda', Ego=False):
        super(CLIPAdapterImageEncoder, self).__init__()
        self.model, _ = clip.load("RN50", device=device)
        # self.model, _ = clip.load("ViT-B/32", device=device)
        self.to_pil_image = torchvision.transforms.ToPILImage()
        print(colored('[CLIP]', 'red') + f'CLIP Module model has initialized.')

        for name, param in self.model.named_parameters():
            param.requires_grad=False #freeze original parameters
        self.model.eval()
        self.device = device
        self.fc = nn.Linear(512, output_dim)
        self.preprocess = self._transform(self.model.visual.input_resolution)
        self.static_bottom_adapter = BottomAdapter()
        self.static_top_adapter = TopAdapter(input_dim=feature_dim)
        if Ego:
            self.ego_bottom_adapter = BottomAdapter()
            self.ego_top_adapter = TopAdapter(input_dim=feature_dim)
        
    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            # _convert_image_to_rgb,
            # ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def forward(self, x, Ego=False):
        with torch.no_grad():
            # image_tensor = []
            # for batch in x:
            #     image = self.to_pil_image(batch)
            #     image_tensor.append(self.preprocess(image))
            # image_tensor = torch.stack(image_tensor, dim=0)
            image_tensor = self.preprocess(x)
        if not Ego:
            phi = self.static_bottom_adapter.forward(image_tensor.to(self.device)) #feed raw image into bottom adapter
            with torch.no_grad():
                image_features = self.model.encode_image(phi) #feed into model, no gradients
                image_features = image_features.float()
            phi = self.static_top_adapter.forward(image_features) #feed into top mlp module before fc head
        else:
            phi = self.ego_bottom_adapter.forward(image_tensor.to(self.device)) #feed raw image into bottom adapter
            with torch.no_grad():
                image_features = self.model.encode_image(phi) #feed into model, no gradients
                image_features = image_features.float()
            phi = self.ego_top_adapter.forward(image_features) #feed into top mlp module before fc head

        phi = self.fc(phi.float())
        return phi
    
    def state_dict(self, *args, **kwargs):
        # Only return the state_dict of fully connected layer
        return self.fc.state_dict(*args, **kwargs)
        # return super().state_dict(*args, **kwargs)


class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linearity = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.skip_connection = nn.Identity()

    def forward(self, x):
        return self.skip_connection(x) + self.up_proj(self.non_linearity(self.down_proj(x)))

class TopAdapter(nn.Module): #mlp
    def __init__(self, input_dim, hidden_dim=256, output_dim=512):
        super(TopAdapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        return x
        # return x + self.up_proj(self.relu(self.down_proj(x)))

class BottomAdapter(nn.Module):
    def __init__(self, in_channels=3, bottleneck_channels=16, kernel_size=1):
        super(BottomAdapter, self).__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.down_proj = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=kernel_size, padding=padding)
        self.non_linearity = nn.ReLU(inplace=True)
        self.up_proj = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x):
        adapter_out = self.up_proj(self.non_linearity(self.down_proj(x)))
        return x + adapter_out

class MiddleAdapter(nn.Module):
    def __init__(self, in_channels, bottleneck_channels=32):
        super(MiddleAdapter, self).__init__()
        self.down = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.up   = nn.Conv2d(bottleneck_channels, in_channels, 1)
        
    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))  # skip connection



######################################## FILM  ##########################################

'''
FiLM Block implementation taken from:https://github.com/caffeinism/FiLM-pytorch.git
@InProceedings{perez2018film,
  title={FiLM: Visual Reasoning with a General Conditioning Layer},
  author={Ethan Perez and Florian Strub and Harm de Vries and Vincent Dumoulin and Aaron C. Courville},
  booktitle={AAAI},
  year={2018}
}
'''

class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = (1 + gamma) * x + beta
        return x
    

class ResBlock(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_chn, out_chn, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_chn, out_chn, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_chn)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, gamma, beta):

        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, gamma, beta)
        x = self.relu2(x)
        
        x = x + identity
        
        return x
    
class FilmResnet(nn.Module):
    def __init__(self, token_dim, block_num = 4, dim_cond = 768):
        super(FilmResnet, self).__init__()

        self.film_proj = nn.Linear(dim_cond, 2 * block_num * token_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.block_num = block_num
        self.res_blocks = nn.ModuleList([ResBlock(token_dim, token_dim) for _ in range(block_num)])
        self.conv = nn.Conv2d(token_dim, 512, 1, 1, 0)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, token_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, cond):
        x = self.cnn(x)
        cond = self.film_proj(cond)
        cond = cond.view(x.size(0), self.block_num, 2, -1) 

        for idx, res_block in enumerate(self.res_blocks):
            gamma = cond[:, idx, 0, :]
            beta = cond[:, idx, 1, :]

            x = res_block(x, gamma, beta)
        
        # flatten the output
        x = self.conv(x)
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        
        return x