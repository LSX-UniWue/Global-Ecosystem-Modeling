# reference: https://github.com/NVlabs/AFNO-transformer
import argparse
import collections
import os
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from typing import List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from einops import rearrange
from torchinfo import summary
import lightning as L

# This import requires a newer version of torchvision than is installed on the FAU-cluster
# Either comment out this import and the ViTBlock class, or install a newer version of torchvision
# For working version of torchvision and torch see slurm/train.slurm
# from torchvision.models.vision_transformer import EncoderBlock

from utils.YParams import YParams
from utils.darcy_loss import LpLoss
from utils.img_utils import PeriodicPad2d
from utils.metrics import lat_weighted_rmse, lat_weighted_mse
from utils.weighted_acc_rmse import weighted_rmse_torch, weighted_acc_torch


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].real,
                         self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].imag,
                         self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].imag,
                         self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].real,
                         self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo',
                             o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) - \
                torch.einsum('...bi,bio->...bo',
                             o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + \
                self.b2[0]
        )

        o2_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo',
                             o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) + \
                torch.einsum('...bi,bio->...bo',
                             o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + \
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")

        return x + bias


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            block_type: str = "original"
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip
        self.block_type = block_type

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


def modify_first_layer(model, in_channels):
    # Modify the first layer of the model to accept the new number of input channels
    model.patch_embed.proj = nn.Conv2d(in_channels, model.embed_dim, kernel_size=model.patch_embed.patch_size,
                                       stride=model.patch_embed.patch_size)


def monkey_patch_forward_features(model):
    model.forward_features_original = model.forward_features

    def new_forward_features(self, x, time_encoding):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x + time_encoding[0] + time_encoding[1]
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    # model.forward_features = new_forward_features

    pass


def freeze_model_parts(n_blocks_to_freeze, model):
    # reverse the order of the blocks, so that the first blocks are the last ones in the list
    print("Freezing the first {} blocks".format(n_blocks_to_freeze))
    for i, block in enumerate(model.blocks):
        curr_block = model.blocks[i]  # go over the blocks in reverse order
        if i < n_blocks_to_freeze:
            for param in curr_block.parameters():
                param.requires_grad = False
        else:
            for param in curr_block.parameters():
                param.requires_grad = True

    for param in model.patch_embed.proj.parameters():
        param.requires_grad = False

    model.pos_embed.requires_grad = False


def remove_spatial_mixing_layers(model):
    for b in model.blocks:
        b.filter = nn.Identity()
    print("Replaced spatial mixing layers with identity layers (\"no op layer\")")


def remove_blocks(model, n_blocks_to_remove):
    if n_blocks_to_remove > 0:
        model.blocks = nn.ModuleList(model.blocks[:-n_blocks_to_remove])
        print("Removed the last {} blocks".format(n_blocks_to_remove))


def remove_channel_mixing_layers(model):
    for b in model.blocks:
        b.mlp = nn.Identity()
    print("Replaced MLP/channel mixing layers with identity layers (\"no op layer\")")


def add_tanh_to_head(model):
    model.head = nn.Sequential(OrderedDict([
        ('head', model.head),
        ('sigmoid', nn.Tanh()),
    ]))


def switch_channel_and_spatial_mixing_layers(model):
    for b in model.blocks:
        filter_ = b.filter
        mlp = b.mlp
        b.filter = mlp
        b.mlp = filter_

    print("Switched spatial and channel mixing layers")


def replace_pos_embedding(model):
    old_pos_embed = model.pos_embed
    shape = old_pos_embed.shape
    print(shape)
    new_pos_embed = torch.zeros(shape[0], shape[1], shape[2])
    print(new_pos_embed.shape)

    model.pos_embed = nn.Parameter(new_pos_embed, requires_grad=False)

    print("Replaced positional embedding with non-trainable zero embedding")
    return model


class ViTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.vit_block = EncoderBlock(
            num_heads=12,
            hidden_dim=768,
            mlp_dim=768 * 4,
            dropout=0.0,
            attention_dropout=0.0,
        )

    def forward(self, x):
        original_shape = x.shape

        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.vit_block(x)
        return x.view(original_shape)


def switch_first_block_to_VIT_block(model):
    model.blocks[0] = ViTBlock()

    print("Switched first FFN-block to ViT-block")


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size_shrink_factor=2, embed_dim=768, drop=0.1, ):
        super().__init__()
        self.patch_size = (8, 8)
        self.embed_dim = embed_dim
        img_size = (720, 1440)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=20, embed_dim=768)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 768))

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.layer1 = nn.Linear(embed_dim, embed_dim // hidden_size_shrink_factor)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.layer2 = nn.Linear(embed_dim // hidden_size_shrink_factor, embed_dim // (hidden_size_shrink_factor * 2))
        self.layer3 = nn.Linear(embed_dim // (hidden_size_shrink_factor * 2),
                                embed_dim // (hidden_size_shrink_factor * 4))

        self.head = nn.Linear(embed_dim // (hidden_size_shrink_factor * 4), 1 * self.patch_size[0] * self.patch_size[1],
                              bias=False)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.drop(x)
        x = x.reshape(B, self.h, self.w, self.embed_dim)

        x = self.layer1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.layer2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.layer3(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.head(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=720 // self.patch_size[0],
            w=1440 // self.patch_size[1],
        )

        return x


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.tanh(x)
        return x


from typing import List


class SimpleCNNwHead(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, filters: List = [], kernel_sizes: List = []):
        super(SimpleCNNwHead, self).__init__()

        def kernel_size_to_padding(kernel_size):
            return (kernel_size - 1) // 2

        self.conv_layers = []
        n_pool_layers = 0
        for i in range(len(filters)):
            ks = kernel_sizes[i]
            padding_s = kernel_size_to_padding(ks)
            if i == 0:
                first_conv = nn.Conv2d(in_channels=in_channels, out_channels=filters[i], kernel_size=ks, stride=1,
                                       padding=padding_s)
                self.conv_layers.append(first_conv)
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                n_pool_layers += 1
            elif i == len(filters) - 1:
                last_conv = nn.Conv2d(in_channels=filters[i - 1], out_channels=128, kernel_size=ks, padding=padding_s)
                self.conv_layers.append(last_conv)
            else:
                self.conv_layers.append(
                    nn.Conv2d(in_channels=filters[i - 1], out_channels=filters[i], kernel_size=ks, stride=1,
                              padding=padding_s))
                self.conv_layers.append(nn.ReLU())
            if n_pool_layers < 3:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                n_pool_layers += 1

        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.head = nn.Linear(128, out_channels * 8 * 8, bias=False)

    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)

        x = rearrange(
            x,
            "b c h w -> b h w c",
            # (batch_size, height, width, channels), switching from (batch_size, channels, height, width)

        )

        x = self.head(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=8,
            p2=8,
            h=90,
            w=180,
        )

        x = torch.tanh(x)
        return x


class PrecipNet(nn.Module):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class AdapterNet(nn.Module):
    def __init__(self, img_size=(720, 1440), patch_size=(16, 16), in_chans=2, out_chans=2, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        # embedding block
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        # adapter block
        self.adapter_down = nn.Linear(in_features=768, out_features=384, bias=True)
        self.adapter_act = nn.GELU()
        self.adapter_up = nn.Linear(in_features=384, out_features=768, bias=True)

        self.head = nn.Linear(embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], bias=True)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)

        x = self.adapter_down(x)
        x = self.adapter_act(x)
        x = self.adapter_up(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.head(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )

        return x


class AFNONet(L.LightningModule):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            variables=None
    ):
        super().__init__()
        self.params = params
        self.block_type = params.block_type
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        self.depth = params.depth

        self.variables = params.channel_names

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold,
                  hard_thresholding_fraction=hard_thresholding_fraction, block_type=self.block_type)
            for i in range(self.depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.ndvi = True
        self.loss_object = LpLoss(rel_or_abs=params.loss_type)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_lat_lon(self, lat, lon):
        self.lat = np.arange(-90, 90, step=0.25)
        self.lon = np.arange(0, 360, step=0.25)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

    def compute_and_apply_mask(self, target, prediction, exclude_locations: bool = False):
        """
        Computes and applies a mask to the target and prediction tensors if working with ndvi data
        Otherwise, returns the original tensors
        """
        if self.ndvi:
            mask = torch.ones_like(target, device=self.device, dtype=torch.float)
            mask = torch.logical_and(mask, target >= -1.)  # Set to 0 if below -1
            mask = torch.logical_and(mask, target <= 1.)  # Set to 0 if above 1

            # if self.eval_mask is None:
            #   self.eval_mask = self.load_valid_data_mask()

            if exclude_locations:

                if self.geo_mask is None:
                    # create the geo_mask:
                    self.geo_mask = self.create_geo_mask(target)

                target = target * self.geo_mask
                prediction = prediction * self.geo_mask

            # target = target * self.eval_mask
            # prediction = prediction * self.eval_mask

            target = target * mask
            prediction = prediction * mask

        return target, prediction

    def process_batch(self, input_data, target_data):
        total_loss = 0
        stepwise_losses = {k: [] for k in self.variables}

        for i in range(self.params.rollout_steps):
            if i == 0:
                gen_step = self.forward(input_data)
            else:
                gen_step = self.forward(gen_step)

            tar_step_m, gen_step_m = self.compute_and_apply_mask(
                target=target_data[:, (i * self.params.N_out_channels):(i + 1) * self.params.N_out_channels],
                prediction=gen_step,
                exclude_locations=self.params.exclude_locations)

            # contains for each variable the loss, and one overall loss termed "loss"
            step_loss: dict = lat_weighted_mse(pred=gen_step_m, y=tar_step_m, vars=self.variables, lat=self.lat)

            # add the loss for each variable to the stepwise_losses dict
            for k in self.variables:
                stepwise_losses[k].append(step_loss[k])

            total_loss += step_loss["loss"]

        return total_loss, stepwise_losses

    def process_forward_pass(self, batch, batch_idx, mode: str = "train"):
        inp, tar, time_encoding_start, time_encoding_steps = batch  # unpack

        total_loss, stepwise_losses = self.process_batch(inp, tar)

        # stepwise losses has this layout {"variable": [loss step 1, loss step 2, ...]}

        step_logs = {}
        step_logs[f"{mode}/loss"] = total_loss

        for key in stepwise_losses.keys():
            for step, individual_step_loss in enumerate(stepwise_losses[key]):
                step_logs[f"{mode}/{key}_step_{step}"] = individual_step_loss

        return step_logs, total_loss

    def training_step(self, batch, batch_idx):
        step_logs, total_loss = self.process_forward_pass(batch, batch_idx, mode="train")

        self.log_dict(step_logs, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        step_logs, _ = self.process_forward_pass(batch, batch_idx, mode="val")

        self.log_dict(step_logs, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return step_logs

    def test_step(self, batch, batch_idx):
        step_logs, _ = self.process_forward_pass(batch, batch_idx, mode="test")

        self.log_dict(step_logs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return step_logs

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.max_epochs,
                                                                  last_epoch=-1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
            }
        }


class AfnoWithConvOutput(nn.Module):
    def __init__(self, params, in_chans, out_chans, backbone):
        super().__init__()
        self.params = params
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.backbone = backbone
        self.new_one_d_conv = nn.Conv2d(self.in_chans, self.out_chans, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.new_one_d_conv(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def load_model(model, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname, map_location=torch.device('cpu'))
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key
            if 'module.' in key:  # model was stored using ddp which prepends module
                name = str(key[7:])
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def setup(in_chan, img_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='../config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='afno_backbone', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--ndvi-finetune", action='store_true')
    parser.add_argument("--ndvi", action='store_true')
    parser.add_argument("--freeze_params", action='store_true')
    args = parser.parse_args()
    freeze = False
    out_chan = 20
    params = YParams(os.path.abspath(args.yaml_config), args.config,
                     print_params=False)

    params['N_in_channels'] = 20
    params['N_out_channels'] = 20

    print(params.block_type)
    model = AFNONet(params=params, img_size=img_size, patch_size=(8, 8), in_chans=in_chan, out_chans=out_chan,
                    num_blocks=8)

    #    model = load_model(model, "/Users/pascal/Downloads/backbone_fourcastnet.ckpt")

    return model


def add_pretrained_adapters(original_model, adapter_network, in_chans=20, out_chans=20):
    try:
        adapter_down = adapter_network.adapter_down
        adapter_act = adapter_network.adapter_act
        adapter_up = adapter_network.adapter_up
    except AttributeError as e:
        print(e)
        # assuming that we are loading a distributed model
        adapter_down = adapter_network.module.adapter_down
        adapter_act = adapter_network.module.adapter_act
        adapter_up = adapter_network.module.adapter_up

    for b in original_model.blocks:
        new_down = deepcopy(adapter_down)
        new_act = deepcopy(adapter_act)
        new_up = deepcopy(adapter_up)

        b.drop_path = nn.Sequential(
            collections.OrderedDict([('new_adapter_down', new_down),
                                     ('new_adapter_act', new_act),
                                     ('new_adapter_up', new_up),
                                     ]
                                    )

        )

    model = AfnoWithConvOutput(params=None, in_chans=in_chans, out_chans=out_chans,
                               backbone=original_model)

    return model


def add_new_block(model):
    new_block = Block(dim=768, mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, num_blocks=8,
                      sparsity_threshold=0.01, hard_thresholding_fraction=1.0, block_type="original")
    new_head = nn.Linear(768, 20 * 8 * 8, bias=False)
    trunc_normal_(new_head.weight, std=.02)

    model.head = nn.Sequential(OrderedDict([
        ('new_block', new_block),
        ('new_head', new_head),
    ]))

    return model


def add_adapters(model, out_chans):
    for b in model.blocks:
        b.drop_path = nn.Sequential(
            collections.OrderedDict([('new_adapter_down', nn.Linear(in_features=768, out_features=384, bias=True)),
                                     ('new_adapter_act', nn.GELU()),
                                     ('new_adapter_up', nn.Linear(in_features=384, out_features=768, bias=True)),
                                     ]

                                    )

        )

    model = AfnoWithConvOutput(params=None, in_chans=20, out_chans=out_chans, backbone=model)

    return model


if __name__ == "__main__":
    in_chan = 20
    img_size = (720, 1440)
    afno = setup(in_chan, img_size)

    sample = torch.randn(3, in_chan, img_size[0], img_size[1])
    summary(afno, input_size=sample.shape)
