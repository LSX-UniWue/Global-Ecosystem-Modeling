# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import textwrap
import time
import numpy as np
import argparse

import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet, PrecipNet, AfnoWithConvOutput, add_adapters, AdapterNet, add_pretrained_adapters, \
    add_new_block, modify_first_layer, remove_spatial_mixing_layers, remove_channel_mixing_layers, add_tanh_to_head, \
    SimpleCNN, SimpleCNNwHead, freeze_model_parts, replace_pos_embedding, switch_first_block_to_VIT_block

from utils.img_utils import vis_precip
import wandb
from utils.weighted_acc_rmse import weighted_rmse_torch, unlog_tp_torch, weighted_acc_torch
from utils.darcy_loss import LpLoss
import matplotlib.pyplot as plt
from collections import OrderedDict

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict


class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.eval_mask = None
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        if params.log_to_wandb:
            wandb.init(config=params, name=params.name, group=params.group, project=params.project,
                       entity=params.entity)

        logging.info('rank %d, begin data loader init' % world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path,
                                                                                         dist.is_initialized(),
                                                                                         train=True,
                                                                                         years=params['train_years'])
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path,
                                                                     dist.is_initialized(), train=False,
                                                                     years=params['valid_years'])
        if params.mse_loss:
            self.loss_obj = nn.MSELoss()
        else:
            self.loss_obj = LpLoss(rel_or_abs=params.loss_type)
        logging.info('rank %d, data loader initialized' % world_rank)

        params.crop_size_x = self.valid_dataset.crop_size_x
        params.crop_size_y = self.valid_dataset.crop_size_y
        params.img_shape_x = self.valid_dataset.img_shape_x
        params.img_shape_y = self.valid_dataset.img_shape_y

        # precip models
        self.precip = True if "precip" in params else False

        self.ndvi_finetune = params["ndvi_finetune"]
        self.ndvi = params["ndvi"]
        self.freeze_params = params["freeze_params"]

        if self.freeze_params and not self.ndvi_finetune:
            raise Warning("The freeze parameters flag doesn't work without ndvi_finetune flag set")

        if self.precip:
            if 'model_wind_path' not in params:
                raise Exception("no backbone model weights specified")
            # load a wind model
            # the wind model has out channels = in channels
            out_channels = np.array(params['in_channels'])
            params['N_out_channels'] = len(out_channels)

            if params.nettype_wind == 'afno':
                self.model_wind = AFNONet(params).to(self.device)
            else:
                raise Exception("not implemented")

            if dist.is_initialized():
                self.model_wind = DistributedDataParallel(self.model_wind,
                                                          device_ids=[params.local_rank],
                                                          output_device=[params.local_rank],
                                                          find_unused_parameters=True)
            self.load_model_wind(params.model_wind_path)
            self.switch_off_grad(self.model_wind)  # no backprop through the wind model

        # reset out_channels for precip models
        if self.precip:
            params['N_out_channels'] = len(params['out_channels'])

        if params.nettype == 'afno':
            if self.ndvi_finetune:
                N_out_temp = params.N_out_channels
                N_in_temp = params.N_in_channels
                params.N_out_channels = 20  # to support loading the pre-trained model
                params.N_in_channels = 20
            self.model = AFNONet(params)
            if self.ndvi_finetune:
                # load pretrained model
                print("Loading pretrained model at %s" % params.pretrained_ckpt_path)
                self.load_pretrained_model(params.pretrained_ckpt_path)

                # Reset N_out_channels to the correct value
                params.N_out_channels = N_out_temp
                params.N_in_channels = N_in_temp

                self.model.head = nn.Linear(768, params.N_out_channels * 8 * 8, bias=params.use_bias_in_new_head)
                self.model._init_weights(self.model.head)

                modify_first_layer(self.model, params.N_in_channels)

                add_tanh_to_head(self.model)
                print("Loaded pretrained model %s" % params.pretrained_ckpt_path)

                if self.freeze_params:
                    freeze_model_parts(n_blocks_to_freeze=args.n_blocks_to_freeze, model=self.model)

            elif self.ndvi:
                print("Training complete network for NDVI from scratch")

            self.model = self.model.to(self.device)



        elif params.nettype == "higgins":
            if self.ndvi_finetune:
                N_out_temp = params.N_out_channels
                params.N_out_channels = 4  # to support loading the pre-trained model, which has been trained on channels [ 2, 20, 21, 22 ]
            self.model = AFNONet(params)
            if self.ndvi_finetune:
                print("Loading pretrained model at %s" % params.pretrained_ckpt_path)
                self.load_pretrained_model(params.pretrained_ckpt_path)
                # Reset N_out_channels to the correct value
                params.N_out_channels = N_out_temp
                # self.model = add_adapters(self.model)
                self.model.head = nn.Linear(768, params.N_out_channels * 8 * 8, bias=params.use_bias_in_new_head)
                # self.model.head =  # original head also has no bias
                self.model._init_weights(self.model.head)
            self.model = self.model.to(self.device)

        elif params.nettype == "higgins_new_finetune":
            print("Finetuning Higgins model with new input layer")
            N_out_temp = params.N_out_channels
            N_in_temp = params.N_in_channels

            params.N_out_channels = 20
            params.N_in_channels = 20

            self.model = AFNONet(params)

            print("Loading pretrained model at %s" % params.pretrained_ckpt_path)
            self.load_pretrained_model(params.pretrained_ckpt_path)

            params.N_out_channels = N_out_temp
            params.N_in_channels = N_in_temp

            self.model.head = nn.Linear(768, params.N_out_channels * 8 * 8, bias=params.use_bias_in_new_head)
            self.model._init_weights(self.model.head)
            add_tanh_to_head(self.model)

            modify_first_layer(self.model, params.N_in_channels)

            print(f"Model adapted with new input layer to accept input with {str(params.N_in_channels)} channels")

            self.model = self.model.to(self.device)


        elif params.nettype == 'adapter':
            print("Training adapter network for ground knowledge transfer")
            self.model = AdapterNet(img_size=(params.img_shape_x, params.img_shape_y),
                                    patch_size=(params.patch_size, params.patch_size), in_chans=params.N_in_channels,
                                    out_chans=params.N_out_channels)
            self.model = self.model.to(self.device)

        elif params.nettype == 'adapter_transfer':
            print("Finetuning weather model with pre-trained adapters inserted")
            N_out_temp = params.N_out_channels
            params.N_out_channels = 20  # to support loading the pre-trained model
            # load the pretrained model
            self.model = AFNONet(params)
            self.load_pretrained_model(params.pretrained_ckpt_path)
            params.N_out_channels = N_out_temp

            # load trained adapter network
            temp_adapter_network = torch.load(params.adapter_network_path)
            self.model = add_pretrained_adapters(self.model, temp_adapter_network, in_chans=params.N_in_channels,
                                                 out_chans=params.N_out_channels)
            self.model = self.model.to(self.device)



        elif params.nettype == "new_output_stage":
            print("Finetuning weather model with pre-trained adapters inserted and novel output stage")
            N_out_temp = params.N_out_channels
            params.N_out_channels = 20  # to support loading the pre-trained model
            # load the pretrained model
            self.model = AFNONet(params)
            self.load_pretrained_model(params.pretrained_ckpt_path)
            params.N_out_channels = N_out_temp
            # add the novel output stage
            self.model = add_new_block(self.model)
            # load trained adapter network
            temp_adapter_network = torch.load(params.adapter_network_path)
            self.model = add_pretrained_adapters(self.model, temp_adapter_network, in_chans=params.N_in_channels,
                                                 out_chans=params.N_out_channels)
            self.model = self.model.to(self.device)

        elif params.nettype == 'conv_output':
            print("Training conv_output network with pre-trained frozen base model")

            N_out_temp = params.N_out_channels
            params.N_out_channels = 20  # to support loading the pre-trained model
            # load the pretrained model
            self.model = AFNONet(params)
            self.load_pretrained_model(params.pretrained_ckpt_path)

            params.N_out_channels = N_out_temp

            self.model = AfnoWithConvOutput(params=params, in_chans=params.N_in_channels,
                                            out_chans=params.N_out_channels, backbone=self.model)

            if self.freeze_params:
                print("Freezing all layers except specific layers")
                for name, param in self.model.named_parameters():
                    if "new" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            self.model = self.model.to(self.device)

        else:
            raise Exception("not implemented")

        if self.params.vit_block:
            switch_first_block_to_VIT_block(self.model)
            self.model.to(self.device)

        # precip model
        if self.precip:
            self.model = PrecipNet(params, backbone=self.model).to(self.device)

        if self.params.enable_nhwc:
            # NHWC: Convert model to channels_last memory format
            self.model = self.model.to(memory_format=torch.channels_last)

        if params.log_to_wandb:
            wandb.watch(self.model)

        # if params.optimizer_type == 'FusedAdam':
        #  self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = params.lr)
        # else:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        # Used for geographical train/test split
        self.geo_mask = None

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)  # todo: fix this for custom model
        if params.two_step_training:
            if not params.resuming and params.pretrained:
                logging.info("Starting from pretrained one-step afno model at %s" % params.pretrained_ckpt_path)
                self.restore_checkpoint(params.pretrained_ckpt_path)
                self.iters = 0
                self.startEpoch = 0
                # logging.info("Pretrained checkpoint was trained for %d epochs"%self.startEpoch)
                # logging.info("Adding %d epochs specified in config file for refining pretrained model"%self.params.max_epochs)
                # self.params.max_epochs += self.startEpoch

        if dist.is_initialized():
            torch.cuda.set_device(params.local_rank)
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[params.local_rank],
                                                 output_device=[params.local_rank], find_unused_parameters=True)

        self.epoch = self.startEpoch

        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5,
                                                                        mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs,
                                                                        last_epoch=self.startEpoch - 1)
        elif params.scheduler == "CyclicLR":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=params.lr, max_lr=0.1,
                                                               step_size_up=10, step_size_down=10, mode='triangular2',
                                                               cycle_momentum=False)
        else:
            self.scheduler = None

        '''if params.log_to_screen:
          logging.info(self.model)'''
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
            #        self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            # on every 10th epoch, run validation
            valid_time, valid_logs = self.validate_one_epoch()
            if epoch == self.params.max_epochs - 1 and self.params.prediction_type == 'direct':
                valid_weighted_rmse, valid_loss, valid_l1 = self.validate_final()

            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        # logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # logging.info('train data time={}, train step time={}, valid step time={}'.format(data_time, tr_time, valid_time))
                logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))
                log_string = ""
                for metric_name, metric_value in valid_logs.items():
                    log_string += "{}: {} | ".format(metric_name, metric_value)
                fmted_log_string = textwrap.fill(log_string, 80)
                logging.info(fmted_log_string)

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info(
                        "Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    self.save_to_wandb()
                    exit()
            elif self.params.scheduler == "CyclicLR":
                if self.epoch >= self.params.max_epochs:
                    logging.info(
                        "Terminating training after reaching params.max_epochs while LR scheduler is set to CyclicLR")
                    self.save_to_wandb()
                    exit()

        self.save_to_wandb()

    #        if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
    #        if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct': Todo: re-integrate metrics from final epoch
    #          logging.info('Final Valid RMSE: Z500- {}. T850- {}, 2m_T- {}'.format(valid_weighted_rmse[0], valid_weighted_rmse[1], valid_weighted_rmse[2]))

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            # adjust_LR(optimizer, params, iters)
            data_start = time.time()
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
            if self.params.orography and self.params.two_step_training:
                orog = inp[:, -2:-1]

            if self.params.enable_nhwc:
                inp = inp.to(memory_format=torch.channels_last)
                tar = tar.to(memory_format=torch.channels_last)

            if 'residual_field' in self.params.target:
                tar -= inp[:, 0:tar.size()[1]]
            data_time += time.time() - data_start

            tr_start = time.time()

            self.model.zero_grad()
            if self.params.two_step_training:
                with amp.autocast(self.params.enable_amp):
                    gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
                    loss_step_one = self.loss_obj(gen_step_one, tar[:, 0:self.params.N_out_channels])
                    if self.params.orography:
                        gen_step_two = self.model(torch.cat((gen_step_one, orog), axis=1)).to(self.device,
                                                                                              dtype=torch.float)
                    else:
                        gen_step_two = self.model(gen_step_one).to(self.device, dtype=torch.float)
                    loss_step_two = self.loss_obj(gen_step_two,
                                                  tar[:, self.params.N_out_channels:2 * self.params.N_out_channels])
                    loss = loss_step_one + loss_step_two
            else:
                with amp.autocast(self.params.enable_amp):
                    if self.precip:  # use a wind model to predict 17(+n) channels at t+dt
                        with torch.no_grad():
                            inp = self.model_wind(inp).to(self.device, dtype=torch.float)
                        gen = self.model(inp.detach()).to(self.device, dtype=torch.float)
                    else:
                        gen = self.model(inp)

                    tar, gen = self.compute_and_apply_mask(target=tar,
                                                           prediction=gen,
                                                           exclude_locations=params.exclude_locations)  # mask out nonavailable data if working with ndvi data
                    loss = self.loss_obj(gen, tar)

            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()
            if self.params.scheduler == 'CyclicLR':  # CyclicLR scheduler needs to be updated after every batch
                self.scheduler.step()
            tr_time += time.time() - tr_start
        try:
            logs = {'loss': loss, 'loss_step_one': loss_step_one, 'loss_step_two': loss_step_two}
        except:
            logs = {'loss': loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, logs

    def compute_and_apply_mask(self, target, prediction, exclude_locations: bool = False):
        """
        Computes and applies a mask to the target and prediction tensors if working with ndvi data
        Otherwise, returns the original tensors
        """
        if self.ndvi:
            mask = torch.ones_like(target, device=self.device, dtype=torch.float)
            mask = torch.logical_and(mask, target >= -1.)  # Set to 0 if below -1
            mask = torch.logical_and(mask, target <= 1.)  # Set to 0 if above 1

            if self.eval_mask is None:
                self.eval_mask = self.load_valid_data_mask()

            if exclude_locations:

                if self.geo_mask is None:
                    # create the geo_mask:
                    self.geo_mask = self.create_geo_mask(target)

                target = target * self.geo_mask
                prediction = prediction * self.geo_mask

            target = target * self.eval_mask
            prediction = prediction * self.eval_mask

            target = target * mask
            prediction = prediction * mask

        return target, prediction

    def load_valid_data_mask(self):
        self.eval_mask = np.load(self.params.eval_mask)
        self.eval_mask = torch.from_numpy(self.eval_mask).to(self.device, dtype=torch.float)

        # create correctly shaped mask
        self.eval_mask = self.eval_mask.view(1, 1, self.eval_mask.shape[0], self.eval_mask.shape[1])

        return self.eval_mask

    def create_geo_mask(self, target):
        """
        Creates a mask for the target tensor, excluding test locations
        """
        coordinates = {"burkina": [311, 8], "random": [205, 180], "brasil": [481, 1216], "chile": [489, 1157],
                       "usa_coast": [250, 985], "australia": [437, 550], "wuerzburg": [160, 39],
                       "steigerwald": [160, 42],
                       "bayreuth": [160, 46], "poland": [149, 95]}

        print("Creating geo mask for target tensor")
        print(f"Target tensor shape: {target.shape}")

        size = self.params.exclude_locations_size

        mask = torch.ones_like(target, device=self.device, dtype=torch.bool)
        for location in coordinates.keys():
            lat, long = coordinates[location]
            for i in range(-size, size):
                for j in range(-size, size):
                    mask[:, :, lat + i, long + j] = 0

        if params.mask_corridor_west_east:
            for location in coordinates.keys():
                lat, long = coordinates[location]
                for i in range(-size, size):
                    mask[:, :, lat + i, long:mask.shape[3]] = 0

        elif params.mask_america:
            mask = torch.ones_like(target, device=self.device, dtype=torch.bool)
            mask[:, :, 0:mask.shape[2], 910:1310] = 0

        return mask

    def validate_one_epoch(self):
        self.model.eval()
        n_valid_batches = 20  # do validation on first 20 images, just for LR scheduler
        if self.params.normalization == 'minmax':
            raise Exception("minmax normalization not supported")
        elif self.params.normalization == 'zscore':
            if not self.ndvi:
                std_dev = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(
                    self.device)  # std-dev loaded from pre-computed global training stds

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)
        valid_weighted_rmse = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)
        valid_weighted_acc = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)

        valid_start = time.time()

        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                if self.params.orography and self.params.two_step_training:
                    orog = inp[:, -2:-1]

                if self.params.two_step_training:
                    gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
                    loss_step_one = self.loss_obj(gen_step_one, tar[:, 0:self.params.N_out_channels])

                    if self.params.orography:
                        gen_step_two = self.model(torch.cat((gen_step_one, orog), axis=1)).to(self.device,
                                                                                              dtype=torch.float)
                    else:
                        gen_step_two = self.model(gen_step_one).to(self.device, dtype=torch.float)

                    loss_step_two = self.loss_obj(gen_step_two,
                                                  tar[:, self.params.N_out_channels:2 * self.params.N_out_channels])
                    valid_loss += loss_step_one + loss_step_two
                    valid_l1 += nn.functional.l1_loss(gen_step_one, tar[:, 0:self.params.N_out_channels])
                else:
                    if self.precip:
                        with torch.no_grad():
                            inp = self.model_wind(inp).to(self.device, dtype=torch.float)
                        gen = self.model(inp.detach())
                    else:
                        gen = self.model(inp)

                    # mask out nonavailable data if working with ndvi data
                    tar, gen = self.compute_and_apply_mask(target=tar, prediction=gen,
                                                           exclude_locations=params.exclude_locations)

                    valid_loss += self.loss_obj(gen, tar)
                    valid_l1 += nn.functional.l1_loss(input=gen, target=tar)

                valid_steps += 1.
                # save fields for vis before log norm
                if (i == sample_idx) and (self.precip and self.params.log_to_wandb):
                    fields = [gen[0, 0].detach().cpu().numpy(), tar[0, 0].detach().cpu().numpy()]

                if self.precip:
                    gen = unlog_tp_torch(gen, self.params.precip_eps)
                    tar = unlog_tp_torch(tar, self.params.precip_eps)

                # direct prediction weighted rmse
                if self.params.two_step_training:
                    if 'residual_field' in self.params.target:
                        valid_weighted_rmse += weighted_rmse_torch((gen_step_one + inp),
                                                                   (tar[:, 0:self.params.N_out_channels] + inp))
                        valid_weighted_acc += weighted_acc_torch((gen_step_one + inp),
                                                                 (tar[:, 0:self.params.N_out_channels] + inp))
                    else:
                        valid_weighted_rmse += weighted_rmse_torch(gen_step_one, tar[:, 0:self.params.N_out_channels])
                        valid_weighted_acc += weighted_acc_torch(gen_step_one, tar[:, 0:self.params.N_out_channels])
                else:
                    if 'residual_field' in self.params.target:
                        valid_weighted_rmse += weighted_rmse_torch((gen + inp), (tar + inp))
                        valid_weighted_acc += weighted_acc_torch((gen + inp), (tar + inp))
                    else:
                        valid_weighted_rmse += weighted_rmse_torch(gen, tar)
                        valid_weighted_acc += weighted_acc_torch(gen, tar)

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)
            dist.all_reduce(valid_weighted_acc)

        logs = {}

        if world_rank == 0:
            # divide by number of steps
            valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
            valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
            valid_weighted_acc = valid_weighted_acc / valid_buff[2]
            if not self.precip:
                if not self.ndvi:
                    valid_weighted_rmse *= std_dev  # scaling back to original units only for previously normalized channels
                    valid_weighted_acc *= std_dev

            # download buffers
            valid_buff_cpu = valid_buff.detach().cpu().numpy()
            valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()
            valid_weighted_acc_cpu = valid_weighted_acc.detach().cpu().numpy()

            logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0]}

            # valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse, axis=0)
            for idx, channel_idx in enumerate(params['out_channels']):
                channel_name = params['channel_names'][channel_idx]
                logs['valid_rmse_' + channel_name] = valid_weighted_rmse_cpu[idx]
                logs['valid_acc_' + channel_name] = valid_weighted_acc_cpu[idx]

            if self.params.log_to_wandb:
                if self.precip:
                    fig = vis_precip(fields)
                    logs['vis'] = wandb.Image(fig)
                    plt.close(fig)
                wandb.log(logs, step=self.epoch)

        valid_time = time.time() - valid_start

        return valid_time, logs

    def validate_final(self):
        self.model.eval()
        n_valid_batches = int(
            self.valid_dataset.n_patches_total / self.valid_dataset.n_patches)  # validate on whole dataset
        valid_weighted_rmse = torch.zeros(n_valid_batches, self.params.N_out_channels)
        if self.params.normalization == 'minmax':
            raise Exception("minmax normalization not supported")
        elif self.params.normalization == 'zscore':
            mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(
                self.device)

        with torch.no_grad():
            valid_loss = torch.zeros(n_valid_batches, dtype=torch.float32, device=self.device)
            valid_l1 = torch.zeros(n_valid_batches, dtype=torch.float32, device=self.device)
            for i, data in enumerate(self.valid_data_loader):
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                if self.params.orography and self.params.two_step_training:
                    orog = inp[:, -2:-1]
                if 'residual_field' in self.params.target:
                    tar -= inp[:, 0:tar.size()[1]]

            if self.params.two_step_training:
                gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
                loss_step_one = self.loss_obj(gen_step_one, tar[:, 0:self.params.N_out_channels])

                if self.params.orography:
                    gen_step_two = self.model(torch.cat((gen_step_one, orog), axis=1)).to(self.device,
                                                                                          dtype=torch.float)
                else:
                    gen_step_two = self.model(gen_step_one).to(self.device, dtype=torch.float)

                loss_step_two = self.loss_obj(gen_step_two,
                                              tar[:, self.params.N_out_channels:2 * self.params.N_out_channels])
                valid_loss[i] = loss_step_one + loss_step_two
                valid_l1[i] = nn.functional.l1_loss(gen_step_one, tar[:, 0:self.params.N_out_channels])
            else:
                gen = self.model(inp)
                tar, gen = self.compute_and_apply_mask(target=tar, prediction=gen,
                                                       exclude_locations=params.exclude_locations)
                # += does not fail because the array is zero-initialized
                valid_loss[i] += self.loss_obj(gen, tar)
                valid_l1[i] += nn.functional.l1_loss(gen, tar)

            if self.params.two_step_training:
                for c in range(self.params.N_out_channels):
                    if 'residual_field' in self.params.target:
                        valid_weighted_rmse[i, c] = weighted_rmse_torch((gen_step_one[0, c] + inp[0, c]),
                                                                        (tar[0, c] + inp[0, c]), self.device)
                    else:
                        valid_weighted_rmse[i, c] = weighted_rmse_torch(gen_step_one[0, c], tar[0, c], self.device)
            else:
                for c in range(self.params.N_out_channels):
                    if 'residual_field' in self.params.target:
                        valid_weighted_rmse[i, c] = weighted_rmse_torch((gen[0, c] + inp[0, c]),
                                                                        (tar[0, c] + inp[0, c]), self.device)
                    else:
                        valid_weighted_rmse[i, c] = weighted_rmse_torch(gen[0, c], tar[0, c], self.device)

                        # un-normalize
            if not self.ndvi:
                # only un-normalize if not ndvi, since NDVI naturally is in range -1 to 1
                valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse, axis=0).to(self.device)

        return valid_weighted_rmse, valid_loss, valid_l1

    def load_model_wind(self, model_path):
        if self.params.log_to_screen:
            logging.info('Loading the wind model weights from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))
        if dist.is_initialized():
            self.model_wind.load_state_dict(checkpoint['model_state'])
        else:
            new_model_state = OrderedDict()
            model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
            for key in checkpoint[model_key].keys():
                if 'module.' in key:  # model was stored using ddp which prepends module
                    name = str(key[7:])
                    new_model_state[name] = checkpoint[model_key][key]
                else:
                    new_model_state[key] = checkpoint[model_key][key]
            self.model_wind.load_state_dict(new_model_state)
            self.model_wind.eval()

    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        if dist.is_initialized():
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)
        else:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)
        torch.save(model, str(checkpoint_path).replace(".tar", ".pt"))  # save as .pt for inference

    def save_to_wandb(self):
        if self.params.log_to_wandb:
            if dist.is_initialized():
                if self.world_rank == 0:
                    logging.info("Saving final model to W&B")
                    wandb.save(params['best_checkpoint_path'].replace(".tar", ".pt"))  # save as .pt for inference
            else:
                logging.info("Saving final model to W&B")
                wandb.save(params['best_checkpoint_path'].replace(".tar", ".pt"))  # save as .pt for inference

    def load_pretrained_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        try:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                if name != 'ged':
                    new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        except:
            self.model.load_state_dict(checkpoint['model_state'])

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        self.model.load_state_dict(checkpoint['model_state'])
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        if self.params.resuming:  # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='afno_backbone', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--ndvi-finetune", action='store_true')
    parser.add_argument("--ndvi", action='store_true')
    parser.add_argument("--freeze_params", action='store_true')
    parser.add_argument("--roll", action='store_true')
    parser.add_argument("--add_noise", action='store_true')
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument("--appendix-for-run-name", default="", type=str)
    parser.add_argument("--n-blocks-to-freeze", default=0, type=int)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config,
                     print_params=(not 'WORLD_SIZE' in os.environ or int(os.environ["LOCAL_RANK"] == 0)))
    params['epsilon_factor'] = args.epsilon_factor
    params['ndvi_finetune'] = args.ndvi_finetune
    params['ndvi'] = args.ndvi or args.ndvi_finetune
    params['ndvi_data'] = params['ndvi']
    params['cluster'] = args.cluster
    params['freeze_params'] = args.freeze_params
    params['roll'] = args.roll  # whether to roll the input data, overrides the config file; added here for convenience
    params[
        'add_noise'] = args.add_noise  # whether to add noise to the input data, overrides the config file; added here for convenience

    params['max_epochs'] = int(args.epochs) if int(args.epochs) > 0 else params['max_epochs']

    if params.train_years is not None:
        params['train_years'] = params.train_years
    else:
        params['train_years'] = None

    if params.valid_years is not None:
        params['valid_years'] = params.valid_years
    else:
        params['valid_years'] = None

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])

    world_rank = 0
    local_rank = 0
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])
        args.gpu = local_rank
        world_rank = dist.get_rank()
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size // params['world_size'])

    # set cuda device
    if args.cluster:
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(-1)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    # Do not comment this line out please:
    args.resuming = True if os.path.isfile(params.checkpoint_path) else False
    args.resuming = False if args.ndvi_finetune else args.resuming

    params['resuming'] = args.resuming
    params['local_rank'] = local_rank
    params['enable_amp'] = args.enable_amp

    # this will be the wandb name
    #  params['name'] = args.config + '_' + str(args.run_num)
    #  params['group'] = "era5_wind" + args.config
    # params['name'] = args.config + '_' + str(args.run_num)
    params['name'] = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    if args.appendix_for_run_name != "":
        params['name'] = params['name'] + '_' + args.appendix_for_run_name
    params['group'] = "ndvi" + args.config
    params['project'] = ""
    params['entity'] = ""
    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) + 1
    else:
        params['N_in_channels'] = len(params['in_channels'])

    params['N_out_channels'] = len(params['out_channels'])

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)
