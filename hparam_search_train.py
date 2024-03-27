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
import sys
import textwrap
import time
import traceback

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import SimpleCNNwHead
from utils.img_utils import vis_precip
import wandb
from utils.weighted_acc_rmse import weighted_rmse_torch, unlog_tp_torch, weighted_acc_torch
from utils.darcy_loss import LpLoss
import matplotlib.pyplot as plt

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict


class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

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

        self.model = SimpleCNNwHead(in_channels=20, out_channels=1, filters=params.filters,
                                    kernel_sizes=params.kernel_sizes)

        self.model = self.model.to(self.device)

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

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        hparam_target = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
            #        self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            # on every 10th epoch, run validation
            valid_time, valid_logs = self.validate_one_epoch()

            if self.params.log_to_wandb:
                hparam_target = valid_logs["valid_rmse_ndvi"]

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info(
                        "Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")

                    return hparam_target
            elif self.params.scheduler == "CyclicLR":
                if self.epoch >= self.params.max_epochs:
                    logging.info(
                        "Terminating training after reaching params.max_epochs while LR scheduler is set to CyclicLR")
                    return hparam_target

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

        return hparam_target

    #
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

            if exclude_locations:

                if self.geo_mask is None:
                    # create the geo_mask:
                    self.geo_mask = self.create_geo_mask(target)

                target = target * self.geo_mask
                prediction = prediction * self.geo_mask

            target = target * mask
            prediction = prediction * mask

        return target, prediction

    def create_geo_mask(self, target, size: int = 5):
        """
        Creates a mask for the target tensor, excluding test locations
        """
        coordinates = {"burkina": [311, 8], "random": [205, 180], "brasil": [481, 1216], "chile": [489, 1157],
                       "usa_coast": [250, 985], "australia": [437, 550], "wuerzburg": [160, 39],
                       "steigerwald": [160, 42],
                       "bayreuth": [160, 46], "poland": [149, 95]}

        print("Creating geo mask for target tensor")
        print(f"Target tensor shape: {target.shape}")

        mask = torch.ones_like(target, device=self.device, dtype=torch.bool)
        for location in coordinates.keys():
            lat, long = coordinates[location]
            for i in range(-size, size):
                for j in range(-size, size):
                    mask[:, :, lat + i, long + j] = 0

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


def update_params_with_sweep_values():
    cfg = wandb.config
    logging.info("Rank %d updating params with sweep values" % world_rank)

    n_filters_l1 = cfg.n_filters_l1
    n_filters_l2 = cfg.n_filters_l2
    n_filters_l3 = cfg.n_filters_l3
    n_filters_l4 = cfg.n_filters_l4
    n_filters_l5 = cfg.n_filters_l5
    n_filters_l6 = cfg.n_filters_l6
    n_filters_l7 = cfg.n_filters_l7
    n_filters_l8 = cfg.n_filters_l8

    kernel_size_l1 = cfg.kernel_size_l1
    kernel_size_l2 = cfg.kernel_size_l2
    kernel_size_l3 = cfg.kernel_size_l3
    kernel_size_l4 = cfg.kernel_size_l4
    kernel_size_l5 = cfg.kernel_size_l5
    kernel_size_l6 = cfg.kernel_size_l6
    kernel_size_l7 = cfg.kernel_size_l7
    kernel_size_l8 = cfg.kernel_size_l8

    filters = [n_filters_l1, n_filters_l2, n_filters_l3, n_filters_l4, n_filters_l5, n_filters_l6, n_filters_l7,
               n_filters_l8]
    kernel_sizes = [kernel_size_l1, kernel_size_l2, kernel_size_l3, kernel_size_l4, kernel_size_l5, kernel_size_l6,
                    kernel_size_l7, kernel_size_l8]

    params.n_layers = cfg.n_layers

    params.filters = filters[:params.n_layers]
    params.kernel_sizes = kernel_sizes[:params.n_layers]

    params.lr = cfg.lr
    params.max_epochs = cfg.epochs


def load_sweep_values():
    logging.info("Loading param values for rank %d" % world_rank)
    param_dictionary = np.load(os.path.join(params["experiment_dir"], f"sweep_values_{world_rank}.npy"),
                               allow_pickle=True).item()
    params.patch_size = param_dictionary["patch_size"]
    params.embed_dim = param_dictionary["embed_dim"]
    params.depth = param_dictionary["depth"]
    params.lr = param_dictionary["lr"]
    params.drop_rate = param_dictionary["drop_rate"]
    params.num_blocks = param_dictionary["num_blocks"]
    params.max_epochs = param_dictionary["epochs"]
    params.mlp_ratio = param_dictionary["mlp_ratio"]
    params.sparsity_threshold = param_dictionary["sparsity_threshold"]
    params.hard_thresholding_fraction = param_dictionary["hard_thresholding_fraction"]
    logging.info("Loaded param values for rank %d" % world_rank)


def runner_sweep():
    wandb.init(config=params, name=params.name, group=params.group, project=params.project,
               entity=params.entity)
    update_params_with_sweep_values()

    try:
        main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)


def main():
    trainer = Trainer(params, world_rank)

    optim_target = trainer.train()

    # optim target is only a valid metric if it comes from from the main rank!
    if params.log_to_wandb:
        wandb.log({"optim_target": optim_target})


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
    parser.add_argument("--sweep-id", default=None, type=str)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config,
                     print_params=(not 'WORLD_SIZE' in os.environ or int(os.environ["LOCAL_RANK"] == 0)))
    params['epsilon_factor'] = args.epsilon_factor
    params['ndvi_finetune'] = args.ndvi_finetune
    params['ndvi'] = args.ndvi or args.ndvi_finetune
    params['ndvi_data'] = params['ndvi']
    params['cluster'] = args.cluster
    params['freeze_params'] = args.freeze_params

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
    params["name"] = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    params['group'] = "sweep"
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

    if params.log_to_wandb:
        wandb.agent(args.sweep_id, runner_sweep, count=1, project=params.project, entity=params.entity)

    logging.info('DONE ---- rank %d' % world_rank)
