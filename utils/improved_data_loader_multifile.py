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

import logging
import glob
import time

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import h5py
from utils.img_utils import reshape_fields, reshape_precip

# Used for geographical train/test split
coordinates = {"burkina": [311, 8], "random": [205, 180], "brasil": [481, 1216], "chile": [489, 1157],
               "usa_coast": [250, 985], "australia": [437, 550], "wuerzburg": [160, 39], "steigerwald": [160, 42],
               "bayreuth": [160, 46], "poland": [149, 95]}
geo_mask = None


def compute_mask(target, prediction, device, ndvi_finetune: bool = False, exclude_locations: bool = False):
    """Computes and applies a mask to the target and prediction tensors if working with ndvi data
    Otherwise, returns the original tensors
    """
    if ndvi_finetune:
        mask = torch.ones_like(target, device=device, dtype=torch.float)
        mask = torch.logical_and(mask, target >= -1)  # Set to 0 if below -1
        mask = torch.logical_and(mask, target <= 1)  # Set to 0 if above 1

        if exclude_locations:
            global geo_mask

            if geo_mask is None:
                # create the geo_mask:
                geo_mask = create_geo_mask(target)

            target = target * geo_mask
            prediction = prediction * geo_mask

        target = target * mask
        prediction = prediction * mask

    return target, prediction


def create_geo_mask(self, target, size: int = 5):
    """
    Creates a mask for the target tensor, excluding test locations
    """
    coordinates = {"burkina": [311, 8], "random": [205, 180], "brasil": [481, 1216], "chile": [489, 1157],
                   "usa_coast": [250, 985], "australia": [437, 550], "wuerzburg": [160, 39], "steigerwald": [160, 42],
                   "bayreuth": [160, 46], "poland": [149, 95]}

    mask = torch.ones_like(target, device=self.device, dtype=torch.bool)
    for location in coordinates.keys():
        lat, long = coordinates[location]
        for i in range(-size, size):
            for j in range(-size, size):
                mask[:, lat + i, long + j] = 0

    return mask


def get_data_loader(params, files_pattern, distributed, train, years=None):
    dataset = GetDataset(params, files_pattern, train, years)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    valid_sampler = DistributedSampler(dataset, shuffle=False) if distributed else None

    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,  # (sampler is None),
                            sampler=sampler if train else valid_sampler,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, location, train, years=None):
        self.params = params
        self.location = location
        self.train = train
        self.years = years
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.roll = params.roll
        self._get_files_stats()
        self.two_step_training = params.two_step_training
        self.orography = params.orography
        self.precip = True if "precip" in params else False
        self.add_noise = params.add_noise if train else False
        self.ndvi_data = True if params.ndvi_data else False

        # Check if we are predicting ndvi as well as other variables
        self.ndvi_multi = True if self.ndvi_data and len(self.out_channels) > 1 else False

        # if so, we need to know the index of the ndvi channel in the out_channels list for further steps
        if self.ndvi_multi:
            print("----- Recognized the use of ndvi and other variables as output channels: -----")
            print("----- Loading NDVI from current timestamp and other variables from next timestamp -----")
            self.ndvi_channel_index = self.params.channel_names.index('ndvi')
            self.insert_position = np.where(self.out_channels == self.ndvi_channel_index)[0]

        if self.precip:
            path = params.precip + '/train' if train else params.precip + '/test'
            self.precip_paths = glob.glob(path + "/*.h5")
            self.precip_paths.sort()

        try:
            self.normalize = params.normalize
        except:
            self.normalize = True  # by default turn on normalization if not specified in config

        if self.orography:
            self.orography_path = params.orography_path

        self.means = np.load(params.global_means_path)
        self.stds = np.load(params.global_stds_path)

        self.data_cache = {}  # Initialize an empty cache

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/**/*.h5", recursive=True)
        self.files_paths.sort(key=lambda s: list(map(int, s[:-3].split('/')[-1].split('_'))))

        if self.years is not None:
            print("----- Loading data only from years: {} -----".format(self.years))
            self.files_paths = list(filter(lambda x: int(x.split('/')[-1][:4]) in self.years, self.files_paths))
            print("----- Found {} files -----".format(len(self.files_paths)))

        self.n_files = len(self.files_paths)
        self.samples_per_file: list = []
        self.cum_samples_per_file: list = []

        logging.info("Getting file stats from all the files")
        for idx, file in enumerate(self.files_paths, 0):
            with h5py.File(self.files_paths[idx], 'r') as _f:
                self.samples_per_file.append(_f['fields'].shape[0])

                if idx == 0:
                    # original image shape (before padding)
                    self.img_shape_x = _f['fields'].shape[2]  # just get rid of one of the pixels
                    self.img_shape_y = _f['fields'].shape[3]

                else:
                    continue

        self.cum_samples_per_file = np.cumsum(self.samples_per_file)
        self.n_samples_total = sum(self.samples_per_file)
        self.files = [None for _ in range(self.n_files)]
        self.precip_files = [None for _ in range(self.n_files)]
        logging.info("Average Number of samples per file/year: {}".format(np.mean(self.samples_per_file)))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location,
                                                                                                       self.n_samples_total,
                                                                                                       self.img_shape_x,
                                                                                                       self.img_shape_y,
                                                                                                       self.n_in_channels))
        logging.info("Delta t: {} hours".format(6 * self.dt))
        logging.info("Including {} hours of past history in training at a frequency of {} hours".format(
            6 * self.dt * self.n_history, 6 * self.dt))

    def __len__(self):
        return self.n_samples_total

    def load_data_into_memory(self):
        # Load the entire dataset into memory once
        self.data = []
        for file_path in self.files_paths:
            with h5py.File(file_path, 'r') as file:
                self.data.append(file['fields'][:])

    def __getitem__(self, global_idx):

        file_idx = np.searchsorted(self.cum_samples_per_file, global_idx, side='right')
        local_idx = global_idx - (0 if file_idx == 0 else self.cum_samples_per_file[file_idx - 1])

        # if we are not at least self.dt*n_history timesteps into the prediction
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # Check if data is in the cache
        cache_key = (file_idx, local_idx)
        if cache_key in self.data_cache:
            data_subset = self.data_cache[cache_key]
        else:
            # Load data from disk since it's not in the cache
            with h5py.File(self.files_paths[file_idx], 'r') as file:
                data_subset = file['fields'][local_idx - self.dt * self.n_history:local_idx + 1:self.dt]

            # Store data in the cache
            self.data_cache[cache_key] = data_subset

        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0

        if self.orography:
            orog = self.orography_field[0:720]
        else:
            orog = None

        if self.train and (self.crop_size_x or self.crop_size_y):
            rnd_x = random.randint(0, self.img_shape_x - self.crop_size_x)
            rnd_y = random.randint(0, self.img_shape_y - self.crop_size_y)
        else:
            rnd_x = 0
            rnd_y = 0

        inp_img = data_subset[:, self.in_channels]
        out_img = data_subset[:, self.out_channels]

        try:
            return reshape_fields(
                img=inp_img, inp_or_tar='inp', crop_size_x=self.crop_size_x, crop_size_y=self.crop_size_y,
                rnd_x=rnd_x, rnd_y=rnd_y, params=self.params, y_roll=y_roll,
                train=self.train, means=self.means, stds=self.stds, normalize=self.normalize, orog=orog,
                add_noise=self.add_noise, ), \
                reshape_fields(img=out_img, inp_or_tar='tar',
                               crop_size_x=self.crop_size_x, crop_size_y=self.crop_size_y,
                               rnd_x=rnd_x, rnd_y=rnd_y, params=self.params, y_roll=y_roll,
                               train=self.train, means=self.means, stds=self.stds,
                               normalize=self.normalize, orog=orog)
        except:
            print("Error in return-statement of getitem")
            print(
                f"local_idx is {local_idx}, Inp_img shape is {inp_img.shape}, out_img shape is {out_img.shape}")
            return None


if __name__ == '__main__':
    data = torch.randn((1, 1, 720, 1440))
    pass
