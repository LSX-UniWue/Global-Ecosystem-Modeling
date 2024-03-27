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
import time
from typing import List, Union

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from sklearn.metrics import r2_score
import torch.distributed as dist
from torch.nn.functional import mse_loss, l1_loss

import torch


class GELU(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(input)


torch.nn.modules.activation.GELU = GELU
import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.data_loader_multifiles import get_data_loader, compute_mask
from utils.weighted_acc_rmse import weighted_rmse_torch, weighted_acc_torch

from utils.distributed_utils import setup_distributed
from networks.afnonet import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

DECORRELATION_TIME = 1

coordinates = {"burkina": [311, 8], "random": [205, 180], "brasil": [481, 1216], "chile": [489, 1157],
               "usa_coast": [250, 985], "australia": [437, 550], "wuerzburg": [160, 39], "steigerwald": [160, 42],
               "bayreuth": [160, 46], "poland": [149, 95]}

biomes = {"BF-CAN-BGR": [-106.37, 53.79], "BF-RUS-KRS": [58.62, 61.37], "BF-USA-ALS": [-153.87, 66.37],
          "GR-ARG-BEL": [-70.54, -51.54], "GR-BRA-ENV": [-55.79, -30.38], "GR-NZL-LMM": [169.79, -45.63],
          "GR-TZA-SER": [35.21, -2.79], "MT-CHL-QUE": [-71.28, -31.45], "MT-ZAF-DEH": [24.21, -33.71],
          "RF-AUS-MIR": [143.38, -13.79],
          "RF-CMR-EBO": [10.46, 4.38],
          "RF-COD-ITO": [28.29, -3.71],
          "RF-IDN-BOR": [115.79, 1.87],
          "RF-LKA-THA": [81.37, 6.62],
          "SA-AGO-MAV": [20.88, -15.54],
          "SA-BFA-BNP": [2.13, 12.12],
          "SA-CAF-MMM": [19.79, 8.70],
          "SA-ZMB-LUP": [32.38, -13.04],
          "SH-BRA-CAA": [-38.62, -9.21],
          "SH-BWA-CEN": [23.88, -22.21],
          "SH-MEX-POR": [-113.54, 27.46],
          "SH-SSD-LOE": [34.54, 5.12],
          "SH-USA-LAR": [-115.29, 34.13],
          "TF-CAN-VAN": [-127.54, 50.21],
          "TF-JPN-ECH": [139.30, 37.20],
          "TF-USA-CRA": [-80.29, 38.29],
          "TU-CAN-PIN": [-72.46, 60.54],
          "TU-NOR-FNM": [24.54, 69.71],
          "UNK-GNI-UNK": [93.73, 7.12]}


def lat_long_to_indices(lat, long):
    resolution = 0.25
    # Calculate latitude index
    lat_index = int((90 - lat) / resolution)

    # Calculate longitude index
    lon_index = int(long / resolution)
    if lon_index < 0:
        lon_index += 1440

    return lat_index, lon_index


def group_biomes_by_biometype():
    # use the keys' first two letters to group biomes by type
    biome_types = {}
    for key in biomes.keys():
        biome_type = key[:2]
        if biome_type in biome_types:
            biome_types[biome_type].append(key)
        else:
            biome_types[biome_type] = [key]

    return biome_types


# Higgins used T_air, T_soil, Radiation, M_soil, CO2 as variables
# first ic: change nothing
# second ic: change T_air + 10%
# third ic: model a change of water availability by changing relative humidity across all levels -10%
# fourth ic: model a change of water availability by changing relative humidity across all levels -50%
# naming is `ic_name: {channel: change}`
ic_changes = {0: None}


def find_all_indices_with_valid_data(data):
    # find all indices with missing data, that is, where data is < -1
    indices = np.where(data > -1.)[0]
    return indices


def preprocess_time_series(gt, prediction, subsample: int = 14):
    gt = np.hstack(gt)
    prediction = np.hstack(prediction)

    gt_without_missing_data = find_all_indices_with_valid_data(gt)
    gt_valid = gt[gt_without_missing_data]
    prediction_valid = prediction[gt_without_missing_data]

    if subsample > 1:
        gt_valid = gt_valid[::subsample]
        prediction_valid = prediction[::subsample]

    return gt_valid, prediction_valid, gt, prediction


def compute_rmse_rsquare(ground_truth, prediction, title: str = "NDVI", verbose: bool = True):
    r2_val = r2_score(ground_truth, prediction)
    r2_val = round(r2_val, 4)

    rmse = torch.sqrt(mse_loss(torch.from_numpy(prediction), torch.from_numpy(ground_truth)))
    rmse = round(float(rmse), 4)

    mae = l1_loss(torch.from_numpy(prediction), torch.from_numpy(ground_truth))
    mae = round(float(mae), 4)

    if verbose:
        logging.info(f"{title} RMSE: {rmse}, R2: {r2_val}, MAE: {mae}")

    return rmse, r2_val, mae


def plot_inset(ax, x, y_true, y_pred):
    # Create inset axes with specific width and height
    iax = inset_axes(ax, width="30%", height="30%", loc='lower left')

    # Plot the inset
    iax.plot(x, y_pred, label='predicted')
    iax.plot(x, y_true, label='ground truth', alpha=0.5)

    iax.set_xticks([])
    iax.set_yticks([])
    mark_inset(ax, iax, loc1=2, loc2=1, fc="none", ec="0.5")


def compute_longtime_ts_error_biome(channel_name, data, subsample: int = -1, out_path: str = None, ic: int = 0,
                                    inset_samples: int = 100, xlabel: str = "Time"):
    # first, group the biomes by type
    biome_types = group_biomes_by_biometype()

    # second, create a separate plot for each biome type
    for biome_type, locations in biome_types.items():
        temp_path = os.path.join(out_path, biome_type)
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)
        # get the data for the biomes of this type
        biome_data = {}
        places = []
        for key in locations:
            places.append(key)
            key_name_gt = ch_name + "_" + str(key) + "_gt"
            key_name_pred = ch_name + "_" + str(key) + "_pred"
            biome_data[key_name_gt] = data[key_name_gt]
            biome_data[key_name_pred] = data[key_name_pred]

        _ic = f"{biome_type}_{ic}"
        compute_longtime_ts_error(channel_name, biome_data, subsample=subsample, out_path=temp_path, ic=_ic,
                                  inset_samples=inset_samples, xlabel=xlabel, verbose=False,
                                  places=places)


def compute_longtime_ts_error(channel_name, data, subsample: int = -1, out_path: str = None, ic: Union[str, int] = 0,
                              inset_samples: int = 100, xlabel: str = "Time", verbose: bool = True,
                              places=list(coordinates.keys())):
    n = len(places)
    fig, axs = plt.subplots(n, 1, figsize=(8, 3 * n))
    # Loop through each subplot
    if n == 1:
        axs = [axs]
    subplot_idx = 0

    for place in places:
        prediction = data[f"{channel_name}_{place}_pred"]
        gt = data[f"{channel_name}_{place}_gt"]
        if place == "random":
            place = "iran"
        place = place.capitalize()
        gt_valid, prediction_valid, gt, prediction = preprocess_time_series(gt, prediction, subsample=subsample)

        gt_masked = np.ma.masked_where(gt < -1., gt)
        prediction_masked = np.ma.masked_where(gt < -1., prediction)

        axs[subplot_idx].plot(prediction_masked, label=f'predicted')
        axs[subplot_idx].plot(gt_masked, label=f'ground truth', alpha=0.5)
        if "ndvi" in channel_name:
            axs[subplot_idx].set_ylim(-1.10, 1.10)
        axs[subplot_idx].set_xlabel(f"{xlabel}")
        axs[subplot_idx].set_ylabel(f'{channel_name} value')
        axs[subplot_idx].legend()
        axs[subplot_idx].set_title(f'{place}')
        verbose = verbose and "ndvi" in channel_name
        rmse, r2_val, mae = compute_rmse_rsquare(gt_valid, prediction_valid, title=f"{channel_name} {place}",
                                                 verbose=verbose)

        plot_inset(axs[subplot_idx], np.arange(inset_samples), gt_masked[:inset_samples],
                   y_pred=prediction_masked[:inset_samples])
        metric_text = f"RMSE: {rmse} | R2: {r2_val} | MAE: {mae}"
        axs[subplot_idx].text(0.7, 0.05, metric_text, transform=axs[subplot_idx].transAxes, ha='right', va='bottom',
                              bbox=dict(facecolor='white', alpha=0.8), fontsize=8)
        subplot_idx += 1
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"{channel_name}_longtime_{ic}.svg"))
    plt.close("all")


def load_model(checkpoint_file):
    try:
        model = torch.load(checkpoint_file)
        model.zero_grad()
        model.eval()
    except RuntimeError as e:
        print(e)
        setup_distributed(0, 1)
        model = torch.load(checkpoint_file)
        model.zero_grad()
        model.eval()

    return model


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    # get data loader
    data_loader, dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False,
                                           years=params["years"])
    img_shape_x = dataset.img_shape_x
    img_shape_y = dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    if params["orography"]:
        params['N_in_channels'] = n_in_channels + 1
    else:
        params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[
        0]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0]

    checkpoint_file = params['best_checkpoint_path']
    model = load_model(checkpoint_file)
    model = model.to(device)

    return data_loader, model, len(dataset)


def change_specific_channel(model_input: torch.Tensor, channels_to_change: dict):
    for channel, change in channels_to_change.items():
        model_input[:, channel, :, :] = model_input[:, channel, :, :] * change

    return model_input


def inference_step(inp, model, ic):
    if ic == 0:
        prediction = model(inp)
    else:
        channels_to_change = ic_changes[ic]
        modified_input = change_specific_channel(inp, channels_to_change)
        prediction = model(modified_input)
    return prediction


def inference(params, ic, valid_data_loader, model, timesteps_per_year, years: List[int]):
    ic = int(ic)
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    prediction_length = timesteps_per_year
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    out_channels = np.array(params.out_channels)
    n_out_channels = len(out_channels)
    model_name = params['best_checkpoint_path'].split("/")[-2]

    long_time_series = {}
    for key in coordinates.keys():
        for c in params.out_channels:
            ch_name = params.channel_names[c]
            long_time_series[ch_name + "_" + str(key) + "_gt"] = []
            long_time_series[ch_name + "_" + str(key) + "_pred"] = []

    biome_timeseries = {}
    for key in biomes.keys():
        for c in params.out_channels:
            ch_name = params.channel_names[c]
            biome_timeseries[ch_name + "_" + str(key) + "_gt"] = []
            biome_timeseries[ch_name + "_" + str(key) + "_pred"] = []

    ndvi_timeseries = {}
    ndvi_timeseries['data'] = np.empty((0))

    evaluation_metrics = {}
    for c in params.out_channels:
        ch_name = params.channel_names[c]
        evaluation_metrics[ch_name + "_l1"] = []
        evaluation_metrics[ch_name + "_weighted_rmse"] = []
        evaluation_metrics[ch_name + "_weighted_acc"] = []

    # autoregressive inference
    if params.log_to_screen:
        logging.info('Begin autoregressive inference')

    cur_year = 0
    with torch.no_grad():
        place_loc = 0
        for data in valid_data_loader:
            inp, target = map(lambda x: x.to(device, dtype=torch.float), data)

            prediction = inference_step(inp, model, ic)
            collect_data(long_time_series=long_time_series, ground_truth=target, predictions=prediction,
                         params=params, biome_timeseries=biome_timeseries, ndvi_timeseries=ndvi_timeseries,
                         evaluation_metrics=evaluation_metrics, device=device)

            place_loc += (1 * params['batch_size'])
            if place_loc >= timesteps_per_year:
                save_decade_data(long_time_series=long_time_series, biome_timeseries=biome_timeseries,
                                 ndvi_timeseries=ndvi_timeseries, params=params, ic=ic, cur_year=cur_year + years[0],
                                 evaluation_metrics=evaluation_metrics)

                # reset ndvi timeseries
                ndvi_timeseries['data'] = np.empty((0))

                logging.info(f"Saved data for {cur_year + years[0]}")
                cur_year += 1
                place_loc = 0

        return long_time_series, biome_timeseries, evaluation_metrics


def save_decade_data(long_time_series, biome_timeseries, ndvi_timeseries, params, ic, cur_year, evaluation_metrics):
    out_loc = os.path.join(params['experiment_dir'], f"longtime_ts_initial_condition_{ic}.npy")
    if params.log_to_screen:
        logging.info("Saving files at {}".format(out_loc))
    np.save(out_loc, long_time_series)

    out_loc = os.path.join(params['experiment_dir'], f"biome_timeseries_initial_condition_{ic}.npy")
    if params.log_to_screen:
        logging.info("Saving biome files at {}".format(out_loc))
    np.save(out_loc, biome_timeseries)

    out_loc = os.path.join(params['experiment_dir'], f"metrics_timeseries_initial_condition_{ic}.npy")
    if params.log_to_screen:
        logging.info("Saving metrics files at {}".format(out_loc))
    np.save(out_loc, evaluation_metrics)

    if params['global_ndvi']:
        out_loc = os.path.join(params['experiment_dir'], f"ndvi_timeseries_{cur_year}.npy")
        if params.log_to_screen:
            logging.info("Saving NDVI files at {}".format(out_loc))
        np.save(out_loc, ndvi_timeseries['data'])


def compute_and_apply_mask(target, prediction, device):
    """
    Computes and applies a mask to the target and prediction tensors if working with ndvi data
    Otherwise, returns the original tensors
    """
    mask = torch.ones_like(target, device=device, dtype=torch.float)
    mask = torch.logical_and(mask, target >= -1.)  # Set to 0 if below -1
    mask = torch.logical_and(mask, target <= 1.)  # Set to 0 if above 1

    target = target * mask
    prediction = prediction * mask

    return target, prediction


def collect_data(long_time_series, ground_truth, predictions, params, biome_timeseries, evaluation_metrics,
                 device, ndvi_timeseries=None):
    for idx, c in enumerate(params.out_channels):
        ch_name = params.channel_names[c]

        for key, indices in coordinates.items():
            pred_ts = predictions[:, idx, indices[0], indices[1]]
            gt_ts = ground_truth[:, idx, indices[0], indices[1]]
            pred_ts = pred_ts.cpu().numpy().tolist()
            gt_ts = gt_ts.cpu().numpy().tolist()

            long_time_series[ch_name + "_" + str(key) + "_gt"].append(gt_ts)
            long_time_series[ch_name + "_" + str(key) + "_pred"].append(pred_ts)

    for idx, c in enumerate(params.out_channels):
        ch_name = params.channel_names[c]

        for key, indices in biomes.items():
            lon, lat = indices
            indices = lat_long_to_indices(lat, lon)  # indices is now (lat, lon)
            pred_ts = predictions[:, idx, indices[0], indices[1]]
            gt_ts = ground_truth[:, idx, indices[0], indices[1]]
            pred_ts = pred_ts.cpu().numpy().tolist()
            gt_ts = gt_ts.cpu().numpy().tolist()

            biome_timeseries[ch_name + "_" + str(key) + "_gt"].append(gt_ts)
            biome_timeseries[ch_name + "_" + str(key) + "_pred"].append(pred_ts)

    if params['global_ndvi']:
        ndvi_channel_idx = params.channel_names.index('ndvi')
        output_ndvi_idx = np.where(np.array(params['out_channels']) == ndvi_channel_idx)[0]

        out = np.empty((predictions.shape[0], 2, predictions.shape[2], predictions.shape[3]), dtype=np.float32)
        out[:, 0, :, :] = ground_truth[:, output_ndvi_idx, :, :].cpu().numpy()
        out[:, 1, :, :] = predictions[:, output_ndvi_idx, :, :].cpu().numpy()

        if ndvi_timeseries['data'].shape[1:] != out.shape[1:]:
            ndvi_timeseries['data'] = out

        else:
            ndvi_timeseries['data'] = np.concatenate((ndvi_timeseries['data'], out), axis=0)

    target, prediction = compute_and_apply_mask(ground_truth, predictions, device=device)
    rmse = weighted_rmse_torch(prediction, target)
    acc = weighted_acc_torch(prediction, target)
    for idx, c in enumerate(params.out_channels):
        ch_name = params.channel_names[c]

        l1 = l1_loss(prediction, target)
        l1 = float(l1.cpu())

        weighted_rmse = rmse[idx]
        weighted_rmse = float(weighted_rmse.cpu())

        weighted_acc = acc[idx]
        weighted_acc = float(weighted_acc.cpu())

        evaluation_metrics[ch_name + "_l1"].append(l1)
        evaluation_metrics[ch_name + "_weighted_rmse"].append(weighted_rmse)
        evaluation_metrics[ch_name + "_weighted_acc"].append(weighted_acc)


def visualize_ts(predictions, ground_truth, year: int, ic: int, model_name: str, params):
    for idx, c in enumerate(params.out_channels):
        ch_name = params.channel_names[c]

        n = len(coordinates)
        fig, axs = plt.subplots(n, 1, figsize=(8, 3 * n))

        # Loop through each subplot
        subplot_idx = 0
        for key, indices in coordinates.items():
            pred_ts = predictions[:, idx, indices[0], indices[1]]
            gt_ts = ground_truth[:, idx, indices[0], indices[1]]

            axs[subplot_idx].plot(pred_ts, label=f'predicted')
            axs[subplot_idx].plot(gt_ts, label=f'ground truth')
            if "ndvi" in ch_name:
                axs[subplot_idx].set_ylim(-1.1, 1.1)
            axs[subplot_idx].set_xlabel('Day of year')
            axs[subplot_idx].set_ylabel(f'{ch_name}')
            axs[subplot_idx].legend()
            axs[subplot_idx].set_title(f'{key}')
            subplot_idx += 1

        # Adjust spacing between subplots
        plt.suptitle(f'{ch_name} for {year}')
        plt.tight_layout()
        out_name = os.path.join(args.override_dir,
                                f'{ch_name}_prediction_{year}_initial_condition_{ic}_{model_name}.svg')
        logging.info("Saving figure at {}".format(out_name))
        plt.savefig(out_name)
        plt.close("all")


def count_num_files(years):
    years = sorted(years)

    return years


def find_prediction_files(exp_dir, years):
    data = None
    for year in years:
        file = os.path.join(exp_dir, f"ndvi_timeseries_{year}.npy")
        if os.path.isfile(file):
            logging.info(f"Found data for visualization year {year}")
            if data is None:
                data = np.load(file)
            else:
                data = np.concatenate([data, np.load(file)], axis=0)
        else:
            logging.info(f"No visualization data for year {year}")

    return data


def create_visualizations(data, split: str, out_dir: str):
    # First, a visualization for the entire range
    error = np.mean(np.abs(data[:, 0] - data[:, 1]), axis=0)
    plt.figure(figsize=(20, 8))
    plt.imshow(error, vmin=0, vmax=1, cmap='turbo', extent=[0, 360, -90, 90])
    plt.colorbar()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot longitude and latitude ticks
    plt.xticks(np.arange(0, 361, 100))
    plt.yticks(np.arange(-90, 91, 100))

    # Remove grid
    plt.grid(False)

    plt.savefig(os.path.join(out_dir, f"mean_absolute_error_{split}.svg"))
    plt.close()

    # Now, create a visualization for each month, that is for each 30 frames
    for i in range(0, len(data), 30):
        error = np.mean(np.abs(data[i:i + 30, 0] - data[i:i + 30, 1]), axis=0)
        plt.figure(figsize=(20, 8))
        plt.imshow(error, vmin=0, vmax=1, cmap='turbo', extent=[0, 360, -90, 90])
        plt.colorbar()
        plt.title(f"Mean Absolute Error for {split} - {i // 30}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(False)
        plt.savefig(os.path.join(out_dir, f"mean_absolute_error_{split}_{i // 30}.svg"))
        plt.close()


def create_error_visualizations(exp_dir, years, split):
    data = find_prediction_files(exp_dir, years)

    if data is None:
        logging.info("No visualization data found at all")
        return

    else:
        logging.info(f"Running vizualisation with data shape {data.shape}")
        create_visualizations(data, split, exp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='../config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='afno_backbone_finetune_ndvi', type=str)
    parser.add_argument("--use_daily_climatology", action='store_true')
    parser.add_argument("--override_dir", type=str,
                        help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument("--weights",
                        type=str,
                        help='Path to model weights, for use with override_dir option')
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--global_ndvi", action='store_true')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    params['interp'] = args.interp
    params['use_daily_climatology'] = args.use_daily_climatology
    params['batch_size'] = 1
    params['ndvi_data'] = True
    params['global_ndvi'] = args.global_ndvi

    if args.cluster:
        torch.cuda.set_device(0 if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(-1)
    torch.backends.cudnn.benchmark = True

    dirs = params.inf_data_path.split("/")
    new_inf_dir = "/".join(dirs[:-1])
    params.inf_data_path = new_inf_dir

    for inf_dir in ("valid", "out_of_sample", "train", "all"):
        params["global_ndvi"] = args.global_ndvi  # reset global_ndvi setting
        logging.info("=====================================")
        logging.info("Processing {}".format(inf_dir))

        if inf_dir == "out_of_sample":
            params["years"] = params.out_of_sample_years
        elif inf_dir == "train":
            params["years"] = params.train_years
        elif inf_dir == "valid":
            params["years"] = params.valid_years
        elif inf_dir == "all":
            params["years"] = [1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996,
                               1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                               2012, 2013, 2014, 2015, 2016, 2017, 2018]
        else:
            raise ValueError(f"Unknown inference directory {inf_dir}")

        if args.global_ndvi:  # only compute for val and test years if global_ndvi is set
            if inf_dir in ("valid", "out_of_sample"):
                params["global_ndvi"] = True
            else:
                params["global_ndvi"] = False

        # Set up directory
        if args.override_dir is not None:
            assert args.weights is not None, 'Must set --weights argument if using --override_dir'
            expDir = os.path.join(args.override_dir, "inference", inf_dir)
            params['experiment_dir'] = expDir
        else:
            assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
            expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
            params['experiment_dir'] = os.path.join(os.path.abspath(expDir), "inference", inf_dir)

        if not os.path.isdir(params['experiment_dir']):
            os.makedirs(params['experiment_dir'])

        params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir,
                                                                                                         'training_checkpoints/best_ckpt.pt')

        params['resuming'] = False
        params['ndvi_inference'] = True
        params['local_rank'] = 0

        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
        logging_utils.log_versions()
        params.log()

        n_samples_per_year = 365
        n_inset_samples = 100
        xlabel = "Day of year"
        if "weekly" in params.inf_data_path:
            n_samples_per_year = 52
            n_inset_samples = 156
            xlabel = "Week"
        elif "monthly" in params.inf_data_path:
            n_samples_per_year = 12
            n_inset_samples = 36
            xlabel = "Month"
        elif "daily" in params.inf_data_path:
            n_samples_per_year = 365
            n_inset_samples = 365
            xlabel = "Day"
        elif "hourly" in params.inf_data_path:
            n_samples_per_year = 365 * 4
            n_inset_samples = 365
            xlabel = "Hour"

        ics = range(len(ic_changes))


        def check_if_ic_possible(ic):
            if ic == 0:
                return True
            for channel, change in ic_changes[ic].items():
                if channel not in params.in_channels:
                    print(
                        f"Condition {ic} not possible: channel {channel} is not in input channels -- discarding it")
                    return False
            return True


        # filter out initial conditions that are not possible, because the channel to be modified is not in the input channels
        ics = [ic for ic in ics if check_if_ic_possible(ic)]

        n_ics = len(ics)

        logging.info("Inference for {} initial conditions".format(n_ics))

        # get data and models
        valid_data_loader, model, timesteps = setup(params)

        years = count_num_files(params["years"])

        # run inference for multiple initial conditions
        for i, ic in enumerate(ics):
            logging.info("Initial condition {} of {}".format(i + 1, n_ics))

            longtime_ts, biome_timeseries, evaluation_metrics = inference(params, ic,
                                                                          valid_data_loader, model,
                                                                          timesteps_per_year=n_samples_per_year,
                                                                          years=years)
            np.save(os.path.join(params['experiment_dir'], f"longtime_ts_initial_condition_{ic}.npy"), longtime_ts)
            np.save(os.path.join(params['experiment_dir'], f"biome_timeseries_initial_condition_{ic}.npy"),
                    biome_timeseries)
            np.save(os.path.join(params['experiment_dir'], f"metrics_timeseries_initial_condition_{ic}.npy"),
                    evaluation_metrics)
            for channel in params.out_channels:
                try:
                    ch_name = params.channel_names[channel]
                    compute_longtime_ts_error(channel_name=ch_name, data=longtime_ts, subsample=-1,
                                              out_path=params['experiment_dir'], ic=ic, inset_samples=n_inset_samples,
                                              xlabel=xlabel)
                    compute_longtime_ts_error_biome(channel_name=ch_name, data=biome_timeseries, subsample=-1,
                                                    out_path=params['experiment_dir'], ic=ic,
                                                    inset_samples=n_inset_samples,
                                                    xlabel=xlabel)
                except Exception as e:
                    print(e)
                    print("Could not compute longtime ts error for channel {}".format(channel))
        del biome_timeseries, longtime_ts, valid_data_loader, model

        if params["global_ndvi"]:
            create_error_visualizations(params['experiment_dir'], years=params["years"], split=inf_dir)

        time.sleep(5)
