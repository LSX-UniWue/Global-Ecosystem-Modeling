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
import argparse
import glob
import shutil
from datetime import datetime, timedelta

# Instructions:
# Set Nimgtot correctly

import h5py
import time
from netCDF4 import Dataset as DS

import zipfile
import os

import numpy as np
import xarray as xr
import tempfile

from utils import fill_missing_date_by_copying, extract_date_from_filename

timesteps = 365  # we loose 7 samples by disregarding years with leap days


def extract_year_from_filename(filename) -> int:
    date_str = filename.split('_')[-2]
    return int(date_str[:4])


def check_if_all_files_exist(files):
    # Create a set of all existing dates
    year = extract_year_from_filename(files[0])
    existing_dates = set(extract_date_from_filename(filename) for filename in files)

    # Define the date range for the year 2016
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    # Generate a list of all dates within the date range
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # Find the missing dates by subtracting the existing dates from all dates
    missing_dates = list(set(all_dates) - existing_dates)

    if len(missing_dates) == 0:
        print('No missing dates found')
        return

    # Sort the missing dates
    missing_dates.sort()

    closest_dates = {}
    for missing_date in missing_dates:
        closest_date = min(existing_dates, key=lambda date: abs((date - missing_date).days))
        closest_dates[missing_date.strftime('%Y%m%d')] = closest_date.strftime('%Y%m%d')

    fill_missing_date_by_copying(closest_dates, files)


def custom_sort_key(filename):
    # Extract the base filename using os.path.basename
    base_filename = os.path.basename(filename)
    # Split the base filename using underscores
    parts = base_filename.split('_')
    # Find the part containing the date (assuming it's in the format YYYYMMDD)
    for part in parts:
        if len(part) == 8 and part.isdigit():
            return part
    # Return a default value if no date is found
    return '00000000'


def append_data(ndvi_data, timesteps):
    data_shape = ndvi_data.shape  # Get the current shape (x, 720, 1440)

    # Calculate how many times to repeat the last entry
    num_repeats = timesteps - data_shape[0]  # We want a total of timesteps entries, so calculate the difference

    # Repeat the last entry along the first axis
    if num_repeats > 0:
        repeated_data = np.repeat(ndvi_data[-1:], num_repeats, axis=0)
        # Now, 'repeated_data' will have shape (num_repeats, 720, 1440)

        # Concatenate 'data' and 'repeated_data' along the first axis to get the final shape (timesteps, 720, 1440)
        final_data = np.concatenate([ndvi_data, repeated_data], axis=0)
    else:
        # If 'num_repeats' is non-positive, no repetition is needed
        final_data = ndvi_data

    return final_data


def open_daily_files(zip_file: str):
    data = None
    # Create a temporary directory to store the extracted files.
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_file:
            file_list = sorted(zip_file.namelist(),
                               key=custom_sort_key)  # sort by date, January 1st to December 31st
            for fname in file_list:
                # Extract the file to the temporary directory.
                zip_file.extract(fname, tmp_dir)

        # Get a list of extracted file paths.
        extracted_files = glob.glob(os.path.join(tmp_dir, '**/*.nc'), recursive=True)
        extracted_files = sorted(extracted_files, key=custom_sort_key)

        check_if_all_files_exist(extracted_files)

        # perform again to get new list of files
        extracted_files = glob.glob(os.path.join(tmp_dir, '**/*.nc'), recursive=True)
        extracted_files = sorted(extracted_files, key=custom_sort_key)
        try:
            # Open the extracted files with xarray.
            dataset = xr.open_mfdataset(extracted_files, combine='by_coords')
            target_lat = xr.DataArray(data=np.arange(90.0, -90., -0.25), dims='lat', name='lat')
            target_lon = xr.DataArray(data=np.arange(-180.0, 180., 0.25), dims='lon', name='lon')
            # Regrid the data using bilinear interpolation
            data = dataset.interp(latitude=target_lat, longitude=target_lon, method='linear')
            data = data.roll(lon=720, roll_coords=True)  # makes alignment easier

            data = data['NDVI'].fillna(-2)  # https://www.streambatch.io/knowledge/ndvi-from-first-principles
            # use missing value of -2, which is outside the range of possible NDVI values (-1 to 1)
            data = data.to_numpy()
        except Exception as e:
            print(e)

    return data


def writetofile(src, dest, channel_idx, varslist, src_idx=0, frmt='nc'):
    global timesteps
    if os.path.isfile(src) or frmt == 'array':
        batch = 2 ** 4
        rank = 0
        Nproc = 1
        Nimgtot = timesteps  # src_shape[0] #for monthly data

        Nimg = Nimgtot // Nproc
        base = rank * Nimg
        end = (rank + 1) * Nimg if rank < Nproc - 1 else Nimgtot
        idx = base

        for variable_name in varslist:

            if frmt == 'nc':
                fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
            elif frmt == 'h5':
                fsrc = h5py.File(src, 'r')[varslist[0]]
            elif frmt == 'array':
                fsrc = src
            print("fsrc shape", fsrc.shape)
            fdest = h5py.File(dest, 'a', )

            while idx < end:
                if end - idx < batch:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:end, src_idx]
                    else:
                        ims = fsrc[idx:end]
                    # print(ims.shape)
                    ims = ims[0:1440, 0:720, :]
                    # print("ims shape after removing last pixel", ims.shape)
                    fdest['fields'][idx:end, channel_idx, :, :] = ims
                    break
                else:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:idx + batch, src_idx]
                    else:
                        ims = fsrc[idx:idx + batch]
                    # ims = fsrc[idx:idx+batch]
                    # print("ims shape", ims.shape)
                    # following  https://github.com/NVlabs/FourCastNet/blob/93360c1720a9f97aabf970689f21c9fad8737788/utils/img_utils.py#L88C46-L88C46
                    ims = ims[0:1440, 0:720, :]
                    # print("ims shape after removing last pixel", ims.shape)
                    fdest['fields'][idx:idx + batch, channel_idx, :, :] = ims
                    idx += batch

            channel_idx += 1


def main(args):
    global timesteps
    start_total_time = time.time()

    files = glob.glob(args.src + "/*_surface_*.nc")  # get all surface level files
    files.sort()
    timesteps = 365  # we loose 7 samples by disregarding leap days
    ndvi_file_dict = {}

    ndvi_zip_files = glob.glob(os.path.join(args.ndvi_src, "*.zip"))
    for zip_file in ndvi_zip_files:
        # Extract the filename from the path.
        file_name = os.path.basename(zip_file)

        # Assume the filename follows the '1981.zip', '1982.zip', etc., convention.
        # You can adjust the parsing logic based on your actual filenames.
        year = file_name.split('.')[0].split("_")[0]  # Get the part before the first period.
        # Add the file to the dictionary with the year as the key.
        ndvi_file_dict[year] = zip_file

    n_variables = 24  # add an additional channel for ndvi

    for surface_file in files:
        if args.year > 0:
            year = surface_file.split("/")[-1].split(".")[0].split("_")[-1]
            if int(year) != args.year:  # only process files for the given year
                continue

        start_year_time = time.time()
        year = surface_file.split("/")[-1].split(".")[0].split("_")[-1]
        dest = os.path.join(args.dest,
                            year + ".h5")
        with h5py.File(dest, 'w') as f:
            f.create_dataset('fields', shape=(timesteps, n_variables, 720, 1440), dtype='f',
                             maxshape=(None, None, 720, 1440))
        print("dest", dest)
        pl_file = surface_file.replace("surface", "pl")

        daily_ndvi = open_daily_files(ndvi_file_dict[year])
        print("daily ndvi shape before", daily_ndvi.shape)
        daily_ndvi = append_data(daily_ndvi, timesteps)
        print("daily ndvi shape after", daily_ndvi.shape)
        print("daily ndvi min", daily_ndvi.min(), "daily ndvi max", daily_ndvi.max(), "daily ndvi mean",
              daily_ndvi.mean(), )
        writetofile(daily_ndvi, dest, 23, ['ndvi'], frmt='array')

        # u10, v10, t2m
        writetofile(surface_file, dest, 0, ['u10'])  # u component of wind at 10 m
        writetofile(surface_file, dest, 1, ['v10'])  # v component of wind at 10 m
        writetofile(surface_file, dest, 2, ['t2m'])  # 2 meter temperature

        # sp mslp
        writetofile(surface_file, dest, 3, ['sp'])  # surface pressure
        writetofile(surface_file, dest, 4, ['msl'])  # mean sea level pressure

        # t850
        writetofile(pl_file, dest, 5, ['t'], src_idx=2)  # temperature at 850 hPa

        # uvz1000
        writetofile(pl_file, dest, 6, ['u'], src_idx=3)  # u component of wind at 1000 hPa
        writetofile(pl_file, dest, 7, ['v'], src_idx=3)  # v component of wind at 1000 hPa
        writetofile(pl_file, dest, 8, ['z'], src_idx=3)  # geo potential at 1000 hPa

        # uvz 850
        writetofile(pl_file, dest, 9, ['u'], src_idx=2)  # u component of wind at 850 hPa
        writetofile(pl_file, dest, 10, ['v'], src_idx=2)  # v component of wind at 850 hPa
        writetofile(pl_file, dest, 11, ['z'], src_idx=2)  # geo potential at 850 hPa

        # uvz 500
        writetofile(pl_file, dest, 12, ['u'], src_idx=1)  # u component of wind at 500 hPa
        writetofile(pl_file, dest, 13, ['v'], src_idx=1)  # v component of wind at 500 hPa
        writetofile(pl_file, dest, 14, ['z'], src_idx=1)  # geo potential at 500 hPa

        # t 500
        writetofile(pl_file, dest, 15, ['t'], src_idx=1)  # temperature at 500 hPa

        # z 50
        writetofile(pl_file, dest, 16, ['z'], src_idx=0)  # geopotential at 50 hPa

        # r500
        writetofile(pl_file, dest, 17, ['r'], src_idx=1)  # relative humidity at 500 hPa

        # r 850
        writetofile(pl_file, dest, 18, ['r'], src_idx=2)  # relative humidity at 850 hPa

        writetofile(surface_file, dest, 19, ['tcwv'])  # total column water vapour

        # data from Higgins et. al.
        writetofile(surface_file, dest, 20, ['swvl1'])  # volumetric soil water layer 1
        writetofile(surface_file, dest, 21, ['stl1'])  # soil temperature level 1
        writetofile(surface_file, dest, 22, ['ssr'])  # surface net solar radiation

        print("done with file", surface_file)
        print("----------------------------------")
        end_year_time = time.time()
        year_elapsed_time = end_year_time - start_year_time
        print(f"Year {year} took {year_elapsed_time} seconds")

    end_total_time = time.time()
    total_elapsed_time = end_total_time - start_total_time
    print(f"Total Elapsed Time: {total_elapsed_time} seconds")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--src', type=str, default='/Users/pascal/PycharmProjects/FourCastNet/data/raw/daily/')
    args.add_argument('--dest', type=str, default='/Users/pascal/PycharmProjects/FourCastNet/data/preprocessed/')
    args.add_argument("--include-ndvi", action="store_true", default=True)
    args.add_argument("--ndvi-src", type=str, default="/Users/pascal/ndvi/")
    args.add_argument("--year", type=int, default=-1, help="year to process")
    args = args.parse_args()
    main(args)
    # main(args)
