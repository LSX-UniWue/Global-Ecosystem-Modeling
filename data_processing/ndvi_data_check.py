import argparse
import glob
import os
import tempfile
import zipfile
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt


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


def open_daily_files(zip_file: str):
    data = None
    # Create a temporary directory to store the extracted files.
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_file:

            file_list = sorted(zip_file.namelist(),
                               key=custom_sort_key)  # sort by date, January 1st to December 31st
            for fname in file_list[1:365:30]:
                # Extract the file to the temporary directory.
                zip_file.extract(fname, tmp_dir)

        # Get a list of extracted file paths.
        extracted_files = glob.glob(os.path.join(tmp_dir, '**/*.nc'), recursive=True)

        try:
            # Open the extracted files with xarray.
            dataset = xr.open_mfdataset(extracted_files, combine='by_coords')
            # dataset.fillna(value={'NDVI': -2})
            target_lat = xr.DataArray(data=np.arange(90.0, -90., -0.25), dims='lat', name='lat')
            target_lon = xr.DataArray(data=np.arange(-180.0, 180., 0.25), dims='lon', name='lon')
            # Regrid the data using bilinear interpolation
            data = dataset.interp(latitude=target_lat, longitude=target_lon, method='linear')

            # data = data['NDVI'].fillna(-2)  # https://www.streambatch.io/knowledge/ndvi-from-first-principles
            # use missing value of -2, which is outside the range of possible NDVI values (-1 to 1)
            data = data['NDVI'].to_numpy()
            dataset.close()
        except Exception as e:
            print(e)

    return data


def get_out_of_range_value(data):
    # check

    idx = np.argwhere((data < -1) | (data > 1))

    return idx


def count_nans(arr):
    # Step 2: Create a Boolean mask for NaN values
    nan_mask = np.isnan(arr)

    # Step 3: Count the number of NaN values
    nan_count = np.sum(nan_mask)

    # Step 4: Calculate the percentage of NaN values
    total_elements = arr.size
    nan_percentage = (nan_count / total_elements) * 100
    return nan_count, nan_percentage


def plot_distribution(data, name):
    n_bins = 100
    plt.hist(data, bins=n_bins)  # You can adjust the number of bins as needed
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # set x range
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 3e7)
    plt.title(f'Histogram of Data (Ignoring NaN), {n_bins} bins\n{name}')
    plt.savefig(os.path.join(args.out_dir, f"{name}_histogram.svg"))
    plt.close()


#
def main(args):
    statistics_df = pd.DataFrame(
        columns=["File", "Shape", "Max", "Min", "Mean", "Median", "99.9th Percentile Range", "NaN count", "NaN %"])

    zip_files = glob.glob(os.path.join(args.ndvi_dir, "*.zip"))
    zip_files = sorted(zip_files)

    all_data = []
    for zf in zip_files:
        try:
            data = open_daily_files(zf)
            all_data.append(data.flatten())
            percentile = 99.9
            lower_bound = np.nanpercentile(data, (100 - percentile) / 2)
            upper_bound = np.nanpercentile(data, 100 - (100 - percentile) / 2)
            nan_count, nan_percentage = count_nans(data)
            file_stats = {
                "File": zf,
                "Max": np.nanmax(data),
                "Min": np.nanmin(data),
                "Mean": np.nanmean(data),
                "Median": np.nanmedian(data),
                "99.9th Percentile Range": (lower_bound, upper_bound),
                "NaN count": nan_count,
                "Nan %": nan_percentage,
            }
            statistics_df = statistics_df._append(file_stats, ignore_index=True)
            flat_data = data[~np.isnan(data)]
            plot_distribution(flat_data, os.path.basename(zf))
        except Exception as e:
            print(e)
        print("Finished processing", zf)
        # Save the statistics to a CSV file
        statistics_df.to_csv(os.path.join(args.out_dir, "ndvi_data_stats.csv"), index=False, )

    filtered_data = [x[~np.isnan(x)] for x in all_data]

    np.save(os.path.join(args.out_dir, "all_data.npy"), np.array(filtered_data))
    statistics_df.to_csv(os.path.join(args.out_dir, "ndvi_data_stats.csv"), index=False)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--ndvi-dir", type=str, required=True)
    argument_parser.add_argument("--out-dir", type=str, required=True)
    args = argument_parser.parse_args()

    main(args)
