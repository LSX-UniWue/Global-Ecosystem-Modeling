import shutil
from datetime import datetime
from typing import Tuple

import xarray


def extract_date_from_filename(filename):
    date_str = filename.split('_')[-2]
    return datetime.strptime(date_str, '%Y%m%d')


def get_files_by_date(files):
    # Create a set of all existing dates
    d = {}
    for f in files:
        t = extract_date_from_filename(f).strftime('%Y%m%d')
        d[t] = f
    return d


def fill_missing_date_by_copying(closest_dates, files):
    parts = files[0].split('_')

    files_for_date = get_files_by_date(files)
    for missing_date, closest_date in closest_dates.items():
        # Create a new filename by replacing the date part of the closest date with the missing date
        new_filename = '_'.join(parts[:-2] + [missing_date] + parts[-1:])
        closest_filename = files_for_date[closest_date]

        print(f'Copying {closest_filename} to {new_filename}')
        old_data = xarray.open_dataset(closest_filename)
        new_datetime = datetime.strptime(missing_date, '%Y%m%d')
        old_data["time"] = [new_datetime]
        old_data.to_netcdf(new_filename)

        old_data.close()

        ## use shutil.copyfile instead of os.system
        # shutil.copyfile(closest_filename, new_filename)


def get_year_month_from_file_name(file_name) -> Tuple[int, int]:
    parts = file_name.split('_')
    year = int(parts[-2])
    month = int(parts[-1].split('.')[0])
    return year, month
