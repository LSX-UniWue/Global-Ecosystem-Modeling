import matplotlib.pyplot as plt
import numpy as np

import xarray as xr


def create_eval_mask(land_sea_mask, land_cover_types):
    land_sea_mask = land_sea_mask[:720, :1440]

    # create mask for evaluation
    # pixels with more than 20% water are dropped to exclude coastal areas and seas
    land_sea_mask[land_sea_mask < 0.8] = 0.0
    land_sea_mask[land_sea_mask > 0.] = 1.0
    # map to integer
    land_sea_mask = np.round(land_sea_mask).astype(int)
    # map to boolean
    land_sea_mask = land_sea_mask.astype(bool)

    # and such with more than 50% barren were removed to exclude deserts
    land_cover_types = land_cover_types[:720, :1440]
    land_cover_types[land_cover_types >= 200] = np.nan
    # see https://datastore.copernicus-climate.eu/documents/satellite-land-cover/D5.3.1_PUGS_ICDR_LC_v2.1.x_PRODUCTS_v1.1.pdf , page 15: 201, 2003, 2002 is bare land (desert)

    # replace non-nans with 1
    land_cover_types[~np.isnan(land_cover_types)] = 1
    # replace nans with 0
    land_cover_types[np.isnan(land_cover_types)] = 0
    # map to integer
    land_cover_types = land_cover_types.astype(int)
    # map to boolean
    land_cover_types = land_cover_types.astype(bool)

    # remove high altitude regions (see text and figure 5 in paper)
    high_latitude_mask = np.ones_like(land_sea_mask)
    high_latitude_mask[:75, :] = 0
    plt.imshow(high_latitude_mask)
    plt.colorbar()
    plt.show()

    # map to boolean
    high_latitude_mask = high_latitude_mask.astype(bool)

    # combine masks
    eval_mask = np.logical_and(land_sea_mask, land_cover_types)
    eval_mask = np.logical_and(eval_mask, high_latitude_mask)

    print(eval_mask.shape)

    return eval_mask


def create_low_res_lulc_mask(path_to_lulc_mask):
    # data can be downloaded from here:
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover?tab=overview

    lulc_mask = xr.open_dataset(path_to_lulc_mask)

    # reshaping using xarray because cdo failed due to large data size (is 300m resolution!)
    target_lat = xr.DataArray(data=np.arange(90.0, -90., -0.25), dims='lat', name='lat')
    target_lon = xr.DataArray(data=np.arange(-180.0, 180., 0.25), dims='lon', name='lon')

    data = lulc_mask.interp(lat=target_lat, lon=target_lon, method='linear')
    data = data.roll(lon=720, roll_coords=True)  # makes alignment easier

    land_cover_types = data.variables["lccs_class"][0].to_numpy()

    np.save("land_cover_types.npy", land_cover_types)


land_sea_mask = np.load("/Users/pascal/Downloads/land_sea_mask.npy")
land_cover_type_mask = np.load("/Users/pascal/PycharmProjects/FourCastNet/jupyter/land_cover_types.npy")

eval_mask = create_eval_mask(land_sea_mask, land_cover_type_mask)
plt.imshow(eval_mask)
plt.show()
np.save("eval_mask.npy", eval_mask)
