import numpy as np


def lat_np(j, num_lat):
    return 90 - j * 180 / (num_lat - 1)


def latitude_weighting_factor(j, num_lat):
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))

    return num_lat * np.cos(np.pi / 180. * lat_np(j, num_lat)) / s


def lat_xarray_method(lat):
    return np.cos(np.deg2rad(lat))


matrix = np.zeros((720, 1440))
num_lat = np.shape(matrix)[0]
num_long = np.shape(matrix)[1]
for i in range(num_lat):
    matrix[i, :] = latitude_weighting_factor(i, num_lat)

import matplotlib.pyplot as plt

plt.imshow(matrix)
plt.colorbar()
plt.show()

np.save("latitude_weighting_factor.npy", matrix)
