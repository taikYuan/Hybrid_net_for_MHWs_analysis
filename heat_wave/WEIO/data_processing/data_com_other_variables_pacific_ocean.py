# 数据合并
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from netCDF4 import Dataset, num2date
import netCDF4 as nc
import time
import datetime

path = r'H:\radiction_0.25\WEIO\surface_net_thermal_radiation.nc'
nc0 = Dataset(path)
time0 = nc0.variables['time'][:]  # 815232
time0 = time0.data
time0 = ((time0 - 12) / 24) - 33967
print(time0)

# print(nc0.variables)  # time, latitude, longitude  sshf
#
lon = nc0.variables['longitude'][:].data
lat = nc0.variables['latitude'][:].data  # current shape = (94,)

lon0 = lon[:]  # -30 -29.75 ... -10.25  -10   经度
lat0 = lat[:]  #  -30  -30.25 ...  -39.75 -40  纬度
# print(lon0.shape)
print(lat0)
# print(lat0.shape)
# print(lat0)


# # # lat lon
sshf0 = nc0.variables['str'][:9861,:,:].data # 09/01/01 到 ... 19/12/31
print(sshf0.shape) # (3287, 41, 201)
# dlwrf0.reshape(-1, 1)
# dlwrf0 = dlwrf0.data

np.savez(r'D:\heat_wave\WEIO\last\EC_str_93_19_WEIO_ocean_area.npz', time=time0,lat=lat0, lon=lon0, str = sshf0)

