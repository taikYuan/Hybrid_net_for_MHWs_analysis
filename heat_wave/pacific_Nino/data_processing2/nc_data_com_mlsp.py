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

path = r'G:\data\mean_sea_lever_anom\mslp.1993.nc'
nc0 = Dataset(path)
# print(nc0.variables)  # time    lat   lon   eastward_wind    northward_wind   wind_vector_divergence    wind_stress
                      # surface_downward_eastward_stress   surface_downward_northward_stress   wind_stress_curl    wind_stress_divergence    wind_speed_rms
                      #  eastward_wind_rms    northward_wind_rms    sampling_length   surface_type   height

lon = nc0.variables['lon']  # current shape = (192,)
# print(lon.shape)#1440
lat = nc0.variables['lat']  # current shape = (94,)

mslp0 = nc0.variables['mslp'][:]




# # # # # #

mslp0 = mslp0.data




# # #
todo1 = r'G:\data\mean_sea_lever_anom\*.nc'
# # # todo1 = todo1[1:]
list_nc = glob.glob(todo1)
list_nc = sorted(list_nc[1:])
# # print(list_nc)
l = len(list_nc)
print(l)  # 36520
list_nc0 = list_nc.copy()
# print(list_nc0)
for i in list_nc0:
    nc_file = i
    nc1 = Dataset(nc_file)

    mslp1 = nc0.variables['mslp'][:]

    mslp1 = mslp1.data



    #     print(time0.shape)
    mslp0 = np.concatenate((mslp0, mslp1), axis=0)  # (13271, 1, 1)

    # print('np.mean(eastward_wind0):{}'.format(np.mean(eastward_wind0)))
    # print(time0.shape)
    # print(sst0.shape)

    # time = list(time0)
    # print(time)
    # for i in range(len(time) - 1):
    #     if time[i + 1] - time[i] != 1:
    #         print('错误')
    #         break
# #
    np.savez(r'D:\heat_wave\sea_level_pressure_93_19_all_area.npz', lat=lat, lon=lon, mslp = mslp0
             )

