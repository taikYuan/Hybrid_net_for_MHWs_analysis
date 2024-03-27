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

path = r'G:\data\wind\shw\wind_12_time\1993\1993010112-IFR-L4-EWSB-BlendedWind-GLO-025-6H-REPv6-20181220T174413-fv1.0.nc'
nc0 = Dataset(path)
# print(nc0.variables)  # time    lat   lon   eastward_wind    northward_wind   wind_vector_divergence    wind_stress
                      # surface_downward_eastward_stress   surface_downward_northward_stress   wind_stress_curl    wind_stress_divergence    wind_speed_rms
                      #  eastward_wind_rms    northward_wind_rms    sampling_length   surface_type   height
time0 = nc0.variables['time'][:]  # 815232
# print((time0 / 24) - 65377)
time0 = time0.data
time0 = time0 / 24 - 33967.5
print(time0) # [815244.]
lon = nc0.variables['lon']  # current shape = (192,)
# print(lon.shape)#1440
lat = nc0.variables['lat']  # current shape = (94,)
# lon0 = lon[56:67]  # 105.    106.875     108.75    110.625     112.5    114.375     116.25    118.125    120.   121.875    123.75
# lat0 = lat[33:48]  #25.7139   23.8092    21.9044    19.9997    18.095    16.1902    14.2855   12.3808   10.47604   8.57131   6.66657    4.76184    2.8571     0.952368     -0.952368
# lon00 = lon[420:495] #105.125 ---123.625
lat00 = lat[360:425] # 0.125 0.375 ... 15.875 16.125
lon00 = lon[540:661] # 135.125  125.375 ... 164.875  165.125 E
print(lon00)
# # print(lat00)
# # lat lon
# sst0 = nc0.variables['sst'][:,360:425,540:661]  # lat  lon
#
# # # # # #
# sst0 = sst0.data
# print(np.mean(sst0)) # 365, 720, 1440)
# #
# # #
# time0 = time0
# # # #
# todo1 = r'G:\MLD_temp_eq\1981-2022\SST\*.nc'
# # # # todo1 = todo1[1:]
# list_nc = glob.glob(todo1)
# list_nc = sorted(list_nc[1:])
# # # print(list_nc)
# l = len(list_nc)
# print(l)  # 36520
# list_nc0 = list_nc.copy()
# # print(list_nc0)
# for i in list_nc0:
#     nc_file = i
#     nc1 = Dataset(nc_file)
#     time1 = nc1.variables['time'][:]
#     #     print(time1)
#     time1 = time1.data
#     time1 = time1  -66473
#     # print(time1)
# #
#     sst1 = nc1.variables['sst'][:,360:425,540:661]
#
#     sst1 = sst1.data
#
#     print('np.mean(sst1):{}'.format(np.mean(sst1)))
#     time0 = np.concatenate((time0, time1), axis=0)
#     #     print(time0.shape)
#     sst0 = np.concatenate((sst0, sst1), axis=0)  # (13271, 1, 1)
#     print('np.mean(sst0):{}'.format(np.mean(sst0)))
#     # print(time0.shape)
#     # print(sst0.shape)
#
#     time = list(time0)
#     #     print(time)
#     for i in range(len(time) - 1):
#         if time[i + 1] - time[i] != 1:
#             print('错误')
#             break
# #
#     # np.savez(r'D:\heat_wave\pacific\SST_82_21_pacific_warm_pool_area.npz', time=time0, lat=lat00, lon=lon00, sst = sst0,
#     #          )
# #
