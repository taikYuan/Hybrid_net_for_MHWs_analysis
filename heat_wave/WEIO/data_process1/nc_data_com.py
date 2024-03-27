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

path = r'G:\MLD_temp_eq\1981-2022\SST\sst.day.mean.1982.nc'
nc0 = Dataset(path)
time0 = nc0.variables['time'][:]  # 815232
# print((time0 / 24) - 65377)
time0 = time0.data
time0 = time0  - 66473
print(time0)
lon = nc0.variables['lon']  # current shape = (192,)
# print(lon.shape)#1440
lat = nc0.variables['lat']  # current shape = (94,)
# lon0 = lon[56:67]  # 105.    106.875     108.75    110.625     112.5    114.375     116.25    118.125    120.   121.875    123.75
# lat0 = lat[33:48]  #25.7139   23.8092    21.9044    19.9997    18.095    16.1902    14.2855   12.3808   10.47604   8.57131   6.66657    4.76184    2.8571     0.952368     -0.952368
lon00 = lon[216:265] #  54.125 54.375 ... 59.875  66.125
lat00 = lat[327:393] #  - 8.125 -7.875 ... 7.875 8.125


print(lat00.shape)


# lat11 = lat[216:265]  # -35.875 -35.625 ... -24.125 -23.875  之前分析的expand_WEIO为这个
# lon11 = lon[327:393]  # 81.875  82.125 ... 97.875 98.125
# print(lon11)
# # lat lon
# sst0 = nc0.variables['sst'][:,360:457,400:481]
sst0 = nc0.variables['sst'][:,327:393,216:265]
# print(sst0.shape) # 365, 720, 1440)
# #
# # # # #
sst0.reshape(-1, 1)
sst0 = sst0.data
print(np.mean(sst0))
#
# #
time0 = time0
# # #
todo1 = r'G:\MLD_temp_eq\1981-2022\SST\*.nc'
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
    time1 = nc1.variables['time'][:]
    #     print(time1)
    time1 = time1.data
    time1 = time1  -66473
    print(time1)
#
    sst1 = nc1.variables['sst'][:,327:393,216:265]

    sst1.reshape(-1,1)



    time0 = np.concatenate((time0, time1), axis=0)
    #     print(time0.shape)
    sst0 = np.concatenate((sst0, sst1), axis=0)  # (13271, 1, 1)

    print(time0.shape)
    print(sst0.shape)

    time = list(time0)
    #     print(time)
    for i in range(len(time) - 1):
        if time[i + 1] - time[i] != 1:
            print('错误')
            break

    np.savez(r'D:\heat_wave\WEIO\expand_WEIO\SST_82_21_expand_WEIO_area.npz', time=time0, lat=lat, lon=lon, sst = sst0,
             )

