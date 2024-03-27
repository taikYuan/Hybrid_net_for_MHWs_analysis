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

path = r'G:\data\SST\SSTA---NOAA\sst.day.anom.19822.nc'
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
# lon00 = lon[420:495] #105.125 ---123.625
# lat00 = lat[360:457] #  0.125 0.375 ... 23.875 24.125
# lon00 = lon[400:481] # 100.125 ... 120.125
# print(lat00)
# print(lat00)
# lat lon
# sst0 = nc0.variables['sst'][:,360:457,400:481]
anom0 = nc0.variables['anom'][:,:,:]
# print(sst0.shape) # 365, 720, 1440)
# #
# # # # #
anom0.reshape(-1, 1)
anom0 = anom0.data
#
#
# #
time0 = time0
# # #
todo1 = r'G:\data\SST\SSTA---NOAA\*.nc'
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
    anom1 = nc1.variables['anom'][:,:,:]

    anom1.reshape(-1,1)



    time0 = np.concatenate((time0, time1), axis=0)
    #     print(time0.shape)
    anom0 = np.concatenate((anom0, anom1), axis=0)  # (13271, 1, 1)

    print(time0.shape)
    print(anom0.shape)

    time = list(time0)
    #     print(time)
    for i in range(len(time) - 1):
        if time[i + 1] - time[i] != 1:
            print('错误')
            break

    np.savez(r'H:\SST_82_21_expand_all_area.npz', time=time0, lat=lat, lon=lon, anom = anom0,
             )

