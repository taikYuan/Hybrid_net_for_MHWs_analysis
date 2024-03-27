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

path =r'D:\heat_wave\WEIO\expand_WEIO\other_variables\total_cloud_cover.nc'
nc0 = Dataset(path)
print(nc0.variables)  # time    lat   lon   eastward_wind    northward_wind   wind_vector_divergence    wind_stress
#                       # surface_downward_eastward_stress   surface_downward_northward_stress   wind_stress_curl    wind_stress_divergence    wind_speed_rms
#                       #  eastward_wind_rms    northward_wind_rms    sampling_length   surface_type   height

time0 = nc0.variables['time'][:]  # 815232
time0 = time0.data
# # print((time0 / 24) - 65377)
# time0 = time0.data
# time0 = time0 / 24 - 33967.5
# print(time0) # [815244.]
lon = nc0.variables['longitude'][:]  # current shape = (192,)  # 54 54.25 54.75 55 ... 65.5 65.75 66
# print(lon.shape)#1440
lat = nc0.variables['latitude'][:] # current shape = (65,)   # 8  7.75 7.5 ... -7.5 7.75 -8
print(lat.shape)
print(lon)

total_cloud_cover0 = nc0.variables['tcc'][:]

total_cloud_cover0 = total_cloud_cover0.data

print(total_cloud_cover0)
# #对维度倒叙
list3 = []
for i in range(9855):
        for j in range(65):
            for k in range(49):
                list3.append(total_cloud_cover0[i, -j-1, k])


total_cloud_cover00 = np.array(list3)
print(total_cloud_cover00.shape) #(2410650, 1)
total_cloud_cover00 = total_cloud_cover00.reshape(9855,65,49)
# print(sst00)
print(np.mean(total_cloud_cover00))
print(np.max(total_cloud_cover00))
print(np.min(total_cloud_cover00))


#

np.savez(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\海气反馈\total_cloud_cover_WEIO_chazhi', lat = lat, lon = lon,
         total_cloud_cover = total_cloud_cover00 )


