# 数据合并
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from netCDF4 import Dataset, num2date
import netCDF4 as nc
import time
import datetime

path =r'D:\博士工作\第三篇论文_MHW_AI\data\WEIO\expand_WEIO\other_variables\low_cloud_cover.nc'
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

low_cloud_cover0 = nc0.variables['lcc'][:]

low_cloud_cover0 = low_cloud_cover0.data

print(low_cloud_cover0)
# #对维度倒叙
list3 = []
for i in range(9861):
        for j in range(65):
            for k in range(49):
                list3.append(low_cloud_cover0[i, -j-1, k])


low_cloud_cover00 = np.array(list3)
print(low_cloud_cover00.shape) #(2410650, 1)
low_cloud_cover00 = low_cloud_cover00.reshape(9861,65,49)
# print(sst00)
print(np.mean(low_cloud_cover00))
print(np.max(low_cloud_cover00))
print(np.min(low_cloud_cover00))


#

np.savez(r'D:\博士工作\第三篇论文_MHW_AI\data\WEIO\expand_WEIO\other_variables\海气反馈\low_cloud_cover_WEIO_chazhi', lat = lat, lon = lon,
         low_cloud_cover = low_cloud_cover00 )


