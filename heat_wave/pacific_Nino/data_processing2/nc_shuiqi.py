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

path =r'D:\heat_wave\pacific\last\low_cloud\potential_evaporation.nc'
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
lat = nc0.variables['latitude'][12:53] # current shape = (65,)   # 8  7.75 7.5 ... -7.5 7.75 -8
print(lat.shape)
print(lat)

potential_evaporation0 = nc0.variables['pev'][:,12:53,:]

potential_evaporation0 = potential_evaporation0.data

print(potential_evaporation0.shape)

# #对维度倒叙
list3 = []
for i in range(9861):
        for j in range(41):
            for k in range(201):
                list3.append(potential_evaporation0[i, -j-1, k])


potential_evaporation00 = np.array(list3)
print(potential_evaporation00.shape) #(2410650, 1)
potential_evaporation00 = potential_evaporation00.reshape(9861,41,201)
# print(sst00)
print(np.mean(potential_evaporation00))
print(np.max(potential_evaporation00))
print(np.min(potential_evaporation00))


#

np.savez(r'D:\heat_wave\pacific\last\low_cloud\potential_evaporation_pacific_chazhi', lat = lat, lon = lon,
         potential_evaporation = potential_evaporation00 )


