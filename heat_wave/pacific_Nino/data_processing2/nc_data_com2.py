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

path = r'G:\data\wind\shw\wind_12_time\2019\2019010112-IFR-L4-EWSB-BlendedWind-GLO-025-6H-REPv6-20201012T170326-fv1.0.nc'
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
lon00 = lon[40:241] #  -170 -169.75 ... -120.25  -120
lat00 = lat[300:341] #  - 5 -4.75 ... 4.75  5   # 66

print(lat00)
# # print(lat00)
# # lat lon
eastward_wind0 = nc0.variables['eastward_wind'][:,300:341,40:241]  # lat  lon
northward_wind0 = nc0.variables['northward_wind'][:,300:341,40:241]  # lat  lon
wind_vector_divergence0 = nc0.variables['wind_vector_divergence'][:,300:341,40:241]  # lat  lon
wind_stress0 = nc0.variables['wind_stress'][:,300:341,40:241] # lat  lon
surface_downward_eastward_stress0 = nc0.variables['surface_downward_eastward_stress'][:,300:341,40:241]  # lat  lon
surface_downward_northward_stress0 = nc0.variables['surface_downward_northward_stress'][:,300:341,40:241]  # lat  lon
wind_stress_curl0 = nc0.variables['wind_stress_curl'][:,300:341,40:241]  # lat  lon
wind_stress_divergence0 = nc0.variables['wind_stress_divergence'][:,300:341,40:241] # lat  lon
wind_speed_rms0 = nc0.variables['wind_speed_rms'][:,300:341,40:241]  # lat  lon
eastward_wind_rms0 = nc0.variables['eastward_wind_rms'][:,300:341,40:241]  # lat  lon
northward_wind_rms0 = nc0.variables['northward_wind_rms'][:,300:341,40:241]  # lat  lon
sampling_length0 = nc0.variables['sampling_length'][:,300:341,40:241] # lat  lon
surface_type0 = nc0.variables['surface_type'][:,300:341,40:241]  # lat  lon
# height = nc0.variables['height'][:,287:353,936:985]  # lat  lon



# # # # # #
eastward_wind0 = eastward_wind0.data
northward_wind0 = northward_wind0.data
wind_vector_divergence0 = wind_vector_divergence0.data
wind_stress0 = wind_stress0.data
surface_downward_eastward_stress0 = surface_downward_eastward_stress0.data
surface_downward_northward_stress0 = surface_downward_northward_stress0.data
wind_stress_curl0 = wind_stress_curl0.data
wind_stress_divergence0 = wind_stress_divergence0.data
wind_speed_rms0 = wind_speed_rms0.data
eastward_wind_rms0 = eastward_wind_rms0.data
northward_wind_rms0 = northward_wind_rms0.data
sampling_length0 = sampling_length0.data
surface_type0 = surface_type0.data

# #
# # #
time0 = time0
# # #
todo1 = r'G:\data\wind\shw\wind_12_time\2019\*.nc'
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
    time1 = time1 / 24 - 33967.5
    # print(time1)
#
    eastward_wind1 = nc1.variables['eastward_wind'][:,300:341,40:241]  # lat  lon
    northward_wind1 = nc1.variables['northward_wind'][:,300:341,40:241]  # lat  lon
    wind_vector_divergence1 = nc1.variables['wind_vector_divergence'][:,300:341,40:241]  # lat  lon
    wind_stress1 = nc1.variables['wind_stress'][:,300:341,40:241]  # lat  lon
    surface_downward_eastward_stress1 = nc1.variables['surface_downward_eastward_stress'][:,300:341,40:241]  # lat  lon
    surface_downward_northward_stress1 = nc1.variables['surface_downward_northward_stress'][:,300:341,40:241] # lat  lon
    wind_stress_curl1 = nc1.variables['wind_stress_curl'][:,300:341,40:241]  # lat  lon
    wind_stress_divergence1 = nc1.variables['wind_stress_divergence'][:,300:341,40:241]  # lat  lon
    wind_speed_rms1 = nc1.variables['wind_speed_rms'][:,300:341,40:241] # lat  lon
    eastward_wind_rms1 = nc1.variables['eastward_wind_rms'][:,300:341,40:241]  # lat  lon
    northward_wind_rms1 = nc1.variables['northward_wind_rms'][:,300:341,40:241] # lat  lon
    sampling_length1 = nc1.variables['sampling_length'][:,300:341,40:241]  # lat  lon
    surface_type1 = nc1.variables['surface_type'][:,300:341,40:241]  # lat  lon

    eastward_wind1 = eastward_wind1.data
    northward_wind1 = northward_wind1.data
    wind_vector_divergence1 = wind_vector_divergence1.data
    wind_stress1 = wind_stress1.data
    surface_downward_eastward_stress1 = surface_downward_eastward_stress1.data
    surface_downward_northward_stress1 = surface_downward_northward_stress1.data
    wind_stress_curl1 = wind_stress_curl1.data
    wind_stress_divergence1 = wind_stress_divergence1.data
    wind_speed_rms1 = wind_speed_rms1.data
    eastward_wind_rms1 = eastward_wind_rms1.data
    northward_wind_rms1 = northward_wind_rms1.data
    sampling_length1 = sampling_length1.data
    surface_type1 = surface_type1.data

    # print('np.mean(eastward_wind1):{}'.format(np.mean(eastward_wind1)))
    time0 = np.concatenate((time0, time1), axis=0)
    #     print(time0.shape)
    eastward_wind0 = np.concatenate((eastward_wind0, eastward_wind1), axis=0)  # (13271, 1, 1)
    northward_wind0 = np.concatenate((northward_wind0, northward_wind1), axis=0)  # (13271, 1, 1)
    wind_vector_divergence0 = np.concatenate((wind_vector_divergence0, wind_vector_divergence1), axis=0)  # (13271, 1, 1)
    wind_stress0 = np.concatenate((wind_stress0, wind_stress1), axis=0)  # (13271, 1, 1)
    surface_downward_eastward_stress0 = np.concatenate((surface_downward_eastward_stress0, surface_downward_eastward_stress1), axis=0)  # (13271, 1, 1)
    surface_downward_northward_stress0 = np.concatenate((surface_downward_northward_stress0, surface_downward_northward_stress1), axis=0)  # (13271, 1, 1)
    wind_stress_curl0 = np.concatenate((wind_stress_curl0, wind_stress_curl1), axis=0)  # (13271, 1, 1)
    wind_stress_divergence0 = np.concatenate((wind_stress_divergence0, wind_stress_divergence1), axis=0)  # (13271, 1, 1)
    wind_speed_rms0 = np.concatenate((wind_speed_rms0, wind_speed_rms1), axis=0)  # (13271, 1, 1)
    eastward_wind_rms0 = np.concatenate((eastward_wind_rms0, eastward_wind_rms1), axis=0)  # (13271, 1, 1)
    northward_wind_rms0 = np.concatenate((northward_wind_rms0, northward_wind_rms1), axis=0)  # (13271, 1, 1)
    sampling_length0 = np.concatenate((sampling_length0, sampling_length1), axis=0)  # (13271, 1, 1)
    surface_type0 = np.concatenate((surface_type0, surface_type1), axis=0)  # (13271, 1, 1)

    # print('np.mean(eastward_wind0):{}'.format(np.mean(eastward_wind0)))
    # print(time0.shape)
    # print(sst0.shape)

    time = list(time0)
    print(time)
    for i in range(len(time) - 1):
        if time[i + 1] - time[i] != 1:
            print('错误')
            break
# #
    np.savez(r'D:\heat_wave\pacific\last\wind_19_pacific_area.npz', time=time0, lat=lat00, lon=lon00,
             eastward_wind0 = eastward_wind0, northward_wind0 = northward_wind0, wind_vector_divergence0 = wind_vector_divergence0,
             wind_stress0 = wind_stress0, surface_downward_eastward_stress0 = surface_downward_eastward_stress0,
             surface_downward_northward_stress0 = surface_downward_northward_stress0, wind_stress_curl0 = wind_stress_curl0,
             wind_stress_divergence0 = wind_stress_divergence0, wind_speed_rms0 = wind_speed_rms0, eastward_wind_rms0 = eastward_wind_rms0,
             northward_wind_rms0 = northward_wind_rms0, sampling_length0 = sampling_length0, surface_type0 = surface_type0
             )

