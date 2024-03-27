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

data = np.load(r'D:\heat_wave\WEIO\last\EC_slhf_93_19_WEIO_ocean_area.npz')

print(data.files)  # ['time', 'lat', 'lon', 'slfh']

time1 = data['time'][:].data # 1 2 ... 10592
print(time1)

lat = data['lat'][:]
# print(lat) #  2 1.75 ... -1.75 -2
print(lat) #(681,)

lon = data['lon'][:] # 48 48.25 ... 53.75 54
# print(lon)

slfh1 = data['slhf'][:] # (9861, 17, 25)
print(slfh1.shape)

print(1)


data1 = np.load(r'D:\heat_wave\WEIO\last\EC_sshf_93_19_WEIO_ocean_area.npz')

print(data1.files)  # ['time', 'lat', 'lon', 'ssfh']


sshf1 = data1['sshf'][:] # (3287, 41, 201)
# print(sshf1)

print(2)

data2 = np.load(r'D:\heat_wave\WEIO\last\EC_ssr_93_19_WEIO_ocean_area.npz')

print(data2.files)  # ['time', 'lat', 'lon', 'ssr']


ssr1 = data2['ssr'][:] # (3287, 41, 201)
# print(ssr1)

print(3)

data3 = np.load(r'D:\heat_wave\WEIO\last\EC_str_93_19_WEIO_ocean_area.npz')

print(data3.files)  # ['time', 'lat', 'lon', 'str']


str1 = data3['str'][:] # (9861, 17, 25)
# print(str1)

print(3)


# # #对维度倒叙
list1 = []
list2 = []
list3 = []
list4 = []
for i in range(9861):
    for j in range(17):
        for k in range(25):
            list1.append(slfh1[i, -j - 1, k])
            list2.append(sshf1[i, -j - 1, k])
            list3.append(ssr1[i, -j - 1, k])
            list4.append(str1[i, -j - 1, k])

slfh000 = np.array(list1)
sshf000 = np.array(list2)
ssr000 = np.array(list3)
str000 = np.array(list4)

slfh000 = slfh000.reshape(9861,17,25)
sshf000 = sshf000.reshape(9861,17,25)
ssr000 = ssr000.reshape(9861,17,25)
str000 = str000.reshape(9861,17,25)



np.savez(r'D:\heat_wave\WEIO\last\radiation_data_93_19_WEIO_ocean_last.npz', lat = lat, lon = lon, slfh = slfh000, sshf = sshf000, ssr = ssr000, str = str000)