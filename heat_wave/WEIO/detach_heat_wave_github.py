# Load required modules
import numpy as np
from datetime import date
from matplotlib import pyplot as plt

import marineHeatWaves as mhw

# Generate time vector using datetime format (January 1 of year 1 is day 1)
t = np.arange(date(1993,1,1).toordinal(),date(2022,12,31).toordinal()+1)
print(t.shape)
dates = [date.fromordinal(tt.astype(int)) for tt in t]
# Generate synthetic temperature time series
from netCDF4 import Dataset, num2date

# path11 = r'I:\\daily_sst_1982-2019.nc'
# nc11 = Dataset(path11) # northward_wind  time   latitude    longitude    slhf   ssr(净向下的短波)   sshf   # (time, latitude, longitude)
# print(nc11.variables)
# data = nc11.variables['sst'][:]
data = np.load(r'D:\博士工作\第四篇论文_SCS_MHW_ENSO\data\oisst_93_22_south_ocean.npz')['sst']
# print(data.files) # ['time', 'lat', 'lon', 'sst']
data = np.squeeze(data)
print(data.shape) #(9861, 30, 41, 201)
data = data[:,:,:]
# print(np.min(data))
# print(data)

data = np.where(data<-100, np.nan, data)

sst = data
sst = np.nanmean(data, axis = (1,2))

# print(sst)
mhws, clim = mhw.detect(t, sst)
a = 32

# print(mhws['n_events']) # 81
print('duration:{}'.format(mhws['duration'][:]))
print('data_star:{}'.format(mhws['date_start'][a-1:a]))
print('date_end:{}'.format(mhws['date_end'][a-1:a]))
print('duration:{}'.format(mhws['duration'][a-1:a]))
print('intensity_max:{}'.format(mhws['intensity_max'][a-1:a]))
print('intensity_mean:{}'.format(mhws['intensity_mean'][a-1:a]))
print('intensity_cumulative:{}'.format(mhws['intensity_cumulative'][a-1:a]))
print('rate_onset:{}'.format(mhws['rate_onset'][a-1:a]))
#
#
# a

# #
# n_events = []
# duration = []
# date_start = []
# date_end = []
# intensity_max = []
# intensity_mean = []
# intensity_cumulative = []
# rate_onset = []
# for i in range(81):
#     for j in range(81):   # (13879, 101, 140)
#         # print(sst.shape)# (14610,)
#         print('sst.shape:{}'.format(sst.shape))
#         sst1 = sst
#         print(sst1.shape)
#         sst2 = sst1[:,j,i]
#         print(sst2.shape)
#
#         # print(sst)
#         mhws, clim = mhw.detect(t, sst2)
#         # a = 58
#         n_events.append(mhws['n_events'])
#         # print(n_events)
#         duration.append(mhws['duration'])
#
#         date_start.append(mhws['date_start'])
#
#         date_end.append(mhws['date_end'])
#
#         intensity_max.append(mhws['intensity_max'])
#
#         intensity_mean.append(mhws['intensity_mean'])
#
#         intensity_cumulative.append(mhws['intensity_cumulative'])
#         # rate_onset.append(mhws['rate_onset'])
#         # print(rate_onset)
#
#         # print('{}{}'.format(i,j))
# print(n_events)
# np.savez(r'D:\south_93_22_point_MHW_1.npz', n_events = n_events, duration = duration, date_start = date_start, date_end = date_end,
#          intensity_max = intensity_max, intensity_mean = intensity_mean, intensity_cumulative = intensity_cumulative)
# # np.savez(r'D:\pengshen.npz', n_events = n_events, duration = duration, date_start = date_start, date_end = date_end)
#
# # np.savez('D:\MHW_TC_SO\MHW_profile_solo_point_rate_onset', rate_onset = rate_onset)
#
# #         # print(mhws['n_events']) # 81
# #         # print('duration:{}'.format(mhws['duration'][:]))
# #         # print('data_star:{}'.format(mhws['date_start'][a-1:a]))
# #         # print('date_end:{}'.format(mhws['date_end'][a-1:a]))
# #         # print('duration:{}'.format(mhws['duration'][a-1:a]))
# #         # print('intensity_max:{}'.format(mhws['intensity_max'][a-1:a]))
# #         # print('intensity_mean:{}'.format(mhws['intensity_mean'][a-1:a]))
# #         # print('intensity_cumulative:{}'.format(mhws['intensity_cumulative'][a-1:a]))
# #         # print('rate_onset:{}'.format(mhws['rate_onset'][a-1:a]))