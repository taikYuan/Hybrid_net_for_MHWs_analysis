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

data = np.load(r'D:\heat_wave\WEIO\reanalysis_data_1993_WEIO_area.npz')

print(data.files)  #['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']

time1 = data['time'][:]
# print(time1)

lat = data['lat'][:]
# print(lat)#-80 -79.75  ... 89.75 90
# print(lat) #(681,)

lon = data['lon'][:]
# print(lon)
mld1 = data['mld']
# print(mld1.shape)
salinity1 = data['salinity'][:]
temp1 = data['temp'][:]
u1 = data['u'][:]
v1 = data['v'][:]
print(1)
data2 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_1994_WEIO_area.npz')

time2 = data2['time'][:]
mld2 = data2['mld']
salinity2 = data2['salinity'][:]
temp2 = data2['temp'][:]
u2 = data2['u'][:]
v2 = data2['v'][:]
print(2)

data3 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_1995_WEIO_area.npz')

time3 = data3['time'][:]
mld3 = data3['mld']
salinity3 = data3['salinity'][:]
temp3 = data3['temp'][:]
u3 = data3['u'][:]
v3 = data3['v'][:]
print(3)

data4 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_1996_WEIO_area.npz')

time4 = data4['time'][:]
mld4 = data4['mld']
salinity4 = data4['salinity'][:]
temp4 = data4['temp'][:]
u4 = data4['u'][:]
v4 = data4['v'][:]
print(4)

data5 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_1997_WEIO_area.npz')

time5 = data5['time'][:]
mld5 = data5['mld']
salinity5 = data5['salinity'][:]
temp5 = data5['temp'][:]
u5 = data5['u'][:]
v5 = data5['v'][:]
print(5)

data6 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_1998_WEIO_area.npz')

time6 = data6['time'][:]
mld6 = data6['mld']
salinity6 = data6['salinity'][:]
temp6 = data6['temp'][:]
u6 = data6['u'][:]
v6 = data6['v'][:]
print(6)
data7 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_1999_WEIO_area.npz')

time7 = data7['time'][:]
mld7 = data7['mld']
salinity7 = data7['salinity'][:]
temp7 = data7['temp'][:]
u7 = data7['u'][:]
v7 = data7['v'][:]
print(7)

data8 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2000_WEIO_area.npz')

time8 = data8['time'][:]
mld8 = data8['mld']
salinity8 = data8['salinity'][:]
temp8 = data8['temp'][:]
u8 = data8['u'][:]
v8 = data8['v'][:]
print(8)

data9 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2001_WEIO_area.npz')

time9 = data9['time'][:]
mld9 = data9['mld']
salinity9 = data9['salinity'][:]
temp9 = data9['temp'][:]
u9 = data9['u'][:]
v9 = data9['v'][:]
print(9)

data10 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2002_WEIO_area.npz')

time10 = data10['time'][:]
mld10 = data10['mld']
salinity10 = data10['salinity'][:]
temp10 = data10['temp'][:]
u10 = data10['u'][:]
v10 = data10['v'][:]
print(10)

data11 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2003_WEIO_area.npz')

time11 = data11['time'][:]
mld11 = data11['mld']
salinity11 = data11['salinity'][:]
temp11 = data11['temp'][:]
u11 = data11['u'][:]
v11 = data11['v'][:]
print(11)

data12 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2004_WEIO_area.npz')

time12 = data12['time'][:]
mld12 = data12['mld']
salinity12 = data12['salinity'][:]
temp12 = data12['temp'][:]
u12 = data12['u'][:]
v12 = data12['v'][:]
print(12)

data13 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2005_WEIO_area.npz')

time13 = data13['time'][:]
mld13 = data13['mld']
salinity13 = data13['salinity'][:]
temp13 = data13['temp'][:]
u13 = data13['u'][:]
v13 = data13['v'][:]
print(13)

data14 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2006_WEIO_area.npz')

time14 = data14['time'][:]
mld14 = data14['mld']
salinity14 = data14['salinity'][:]
temp14 = data14['temp'][:]
u14 = data14['u'][:]
v14 = data14['v'][:]
print(14)

data15 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2007_WEIO_area.npz')

time15 = data15['time'][:]
mld15 = data15['mld']
salinity15 = data15['salinity'][:]
temp15 = data15['temp'][:]
u15 = data15['u'][:]
v15 = data15['v'][:]
print(15)

data16 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2008_WEIO_area.npz')

time16 = data16['time'][:]
mld16 = data16['mld']
salinity16 = data16['salinity'][:]
temp16 = data16['temp'][:]
u16 = data16['u'][:]
v16 = data16['v'][:]
print(16)

data17 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2009_WEIO_area.npz')

time17 = data17['time'][:]
mld17 = data17['mld']
salinity17 = data17['salinity'][:]
temp17 = data17['temp'][:]
u17 = data17['u'][:]
v17 = data17['v'][:]
print(17)

data18 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2010_WEIO_area.npz')

time18 = data18['time'][:]
mld18 = data18['mld']
salinity18 = data18['salinity'][:]
temp18 = data18['temp'][:]
u18 = data18['u'][:]
v18 = data18['v'][:]
print(18)

data19 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2011_WEIO_area.npz')

time19 = data19['time'][:]
mld19 = data19['mld']
salinity19 = data19['salinity'][:]
temp19 = data19['temp'][:]
u19 = data19['u'][:]
v19 = data19['v'][:]
print(19)

data20 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2012_WEIO_area.npz')

time20 = data20['time'][:]
mld20 = data20['mld']
salinity20 = data20['salinity'][:]
temp20 = data20['temp'][:]
u20 = data20['u'][:]
v20 = data20['v'][:]
print(20)

data21 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2013_WEIO_area.npz')

time21 = data21['time'][:]
mld21 = data21['mld']
salinity21 = data21['salinity'][:]
temp21 = data21['temp'][:]
u21 = data21['u'][:]
v21 = data21['v'][:]
print(21)

data22 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2014_WEIO_area.npz')

time22 = data22['time'][:]
mld22 = data22['mld']
salinity22 = data22['salinity'][:]
temp22 = data22['temp'][:]
u22 = data22['u'][:]
v22 = data22['v'][:]
print(22)

data23 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2015_WEIO_area.npz')

time23 = data23['time'][:]
mld23 = data23['mld']
salinity23 = data23['salinity'][:]
temp23 = data23['temp'][:]
u23 = data23['u'][:]
v23 = data23['v'][:]
print(23)

data24 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2016_WEIO_area.npz')

time24 = data24['time'][:]
mld24 = data24['mld']
salinity24 = data24['salinity'][:]
temp24 = data24['temp'][:]
u24 = data24['u'][:]
v24 = data24['v'][:]
print(24)

data25 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2017_WEIO_area.npz')

time25 = data25['time'][:]
mld25 = data25['mld']
salinity25 = data25['salinity'][:]
temp25 = data25['temp'][:]
u25 = data25['u'][:]
v25 = data25['v'][:]
print(25)

data26 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2018_WEIO_area.npz')

time26 = data26['time'][:]
mld26 = data26['mld']
salinity26 = data26['salinity'][:]
temp26 = data26['temp'][:]
u26 = data26['u'][:]
v26 = data26['v'][:]
print(26)

data27 = np.load(r'D:\heat_wave\WEIO\reanalysis_data_2019_WEIO_area.npz')

time27 = data27['time'][:]
mld27 = data27['mld']
salinity27 = data27['salinity'][:]
temp27 = data27['temp'][:]
u27 = data27['u'][:]
v27 = data27['v'][:]
print(27)

time = np.concatenate((time1,time2,time3,time4,time5,time6,time7,time8,time9,time10,time11,time12,time13,time14,time15,time16,time17,time18,time19,time20,time21,time22,time23,time24,time25,time26,time27),axis=0)
mld = np.concatenate((mld1,mld2,mld3,mld4,mld5,mld6,mld7,mld8,mld9,mld10,mld11,mld12,mld13,mld14,mld15,mld16,mld17,mld18,mld19,mld20,mld21,mld22,mld23,mld24,mld25,mld26,mld27),axis=0)
salinity = np.concatenate((salinity1,salinity2,salinity3,salinity4,salinity5,salinity6,salinity7,salinity8,salinity9,salinity10,salinity11,salinity12,salinity13,salinity14,salinity15,salinity16,salinity17,salinity18,salinity19,salinity20,salinity21,salinity22,salinity23,salinity24,salinity25,salinity26,salinity27),axis=0)
temp = np.concatenate((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12,temp13,temp14,temp15,temp16,temp17,temp18,temp19,temp20,temp21,temp22,temp23,temp24,temp25,temp26,temp27),axis=0)
u = np.concatenate((u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27),axis=0)
v = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27),axis=0)

print("mld.shape:{}".format(mld.shape))
print("time.shape:{}".format(time.shape))
print("salinity.shape:{}".format(salinity.shape))
print("temp.shape:{}".format(temp.shape))
print("u.shape:{}".format(u.shape))
print("v.shape:{}".format(v.shape))

np.savez(r'D:\heat_wave\WEIO\reanalysis_data_1993-2019_WEIO_area.npz',time = time, lat = lat, lon = lon, mld = mld,
                       salinity = salinity, temp = temp, u = u, v = v)