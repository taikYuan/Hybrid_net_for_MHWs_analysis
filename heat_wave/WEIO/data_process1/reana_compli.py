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
#
for cc in range(0,1):
#
#
    data = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_01_deep30_all_area.npz'.format(cc,cc))
    print(data.files)  #['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']

    time1 = data['time'][:]
    print(time1)

    lat = data['lat'][:]
    # print(lat)# -2 -1.75 -1.5 ... 1.5 1.75 2
    print(lat) #(681,) #

    lat1 = lat[288:353]
    # print(lat1) # [-8,8]
#
    lon = data['lon'][:]
#     # print(lon) #-180 -179.75   179.5, 179.75
#
    lon1 = lon[936:985]
    print(lon1) # 54E - 66E
#
    mld1 = data['mld'][:,288:353,936:985]
    print(mld1.shape) #(31, 41, 201)

    salinity1 = data['salinity'][:]
    print('salinity1.shape:{}'.format(salinity1.shape)) #(31, 30, 681, 1440) #30是深度

    salinity1 = salinity1[:,:,288:353,936:985]
    print(salinity1.shape) #(31, 30, 41, 201) #30是深度

    temp1 = data['temp'][:,:,288:353,936:985]
    print(temp1.shape)
    #
    u1 = data['u'][:,:,288:353,936:985]
    #
    v1 = data['v'][:,:,288:353,936:985]
    #
    #
    data2 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_02_deep30_all_area.npz'.format(cc,cc))
    time2 = data2['time'][:]
    mld2 = data2['mld'][:,288:353,936:985]
    salinity2 = data2['salinity'][:,:,288:353,936:985]
    temp2 = data2['temp'][:,:,288:353,936:985]
    u2 = data2['u'][:,:,288:353,936:985]
    v2 = data2['v'][:,:,288:353,936:985]
    print("salinity2.shape:{}".format(salinity2.shape))
    print("mld2.shape:{}".format(mld2.shape))
    print(2)

    data3 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_03_deep30_all_area.npz'.format(cc,cc))
    time3 = data3['time'][:]
    mld3 = data3['mld'][:,288:353,936:985]
    salinity3 = data3['salinity'][:,:,288:353,936:985]
    temp3 = data3['temp'][:,:,288:353,936:985]
    u3 = data3['u'][:,:,288:353,936:985]
    v3 = data3['v'][:,:,288:353,936:985]
    print("salinity3.shape:{}".format(salinity3.shape))
    print("mld3.shape:{}".format(mld3.shape))

    print(3)

    data4 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_04_deep30_all_area.npz'.format(cc,cc))
    time4 = data4['time'][:]
    mld4 = data4['mld'][:,288:353,936:985]
    salinity4 = data4['salinity'][:,:,288:353,936:985]
    temp4 = data4['temp'][:,:,288:353,936:985]
    u4 = data4['u'][:,:,288:353,936:985]
    v4 = data4['v'][:,:,288:353,936:985]
    print("salinity4.shape:{}".format(salinity4.shape))
    print("mld4.shape:{}".format(mld4.shape))
    print(4)

    data5 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_05_deep30_all_area.npz'.format(cc,cc))
    time5 = data5['time'][:]
    mld5 = data5['mld'][:,288:353,936:985]
    salinity5 = data5['salinity'][:,:,288:353,936:985]
    temp5 = data5['temp'][:,:,288:353,936:985]
    u5 = data5['u'][:,:,288:353,936:985]
    v5 = data5['v'][:,:,288:353,936:985]
    print("salinity5.shape:{}".format(salinity5.shape))
    print("mld5.shape:{}".format(mld5.shape))
    print(5)

    data6 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_06_deep30_all_area.npz'.format(cc,cc))
    time6 = data6['time'][:]
    mld6 = data6['mld'][:,288:353,936:985]
    salinity6 = data6['salinity'][:,:,288:353,936:985]
    temp6 = data6['temp'][:,:,288:353,936:985]
    u6 = data6['u'][:,:,288:353,936:985]
    v6 = data6['v'][:,:,288:353,936:985]
    print("salinity6.shape:{}".format(salinity6.shape))
    print("mld6.shape:{}".format(mld6.shape))
    print(6)

    data7 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_07_deep30_all_area.npz'.format(cc,cc))
    time7 = data7['time'][:]
    mld7 = data7['mld'][:,288:353,936:985]
    salinity7 = data7['salinity'][:,:,288:353,936:985]
    temp7 = data7['temp'][:,:,288:353,936:985]
    u7 = data7['u'][:,:,288:353,936:985]
    v7 = data7['v'][:,:,288:353,936:985]
    print("salinity7.shape:{}".format(salinity7.shape))
    print("mld7.shape:{}".format(mld7.shape))
    print(7)

    data8 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_08_deep30_all_area.npz'.format(cc,cc))
    time8 = data8['time'][:]
    mld8 = data8['mld'][:,288:353,936:985]
    salinity8 = data8['salinity'][:,:,288:353,936:985]
    temp8 = data8['temp'][:,:,288:353,936:985]
    u8 = data8['u'][:,:,288:353,936:985]
    v8 = data8['v'][:,:,288:353,936:985]
    print("salinity8.shape:{}".format(salinity8.shape))
    print("mld8.shape:{}".format(mld8.shape))
    print(8)

    data9 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_09_deep30_all_area.npz'.format(cc,cc))
    time9 = data9['time'][:]
    mld9 = data9['mld'][:,288:353,936:985]
    salinity9 = data9['salinity'][:,:,288:353,936:985]
    temp9 = data9['temp'][:,:,288:353,936:985]
    u9 = data9['u'][:,:,288:353,936:985]
    v9 = data9['v'][:,:,288:353,936:985]
    print("salinity9.shape:{}".format(salinity9.shape))
    print("mld9.shape:{}".format(mld9.shape))
    print(9)

    data10 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_10_deep30_all_area.npz'.format(cc,cc))
    time10 = data10['time'][:]
    mld10 = data10['mld'][:,288:353,936:985]
    salinity10 = data10['salinity'][:,:,288:353,936:985]
    temp10 = data10['temp'][:,:,288:353,936:985]
    u10 = data10['u'][:,:,288:353,936:985]
    v10 = data10['v'][:,:,288:353,936:985]
    print("salinity10.shape:{}".format(salinity10.shape))
    print("mld10.shape:{}".format(mld10.shape))
    print(10)

    data11 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_11_deep30_all_area.npz'.format(cc,cc))
    time11 = data11['time'][:]
    mld11 = data11['mld'][:,288:353,936:985]
    salinity11 = data11['salinity'][:,:,288:353,936:985]
    temp11 = data11['temp'][:,:,288:353,936:985]
    u11 = data11['u'][:,:,288:353,936:985]
    v11 = data11['v'][:,:,288:353,936:985]
    print("salinity11.shape:{}".format(salinity11.shape))
    print("mld11.shape:{}".format(mld11.shape))
    print(11)

    data12 = np.load(r'G:\synthesis_data\200{}\reanalysis_0{}_12_deep30_all_area.npz'.format(cc,cc))
    time12 = data12['time'][:]
    mld12 = data12['mld'][:,288:353,936:985]
    salinity12 = data12['salinity'][:,:,288:353,936:985]
    temp12 = data12['temp'][:,:,288:353,936:985]
    u12 = data12['u'][:,:,288:353,936:985]
    v12 = data12['v'][:,:,288:353,936:985]
    print("salinity12.shape:{}".format(salinity12.shape))
    print("mld12.shape:{}".format(mld12.shape))
    print(12)

    time = np.concatenate((time1,time2,time3,time4,time5,time6,time7,time8,time9,time10,time11,time12),axis=0)
    mld = np.concatenate((mld1,mld2,mld3,mld4,mld5,mld6,mld7,mld8,mld9,mld10,mld11,mld12),axis=0)
    salinity = np.concatenate((salinity1,salinity2,salinity3,salinity4,salinity5,salinity6,salinity7,salinity8,salinity9,salinity10,salinity11,salinity12),axis=0)
    temp = np.concatenate((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12),axis=0)
    u = np.concatenate((u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12),axis=0)
    v = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12),axis=0)

    print("mld.shape:{}".format(mld.shape))
    print("time.shape:{}".format(time.shape))
    print("salinity.shape:{}".format(salinity.shape))
    print("temp.shape:{}".format(temp.shape))
    print("u.shape:{}".format(u.shape))
    print("v.shape:{}".format(v.shape))

    np.savez(r'D:\heat_wave\WEIO\expand_WEIO\reanalysis\reanalysis_data_200{}_expand_WEIO_area.npz'.format(cc),time = time, lat = lat1, lon = lon1, mld = mld,
                           salinity = salinity, temp = temp, u = u, v = v)

#
# for aa in range(1,10):
#
#
#     data = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_01_deep30_all_area.npz'.format(aa,aa))
#     print(data.files)  #['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']
#
#     time1 = data['time'][:]
#     print(time1)
#
#     lat = data['lat'][:]
#     # print(lat)# -2 -1.75 -1.5 ... 1.5 1.75 2
#     # print(lat) #(681,) #
#
#     lat1 = lat[288:353]
#     # print(lat1) # [-2,2]
#
#     lon = data['lon'][:]
#     # print(lon) #-180 -179.75   179.5, 179.75
#
#     lon1 = lon[936:985]
#     print(lon1) # 48E - 54E
#
#     mld1 = data['mld'][:,288:353,936:985]
#     print(mld1.shape) #(31, 41, 201)
#
#     salinity1 = data['salinity'][:]
#     print('salinity1.shape:{}'.format(salinity1.shape)) #(31, 30, 681, 1440) #30是深度
#
#     salinity1 = salinity1[:,:,288:353,936:985]
#     print(salinity1.shape) #(31, 30, 41, 201) #30是深度
#
#     temp1 = data['temp'][:,:,288:353,936:985]
#     print(temp1.shape)
#     #
#     u1 = data['u'][:,:,288:353,936:985]
#     #
#     v1 = data['v'][:,:,288:353,936:985]
#     #
#     #
#     data2 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_02_deep30_all_area.npz'.format(aa,aa))
#     time2 = data2['time'][:]
#     mld2 = data2['mld'][:,288:353,936:985]
#     salinity2 = data2['salinity'][:,:,288:353,936:985]
#     temp2 = data2['temp'][:,:,288:353,936:985]
#     u2 = data2['u'][:,:,288:353,936:985]
#     v2 = data2['v'][:,:,288:353,936:985]
#     print("salinity2.shape:{}".format(salinity2.shape))
#     print("mld2.shape:{}".format(mld2.shape))
#     print(2)
#
#     data3 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_03_deep30_all_area.npz'.format(aa,aa))
#     time3 = data3['time'][:]
#     mld3 = data3['mld'][:,288:353,936:985]
#     salinity3 = data3['salinity'][:,:,288:353,936:985]
#     temp3 = data3['temp'][:,:,288:353,936:985]
#     u3 = data3['u'][:,:,288:353,936:985]
#     v3 = data3['v'][:,:,288:353,936:985]
#     print("salinity3.shape:{}".format(salinity3.shape))
#     print("mld3.shape:{}".format(mld3.shape))
#
#     print(3)
#
#     data4 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_04_deep30_all_area.npz'.format(aa,aa))
#     time4 = data4['time'][:]
#     mld4 = data4['mld'][:,288:353,936:985]
#     salinity4 = data4['salinity'][:,:,288:353,936:985]
#     temp4 = data4['temp'][:,:,288:353,936:985]
#     u4 = data4['u'][:,:,288:353,936:985]
#     v4 = data4['v'][:,:,288:353,936:985]
#     print("salinity4.shape:{}".format(salinity4.shape))
#     print("mld4.shape:{}".format(mld4.shape))
#     print(4)
#
#     data5 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_05_deep30_all_area.npz'.format(aa,aa))
#     time5 = data5['time'][:]
#     mld5 = data5['mld'][:,288:353,936:985]
#     salinity5 = data5['salinity'][:,:,288:353,936:985]
#     temp5 = data5['temp'][:,:,288:353,936:985]
#     u5 = data5['u'][:,:,288:353,936:985]
#     v5 = data5['v'][:,:,288:353,936:985]
#     print("salinity5.shape:{}".format(salinity5.shape))
#     print("mld5.shape:{}".format(mld5.shape))
#     print(5)
#
#     data6 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_06_deep30_all_area.npz'.format(aa,aa))
#     time6 = data6['time'][:]
#     mld6 = data6['mld'][:,288:353,936:985]
#     salinity6 = data6['salinity'][:,:,288:353,936:985]
#     temp6 = data6['temp'][:,:,288:353,936:985]
#     u6 = data6['u'][:,:,288:353,936:985]
#     v6 = data6['v'][:,:,288:353,936:985]
#     print("salinity6.shape:{}".format(salinity6.shape))
#     print("mld6.shape:{}".format(mld6.shape))
#     print(6)
#
#     data7 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_07_deep30_all_area.npz'.format(aa,aa))
#     time7 = data7['time'][:]
#     mld7 = data7['mld'][:,288:353,936:985]
#     salinity7 = data7['salinity'][:,:,288:353,936:985]
#     temp7 = data7['temp'][:,:,288:353,936:985]
#     u7 = data7['u'][:,:,288:353,936:985]
#     v7 = data7['v'][:,:,288:353,936:985]
#     print("salinity7.shape:{}".format(salinity7.shape))
#     print("mld7.shape:{}".format(mld7.shape))
#     print(7)
#
#     data8 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_08_deep30_all_area.npz'.format(aa,aa))
#     time8 = data8['time'][:]
#     mld8 = data8['mld'][:,288:353,936:985]
#     salinity8 = data8['salinity'][:,:,288:353,936:985]
#     temp8 = data8['temp'][:,:,288:353,936:985]
#     u8 = data8['u'][:,:,288:353,936:985]
#     v8 = data8['v'][:,:,288:353,936:985]
#     print("salinity8.shape:{}".format(salinity8.shape))
#     print("mld8.shape:{}".format(mld8.shape))
#     print(8)
#
#     data9 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_09_deep30_all_area.npz'.format(aa,aa))
#     time9 = data9['time'][:]
#     mld9 = data9['mld'][:,288:353,936:985]
#     salinity9 = data9['salinity'][:,:,288:353,936:985]
#     temp9 = data9['temp'][:,:,288:353,936:985]
#     u9 = data9['u'][:,:,288:353,936:985]
#     v9 = data9['v'][:,:,288:353,936:985]
#     print("salinity9.shape:{}".format(salinity9.shape))
#     print("mld9.shape:{}".format(mld9.shape))
#     print(9)
#
#     data10 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_10_deep30_all_area.npz'.format(aa,aa))
#     time10 = data10['time'][:]
#     mld10 = data10['mld'][:,288:353,936:985]
#     salinity10 = data10['salinity'][:,:,288:353,936:985]
#     temp10 = data10['temp'][:,:,288:353,936:985]
#     u10 = data10['u'][:,:,288:353,936:985]
#     v10 = data10['v'][:,:,288:353,936:985]
#     print("salinity10.shape:{}".format(salinity10.shape))
#     print("mld10.shape:{}".format(mld10.shape))
#     print(10)
#
#     data11 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_11_deep30_all_area.npz'.format(aa,aa))
#     time11 = data11['time'][:]
#     mld11 = data11['mld'][:,288:353,936:985]
#     salinity11 = data11['salinity'][:,:,288:353,936:985]
#     temp11 = data11['temp'][:,:,288:353,936:985]
#     u11 = data11['u'][:,:,288:353,936:985]
#     v11 = data11['v'][:,:,288:353,936:985]
#     print("salinity11.shape:{}".format(salinity11.shape))
#     print("mld11.shape:{}".format(mld11.shape))
#     print(11)
#
#     data12 = np.load(r'H:\synthesis_data\synthesis_data\200{}\reanalysis_0{}_12_deep30_all_area.npz'.format(aa,aa))
#     time12 = data12['time'][:]
#     mld12 = data12['mld'][:,288:353,936:985]
#     salinity12 = data12['salinity'][:,:,288:353,936:985]
#     temp12 = data12['temp'][:,:,288:353,936:985]
#     u12 = data12['u'][:,:,288:353,936:985]
#     v12 = data12['v'][:,:,288:353,936:985]
#     print("salinity12.shape:{}".format(salinity12.shape))
#     print("mld12.shape:{}".format(mld12.shape))
#     print(12)
#
#     time = np.concatenate((time1,time2,time3,time4,time5,time6,time7,time8,time9,time10,time11,time12),axis=0)
#     mld = np.concatenate((mld1,mld2,mld3,mld4,mld5,mld6,mld7,mld8,mld9,mld10,mld11,mld12),axis=0)
#     salinity = np.concatenate((salinity1,salinity2,salinity3,salinity4,salinity5,salinity6,salinity7,salinity8,salinity9,salinity10,salinity11,salinity12),axis=0)
#     temp = np.concatenate((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12),axis=0)
#     u = np.concatenate((u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12),axis=0)
#     v = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12),axis=0)
#
#     print("mld.shape:{}".format(mld.shape))
#     print("time.shape:{}".format(time.shape))
#     print("salinity.shape:{}".format(salinity.shape))
#     print("temp.shape:{}".format(temp.shape))
#     print("u.shape:{}".format(u.shape))
#     print("v.shape:{}".format(v.shape))
#
#     np.savez(r'D:\heat_wave\WEIO\expand_WEIO\reanalysis\reanalysis_data_200{}_expand_WEIO_area.npz'.format(aa),time = time, lat = lat1, lon = lon1, mld = mld,
#                            salinity = salinity, temp = temp, u = u, v = v)
#
#
# for bb in range(10,20):
#
#
#     data = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_01_deep30_all_area.npz'.format(bb,bb))
#     print(data.files)  #['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']
#
#     time1 = data['time'][:]
#     print(time1)
#
#     lat = data['lat'][:]
#     # print(lat)# -2 -1.75 -1.5 ... 1.5 1.75 2
#     # print(lat) #(681,) #
#
#     lat1 = lat[288:353]
#     # print(lat1) # [-2,2]
#
#     lon = data['lon'][:]
#     # print(lon) #-180 -179.75   179.5, 179.75
#
#     lon1 = lon[936:985]
#     print(lon1) # 48E - 54E
#
#     mld1 = data['mld'][:,288:353,936:985]
#     print(mld1.shape) #(31, 41, 201)
#
#     salinity1 = data['salinity'][:]
#     print('salinity1.shape:{}'.format(salinity1.shape)) #(31, 30, 681, 1440) #30是深度
#
#     salinity1 = salinity1[:,:,288:353,936:985]
#     print(salinity1.shape) #(31, 30, 41, 201) #30是深度
#
#     temp1 = data['temp'][:,:,288:353,936:985]
#     print(temp1.shape)
#
#     u1 = data['u'][:,:,288:353,936:985]
#
#     v1 = data['v'][:,:,288:353,936:985]
#
#
#     data2 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_02_deep30_all_area.npz'.format(bb,bb))
#     time2 = data2['time'][:]
#     mld2 = data2['mld'][:,288:353,936:985]
#     salinity2 = data2['salinity'][:,:,288:353,936:985]
#     temp2 = data2['temp'][:,:,288:353,936:985]
#     u2 = data2['u'][:,:,288:353,936:985]
#     v2 = data2['v'][:,:,288:353,936:985]
#     print("salinity2.shape:{}".format(salinity2.shape))
#     print("mld2.shape:{}".format(mld2.shape))
#     print(2)
#
#     data3 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_03_deep30_all_area.npz'.format(bb,bb))
#     time3 = data3['time'][:]
#     mld3 = data3['mld'][:,288:353,936:985]
#     salinity3 = data3['salinity'][:,:,288:353,936:985]
#     temp3 = data3['temp'][:,:,288:353,936:985]
#     u3 = data3['u'][:,:,288:353,936:985]
#     v3 = data3['v'][:,:,288:353,936:985]
#     print("salinity3.shape:{}".format(salinity3.shape))
#     print("mld3.shape:{}".format(mld3.shape))
#
#     print(3)
#
#     data4 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_04_deep30_all_area.npz'.format(bb,bb))
#     time4 = data4['time'][:]
#     mld4 = data4['mld'][:,288:353,936:985]
#     salinity4 = data4['salinity'][:,:,288:353,936:985]
#     temp4 = data4['temp'][:,:,288:353,936:985]
#     u4 = data4['u'][:,:,288:353,936:985]
#     v4 = data4['v'][:,:,288:353,936:985]
#     print("salinity4.shape:{}".format(salinity4.shape))
#     print("mld4.shape:{}".format(mld4.shape))
#     print(4)
#
#     data5 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_05_deep30_all_area.npz'.format(bb,bb))
#     time5 = data5['time'][:]
#     mld5 = data5['mld'][:,288:353,936:985]
#     salinity5 = data5['salinity'][:,:,288:353,936:985]
#     temp5 = data5['temp'][:,:,288:353,936:985]
#     u5 = data5['u'][:,:,288:353,936:985]
#     v5 = data5['v'][:,:,288:353,936:985]
#     print("salinity5.shape:{}".format(salinity5.shape))
#     print("mld5.shape:{}".format(mld5.shape))
#     print(5)
#
#     data6 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_06_deep30_all_area.npz'.format(bb,bb))
#     time6 = data6['time'][:]
#     mld6 = data6['mld'][:,288:353,936:985]
#     salinity6 = data6['salinity'][:,:,288:353,936:985]
#     temp6 = data6['temp'][:,:,288:353,936:985]
#     u6 = data6['u'][:,:,288:353,936:985]
#     v6 = data6['v'][:,:,288:353,936:985]
#     print("salinity6.shape:{}".format(salinity6.shape))
#     print("mld6.shape:{}".format(mld6.shape))
#     print(6)
#
#     data7 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_07_deep30_all_area.npz'.format(bb,bb))
#     time7 = data7['time'][:]
#     mld7 = data7['mld'][:,288:353,936:985]
#     salinity7 = data7['salinity'][:,:,288:353,936:985]
#     temp7 = data7['temp'][:,:,288:353,936:985]
#     u7 = data7['u'][:,:,288:353,936:985]
#     v7 = data7['v'][:,:,288:353,936:985]
#     print("salinity7.shape:{}".format(salinity7.shape))
#     print("mld7.shape:{}".format(mld7.shape))
#     print(7)
#
#     data8 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_08_deep30_all_area.npz'.format(bb,bb))
#     time8 = data8['time'][:]
#     mld8 = data8['mld'][:,288:353,936:985]
#     salinity8 = data8['salinity'][:,:,288:353,936:985]
#     temp8 = data8['temp'][:,:,288:353,936:985]
#     u8 = data8['u'][:,:,288:353,936:985]
#     v8 = data8['v'][:,:,288:353,936:985]
#     print("salinity8.shape:{}".format(salinity8.shape))
#     print("mld8.shape:{}".format(mld8.shape))
#     print(8)
#
#     data9 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_09_deep30_all_area.npz'.format(bb,bb))
#     time9 = data9['time'][:]
#     mld9 = data9['mld'][:,288:353,936:985]
#     salinity9 = data9['salinity'][:,:,288:353,936:985]
#     temp9 = data9['temp'][:,:,288:353,936:985]
#     u9 = data9['u'][:,:,288:353,936:985]
#     v9 = data9['v'][:,:,288:353,936:985]
#     print("salinity9.shape:{}".format(salinity9.shape))
#     print("mld9.shape:{}".format(mld9.shape))
#     print(9)
#
#     data10 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_10_deep30_all_area.npz'.format(bb,bb))
#     time10 = data10['time'][:]
#     mld10 = data10['mld'][:,288:353,936:985]
#     salinity10 = data10['salinity'][:,:,288:353,936:985]
#     temp10 = data10['temp'][:,:,288:353,936:985]
#     u10 = data10['u'][:,:,288:353,936:985]
#     v10 = data10['v'][:,:,288:353,936:985]
#     print("salinity10.shape:{}".format(salinity10.shape))
#     print("mld10.shape:{}".format(mld10.shape))
#     print(10)
#
#     data11 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_11_deep30_all_area.npz'.format(bb,bb))
#     time11 = data11['time'][:]
#     mld11 = data11['mld'][:,288:353,936:985]
#     salinity11 = data11['salinity'][:,:,288:353,936:985]
#     temp11 = data11['temp'][:,:,288:353,936:985]
#     u11 = data11['u'][:,:,288:353,936:985]
#     v11 = data11['v'][:,:,288:353,936:985]
#     print("salinity11.shape:{}".format(salinity11.shape))
#     print("mld11.shape:{}".format(mld11.shape))
#     print(11)
#
#     data12 = np.load(r'H:\synthesis_data\synthesis_data\20{}\reanalysis_{}_12_deep30_all_area.npz'.format(bb,bb))
#     time12 = data12['time'][:]
#     mld12 = data12['mld'][:,288:353,936:985]
#     salinity12 = data12['salinity'][:,:,288:353,936:985]
#     temp12 = data12['temp'][:,:,288:353,936:985]
#     u12 = data12['u'][:,:,288:353,936:985]
#     v12 = data12['v'][:,:,288:353,936:985]
#     print("salinity12.shape:{}".format(salinity12.shape))
#     print("mld12.shape:{}".format(mld12.shape))
#     print(12)
#
#     time = np.concatenate((time1,time2,time3,time4,time5,time6,time7,time8,time9,time10,time11,time12),axis=0)
#     mld = np.concatenate((mld1,mld2,mld3,mld4,mld5,mld6,mld7,mld8,mld9,mld10,mld11,mld12),axis=0)
#     salinity = np.concatenate((salinity1,salinity2,salinity3,salinity4,salinity5,salinity6,salinity7,salinity8,salinity9,salinity10,salinity11,salinity12),axis=0)
#     temp = np.concatenate((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12),axis=0)
#     u = np.concatenate((u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12),axis=0)
#     v = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12),axis=0)
#
#     print("mld.shape:{}".format(mld.shape))
#     print("time.shape:{}".format(time.shape))
#     print("salinity.shape:{}".format(salinity.shape))
#     print("temp.shape:{}".format(temp.shape))
#     print("u.shape:{}".format(u.shape))
#     print("v.shape:{}".format(v.shape))
#
#     np.savez(r'D:\heat_wave\WEIO\expand_WEIO\reanalysis\reanalysis_data_20{}_expand_WEIO_area.npz'.format(bb),time = time, lat = lat1, lon = lon1, mld = mld,
#                            salinity = salinity, temp = temp, u = u, v = v)
