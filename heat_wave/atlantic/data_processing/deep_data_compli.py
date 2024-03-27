import numpy as np

data = np.load(r'D:\heat_wave\atlantic\reanalysis_data_1993-2019_expand_atlantic_area.npz')
# data1 = np.load(r'D:\data_3th_paper_3Dtemperature\pacific_ocean\reanalysis_data_2009-2017_pacific_ocean.npz')
print(data.files)   # ['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']
# print(data1.files)  # ['time', 'lat', 'lon', 'mld', 'sss', 'sst', 'u', 'v']

# 深度最高80m   根据数据本身  第23个数据位86m
# depth = data['depth'][:] # 0.5 25 50 75 100 125 150
# print(depth)
temp = data['temp'][:]  # (9861, 30, 41, 201)  #
u = data['u'][:]   # (9861, 30, 41, 201)
v = data['v'][:]   # (9861, 30, 41, 201)
mld = data['mld'][:]  # (9861, 41, 201)
print('temp.shape:{}'.format(temp.shape))
print('u.shape:{}'.format(u.shape))
print('v.shape:{}'.format(v.shape))
print('mld.shape:{}'.format(mld.shape))
print(np.max(mld))  #227.96114    第30个深度为 180.54
print(np.mean(mld))


print(temp.shape)

sst_0 = temp[:,0,:,:]    #表面sst   0.5m
sst_1 = temp[:,29,:]  #147.40m的 temp
# # # #
u_0= u[:,0,:,:]
u_1 = u[:,29,:,:]
# #
v_0 = v[:,0,:,:]
v_1 = v[:,29,:,:]
print('sst_0.shape:{}'.format(sst_0.shape))
print('sst_1.shape:{}'.format(sst_1.shape))
print('mld.shape:{}'.format(mld.shape))
T_d = sst_0 - ((sst_0 - sst_1) * (0.5 - mld - 10) / (180.54 - 0.5))  #(1826, 12, 15)  混合层10m下的温度
# #
u_d = u_0 - ((u_0 - u_1) * (0.5 - mld - 10) / (180.54 - 0.5))   #(1826, 12, 15)  混合层10m下的u
# #
v_d = u_0 - ((v_0 - v_1) * (0.5 - mld - 10) / (180.54 - 0.5))   #(1826, 12, 15)  混合层10m下的v
#
print(T_d.shape) # (3287, 41, 201)

np.savez(r'D:\heat_wave\atlantic\T_d_u_d_v_d_93_19_deep_expand_atlantic_last.npz', T_d = T_d, u_d = u_d, v_d = v_d)



