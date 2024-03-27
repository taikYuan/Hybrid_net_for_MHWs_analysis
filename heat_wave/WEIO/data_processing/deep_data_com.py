import numpy as np
from scipy.interpolate import interp1d

data = np.load(r'D:\heat_wave\WEIO\last\reanalysis_data_1993-2019_WEIO_area.npz')
# data1 = np.load(r'D:\data_3th_paper_3Dtemperature\pacific_ocean\reanalysis_data_2009-2017_pacific_ocean.npz')
print(data.files)   # ['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']
# print(data1.files)  # ['time', 'lat', 'lon', 'mld', 'sss', 'sst', 'u', 'v']

# 深度最高80m   根据数据本身  第23个数据位86m
# depth = data['depth'][:] # 0.5 25 50 75 100 125 150
# print(depth)
temp = data['temp'][:]  # (9861, 17, 25)  #
u = data['u'][:]   # (9861, 17, 25)
v = data['v'][:]   # (9861, 17, 25)
mld = data['mld'][:]  # (9861, 17, 25)
print(temp.shape)
print(np.mean(temp))
print(np.max(mld))  #360.59982   第7个深度为 150
# print(mld)
print(temp.shape)




sst_0 = temp[:,0,:,:]    #表面sst   0.5m
sst_1 = temp[:,6,:,:]  #147.40m的 temp
# # # #
u_0= u[:,0,:,:]
u_1 = u[:,6,:,:]
# #
v_0 = v[:,0,:,:]
v_1 = v[:,6,:,:]
#
T_d = sst_0 - ((sst_0 - sst_1) * (0.5 - mld - 10) / (0.5 - 150))  #(1826, 12, 15)  混合层10m下的温度
# #
u_d = u_0 - ((u_0 - u_1) * (0.5 - mld - 10) / (0.5 - 150))   #(1826, 12, 15)  混合层10m下的u
# #
v_d = u_0 - ((v_0 - v_1) * (0.5 - mld - 10) / (0.5 - 150))   #(1826, 12, 15)  混合层10m下的v
#
# print(T_d.shape) # (3287, 41, 201)
print(np.mean(T_d))
print(np.mean(v_d))
print(np.mean(u_d))


# np.savez(r'D:\data_3th_paper_3Dtemperature\atlantic_ocean\available_last_data\T_d_u_d_v_d_09_17_deep_atlantic_last.npz', T_d = T_d, u_d = u_d, v_d = v_d)



