import numpy as np

data = np.load(r'D:\heat_wave\WEIO\expand_WEIO\reanalysis_data_1993-2019_expand_WEIO_area.npz')
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
print(np.max(mld))  #112.39484    第26个深度为 120
print(np.mean(mld))
print(np.min(mld))

print(temp.shape)

sst_0 = temp[:,0,:,:]    #表面sst   0.5m
sst_1 = temp[:,25,:,:]  #147.40m的 temp
# # # #
u_0= u[:,0,:,:]
u_1 = u[:,25,:,:]
# #
v_0 = v[:,0,:,:]
v_1 = v[:,25,:,:]
print('sst_0.shape:{}'.format(sst_0.shape))
print('sst_1.shape:{}'.format(sst_1.shape))
print('mld.shape:{}'.format(mld.shape))



#将SST大于100的数设置为nan
sst_0 = np.where(sst_0>100, np.nan, sst_0)
print(np.isnan(sst_0).sum())


sst_1 = np.where(sst_1>100, np.nan, sst_1)
print(np.isnan(sst_1).sum())

u_1 = np.where(u_1>100, np.nan, u_1)
print(np.isnan(u_1).sum())


v_1 = np.where(v_1>100, np.nan, v_1)
print(np.isnan(v_1).sum())



from scipy import interpolate
def fill_nan(arr):
    isnan = np.isnan(arr)
    if np.sum(isnan) == 0:
        return arr
    notnan = np.logical_not(isnan)
    arr[isnan] = interpolate.griddata(np.where(notnan), arr[notnan], np.where(isnan), method='nearest')
    return arr

sst_1 = fill_nan(sst_1)
print(np.isnan(sst_1).sum())


u_1 = fill_nan(u_1)
print(np.isnan(u_1).sum())

v_1 = fill_nan(v_1)
print(np.isnan(v_1).sum())

T_d = sst_0 - ((sst_0 - sst_1) * (0.5 - mld - 10) / (120 - 0.5))  #(1826, 12, 15)  混合层10m下的温度
# #
u_d = u_0 - ((u_0 - u_1) * (0.5 - mld - 10) / (120 - 0.5))   #(1826, 12, 15)  混合层10m下的u
# #
v_d = u_0 - ((v_0 - v_1) * (0.5 - mld - 10) / (120 - 0.5))   #(1826, 12, 15)  混合层10m下的v

diff = sst_0[0,10,11] - sst_1[0,10,11]
print(sst_0[0,10,11])
print(sst_1[0,10,11])
# min_diff = np.min(diff)
# print(diff)
# print('min_diff:{}'.format(min_diff))
# print('indices of min diff:{}'.format(np.where(diff == min_diff)))

print(np.min((sst_0 - sst_1)))
print(T_d.shape) # (3287, 41, 201)

print(np.min(T_d))
print(np.max(T_d))
print(np.min(u_d))
print(np.max(u_d))
print(np.min(v_d))
print(np.max(v_d))

np.savez(r'D:\heat_wave\WEIO\expand_WEIO\T_d_u_d_v_d_93_19_deep_expand_WEIO_last_chazhi.npz', T_d = T_d, u_d = u_d, v_d = v_d)



