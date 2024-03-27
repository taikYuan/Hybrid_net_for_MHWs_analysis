import numpy as np
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

data1 = np.load(r'D:\heat_wave\sea_level_pressure_93_19_all_area.npz')
print(data1.files)  #['lat', 'lon', 'mslp']


lat1 = data1['lat'][:]  # (73)
lon1 = data1['lon'][:] # (144)
mslp1 = data1['mslp'][:,:,:]  #(9855, 73, 144)



# print(lat1)  # 90 87.5 ... -87.5 -90
# print(lon1) # 0 2.5 5 ... 352.5 355 357.5

lat2 = lat1[32:41]  # 10 7.5 5 2.5 0 -2.5 -5 -7.5 -10

lon2 = lon1[21:28]  # 52.5 55 ... 65  67.5

mslp2 = mslp1[:,32:41,21:28]  # 单位 pa

mslp2 = mslp2 / 100  # ---》  单位hpa

print(lat2)
# 去除季节性和长期趋势

# 读取SLPA数据，假设数据存储在名为data.npz的文件中

# 季节性分解，使用additive模型

# 假设数据为3D数组（time，lat，lon）
nt, nlats, nlons = mslp2.shape
print(mslp2.shape) #(9855, 9, 7)
# 重塑为2D数组（time，lat*lon）
mslp2_2d = mslp2.reshape(nt, nlats*nlons)

# 对每个经纬度点进行季节性分解
decomposition = seasonal_decompose(mslp2_2d, model='additive', period=12)

# 将每个经纬度点的趋势项去除
trend = decomposition.trend
for i in range(nlats*nlons):
    model = LinearRegression().fit(np.arange(nt).reshape(-1, 1), decomposition.observed[:, i])
    trend[:, i] = model.predict(np.arange(nt).reshape(-1, 1))
mslp_detrended = decomposition.observed - trend

# 重塑为3D数组（time，lat，lon）
mslp_detrended = mslp_detrended.reshape(nt, nlats, nlons)

mslp_detrended = mslp_detrended.reshape(-1,9,7)
print(mslp_detrended.shape)


print(np.mean(mslp_detrended))
print(np.max(mslp_detrended))
print(np.min(mslp_detrended))

# y = np.arange(54, 66.2, 0.25) #(201,)
# x = np.arange(-8, 5.2, 0.25)   #(41,)
# print(x.shape)
# print(y.shape)
# # #对维度倒叙
# list3 = []
# list4 = []
# for i in range(9855):
#         for j in range(9):
#             for k in range(7):
#                 list3.append(mslp2[i, -j-1, k])
#                 list4.append(mslp_detrended[i, -j-1, k])
#
#
# mslp00 = np.array(list3)
# print(mslp00.shape) #(2410650, 1)
# mslp00 = mslp00.reshape(9855,9,7)
# # print(sst00)
# print(np.mean(mslp00))
# print(np.max(mslp00))
# print(np.min(mslp00))
#
# mslp_detrended00 = np.array(list4)
# print(mslp_detrended00.shape) #(2410650, 1)
# mslp_detrended00 = mslp_detrended00.reshape(9855,9,7)
#
# print(mslp00[0,:,:])
# print(mslp2[0,:,:])
#
#
# mslp2 = mslp00
#
# mslp_detrended = mslp_detrended00
#
# y = np.array(lon2)
# x = np.array(lat2)
# # print(y)
# print(x.shape)
# print('y.shape:{}'.format(y.shape))
# y_new = np.arange(54, 66.2, 0.25) #(201   ,)
# x_new = np.arange(-8, 8.2, 0.25)  #(41,)
# print(x_new.shape)
# print(y_new.shape)
# # # # #
#
# xx,yy = np.meshgrid(x,y)              #(97, 80)
# # print(xx.shape) #(97, 80)
# # print(yy.shape) #(97, 80)
# list1 = []
# list2 = []
# # list3 = []
# # list4 = []
# #
# for i in range(9855):
#     z1 = mslp2[i,:,:] #(97, 80)
#     z2 = mslp_detrended[i, :, :]  # (97, 80)
#
#     f1 = interpolate.interp2d(y,x,z1, kind='cubic')
#     f2 = interpolate.interp2d(y, x, z2, kind='cubic')
#
# # #
# # #
#     for j in y_new:
#         for k in x_new:
#             ssh0 = f1(j,k)
#             list1.append(ssh0)
#
#             ssha0 = f2(j, k)
#             list2.append(ssha0)
#
#
# # # #
# # # #
# mslp00 = np.array(list1)
# print(mslp00.shape) #(2410650, 1)
# sst00 = mslp00.reshape(9855, 65,49)
# print(mslp00.shape)
# print(np.mean(mslp00))
# print(np.max(mslp00))
# print(np.min(mslp00))
#
#
# mslp_detrended00 = np.array(list1)
# print(mslp_detrended00.shape) #(2410650, 1)
# mslp_detrended00 = mslp_detrended00.reshape(9855, 65,49)
#
#
#
#
# #
# #
# #
# # # #
# np.savez(r'D:\heat_wave\WEIO\sea_level_press_WEIO_chazhi', lat = x_new, lon = y_new,
#          mslp = mslp00, mslpa = mslp_detrended00)

