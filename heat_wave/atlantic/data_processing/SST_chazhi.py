import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_squared_error

data1 = np.load(r'D:\heat_wave\atlantic\SST_93_19_expand_atlantic_area.npz')
print(data1.files)  #['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']


lat1 = data1['lat'][1:]  # -4.875 -4.625 ... 4.875 5.125
lon1 = data1['lon'][1:] # 190.125 190.375 ... 239.875  240.125
sst1 = data1['sst'][:,1:,1:]  #(1826, 98, 98)

# print(lat1)   #-4.875   5.125
# print(lon1)   #189.875 ...239.875
print('lon1.shape:{}'.format(lon1.shape))
print(sst1.shape) #(9861, 41, 210)

#
print(np.mean(sst1))
print(np.max(sst1))
print(np.min(sst1))
#
y = lon1.reshape(-1,)
# print(y)
x = lat1.reshape(-1,)
# print(x.shape)
y_new = np.arange(330, 350.2, 0.25) #(201,)
x_new = np.arange(-30, -9.8, 0.25)  #(41,)
# y_new = y
# x_new = x

# print(x_new.shape)
# print(y_new.shape)
# # # # #
#
yy,xx = np.meshgrid(y,x)              #(97, 80)
# print(xx.shape) #(201, 41)
# print(yy.shape) #(201, 41)
list1 = []
# list2 = []
# print(sst1.shape)  #(1826, 80, 97)
#
for i in range(9861):
    z1 = sst1[i,:,:] #(41, 201)
    # print(z1.shape)  #(80, 97)
    f1 = interpolate.interp2d(y,x,z1, kind='cubic')
# # #
# # #
    for j in y_new:
        for k in x_new:
            sst0 = f1(j,k)
            list1.append(sst0)
# # # #
# # # #
sst00 = np.array(list1)
# print(sst00.shape) #(2410650, 1)
sst00 = sst00.reshape(9861,81,81)
# print(sst00)
print(np.mean(sst00))
print(np.max(sst00))
print(np.min(sst00))


def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred=y_preds, y_true=y_true))
# sst_obser = sst_obser[5844:9131,:,:]
# analysis_temp = analysis_temp[5844:9131,:,:]
a = sst00.reshape(-1,1)
b = sst1.reshape(-1,1)

s = rmse(a,b)
print('RMSE: {:.3f}'.format(s))
# #
# mld00 = np.array(list2)
# print(mld00.shape) #(2410650, 1)
# mld00 = mld00.reshape(324,6,27)
# print(mld00)
# print(np.mean(mld00))
# print(np.max(mld00))
# print(np.min(mld00))
# # #
#对维度倒叙
# list3 = []
# list4 = []
# for i in range(200):
#     for j in range(41):
#         for k in range(201):
#             list3.append(sst00[i, j, -k - 1])
#
# sst000 = np.array(list3)
# sst000 = sst000.reshape(200,41,201)
# # print(sst000)
# # print(sst000)
# print(np.mean(sst000))
# print(np.max(sst000))
# print(np.min(sst000))
#
# c = sst000.reshape(-1,1)
# b = sst1.reshape(-1,1)
#
# s1 = rmse(c,b)
# print('RMSE1: {:.3f}'.format(s1))

#
#
np.savez(r'D:\heat_wave\atlantic\obser_sst_1993_2019_expand_atlantic_area_chazhi_reverse_surface.npz', lat = x_new, lon = y_new,sst = sst00)


