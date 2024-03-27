import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

mm = MinMaxScaler()



class ConvNet(nn.Module):
    def __init__(self,input_dim, input_dim1, hidden_dim, kernel_size1,padding1,kernel_size2,padding2,kernel_size3,padding3):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.input_dim1 = input_dim1

        self.hidden_dim = hidden_dim
        self.kernel_size1 = kernel_size1
        self.padding1 = padding1

        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.padding2 = padding2
        self.padding3 = padding3
        #stage1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.input_dim ,
                              out_channels=16,
                              kernel_size=kernel_size1,
                              padding=self.padding1,
                              ))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=kernel_size2,
                               padding=self.padding2,
                               ))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                             out_channels=1,
                                             kernel_size=kernel_size3,
                                             padding=self.padding3,
                                             ))
        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64,
        #                                      out_channels=1,
        #                                      kernel_size=kernel_size,
        #                                      padding=self.padding,
        #                                      ))
        # stage2
        # self.conv22 = nn.Sequential(nn.Conv3d(in_channels=16,
        #                                      out_channels=32,
        #                                      kernel_size=kernel_size2,
        #                                      padding=self.padding2,
        #                                      ))
        # self.conv33 = nn.Sequential(nn.Conv3d(in_channels=32,
        #                                      out_channels=10,
        #                                      kernel_size=kernel_size3,
        #                                      padding=self.padding3,
        #                                      ))
        # # self.fc = nn.Linear

 # 前馈网络过程
    def forward(self, x):  #x 为(batch, channel, h, w)

        out = self.conv1(x) #x 为(batch, channel, h, w)
        out = self.conv2(out)
        out = self.conv3(out)
        # out1 = self.conv22(out1)
        # out = self.conv4(out)
        return  out


# 只要表面数据
data1 = np.load(r'D:\heat_wave\atlantic\expand_area1\radiation_93_19_expand_atlantic1_ocean_last.npz')
print(data1.files)  # ['lat', 'lon', 'slfh', 'sshf', 'ssr', 'str']

slfh = data1['slfh'][:,:,1:]  # (9861, 81, 81)
sshf = data1['sshf'][:,:,1:]  # (9861, 81, 81)
ssr = data1['ssr'][:,:,1:]  # (9861, 81, 81)
str = data1['str'][:,:,1:]  # (9861, 81, 81)

lon_radiation = data1['lon'][:]  # -35 -34.75  -29.5 ... -10.5  -15.25 -15  ---> 325 325.25 325.5 ... 345.5  345.75  345
lat_radiation = data1['lat'][:]  # -10 -9.75 ... 9.75 10
print(sshf.shape)

data2 = np.load(r'D:\heat_wave\atlantic\expand_area1\SST_93_19_expand_atlantic_1_area.npz')
print(data2.files)  # ['time', 'lat', 'lon', 'sst']

sst_obser = data2['sst'][:,:81,:81] # (9861, 81, 80)

lon_obser = data2['lon'][:] # 325.375  325.625 ... 344.875 345.125
lat_obser = data2['lat'][:] # -10

data3 = np.load(r'D:\heat_wave\atlantic\expand_area1\T_d_u_d_v_d_93_19_deep_expand_atlantic1_last.npz')
print(data3.files) #['T_d', 'u_d', 'v_d']

T_d = data3['T_d'][:] # (9861, 81, 80)
u_d = data3['u_d'][:] # (9861, 81, 80)
v_d = data3['v_d'][:] # (9861, 81, 80)
print(T_d.shape)

data4 = np.load(r'D:\heat_wave\atlantic\expand_area1\reanalysis_data_1993-2019_expand_atlantic_area1.npz')
print(data4.files) #['time', 'lat', 'lon', 'mld', 'salinity', 'temp', 'u', 'v']


mld = data4['mld'][:,:,:]  # (9861, 40, 81)
analysis_temp = data4['temp'][:,0,:,:] # (9861, 81, 80)
u = data4['u'][:,0,:,:] # (9861, 30, 81, 81)
v = data4['v'][:,0,:,:] # (9861, 30, 81, 81)
lon_ana = data4['lon'][:] # -34.75 -34.5 ... -15.25 -15
print(lon_ana)
print('analysis_temp.shape:{}'.format(analysis_temp.shape))
#
# 用再分析计算 d_dt  d_dx  d_dy

#1) d_dt
analysis_temp_to0 = analysis_temp[:9860,:,:] #(9860, 81, 81)
analysis_temp_to1 = analysis_temp[1:,:,:] #(9860, 81, 81)

dT_dt = analysis_temp_to1 - analysis_temp_to0 #(9860, 81, 81)

y = np.arange(325.25, 345.2, 0.25) #(81,)
x = np.arange(-10, 10.2, 0.25)  #(81,)

yy, xx = np.meshgrid(y, x)
xx = np.expand_dims(xx, 0).repeat(9861, axis=0)
yy = np.expand_dims(yy, 0).repeat(9861, axis=0)

# 2)d_dx  # 纬向  维度  -30 --- -10
list = []
for i in range(80):
    xx_i = xx[:,i,:]
    xx_j = xx[:,i+1,:]
    T_i = analysis_temp[:,i,:]
    T_j = analysis_temp[:,i+1,:]
    dT_dx = (T_j - T_i) / (xx_j - xx_i)
    list.append(dT_dx)
    # print(dT_dx.shape)
a = np.array(list)
print('a.shape:{}'.format(a.shape)) # a.shape:(80, 9861, 81)

dT_dx = a.transpose(1,0,2)
print(dT_dx.shape)  # (3281, 7, 40, 81)

# 3)d_dy  # 经向  维度  330 --- 350
list1 = []
for j in range(79):
    yy_i = yy[:,:,j]
    yy_j = yy[:,:,j+1]
    T_i = analysis_temp[:,:,j]
    T_j = analysis_temp[:,:,j+1]
    dT_dy = (T_j - T_i) / (yy_j - yy_i)
    list1.append(dT_dy)
    # print(dT_dy.shape)
b = np.array(list1)
print(b.shape) # (80, 9861, 81)

dT_dy = b.transpose(1,2,0)
print(dT_dy.shape)  # (3281, 7, 41, 80)

np.savez(r'D:\heat_wave\atlantic\expand_area1\last\optimized_formula_all_variables_expand_atlantic1_surface.npz',
         slfh = slfh, sshf = sshf, ssr = ssr, str = str, sst_obser = sst_obser, T_d = T_d, u_d = u_d, v_d = v_d, mld = mld, analysis_temp = analysis_temp, u = u, v = v,
         dT_dt = dT_dt, xx =xx, yy = yy, dT_dx = dT_dx, dT_dy = dT_dy )


train_size = 5920  # 前60%
valid_size = 7872  # 中间20%   作为验证    剩下的20%的作为测试


Q_net = (slfh + sshf + ssr + str)


# dT_dt1 = hycom_temp[:,1,:,:] - hycom_temp[:,0,:,:]   # (2,3,4,5,6,7,8)      - (1,2,3,4,5,6,7)    #dT_dt1.shape:torch.Size([3281, 41, 81])

# dT_dt1 = dT_dt1.reshape(-1,1,41,81)

#
# dT_dt = np.concatenate((dT_dt1, dT_dt2, dT_dt3, dT_dt4, dT_dt5, dT_dt6), axis=1)  # dT_dt.shape:(3281, 6, 41, 81)   N, time, lat, lon
#
# print('dT_dt.shape:{}'.format(dT_dt.shape))
#
# list = []
# for i in range(40):
#     xx_i = xx[:,:,i,:]
#     xx_j = xx[:,:,i+1,:]
#     T_i = xx[:,:,i,:]
#     T_j = xx[:,:,i+1,:]
#     dT_dx = (T_j - T_i) / (xx_j - xx_i)
#     list.append(dT_dx)
#     print(dT_dx.shape)
# a = np.array(list)
# print(a.shape) # (40, 3281, 7, 81)
#
# dT_dx = a.transpose(1,2,0,3)
# print(dT_dx.shape)  # (3281, 7, 40, 81)
#
#
# list1 = []
# for j in range(80):
#     yy_i = yy[:,:,:,j]
#     yy_j = yy[:,:,:,j+1]
#     T_i = yy[:,:,:,j]
#     T_j = yy[:,:,:,j+1]
#     dT_dy = (T_j - T_i) / (yy_j - yy_i)
#     list1.append(dT_dy)
#     print(dT_dy.shape)
# b = np.array(list1)
# print(b.shape) # (80, 3281, 7, 81)
#
# dT_dy = b.transpose(1,2,3,0)
# print(dT_dy.shape)  # (3281, 7, 41, 80)


# sst_obser = sst_obser.reshape(-1,1,40,200)
# sst_obser = torch.Tensor(sst_obser)
# sst_obser_train = sst_obser[0:train_size,:,:,:]
# sst_obser_valid = sst_obser[train_size:valid_size,:,:,:]
# sst_obser_test = sst_obser[valid_size:,:,:,:]
#
# Q_net = Q_net.reshape(-1,1,40,200)
# Q_net = torch.Tensor(Q_net)
# Q_net_train = Q_net[0:train_size,:,:,:]
# Q_net_valid = Q_net[train_size:valid_size,:,:,:]
# Q_net_test = Q_net[valid_size:,:,:,:]
#
#
# mld = mld.reshape(-1,1,40,200)
# mld = torch.Tensor(mld)
# mld_train = mld[0:train_size,:,:,:]
# mld_valid = mld[train_size:valid_size,:,:,:]
# mld_test = mld[valid_size:,:,:,:]
#
# analysis_temp = analysis_temp.reshape(-1,1,40,200)
# analysis_temp = torch.Tensor(analysis_temp)
#
# analysis_temp_train = analysis_temp[0:train_size,:,:,:]
# analysis_temp_valid = analysis_temp[train_size:valid_size,:,:,:]
# analysis_temp_test = analysis_temp[valid_size:,:,:,:]
#
#
# u = u.reshape(-1,1,40,200)
# u = torch.Tensor(u)
# u_train = u[0:train_size,:,:,:]
# u_valid = u[train_size:valid_size,:,:,:]
# u_test = u[valid_size:,:,:,:]
#
# v = v.reshape(-1,1,40,200)
# v = torch.Tensor(v)
# v_train = v[0:train_size,:,:,:]
# v_valid = v[train_size:valid_size,:,:,:]
# v_test = v[valid_size:,:,:,:]
#
# T_d = T_d.reshape(-1,1,40,200)
# T_d = torch.Tensor(T_d)
# T_d_train = T_d[0:train_size,:,:,:]
# T_d_valid = T_d[train_size:valid_size,:,:,:]
# T_d_test = T_d[valid_size:,:,:,:]
#
#
# u_d = u_d.reshape(-1,1,40,200)
# u_d = torch.Tensor(u_d)
# u_d_train = u_d[0:train_size,:,:,:]
# u_d_valid = u_d[train_size:valid_size,:,:,:]
# u_d_test = u_d[valid_size:,:,:,:]
#
#
# v_d = v_d.reshape(-1,1,40,200)
# v_d = torch.Tensor(v_d)
# v_d_train = v_d[0:train_size,:,:,:]
# v_d_valid = v_d[train_size:valid_size,:,:,:]
# v_d_test = v_d[valid_size:,:,:,:]
#
#
# xx = xx.reshape(-1,1,40,200)
# xx = torch.Tensor(xx)
# xx_train = xx[0:train_size,:,:,:]
# xx_valid = xx[train_size:valid_size,:,:,:]
# xx_test = xx[valid_size:,:,:,:]
#
#
# yy = yy.reshape(-1,1,40,200)
# yy = torch.Tensor(yy)
# yy_train = yy[0:train_size,:,:,:]
# yy_valid = yy[train_size:valid_size,:,:,:]
# yy_test = yy[valid_size:,:,:,:]
#
#
#
# train_data = torch.cat((sst_obser_train, Q_net_train, mld_train, u_train, v_train, T_d_train, v_d_train, xx_train, yy_train, dT_dt_train, dT_dx_train, dT_dy_train), dim=1) # train_data.shape:torch.Size([1952, 12, 7, 40, 80])  # 数据  第一维时间  第二维变量和变量 第三维7dat seq  4  5lat lon
#
#
# valid_data = torch.cat((sst_obser_valid, Q_net_valid, mld_valid, u_valid, v_valid, T_d_valid, v_d_valid, xx_valid, yy_valid, dT_dt_valid, dT_dx_valid, dT_dy_valid), dim=1) # valid_data.shape:torch.Size([672, 12, 32, 40, 80])
#
# test_data = torch.cat((sst_obser_test, Q_net_test, mld_test, u_test, v_test, T_d_test, v_d_test,xx_test, yy_test, dT_dt_test, dT_dx_test, dT_dy_test), dim=1) # test_data.shape:torch.Size([657, 12, 7, 40, 80])
# test_data = test_data[:1988,:,:,:]
#
# train_label = analysis_temp[1:train_size + 1,:,:,:]  # train_label.shape:torch.Size([1952, 1, 7, 40, 80])
#
# valid_label = analysis_temp[train_size + 1: valid_size + 1,:,:,:] # test_label.shape:torch.Size([672, 1, 7, 40, 80])
#
# test_label = analysis_temp[valid_size + 1:,:,:,:] # test_label.shape:torch.Size([650, 1, 7, 40, 80])
#
#
#
# train_label = torch.Tensor(train_label)
# train_label = train_label[:]
# valid_label = torch.Tensor(valid_label)
# valid_label = valid_label[:]
# test_label = torch.Tensor(test_label)
# test_label = test_label[:]
# print('test_label.shape:{}'.format(test_label.shape)) # test_label.shape:torch.Size([1988, 1, 40, 200])
#
# #构建数据管道
# class MyDataset(Dataset):
#     def __init__(self, data, label):
#         self.data = torch.Tensor(data)
#         self.label = torch.Tensor(label)
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.label[idx]
#
#
# batch_size1 = 32
# batch_size2 = 32
# batch_size3 = 3000
#
# trainset = MyDataset(train_data, train_label)
# trainloader = DataLoader(trainset, batch_size=batch_size1, shuffle=True, drop_last=False,pin_memory=True, num_workers=0)
#
# validset = MyDataset(valid_data, valid_label)
# validloader = DataLoader(validset, batch_size=batch_size2, shuffle=True, drop_last=False,pin_memory=True, num_workers=0)
#
# testset = MyDataset(test_data, test_label)
# testloader = DataLoader(testset, batch_size=batch_size3, shuffle=False, drop_last=False,pin_memory=True, num_workers=0)
#
# pred_val = np.zeros((1952, 1, 40, 200))

# # 计算温度关于时间的梯度    这部分 分部分用神经网络描述
# # dT_dt = Q/(p * C_p * h_m)   +  u * (d_T / d_x)  + v * (d_T / d_y) + w_e * ((T_m - T_d) / h_m)
#
#
# # 积分   这部分 参与 loss计算   可以向后积分7天  也可以向后积分一天
# # dT_1 = Q1/(p * C_p * h_m1) + u1 * (d_T / d_x1)  + v1 * (d_T / d_y1) + w_e1 * ((T_m1 - T_d1) / h_m1)
# # dT_2 = Q2/(p * C_p * h_m1) + u2 * (d_T / d_x1)  + v2 * (d_T / d_y1) + w_e2 * ((T_m1 - T_d2) / h_m2)
# # dT_3 = Q3/(p * C_p * h_m1) + u3 * (d_T / d_x1)  + v3 * (d_T / d_y1) + w_e3 * ((T_m1 - T_d3) / h_m3)
# # dT_4 = Q4/(p * C_p * h_m1) + u4 * (d_T / d_x1)  + v4 * (d_T / d_y1) + w_e4 * ((T_m1 - T_d4) / h_m4)
# # dT_5 = Q5/(p * C_p * h_m1) + u5 * (d_T / d_x1)  + v5 * (d_T / d_y1) + w_e5 * ((T_m1 - T_d5) / h_m5)
# # dT_6 = Q6/(p * C_p * h_m1) + u6 * (d_T / d_x1)  + v6 * (d_T / d_y1) + w_e6 * ((T_m1 - T_d6) / h_m6)
#
# # T_1取 hycom的1;
# # T_2 = T_1 + dT_1
# # T_3 = T_1 + dT_1 + dT_2
# # T_4 = T_1 + dT_1 + dT_2 + dT_3
# # T_5 = T_1 + dT_1 + dT_2 + dT_3 + dT_4
# # T_6 = T_1 + dT_1 + dT_2 + dT_3 + dT_4 + dT_5
# # T_7 = T_1 + dT_1 + dT_2 + dT_3 + dT_4 + dT_5 + dT_6
#
# # T_8 = 1/18 * (T_0 + 4T_1 + 2T_2 + 4T_3 + 2T_4 + 4T_5 + 2T_6 + T_7)
# for seed in range(1,6):
#     for epoches in range(200,201,1):
#         def setup_seed(seed):
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)
#             np.random.seed(seed)
#             random.seed(seed)
#             torch.backends.cudnn.deterministic = True
#
#
#         # 设置随机数种子
#         setup_seed(seed)
#
#         model_weights1 = './cnn_guiding_intuotion_f(2)_alpha2_epo{}_lay3_lr1e-3_e{}_1day_model_weights.pth'.format(epoches,seed)
#         torch.backends.cudnn.enabled = False
#
#         model = ConvNet(input_dim=1, input_dim1=7, hidden_dim=1, kernel_size1=(1, 3), padding1=(0, 1),
#                             kernel_size2=(1, 3), padding2=(0, 1), kernel_size3=(1, 3), padding3=(0, 1)).cuda()#                 padding2=(0, 1, 1), kernel_size3=(1, 3, 3), padding3=(0, 1, 1)).cuda()
#         criterion = nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#         epochs = epoches
#
#         train_losses, valid_losses = [], []
#         best_score = float('inf')
#         best_score1 = float('inf')
#
#         sores = []
#         def rmse(y_true, y_preds):
#             return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))
#
#         for epoch in range(epochs):
#             print('Epoch: {}/{}'.format(epoch + 1, epochs))
#             # print(var_y)
#             #模型训练
#             model.train()
#             losses = 0
#             loss1 = 0
#             for i, data in tqdm(enumerate(trainloader)):
#                 data, label = data
#                 data = data.cuda()
#                 label = label.cuda()
#                 optimizer.zero_grad()
#
#                 T_m = data[:, 0, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 Q = data[:, 1, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 h_m = data[:, 2, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 u = data[:, 3, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 v = data[:, 4, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 T_d = data[:, 5, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 w_e = data[:, 6, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 xx = data[:, 7, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 yy = data[:, 8, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 dT_dt = data[:, 9, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 dT_dx = data[:, 10, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 dT_dy = data[:, 11, :, :].reshape(-1, 1, 40, 200)
#
#
#                 #只取当前时刻预测下一时刻
#
#                 # T_m = T_m[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # Q = Q[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # h_m = h_m[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # u = u[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # v = v[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # T_d = T_d[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # w_e = w_e[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # xx = xx[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # yy = yy[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # dT_dt = dT_dt[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # dT_dx = dT_dx[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#                 # dT_dy = dT_dy[:, 0, :, :].reshape(-1,1,40,80)  # ([32, 40, 80])
#
#                 # (1) 输入为 Q/(p * C_p * h_m)  out 为 f(1)
#                 # d_t = f(1) + (2) + (3) + (4)
#
#                 # (2) 输入为 u * (d_T / d_x)  out 为 f(2)
#                 # d_t = (1) + f(2) + (3) + (4)
#
#                 # (3) 输入为 v * (d_T / d_y)  out 为 f(3)
#                 # d_t = (1) + (2) + f(3) + (4)
#
#                 # (4) 输入为 w_e * ((T_m - T_d) / h_m)  out 为 f(4)
#                 # d_t = (1) + (2) + (3) + f(4)
#
#                 # ...
#                 data1 =  u* (dT_dx)   #(1)  data1.shape:torch.Size([32, 1, 40, 80])
#
#                 data1 = data1.reshape(-1,1,40,200)
#
#                 out = model(data1)  # ---> f(1)  ([32, 1, 40, 80])
#
#                 dT_dt1 = Q / (1025 * 4000 * h_m) + v * (dT_dy) + out + w_e * ((T_m - T_d) / h_m)      #torch.Size([32, 1, 40, 80])
#
#                 T_1 = T_m.reshape(-1,1,40,200)  #hycom  假设为当前时刻的观测
#
#                 T_2 = T_1 + dT_dt1   #torch.Size([32, 1, 40, 80])
#                 # label 取为下一时刻
#                 # print('label.shape:{}'.format(label.shape))
#                 loss = criterion(T_2, label)
#
#                 losses += loss
#
#                 # 反向传播
#                 loss.backward()
#                 optimizer.step()
#             train_loss = losses / len(trainloader)
#             train_losses.append(train_loss)
#             print('Training Loss: {:.10f}'.format(train_loss))
#
#
#             #模型验证
#             model.eval()
#             losses = 0
#
#             for i, data in tqdm(enumerate(validloader)):
#                 data, label = data
#                 data = data.cuda()
#                 label = label.cuda()
#                 optimizer.zero_grad()
#
#                 T_m = data[:, 0, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 Q = data[:, 1, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 h_m = data[:, 2, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 u = data[:, 3, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 v = data[:, 4, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 T_d = data[:, 5, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 w_e = data[:, 6, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 xx = data[:, 7, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 yy = data[:, 8, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 dT_dt = data[:, 9, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 dT_dx = data[:, 10, :, :].reshape(-1, 1, 40, 200)  # ([32, 7, 40, 80])
#                 dT_dy = data[:, 11, :, :].reshape(-1, 1, 40, 200)
#
#                 # 只取当前时刻预测下一时刻
#
#                 # T_m = T_m[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # Q = Q[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # h_m = h_m[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # u = u[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # v = v[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # T_d = T_d[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # w_e = w_e[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # xx = xx[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # yy = yy[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # dT_dt = dT_dt[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # dT_dx = dT_dx[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#                 # dT_dy = dT_dy[:, 0, :, :].reshape(-1, 1, 40, 80)  # ([32, 40, 80])
#
#                 data1 = u* (dT_dx)   #(1)  data1.shape:torch.Size([32, 1, 40, 80])
#
#                 data1 = data1.reshape(-1, 1, 40, 200)
#
#                 out = model(data1)  # ---> f(1)  ([32, 1, 40, 80])
#
#                 dT_dt1 = Q / (1025 * 4000 * h_m) + v * (dT_dy) + w_e * ((T_m - T_d) / h_m) + out     # torch.Size([32, 1, 40, 80])
#
#                 T_1 = T_m.reshape(-1, 1, 40, 200)  # hycom  假设为当前时刻的观测
#
#                 T_2 = T_1 + dT_dt1  # torch.Size([32, 1, 40, 80])
#                 # label 取为下一时刻
#                 # print('label.shape:{}'.format(label.shape))
#                 loss = criterion(T_2, label)
#                 losses += float(loss)
#
#                 T_2 = T_2.detach().cpu().numpy()
#                 pred_val[i * batch_size2:(i + 1) * batch_size2] = np.array(T_2)
#
#             valid_loss = losses / len(validloader)
#             valid_losses.append(valid_loss)
#
#
#             valid_label = valid_label.reshape(-1,1)
#             # a = np.mean(valid_label)
#
#             T_2 = pred_val.reshape(-1,1)
#             s = rmse(valid_label, T_2)
#             sores.append(s)
#             print('Score:{:.3f}'.format(s))
#
#             if valid_loss < best_score1:  # 求s的最小值  ---》最大值反过来  inf符号也要反过来
#                 best_score1 = valid_loss
#                 checkpoint = {'best_score': valid_loss,
#                               'state_dict': model.state_dict()}
#                 torch.save(checkpoint, model_weights1)  # if valid_loss < best_loss:
#                 best_loss = valid_loss
#                 torch.save(model.state_dict(),
#                            './cnn_guiding_intuition_lr1e-3_f(2)_alpha2_model_{}_layer3_1day_e{}.pt'.format(epoches, seed))
#             # checkpoint = {'state_dict': model.state_dict()}
#             # torch.save(checkpoint, model_weights1)  # if valid_loss < best_loss:
#             # torch.save(model.state_dict(),
#             #        './cnn_guiding_intuition_lr1e-3_f(1)_month_nino_model_{}_layer3_1day_e{}.pt'.format(epoches, seed))
#         # print(sores)
#         print(sores)
#         print(best_score)
#         print(s)


