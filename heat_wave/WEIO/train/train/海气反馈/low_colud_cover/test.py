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
    def __init__(self,input_dim, input_dim1, hidden_dim, kernel_size1,padding1,kernel_size2,padding2,kernel_size3,padding3,kernel_size4,padding4):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.input_dim1 = input_dim1

        self.hidden_dim = hidden_dim
        self.kernel_size1 = kernel_size1
        self.padding1 = padding1

        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4

        self.padding2 = padding2
        self.padding3 = padding3
        self.padding4 = padding4

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
        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels=32,
        #                                      out_channels=1,
        #                                      kernel_size=kernel_size4,
        #                                      padding=self.padding4,
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
        # out = self.conv4(out)
        # out1 = self.conv22(out1)
        # out = self.conv4(out)
        return  out


data = np.load(r'D:\heat_wave\WEIO\expand_WEIO\last\optimized_formula_all_variables_expand_WEIO_surface_deepchazhi_1.npz')
print(data.files) # ['sst_obser', 'slfh', 'sshf', 'ssr', 'str', 'mld', 'analysis_temp', 'u', 'v', 'T_d', 'u_d', 'v_d', 'xx', 'yy', 'dT_dt', 'dT_dx', 'dT_dy']

sst_obser = data['sst_obser'][:,:64,:48] # (9861, 41, 201)  #
print(sst_obser.shape) # (9861, 41, 201)
slfh = data['slfh'][:,:64,:48]  # (9861, 41, 201)
sshf = data['sshf'][:,:64,:48]  # (9861, 41, 201)
ssr = data['ssr'][:,:64,:48]  # (9861, 41, 201)   # 可惜这个是 0
str = data['str'][:,:64,:48] # (9861, 41, 201)
mld = data['mld'][:,:64,:48]  # (9861, 41, 201)
analysis_temp = data['analysis_temp'][:,:64,:48] # (9861, 41, 201)
u = data['u'][:,:64,:48] # (9861, 41, 201)
v = data['v'][:,:64,:48] # (9861, 41, 201)
T_d = data['T_d'][:,:64,:48] # (9861, 41, 201)
u_d = data['u_d'][:,:64,:48] # (9861, 41, 201)
v_d = data['v_d'][:,:64,:48] # (9861, 41, 201)
xx = data['xx'][:,:64,:48] # (9861, 41, 201)
yy = data['yy'][:,:64,:48] # (9861, 41, 201)
print(np.min(T_d))
print(np.max(T_d))
# print(xx)  # -5 -4.75 ... 4.75
# print(yy)  # 190  190.25 ... 239.75

dT_dt = data['dT_dt'][:,:64,:48]
dT_dx = data['dT_dx'][:,:64,:48]
dT_dy = data['dT_dy'][:,:64,:48]

dT_dt0 = torch.zeros(1,64,48)
dT_dt = np.concatenate((dT_dt,dT_dt0),axis=0)

data1 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\海气反馈\low_cloud_cover_WEIO_chazhi.npz')
print(data1.files) #['time', 'lat', 'lon', 'eastward_wind', 'northward_wind', 'wind_vector_divergence', 'wind_stress', 'surface_downward_eastward_stress', 'surface_downward_northward_stress', 'wind_stress_curl', 'wind_stress_divergence', 'wind_speed_rms', 'eastward_wind_rms', 'northward_wind_rms', 'sampling_length', 'surface_type']
low_cloud_cover = data1['low_cloud_cover'][:,:64,:48] # (9848, 66, 49)


train_size = 5920  # 前60%
valid_size = 7872  # 中间20%   作为验证    剩下的20%的作为测试

dT_dt = dT_dt.reshape(-1,1,64,48)
dT_dt = torch.Tensor(dT_dt)
dT_dt_train = dT_dt[0:train_size,:,:,:]
dT_dt_valid = dT_dt[train_size:valid_size,:,:,:]
dT_dt_test = dT_dt[valid_size:,:,:,:]

dT_dx = dT_dx.reshape(-1,1,64,48)
dT_dx = torch.Tensor(dT_dx)
dT_dx_train = dT_dx[0:train_size,:,:,:]
dT_dx_valid = dT_dx[train_size:valid_size,:,:,:]
dT_dx_test = dT_dx[valid_size:,:,:,:]

dT_dy = dT_dy.reshape(-1,1,64,48)
dT_dy = torch.Tensor(dT_dy)
dT_dy_train = dT_dy[0:train_size,:,:,:]
dT_dy_valid = dT_dy[train_size:valid_size,:,:,:]
dT_dy_test = dT_dy[valid_size:,:,:,:]

Q_net = (slfh + sshf + ssr + str)

# hycom_temp = hycom_temp.reshape(-1,1,7,40,80)
# hycom_temp = torch.Tensor(hycom_temp)
# hycom_temp_train = hycom_temp[0:train_size,:,:,:]
# hycom_temp_valid = hycom_temp[train_size:valid_size,:,:,:]
# hycom_temp_test = hycom_temp[valid_size:,:,:,:]

# dT_dt1 = hycom_temp[:,1,:,:] - hycom_temp[:,0,:,:]   # (2,3,4,5,6,7,8)      - (1,2,3,4,5,6,7)    #dT_dt1.shape:torch.Size([3281, 41, 81])
# dT_dt2 = hycom_temp[:,2,:,:] - hycom_temp[:,1,:,:]   # (3,4,5,6,7,8,9)      - (2,3,4,5,6,7,8)
# dT_dt3 = hycom_temp[:,3,:,:] - hycom_temp[:,2,:,:]   # (4,5,6,7,8,9,10)     - (3,4,5,6,7,8,9)
# dT_dt4 = hycom_temp[:,4,:,:] - hycom_temp[:,3,:,:]   # (5,6,7,8,9,10,11)    - (4,5,6,7,8,9,10)
# dT_dt5 = hycom_temp[:,5,:,:] - hycom_temp[:,4,:,:]   # (6,7,8,9,10,11,12)   - (5,6,7,8,9,10,11)
# dT_dt6 = hycom_temp[:,6,:,:] - hycom_temp[:,5,:,:]   # (7,8,9,10,11,12,13)  - (6,7,8,9,10,11,12)
#
# dT_dt1 = dT_dt1.reshape(-1,1,41,81)
# dT_dt2 = dT_dt2.reshape(-1,1,41,81)
# dT_dt3 = dT_dt3.reshape(-1,1,41,81)
# dT_dt4 = dT_dt4.reshape(-1,1,41,81)
# dT_dt5 = dT_dt5.reshape(-1,1,41,81)
# dT_dt6 = dT_dt6.reshape(-1,1,41,81)
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
sst_obser = np.array(sst_obser)
sst_obser = sst_obser.reshape(-1,1,64,48)
sst_obser = torch.Tensor(sst_obser)
sst_obser_train = sst_obser[0:train_size,:,:,:]
sst_obser_valid = sst_obser[train_size:valid_size,:,:,:]
sst_obser_test = sst_obser[valid_size:,:,:,:]

Q_net = Q_net.reshape(-1,1,64,48)
Q_net = torch.Tensor(Q_net)
Q_net_train = Q_net[0:train_size,:,:,:]
Q_net_valid = Q_net[train_size:valid_size,:,:,:]
Q_net_test = Q_net[valid_size:,:,:,:]


mld = mld.reshape(-1,1,64,48)
mld = torch.Tensor(mld)
mld_train = mld[0:train_size,:,:,:]
mld_valid = mld[train_size:valid_size,:,:,:]
mld_test = mld[valid_size:,:,:,:]

analysis_temp = analysis_temp.reshape(-1,1,64,48)
analysis_temp = torch.Tensor(analysis_temp)

analysis_temp_train = analysis_temp[0:train_size,:,:,:]
analysis_temp_valid = analysis_temp[train_size:valid_size,:,:,:]
analysis_temp_test = analysis_temp[valid_size:,:,:,:]


u = u.reshape(-1,1,64,48)
u = torch.Tensor(u)
u_train = u[0:train_size,:,:,:]
u_valid = u[train_size:valid_size,:,:,:]
u_test = u[valid_size:,:,:,:]

v = v.reshape(-1,1,64,48)
v = torch.Tensor(v)
v_train = v[0:train_size,:,:,:]
v_valid = v[train_size:valid_size,:,:,:]
v_test = v[valid_size:,:,:,:]

T_d = T_d.reshape(-1,1,64,48)
T_d = torch.Tensor(T_d)
T_d_train = T_d[0:train_size,:,:,:]
T_d_valid = T_d[train_size:valid_size,:,:,:]
T_d_test = T_d[valid_size:,:,:,:]


u_d = u_d.reshape(-1,1,64,48)
u_d = torch.Tensor(u_d)
u_d_train = u_d[0:train_size,:,:,:]
u_d_valid = u_d[train_size:valid_size,:,:,:]
u_d_test = u_d[valid_size:,:,:,:]


v_d = v_d.reshape(-1,1,64,48)
v_d = torch.Tensor(v_d)
v_d_train = v_d[0:train_size,:,:,:]
v_d_valid = v_d[train_size:valid_size,:,:,:]
v_d_test = v_d[valid_size:,:,:,:]


xx = xx.reshape(-1,1,64,48)
xx = torch.Tensor(xx)
xx_train = xx[0:train_size,:,:,:]
xx_valid = xx[train_size:valid_size,:,:,:]
xx_test = xx[valid_size:,:,:,:]


yy = yy.reshape(-1,1,64,48)
yy = torch.Tensor(yy)
yy_train = yy[0:train_size,:,:,:]
yy_valid = yy[train_size:valid_size,:,:,:]
yy_test = yy[valid_size:,:,:,:]

low_cloud_cover = low_cloud_cover.reshape(-1,1,64,48)
low_cloud_cover = torch.Tensor(low_cloud_cover)
low_cloud_cover_train = low_cloud_cover[0:train_size,:,:,:]
low_cloud_cover_valid = low_cloud_cover[train_size:valid_size,:,:,:]
low_cloud_cover_test = low_cloud_cover[valid_size:,:,:,:]


# train_data = torch.cat((sst_obser_train, Q_net_train, mld_train, u_train, v_train, T_d_train, v_d_train, xx_train, yy_train, dT_dt_train, dT_dx_train, dT_dy_train, eastward_wind_train, northward_wind_train,
#                         surface_downward_eastward_stress_train, surface_downward_northward_stress_train), dim=1) # train_data.shape:torch.Size([1952, 12, 7, 40, 80])  # 数据  第一维时间  第二维变量和变量 第三维7dat seq  4  5lat lon
#
#
# valid_data = torch.cat((sst_obser_valid, Q_net_valid, mld_valid, u_valid, v_valid, T_d_valid, v_d_valid, xx_valid, yy_valid, dT_dt_valid, dT_dx_valid, dT_dy_valid,
#                         eastward_wind_valid, northward_wind_valid, surface_downward_eastward_stress_valid, surface_downward_northward_stress_valid), dim=1) # valid_data.shape:torch.Size([672, 12, 32, 40, 80])

sst_obser_test = sst_obser_test[:1976,:,:,:].reshape(-1,1,64,48)
Q_net_test = Q_net_test[:1976,:,:,:].reshape(-1,1,64,48)
mld_test = mld_test[:1976,:,:,:].reshape(-1,1,64,48)
u_test = u_test[:1976,:,:,:].reshape(-1,1,64,48)
v_test = v_test[:1976,:,:,:].reshape(-1,1,64,48)
T_d_test = T_d_test[:1976,:,:,:].reshape(-1,1,64,48)
v_d_test = v_d_test[:1976,:,:,:].reshape(-1,1,64,48)
xx_test = xx_test[:1976,:,:,:].reshape(-1,1,64,48)
yy_test = yy_test[:1976,:,:,:].reshape(-1,1,64,48)
dT_dt_test = dT_dt_test[:1976,:,:,:].reshape(-1,1,64,48)
dT_dx_test = dT_dx_test[:1976,:,:,:].reshape(-1,1,64,48)
dT_dy_test = dT_dy_test[:1976,:,:,:].reshape(-1,1,64,48)
low_cloud_cover_test = low_cloud_cover_test[:1976,:,:,:].reshape(-1,1,64,48)
test_data = torch.cat((sst_obser_test, Q_net_test, mld_test, u_test, v_test, T_d_test, v_d_test, xx_test, yy_test, dT_dt_test, dT_dx_test, dT_dy_test
                       ,low_cloud_cover_test), dim=1) # test_data.shape:torch.Size([657, 12, 7, 40, 80])



test_data = test_data[:1976,:,:,:]
train_label = analysis_temp[1:train_size + 1,:,:,:]  # train_label.shape:torch.Size([1952, 1, 7, 40, 80])

valid_label = analysis_temp[train_size + 1: valid_size + 1,:,:,:] # test_label.shape:torch.Size([672, 1, 7, 40, 80])

test_label = analysis_temp[valid_size + 1:,:,:,:] # test_label.shape:torch.Size([650, 1, 7, 40, 80])
test_label = test_label[:1976,:,:,:]


train_label = torch.Tensor(train_label)
train_label = train_label[:]
valid_label = torch.Tensor(valid_label)
valid_label = valid_label[:]
test_label = torch.Tensor(test_label)
test_label = test_label[:]
print('test_label.shape:{}'.format(test_label.shape)) # test_label.shape:torch.Size([1988, 1, 40, 200])

#构建数据管道
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


batch_size1 = 32
batch_size2 = 32
batch_size3 = 3000

# trainset = MyDataset(train_data, train_label)
# trainloader = DataLoader(trainset, batch_size=batch_size1, shuffle=True, drop_last=False,pin_memory=True, num_workers=0)
#
# validset = MyDataset(valid_data, valid_label)
# validloader = DataLoader(validset, batch_size=batch_size2, shuffle=True, drop_last=False,pin_memory=True, num_workers=0)

testset = MyDataset(test_data, test_label)
testloader = DataLoader(testset, batch_size=batch_size3, shuffle=False, drop_last=False,pin_memory=True, num_workers=0)

pred_val = np.zeros((1952, 1, 64, 48))

# 计算温度关于时间的梯度    这部分 分部分用神经网络描述
# dT_dt = Q/(p * C_p * h_m)   +  u * (d_T / d_x)  + v * (d_T / d_y) + w_e * ((T_m - T_d) / h_m)


# 积分   这部分 参与 loss计算   可以向后积分7天  也可以向后积分一天
# dT_1 = Q1/(p * C_p * h_m1) + u1 * (d_T / d_x1)  + v1 * (d_T / d_y1) + w_e1 * ((T_m1 - T_d1) / h_m1)
# dT_2 = Q2/(p * C_p * h_m1) + u2 * (d_T / d_x1)  + v2 * (d_T / d_y1) + w_e2 * ((T_m1 - T_d2) / h_m2)
# dT_3 = Q3/(p * C_p * h_m1) + u3 * (d_T / d_x1)  + v3 * (d_T / d_y1) + w_e3 * ((T_m1 - T_d3) / h_m3)
# dT_4 = Q4/(p * C_p * h_m1) + u4 * (d_T / d_x1)  + v4 * (d_T / d_y1) + w_e4 * ((T_m1 - T_d4) / h_m4)
# dT_5 = Q5/(p * C_p * h_m1) + u5 * (d_T / d_x1)  + v5 * (d_T / d_y1) + w_e5 * ((T_m1 - T_d5) / h_m5)
# dT_6 = Q6/(p * C_p * h_m1) + u6 * (d_T / d_x1)  + v6 * (d_T / d_y1) + w_e6 * ((T_m1 - T_d6) / h_m6)

# T_1取 hycom的1;
# T_2 = T_1 + dT_1
# T_3 = T_1 + dT_1 + dT_2
# T_4 = T_1 + dT_1 + dT_2 + dT_3
# T_5 = T_1 + dT_1 + dT_2 + dT_3 + dT_4
# T_6 = T_1 + dT_1 + dT_2 + dT_3 + dT_4 + dT_5
# T_7 = T_1 + dT_1 + dT_2 + dT_3 + dT_4 + dT_5 + dT_6

# T_8 = 1/18 * (T_0 + 4T_1 + 2T_2 + 4T_3 + 2T_4 + 4T_5 + 2T_6 + T_7)
for seed in range(1,6):
    for epoches in range(100,101,100):

        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True


        checkpoint = torch.load(
            r'C:\Users\mymylove\Desktop\zege\heat_wave\WEIO\train\正式训练\海气反馈\low_colud_cover\cnn_guiding_intuotion_f(1)_with_low_cloud_cover_epo{}_lay3_lr1e-3_e{}_1day_model_weights.pth'.format(epoches,seed))

        # 设置随机数种子
        setup_seed(seed)


        torch.backends.cudnn.enabled = False

        model = ConvNet(input_dim=1, input_dim1=7, hidden_dim=1, kernel_size1=(1, 3), padding1=(0, 1),
                        kernel_size2=(1, 3), padding2=(0, 1), kernel_size3=(1, 3), padding3=(0, 1), kernel_size4=(1, 3),
                        padding4=(0, 1)) # padding2=(0, 1, 1), kernel_size3=(1, 3, 3), padding3=(0, 1, 1)).cuda()

        model.load_state_dict(checkpoint['state_dict'])

        criterion = nn.MSELoss()
        epochs = epoches

        sores = []
        def rmse(y_true, y_preds):
            return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))


        for i, data in tqdm(enumerate(testloader)):
            data, label = data

            data = data
            label = label
            T_m = data[:, 0, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            Q = data[:, 1, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            h_m = data[:, 2, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            u = data[:, 3, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            v = data[:, 4, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            T_d = data[:, 5, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            w_e = data[:, 6, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            xx = data[:, 7, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            yy = data[:, 8, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            dT_dt = data[:, 9, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            dT_dx = data[:, 10, :, :].reshape(-1, 1, 64, 48)  # ([32, 7, 40, 80])
            dT_dy = data[:, 11, :, :].reshape(-1, 1, 64, 48)
            low_cloud_cover = data[:, 12, :, :].reshape(-1, 1, 64, 48)


            #traditional

            data1 = Q / (1025 * 4000 * h_m)+ low_cloud_cover #1988

            data1 = data1.reshape(-1, 1, 64, 48)

            out = model(data1)

            # for alfah in np.arange(1):
                # print(u * (dT_dx))
            dT_dt1 = u * (dT_dx)  + v * (dT_dy)  + w_e * ((T_m - T_d) / h_m)  + out # torch.Size([32, 1, 40, 80])

            dT_dt1 = dT_dt1[:,:,:,:]
            T_1 = T_m.reshape(-1, 1, 64, 48)  # hycom  假设为当前时刻的观测
            T_1 = T_1[:,:,:,:]
            # print(dT_dt1)
            T_2 = T_1 + dT_dt1  # torch.Size([32, 1, 40, 80])
            # print(T_2)
            # label 取为下一时刻



            losses = 0
            loss = criterion(T_2, label)
            losses += loss

            pred_test1 = T_2.reshape(-1, 1).detach().numpy()
            test_label = label.reshape(-1, 1).detach().numpy()

            s = rmse(test_label, pred_test1)
            print('epo:{},e:{},RMSE: {:.3f}'.format(epoches,seed,s))

            # np.save("plot_pinn_pred.npy", plot_pinn_pred)

            test_label = np.array(test_label)
            pred_test1 = np.array(pred_test1)
            MAE = np.mean(np.abs(test_label - pred_test1))
            print('epo:{}, e:{},MAE: {:.3f}'.format(epoches, seed,MAE))

            MSE = np.mean(np.square(test_label - pred_test1))
            print('epo:{}, e:{},MSE: {:.3f}'.format(epoches, seed,MSE))

            MAPE = np.mean(np.abs((test_label - pred_test1) / test_label))
            print('epo:{}, e:{},MAPE: {:.3f}'.format(epoches, seed,MAPE))

            test_label1 = np.array((test_label))
            pred_test1 = np.array(pred_test1)

            x_hat = np.mean(test_label1)
            # print(x_hat)
            x = test_label1
            y_hat = np.mean(pred_test1)
            y = pred_test1

            fenzi = np.sum((x - x_hat) * (y - y_hat))
            fenmu = np.sqrt((np.sum(np.square(x - x_hat))) * (np.sum(np.square(y - y_hat))))

            cor = fenzi / fenmu
            print('epo:{}, e:{},cor: {:.3f}'.format(epoches, seed,cor))


            # Q_net
