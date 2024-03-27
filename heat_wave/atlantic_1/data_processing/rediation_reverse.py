import numpy as np

data1 = np.load(r'D:\heat_wave\atlantic\expand_area1\EC_slhf_93_19_atlantic_expand_ocean_area.npz')
lat = data1['lat'][:]   # 5  4.75 .... -4.75  -5
lon = data1['lon'][:]

data2 = np.load(r'D:\heat_wave\atlantic\expand_area1\EC_sshf_93_19_atlantic_expand_ocean_area.npz')
data3 = np.load(r'D:\heat_wave\atlantic\expand_area1\EC_ssr_93_19_atlantic_expand_ocean_area.npz')
data4 = np.load(r'D:\heat_wave\atlantic\expand_area1\EC_str_93_19_atlantic_expand_ocean_area.npz')

slfh = data1['slhf'][:]
sshf = data2['sshf'][:]
ssr = data3['ssr'][:]
str = data4['str'][:]
print(slfh.shape)


# #对维度倒叙
list1 = []
list2 = []
list3 = []
list4 = []
for i in range(9861):
    for j in range(81):
        for k in range(81):
            list1.append(slfh[i, -j - 1, k])
            list2.append(sshf[i, -j - 1, k])
            list3.append(ssr[i, -j - 1, k])
            list4.append(str[i, -j - 1, k])

slfh000 = np.array(list1)
sshf000 = np.array(list2)
ssr000 = np.array(list3)
str000 = np.array(list4)

slfh000 = slfh000.reshape(9861,81,81)
sshf000 = sshf000.reshape(9861,81,81)
ssr000 = ssr000.reshape(9861,81,81)
str000 = str000.reshape(9861,81,81)
#
#
# #
np.savez(r'D:\heat_wave\atlantic\expand_area1\radiation_93_19_expand_atlantic1_ocean_last.npz', lat = lat, lon = lon, slfh = slfh000, sshf = sshf000, ssr = ssr000, str = str000)