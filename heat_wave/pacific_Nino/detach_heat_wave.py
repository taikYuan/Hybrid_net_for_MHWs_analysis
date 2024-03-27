import numpy as np

# 加载数据
sst = np.load(r'D:\MHW_TC_SO\oisst_93_22_south_ocean.npz')['sst']
sst = sst[:]
print(sst.shape) #(10957, 1, 81, 81)
sst = np.squeeze(sst)
data = sst
# data = np.load('data.npz')['arr_0']
n_days, n_lat, n_lon = data.shape
print(data.shape) # (10957, 81, 81)
data = np.where(data<-100, np.nan, data)
print(data)

# 计算每日90th百分位
window = 11
pct = np.zeros((n_days, n_lat, n_lon))
for d in range(n_days):
    pct[d] = np.percentile(data[max(0, d-window//2):min(n_days, d+window//2+1)], 90, axis=0)

# 按照MHW定义识别热浪事件并计算矩阵信息
threshold = np.tile(np.mean(pct, axis=0), (n_days, 1, 1))  # 热浪事件阈值为每日90th百分位平均值 #(14610, 18, 26)

mask = data > threshold  # 识别每日是否为热浪事件
events = []
label, cnt = np.zeros(n_days), 0
for d in range(n_days):
    if np.all(mask[d]):
        cnt += 1
        label[d] = cnt
    elif cnt >= 5:
        # 符合MHW定义，记录热浪事件信息
        indices = np.nonzero(label)[0]
        start, end = indices[0], indices[-1]
        duration = end - start + 1
        if duration >= 5 and f"热浪事件({start}, {end})" not in events:
            max_intensity = np.max(data[start:end+1])
            mean_intensity = np.mean(data[start:end+1])
            cum_intensity = np.sum(data[start:end+1] - threshold[start:end+1])
            growth_rate = max_intensity - data[start]
            event_info = f"热浪事件({start}, {end})信息：持续时间{duration}天，最大强度{max_intensity}，平均强度{mean_intensity}，积累强度{cum_intensity}"
            events.append(f"热浪事件({start}, {end})")
            print(event_info)
        label, cnt = np.zeros(n_days), 0
    elif cnt > 0:
        start, end = 0, n_days - 1
        duration = end - start + 1
        if duration >= 5 and f"热浪事件({start}, {end})" not in events:
            max_intensity = np.max(data[start:end+1])
            mean_intensity = np.mean(data[start:end+1])
            cum_intensity = np.sum(data[start:end+1] - threshold[start:end+1])
            growth_rate = max_intensity - data[start]
            event_info = f"热浪事件({start}, {end})信息：持续时间{duration}天，最大强度{max_intensity}，平均强度{mean_intensity}，积累强度{cum_intensity}"
            events.append(f"热浪事件({start}, {end})")
        label, cnt = np.zeros(n_days), 0
print(f"总共发生了{len(events)}个热浪事件")