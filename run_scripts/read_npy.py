import numpy as np
import os

# 定义npy文件所在的目录
data_path = r"E:\postgraduate\madiff_mpe\diffuser\datasets\data\mpe\simple_tag\expert\seed_9_data"

# 定义要读取的npy文件
# npy_files = ["acs_0.npy", "acs_1.npy", "acs_2.npy",
#              "dones_0.npy", "dones_1.npy", "dones_2.npy",
#              "obs_0.npy", "obs_1.npy", "obs_2.npy",
#              "rews_0.npy", "rews_1.npy", "rews_2.npy"]
npy_files = ["obs_1.npy"]
# 创建一个字典来存储每个文件的数据
npy_data = {}

# 读取每个npy文件并存储到字典中
for file in npy_files:
    file_path = os.path.join(data_path, file)
    if os.path.exists(file_path):
        npy_data[file] = np.load(file_path)
        print(f"{file} shape：\n", npy_data[file])
    else:
        print(f"{file} not exist")