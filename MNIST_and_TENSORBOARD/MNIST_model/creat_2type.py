import matplotlib.pyplot as plt
import numpy as np
import math
import random

# n_batch是每个batch的个数，最好能整除；data是原数据（二维数组），返回一个【n，n-batch，2】数组,n是有多少个batch
def batch(n_batch, data):
    n_count = np.arange(0, data.size)
    np.random.shuffle(n_count)
    n = data.size//n_batch
    batch = np.zeros((n, n_batch, 2))
    for i in range(n):
        for j in range(n_batch):
            batch[i, j, 0] = n_count[i*n + j]//data.shape[1]
            batch[i, j, 1] = n_count[i*n + j]%data.shape[0]
    return(batch)


# 产生八组有噪声的二分类数据，例：第一组为n_data【0】，其值为类别
n_data = np.random.randint(8, 20, (8, 30, 30))
for h in range(8):
    for i in range(n_data[0].shape[0]):
        for j in range(n_data[0].shape[1]):
            k = n_data[h, i, j]/math.sqrt((i - 15)*(i - 15) + (j - 15)*(j - 15))
            if k > 1:
                n_data[h, i, j] = 1
            else:
                n_data[h, i, j] = 0

batch_1 = batch(30, n_data[1])
print(batch_1[0])

# 可视化batch输出
batch_1 =batch_1.astype('int64')
nn = np.zeros((30, 30))
for i in range(30):
    nn[batch_1[i, 0], batch_1[i, 1]] = 1

plt.imshow(nn, cmap='gray')
plt.show()
