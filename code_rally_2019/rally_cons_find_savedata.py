from keras.models import load_model
from pylab import *
import pandas as pd


# 本程序为可视化text集位置输出，并保存预测数据
# klkcon作用是将原一维矩阵变成适合于卷积层的二维矩阵，并且这个矩阵相邻项可以取到任意两元素组合。（可以最大限度提取空间信息）
def klkcon(BS_data):
    n_con = np.zeros((BS_data.shape[0], BS_data.shape[1] - 1, BS_data.shape[1]))
    for i in range(BS_data.shape[1] - 1):
        i = i + 1
        j = 0
        for h in range(i):
            for k in range(BS_data.shape[1]):
                if (k * i + h) >= (BS_data.shape[1]):
                    break
                n_con[:, i - 1, j] = BS_data[:, k * i + h]
                j += 1
                if j > (BS_data.shape[1] - 1):
                    break
            if j > (BS_data.shape[1] - 1):
                break
    return n_con


BS_data = np.loadtxt('test.csv', skiprows=1, delimiter=',',
                        usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))

# BS_data_1是正常卷积输入，经过变换的BS_data为正弦输入;BS_data_2为线性直接输入
BS_data_1 = BS_data
BS_data_2 = BS_data
BS_data = BS_data*(1.0/130)
BS_data = np.sin(BS_data)

# XY_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',usecols=(0, 1))
BS_data = klkcon(BS_data)
BS_data_1 = klkcon(BS_data_1)
BS_data = BS_data.reshape(BS_data.shape[0], 1, 11, 12)
BS_data_1 = BS_data_1.reshape(BS_data_1.shape[0], 1, 11, 12)

model = load_model('C:\\my_ml\\code_rally_2019\\model\\my_model_2_con.h5')
# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
result = model.predict(BS_data_1, batch_size=100, verbose=1)
# 0和1代表xy坐标预测值
X = result[0:205, 0]
Y = result[0:205, 1]
# TURE = XY_data[10000:18000, 1]

# X = np.linspace(0, 205, 205, endpoint=True)
plot(X, Y)
# plot(X, TURE)
show()

data1 = pd.DataFrame(result)
data1.to_csv('C:\\my_ml\\code_rally_2019\\predata\\predata2.csv')

