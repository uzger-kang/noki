from keras.models import load_model
import numpy as np
from pylab import *


# 本程序为输出可视化输出和输入的对比图/输出mae
# testdata与predata都是二维位置信息，并且其维度应相等
def my_mae(testdata, predata):
    ae = np.abs(testdata - predata)
    total_ae = sum(ae)
    mymae = total_ae/testdata.size
    return mymae


# 将test 和 train 的基站信号和位置处理成三输入的形式
# 如果应用历史信号预测，则应用此函数处理输入数据
# input为输入数组， n_out为输入神经网络数据组数
def n_bx(my_input, n_need):
    shape_n = my_input.shape[0]
    my_input = my_input.tolist()
    my_output = []
    for i in range(shape_n - n_need + 1):
        my_output.append(my_input[int(i):int(n_need + i)])
    my_output = np.array(my_output)
    return my_output


# 本函数实现输出与多输入相对应
def n_xy(my_input, n_need):
    my_input= np.delete(my_input, slice(0, n_need - 1), axis=0)
    return my_input


n_need = 2
BS_test_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                           usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
XY_test_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                          usecols=(0, 1))
BS_test_data = BS_test_data*(1.0/130)
BS_test_data = np.sin(BS_test_data)
BS_test_data = n_bx(BS_test_data, n_need)
XY_test_data = n_xy(XY_test_data, n_need)
BS_test_data = BS_test_data.reshape(BS_test_data.shape[0], 1, n_need, 12)


model = load_model('C:\\my_ml\\code_rally_2019\\model\\n_in_con_model_3in.h5')
# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
result = model.predict(BS_test_data, batch_size=100, verbose=1)

print('test mae = ', my_mae(XY_test_data, result))

# 0和1代表xy坐标预测值
PRE_X = result[:, 0]
PRE_Y = result[:, 1]
TEST_X = XY_test_data[:, 0]
TEST_Y = XY_test_data[:, 1]

X = np.linspace(0, result.shape[0], result.shape[0], endpoint=True)
plot(X, PRE_X)
plot(X, PRE_Y)
plot(X, TEST_X)
plot(X, TEST_Y)
show()