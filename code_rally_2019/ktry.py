import numpy as np


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


BS_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                     usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
print(BS_data.shape[0])
XY_train_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',
                           usecols=(0, 1))
klk = n_bx(BS_data, 5)
print(klk)
print(klk.shape)
