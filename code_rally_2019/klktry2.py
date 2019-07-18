import numpy as np


# 本函数实现输出与多输入相对应
def n_xy(my_intput, n_need):
    my_intput= np.delete(my_intput, slice(0, n_need - 1), axis=0)
    return my_intput


XY_train_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                           usecols=(0, 1))
XY_train_data = n_xy(XY_train_data, 5)
print(XY_train_data)
print(XY_train_data.shape[0])
