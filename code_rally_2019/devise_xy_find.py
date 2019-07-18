from keras.models import load_model
from pylab import *


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


def my_mae(testdata, predata):
    ae = np.abs(testdata - predata)
    print(ae.shape)
    total_ae = sum(ae)
    print(total_ae.shape)
    mymae = total_ae/testdata.size
    return mymae


n_need = 5

BS_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                     usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
X_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                    usecols=(0,))
Y_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                    usecols=(1,))
XY_test_data = np.loadtxt('sample_submission.csv', skiprows=1, delimiter=',',
                          usecols=(0, 1))
TEST_X = XY_test_data[:, 0]
TEST_Y = XY_test_data[:, 1]
BS_data = BS_data*(1.0/130)
BS_data = np.sin(BS_data)
BS_data = klkcon(BS_data)

BS_data = n_bx(BS_data, n_need)
X_data = n_xy(X_data, n_need)
Y_data = n_xy(Y_data, n_need)

TEST_X = n_xy(TEST_X, n_need)
TEST_Y = n_xy(TEST_Y, n_need)

# 将数据转换为适合卷积层输入格式（四维）
BS_data = BS_data.reshape(BS_data.shape[0], 1, n_need, 11*12)

# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
model = load_model('C:\\my_ml\\code_rally_2019\\model\\n_in_con_X3.h5')
result_X = model.predict(BS_data, batch_size=100, verbose=1)
model = load_model('C:\\my_ml\\code_rally_2019\\model\\n_in_con_Y3.h5')
result_Y = model.predict(BS_data, batch_size=100, verbose=1)

X_mae = my_mae(result_X.T, TEST_X)
Y_mae = my_mae(result_Y.T, TEST_Y)
print('x mae:', X_mae, '      y mae:', Y_mae)
print('total mae:', (X_mae + Y_mae)/2.0)

X = np.linspace(0, result_X.shape[0], result_X.shape[0], endpoint=True)
plot(X, result_X)
plot(X, result_Y)
plot(X, TEST_X)
plot(X, TEST_Y)
show()
