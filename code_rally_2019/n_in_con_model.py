# coding:utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout
import numpy as np
import datetime


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


start_time = datetime.datetime.now()
# 设置随机种子
np.random.seed(1000)

# 从csv读取数据，并转化为多输入格式
n_need = 2
BS_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',
                     usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
XY_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',
                     usecols=(0, 1))
BS_data = BS_data*(1.0/130)
BS_data = np.sin(BS_data)
BS_data = n_bx(BS_data, n_need)
XY_data = n_xy(XY_data, n_need)


# 将数据转换为适合卷积层输入格式（四维）
BS_data = BS_data.reshape(BS_data.shape[0], 1, n_need, 12)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)

# 构建模型
model = Sequential()

# 卷积层,64个卷积核
model.add(Convolution2D(nb_filter=64, nb_row=2, nb_col=2,
                        activation='relu',
                        border_mode='same', input_shape=(1, 2, 12)))

# 将数据展平
model.add(Flatten())

# 全连接层
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(2,))
model.add(Activation('linear'))


# 编译模型
history = model.compile(optimizer='rmsprop', loss='MSE', metrics=['MAE'])

# 训练模型
# shuffle就是是否把数据随机打乱之后再进行训练
# verbose是屏显进度条
# validation_split就是拿出百分之多少用来做交叉验证
model.fit(BS_data, XY_data, nb_epoch=80, batch_size=64,
          shuffle=True, verbose=2, validation_split=0.2,
          callbacks=[tbCallBack])
model.save('C:\\my_ml\\code_rally_2019\\model\\n_in_con_model_3in.h5')

end_time = datetime.datetime.now()
total_time = (end_time - start_time).seconds
print('total time is:', total_time)

# 测试（针对训练集损失）
loss, accuracy = model.evaluate(BS_data, XY_data)
print('Test loss:', loss)
print('Test Accuracy:', accuracy)
