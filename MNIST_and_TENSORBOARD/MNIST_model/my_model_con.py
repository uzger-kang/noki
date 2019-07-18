#coding:utf-8

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import datetime

start_time = datetime.datetime.now()
# 设置随机种子
np.random.seed(1000)

# 数据
mnist = input_data.read_data_sets('c:/my_ml/MNIST_data', one_hot=True)

# 构建模型
model = Sequential()
# reshape
train_data = mnist.train.images.reshape(mnist.train.images.shape[0],1,28,28)
test_data = mnist.test.images.reshape(mnist.test.images.shape[0],1,28,28)

# 卷积层,32个卷积核,每个卷积核大小5*5,采用same_padding的方式
model.add(Convolution2D(nb_filter=32,nb_row=5,nb_col=5,border_mode='same',input_shape=(1,28,28)))
# pooling层,采用same padding
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
model.add(Convolution2D(nb_filter=64,nb_row=5,nb_col=5,border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
# 将数据展平
model.add(Flatten())
# 全连接层
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# 编译模型sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
model.compile(optimizer=Adam(lr = 0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# 训练模型
# shuffle就是是否把数据随机打乱之后再进行训练
# verbose是屏显进度条
# validation_split就是拿出百分之多少用来做交叉验证
model.fit(train_data, mnist.train.labels,nb_epoch=10,batch_size=32,shuffle=True,verbose=1,validation_split=0.2)
model.save_weights('my_model_weights_con.h5')
model.save('my_model_1_con.h5')

end_time = datetime.datetime.now()
total_time = (end_time - start_time).seconds
print('total time is:',total_time)

# 测试（针对训练集损失）
loss, accuracy = model.evaluate(train_data, mnist.train.labels)
print('Test loss:', loss)
print('Test Accuracy:', accuracy)
