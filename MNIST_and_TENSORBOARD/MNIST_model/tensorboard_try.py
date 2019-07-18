from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

# 读取数据集，第一次TensorFlow会自动下载数据集到下面的路径中, label 采用 one_hot 形式
# label 默认采用 0~9 来表示，等价于 one_hot=False, read_data_sets 时会默认把图像 reshape(展平)
# 若想保留图像的二维结构，可以传入 reshape=False
mnist = input_data.read_data_sets('c:/my_ml/MNIST_data', one_hot=True)
keras.callbacks.TensorBoard(log_dir='./Graph',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq= 0,
                                         write_graph=True,
                                         write_images=True)

# 搭建模型
model = Sequential()

model.add(Dense(64, input_shape=(784,), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(mnist.train.images, mnist.train.labels,
                    epochs=10, batch_size=64,
                    verbose=1,
                    callbacks=[tbCallBack],
                    validation_split=0.05)

# 测试
loss, accuracy = model.evaluate(mnist.train.images, mnist.train.labels,)
print('Test loss:', loss)
print('Test Accuracy:', accuracy)