from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout


# minist数据集分类模型
# 读取数据集，第一次TensorFlow会自动下载数据集到下面的路径中, label 采用 one_hot 形式
# label 默认采用 0~9 来表示，等价于 one_hot=False, read_data_sets 时会默认把图像 reshape(展平)
# 若想保留图像的二维结构，可以传入 reshape=False
mnist = input_data.read_data_sets('c:/my_ml/MNIST_data', one_hot=True)

# 显示默认数据集的大小
print("Training data size: ", mnist.train.num_examples)              # 数据格式
print("Validating data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)

# 图片大小(28*28), TensorFlow 默认把它展开了，但这样丢失了图片的二维结构信息
print("Example training data0: ", mnist.train.images[0].shape)
print("Example training data0 label: ", mnist.test.labels[0])

temp = mnist.train.images[0]
img=np.reshape(temp, (28, 28))
plt.imshow(img,cmap='gray')
plt.show()

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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练并保存
model.fit(mnist.train.images, mnist.train.labels, epochs=2, batch_size=64, verbose=1, validation_split=0.05)
model.save_weights('my_model_weights_1.h5')
model.save('my_model_1.h5')

# 测试
loss, accuracy = model.evaluate(mnist.train.images, mnist.train.labels,)
print('Test loss:', loss)
print('Test Accuracy:', accuracy)






