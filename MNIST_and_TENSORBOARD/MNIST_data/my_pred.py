from tensorflow.examples.tutorials.mnist import input_data
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# 加载模型\数据
mnist = input_data.read_data_sets('c:/my_ml/MNIST_data', one_hot=True)

# 根据是否是卷积网络决定注释与否，因为输入数据格式不同
test_data = mnist.test.images.reshape(mnist.test.images.shape[0],1,28,28)

model = load_model('c:/my_ml/my_model_1_con.h5')
# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
result = model.predict(test_data, batch_size=100, verbose=1)

# 找到每行最大的序号
result_max = np.argmax(result, axis=1)  # axis=1表示按行 取最大值   如果axis=0表示按列 取最大值 axis=None表示全部
test_max = np.argmax(mnist.test.labels, axis=1)  # 这是结果的真实序号
result_bool = np.equal(result_max, test_max)  # 预测结果和真实结果一致的为真（按元素比较）
true_num = np.sum(result_bool)  # 正确结果的数量

print("The accuracy of the model is %f" % (true_num / len(result_bool)))  # 验证结果的准确率

# 打印预测结果
plt.figure()
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    ax.set_title("my pre is%u" % result_max[i])
    temp = mnist.test.images[i]
    img=np.reshape(temp, (28, 28))
    plt.imshow(img,cmap='gray')
plt.show()
