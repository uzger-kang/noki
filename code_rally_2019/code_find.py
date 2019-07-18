from tensorflow.examples.tutorials.mnist import input_data
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy as np
from pylab import *


BS_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',
                        usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
XY_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',
                     usecols=(0, 1))

print(BS_data[7])

model = load_model('C:\\my_ml\\code_rally_2019\\model\\my_model_1.h5')
# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
result = model.predict(BS_data, batch_size=100, verbose=1)
PRE = result[0:10000, 0]
TURE = XY_data[0:10000, 0]

X = np.linspace(0, 10000, 10000, endpoint=True)
plot(X, PRE)
plot(X, TURE)
show()