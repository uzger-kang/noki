from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
from keras import layers
import keras.backend as K


BS_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',
                     usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
XY_data = np.loadtxt('train.csv', skiprows=1, delimiter=',',
                     usecols=(0,1))
np.abs(BS_data)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)

# 搭建模型
model = Sequential()

model.add(Dense(64, input_shape=(12,), kernel_initializer='he_normal'))
model.add(Activation('selu'))
model.add(Dropout(0.1))

model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('selu'))
model.add(Dropout(0.1))

model.add(Dense(2,))

# compile
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
history = model.fit(BS_data, XY_data,
                    epochs=400, batch_size=256,
                    verbose=2, shuffle=True,
                    callbacks=[tbCallBack],)
model.save('C:\\my_ml\\code_rally_2019\\model\\my_model_1.h5')
# 测试
loss = model.evaluate(BS_data, XY_data)
print('Test loss:', loss)

result = model.predict(BS_data, batch_size=100, verbose=1)
print(result[0:50])