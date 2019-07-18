import numpy as np


# sigmod激活函数
def sigmod(x):
    y = 1/(1 + np.exp(-x))
    return y


# 定义最简单的神经元基元：输入：权重；偏置（线性）/输出：经神经元归一化的值（0-1）
# 将class作为神经元，每一个神经元是其实例，通过调用其输出得到结果
# 输入值个数由weights和bias维数决定
class Neuron:
    def __init__(self, weights):
        self.weights = weights
#        self.bias = bias

    def output(self,input):       # input是神经元输入数组
        in_total = np.dot(self.weights, input)  # + self.bias
        neu_out = sigmod(in_total)
        return(neu_out)


weights = np.array([0,1])
n = Neuron(weights)
x = np.array([2,3])
print(n.output(x))



