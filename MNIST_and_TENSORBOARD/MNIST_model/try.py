import matplotlib.pyplot as plt
import numpy as np
import math
import random

n_count = np.zeros((10, 10,2))
for i in range(10):
    for j in range(10):
        n_count[i, j, 1] = j
        n_count[i, j, 0] = i
n_count = n_count.reshape((100,2))
print(n_count)

n = random.sample(n_count.tolist(), 10)
print(n)
