# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:39:00 2020

@author: VISHAL
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
np.random.seed(0)

data = np.random.uniform(-1, 1, (1500, 2))

initial_W = np.random.uniform(-1, 1, (100,2))
#plt.scatter(initial_W[:, 0], initial_W[:, 1])
#plt.title("Initial weights of Network")
alpha = 0.1

prev = np.zeros(data.shape[0])
for epochs in range(100):
    count = 0
    for i in range(data.shape[0]): 
        d = data[i] - initial_W
        out = np.argmin(np.linalg.norm(d, axis = 1))
        if epochs != 0:
            count += 1 if prev[i] != out else 0
        prev[i] = out
        initial_W[out] += alpha*(data[i] - initial_W[out])
#        lr -= 0.1*lr
    print("For epoch: ", epochs, " = ", count)
    if epochs != 0 and count == 0:
        break
print("Total epochs : ",epochs)

plt.scatter(initial_W[:, 0], initial_W[:, 1])
plt.title("Final weights of Network")
#testing 
data_test = np.array([[0.1,0.8], [0.5, -0.2], [-0.8, -0.9], [-0.6, 0.9]])
for i in range(data_test.shape[0]):
    d = data_test[i] - initial_W
    j = np.argmin(np.linalg.norm(d, axis = 1))
    print("Input", i, "=",data_test[i],"Result:", j)
    


