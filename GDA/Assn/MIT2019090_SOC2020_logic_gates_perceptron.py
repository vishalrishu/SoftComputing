# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:42:20 2020

@author: VISHAL
"""

import numpy as np

#Initial weights
w0 = np.random.randn()
w1 = np.random.randn()
w2 = np.random.randn()

print("Initial weights - ")
print("w0 = ",w0,"w1 = ", w1,"w2 = ", w2)

del_w0 = 1
del_w1 = 1
del_w2 = 1

alpha = 0.1
epochs = 1000

#Training data
train_data_temp = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
train_data = np.asarray(train_data_temp)

# op_f = [0, 0, 0, 1] # AND
# op_f = [0, 1, 1, 1] # OR
# op_f = [1, 0, 0, 0] # NOR
op_f = [1, 1, 1, 0] # NAND

op = np.asarray(op_f)

#training for y = w0 + w1*x1 + w2*x2
for i in range(epochs):
    j = 0
    for x in train_data:
        res = w0*x[0] + w1*x[1] + w2*x[2]

        if (res >= 0):
            act = 1
        else:
            act = 0
        
        err = op[j] - act

        del_w0 = alpha*x[0]*err
        del_w1 = alpha*x[1]*err
        del_w2 = alpha*x[2]*err

        w0 = w0 + del_w0
        w1 = w1 + del_w1
        w2 = w2 + del_w2

        j = j + 1
        
print("\nFinal weights - ")
print("w0 = ",w0,"w1 = ",w1,"w2 = ",w2)

#Test Model
test_data = [[0.98, 1],[0.01, 0.97],[0.77, 0.99],[0.912, 1.002],[0.88, 0.11],[0.82, 0.9],[0.8, 1],[0.02, 0.01],[0.21, 0.99],[0.11, 0.2],[0.79, 1],[0.11, 1.02],[0.98, 0.87],[0.2, 1.3],[0.2, 0.003]]
# test_op = [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0] # AND
# test_op = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0] # OR
# test_op = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1] # NOR
test_op = [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1] # NAND

total = 15
correct = 0

for i in range(len(test_data)):
    temp = w0 + w1*test_data[i][0] + w2*test_data[i][1]
    # ans = 1/(1 + math.exp(-temp))
    
    if ( temp >= 0 ):
        ans = 1
    else:
        ans = 0

    print("\nx1 - ",test_data[i][0],",x2 - ",test_data[i][1])
    print("Predicted - ",ans,",Actual - ",test_op[i])

    if ( ans == test_op[i] ):
        correct = correct + 1

print ( "\nAccuracy - ", correct/total)