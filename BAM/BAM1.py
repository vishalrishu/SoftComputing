# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:13:18 2020

@author: VISHAL
"""

import numpy as np


#    return ((delta+1)*x) - (delta*(x**3))

def activation(arr):
    for i in range(arr.shape[0]):
        arr[i] = get_val(arr[i])
    return arr

def get_val(x):
    if x > 1:
        return 1
    if x < -1:
        return -1
    return 0

X = np.array([[1,1,1,1,1,1 ],[-1,-1,-1,-1,-1,-1],[1,-1,-1,1,1,1],[1,1,-1,-1,-1,-1]])
y = np.array([[1,1,1],[-1,-1,-1],[-1,1,1],[1,-1,1]])


W = np.zeros((X.shape[1], y.shape[1]))

for i in range(X.shape[0]):
    W += np.outer(X[i],y[i])
    
V = W.copy().T
    
changed = True
epochs = 0
lr = 0.

while changed:
    changed = False
    epochs+=1
    for i in range(X.shape[0]):
        pred = activation(X[i] @ W)
        new_x = activation(pred @ V)
        error = y[i] - pred
        if np.any(error):
            changed = True
            W += lr*np.outer(error, X[i] + new_x).T
        
        error_x = X[i] - new_x
        if np.any(error_x):
            changed = True
            V += lr*np.outer(error_x, y[i] + pred).T
    
print(epochs)
for i in range(X.shape[0]):
	y_pred = activation(X[i] @ W)
	print(X[i],y_pred)
