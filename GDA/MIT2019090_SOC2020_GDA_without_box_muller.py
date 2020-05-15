# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:25:38 2020

@author: VISHAL
"""
import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

  
def calculate_px_py(x, mu, sigma):
  n = 1
  pi = 3.14
  dim = mu.shape[0]
  c = 1/(((2*pi)**(dim/2))*np.sqrt(np.linalg.det(sigma)))
  return (c * np.exp(-0.5* np.dot(np.dot((x-mu), inv_sigma),(x-mu).T)))
  

def calculate_py(y, phi):
  if y== 1:
    return phi
  else:
    return (1-phi)


data = pd.read_csv('D:\\MTECH\\2nd_Sem\\Assignments\\dataset\\Microchip.csv')
data = shuffle(data).reset_index(drop=True)
print(data)

X = data.iloc[:84,:-1]
y = data.iloc[:84,2]
#test data
X_test = data.iloc[85:,:-1]
y_test = data.iloc[85:,2]

phi = np.mean(y)
a = data[data['Result']==0]
b = data[data['Result']==1]

neg_data = a.iloc[:,:2]
pos_data = b.iloc[:,:2]

mu0 = np.mean(neg_data, axis=0)
mu1 = np.mean(pos_data, axis=0)
print(mu0)

n_x = neg_data - mu0
p_x = pos_data - mu1

sigma = ((n_x.T).dot(n_x) + (p_x.T).dot(p_x))/X.shape[0]
inv_sigma = np.linalg.inv(sigma)
print(inv_sigma.shape)


y_pred = []
for i in range(0, len(X_test)):
  px0_0 = calculate_px_py(X_test.iloc[i,:].values, mu0, sigma)*calculate_py(0, phi)
  px0_1 = calculate_px_py(X_test.iloc[i,:].values, mu1, sigma)*calculate_py(1, phi)
 # px0_0  = px0_0 + px0_1
  if px0_0 < px0_1:
    px0_0 = 1
  else:
    px0_0 = 0
  y_pred.append([px0_0])
  
y_pred = pd.DataFrame(y_pred)

print(accuracy_score(y_test, y_pred))
print(cm(y_test, y_pred))
plt.scatter(pos_data.iloc[:,0], pos_data.iloc[:,1], color="green")
plt.scatter(neg_data.iloc[:,0], neg_data.iloc[:,1], color="red")