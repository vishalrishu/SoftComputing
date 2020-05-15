# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:38:14 2020

@author: VISHAL
"""
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
fig = plt.figure()
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from numpy import sqrt, log, sin, cos, pi
from pylab import hist,subplot,figure

def calculate_px_py(x, mu, sigma):
  n = 1
  pi = 3.14
  dim = mu.shape[0]
  c = 1/(((2*pi)**(dim/2))*np.sqrt(np.linalg.det(sigma)))
  return (c * np.exp(-0.5* np.dot(np.dot((x-mu), inv_sigma),(x-mu).T)))
  

def calculate_py(y, phi):
  if y==1:
    return phi
  else:
    return (1-phi)


def box_muller(u1,u2):
    s = np.sqrt(u1**2 + u2**2)
    z1 = u1*np.sqrt(-2*np.log(s)/s)
    z2 = u2*np.sqrt(-2*np.log(s)/s)
    return z1,z2

def gaussian(u1,u2):
  z1 = sqrt(-2*log(u1))*cos(2*pi*u2)
  z2 = sqrt(-2*log(u1))*sin(2*pi*u2)
  return z1,z2



data1 = pd.read_csv('D:\\MTECH\\2nd_Sem\\Assignments\\dataset\\Microchip.csv')
data = shuffle(data1).reset_index(drop=True)
scaler = MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data))

X = data.iloc[:,:2]
y = data1.iloc[:100,2]
u1 = data.iloc[:,0]
u2 = data.iloc[:,1]
#TEST DATA
z = []
for i in range(0, len(u1)):
  z1,z2 = gaussian(data.iloc[i,0],data.iloc[i,1])
  z.append([z1,z2])

z = pd.DataFrame(z)
z = z.join(data.iloc[:,2])
z.columns = ['test1','test2','Result']
#print(z)
z = z[np.isfinite(z).all(1)]
z = pd.DataFrame(z)

X = z.iloc[:100,:2]
#TEST DATA
X_test = z.iloc[100:,:2]
y_test = z.iloc[100:,2]
print(y_test)

#plotting
figure()
#subplot(221)
#hist(u1)
#subplot(222)
#hist(u2)
#subplot(223)
#hist(z.iloc[:,0]) 
#subplot(224)
#hist(z.iloc[:,1])
#phi = np.mean(y)
a = z[z['Result']==0]
b = z[z['Result']==1]

neg_data = a.iloc[:100,:2]
pos_data = b.iloc[:100,:2]
#print(b)
mu0 = np.mean(neg_data, axis=0)
mu1 = np.mean(pos_data, axis=0)
print(mu0)

n_x = neg_data - mu0
p_x = pos_data - mu1

sigma = ((n_x.T).dot(n_x) + (p_x.T).dot(p_x))/z.shape[0]
inv_sigma = np.linalg.inv(sigma)
print(inv_sigma)

y_pred = []
for i in range(0, len(X_test)):
  px0_0 = calculate_px_py(X_test.iloc[i,:].values, mu0, sigma)*calculate_py(0, phi)
  px0_1 = calculate_px_py(X_test.iloc[i,:].values, mu1, sigma)*calculate_py(1, phi)
  if px0_0 > px0_1:
    px0_0 = 0
  else:
    px0_0 = 1
  y_pred.append([px0_0])
  
y_pred = pd.DataFrame(y_pred)
print(accuracy_score(y_test, y_pred))
print(cm(y_test, y_pred))
#plt.scatter(pos_data.iloc[:,0], pos_data.iloc[:,1], color="green")
#plt.scatter(neg_data.iloc[:,0], neg_data.iloc[:,1], color="red")

