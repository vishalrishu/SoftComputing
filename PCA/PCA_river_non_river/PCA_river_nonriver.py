# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:49:59 2020

@author: VISHAL
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image
import cv2
import imtools

r = np.array(Image.open('1.gif')).flatten()
g = np.array(Image.open('2.gif')).flatten()
b = np.array(Image.open('3.gif')).flatten()
i = np.array(Image.open('4.gif')).flatten()

cov = np.cov([r,g,b,i])
_, vec = np.linalg.eig(cov)
PC = [[],[],[],[]]

for j in range(r.shape[0]):
    for k in range(4):
        PC[k].append(np.dot(vec[:,k], [r[j],g[j],b[j],i[j]]))
        
new_r = np.array(PC[0]).reshape((512,512)).astype(np.uint8)
new_g = np.array(PC[1]).reshape((512,512)).astype(np.uint8)
new_b = np.array(PC[2]).reshape((512,512)).astype(np.uint8)
new_i = np.array(PC[3]).reshape((512,512)).astype(np.uint8)

img_r = Image.fromarray(new_r)
img_g = Image.fromarray(new_g)
img_b = Image.fromarray(new_b)
img_i = Image.fromarray(new_i)

img_r.save("PCA_1.png")
img_g.save("PCA_2.png")
img_b.save("PCA_3.png")
img_i.save("PCA_4.png")

img = cv2.imread("PCA_1.png")

plt.imshow(img);