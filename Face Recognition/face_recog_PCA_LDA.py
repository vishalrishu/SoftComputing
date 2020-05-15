# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:29:46 2020

@author: Admin
"""

import numpy as np
import cv2
import os
from PIL import Image
from sklearn.metrics import accuracy_score

def mah_dist(x,y,S):
    diff = (x - y).reshape((x.shape[0], 1))
    dist = np.sqrt(abs(diff.T @ S @ diff))
    return dist[0][0]
    

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#            cv2.imshow('Gray image', img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            images.append(gray.flatten())
    return images

path = os.getcwd() + "\\training_faces"
training_images = np.zeros((6,10304))
for n,folder in enumerate(os.listdir(path)):
    if(n==0):
        training_images+=(load_images_from_folder(path+"\\"+folder))
    else:
        training_images = np.row_stack((training_images,(load_images_from_folder(path+"\\"+folder))))


#files = os.listdir('faces')
#
##training part
#
#images = []
#
##img = Image.open('faces/'+files[0])
##img = img.resize((50,50))
#
#for file in files:
#    images.append(np.array(Image.open('faces/'+file).resize((50,50))).flatten())
#    
#images = np.array(images).T
mean = training_images.mean(axis = 0)

normalized = training_images - mean

cov = (normalized @ normalized.T) / normalized.shape[1]

val,vec = np.linalg.eig(cov)
k = 30
m = 8
order = np.flip(np.argsort(val))
vec = vec[:,order]
new_vec = vec[:k]


##del(normalized)
#del(cov)
eig_faces = new_vec @ normalized
#
W = eig_faces @ normalized.T
#

classes = {}
class_mean = np.zeros((10,W.shape[0]))
for i in range(10):
    classes[i] = W.T[i*6:i*6+6]
    class_mean[i] = classes[i].mean(axis = 0)
    
global_mean = W.T.mean(axis = 0)
SW = np.zeros((W.shape[0],W.shape[0]))
for i in range(10):
    temp = classes[i] - class_mean[i]
    SW += temp.T @ temp
    
SB = np.zeros(SW.shape)

for i in range(10):
    temp = class_mean[i] - global_mean
    SB += temp.T @ temp

J = np.linalg.inv(SW) @ SB
val , vec = np.linalg.eig(J)

order = np.flip(np.argsort(val))
vec = vec[order]
feats = np.real(vec[:m])

fisher = feats @ W

S = np.linalg.inv(np.cov(fisher))

##testing part
path = os.getcwd() + "\\testing_faces"
testing_images = np.zeros((4,10304))
y = [0]*4
for n,folder in enumerate(os.listdir(path)):
#    temp = int(folder.strip('s')) - 1
#    y.extend([temp]*4)
    if(n==0):
        testing_images+=(load_images_from_folder(path+"\\"+folder))
    else:
        y.extend([n]*4)
        testing_images = np.row_stack((testing_images,(load_images_from_folder(path+"\\"+folder))))


#files = os.listdir('test')
#testing_file = files[3]
#
#test_img = np.array(Image.open('test/'+testing_file).resize((50,50))).reshape(mean.shape)
y_pred = []
for test_img in testing_images:
#    test_img = testing_images[5]
    test_norm = test_img - mean
    test_W = eig_faces @ test_norm.T
    test_fisher = feats @ test_W
    index = np.argmin(np.array([mah_dist(test_fisher, img, S) for img in fisher.T]))
#    index = np.argmin(np.linalg.norm(test_fisher - fisher.T, axis = 1))
    y_pred.append(int(np.floor(index/6)))
    
print(accuracy_score(y, y_pred))
