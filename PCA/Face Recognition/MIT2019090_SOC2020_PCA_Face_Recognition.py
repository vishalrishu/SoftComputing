# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:33:18 2020

@author: VISHAL
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import re
import random
import cv2
from copy import deepcopy
from sklearn.linear_model import LogisticRegression

new_image_height = 40
new_image_width = 40
old_directory = "D:\\MTECH\\2nd_Sem\\Assignments\\dataset\\Edited"
new_directory = "D:\\MTECH\\2nd_Sem\\Assignments\\dataset\\Reduced"


c = 0
for file in os.listdir(old_directory) :
    if(file != ".DS_Store") :
        old_image_path = os.path.join(old_directory, file)
        old_image = Image.open(old_image_path)
        old_image = old_image.resize((new_image_height, new_image_width), Image.ANTIALIAS)
        
        new_image_path = os.path.join(new_directory, file)
        old_image.save(new_image_path)
        
Y = [] # labels
Y_values = {} # mapping from person name to number
Y_inverse_values = {} # mapping from number to person name, to be used during prediction

X = [] # training dataset

for file in os.listdir(new_directory) :
    if(file != ".DS_Store") :
        # for X
        im = cv2.imread(os.path.join(new_directory, file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im.flatten()
        X.append(im)

        # for Y
        file = file.split(".")[0]
        label_name = str(" ".join(re.split("[^a-zA-Z]*", file)))
        if(label_name not in Y_values) :
            new_val = len(Y_values) + 1
            Y_values[label_name] = new_val
            Y_inverse_values[new_val] = label_name
        Y.append(Y_values[label_name])
        
X = np.array(X)
Y = np.array(Y)

fig = plt.figure()
plt.rcParams['figure.figsize'] = [50, 50]

#
#random_indices = random.sample(range(0, X.shape[0]), 10)
#for i in range(len(random_indices)) :
#    tmp = X[random_indices[i]].reshape(new_image_height, new_image_width)
#    plt.subplot(2, 5, i + 1)
#    plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB))
#    

# mean normalization
mean_intensities = np.mean(X, axis=0)

# split into training and testing data
X_train = []
X_test = []
Y_train = []
Y_test = []

training_perc = 75 
sample_size = int((training_perc / float(100)) * X.shape[0])
training_indices = random.sample(range(0, X.shape[0]), sample_size)
for i in range(X.shape[0]) :
    if(i in training_indices) :
        X_train.append(X[i])
        Y_train.append(Y[i])
    else :
        X_test.append(X[i])
        Y_test.append(Y[i])
        
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

mean_intensities = np.mean(X_train)
X_train = X_train - mean_intensities

X_test_original = deepcopy(X_test)
mean_intensities = np.mean(X_test)
X_test = X_test - mean_intensities

# finding covariance matrix 
cov_matrix = (1 / float(X_train.shape[0])) * np.dot(np.transpose(X_train), X_train)



# finding best K values

u, s, v = np.linalg.svd(cov_matrix)
total_sum = 0
for i in range(s.shape[0]) :
    total_sum = total_sum + s[i]
prefix_sum = 0
for i in range(1600) :
    prefix_sum = prefix_sum + s[i]
    val = float(prefix_sum) / float(total_sum)
    if(val >= 0.99) :
        K = i + 1
        break
    
best_eigen_vectors = []
for i in range(K) :
    best_eigen_vectors.append(u[:, i])
best_eigen_vectors = np.array(best_eigen_vectors)


# getting the eigen faces
eigen_faces = np.dot(X_train, np.transpose(best_eigen_vectors))

# for testing data => already done mean normalization

projected_data = np.dot(X_test, np.transpose(best_eigen_vectors))

# now we will work with reduced dimension data

X_train_original = deepcopy(X_train)
X_test_original_1 = deepcopy(X_test)

X_train = eigen_faces
X_test = projected_data

# mean vectors

mean_vectors = []
for i in range(1, len(Y_values) + 1) :
    mean_vectors.append(np.mean(X_train[Y_train == i], axis = 0))
mean_vectors = np.array(mean_vectors)

# within class scatter

S_W = np.zeros((K, K))
for i in range(1, mean_vectors.shape[0] + 1) :
    class_sc_mat = np.zeros((K, K))
    for row in X_train[Y_train == i] :
        row1, mean1 = row.reshape(K, 1), mean_vectors[i - 1].reshape(K, 1)
        class_sc_mat += np.dot((row1 - mean1), np.transpose(row1 - mean1))
    S_W += class_sc_mat
    
# between class scatter

overall_mean = np.mean(X_train, axis = 0)

S_B = np.zeros((K, K))
for i in range(1, mean_vectors.shape[0] + 1) :
    n = X_train[Y_train == i].shape[0]
    mean1 = mean_vectors[i - 1].reshape(K, 1)
    overall_mean1 = overall_mean.reshape(K, 1)
    S_B += n * np.dot((mean1 - overall_mean1), np.transpose(mean1 - overall_mean1))

cov_matrix = np.dot(np.linalg.inv(S_W), S_B)

u, s, v = np.linalg.svd(cov_matrix)

new_K = 20
W = []

for i in range(new_K) :
    W.append(u[:, i])
W = np.array(W)

X_train_transformed = np.dot(X_train, np.transpose(W))

X_test_transformed = np.dot(X_test, np.transpose(W))

   

# prediction using euclidean distnace

predictions = []

for i in range(X_test_transformed.shape[0]) :
    min_distance = -1
    for j in range(X_train_transformed.shape[0]) :
        distance = np.linalg.norm(X_train_transformed[j] - X_test_transformed[i]) 
        if(min_distance == -1 or min_distance > distance) :
            min_distance = distance
            label = Y_train[j]
    predictions.append(label)


# show prediction v/s actual

total_correct = 0

fig = plt.figure()
k = 10
for i in range(3) :
    for j in range(3) :
        tmp = deepcopy(X_test_original[2 * j + i])
        tmp = tmp.reshape(new_image_height, new_image_width)
        plt.subplot(3, 6, k)
        
        predicted_label = Y_inverse_values[predictions[2 * j + i]]
        actual_label = Y_inverse_values[Y_test[2 * j + i]]
        
        if(predicted_label == actual_label) :
            total_correct += 1
        
        title = "actual: " + actual_label + "\npredicted: " + predicted_label 
        
        plt.title(title, fontsize=20)
        k = k + 1
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB))
        
        
print("percentage accuracy: ", (total_correct / X_test.shape[0]) * 100)