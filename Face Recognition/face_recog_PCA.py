import numpy as np
import cv2
import os
from PIL import Image
from sklearn.metrics import accuracy_score

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

#cov = (normalized @ normalized.T) / normalized.shape[0]
cov = np.cov(normalized)

val,vec = np.linalg.eig(cov)
order = np.flip(np.argsort(val))
new_vec = np.real(vec[:,order])

k = 3

k_vec = new_vec[:k]
##del(normalized)
#del(cov)
eig_faces = k_vec @ normalized
#
W = eig_faces @ normalized.T
#
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
    index = np.argmin(np.linalg.norm(test_W - W.T, axis = 1))
    y_pred.append(int(np.floor(index/6)))
    
print(accuracy_score(y, y_pred))
