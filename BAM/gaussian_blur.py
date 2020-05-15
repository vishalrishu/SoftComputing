# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:15:20 2020

@author: VISHAL
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

blur = cv2.blur(img,(5,5))