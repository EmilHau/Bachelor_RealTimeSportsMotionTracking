import cv2
import tensorflow as tf
import os
import keras as keras
import numpy as np
import pickle
import time
import random
from skimage import data, color
from skimage.transform import rescale, rotate
from skimage.util import random_noise
# COMMMENT OUT WHEN FINDING BALLCENTERS

cap = cv2.VideoCapture('final_FIFACut.mp4')
ret, frame = cap.read()
blackImage = frame * 0

ballPosition = pickle.load(open("ballPosition.pickle","rb"))

for index in range(len(ballPosition)):
    cv2.circle(blackImage,(ballPosition[index][0],ballPosition[index][1]), 5, (255,255,255), -1)

cv2.imwrite("C:/Users/emilh/Desktop/keras_bachelor/perfect.png",blackImage)
