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

IMG_SIZE = 25
CIRCLE_RADIUS = 5

second_frame_count = 1

ballPosition = pickle.load(open("ballPosition.pickle","rb"))

while(cap.isOpened()):
    if (second_frame_count == 1500):
         break

    ret, frameRead = cap.read()
    frame = random_noise(frameRead, mode='gaussian', seed=None, clip=True)

    blank_image = np.zeros((674,1200,3), np.uint8)

    cv2.circle(blank_image,(ballPosition[second_frame_count - 1][0],ballPosition[second_frame_count - 1][1]), CIRCLE_RADIUS, (0,255,255), -1)
    print(second_frame_count)
    start = time.time()
    if ballPosition[second_frame_count - 1][0] < IMG_SIZE and ballPosition[second_frame_count - 1][1] < IMG_SIZE:
        end = time.time()
        print(end - start)
        second_frame_count = second_frame_count + 1
        continue
    for i in range(1,5000):
        # print("frame: "+str(second_frame_count))
        # print("iteration "+str(i))
        width = random.randint(1, 1150)
        height = random.randint(1, 624)
        rotation = random.randint(1, 180)
        scale = random.uniform(0.5, 1)


        randompiece_frameWithCircle = blank_image[height:height+IMG_SIZE, width:width+IMG_SIZE]
        randompiece_frameToData = frame[height:height+IMG_SIZE, width:width+IMG_SIZE]

        rotation_frameWithCircle = rotate(randompiece_frameWithCircle, angle=rotation, mode='reflect')
        scale_frameWithCircle = rescale(rotation_frameWithCircle, scale=scale, mode='constant')

        rotation_frameToData = rotate(randompiece_frameToData, angle=rotation, mode='reflect')
        scale_frameToData = rescale(rotation_frameToData, scale=scale, mode='constant')
        # frameWithCircle = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=45, width_shift_range=1200, height_shift_range=674, brightness_range=None, shear_range=0.0, zoom_range=[0.025,0.025], channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None).random_transform(blank_image,seed=i)
        # frameToData = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=45, width_shift_range=1200, height_shift_range=674, brightness_range=None, shear_range=0.0, zoom_range=[0.025,0.025], channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None).random_transform(frame,seed=i)
        frameWithCircle = cv2.resize(scale_frameWithCircle, (IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_AREA)
        frameToData = cv2.resize(scale_frameToData, (IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_AREA)


        count = np.count_nonzero(frameWithCircle)

        if(count > 250):
            print(count)
            name = str(second_frame_count)+"_"+str(i)+".png"
            frameToData = frameToData * 255
            image_to_save = frameToData.astype('uint8')
            cv2.imwrite(os.path.join("C:/Users/emilh/Desktop/keras_bachelor/FifaBall/",name), image_to_save )
        else:
            random_number = random.uniform(0, 1)
            if random_number > 1 - 1 / 1000:
                 name = str(second_frame_count)+"_"+str(i)+".png"
                 frameToData = frameToData * 255
                 image_to_save = frameToData.astype('uint8')
                 cv2.imwrite(os.path.join("C:/Users/emilh/Desktop/keras_bachelor/FifaNotBall/",name), image_to_save )

    end = time.time()
    print(end - start)
    second_frame_count = second_frame_count + 1

cap.release()
cv2.destroyAllWindows()
