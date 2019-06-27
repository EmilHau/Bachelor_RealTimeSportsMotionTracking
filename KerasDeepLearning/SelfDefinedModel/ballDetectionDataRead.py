import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

DATADIR = "C:/Users/emilh/Desktop/keras_bachelor/"

CATEGORIES = ['FifaBall','FifaNotBall']

# Insure same size
IMG_SIZE = 25

training_data = []
def create_training_data():
    for category in CATEGORIES:
        print(len(training_data))
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        counter = 0
        num = random.randint(1,5000)
        for img in os.listdir(path):
            if(class_num == 1 and counter==16500):
                break
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_img_array, class_num])
                counter = counter + 1
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)

# Features
X = []
# Labels
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


pickle_out = open("Xsmall.pickle","wb")
pickle.dump(X,pickle_out, protocol=4)
pickle_out.close()

pickle_out = open("ysmall.pickle","wb")
pickle.dump(y,pickle_out, protocol=4)
pickle_out.close()
