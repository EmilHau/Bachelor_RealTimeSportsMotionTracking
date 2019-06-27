import tensorflow as tf
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("Xsmall.pickle","rb"))
y = pickle.load(open("ysmall.pickle","rb"))

X = X / 255

vgg16_model = keras.applications.vgg16.VGG16()

model = keras.Sequential()

for layer in vgg16_model.layers:
    model.add(layer)

model.layers.pop()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(1,activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=10, epochs=1, validation_split=0.3)

model.save('ball_detector.model')
