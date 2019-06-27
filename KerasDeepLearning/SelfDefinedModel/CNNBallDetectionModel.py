import tensorflow as tf
import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("Xsmall.pickle","rb"))
y = pickle.load(open("ysmall.pickle","rb"))

X = X / 255

model = Sequential()

model.add(Conv2D(64, (3, 3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(256, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(256, (2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(10))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=30, epochs=10, validation_split=0.3)

model.save('ball_detector.model')
new_model = tf.keras.models.load_model('ball_detector.model')
