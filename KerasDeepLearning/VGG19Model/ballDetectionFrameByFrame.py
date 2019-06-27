import numpy as np
import cv2
import tensorflow as tf
import time
import keras as keras

cap = cv2.VideoCapture('final_FIFACut.mp4')

IMG_SIZE = 224
frame_count = 1

#CATEGORIES = ['testOnePictureBall','testOnePictureNotBall']
CATEGORIES = ['trainingDataBall','trainingDataNotBall']

SCAN_BALL_HEIGHT = 25
SCAN_BALL_WIDTH = 25

next = False
ret, frame = cap.read()
cv2.imshow('frame',frame)

model = tf.keras.models.load_model("ball_detector.model",compile=False)

while(cap.isOpened()):
    if cv2.waitKey(1) & 0xFF == ord('p'):
        next = not next

    if( next):
        ret, frame = cap.read()

        h, w = frame.shape[:2]

        drawingFrame = frame.copy()

        if (frame_count > 1500 and frame_count % 20 == 0):
            next = not next
            images = []
            positions = []
            for x in range(int(SCAN_BALL_HEIGHT / 2),  h - int(SCAN_BALL_HEIGHT / 2), 25):
                for y in range(int(SCAN_BALL_WIDTH / 2),  w - int(SCAN_BALL_WIDTH / 2), 25):
                    roi = frame[x:x+SCAN_BALL_HEIGHT, y:y+SCAN_BALL_WIDTH]
                    piece = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                    new_piece = piece.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
                    images.append(new_piece)
                    positions.append((x,y))
                    # start = time.time()
                    # prediction = model.predict([new_piece],batch_size=32)
                    # end = time.time()
                    # print(end - start)
                    # prediction_class = CATEGORIES[int(prediction[0][0])]
                    # if prediction_class == 'trainingDataBall' :
                    #     cv2.rectangle(drawingFrame,(y, x),(y + SCAN_BALL_HEIGHT, x + SCAN_BALL_WIDTH ),(0,255,0),3)

            start = time.time()

            prediction = model.predict_classes(np.vstack(images),batch_size=128)
            end = time.time()
            print(end - start)

            for index in range(len(prediction)):
                if prediction[index] == 0:
                    cv2.rectangle(frame,(positions[index][1], positions[index][0]),(positions[index][1] + SCAN_BALL_HEIGHT, positions[index][0] + SCAN_BALL_WIDTH ),(0,0,255),3)

        frame_count = frame_count + 1
        cv2.imshow('frame',frame)


cap.release()
cv2.destroyAllWindows()
