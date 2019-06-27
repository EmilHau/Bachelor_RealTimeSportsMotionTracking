import cv2
import tensorflow as tf
import os
import keras as keras
import numpy as np
import pickle

cap = cv2.VideoCapture('final_FIFACut.mp4')

IMG_SIZE = 50
CIRCLE_RADIUS = 5

CATEGORIES = ["FifaBall", "FifaNotBall"]

SCAN_BALL_HEIGHT = IMG_SIZE
SCAN_BALL_WIDTH = IMG_SIZE

ret, frame = cap.read()
cv2.imshow('frame',frame)
frame_count = 1

waitForMouseEvent = True

global ballPosition # index is framenumber + 1
ballPosition = []
global ballX
global ballY

def mouse_drawing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global ballX
        global ballY
        ballX = x
        ballY = y
        tempFrame = frame.copy()
        cv2.circle(tempFrame,(ballX,ballY), CIRCLE_RADIUS, (0,255,255), -1)
        cv2.imshow('Temp',tempFrame)

while(cap.isOpened()):
    if (frame_count == 1500):
         break

    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    cv2.imshow('Temp',frame)

    cv2.setMouseCallback('frame', mouse_drawing)

    while(waitForMouseEvent):
        if cv2.waitKey(1) & 0xFF == ord('n'):
            break
        continue

    global ballX
    global ballY

    ballPosition.append((ballX,ballY))
    print(frame_count)
    print((ballX,ballY))
    frame_count = frame_count + 1

pickle_out = open("ballPosition.pickle","wb")
pickle.dump(ballPosition,pickle_out)
pickle_out.close()
