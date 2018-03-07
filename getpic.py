#coding=utf8
#__author__='caichaopeng'
import cv2
import os

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(2)
cv2.namedWindow("camera", 1)

while True:

    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        f = cv2.resize(gray[y:y + h, x:x + w], (299, 299))
        if not os.path.exists('./predictFace/p.jpg'):
            cv2.imwrite('./predictFace/p.jpg' ,f)


    cv2.imshow('camera', frame)

    key = cv2.waitKey(10)


    if key == 27:
        break
cv2.destroyWindow('camera')
camera.release()  # 释放cap对象
