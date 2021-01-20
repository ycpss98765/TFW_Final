import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

img = cv.imread('sample.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    image = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.show()