import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import timeit
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-image', type=str, default="sample1.jpg", help='give a image path')
parser.add_argument('-rotate', type=int, default=0, help='How many degree you want to roated a image')
args = parser.parse_args()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

image_path = args.image
rotate_degree = args.rotate
image  = Image.open(image_path)
image = image.rotate(rotate_degree)
img = cv.cvtColor(np.asarray(image),cv.COLOR_RGB2BGR)

start = timeit.default_timer()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
stop = timeit.default_timer()
print('Time: ', stop - start)  

if (len(faces) != 0):
    for (x, y, w, h) in faces:
        image = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()
else:
    print("No faces detected!!")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()