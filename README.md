# Projeto-Python-OpenCV
# Codigo referente ao video sobre a indentificação de rosto em fotos:

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


test_image = cv.imread('test.jpg')
test_image_gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
plt.imshow(test_image_gray, cmap='gray')

def convertToRGB(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

haar_cascade_face = cv.CascadeClassifier(r'\Users\Lucas Bernardes\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)
print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(convertToRGB(test_image))
plt.show()
