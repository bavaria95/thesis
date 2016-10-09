import cv2
import sys
import numpy as np

# to what size resize image
N = 96

# Create the haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def extract_faces(img):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_faces = []

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # resize images
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (N, N), interpolation=cv2.INTER_LANCZOS4)

        new_faces.append(face)

    return new_faces

# Get user supplied values
imagePath = '1.png'

# Read the image
image = cv2.imread(imagePath)

f = extract_faces(image)
cv2.imshow('', f[0])
cv2.waitKey(0)
