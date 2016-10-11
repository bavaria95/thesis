import cv2
import sys
import numpy as np

# Create the haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_face(img, N):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if faces.shape == (0,):
        return np.array([])

    # assuming that only one face on the image
    face = faces[0]

    # resize the image
    (x, y, w, h) = face
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (N, N), interpolation=cv2.INTER_LANCZOS4)

    return face

# # Get user supplied values
# imagePath = '1.png'

# # Read the image
# image = cv2.imread(imagePath)

# f = extract_faces(image)
# cv2.imshow('', f[0])
# cv2.waitKey(0)
