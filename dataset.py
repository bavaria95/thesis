import os
import numpy as np
import cv2
import collections
import face_detect

def read_all_labels(path):
    d = collections.defaultdict(dict)
    for root, dirs, files in os.walk(path):
        for f in files:
            per, batch = root.split('/')[-2: ]
            with open(os.path.join(root, f), 'r') as fd:
                d[per][batch] = float(fd.read())

    return d

def recognize_face(path):
    image = cv2.imread(path)
    f = face_detect.extract_face(image, 64)

    return f

def read_all_files(path, path_labels):
    d = read_all_labels(path_labels)
    faces = []
    labels = []

    for root, dirs, files in os.walk(path):
        for f in files:
            per, batch = root.split('/')[-2: ]
            if per in d and batch in d[per]:
                face = recognize_face(os.path.join(root, f))
                if face.shape != (0,):
                    faces.append(face)
                    labels.append(d[per][batch])

    faces = np.array(faces)
    labels = np.array(labels)

    return (faces, labels)

faces, labels = read_all_files('/home/bavaria/Downloads/cohn-kanade-images', '/home/bavaria/Downloads/Emotion')
