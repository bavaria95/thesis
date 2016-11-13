import cv2
import sys
import numpy as np    
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def write_matrix_to_textfile(a_matrix, file_to_write):

    def compile_row_string(a_row):
        return str(a_row).strip(']').strip('[').replace(' ','')

    with open(file_to_write, 'w') as f:
        for row in a_matrix:
            f.write(compile_row_string(row)+'\n')

    return True


video_capture = cv2.VideoCapture(0)
counter = 0
ret, frame = video_capture.read()
while True:
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

    # Show only face
    for (x, y, w, h) in faces:
        small = cv2.resize(frame[y:y+h, x:x+w], (128,128))
        cv2.imwrite(str(counter) + ".jpg", small)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("Face found", frame)
        
        write_matrix_to_textfile(small, 'data.dat')
    
    counter = counter + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
