import cv2
import sys
    
# Get user supplied values
#imagePath = sys.argv[1]
video_capture = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("Nariz.xml")
smileCascade = cv2.CascadeClassifier("mouth.xml")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Show only face
    for (x, y, w, h) in faces:
        scale = 125
        sized = frame[y:y+h, x:x+w]
        sizedx2 = cv2.resize(sized, (scale, scale))
        sized_gray = cv2.cvtColor(sizedx2, cv2.COLOR_BGR2GRAY)
        #sized_c = cv2.equalizeHist(sized_gray)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        eyes = eyeCascade.detectMultiScale(sized_gray)
        smile = smileCascade.detectMultiScale(sized_gray)
        nose = noseCascade.detectMultiScale(sized_gray)
        xScale = w/float(scale)
        yScale = h/float(scale)
        # for (ex,ey,ew,eh) in eyes:
            # cv2.rectangle(sized,(int(ex * xScale), int(ey * yScale)),(int((ex + ew) * xScale), int((ey + eh) * yScale)),(0,255,0),2)
        # if len(smile) != 0 :
            # sx,sy,sw,sh = smile[0]
            # cv2.rectangle(sized,(int(sx * xScale), int(sy * yScale)),(int((sx + sw) * xScale), int((sy + sh) * yScale)),(0,0,255),2)
        # if len(nose) != 0 :
            # nx,ny,nw,nh = nose[0]
            # cv2.rectangle(sized,(int(nx * xScale), int(ny * yScale)),(int((nx + nw) * xScale), int((ny + nh) * yScale)),(0,0,0),2)
        cv2.imshow("Face found", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
