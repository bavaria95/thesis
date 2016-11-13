import cv2
import sys

import numpy as np
import theano
import theano.tensor as T

import lasagne
    
size = 96
    
def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, size, size),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=7,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
    
    
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# video
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

# load images
happy = cv2.imread("happy.jpg")
sad = cv2.imread("sad.jpg")
none = cv2.imread("none.jpg")

# prepare network
input_var = T.tensor4('inputs')
network = build_cnn(input_var)

with np.load('model378.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

while True:
    ret, frame = video_capture.read()
    cv2.imshow("Face", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('a'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            small = cv2.resize(frame[y:y+h, x:x+w], (size,size))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = (gray / np.float(256))
            arr = np.array(gray)
            arr = arr.reshape(1,1,size,size)
            
            out = lasagne.layers.get_output(network, arr)
            # print(out.eval())
            tab = out.eval()
            cv2.imshow("Face expression", frame)
            print ("anger = {:.2f}%\ncontempt = {:.2f}%\ndisgust = {:.2f}%\nfear = {:.2f}%\nhappy = {:.2f}%\nsadness = {:.2f}%\nsurprise = {:.2f}%\n".format(tab[0][0]*100,tab[0][1]*100,tab[0][2]*100,tab[0][3]*100,tab[0][4]*100,tab[0][5]*100,tab[0][6]*100))
            break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
