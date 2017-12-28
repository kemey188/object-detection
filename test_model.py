"""
test_model.py
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True)
ap.add_argument("-i", "--image", required = True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image dataset
image = cv2.resize(image, (28,28))
image = image.astype("float")/255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained cnn model
print "[INFO] loading network..."
model = load_model(args["model"])

# do classification
(notSantaClaus, santaClaus) = model.predict(image)[0]

# build the label
label = "Santa Claus" if santa > notSanta else "Not Santa Claus"
proba = santaClaus if santaClaus >  notSantaClaus else notSantaClaus
label = "{}: {:.2f}%".format(label, proba*100)

# draw the label
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,0,0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
