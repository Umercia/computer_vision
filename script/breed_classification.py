import cv2
import pickle
import numpy as np
import sys
import os

# parameters
IMG_SIZE = 100   # size of the resize images
TF_SIZE = 70     # number of features for the term frequency = number of descriptor groups
dir = os.listdir()

# input check
t=0
if len(sys.argv) == 1:
    print("pass the image adress to classify as parameters")
    t = 1
if "kmeans_model.pickle" not in dir:
    print("kmeans_model.pickle is missing in the current directory.")
    t = 1
if "svm_grid.pickle" not in dir:
    print("svm_grid.pickle is missing in the current directory.")
    t = 1
if t == 1:
    sys.exit()

# read image
img_name = sys.argv[1]     # read argument from command line
if img_name not in dir:
    print(img_name, "image is missing in the current directory.")
    sys.exit()
img = cv2.imread(img_name)

#read models
pickle_in = open("kmeans_model.pickle", "rb")
kmeans = pickle.load(pickle_in)

pickle_in = open("svm_grid.pickle", "rb")
svm = pickle.load(pickle_in)

# transformations: resize, grayscale, blur
img_gray = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=1)

# descriptors
orb = cv2.ORB_create(patchSize=10, edgeThreshold=10, nfeatures=50)
keypoints, descriptors = orb.detectAndCompute(img_gray, None)

# term frequency table
labels = kmeans.predict(descriptors)
tf = np.zeros(TF_SIZE)   # maybe define a parameters
tf = tf.astype(int)
for e in labels:
    tf[e] += 1

# prediction
print(svm.predict(tf.reshape(1, -1))[0])
