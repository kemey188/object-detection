# -*- coding:utf-8 –*-

"""
train_model.py
"""

import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imageNetWork.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-m", "--model", required = True)
ap.add_argument("-p", "--plot", type=str, default="plot.png")
args = vars(ap.parse_args())

#初始化参数
EPOCHS  = 25
INIT_LR = 1e-3
#BATCH_SIZE = 32
BATCH_SIZE = 3

#初始化data和labels
print "[INFO] loading images..."
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# 加载图像
idx = 1
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    print "[%s] read done %d" % (imagePath, idx)
    image = cv2.resize(image, (28,28))
    idx += 1
    image = img_to_array(image)
    data.append(image)
    
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "santa_claus" else 0
    labels.append(label)
    

# 正则化 -> [0,1]
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

(train_X, test_X, train_Y, test_Y) = train_test_split(data,
                                                      labels,
                                                      test_size=.25,
                                                      random_state=42)
# 将labels转换为vector
train_Y = to_categorical(train_Y, num_classes=2)
test_Y = to_categorical(test_Y, num_classes=2)

# 通过对训练集进行随机旋转、偏移、翻转、剪切，从而增加数据量
aug = ImageDataGenerator(rotation_range=30, width_shift_range=.1, height_shift_range=.1,
                        shear_range=.2, zoom_range=.2, horizontal_flip=True, fill_mode="nearest")

# 网络初始化
print "[INFO] compiling model..."
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

"""
# for multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
"""

# 训练
print "[INFO] training network..."
H = model.fit_generator(aug.flow(train_X, train_Y, batch_size=BATCH_SIZE),
                        validation_data=(test_X, test_Y),
                        steps_per_epoch=len(train_X) // BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1)

print "[INFO] serializing network..."
model.save(args["model"])

# 输出metrics
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuarcy on santa_claus/not_santa_claus")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuarcy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
