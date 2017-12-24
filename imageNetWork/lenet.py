# -*- coding:utf-8 –*-

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense   #Dense: full collection
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)
        
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            
        # first set of  CONV => RELU => POOLING layers
        model.add(Conv2D(20, (5,5), padding = "same", input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
        
        # second set of CONV => RELU => POOLING layers 
        model.add(Conv2D(50, (5,5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
        
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # return the constructed network architecture
        return model
        
