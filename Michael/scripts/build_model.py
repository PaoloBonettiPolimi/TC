import pandas as pd
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 150)
import numpy as np
import argparse

import decimal
from decimal import Decimal

from pathlib import Path  

from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding labels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

#from imblearn.over_sampling import SMOTE
from collections import Counter
import io
import requests

import tensorflow as tf
from tensorflow import keras
from keras import Sequential # for creating a linear stack of layers for our Neural Network
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout # for adding Concolutional and densely-connected NN layers.
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout,BatchNormalization,Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers
from keras import callbacks
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense # for creating regular densely-connected NN layer.
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout,MaxPooling2D # for adding Concolutional and densely-connected NN layers.
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Cropping2D, ZeroPadding2D

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):    
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    
    s = ZeroPadding2D(padding=((1, 2),(1,2)))(inputs)

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    c10 = Cropping2D(cropping=((1,2),(1,2)))(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

def alt_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    
    s = ZeroPadding2D(padding=((0,1),(4,5)))(inputs)

    #Contraction path
    c1 = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    fl = Flatten()(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    # u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    # u8 = concatenate([u8, c2])
    # c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    # c8 = Dropout(0.1)(c8)
    # c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u9 = concatenate([u9, c2], axis=3)
    c9 = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    c10 = Cropping2D(cropping=((1,2),(5,6)))(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='CNN_base_0')
    parser.add_argument('--savepath_model', default='/work/bk1318/b382633/models/fresh_models/CNN_base_0')
    
    args = parser.parse_args()
    print(args)
    
    model_name = args.model_name
    savepath_model = args.savepath_model
    
    # model_name = "CNN_1_1"   # choose the model you want to build
    
    if model_name=='CNN_base_0':
        model = Sequential()
        model.add(layers.Input(shape=(13, 29, 9)))
        model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
        model.add(layers.UpSampling2D((2,2)))
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
        model.add(layers.Cropping2D(cropping=((0,1),(0,1))))
        model.summary()
        
        model.compile(loss='binary_crossentropy', optimizer='adam')
    
#     elif model_name=='CNN_1_0':
#         model = Sequential()
#         model.add(layers.Input(shape=(31, 71, 9)))
#         model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
#         model.add(layers.MaxPooling2D((2, 2), padding='same'))
#         model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
#         model.add(layers.MaxPooling2D((2, 2), padding='same'))
#         model.add(layers.Flatten())
#         model.add(layers.Dense(7*15*32, activation='relu'))
#         model.add(layers.Reshape((7,15,32)))
#         # model.add(layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same'))
#         model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
#         model.add(layers.UpSampling2D((2,2)))
#         model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # sigmoidal activation to have probabilities
#         model.add(layers.Cropping2D(cropping=((0,1),(0,1))))
#         model.summary()

#         model.compile(loss='binary_crossentropy', optimizer='adam')
        
    elif model_name=='CNN_1_0':
        model = Sequential()
        model.add(layers.Input(shape=(31, 71, 9)))
        model.add(layers.Conv2D(9, (3,3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.Conv2D(9, (3,3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.Conv2D(9, (3,3), activation='relu', padding='same'))
        model.add(layers.UpSampling2D((2,2)))
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # sigmoidal activation to have probabilities
        model.add(layers.Cropping2D(cropping=((2,1),(4,3))))
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam')
        
    elif model_name=='Unet_base_0':
        model = simple_unet_model(13,29,9)

        model.compile(loss='binary_crossentropy', optimizer='adam')
        
    elif model_name=='Unet_1_0':
        model = alt_unet_model(31,71,9)

        model.compile(loss='binary_crossentropy', optimizer='adam')
        
    else:
        print("model name is incorrect")
        
    model.save(savepath_model)   # path in which you want to save the model