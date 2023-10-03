import pandas as pd
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 150)
import numpy as np
import argparse

import decimal
from decimal import Decimal

import matplotlib.pyplot as plt

from pathlib import Path  

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding labels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE
from collections import Counter
import io
import requests
from sklearn import metrics

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense # for creating regular densely-connected NN layer.
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
from sklearn.preprocessing import StandardScaler
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
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Cropping2D

################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

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
 
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def extract_images(df, variables, verbose=False):
    number_of_img, rows, cols = len(df.time.unique()), len(df.latitude.unique()), len(df.longitude.unique())
    images = np.zeros( (number_of_img, rows, cols, len(variables)) )

    df = df.sort_values(by=['time','latitude','longitude'])
    k=0
    
    for day in range(0,number_of_img):
        
        a=df.iloc[377*day:377*(day+1)]
        i=0
        for var in variables:
            images[day,:,:,i] = a.pivot(index='latitude', columns='longitude').sort_index(ascending=False)[var]
            i+=1
        k+=1
        if (k%100==0) & (verbose==True): print(k)
    return images

def dataLoad_target(train_path_tar,val_path_tar,test_path_tar):
    train = pd.read_csv(train_path_tar)
    val = pd.read_csv(val_path_tar)
    test = pd.read_csv(test_path_tar)
    test = test.loc[test.time>='2016-04-01']

    train['target'] = train.apply(lambda x: 1 if x.new_target>=50 else 0,axis=1)
    test['target'] = test.loc[test.time>='2016-04-01'].apply(lambda x: 1 if x.new_target>=50 else 0,axis=1)
    val['target'] = val.apply(lambda x: 1 if x.new_target>=50 else 0,axis=1)

    y_train = train.loc[:,['target','latitude','longitude','time']]
    y_val = val.loc[:,['target','latitude','longitude','time']]
    y_test = test.loc[:,['target','latitude','longitude','time']]

    return y_train, y_val, y_test

def dataLoad_features(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    test = test.loc[test.time>='2016-04-01']

    train['shear'] = train.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)
    val['shear'] = val.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)
    test['shear'] = test.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)

    return train, val, test

def preprocess_data(x_train, x_val, x_test, y_train, y_val, y_test, cols):
    scaler = StandardScaler()
    train_std,val_std,test_std = x_train,x_val,x_test

    train_std['lat'] = train_std['latitude']
    train_std['lon'] = train_std['longitude']
    val_std['lat'] = val_std['latitude']
    val_std['lon'] = val_std['longitude']
    test_std['lat'] = test_std['latitude']
    test_std['lon'] = test_std['longitude']

    train_std,val_std,test_std = x_train,x_val,x_test
    train_std[cols] = scaler.fit_transform(x_train[cols])
    val_std[cols] = scaler.transform(x_val[cols])
    test_std[cols] = scaler.transform(x_test[cols])

    train_img_std = extract_images(train_std, cols, verbose=False)
    print(train_img_std.shape)
    val_img_std = extract_images(val_std, cols, verbose=False)
    print(val_img_std.shape)
    test_img_std = extract_images(test_std, cols, verbose=False)
    print(test_img_std.shape)

    variables = ['target']
    y_train_img = extract_images(y_train, variables, verbose=False)
    print(y_train_img.shape)
    y_val_img = extract_images(y_val, variables, verbose=False)
    print(y_val_img.shape)
    y_test_img = extract_images(y_test, variables, verbose=False)
    print(y_test_img.shape)

    return train_img_std, val_img_std, test_img_std, y_train_img, y_val_img, y_test_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features_path", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/train_dailymeans.csv')
    parser.add_argument("--val_features_path", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/val_dailymeans.csv')
    parser.add_argument("--test_features_path", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/test_dailymeans.csv')
    parser.add_argument("--train_features_path2", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/train_dailymeans_Atlantic.csv')
    parser.add_argument("--val_features_path2", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/val_dailymeans_Atlantic.csv')
    parser.add_argument("--test_features_path2", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/test_dailymeans_Atlantic.csv')
    parser.add_argument("--tar_train_features_path", default='/Users/paolo/Documents/TC_old/data/NewData_csv/training_sets_withrealtom.csv')
    parser.add_argument("--tar_val_features_path", default='/Users/paolo/Documents/TC_old/data/NewData_csv/validation_sets_withrealtom.csv')
    parser.add_argument("--tar_test_features_path", default='/Users/paolo/Documents/TC_old/data/NewData_csv/test_sets_withrealtom.csv')
    parser.add_argument("--tar_train_features_path2", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/target_train_dailymeans_Atlantic.csv')
    parser.add_argument("--tar_val_features_path2", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/target_val_dailymeans_Atlantic.csv')
    parser.add_argument("--tar_test_features_path2", default='/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/target_test_dailymeans_Atlantic.csv')
    parser.add_argument("--features_list", default=['vo','r','u_200','u_850','v_200','v_850','tcwv','sst','shear'])
    parser.add_argument("--model", default='extratree')
    parser.add_argument("--save", default='no')
    parser.add_argument("--savepath", default='trial.csv')
    parser.add_argument("--savepath2", default='trial2.csv')

    args = parser.parse_args()
    print(args)

    ##################### load and standardize data ########################

    x_train, x_val, x_test = dataLoad_features(train_path=args.train_features_path,val_path=args.val_features_path,test_path=args.test_features_path)
    y_train, y_val, y_test = dataLoad_target(train_path_tar=args.tar_train_features_path,val_path_tar=args.tar_val_features_path,test_path_tar=args.tar_test_features_path)

    x_train2, x_val2, x_test2 = dataLoad_features(train_path=args.train_features_path2,val_path=args.val_features_path2,test_path=args.test_features_path2)
    y_train2, y_val2, y_test2 = dataLoad_target(train_path_tar=args.tar_train_features_path2,val_path_tar=args.tar_val_features_path2,test_path_tar=args.tar_test_features_path2)

    train_img_std,val_img_std,test_img_std,y_train_img,y_val_img,y_test_img = preprocess_data(x_train,x_val,x_test,y_train,y_val,y_test,args.features_list)
    train_img_std2,val_img_std2,test_img_std2,y_train_img2,y_val_img2,y_test_img2 = preprocess_data(x_train2,x_val2,x_test2,y_train2,y_val2,y_test2,args.features_list)

    train_img_std = np.concatenate((train_img_std,train_img_std2),axis=0)
    val_img_std = np.concatenate((val_img_std,val_img_std2),axis=0)
    #test_img_std = np.concatenate((test_img_std,test_img_std2),axis=0)

    y_train_img = np.concatenate((y_train_img,y_train_img2),axis=0)
    y_val_img = np.concatenate((y_val_img,y_val_img2),axis=0)
    #y_test_img = np.concatenate((y_test_img,y_test_img2),axis=0)

    print(y_train_img.shape)
    print(y_val_img.shape)

    ##################### model training and testing ########################
    
    test = pd.read_csv(args.test_features_path)
    test = test.loc[test.time>='2016-04-01']
    test2 = pd.read_csv(args.test_features_path2)
    test2 = test2.loc[test2.time>='2016-04-01']

    for lag in range(14):
        if lag == 0:
            if args.model=='CNN':
                model = Sequential()
                model.add(layers.Input(shape=(13, 29, len(args.features_list))))
                model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
                model.add(layers.MaxPooling2D((2, 2), padding='same'))
                model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
                model.add(layers.UpSampling2D((2,2)))
                model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # sigmoidal activation to have probabilities
                model.add(layers.Cropping2D(cropping=((0,1),(0,1))))
                model.summary()

                model.compile(loss='binary_crossentropy', optimizer='adam')

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)

                model.fit(train_img_std, y_train_img, validation_data=(val_img_std,y_val_img),
                            callbacks=[monitor],epochs=100)
                t = model.predict(test_img_std)
                t2 = model.predict(test_img_std2)

            if args.model=='Unet':
                model = simple_unet_model(16,32,len(args.features_list))

                model.compile(loss='binary_crossentropy', optimizer='adam')

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)
                
                pad_tns = tf.constant([ [0,0], [1,2], [1,2], [0,0] ])
                train_img_std_padded = np.array(tf.pad(train_img_std,pad_tns,mode='CONSTANT',constant_values=0))
                val_img_std_padded = np.array(tf.pad(val_img_std,pad_tns,mode='CONSTANT',constant_values=0))
                test_img_std_padded = np.array(tf.pad(test_img_std,pad_tns,mode='CONSTANT',constant_values=0))
                test_img_std_padded2 = np.array(tf.pad(test_img_std2,pad_tns,mode='CONSTANT',constant_values=0))

                model.fit(train_img_std_padded, y_train_img, validation_data=(val_img_std_padded,y_val_img),
                            callbacks=[monitor],epochs=100)

                t = model.predict(test_img_std_padded)
                t2 = model.predict(test_img_std_padded2)
            ### confusion matrices ###
            
            #ranges = [0.025,0.05,0.075,0.1]

            #for j in ranges:
            #    classes = []
            #    for i in t:
            #        if i<=j: classes.append(0)
            #        else: classes.append(1)

                #ConfusionMatrixDisplay(confusion_matrix(y_test, classes)).plot(colorbar=False,cmap=plt.cm.Blues, values_format='d')
                #ConfusionMatrixDisplay(confusion_matrix(y_test, classes, normalize='true')).plot(colorbar=False,cmap=plt.cm.Blues)

            ### ROC curve ###
            #plot_roc(t,y_test)

            ### Calibration curve ###
            #display = CalibrationDisplay.from_predictions(y_test, t, n_bins=10)

            ### Brier score (w.r.t. class) ###
            print(f'All zeros Brier score 1: {brier_score_loss(y_test_img.reshape(-1,1), np.zeros(len(y_test_img.reshape(-1,1))))}')
            print(f'Model Brier score 1: {brier_score_loss(y_test_img.reshape(-1,1), t.reshape(-1,1))}')  

            print(f'All zeros Brier score 1: {brier_score_loss(y_test_img2.reshape(-1,1), np.zeros(len(y_test_img2.reshape(-1,1))))}')
            print(f'Model Brier score 1: {brier_score_loss(y_test_img2.reshape(-1,1), t2.reshape(-1,1))}') 

            ### save results ###
            if args.save=='yes':
                test['predictions_lag0'] = t.reshape(-1,1)
                test.to_csv(args.savepath)
                test2['predictions_lag0'] = t2.reshape(-1,1)
                test2.to_csv(args.savepath2)

        else:
            if args.model=='CNN':
                model = Sequential()
                model.add(layers.Input(shape=(13, 29, len(args.features_list))))
                model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
                model.add(layers.MaxPooling2D((2, 2), padding='same'))
                model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
                model.add(layers.UpSampling2D((2,2)))
                model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # sigmoidal activation to have probabilities
                model.add(layers.Cropping2D(cropping=((0,1),(0,1))))
                model.summary()

                model.compile(loss='binary_crossentropy', optimizer='adam')

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)

                model.fit(train_img_std[:-lag], y_train_img[lag:], validation_data=(val_img_std[:-lag],y_val_img[lag:]),
                            callbacks=[monitor],epochs=100)

                t = model.predict(test_img_std[:-lag])
                t2 = model.predict(test_img_std2[:-lag])

            if args.model=='Unet':
                model = simple_unet_model(16,32,len(args.features_list))

                model.compile(loss='binary_crossentropy', optimizer='adam')

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)
                
                pad_tns = tf.constant([ [0,0], [1,2], [1,2], [0,0] ])
                train_img_std_padded = np.array(tf.pad(train_img_std,pad_tns,mode='CONSTANT',constant_values=0))
                val_img_std_padded = np.array(tf.pad(val_img_std,pad_tns,mode='CONSTANT',constant_values=0))
                test_img_std_padded = np.array(tf.pad(test_img_std,pad_tns,mode='CONSTANT',constant_values=0))
                test_img_std_padded2 = np.array(tf.pad(test_img_std2,pad_tns,mode='CONSTANT',constant_values=0))

                model.fit(train_img_std_padded[:-lag], y_train_img[lag:], validation_data=(val_img_std_padded[:-lag],y_val_img[lag:]),
                            callbacks=[monitor],epochs=100)

                t = model.predict(test_img_std_padded[:-lag])
                t2 = model.predict(test_img_std_padded2[:-lag])

            ### confusion matrices ###
            #ranges = [0.025,0.05,0.075,0.1]
            
            #for j in ranges:
            #    classes = []
            #    for i in t:
            #        if i<=j: classes.append(0)
            #        else: classes.append(1)
            
                #ConfusionMatrixDisplay(confusion_matrix(y_test[lag*377:], classes)).plot(colorbar=False,cmap=plt.cm.Blues, values_format='d')
                #ConfusionMatrixDisplay(confusion_matrix(y_test[lag*377:], classes, normalize='true')).plot(colorbar=False,cmap=plt.cm.Blues)
            
            ### ROC curve ###
            #plot_roc(t,y_test[lag*377:])
            
            ### Calibration curve ###
            #display = CalibrationDisplay.from_predictions(y_test[lag*377:], t, n_bins=10)

            ### Brier score (w.r.t. class) ###
            print(f'All zeros Brier score 1: {brier_score_loss(y_test_img[lag:].reshape(-1,1), np.zeros(len(y_test_img[lag:].reshape(-1,1))))}')
            print(f'Model Brier score 1: {brier_score_loss(y_test_img[lag:].reshape(-1,1), t.reshape(-1,1))}')
            print(f'All zeros Brier score 2: {brier_score_loss(y_test_img2[lag:].reshape(-1,1), np.zeros(len(y_test_img2[lag:].reshape(-1,1))))}')
            print(f'Model Brier score 2: {brier_score_loss(y_test_img2[lag:].reshape(-1,1), t2.reshape(-1,1))}')

            ### save results ###
            if args.save=='yes':
                z = np.zeros((lag,13,29,1))
                tt = np.concatenate((t,z))
                test['predictions_lag'+str(lag)] = tt.reshape(-1,1)
                test.to_csv(args.savepath)    
                tt2 = np.concatenate((t2,z))
                test2['predictions_lag'+str(lag)] = tt2.reshape(-1,1)
                test2.to_csv(args.savepath2) 
