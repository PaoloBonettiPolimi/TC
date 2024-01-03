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

#from imblearn.over_sampling import SMOTE
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

    #train['shear'] = train.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)
    #val['shear'] = val.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)
    #test['shear'] = test.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)

    return train, val, test

def preprocess_data(x_train, x_val, x_test, y_train, y_val, y_test, cols, standard):
    scaler = StandardScaler()
    train_std,val_std,test_std = x_train,x_val,x_test

    train_std['lat'] = train_std['latitude']
    train_std['lon'] = train_std['longitude']
    val_std['lat'] = val_std['latitude']
    val_std['lon'] = val_std['longitude']
    test_std['lat'] = test_std['latitude']
    test_std['lon'] = test_std['longitude']

    train_std,val_std,test_std = x_train,x_val,x_test

    if standard=='yes':
        scaler = StandardScaler()
        train_std,val_std,test_std = x_train,x_val,x_test
        train_std[cols] = scaler.fit_transform(x_train[cols])
        val_std[cols] = scaler.transform(x_val[cols])
        test_std[cols] = scaler.transform(x_test[cols])

    # remove std of lat-lon
    train_std['lat'] = train_std['latitude']/100
    train_std['lon'] = train_std['longitude']/100
    val_std['lat'] = val_std['latitude']/100
    val_std['lon'] = val_std['longitude']/100
    test_std['lat'] = test_std['latitude']/100
    test_std['lon'] = test_std['longitude']/100
    print(train_std.lat)

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
    parser.add_argument("--data_path", default='/home/b/b382633/TC_adds/csv_extracted/')
#    'NAOP_S', 'BLOCK_S', 'NAOM_S', 'RIDGE_S', 'nino3', 'nino4',
#       'nino34', 'nino12', 'indocW', 'indocE', 'AtlMDR', 'CarMDR', 'GulfMDR',
#       'AtlSub', 'glob', '30_40N', 'weu', 'EOF1', 'EOF2', 'EOF3', 'Arctic',
#       'eq', '60N', '60S', 'weu.1', 'eeu', 'cus', 'NAO', 'EAT', 'ABLOCK',
#       'NAOP', 'BLOCK', 'NAOM', 'RIDGE', 'EAS', 'Arctic.1', 'weu.2',
#       'NAO_classic', 'PNA', 'NPD'])
    parser.add_argument("--save", default='no')
    parser.add_argument("--savepath_model", default='/home/b/b382633/TC_adds/models/Unet/')
    parser.add_argument("--savepath_prediction", default='/home/b/b382633/TC_adds/predictions/Unet/')
    parser.add_argument("--single_std", default='no')

    args = parser.parse_args()
    print(args)

    ##################### load data ########################

    ### x_train, x_val, x_test, y_train, y_val, y_test lists containing dataframes, one for each area considered
    
    train_img_std = np.load('/home/b/b382633/TC_adds/csv_extracted/Sindiantrain_img_std_indices.npy')
    val_img_std = np.load('/home/b/b382633/TC_adds/csv_extracted/Sindianval_img_std_indices.npy')
    test_img_std = np.load('/home/b/b382633/TC_adds/csv_extracted/Sindiantest_img_std_indices.npy')
    #train_img_std = train_img_std[:,:,:,:9]
    #val_img_std = val_img_std[:,:,:,:9]
    #test_img_std = test_img_std[:,:,:,:9]
    
    y_train_img = np.load('/home/b/b382633/TC_adds/csv_extracted/Sindiany_train_img.npy')
    y_val_img = np.load('/home/b/b382633/TC_adds/csv_extracted/Sindiany_val_img.npy')
    y_test_img = np.load('/home/b/b382633/TC_adds/csv_extracted/Sindiany_test_img.npy')

    train_img_std_padded = np.pad(train_img_std, ((0,0),(1,2),(1,2),(0,0)), 'constant')
    val_img_std_padded = np.pad(val_img_std, ((0,0),(1,2),(1,2),(0,0)), 'constant')
    test_img_std_padded = np.pad(test_img_std, ((0,0),(1,2),(1,2),(0,0)), 'constant')

    df = pd.read_csv('/home/b/b382633/TC_adds/csv_extracted/target_test_dailymeans_Sindian.csv')

    ##################### model training and testing ########################
    
    for lag in range(14):
        if lag==0:
            model = simple_unet_model(16,32,54)
            model.compile(loss='binary_crossentropy', optimizer='adam')
            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                        verbose=1, mode='auto', restore_best_weights=True)
            model.fit(train_img_std_padded, y_train_img, validation_data=(val_img_std_padded,y_val_img),
                        callbacks=[monitor],epochs=100)
            t = model.predict(test_img_std_padded)

            ### t contains the predictions for each day and area

            ### Brier scores ### 
            print(f'All zeros Brier score: {brier_score_loss(y_test_img.reshape(-1,1), np.zeros(len(y_test_img.reshape(-1,1))))}')
            print(f'Model Brier score 1: {brier_score_loss(y_test_img.reshape(-1,1), t.reshape(-1,1))}')
            print(f'All Zeros Brier on Probabilities: {np.average((df.new_target/100) ** 2)}')
            print(f'Predictions Brier on Probabilities: {np.average((df.new_target.values.reshape(-1,1)/100 - t.reshape(-1,1)) ** 2)}')
            
            if args.save=='yes':
                df['predictions_lag0'] = t.reshape(-1,1)
                df.to_csv(args.savepath_prediction)

        else:
            model = simple_unet_model(16,32,54)
            model.compile(loss='binary_crossentropy', optimizer='adam')
            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                        verbose=1, mode='auto', restore_best_weights=True)
            model.fit(train_img_std_padded[:-lag], y_train_img[lag:], validation_data=(val_img_std_padded[:-lag],y_val_img[lag:]),
                        callbacks=[monitor],epochs=100)
            t = model.predict(test_img_std_padded)
            ### Brier score ### 
            print(f'All zeros Brier score {brier_score_loss(y_test_img[lag:,:,:,:].reshape(-1,1), np.zeros(len(y_test_img[lag:,:,:,:].reshape(-1,1))))}')
            print(f'Model Brier score 1: {brier_score_loss(y_test_img[lag:,:,:,:].reshape(-1,1), t[:-lag,:,:,:].reshape(-1,1))}')  

            print(f'All Zeros Brier on Probabilities: {np.average((df.new_target/100) ** 2)}')
            print(f'Predictions Brier on Probabilities: {np.average((df.new_target[377*lag:].values.reshape(-1,1)/100 - t.reshape(-1,1)[:-377*lag])**2)}')

            keras.backend.clear_session()
            
            if args.save=='yes':
                z = np.zeros((lag,13,29,1))
                tt = np.concatenate((t[:-lag,:,:,:],z))
                df['predictions_lag'+str(lag)] = tt.reshape(-1,1)
                df.to_csv(args.savepath_prediction) 
            
        ### save
        

