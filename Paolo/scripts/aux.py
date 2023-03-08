#import libraries
import pandas as pd
print('pandas: %s' % pd.__version__)

pd.options.display.max_columns = None
pd.set_option('display.max_rows', 150)

import numpy as np
print('geopandas: %s' % np.__version__)

# Tensorflow / Keras
import tensorflow as tf # used to access argmax function
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
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
from joblib import Parallel, delayed


# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version

import decimal
from decimal import Decimal

import keras 
import tensorflow as tf
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

from pathlib import Path  

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding labels
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from tensorflow.keras.utils import plot_model

def extract_target(train_path,val_path,test_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    y_train = train.loc[:,['Real_tom_lsm','latitude','longitude','time']]
    y_val = val.loc[:,['Real_tom_lsm','latitude','longitude','time']]
    y_test = test.loc[:,['Real_tom_lsm','latitude','longitude','time']]

    return y_train,y_val,y_test

def find_neighbour(actual_clust, keys, radius): 
    for datum in actual_clust:
        lat = datum[0]
        lon = datum[1]
        time = datum[2]
        for c in keys:
            clat = c[0]
            clon = c[1]
            ctime = c[2]
            if ((time==ctime) & (abs(lat-clat)<=radius) & (abs(lon-clon)<=radius)): return c
    return ''

def compute_cyclones(df,radius): 
    actual_cluster = []
    output = []
    df['key']=df.apply(lambda x: [x.latitude,x.longitude,x.time], axis=1)
    keys_to_check = df['key']

    while(len(keys_to_check)>0):
        if (actual_cluster == []):
            actual_key = keys_to_check.iloc[0]
            actual_cluster.append(actual_key)
            keys_to_check = keys_to_check[keys_to_check.apply(lambda x: x!=actual_key)]

        key_to_aggr = find_neighbour(actual_cluster, keys_to_check, radius)
        if key_to_aggr != '':
            actual_cluster.append(key_to_aggr)
            keys_to_check = keys_to_check[keys_to_check.apply(lambda x: x!=key_to_aggr)]
        else:
            output.append(actual_cluster)
            actual_cluster = []
    if (len(actual_cluster)>0): output.append(actual_cluster)
    return output

def compute_centers(cyclones):
    centers = []
    for cyclone in cyclones:
        lats = []
        lons = []
        for point in cyclone:
            lats.append(point[0])
            lons.append(point[1])
        time = cyclone[0][2]
        centers.append([lats[np.argmin(abs(lats-np.mean(lats)))], lons[np.argmin(abs(lons-np.mean(lons)))], time])
    return centers 

def extract_center_points(df,centers):
    df = df.loc[df.apply(lambda x: [x.latitude,x.longitude,x.time] in centers, axis=1)]
    return df


def extract_cyclones(radius=2.5,train_path='/Users/paolo/Documents/TC/data/training_real_tom_target.csv',val_path='/Users/paolo/Documents/TC/data/validation_real_tom_target.csv',test_path='/Users/paolo/Documents/TC/data/test_real_tom_target.csv'):
    y_train,y_val,y_test = extract_target(train_path,val_path,test_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    y_test_cyclones = y_test[y_test.Real_tom_lsm==1]
    y_train_cyclones = y_train[y_train.Real_tom_lsm==1]
    y_val_cyclones = y_val[y_val.Real_tom_lsm==1]

    train_cyclones = compute_cyclones(y_train_cyclones,radius)
    test_cyclones = compute_cyclones(y_test_cyclones,radius)
    val_cyclones = compute_cyclones(y_val_cyclones,radius)

    test_centers = compute_centers(test_cyclones)
    train_centers = compute_centers(train_cyclones)
    val_centers = compute_centers(val_cyclones)

    #train_center_dataset = extract_center_points(train,train_centers)
    #val_center_dataset = extract_center_points(val,val_centers)
    #test_center_dataset = extract_center_points(test,test_centers)

    #return train_center_dataset, val_center_dataset, test_center_dataset

    return train_centers, val_centers, test_centers

def extract_zeros_samples(n_train_centers, train_path='/Users/paolo/Documents/TC/data/training_real_tom_target.csv'):
    train = pd.read_csv(train_path)
    return train.loc[train.Real_tom_lsm==0].sample(n_train_centers).loc[:,['latitude','longitude','time']].values.tolist()


def extract_images_withCenter(df, train_cyclones, variables=[ 'vo', 'r', 'u_200', 'u_850', 'v_200','v_850', 'ttr','sst'], x_extension=9, y_extension=9, scale=2.5, verbose=False):
    number_of_img, rows, cols = len(train_cyclones), x_extension, y_extension
    n_layers = len(variables)
    images = np.zeros( (number_of_img, rows, cols, n_layers) )
    
    #df = df.sort_values(by=['time','latitude','longitude'])
    k=0
    
    df = df.set_index(['time','latitude','longitude'], drop=True)
    df.sort_index(level=['time','latitude', 'longitude'], ascending=[1,1,1], inplace=True)

    for k,center in enumerate(train_cyclones):
        
        x_to_add = scale*(x_extension/2)
        y_to_add = scale*(y_extension/2)

        images[k,:,:,:] = df.loc(axis=0)[
                          center[2], (center[0]-y_to_add):(center[0]+y_to_add),(center[1]-x_to_add):(center[1]+x_to_add)
                      ].values.reshape(x_extension,y_extension,len(variables))
        if (k%100==0) & (verbose==True): print(k)
    return images
'''        i=0
        for var in variables:
        #images[k,:,:,:] = unfolded_img.pivot(index='latitude', columns='longitude',values=variables).values.reshape(x_extension,y_extension,n_layers)
            images[k,:,:,i] = unfolded_img.pivot(index='latitude', columns='longitude', values=var)
            i+=1
        k+=1
        if (k%10==0) & (verbose==True): print(k)
    return images'''
    
def extract_images_all(df, variables, verbose=False):
    number_of_img, rows, cols = len(df.time.unique()), len(df.latitude.unique()), len(df.longitude.unique())
    images = np.zeros( (number_of_img, rows, cols, len(variables)) )
    
    df = df.sort_values(by=['time','latitude','longitude'])
    k=0
    
    for day in range(0,number_of_img):
        
        a=df.iloc[377*day:377*(day+1)]
        i=0
        for var in variables:
            images[day,:,:,i] = a.pivot(index='latitude', columns='longitude')[var]
            i+=1
        k+=1
        if (k%100==0) & (verbose==True): print(k)
    return images
    

