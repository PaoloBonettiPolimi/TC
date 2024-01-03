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

def dataLoad_target_noBinary(train_path_tar,val_path_tar,test_path_tar):
    train = pd.read_csv(train_path_tar)
    val = pd.read_csv(val_path_tar)
    test = pd.read_csv(test_path_tar)
    test = test.loc[test.time>='2016-04-01']

    train['target'] = train.new_target
    test['target'] = test.new_target
    val['target'] = val.new_target

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
    parser.add_argument("--features_list", default=['vo','r','u_200','u_850','v_200','v_850','tcwv','sst','shear'])
#    'NAOP_S', 'BLOCK_S', 'NAOM_S', 'RIDGE_S', 'nino3', 'nino4',
#       'nino34', 'nino12', 'indocW', 'indocE', 'AtlMDR', 'CarMDR', 'GulfMDR',
#       'AtlSub', 'glob', '30_40N', 'weu', 'EOF1', 'EOF2', 'EOF3', 'Arctic',
#       'eq', '60N', '60S', 'weu.1', 'eeu', 'cus', 'NAO', 'EAT', 'ABLOCK',
#       'NAOP', 'BLOCK', 'NAOM', 'RIDGE', 'EAS', 'Arctic.1', 'weu.2',
#       'NAO_classic', 'PNA', 'NPD'])
    parser.add_argument("--model", default='Unet')
    parser.add_argument("--save", default='no')
    parser.add_argument("--savepath", default='/home/b/b382633/TC_adds/predictions/Unet/')
    parser.add_argument("--single_std", default='no')
    args = parser.parse_args()
    print(args)

    # areas = ['Sindian','Natlantic','NWpacific','Australia','Nindian','Epacific','sPacific']
    areas = ['Sindian']
    y_train, y_val, y_test = ([] for i in range(3))

    print(areas, flush=True)

    for area in areas:
        tar_train_features_path = args.data_path+'target_train_dailymeans_'+area+'.csv'
        tar_val_features_path = args.data_path+'target_val_dailymeans_'+area+'.csv'
        tar_test_features_path = args.data_path+'target_test_dailymeans_'+area+'.csv'
        y_train_curr, y_val_curr, y_test_curr = dataLoad_target_noBinary(train_path_tar=tar_train_features_path,val_path_tar=tar_val_features_path,test_path_tar=tar_test_features_path)
        
        y_train.append(y_train_curr)
        y_val.append(y_val_curr)
        y_test.append(y_test_curr)
        print(y_train_curr.shape, y_val_curr.shape, y_test_curr.shape,flush=True)

    for i in range(len(areas)):
        variables = ['target']
        
        if i==0:
            y_train_img = extract_images(y_train[i], variables, verbose=False)
            print(y_train_img.shape)
            y_val_img = extract_images(y_val[i], variables, verbose=False)
            print(y_val_img.shape)
            y_test_img = extract_images(y_test[i], variables, verbose=False)
            print(y_test_img.shape)
            
        else:
            
            y_train_img_curr = extract_images(y_train[i], variables, verbose=False)
            print(y_train_img_curr.shape)
            y_val_img_curr = extract_images(y_val[i], variables, verbose=False)
            print(y_val_img_curr.shape)
            y_test_img_curr = extract_images(y_test[i], variables, verbose=False)
            print(y_test_img_curr.shape)

            y_train_img = np.concatenate((y_train_img,y_train_img_curr),axis=0)
            y_val_img = np.concatenate((y_val_img,y_val_img_curr),axis=0)
            y_test_img = np.concatenate((y_test_img,y_test_img_curr),axis=0)

    print(y_train_img.shape, flush=True)
    print(y_val_img.shape, flush=True)
    print(y_test_img.shape, flush=True)

    np.save(args.savepath+areas[i]+'y_train_img_noBinary.npy', y_train_img) 
    np.save(args.savepath+areas[i]+'y_val_img_noBinary.npy', y_val_img) 
    np.save(args.savepath+areas[i]+'y_test_img_noBinary.npy', y_test_img) 