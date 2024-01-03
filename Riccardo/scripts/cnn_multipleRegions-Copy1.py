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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='/work/bk1318/b382633/npy_1/')
    parser.add_argument("--columns", type=str, default="0,1,2,3,4,5,6,7,10")
#    'NAOP_S', 'BLOCK_S', 'NAOM_S', 'RIDGE_S', 'nino3', 'nino4',
#       'nino34', 'nino12', 'indocW', 'indocE', 'AtlMDR', 'CarMDR', 'GulfMDR',
#       'AtlSub', 'glob', '30_40N', 'weu', 'EOF1', 'EOF2', 'EOF3', 'Arctic',
#       'eq', '60N', '60S', 'weu.1', 'eeu', 'cus', 'NAO', 'EAT', 'ABLOCK',
#       'NAOP', 'BLOCK', 'NAOM', 'RIDGE', 'EAS', 'Arctic.1', 'weu.2',
#       'NAO_classic', 'PNA', 'NPD'])
    parser.add_argument("--model", default='CNN')
    parser.add_argument("--save", default='no')
    parser.add_argument("--sourcepath_model", default='/home/b/b382633/TC_adds/models/base_models/Unet2')
    parser.add_argument("--savepath_model", default='/home/b/b382633/TC_adds/models/Unet_1/')
    parser.add_argument("--savepath_prediction", default='/home/b/b382633/TC_adds/predictions/Unet_1/')
    parser.add_argument("--x_test_path", default='/home/b/b382633/TC_adds/csv_extracted/test_dailymeans_Sindian.csv')
    # parser.add_argument("--single_std", default='no')

    args = parser.parse_args()
    print(args)

    ##################### load numpy arrays ########################

    # areas = ['Sindian','Natlantic','NWpacific','Australia','Nindian','Epacific','sPacific']
    areas = ['Sindian']
    
    columns = args.columns.split(',')
    columns = [int(col) for col in columns]
    print(columns, flush=True)

    print(areas, flush=True)
    
    # train_img_std = np.array([])
    # val_img_std = np.array([])
    # test_img_std = np.array([])
    # y_train_img = np.array([])
    # y_val_img = np.array([])
    # y_test_img = np.array([])

    for i in range(len(areas)):
        X_train_area = np.load(args.data_path+areas[i]+'_train_img_std.npy')[:,:,:,columns]
        X_val_area = np.load(args.data_path+areas[i]+'_val_img_std.npy')[:,:,:,columns]
        X_test_area = np.load(args.data_path+areas[i]+'_test_img_std.npy')[:,:,:,columns]
        y_train_area = np.load(args.data_path+areas[i]+'_y_train_img.npy')
        y_val_area = np.load(args.data_path+areas[i]+'_y_val_img.npy')
        y_test_area = np.load(args.data_path+areas[i]+'_y_test_img.npy')
        
        if i == 0:
            train_img_std = X_train_area
            val_img_std = X_val_area
            test_img_std = X_test_area
            y_train_img = y_train_area
            y_val_img = y_val_area
            y_test_img = y_test_area
        else:
            train_img_std = np.append(train_img_std, X_train_area, axis=0)
            val_img_std = np.append(val_img_std, X_val_area, axis=0)
            test_img_std = np.append(test_img_std, X_test_area, axis=0)
            y_train_img = np.append(y_train_img, y_train_area, axis=0)
            y_val_img = np.append(y_val_img, y_val_area, axis=0)
            y_test_img = np.append(y_test_img, y_test_area, axis=0)

    print(y_train_img.shape, flush=True)
    print(y_val_img.shape, flush=True)
    print(y_test_img.shape, flush=True)
    print(train_img_std.shape, flush=True)
    print(val_img_std.shape, flush=True)
    print(test_img_std.shape, flush=True)
    
    x_test = []
    for area in areas: 
        test = pd.read_csv(args.x_test_path)
        x_test_curr = test.loc[test.time>='2016-04-01']
        x_test.append(x_test_curr)
        
    print(len(x_test))
    
    print(args.sourcepath_model)

    if args.model=='Unet':
        #pad_tns = tf.constant([[0,0], [1,2], [1,2], [0,0] ])
        # train_img_std_padded = np.pad(train_img_std, ((0,0),(1,2),(1,2),(0,0)), 'constant')
        # val_img_std_padded = np.pad(val_img_std, ((0,0),(1,2),(1,2),(0,0)), 'constant')
        # test_img_std_padded = np.pad(test_img_std, ((0,0),(1,2),(1,2),(0,0)), 'constant')
        train_img_std_padded = np.pad(train_img_std, ((0,0),(0,1),(4,5),(0,0)), 'constant')
        val_img_std_padded = np.pad(val_img_std, ((0,0),(0,1),(4,5),(0,0)), 'constant')
        test_img_std_padded = np.pad(test_img_std, ((0,0),(0,1),(4,5),(0,0)), 'constant')

    ##################### model training and testing ########################
    
    ### x_test, y_test contain the original dataframes

    for lag in range(14):
        if lag == 0:
            if args.model=='CNN':
                model = tf.keras.models.load_model(args.sourcepath_model)
                model.summary()

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)

                model.fit(train_img_std, y_train_img, validation_data=(val_img_std,y_val_img),
                            callbacks=[monitor],epochs=100)
                t = model.predict(test_img_std)

            if args.model=='Unet':
                model = tf.keras.models.load_model(args.sourcepath_model)
                model.summary()

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)

                model.fit(train_img_std_padded, y_train_img, validation_data=(val_img_std_padded,y_val_img),
                            callbacks=[monitor],epochs=100)

                t = model.predict(test_img_std_padded)

            ### t contains the predictions for each day and area

            ### Brier score for each area and save results ### 
            for i in range(len(areas)):
                y_test_img_curr = y_test_img[2466*i:2466*(i+1),:,:,:]
                t_curr = t[2466*i:2466*(i+1),:,:,:]
                print(f'All zeros Brier score {areas[i]}: {brier_score_loss(y_test_img_curr.reshape(-1,1), np.zeros(len(y_test_img_curr.reshape(-1,1))))}')
                print(f'Model Brier score 1: {brier_score_loss(y_test_img_curr.reshape(-1,1), t_curr.reshape(-1,1))}')  
                print(t_curr.reshape(-1,1).shape)

                if args.save=='yes':
                    x_test[i]['predictions_lag0'] = t_curr.reshape(-1,1)
                    x_test[i].to_csv(args.savepath_prediction+areas[i]+'.csv')

        else:
            if args.model=='CNN':
                model = tf.keras.models.load_model(args.sourcepath_model)
                model.summary()

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)

                model.fit(train_img_std[:-lag], y_train_img[lag:], validation_data=(val_img_std[:-lag],y_val_img[lag:]),
                            callbacks=[monitor],epochs=100)

                t = model.predict(test_img_std)

            if args.model=='Unet':
                model = tf.keras.models.load_model(args.sourcepath_model)
                model.summary()

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                                            verbose=1, mode='auto', restore_best_weights=True)

                model.fit(train_img_std_padded[:-lag], y_train_img[lag:], validation_data=(val_img_std_padded[:-lag],y_val_img[lag:]),
                            callbacks=[monitor],epochs=100)

                t = model.predict(test_img_std_padded)

            ### Brier score for each area and save results ### 
            for i in range(len(areas)):
                y_test_img_curr = y_test_img[2466*i+lag:2466*(i+1),:,:,:]
                t_curr = t[2466*i:2466*(i+1)-lag,:,:,:] 
                print(f'All zeros Brier score {areas[i]}: {brier_score_loss(y_test_img_curr.reshape(-1,1), np.zeros(len(y_test_img_curr.reshape(-1,1))))}')
                print(f'Model Brier score 1: {brier_score_loss(y_test_img_curr.reshape(-1,1), t_curr.reshape(-1,1))}')  

                z = np.zeros((lag,13,29,1))
                if args.save=='yes':
                    x_test[i]['predictions_lag'+str(lag)] = np.concatenate((t_curr,z)).reshape(-1,1)
                    x_test[i].to_csv(args.savepath_prediction+areas[i]+'.csv')
        
        model.save(args.savepath_model+str(lag)+'_model.keras')
        keras.backend.clear_session()

