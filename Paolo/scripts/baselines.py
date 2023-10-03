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

def dataLoad_target():
    #train = pd.read_csv('/Users/paolo/Documents/TC_old/data/NewData_csv/training_sets_withrealtom.csv')
    #val = pd.read_csv('/Users/paolo/Documents/TC_old/data/NewData_csv/validation_sets_withrealtom.csv')
    #test = pd.read_csv('/Users/paolo/Documents/TC_old/data/NewData_csv/test_sets_withrealtom.csv')  

    #train = pd.read_csv('/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/target_train_dailymeans_Atlantic.csv')
    #val = pd.read_csv('/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/target_val_dailymeans_Atlantic.csv')
    #test = pd.read_csv('/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/target_test_dailymeans_Atlantic.csv')  

    train = pd.read_csv('/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/train_dailymeans_both_target.csv')
    val = pd.read_csv('/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/val_dailymeans_both_target.csv')
    test = pd.read_csv('/Users/paolo/Documents/TC/Paolo/Notebooks_newTarget/GRIB extraction/test_dailymeans_both_target.csv')  

    y_train = train.apply(lambda x: 1 if x.new_target>=50 else 0,axis=1)
    print (y_train)
    y_test = test.loc[test.time>='2016-04-01'].apply(lambda x: 1 if x.new_target>=50 else 0,axis=1)
    print(y_test)
    y_val = val.apply(lambda x: 1 if x.new_target>=50 else 0,axis=1)
    print(y_val)

    return y_train, y_val, y_test

def dataLoad_features(train_path, val_path, test_path, var_list):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    train['shear'] = train.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)
    val['shear'] = val.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)
    test['shear'] = test.apply(lambda x: np.sqrt((x.u_200-x.u_850)**2 + (x.v_200-x.v_850)**2),axis=1)

    x_train = train[var_list]
    print (x_train)
    x_test = test.loc[test.time>='2016-04-01'][var_list]
    print (x_test)
    x_val = val[var_list]
    print(x_val)

    return x_train, x_val, x_test

def preprocess_data(x_train, x_val, x_test, y_train, y_val, y_test):
    trainScaler = StandardScaler()

    x_train_scaled = trainScaler.fit_transform(x_train)
    x_val_scaled = trainScaler.transform(x_val)
    x_test_scaled = trainScaler.transform(x_test)

    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features_path", default='/Users/paolo/Documents/TC_old/data/NewData_csv/training_sets_withrealtom.csv')
    parser.add_argument("--val_features_path", default='/Users/paolo/Documents/TC_old/data/NewData_csv/validation_sets_withrealtom.csv')
    parser.add_argument("--test_features_path", default='/Users/paolo/Documents/TC_old/data/NewData_csv/test_sets_withrealtom.csv')
    parser.add_argument("--features_list", default=[ 'vo', 'r', 'u_200', 'u_850', 'v_200','v_850', 'tcwv','sst','shear','latitude','longitude'])
    parser.add_argument("--model", default='extratree')
    parser.add_argument("--save", default='no')
    parser.add_argument("--savepath", default='trial.csv')

    args = parser.parse_args()
    print(args)

    ##################### load and standardize data ########################

    x_train, x_val, x_test = dataLoad_features(train_path=args.train_features_path,val_path=args.val_features_path,test_path=args.test_features_path, var_list=args.features_list)
    y_train, y_val, y_test = dataLoad_target()

    x_train_scaled,x_val_scaled,x_test_scaled,y_train,y_val,y_test = preprocess_data(x_train,x_val,x_test,y_train,y_val,y_test)

    ##################### model training and testing ########################
    
    test = pd.read_csv(args.test_features_path)
    test = test.loc[test.time>='2016-04-01']

    for lag in range(14):
        if lag == 0:
            if args.model=='extratree':
                clf = ExtraTreesClassifier(n_estimators=200, random_state=0, min_samples_leaf=1000, max_features='sqrt',n_jobs=-1,verbose=3)
                clf.fit(np.concatenate((x_train_scaled,x_val_scaled),axis=0), np.concatenate((y_train,y_val),axis=0))

            if args.model=='adaboost':
                base = DecisionTreeClassifier(splitter='random', min_samples_split=1000, random_state=0)
                clf = AdaBoostClassifier(base_estimator=base, n_estimators=50, random_state=0, learning_rate=1)
                clf.fit(np.concatenate((x_train_scaled,x_val_scaled),axis=0), np.concatenate((y_train,y_val),axis=0))

            if args.model=='FFNN':
                model_std_relu_dropout = Sequential()
                model_std_relu_dropout.add(Dense(512, input_dim=x_train_scaled.shape[1], activation='relu'))
                model_std_relu_dropout.add(Dropout(0.5))
                model_std_relu_dropout.add(Dense(1,activation='sigmoid')) # Output
                model_std_relu_dropout.compile(loss='binary_crossentropy', optimizer='adam')

                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, 
                        verbose=1, mode='auto', restore_best_weights=True)

                model_std_relu_dropout.summary()

                model_std_relu_dropout.fit(x_train_scaled,y_train,validation_data=(x_val_scaled,y_val),
                    callbacks=[monitor],epochs=50, batch_size=512)

            ### confusion matrices ###
            if args.model=='FFNN':
                t = model_std_relu_dropout.predict(x_test_scaled)
            else:
                t = clf.predict_proba(x_test_scaled)
                t = t[:,1]

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
            print(f'All zeros Brier score: {brier_score_loss(y_test, np.zeros(len(y_test)))}')
            print(f'Model Brier score: {brier_score_loss(y_test, t)}')  

            ### save results ###
            if args.save=='yes':
                test['predictions_lag0'] = t
                test.to_csv(args.savepath)

        else:
            if args.model=='extratree':
                clf = ExtraTreesClassifier(n_estimators=200, random_state=0, min_samples_leaf=1000, max_features='sqrt',n_jobs=-1,verbose=3)
                clf.fit(np.concatenate((x_train_scaled[:-lag*377],x_val_scaled[:-lag*377]),axis=0), np.concatenate((y_train[lag*377:],y_val[lag*377:]),axis=0))

            if args.model=='adaboost':
                    base = DecisionTreeClassifier(splitter='random', min_samples_split=1000, random_state=0)
                    clf = AdaBoostClassifier(base_estimator=base, n_estimators=50, random_state=0, learning_rate=1)
                    clf.fit(np.concatenate((x_train_scaled[:-lag*377],x_val_scaled[:-lag*377]),axis=0), np.concatenate((y_train[lag*377:],y_val[lag*377:]),axis=0))

            if args.model=='FFNN':
                model_std_relu_dropout = Sequential()
                model_std_relu_dropout.add(Dense(512, input_dim=x_train_scaled.shape[1], activation='relu'))
                model_std_relu_dropout.add(Dropout(0.5))
                model_std_relu_dropout.add(Dense(1,activation='sigmoid')) # Output
                model_std_relu_dropout.compile(loss='binary_crossentropy', optimizer='adam')
                
                monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, 
                        verbose=1, mode='auto', restore_best_weights=True)

                model_std_relu_dropout.summary()
                model_std_relu_dropout.fit(x_train_scaled[:-lag*377], y_train[lag*377:], validation_data=(x_val_scaled[:-lag*377],y_val[lag*377:]),
                    callbacks=[monitor],epochs=50,batch_size=512)
                

            ### confusion matrices ###
            if args.model=='FFNN':
                t = model_std_relu_dropout.predict(x_test_scaled[:-lag*377])
            else:
                t = clf.predict_proba(x_test_scaled[:-lag*377])
                t = t[:,1]

            ranges = [0.025,0.05,0.075,0.1]
            
            for j in ranges:
                classes = []
                for i in t:
                    if i<=j: classes.append(0)
                    else: classes.append(1)
            
                #ConfusionMatrixDisplay(confusion_matrix(y_test[lag*377:], classes)).plot(colorbar=False,cmap=plt.cm.Blues, values_format='d')
                #ConfusionMatrixDisplay(confusion_matrix(y_test[lag*377:], classes, normalize='true')).plot(colorbar=False,cmap=plt.cm.Blues)
            
            ### ROC curve ###
            #plot_roc(t,y_test[lag*377:])
            
            ### Calibration curve ###
            #display = CalibrationDisplay.from_predictions(y_test[lag*377:], t, n_bins=10)

            ### Brier score (w.r.t. class) ###
            print(f'All zeros Brier score: {brier_score_loss(y_test[lag*377:], np.zeros(len(y_test[lag*377:])))}')
            print(f'Model Brier score: {brier_score_loss(y_test[lag*377:], t)}')

            ### save results ###
            if args.save=='yes':
                z = np.zeros((lag*377,1))
                tt = np.concatenate((t.reshape(-1,1),z))
                test['predictions_lag'+str(lag)] = tt.reshape(-1,1)
                test.to_csv(args.savepath)
            

    

