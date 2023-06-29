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
from keras.layers import Conv2DTranspose as DeConv

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

from sklearn.utils.class_weight import compute_sample_weight

import pickle
from keras.layers import Conv2DTranspose as DeConv

%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss

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

def Generator(File_address, target_address, year_start, year_end, lag):
    #while True:
    pickle_target = pd.read_pickle(target_address)
    
        #with (open(File_address, "rb")) as openfile: 
        #    while True:
        #        pickle_data.append(pickle.load(openfile))
        #with (open(target_address, "rb")) as openfile: 
        #        pickle_target.append(pickle.load(openfile))
    while True:
        i=0
        for year in range(year_start,year_end+1):
            X = np.load(File_address+str(year)+'/img_'+str(year)+'.npy')
            Y = pickle_target[i+lag:i+X.shape[0]] 
            i += X.shape[0]
            X = X[:-lag,:,:,:-1]
            yield X, Y

def create_model():
    input_layer = Input(shape=(73,144, 9))

    input_layer = (layers.Conv2D(3, (1,1), activation='relu', padding='same'))(input_layer)

    resnet = tf.keras.applications.ResNet50(
        input_shape=(73,144, 3),
        weights='imagenet',
        include_top=False,  
        input_tensor=None,
        pooling=None
        )
    resnet.trainable = False

    res_features = resnet(input_layer)

    conv = DeConv(1, padding="valid", activation="sigmoid", kernel_size=32)(res_features)

    conv = layers.Cropping2D(cropping=((10,11),(3,4)))(conv)


    model = Model(inputs=input_layer, outputs=conv)
        
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam')

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
            verbose=1, mode='auto', restore_best_weights=True)

    return model, monitor

def evaluate_perf(t,test_outputs,day=0):
    one_day_t = t
    one_day_y = test_outputs
    
    ranges = [0.025,0.05,0.075,0.1]
    
    for j in ranges:
        classes = []
        for i in one_day_t.reshape(-1,1):
            if i<=j: classes.append(0)
            else: classes.append(1)
    
    # confusion matrix with threshold on 0.1, otherwise always 0 
        ConfusionMatrixDisplay(confusion_matrix(one_day_y.reshape(-1,1), classes)).plot(colorbar=False,cmap=plt.cm.Blues, values_format='d')
        ConfusionMatrixDisplay(confusion_matrix(one_day_y.reshape(-1,1), classes, normalize='true')).plot(colorbar=False,cmap=plt.cm.Blues)
    
    plot_roc(one_day_t.reshape(-1,1),one_day_y.reshape(-1,1))
    
    display = CalibrationDisplay.from_predictions(one_day_y.reshape(-1,1), one_day_t.reshape(-1,1), n_bins=10)

    print(f'All zeros Brier score: {brier_score_loss(one_day_y.reshape(-1,1), np.zeros(len(one_day_y.reshape(-1,1))))}')
    print(f'Model Brier score: {brier_score_loss(one_day_y.reshape(-1,1), one_day_t.reshape(-1,1))}')
    


if __name__ == "__main__":

    x2016 = np.load('/data/tropical/images_extracted/test_stdwith2015/imgs/2016/img_2016.npy')
    x2016 = x2016[91:]
    x2017 = np.load('/data/tropical/images_extracted/test_stdwith2015/imgs/2017/img_2017.npy')
    x2018 = np.load('/data/tropical/images_extracted/test_stdwith2015/imgs/2018/img_2018.npy')
    x2019 = np.load('/data/tropical/images_extracted/test_stdwith2015/imgs/2019/img_2019.npy')
    x2020 = np.load('/data/tropical/images_extracted/test_stdwith2015/imgs/2020/img_2020.npy')
    x2021 = np.load('/data/tropical/images_extracted/test_stdwith2015/imgs/2021/img_2021.npy')
    x2022 = np.load('/data/tropical/images_extracted/test_stdwith2015/imgs/2022/img_2022.npy')
    test_target = pd.read_pickle('/home/cmcc/TC/Paolo/data/test_target_img.pkl')

    for lag in [2,3,4,5,6,7,8,9,10,11,12,13]:
    
        input_layer = Input(shape=(73,144, 9))

        conv0 = (layers.Conv2D(3, (3,3), activation='relu', padding='same'))(input_layer)
        
        resnet = tf.keras.applications.ResNet50(
            input_shape=(73,144, 3),
            weights='imagenet',
            include_top=False,  
            input_tensor=None,
            pooling=None
            )
        resnet.trainable = False
        
        res_features = resnet(conv0)
        
        conv = DeConv(1, padding="valid", activation="sigmoid", kernel_size=3)(res_features)
        convUp = (layers.UpSampling2D((3,5)))(conv)
        
        convUp = layers.Cropping2D(cropping=((1,1),(3,3)))(convUp)
        
        
        model = Model(inputs=input_layer, outputs=convUp)
            
        model.summary()
        
        model.compile(loss='binary_crossentropy', optimizer='adam')
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, 
                verbose=1, mode='auto', restore_best_weights=True)
        
        train_path = '/data/tropical/images_extracted/imgs/'
        train_target_path = '/home/cmcc/TC/Paolo/data/train_target_img.pkl'
        val_path = '/data/tropical/images_extracted/val/imgs/'
        val_target_path = '/home/cmcc/TC/Paolo/data/val_target_img.pkl'
        
        train_gen = Generator(train_path, train_target_path, 1980, 2010, lag)
        val_gen = Generator(val_path, val_target_path, 2011, 2015, lag)
        
        model.fit(train_gen, validation_data=val_gen,
                callbacks=[monitor],epochs=100, steps_per_epoch=31, validation_steps=5)
        
        t2016 = model.predict(x2016[:,:,:,:-1])
        t2017 = model.predict(x2017[:,:,:,:-1])
        t2018 = model.predict(x2018[:,:,:,:-1])
        t2019 = model.predict(x2019[:,:,:,:-1])
        t2020 = model.predict(x2020[:,:,:,:-1])
        t2021 = model.predict(x2021[:,:,:,:-1])
        t2022 = model.predict(x2022[:-lag,:,:,:-1])
        t = np.concatenate((t2016,t2017,t2018,t2019,t2020,t2021,t2022))
        
        evaluate_perf(t,test_target[lag:],day=0)
        
        test = pd.read_csv('/data/cmcc/test_with_newTarget_predictions_CNNglobal_ResNet.csv')
        z = np.zeros((lag,13,29,1))
        tt = np.concatenate((t,z))
        test['predictions_lag'+str(lag)] = tt.reshape(-1,1)
        test.to_csv("/data/cmcc/test_with_newTarget_predictions_CNNglobal_ResNet.csv")