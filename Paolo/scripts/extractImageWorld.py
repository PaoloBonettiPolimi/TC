#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import argparse

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

def extract_images_new(df, n_filters, verbose=False):
    times = df.time.unique()
    number_of_img, rows, cols = len(times), len(df.latitude.unique()), len(df.longitude.unique())
    images = np.zeros( (number_of_img, rows, cols, n_filters) )
    
    df = df.set_index(['time','latitude','longitude'], drop=True)
    df.sort_index(level=['time','latitude', 'longitude'], ascending=[1,0,1], inplace=True)
    k=0
    
    for day in range(0,number_of_img):
        
        images[k,:,:,:] = df.loc(axis=0)[times[day]].values.reshape(rows,cols,n_filters)
        if (k%100==0) & (verbose==True): print(k)
        k += 1
    return images
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_repetitions", default=5, type=int)
    parser.add_argument("--results_train", default='/data/tropical/train_world_img.pkl')
    parser.add_argument("--results_val", default='/data/tropical/val_world_img.pkl')
    parser.add_argument("--results_test", default='/data/tropical/test_world_img.pkl')

    parser.add_argument("--train_path", default='/data/tropical/training_world.csv')
    parser.add_argument("--val_path", default='/data/tropical/training_world.csv')
    parser.add_argument("--test_path", default='/data/tropical/training_world.csv')

    args = parser.parse_args()
    print(args)

    ##################### load csv ########################

    train = pd.read_csv(train_path)
    print('train loaded')
    val = pd.read_csv(val_path)
    print('val loaded')
    test = pd.read_csv(test_path)
    print('test loaded')

    ##################### shear ########################

    train['shear'] = np.sqrt((train['u_200']-train['u_850'])**2 + (train['v_200']-train['v_850'])**2)
    print('train shear computed')
    val['shear'] = np.sqrt((val['u_200']-val['u_850'])**2 + (val['v_200']-val['v_850'])**2)
    print('val shear computed')
    test['shear'] = np.sqrt((test['u_200']-test['u_850'])**2 + (test['v_200']-test['v_850'])**2)
    print('test shear computed')

    ##################### standardization ########################

    cols_to_std = [ 'vo', 'r', 'u_200', 'u_850', 'v_200','v_850', 'ttr','sst','shear']
    scaler = StandardScaler()

    train[cols_to_std] = scaler.fit_transform(train[[ 'vo', 'r', 'u_200', 'u_850', 'v_200','v_850', 'ttr','sst','shear']])
    print('train standardized')
    val[cols_to_std] = scaler.transform(val[[ 'vo', 'r', 'u_200', 'u_850', 'v_200','v_850', 'ttr','sst','shear']])
    print('val standardized')
    test[cols_to_std] = scaler.transform(test[[ 'vo', 'r', 'u_200', 'u_850', 'v_200','v_850', 'ttr','sst','shear']])
    print('test standardized')

    ##################### extract images and save ########################

    variables = ['time', 'latitude', 'longitude', 'vo', 'r', 'u_200', 'u_850', 'v_200', 'v_850', 'ttr', 'sst', 'shear']
    new_train_img =  extract_images_new(train.loc[:,variables], 9, verbose=True)
    with open(args.results_train, 'wb') as f:  
        pickle.dump(new_train_img, f)

    new_val_img =  extract_images_new(val.loc[:,variables], 9, verbose=True)
    with open(args.results_val, 'wb') as f:  
        pickle.dump(new_val_img, f)

    new_test_img =  extract_images_new(test.loc[:,variables], 9, verbose=True)
    with open(args.results_test, 'wb') as f:  
        pickle.dump(new_test_img, f)
    
    

