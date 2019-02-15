


from numpy.random import seed
seed(42)

import pandas as pd
import numpy as np

# This is fast if you have GPU!
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Input, Embedding, 
                          Dropout, Conv1D, MaxPooling1D, 
                          BatchNormalization)
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop
from keras.initializers import Zeros

from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

WELL_NAMES = ['CHURCHMAN BIBLE', 'CROSS H CATTLE',
              'LUKE G U', 'NEWBY', 'NOLAN', 'Recruit F9',
              'SHRIMPLIN', 'SHANKLE']
NORM_KEYS = ['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE',
             'NM_M', 'RELPOS']
NOT_KEYS = ['Facies', 'Formation', 'Well Name']

def load_data():
    all_data = pd.read_csv("training_data.csv")
    # stuff not to normalize
    all_wells = {}
    for key, val in all_data.groupby("Well Name"):
        all_wells[key] = val.copy()

    return all_wells

def chunk(x, y, num_chunks, size=61, random=True):
    rng = x.shape[0] - size
    if random:
        indx = np.int_(
            np.random.rand(num_chunks) * rng) + size//2
    else:
        indx = np.arange(0,rng,1) + size//2
        
    Xwords = np.array([[x[i-size//2:i+size//2+1,:] 
                                for i in indx]])
    ylabel = np.array([y[i] for i in indx])
    return Xwords[0,...], ylabel


def _num_pad(size, batch_size):
    return (batch_size - np.mod(size, batch_size))

def setup_svc(all_wells,
              blind_well='SHANKLE', 
              holdout_wells=["STUART", "CRAWFORD"],
              wvars=['GR', 'DeltaPHI', 'PE', 'PHIND', 
                     'ILD_log10', 'NM_M'],
              win=7):

    X_train = []
    y_train = []
    for key,val in all_wells.items():
        if key == blind_well or key in holdout_wells:
            continue
        X_train.extend(val[wvars].values)
        y_train.extend(val['Facies'].values)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    _scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = _scaler.transform(X_train)    
    
    if blind_well is not None:
    
        blind = all_wells[blind_well].copy() 

        X_test = blind[wvars].values
        y_test = blind['Facies'].values

        X_test = _scaler.transform(X_test)   
    else:
        X_test = None
        y_test = None
    
    return X_train, y_train, X_test, y_test

def setup_lstm(all_wells,
               batch_size,
               blind_well='SHANKLE', 
               holdout_wells=["STUART", "CRAWFORD"],
               wvars=['GR', 'DeltaPHI', 'PE', 'PHIND', 
                      'ILD_log10', 'NM_M'],
               win=7):

    not_blind = np.vstack([
                   val[wvars].values for key,val in all_wells.items() if
                      key != blind_well])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(not_blind)
    
    X_train = []
    y_train = []
    
    for key,val in all_wells.items():
        if key == blind_well or key in holdout_wells:
            continue
        val = val.copy()
        val[wvars] = scaler.transform(val[wvars])    
            
        _X = val[wvars].values
        _y = val['Facies'].values

        __X, __y = chunk(_X, _y, 400, size=win, 
                         random=False)
        X_train.extend(__X)
        y_train.extend(__y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # hot one encoding
    enc = OneHotEncoder(sparse=False, n_values=11)
    y_train = enc.fit_transform(
                        np.atleast_2d(y_train-1).T)
    X_train = X_train.transpose(0,2,1)

    # pad to batch size    
    num_pad = _num_pad(X_train.shape[0], batch_size)
    X_train = np.pad(X_train, 
                ((0,num_pad),(0,0),(0,0)), mode='edge')
    y_train = np.pad(y_train, 
                ((0,num_pad), (0,0)), mode='edge')
    
    if blind_well is not None:
        blind = all_wells[blind_well].copy()
        
        blind[wvars] = scaler.transform(blind[wvars])    

        _X = blind[wvars].values
        _y = blind['Facies'].values

        X_test, y_test = chunk(_X, _y, 400, size=win, 
                               random=False)

        # hot one encoding
        enc = OneHotEncoder(sparse=False, n_values=11)
        y_test = enc.fit_transform(np.atleast_2d(y_test-1).T)
        X_test = X_test.transpose(0,2,1)
        
        # pad to batch size
        num_pad = _num_pad(X_test.shape[0], batch_size)
        X_test = np.pad(X_test, 
                        ((0,num_pad),(0,0),(0,0)), mode='edge')
        y_test  = np.pad(y_test, 
                         ((0,num_pad), (0,0)), mode='edge')
    else:
        X_test = None
        y_test = None
        
 
    return X_train, y_train, X_test, y_test


def train_predict_lstm(all_wells,
                       blind_wells=WELL_NAMES,
                       name="PredictionLSTM",
                       wvars = ['GR', 'DeltaPHI', 'PE', 'PHIND',
                                'ILD_log10', 'NM_M'],                 
                       win=7,
                       batch_size = 128,
                       dropout=0.1,
                       max_epochs=80,
                       num_hidden=50,
                       num_classes=11):

    # Well-fold X-validation
    for blind in blind_wells:  
    
        X_train, y_train, X_test, y_test = setup_lstm(all_wells,
                                                      batch_size,
                                                      wvars=wvars,
                                                      blind_well=blind)
    

        """
        # burn-in a little
        lstm_model = Sequential()
        lstm_model.add(LSTM(num_hidden,batch_input_shape=(batch_size, 
                        X_train.shape[1], X_train.shape[2]),
                        stateful=True, kernel_initializer=Zeros()))
        lstm_model.add(Dropout(dropout))
        lstm_model.add(Dense(num_classes, activation='sigmoid'))
        lstm_model.compile(loss='categorical_crossentropy', 
                           optimizer=RMSprop(), metrics=['accuracy'])    
        history = lstm_model.fit(X_train,y_train,epochs=20, 
                                 batch_size=batch_size, 
                                 validation_data=(X_test, y_test),verbose=0)
        lstm_model.save_weights("tmp.h5")
        """
        
        lstm_model = Sequential()
        lstm_model.add(LSTM(num_hidden,batch_input_shape=(batch_size, 
                        X_train.shape[1],X_train.shape[2]),
                        stateful=True))
        lstm_model.add(Dropout(dropout))
        lstm_model.add(Dense(num_classes, activation='sigmoid'))
        lstm_model.compile(loss='categorical_crossentropy', 
                           optimizer='adam', metrics=['accuracy'])    
        #lstm_model.load_weights("tmp.h5")
        
        history = lstm_model.fit(X_train,y_train,epochs=max_epochs, 
                                 batch_size=batch_size, verbose=0)
    
        # Save our prediction
        prediction = lstm_model.predict(X_test, batch_size=batch_size)
        all_wells[blind][name] = np.nan
        upper = all_wells[blind].shape[0] - win
    
        all_wells[blind][name][win//2:-win//2] = \
                            np.argmax(prediction[:upper], axis=1)+1

        all_wells[blind][name].fillna(
            method='bfill', inplace=True)
        all_wells[blind][name].fillna(
            method='ffill', inplace=True)   

        for i in range(num_classes):
            label = name + "-%d" % i
            all_wells[blind][label] = np.nan
            
            upper = all_wells[blind].shape[0] - win
        
            all_wells[blind][label][win//2:-win//2] = prediction[:upper,i]
            all_wells[blind][label].fillna(
                method='bfill', inplace=True)
            all_wells[blind][label].fillna(
                method='ffill', inplace=True)   
        
    return all_wells

def train_predict_svc(all_wells,
                       blind_wells=WELL_NAMES,
                       name="PredictionSVC",
                       wvars = ['GR', 'DeltaPHI', 'PE', 'PHIND',
                                    'ILD_log10', 'NM_M'],
                       num_classes=11):

    for blind in blind_wells:  
        print (blind)
    
        X_train, y_train, X_test, y_test = setup_svc(all_wells,
                                                     wvars=wvars,
                                                     blind_well=blind)
        svc_model = svm.SVC(C=10, gamma=1, probability=True)
        svc_model.fit(X_train,y_train)       
        all_wells[blind][name] = svc_model.predict(X_test)
        proba = svc_model.predict_proba(X_test)
        for i in range(num_classes):
            label = name + '-%d' % i
            if(i < proba.shape[1]):
                all_wells[blind][label] = proba[:,i]
            else:
                all_wells[blind][label] = 0.0
       
    return all_wells
