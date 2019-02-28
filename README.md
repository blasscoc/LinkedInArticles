
### Deep Learning - Strength through diversity

Previously we messed around with using a deep learning LSTM method applied to the facies classification competition data. We made a big deal about how much better it was compared to the base-line support vector classifier. We all like to nerd out once in a while, but this isn't a very productive frame of mind. 

Fundamentally these two learners are looking at the problem in different ways, so let's see if we can make them work together, and improve our model. We do this by augmenting our feature set with the predictions made by each classifier, iterating over "blind" wells. We then train a subsequent LSTM to use this augmented feature set for prediction. This process is called model stacking, http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/, and it generally happend that stacked or otherwise ensembled classifiers out-perform.

The results showed improved performance on the SHANKLE well we showed last time, both in terms of f1-score and qualitatively. The stacked classifier doesn't confuse the Wackestone (WS) with the Marine silt-stone and shales (SiSh) in the shallower section. And it does a much better job resolving Dolomite (D) from the (SiSh) deeper. 

Access this notebook on https://blasscoc.github.io/LinkedInArticles/

Other datasets and solutions to the problem can be found here:
https://agilescientific.com/blog/2016/12/8/seg-machine-learning-contest-theres-still-time


```python
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

import numpy as np

from sklearn.metrics import classification_report

# Convenience method from the seminal work of https://github.com/brendonhall
from competition_facies_plots import (make_facies_log_plot, 
                                      compare_facies_plot, 
                                      facies_colors)

from utils import (load_data, chunk, setup_svc, setup_lstm, 
                   train_predict_lstm, train_predict_svc,
                   WELL_NAMES)
```

    /Users/blasscock/.edm/envs/DeepLearning36/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


We've pushed much of the code from the notebooks to a utils library. Cleans things up.

Here we will train/predict using the SVC/LSTM. The result "all_wells" will have a PredictionSVC and a set of associate class probabilities. There's no flow of information here, each prediction is made "blind", we will use this futher down as a feature for a stack classifier.


```python
all_wells = load_data()

train_predict_svc(all_wells,
                  wvars = ['GR', 'DeltaPHI', 'PE', 'PHIND',
                           'ILD_log10', 'NM_M']);

train_predict_lstm(all_wells,
                  wvars = ['GR', 'DeltaPHI', 'PE', 'PHIND',
                           'ILD_log10', 'NM_M'],                 
                  win=7,
                  batch_size = 128,
                  dropout=0.1,
                  max_epochs=80,
                  num_hidden=100,
                  num_classes=11);


```

    CHURCHMAN BIBLE
    CROSS H CATTLE
    LUKE G U
    NEWBY
    NOLAN
    Recruit F9
    SHRIMPLIN
    SHANKLE
    CHURCHMAN BIBLE
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

    CROSS H CATTLE
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

    LUKE G U
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

    NEWBY
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

    NOLAN
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

    Recruit F9
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

    SHRIMPLIN
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

    SHANKLE
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']



We've created a new "feature" out of our prediction iterating over blind wells. There's no flow of information at this point *that's important, even standardizing colllectively will be an issue*. Now train a new LSTM to use this as a new feature, the result combines the predictive power of the SVC/LSTM together, they get to be friends. The training has to be done carefully to prevent any flow of information, once again iterate over blind wells to validate the performance.


```python
proba_svc = ["PredictionSVC-%d" %i for i in range(11)]
proba_lstm = ["PredictionLSTM-%d" %i for i in range(11)]
svars = ['PredictionSVC', 'PredictionLSTM'] 
_wvars = ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']

# play around here, for now just use the prediction
all_vars = _wvars + svars

train_predict_lstm(all_wells,
                   name="PredictionStacked",
                   wvars = all_vars,                 
                   win=7,
                   batch_size = 128,
                   dropout=0.1,
                   max_epochs=80,
                   num_hidden=100,
                   num_classes=11);
```

    CHURCHMAN BIBLE
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']

    CROSS H CATTLE
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']

    LUKE G U
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']

    NEWBY
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']

    NOLAN
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']

    Recruit F9
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']

    SHRIMPLIN
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']

    SHANKLE
    Variables used :  ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M', 'PredictionSVC', 'PredictionLSTM']




```python
prediction_svc = np.hstack(
    [all_wells[well]['PredictionSVC'].values for well in WELL_NAMES])
prediction_lstm = np.hstack(
    [all_wells[well]['PredictionLSTM'].values for well in WELL_NAMES])
prediction_stacked = np.hstack(
    [all_wells[well]['PredictionStacked'].values for well in WELL_NAMES])

facies = np.hstack([
    all_wells[well]['Facies'].values for well in WELL_NAMES])

print ("\n\nHold-one/Predict Cross-Validation performance\n\n")
print(classification_report(prediction_svc, facies))
print(classification_report(prediction_lstm, facies))
print(classification_report(prediction_stacked, facies))

print ("\n\nHold-one/Predict SHANKLE-Well example\n\n")
print(classification_report(all_wells['SHANKLE']['PredictionSVC'], 
                            all_wells['SHANKLE']['Facies']))
print(classification_report(all_wells['SHANKLE']['PredictionLSTM'], 
                            all_wells['SHANKLE']['Facies']))
print(classification_report(all_wells['SHANKLE']['PredictionStacked'], 
                            all_wells['SHANKLE']['Facies']))
```

    
    
    Hold-one/Predict Cross-Validation performance
    
    
                  precision    recall  f1-score   support
    
               1       0.08      0.17      0.11       121
               2       0.58      0.48      0.52       892
               3       0.50      0.48      0.49       635
               4       0.40      0.43      0.42       171
               5       0.04      0.06      0.05       150
               6       0.40      0.41      0.40       453
               7       0.15      0.23      0.18        65
               8       0.52      0.45      0.48       570
               9       0.73      0.67      0.70       175
    
       micro avg       0.44      0.44      0.44      3232
       macro avg       0.38      0.38      0.37      3232
    weighted avg       0.47      0.44      0.45      3232
    
                  precision    recall  f1-score   support
    
             1.0       0.28      0.74      0.41        99
             2.0       0.72      0.54      0.62       987
             3.0       0.54      0.62      0.58       538
             4.0       0.39      0.38      0.38       188
             5.0       0.01      0.18      0.02        11
             6.0       0.56      0.43      0.49       593
             7.0       0.43      0.61      0.50        69
             8.0       0.60      0.48      0.53       619
             9.0       0.40      0.51      0.45       128
    
       micro avg       0.52      0.52      0.52      3232
       macro avg       0.44      0.50      0.44      3232
    weighted avg       0.58      0.52      0.54      3232
    
                  precision    recall  f1-score   support
    
             1.0       0.36      0.46      0.40       198
             2.0       0.67      0.51      0.58       968
             3.0       0.46      0.61      0.52       459
             4.0       0.36      0.40      0.38       168
             5.0       0.00      0.00      0.00         0
             6.0       0.58      0.45      0.51       602
             7.0       0.46      0.47      0.46        96
             8.0       0.56      0.43      0.49       649
             9.0       0.24      0.41      0.30        92
    
       micro avg       0.48      0.48      0.48      3232
       macro avg       0.41      0.42      0.40      3232
    weighted avg       0.55      0.48      0.51      3232
    
    
    
    Hold-one/Predict SHANKLE-Well example
    
    
                  precision    recall  f1-score   support
    
               1       0.04      0.33      0.08        12
               2       0.79      0.36      0.49       194
               3       0.51      0.68      0.59        88
               4       0.14      0.06      0.08        18
               5       0.00      0.00      0.00        24
               6       0.41      0.69      0.51        42
               7       0.06      0.50      0.11         2
               8       0.78      0.45      0.57        69
    
       micro avg       0.44      0.44      0.44       449
       macro avg       0.34      0.38      0.30       449
    weighted avg       0.60      0.44      0.47       449
    
                  precision    recall  f1-score   support
    
             1.0       0.24      0.72      0.36        29
             2.0       0.87      0.36      0.51       213
             3.0       0.42      0.91      0.57        54
             4.0       0.00      0.00      0.00        23
             5.0       0.00      0.00      0.00         0
             6.0       0.68      0.69      0.68        70
             7.0       0.65      1.00      0.79        11
             8.0       0.78      0.63      0.70        49
    
       micro avg       0.53      0.53      0.53       449
       macro avg       0.45      0.54      0.45       449
    weighted avg       0.68      0.53      0.54       449
    
                  precision    recall  f1-score   support
    
             1.0       0.46      0.79      0.58        52
             2.0       0.84      0.38      0.52       197
             3.0       0.36      0.91      0.52        46
             4.0       0.00      0.00      0.00        12
             5.0       0.00      0.00      0.00         0
             6.0       0.73      0.70      0.72        74
             7.0       0.82      0.88      0.85        16
             8.0       0.80      0.62      0.70        52
    
       micro avg       0.57      0.57      0.57       449
       macro avg       0.50      0.53      0.49       449
    weighted avg       0.70      0.57      0.58       449
    


```python
import matplotlib.pylab as plt
compare_facies_plot(all_wells['SHANKLE'], 
                    'PredictionSVC', facies_colors)
plt.show()
compare_facies_plot(all_wells['SHANKLE'], 
                    'PredictionLSTM', facies_colors)
plt.show()
compare_facies_plot(all_wells['SHANKLE'], 
                    'PredictionStacked', facies_colors)
plt.show()
```


![png](output_7_0.png)



![png](output_7_1.png)



![png](output_7_2.png)



### Facies Prediction - Deep Learning

Here we will get you started using deep learning to associate a suite of well-log measurements, with set of lithofacies. To start with we have a set of facies labels made from an interpreter, we train a model and predict to a blind well. Exact details of the problem can be found in the seminal paper by Hall (2016); https://doi.org/10.1190/tle35100906.1. A with problems with a simple point-wise machine learning solution it exhibits high frequency noise in its prediction. Instead of looking at the data point-wise, we want a machine to look at a scene in a broader context. Borrowing from natural language processing and deep learning we show a baseline implementation using a method call LSTM, looking at the broader scene gives commensurate predictive power, but without the unphysical noise. 

If you want to try this out for yourself you can find the datasets and example notebooks here
https://agilescientific.com/blog/2016/12/8/seg-machine-learning-contest-theres-still-time


```python
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

import pandas as pd
import numpy as np

import matplotlib.pylab as plt

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
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Convenience method from the seminal work of https://github.com/brendonhall
from competition_facies_plots import (make_facies_log_plot, 
                                      compare_facies_plot, 
                                      facies_colors)
```

    Using TensorFlow backend.


Find code and datasets at the agile geoscience github repos https://github.com/seg/2016-ml-contest. It's crucial we do our test/train split in a realistic way, here we will hold out a blind-well and train on the remainder of the collection. In later version we will show how to iterate this to predict likely performance in the wild. The data we might use in training needs to be normalized to help the model learn.


```python
all_data = pd.read_csv("training_data.csv")
# stuff not to normalize
_not_keys = ['Facies', 'Formation', 'Well Name']
# stuff to normalize
_norm_keys = ['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 
              'PHIND', 'PE', 'NM_M', 'RELPOS']

all_wells = {}
for key, val in all_data.groupby("Well Name"):
    all_wells[key] = val.copy()
```

Tell us about the scene; load the well-data into short snippets, and associate label with the center of of the scene.


```python
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
```

Organize the test/train split. The LSTM requires it's ground truth as in "Hot-One" format, and we need to make sure to pad the size of the data to be a multiple of the batch size using in training. This let's use take advantage of the "stateful" option, that lets information flow from one epoch of training to the next. Letting LSTM have a better view of the overall scene. 


```python
def _num_pad(size, batch_size):
    return (batch_size - np.mod(size, batch_size))

def setup_svc(blind_well='SHANKLE', 
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

def setup_lstm(batch_size,
               blind_well='SHANKLE', 
               holdout_wells=["STUART", "CRAWFORD"],
               wvars=['GR', 'DeltaPHI', 'PE', 'PHIND', 
                      'ILD_log10', 'NM_M'],
               win=7):
    
    not_blind = all_data[
            all_data['Well Name'] != blind_well].copy()    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(not_blind[_norm_keys])
    
    X_train = []
    y_train = []
    
    for key,val in all_wells.items():
        if key == blind_well or key in holdout_wells:
            continue
        val = val.copy()
        val[_norm_keys] = scaler.transform(
                                    val[_norm_keys])    
            
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
        
        blind[_norm_keys] = scaler.transform(blind[_norm_keys])    

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
```

### Cross-Validation
In a practical application, we will wonder, how well is this method likely to work on unseen wells I might apply it to in the future. To start answering that question, let's treat each well in the training set as "blind" and make a prediction. We can then evaluate the performance of these predictions in terms of standard metrics of precision, recall and f1-score.

The LSTM example is implemented in keras. There are numerous "hyper-parameters" to be calibrated, the number of hidden-neurons in the LSTM (we used 50), the dropout rate (used to discourage overfitting), batch size, etc. These can be decided objectively using the "hold-out one/predict/repeat" method shown below. 

For comparison, the support vector classifier is also applied, with hyper-paramters suggested by https://github.com/brendonhall/facies_classification/blob/master/Facies%20Classification%20-%20SVM.ipynb


```python
blind_wells = ['CHURCHMAN BIBLE', 'CROSS H CATTLE', 'LUKE G U', 
               'NEWBY', 'NOLAN', 'Recruit F9', 'SHRIMPLIN', 
               'SHANKLE']
wvars = ['GR', 'DeltaPHI', 'PE', 'PHIND', 'ILD_log10', 'NM_M']
win = 7
batch_size = 128

# Well-fold X-validation
for blind in blind_wells:  
    print (blind)
    
    X_train, y_train, X_test, y_test = setup_lstm(batch_size, 
                                                  blind_well=blind)
    
    
    # burn-in a little
    lstm_model = Sequential()
    lstm_model.add(LSTM(50,batch_input_shape=(batch_size, 
                                X_train.shape[1], X_train.shape[2]),
                   stateful=True, kernel_initializer=Zeros()))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(11, activation='sigmoid'))
    lstm_model.compile(loss='categorical_crossentropy', 
                       optimizer=RMSprop(), metrics=['accuracy'])    
    history = lstm_model.fit(X_train,y_train,epochs=20, 
                             batch_size=batch_size, 
                        validation_data=(X_test, y_test),verbose=0)
    lstm_model.save_weights("tmp.h5")
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50,batch_input_shape=(batch_size, 
                                X_train.shape[1],X_train.shape[2]),
                   stateful=True))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(11, activation='sigmoid'))
    lstm_model.compile(loss='categorical_crossentropy', 
                       optimizer='adam', metrics=['accuracy'])    
    lstm_model.load_weights("tmp.h5")
    
    history = lstm_model.fit(X_train,y_train,epochs=80, 
                             batch_size=batch_size, verbose=0)
    
    # Save our prediction
    prediction = lstm_model.predict(X_test, batch_size=batch_size)
    all_wells[blind]['PredictionLSTM'] = np.nan
    upper = all_wells[blind].shape[0] - win
    
    all_wells[blind]['PredictionLSTM'][win//2:-win//2] = \
                            np.argmax(prediction[:upper], axis=1)+1

    all_wells[blind]['PredictionLSTM'].fillna(
                                method='bfill', inplace=True)
    all_wells[blind]['PredictionLSTM'].fillna(
                                method='ffill', inplace=True)   
    
    
    X_train, y_train, X_test, y_test = \
                                setup_svc(blind_well=blind)
    svc_model = svm.SVC(C=10, gamma=1)
    svc_model.fit(X_train,y_train)       
    all_wells[blind]['PredictionSVC'] = \
                                svc_model.predict(X_test)
    
#print(classification_report(np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1)))
```

    CHURCHMAN BIBLE


    /Users/blasscock/.edm/envs/learning/lib/python3.6/site-packages/ipykernel_launcher.py:40: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy


    CROSS H CATTLE
    LUKE G U
    NEWBY
    NOLAN
    Recruit F9
    SHRIMPLIN
    SHANKLE



```python
prediction_svc = np.hstack(
    [all_wells[well]['PredictionSVC'].values for well in blind_wells])
prediction_lstm = np.hstack(
    [all_wells[well]['PredictionLSTM'].values for well in blind_wells])
facies = np.hstack([
    all_wells[well]['Facies'].values for well in blind_wells])

print ("\n\nHold-one/Predict Cross-Validation performance\n\n")
print(classification_report(prediction_svc, facies))
print(classification_report(prediction_lstm, facies))

print ("\n\nHold-one/Predict SHANKLE-Well example\n\n")
print(classification_report(all_wells['SHANKLE']['PredictionSVC'], 
                            all_wells['SHANKLE']['Facies']))
print(classification_report(all_wells['SHANKLE']['PredictionLSTM'], 
                            all_wells['SHANKLE']['Facies']))
```

    
    
    Hold-one/Predict Cross-Validation performance
    
    
                 precision    recall  f1-score   support
    
              1       0.08      0.17      0.11       121
              2       0.58      0.48      0.52       892
              3       0.50      0.48      0.49       635
              4       0.40      0.43      0.42       171
              5       0.04      0.06      0.05       150
              6       0.40      0.41      0.40       453
              7       0.15      0.23      0.18        65
              8       0.52      0.45      0.48       570
              9       0.73      0.67      0.70       175
    
    avg / total       0.47      0.44      0.45      3232
    
                 precision    recall  f1-score   support
    
            1.0       0.38      0.56      0.45       176
            2.0       0.66      0.53      0.59       921
            3.0       0.53      0.62      0.57       524
            4.0       0.32      0.38      0.35       152
            5.0       0.00      0.00      0.00         3
            6.0       0.61      0.45      0.52       620
            7.0       0.39      0.47      0.43        80
            8.0       0.62      0.48      0.54       653
            9.0       0.28      0.44      0.34       103
    
    avg / total       0.57      0.51      0.53      3232
    
    
    
    Hold-one/Predict SHANKLE-Well example
    
    
                 precision    recall  f1-score   support
    
              1       0.04      0.33      0.08        12
              2       0.79      0.36      0.49       194
              3       0.51      0.68      0.59        88
              4       0.14      0.06      0.08        18
              5       0.00      0.00      0.00        24
              6       0.41      0.69      0.51        42
              7       0.06      0.50      0.11         2
              8       0.78      0.45      0.57        69
    
    avg / total       0.60      0.44      0.47       449
    
                 precision    recall  f1-score   support
    
            1.0       0.37      0.77      0.50        43
            2.0       0.83      0.38      0.52       197
            3.0       0.43      0.89      0.58        56
            4.0       0.00      0.00      0.00        15
            5.0       0.00      0.00      0.00         0
            6.0       0.79      0.71      0.75        79
            7.0       0.29      1.00      0.45         5
            8.0       0.72      0.54      0.62        54
    
    avg / total       0.68      0.55      0.56       449
    


    /Users/blasscock/.edm/envs/learning/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)



```python
import matplotlib.pylab as plt
compare_facies_plot(all_wells['SHANKLE'], 
                    'PredictionSVC', facies_colors)
plt.show()
compare_facies_plot(all_wells['SHANKLE'], 
                    'PredictionLSTM', facies_colors)
plt.show()
```


![png](output_11_0.png)



![png](output_11_1.png)

