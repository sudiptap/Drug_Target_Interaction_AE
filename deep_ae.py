import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

dataset_name = 'ic_admat_dgc.txt'
encoding_dim = 60
temp = pd.read_table(dataset_name)
output_dim = temp.shape[1]-1

#input_img = pd.read_table(dataset_name)
input_img = np.loadtxt(dataset_name,'\t',skiprows=1, usecols=range(1,output_dim))
print(input_img.shape)
input_tf = input_img
#input = input_img.drop(input_img.columns[0], axis=1).values
#input = input_img.values
#input_tf = tf.Variable(input, dtype='float32')
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=output_dim-1))
model.add(Dense(256, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(output_dim-1, activation='relu', name='decoder'))
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#train, test = train_test_split(input_tf, train_size=0.8)
kfold = KFold(n_splits=5, shuffle=True, random_state=4242)
cvscores = []
aucs = []
#X = 
for train,test in kfold.split(input_tf):
    model.fit(input_tf[train], input_tf[train], epochs=100, batch_size=10, verbose=0)
    scores = model.evaluate(input_tf[test], input_tf[test], verbose=0)
    decoded_img = model.predict(input_tf[test])
    decoded_flattened = decoded_img.flatten()
    truth_flattened = input_tf[test].flatten()
    precision, recall, thresholds = precision_recall_curve(truth_flattened, decoded_flattened)
    area = metrics.auc(recall, precision)
    print('AUPR ==> ',area)
    #print(decoded_flattened.shape)
    #print(truth_flattened.shape)
    #print(decoded_img)
    #print(model.metrics_names[1], scores[1]*100)
    cvscores.append(scores[1]*100)
    aucs.append(area * 100)
print(np.mean(cvscores),np.std(cvscores),np.mean(aucs))




  

