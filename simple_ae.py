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
import time
import os, errno
import shutil

#creates directory
def createDirectory(directory):
    try:
        os.makedirs(directory)
	#print('directory created successfully')
    except OSError as e:
        if e.errno != errno.EEXIST:
	    raise

start_time = time.time()

dataset_name = 'gpcr_admat_dgc.txt'
#get dataset initial
directory_name = dataset_name.split('_')[0]
shutil.rmtree(directory_name)
createDirectory(directory_name)
encoding_dim = 2000
temp = pd.read_table(dataset_name)
output_dim = temp.shape[1]-1

#input_img = pd.read_table(dataset_name)
input_img = np.loadtxt(dataset_name,'\t',skiprows=1, usecols=range(1,output_dim))
print('processing ',dataset_name,' of shape ',input_img.shape)
input_tf = input_img
#input = input_img.drop(input_img.columns[0], axis=1).values
#input = input_img.values
#input_tf = tf.Variable(input, dtype='float32')
model = Sequential()
model.add(Dense(encoding_dim, activation='relu', name='encoder', input_dim=output_dim-1))
model.add(Dense(output_dim-1, activation='relu', name='decoder'))
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#train, test = train_test_split(input_tf, train_size=0.8)
kfold = KFold(n_splits=5, shuffle=True, random_state=4242)
cvscores = []
aucs = []
fold = 1
for train,test in kfold.split(input_tf):
    os.chdir(directory_name)
    fold_name = str(fold)
    createDirectory(fold_name)
    os.chdir(fold_name)
    model.fit(input_tf[train], input_tf[train], epochs=100, batch_size=10, verbose=0)
    scores = model.evaluate(input_tf[test], input_tf[test], verbose=0)
    decoded_img = model.predict(input_tf[test])
    decoded_flattened = decoded_img.flatten()
    truth_flattened = input_tf[test].flatten()
    np.savetxt('prediction_test.out', decoded_flattened, delimiter=',')
    np.savetxt('truth_test.out', truth_flattened, delimiter=',')
    os.chdir('..')
    precision, recall, thresholds = precision_recall_curve(truth_flattened, decoded_flattened)
    area = metrics.auc(recall, precision)
    print('AUPR ==> ',area)
    #print(decoded_flattened.shape)
    #print(truth_flattened.shape)
    #print(decoded_img)
    #print(model.metrics_names[1], scores[1]*100)
    cvscores.append(scores[1]*100)
    aucs.append(area * 100)
    fold=fold+1
    os.chdir('..')
print(np.mean(cvscores),np.std(cvscores),np.mean(aucs))
print('Execution time :', (time.time() - start_time), ' seconds')
os.chdir('..')


  

