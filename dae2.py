import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Activation, Dropout
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
import sys

#list of commandline arguments
dataset_name = sys.argv[1] #'gpcr_admat_dgc.txt'
encoding_dim = int(sys.argv[3]) # 2000
noise_factor = float(sys.argv[2]) #0.1
dropout_rate = 0.0
epochs=100
batch_size=10
verbose=0

#creates directory
def createDirectory(directory):
    try:
        os.makedirs(directory)
	#print('directory created successfully')
    except OSError as e:
        if e.errno != errno.EEXIST:
	    raise

start_time = time.time()

#get dataset initial
directory_name = dataset_name.split('_')[0]
directory_name = directory_name+'_'+sys.argv[2]+'_'+sys.argv[3] #filter(lambda e: e in range(9), str(noise_factor))
#shutil.rmtree(directory_name)
createDirectory(directory_name)

temp = pd.read_table(dataset_name)
output_dim = temp.shape[1]-1

#input_img = pd.read_table(dataset_name)
input_img = np.loadtxt(dataset_name,'\t',skiprows=1, usecols=range(1,output_dim))
print('processing ',dataset_name,' of shape ',input_img.shape)
input_tf = input_img

#input = input_img.drop(input_img.columns[0], axis=1).values
#input = input_img.values
#input_tf = tf.Variable(input, dtype='float32')



#input_tf_noisy = input_tf + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_tf.shape) 


#input_tf_noisy = np.clip(input_tf_noisy, 0., 1.)




model = Sequential()
model.add(Dropout(dropout_rate, input_shape=(output_dim-1,)))
model.add(Dense(encoding_dim, activation='relu', name='encoder'))#, input_dim=output_dim-1))
model.add(Dense(output_dim-1, activation='relu', name='decoder'))
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

kfold = KFold(n_splits=5, shuffle=True, random_state=4242)
cvscores = []
aucs = []
fold_idx = 1 
for train,test in kfold.split(input_tf):
    #print(input_tf[test].sum())
    os.chdir(directory_name)
    fold_name = str(fold_idx)
    createDirectory(fold_name)
    os.chdir(fold_name)
    input_train = np.zeros(input_tf.shape)
    input_test = np.zeros(input_tf.shape)
    input_train = input_tf[train]
    input_test = input_tf[test]
    input_train_noisy = input_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_train.shape)
    input_test_noisy = input_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_test.shape)  

    model.fit(input_train_noisy, input_train_noisy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    scores = model.evaluate(input_test_noisy, input_test_noisy, verbose=verbose)
    decoded_img = model.predict(input_test_noisy)
    decoded_flattened = decoded_img.flatten()
    truth_flattened = input_test.flatten()
    np.savetxt('prediction_test.out', decoded_flattened, delimiter=',')
    np.savetxt('truth_test.out', truth_flattened, delimiter=',')
    os.chdir('..')
    precision, recall, thresholds = precision_recall_curve(truth_flattened, decoded_flattened)
    area = metrics.auc(recall, precision)
    print('fold index = ', fold_idx)
    print('area = ', area)
    cvscores.append(scores[1]*100)
    aucs.append(area * 100)
    fold_idx=fold_idx+1
    os.chdir('..')
'''for train,test in kfold.split(input_tf):
    #print(train)
    model.fit(input_tf[train], input_tf[train], epochs=100, batch_size=10, verbose=0)
    scores = model.evaluate(input_tf[test], input_tf[test], verbose=0)
    decoded_img = model.predict(input_tf[test])
    decoded_flattened = decoded_img.flatten()
    truth_flattened = input_tf[test].flatten()
    precision, recall, thresholds = precision_recall_curve(truth_flattened, decoded_flattened)
    area = metrics.auc(recall, precision)
    print('fold index = ', fold_idx)
    print('area = ', area)
    cvscores.append(scores[1]*100)
    aucs.append(area * 100)
    fold_idx=fold_idx+1
'''
print(np.mean(cvscores),np.std(cvscores),np.mean(aucs))
print(cvscores)
print('Execution time :', (time.time() - start_time), ' seconds')
os.chdir('..')
