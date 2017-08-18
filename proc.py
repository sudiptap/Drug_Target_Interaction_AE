import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

data = sys.argv[1]
noises = range(2,42,2)
noises = map(lambda e: float(e)/100, noises)
noises = [0.01]

dims = range(1000, 3100, 100)

for noise in noises:  # for range noise
#for dim in dims:       # for range dim

  scores = []
  for i in range(5):

    fileprefix = data+'_'+str(noise)+'/'+str(i+1)+'/'  # for range noise
    #fileprefix = data+'_0.1_'+str(dim)+'/'+str(i+1)+'/' # for range dims

    y_true = np.loadtxt(fileprefix+'truth_test.out')
    y_scores = np.loadtxt(fileprefix+'prediction_test.out')
    score = roc_auc_score(y_true, y_scores)
    scores.append(score)
  print np.mean(scores)
