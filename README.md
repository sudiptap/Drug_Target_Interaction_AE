## deepDTI : Deep Autoencoder for Drug-Target Interaction Prediction 
Drug-Target Interaction (DTI) prediction is an active area of research. It has big impact on pharmaceutical research and drug repositioning. Improvement in DTI prediction accuracy could save lot of time, effort and money invested for drug discovery and drug repositioning experiments in wet labs. However, most of the algorithms proposed for DTI prediction suffer from poor accuracy and/or large runtime. Also, most of the algorithms with best results are not easily extended to distributed or parallel settings. In this paper, we propose autoencoder (AE) based approaches to improve prediction accuracy. We also provide a shallow and deep neural network based AE implementation developed using well known deep learning library Keras and Tensorflow. Through extensive experiments we show that our results outperform all competing algorithms both in prediction accuracy and runtime. 
## Pre-requisite
Python, Tensorflow, Keras, Sklearn, numpy, Pandas
## Usage
Please follow the following command to a single file : python dae2.py dataset-name noise-level hidden-layer-dimension
example : python dae2.py 'nr_admat_dgc.txt' 0.1 2048

## Please cite: 
coming soon
