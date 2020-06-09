""" Trying to replicate results from lab 2 """

import numpy as np
from data_handling import loadPreProcData , loadBatch , loadAllFiles , loadTestFiles , reduceDims
from numerical_grads import testGrads , relErr
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost , he_init , xavier_init
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import matplotlib.pyplot as plt

#X , Y , y = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

training_data , validation_data = loadAllFiles(valSize=1000)
test_data = loadTestFiles()

Xin = [training_data[0],validation_data[0]]
Yin = [training_data[1],validation_data[1]]
yin = [training_data[2],validation_data[2]]

Xtest = test_data[0]
ytest = test_data[2]

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

#redDim = d
#redN = N

#Xtrain , Ytrain , ytrain = reduceDims(X[0], Y[0], y[0], redDim, redN)
#Xval , Yval , yval = reduceDims(X[1], Y[1], y[1], redDim, redN)

#Xtest = X[2]
#Ytest = Y[2]
#ytest = y[2]

#Xin = [Xtrain, Xval]
#Yin = [Ytrain, Yval]
#yin = [ytrain, yval]

#d = redDim
#N = redN

recPerEp = 100

nLayers = 1
nBatch = 100
cycles = 1
eta = [1e-5, 1e-1]
lamda = 1.73e-5

batchSize = int(d/nBatch)
n_s = int(2*np.floor(N/nBatch))

layerDims = [50]

net = Network()
net.build_layers(d,k,layerDims,relu,he_init)

net.set_loss(cross_entropy, cross_entropy_prime)
net.set_cost(L2_cost)

net.fit(Xin,Yin,yin,cycles,n_s,nBatch,eta,lamda,recPerEp)

steps = np.linspace(0,2*n_s*cycles,len(net.cost["Training"]))

plt.plot(steps,net.cost["Training"],label="Training")
plt.plot(10*np.arange(len(net.cost["Validation"])),net.cost["Validation"],label="Validation")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()

plt.show()


plt.plot(np.arange(len(net.accuracy["Training"])),net.accuracy["Training"],label="Training")
plt.plot(np.arange(len(net.accuracy["Validation"])),net.accuracy["Validation"],label="Validation")
plt.legend()
plt.show()