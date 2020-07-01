""" Running 9 layer NN with parameters from lab instructions. Test acc. 41.6% """

import numpy as np
from data_handling import loadPreProcData , loadBatch , loadAllFiles2 , loadTestFiles , reduceDims
from numerical_grads import testGrads , relErr
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import matplotlib.pyplot as plt
import plot_funcs
import importlib
#X , Y , y = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

Xin , Yin , yin = loadAllFiles2(valSize=5000)
test_data = loadTestFiles()

Xtest = test_data[0]
ytest = test_data[2]

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

recPerEp = 5

nBatch = 100
cycles = 2
eta = [1e-5, 1e-1]
lamda = .005
#lamda = 1.73e-5
n_s = 5*N/nBatch

#n_s = int(2*np.floor(N/nBatch))

layerDims = [50, 30, 20, 20, 10, 10, 10, 10]

net = Network(normalize=False)
net.build_layers(d,k,layerDims)

net.fit(Xin,Yin,yin,cycles,n_s,nBatch,eta,lamda,recPerEp)

plot_funcs.plotAcc(net,["Training","Validation"],"9_layers_Non_BN_acc.png","Imgs/9_Layers/",size=(10,5))
plot_funcs.plotCost(net,["Training","Validation"],"9_layers_Non_BN_cost.png","Imgs/9_Layers/",size=(10,5))

net.compute_accuracy(Xtest, ytest , key="Test")
print(net.accuracy["Test"])

import numerical_grads
reload(numerical_grads)
# Testing grads after run
numerical_grads.testGrads(Xin[0], Yin[0], yin[0], layerDims = None, lamda=None, h=1e-5, init_func=None, fast=False,normalize=False,alpha=0.5, net=net)