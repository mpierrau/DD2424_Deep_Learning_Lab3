""" Running 9 layer NN with parameters from lab instructions. Test acc. 41.6% """

import numpy as np
from data_handling import loadPreProcData , loadBatch , loadAllFiles , loadTestFiles , reduceDims
from numerical_grads import testGrads , relErr
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import matplotlib.pyplot as plt
import plot_funcs
import importlib
import numerical_grads

Xin , Yin , yin = loadAllFiles(valSize = 5000)
Xtest , _ , ytest = loadTestFiles()

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

recPerEp = 5
nBatch = 100
cycles = 2
eta = [1e-5, 1e-1]
lamda = .005
ns = 2*int(N/nBatch)

layerDims = [50, 30, 20, 20, 10, 10, 10, 10]

net = Network(normalize=True)
net.build_layers(d,k,layerDims,lamda)

net.fit(Xin,Yin,yin,cycles,ns,nBatch,eta,recPerEp)

plot_funcs.plotAcc(net,["Training","Validation"],"9_layer_example_acc.png","Imgs/9_Layers/")
plot_funcs.plotCost(net,["Training","Validation"],"9_layer_example_cost.png","Imgs/9_Layers/")

net.compute_accuracy(Xtest, ytest, key="Test")
print("Final test accuracy: {0}".format(net.accuracy["Test"]))

# Testing grads after run
numerical_grads.testGrads(Xin[0], Yin[0], yin[0], layerDims = None, lamda=None, h=1e-5, init_func=None, fast=False,normalize=False,alpha=0.5, net=net)