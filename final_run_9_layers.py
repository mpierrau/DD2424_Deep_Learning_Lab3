""" Training 9-layer network with best lambda for 3 cycles to see accuracy """


import numpy as np
from data_handling import loadAllFiles2 , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
import matplotlib.pyplot as plt
import plot_funcs

Xin , Yin , yin = loadAllFiles2(valSize=5000)
test_data = loadTestFiles()

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

dims = [50,30,20,20,10,10,10,10]

# Parameters

cycles = 2
nBatch = 100
n_s = 5*N/nBatch
eta_min = 1e-5
eta_max = 1e-1
lamda = .005
rec = 5

net = Network(normalize=True)
net.build_layers(d,k,dims)

net.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],lamda=lamda,recPerEp=rec)

net.compute_accuracy(test_data[0],test_data[2],key="Test")
net.accuracy["Test"]

#reload(plot_funcs)
plot_funcs.plotAcc(net,["Training","Validation"],"9_layers_BN_acc.png","Imgs/9_Layers/",size=(10,5))
plot_funcs.plotCost(net,["Training","Validation"],"9_layers_BN_cost.png","Imgs/9_Layers/",size=(10,5))



# CHECKING IF ALPHA = 0.5 GIVES BETTER RESULTS
alphaNet = Network(normalize=True,alpha=0.9)
alphaNet.build_layers(d,k,dims)

alphaNet.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],lamda=lamda,recPerEp=rec)

alphaNet.compute_accuracy(test_data[0],test_data[2],key="Test")
alphaNet.accuracy["Test"]

#reload(plot_funcs)
plot_funcs.plotAcc(alphaNet,["Training","Validation"],"9_layers_BN_alpha_.5_acc.png","Imgs/9_Layers/",size=(10,5))
plot_funcs.plotCost(alphaNet,["Training","Validation"],"9_layers_BN_alpha_.5_cost.png","Imgs/9_Layers/",size=(10,5))