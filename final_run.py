""" Training 2-layer network with best lambda for 3 cycles to see accuracy of best lambda """


import numpy as np
from data_handling import loadAllFiles2 , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
import matplotlib.pyplot as plt
import plot_funcs 
from imp import importlib

Xin , Yin , yin = loadAllFiles2(valSize=5000)
test_data = loadTestFiles()

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

dims = [50,50]

# Parameters

cycles = 3
nBatch = 100
n_s = 5*N/nBatch
eta_min = 1e-5
eta_max = 1e-1  
lamda = 2.28e-3
rec = 5

net = Network(normalize=True)
net.build_layers(d,k,dims)

net.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],lamda=lamda,recPerEp=rec)

net.compute_accuracy(test_data[0],test_data[2], key="Test")
net.accuracy["Test"]


#importlib.reload(plot_funcs)
plot_funcs.plotAcc(net,["Training","Validation"],"3_layers_50_50_best_lambda_acc.png","Imgs/3_Layers/")
plot_funcs.plotCost(net,["Training","Validation"],"3_layers_50_50_best_lambda_cost.png","Imgs/3_Layers/")

lamda = 3.76e-3

net2 = Network(normalize=True)
net2.build_layers(d,k,dims)

net2.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],lamda=lamda,recPerEp=rec)
net2.compute_accuracy(test_data[0],test_data[2], key="Test")
net2.accuracy["Test"]

reload(plot_funcs)
plot_funcs.plotAcc(net2,["Training","Validation"],"3_layers_50_50_best_lambda2_acc.png","Imgs/3_Layers/")
plot_funcs.plotCost(net2,["Training","Validation"],"3_layers_50_50_best_lambda2_cost.png","Imgs/3_Layers/")

# 51.7 % accuracy with BN using lambda=5e-3 for 3 cycles with standard settings
# 51.85 % accuracy with BN using lambda=2.28e-3 for 3 cycles with standard settings
# 50.88 % accuracy with BN using lambda=3.76e-3 for 3 cycles with standard settings
