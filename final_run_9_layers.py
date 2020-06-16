""" Training 9-layer network with best lambda for 3 cycles to see accuracy """


import numpy as np
from data_handling import loadAllFiles2 , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
import matplotlib.pyplot as plt
import plot_funcs

foldername = "Imgs/9_Layers/New_run/"

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
rec_every = 100

net91 = Network(normalize=False)
net91.build_layers(d,k,dims,lamda,par_seed=1337)

net91.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],rec_every=rec_every,shuffle_seed=1337,fileName="9_lay_no_BN_res")

net91.compute_accuracy(test_data[0],test_data[2],key="Test")
net91.accuracy["Test"]
np.max(net91.accuracy["Training"])

plot_funcs.plotAcc(net91,["Training","Validation"],"9_layers_acc_new_Non_BN.png",foldername,size=(7,5))
plot_funcs.plotCost(net91,["Training","Validation"],"9_layers_cost_new_Non_BN.png",foldername,size=(7,5))

net92 = Network(normalize=True)
net92.build_layers(d,k,dims,lamda,par_seed=1337)

net92.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],rec_every=rec_every,shuffle_seed=1337,fileName="9_lay_BN_res")

net92.compute_accuracy(test_data[0],test_data[2],key="Test")
net92.accuracy["Test"]
np.max(net92.accuracy["Training"])

plot_funcs.plotAcc(net92,["Training","Validation"],"9_layers_acc_new_BN.png",foldername,size=(7,5))
plot_funcs.plotCost(net92,["Training","Validation"],"9_layers_cost_new_BN.png",foldername,size=(7,5))