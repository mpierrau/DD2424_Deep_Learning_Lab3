import numpy as np
from data_handling import loadAllFiles2 , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
import matplotlib.pyplot as plt
from plot_funcs import plotAcc , plotCost

Xin , Yin , yin = loadAllFiles2(valSize=5000)
test_data = loadTestFiles()

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

dims = [50,50]

# Parameters

cycles = 2
nBatch = 100
n_s = 5*N/nBatch
eta_min = 1e-5
eta_max = 1e-1
lamda = .005
rec = 5

net1 = Network(normalize=True)
net1.build_layers(d,k,dims,lamda,par_seed=1337)

net1.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],recPerEp=rec,shuffle_seed=1337)

net1.compute_accuracy(test_data[0],test_data[2],key="Test")
net1.accuracy["Test"]
# 53.4 % accuracy
np.max(net1.accuracy["Training"])
# 73 % accuracy
np.min(net1.cost["Validation"])

plotAcc(net1,["Training","Validation"],savename="3_layers_acc_new_BN.png",foldername="Imgs/3_Layers/New_run/",
        title = "Accuracy for %d-layer network with hidden dimensions %s \n with BN" % (len(net1.layer_dims) + 1 , net1.layer_dims),
        size=(7,5))
plotCost(net1,["Training","Validation"],savename="3_layers_cost_new_BN.png",foldername="Imgs/3_Layers/New_run/",
        title = "Cost for %d-layer network with hidden dimensions %s \n with BN" % (len(net1.layer_dims) + 1 , net1.layer_dims),
        size=(7,5))


net2 = Network(normalize=False)
net2.build_layers(d,k,dims,lamda,par_seed=1337)

net2.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],recPerEp=rec,shuffle_seed=1337)

net2.compute_accuracy(test_data[0],test_data[2],key="Test")
net2.accuracy["Test"]
# 46.7 % accuracy
np.max(net2.accuracy["Training"])
# 58 % accuracy

np.min(net2.cost["Training"])

plotAcc(net2,["Training","Validation"],savename="3_layers_acc_new_Non_BN.png",foldername="Imgs/3_Layers/New_run/",
        title = "Accuracy for %d-layer network with hidden dimensions %s \n without BN" % (len(net2.layer_dims) + 1 , net2.layer_dims),
        size=(7,5))
plotCost(net2,["Training","Validation"],savename="3_layers_cost_new_Non_BN.png",foldername="Imgs/3_Layers/New_run/",
        title = "Cost for %d-layer network with hidden dimensions %s \n without BN" % (len(net2.layer_dims) + 1 , net2.layer_dims),
        size=(7,5))
