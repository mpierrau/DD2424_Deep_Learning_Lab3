import numpy as np
from data_handling import loadPreProcData , loadBatch , loadAllFiles , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
import matplotlib.pyplot as plt
from plot_funcs import plotAcc , plotCost

training_data , validation_data = loadAllFiles(valSize=5000)
test_data = loadTestFiles()

Xtrain = training_data[0]
Ytrain = training_data[1]
ytrain = training_data[2]

Xval = validation_data[0]
Yval = validation_data[1]
yval = validation_data[2]

Xtest = test_data[0]
Ytest = test_data[1]
ytest = test_data[2]

Xin = [Xtrain,Xval]
Yin = [Ytrain,Yval]
yin = [ytrain,yval]

d , N = Xtrain.shape
k = np.shape(Ytrain)[0]

dims = [50,50]

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

net.compute_accuracy(Xtest,ytest,key="Test")
net.accuracy["Test"]
# 51.7 % accuracy

plotAcc(net,["Training","Validation"])
plotCost(net,["Training","Validation"])