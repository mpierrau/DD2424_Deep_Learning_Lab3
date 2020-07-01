""" Training 2-layer network with best lambda for 3 cycles to see accuracy of best lambda """

import numpy as np
from data_handling import loadAllFiles , loadTestFiles , reduceDims
from K_NN_network_class import Network
import plot_funcs 
from numerical_grads import testGrads

Xin , Yin , yin = loadAllFiles(valSize=5000)    # Loads 50 000 images from CIFAR-10 as training and sets aside 5000 for validation.
test_data = loadTestFiles()                     # Loads 10 000 test images.

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

""" Testing gradients """

# Set parameters

lamda = 0       # Using lambda != 0 makes gradients for W to disagree between numerical and analytical.
dims = [50,50]

# It is good practice to reduce dimensionality and samplesize when testing gradients to avoid issues with numerical precision.
dRed = 20
NRed = 100

testNet = Network(normalize=True)
testNet.build_layers(dRed,k,dims,lamda)

Xgrad, Ygrad, ygrad = reduceDims(Xin[0], Yin[0], yin[0], dRed, NRed)

errs , anGrads , numGrads = testGrads(testNet, Xgrad, Ygrad, ygrad, h=1e-6, fast=False)
# gradients of b are close to zero when normalize=True and relative errors are therefore acting up.

""" Train the network """

dims = [50,50]
lamda = 2.28e-3
cycles = 3
nBatch = 100
ns = 5*int(N/nBatch)
eta_min = 1e-5
eta_max = 1e-1

rec = 50

net = Network(normalize=True)
net.build_layers(d, k, dims, lamda)

net.fit(Xin, Yin, yin, nCycles = cycles, ns = ns, nBatch = nBatch, eta = [eta_min, eta_max], recEvery = rec)

net.compute_accuracy(test_data[0], test_data[2], key = "Test")
net.accuracy["Test"]


plot_funcs.plotAcc(net, ["Training", "Validation"], "3_layers_50_50_best_lambda_acc.png", "Imgs/3_Layers/")
plot_funcs.plotCost(net, ["Training", "Validation"], "3_layers_50_50_best_lambda_cost.png", "Imgs/3_Layers/")

# 51.85 % accuracy with BN using lambda=2.28e-3 for 3 cycles with standard settings
