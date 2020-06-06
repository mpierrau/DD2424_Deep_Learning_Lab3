""" Trying to replicate results from lab 2 """

import numpy as np
from data_handling import loadPreProcData , loadBatch
from numerical_grads import testGrads , relErr
from K_NN_funcs import reduceDims , relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import matplotlib.pyplot as plt

X , Y , y = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

d , N = np.shape(X[0])
k = np.shape(Y[0])[0]

redDim = 10
redN = 5

X, Y = reduceDims(X[0], Y[0], redDim, redN)


m = 50
lamda = 0
mu = 0

d = redDim
N = redN

nBatch = N
batchSize = int(d/nBatch)
cycles = 2
n_s = 100
#n_s = int(2*np.floor(N/nBatch))
eta = [1e-5, 1e-3]
net = Network()

net.add_layer(FCLayer(input_size=m,output_size=d,mu=0))
net.add_layer(ActLayer(relu))
net.add_layer(FCLayer(input_size=k,output_size=m,mu=0))

net.set_loss(cross_entropy, cross_entropy_prime)
net.set_cost(L2_cost)

net.fit(X,Y,y[0],nBatch,cycles,n_s,eta,10,lamda)

plt.plot(np.arange(len(net.cost)),net.cost)
plt.show()

net.accuracy