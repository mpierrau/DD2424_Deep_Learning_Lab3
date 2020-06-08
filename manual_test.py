""" Trying to replicate results from lab 2 """

import numpy as np
from data_handling import loadPreProcData , loadBatch , loadAllFiles , loadTestFiles , reduceDims
from numerical_grads import testGrads , relErr
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import matplotlib.pyplot as plt

Xin = np.array([[1.,2.],[3.,4.],[5.,6.]])
Yin = np.array([[1.,0.],[0.,1.]])
yin = np.array([0,1])

W1 = np.array([ [1.,-1.,1.],
                [-1.,-1.,1.],
                [1.,1.,1.],
                [1.,-1.,-1.] ])
W2 = np.array([ [0.5,0.5,-0.5,-0.5],
                [-0.5,0.5,0.5,-0.5] ])

b1 = np.array([ [0.],
                [0.],
                [0.],
                [0.]])

b2 = np.array([ [0.],
                [0.] ])


s1 = W1@Xin + b1
h = relu(s1)


d , N = np.shape(Xin)
k = np.shape(Yin)[0]

recPerEp = 10

nLayers = 1
nBatch = 1
cycles = 1
eta = [1e-1, 1e-1]
lamda = 0.1

batchSize = int(d/nBatch)
n_s = 1
#n_s = int(2*np.floor(N/nBatch))

layerDims = [4]

net = Network(lamda,eta,cycles,n_s,nBatch)
net.build_layers(d,k,layerDims,relu,W=[W1,W2],b=[b1,b2])

net.set_loss(cross_entropy, cross_entropy_prime)
net.set_cost(L2_cost)

net.forward_prop(Xin,"Training")
net.backward_prop(Yin,0.01)
net.forward_prop(Xin,"Training")
net.backward_prop(Yin,0.01)

net.compute_accuracy(Xin,yin,"Training")

pars = net.get_pars(getAll=True)