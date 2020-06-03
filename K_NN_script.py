import numpy as np
from data_handling import loadPreProcData , loadBatch
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
from K_NN_funcs import *
from numerical_grads import testGrads

X , Y , y = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

d , N = np.shape(X[0])
k = np.shape(Y[0])[0]
m = 50
lamda = 0
mu = 0

net = Network()
# Mid layers
net.add_layer(FCLayer(m,d,mu=mu,sig=1/np.sqrt(d),lamda=lamda))
net.add_layer(ActLayer(relu))

# Last FC layer
net.add_layer(FCLayer(k,m,mu=mu,sig=1/np.sqrt(m),lamda=lamda))
#net.add_layer(SoftMaxLayer())

# Choose which cost and loss to use
net.set_cost(L2_cost)
net.set_loss(cross_entropy,cross_entropy_prime)

redDim = 20

grads = net.computeGradsNum(X[0],Y[0],0)

testGrads(X[0],Y[0],2,[[m,redDim],[k,m]],0,1e-5,N,redDim,redDim=True)

net.forward_prop(X[0])

net.fit(X[0], Y[0], y[0], nBatch=100, n_cycles=1, n_s=500, eta=[1e-1,1e-5], rec_every=10, lamda=lamda)

# test

net.forward_prop(X[2])
net.compute_accuracy([X[2]],[y[2]])