import numpy as np
from data_handling import loadPreProcData , loadBatch
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
from K_NN_funcs import *
from numerical_grads import testGrads , relErr , maxErr

X , Y , y = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

doRed = True
redDim = 10
redN = 50
"""
if doRed == True:
    X[0] = X[0][:redDim,:redN]
    Y[0] = Y[0][:,:redN]
    y[0] = y[0][:redN]
"""
d , N = np.shape(X[0])
k = np.shape(Y[0])[0]
m = 50
lamda = 0
mu = 0

"""
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


net.forward_prop(X[0])

P1 = net.P[-1]

net.biases[0][0] = 1

net.forward_prop(X[0])

P2 = net.P[-1]

net.compute_loss(Y[0], P1, 10)
net.compute_cost(net.loss,1)
net.compute_loss(Y[0], P2, 10)
"""
"""
grads = net.computeGradsNum(X[0],Y[0],0,nBatch=10)

net.forward_prop(X[0])

net.backward_prop(net.P[-1],1e-5)

Wgrads = []
bgrads = []
for layer in net.layers:
    if type(layer) == FCLayer:
        Wgrads.append(layer.gradW[-1])
        bgrads.append(layer.gradb[-1])

print(relErr())
"""
errs, grads = testGrads(X[0],Y[0],nLayers=2,layerDims=[[m,redDim],[k,m]],
                        lamda=0,h=1e-5,
                        sampleSize=redN,dataDim=redDim,
                        nBatch=10,
                        redDim=True,printAll=False)

"""
for i in np.arange(0,2,1):
    for j in range(len(grads[i])):
        print("Analytical Grads [%d][%d]:" % (i,j))
        print(grads[i][j])
        print("Numerical Grads [%d][%d]:" % (i+2,j))
        print(grads[i + 2][j])
"""
#net.forward_prop(X[0])

#net.fit(X[0], Y[0], y[0], nBatch=100, n_cycles=1, n_s=500, eta=[1e-1,1e-5], rec_every=10, lamda=lamda)

# test

#net.forward_prop(X[2])
#net.compute_accuracy([X[2]],[y[2]])