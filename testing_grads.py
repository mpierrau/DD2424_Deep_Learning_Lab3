import numpy as np
from data_handling import loadPreProcData
from numerical_grads import testGrads , relErr
from K_NN_funcs import reduceDims

X1 , Y1 , y1 = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

redDim = 20
redN = 5
nBatch = 1

X, Y = reduceDims(X1[0],Y1[0],redDim,redN)

d , N = np.shape(X)
k = np.shape(Y)[0]
m = 50
lamda = 0
mu = 0


errs, Wgrads, bgrads, anNet, numNet = testGrads(X,Y,nLayers=2,layerDims=[[m,redDim],[k,m]],
                        lamda=0.1,h=1e-5,
                        nBatch=nBatch,fast=False,printAll=False)

numNet.compute_loss(Y,numNet.P[-1],1)
anNet.loss[-1]

errs

