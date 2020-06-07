import numpy as np
from data_handling import loadPreProcData
from numerical_grads import testGrads , relErr
from K_NN_funcs import reduceDims

X1 , Y1 , y1 = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

redDim = 10
redN = 5
nBatch = 1

X , Y , y = reduceDims(X1[0],Y1[0],y1[0],redDim,redN)

d , N = np.shape(X)
k = np.shape(Y)[0]
m = 50
l = 50
p = 50
lamda = 0
mu = 0


errs, Wgrads, bgrads, anNet, numNet = testGrads(X,Y,layerDims=[[m,redDim],[l,m],[p,l],[k,p]],
                        lamda=lamda,h=1e-5,
                        nBatch=nBatch,fast=False,printAll=False)

errs