import numpy as np
from data_handling import loadPreProcData , reduceDims
from numerical_grads import testGrads , relErr
from K_NN_funcs import he_init , regular_init

X1 , Y1 , y1 = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

redDim = 20
redN = 5

X , Y , y = reduceDims(X1[0],Y1[0],y1[0],redDim,redN)

dims = [30]
lamda = 0

errs, anGrads, numGrads, anNet, numNet = testGrads(X,Y,y,layerDims=dims,
                                                lamda=lamda,h=1e-6,init_func = regular_init,
                                                fast=False,burnIn=10,normalize=True)

errs