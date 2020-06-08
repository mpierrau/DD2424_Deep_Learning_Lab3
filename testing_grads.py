import numpy as np
from data_handling import loadPreProcData , reduceDims
from numerical_grads import testGrads , relErr

X1 , Y1 , y1 = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

redDim = 20
redN = 2

X , Y , y = reduceDims(X1[0],Y1[0],y1[0],redDim,redN)

dims = [30,20,10]
lamda = 0

errs, Wgrads, bgrads, anNet, numNet = testGrads(X,Y,layerDims=dims,
                        lamda=lamda,h=1e-6,
                        fast=False,burnIn=10)

errs