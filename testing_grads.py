import numpy as np
from data_handling import loadPreProcData , reduceDims , loadAllFiles2
import numerical_grads
from K_NN_funcs import he_init , regular_init
from K_NN_network_class import Network
import importlib

#reload(numerical_grads)

X1 , Y1 , y1 = loadAllFiles2(5000)

redDim = 100
redN = 10

Xtrain , Ytrain , ytrain = reduceDims(X1[0],Y1[0],y1[0],redDim,redN)
Xval , Yval , yval = reduceDims(X1[1],Y1[0],y1[0],redDim,5)

Xin = [Xtrain , Xval]
Yin = [Ytrain , Yval]
yin = [ytrain , yval]

dims = [50,30,20,20,10,10,10,10]

net = Network(normalize=True)
net.build_layers(redDim, 10, dims,par_seed=1337)
net.fit(Xin,Yin,yin,n_cycles=2,n_s=100,nBatch=5,eta=[1e-1,1e-5],lamda=.005,recPerEp=1)

errs, anGrads, numGrads, anNet, numNet = numerical_grads.testGrads(Xtrain,Ytrain,ytrain,layerDims=dims,
                                                lamda=.005,h=1e-6,init_func = he_init,
                                                fast=False,burnIn=100,normalize=True,net=None)

errs


# from numerical_grads import relErr
#relErr(anNet.layers[0].gradW,numNet.layers[0].gradW,eps=1e-5)

"""
 errs
{'W': {0: 4.183811342713687e-09, 2: 9.659857872008717e-09, 4: 1.1942781191901658e-08, 6: 1.1844912721148555e-08, 8: 1.7659871814241274e-08, 10: 9.935932723774688e-09, 12: 2.276816930532646e-07, 14: 1.9442746613840773e-08, 16: 3.817790808723261e-09}, 'b': {0: nan, 2: nan, 4: 1.0, 6: nan, 8: 1.0, 10: 1.0, 12: nan, 14: nan, 16: 9.024283692851775e-09}, 'beta': {0: 2.187446251066e-09, 2: 3.9382149728307334e-09, 4: 3.745799487412595e-09, 6: 2.521055940895449e-09, 8: 7.21274574107518e-09, 10: 1.0079434598015203e-08, 12: 3.0145373072441064e-09, 14: 5.714760751783105e-10, 16: 9.026766463244642e-09}, 'gamma': {0: 1.3452217514353782e-08, 2: 1.3721941044593905e-09, 4: 1.3283050590144924e-08, 6: 1.4747124161413685e-08, 8: 1.6904665298909858e-08, 10: 4.9189835997370614e-09, 12: 1.9674724368565837e-08, 14: 1.0082406931354447e-09, 16: 2.760164212322976e-09}}"""