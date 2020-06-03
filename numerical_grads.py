import numpy as np
import copy
from K_NN_funcs import L2_cost , cross_entropy , cross_entropy_prime , relu
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer

""" Test function for numerical gradient computation """

def testGrads(X, Y, nLayers, layerDims, lamda, h, sampleSize, dataDim, nBatch=100, mu=0, sig=0.01, redDim=False, fast=True, debug=False):
    # Compares results of analytically computed
    # gradients to numerical approximations to
    # ensure correctness.
    #
    # sampleSize is used for dimensionality reduction (how many samples to use)
    # dataDim is used for dimensionality reduction (how many dimensions to consider)

    if redDim:
        N = sampleSize
        d = dataDim
    else:
        N = np.shape(Y)[1]
        d = np.shape(X)[0]

    XbatchRedDim = X[0:d,0:N]
    YbatchRedDim = Y[:,0:N]

    testNet = Network()

    for i in range(nLayers-1):
        testNet.add_layer(FCLayer(layerDims[i][0],layerDims[i][1], mu=mu, sig=sig, lamda=lamda))
        testNet.add_layer(ActLayer(relu))
    
    testNet.add_layer(FCLayer(layerDims[nLayers-1][0],layerDims[nLayers-1][1],mu=mu,sig=sig,lamda=lamda))

    testNet.set_cost(L2_cost)
    testNet.set_loss(cross_entropy, cross_entropy_prime)

    P = testNet.forward_prop(XbatchRedDim)
    testNet.backward_prop(P,0)

    print("Computing grads for dim: %d , N: %d, dimRed=%s" % (d,N,redDim))
    print("Using %s algorithm..." % ("fast but inaccurate" if fast else "slow but accurate"))
    
    if fast:
        W_grad_num , b_grad_num = testNet.computeGradsNum(XbatchRedDim,YbatchRedDim,h,nBatch)
    else:
        W_grad_num , b_grad_num = testNet.computeGradsNumSlow(XbatchRedDim,YbatchRedDim,h,nBatch)

    gradWList = []
    gradbList = []

    for layer in testNet.layers:
        if type(layer) == FCLayer:
            gradWList.append(layer.gradW)
            gradbList.append(layer.gradb)

    relErrsW = []
    relErrsb  = []
    
    for i in range(len(gradWList)):
        relErrsW.append(relErr(gradWList[i],W_grad_num[i]))
    
    for i in range(len(gradbList)):
        relErrsb.append(relErr(gradbList[i],b_grad_num[i]))
    
    errs = [relErrsW, relErrsb]
    grads = [gradWList, gradbList, W_grad_num, b_grad_num]
    
    print("Largest relative error: %e for grad[%s]." % (np.max(errs),np.argmax(errs)))
    if np.max(errs) > 1e-3: print("Large errors in gradient!")

    return errs , grads


def relErr(Wan,Wnum,eps=1e-10):
    # Computes relative error between Jacobians Wan and Wnum
    return np.mean(np.abs(Wan - Wnum)/np.maximum(eps,(np.abs(Wan) + np.abs(Wnum))))

def maxErr(Wan,Wnum,eps=1e-10):
    # Computes absolute error between Jacobians Wan and Wnum
    return np.mean(np.abs(Wan-Wnum))
