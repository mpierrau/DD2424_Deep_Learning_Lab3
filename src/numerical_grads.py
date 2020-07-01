import numpy as np
import copy
from K_NN_funcs import L2_cost , cross_entropy , cross_entropy_prime , relu
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import sys
from tqdm import tqdm , trange

""" Test function for numerical gradient computation """

def testGrads(anNet, X, Y, y, h, fast=True, debug=False, burnIn=10, eta=.005):
    # Compares results of analytically computed
    # gradients to numerical approximations to
    # ensure correctness.
    #
    # sampleSize is used for dimensionality reduction (how many samples to use)
    # dataDim is used for dimensionality reduction (how many dimensions to consider)

    keys = ["W","b","beta","gamma"] if anNet.normalize else ["W","b"]

    for _ in range(burnIn-1):
        anNet.forward_prop(X)
        anNet.backward_prop(Y, eta)
    
    anNet.compute_loss(Y)
    anNet.compute_cost()

    print("Copying net for test...")

    testNet = copy.deepcopy(anNet)

    print("Computing grads using %s algorithm...\n" % ("fast but inaccurate" if fast else "slow but accurate"))
    numGrads = computeGradsNum(testNet, X, Y, y, h, fast)
    print("Numerical grads computed!\n")
    anNet.forward_prop(X)
    anNet.backward_prop(Y, eta)

    
    anGrads = {k : {} for k in keys}

    for i,layer in enumerate(anNet.layers):
        if type(layer) == FCLayer:
            for k in anGrads.keys():
                anGrads[k][i] = layer.grads[k]

    relErrs = {k: {} for k in anGrads.keys()}

    for par in anGrads.keys():
        for lay in anGrads[par].keys():
            err = relErr(anGrads[par][lay], numGrads[par][lay])
            relErrs[par][lay] = err
    
    maxerror = -sys.maxsize - 1

    for key in relErrs.keys():
        for idx in relErrs[key].keys():
            tmpVal = relErrs[key][idx]
            maxerror = tmpVal if tmpVal > maxerror else maxerror
    
    print("\nLargest relative error: %e" % (maxerror))

    if maxerror > 1e-2:
        print("Probably issue with gradient")
    elif maxerror > 1e-4:
        print("You should feel uncomfortable...")
    elif maxerror > 1e-7:
        print("Ok for objectives with kinks (non-linear functions)")
    else:
        print("You should be happy!")

    return relErrs , anGrads , numGrads


def computeGradsNum(net, X, Y, y, h=1e-6, fast=False):
        """ Uses finite or centered difference approx depending on fast boolean 
            Good practice: let net burn in a few steps before computing grads"""

        keys = ["W","b","beta","gamma"] if net.normalize else ["W","b"]

        numGrads = {k : {} for k in keys}

        if fast:
            print("Using finite difference method")
            c = net.cost["Training"][-1]
            approx_func = lambda net,key,layer_idx,el_idx : finite_diff(net,X,Y,key,layer_idx,el_idx,c,h)
        else:
            print("Using centered difference method")
            approx_func = lambda net,key,layer_idx,el_idx : centered_diff(net,X,Y,key,layer_idx,el_idx,h)
        
        layers = net.get_pars("W")["W"].keys()

        for par in tqdm(keys, leave=False):
            for layIdx in tqdm(layers, leave=False):
                numGrads[par][layIdx] = recurseGrads(net,par,layIdx,approx_func)

        return numGrads


def recurseGrads(net,key,layIdx,approx_func):
    """ Dynamic for-loop that uses recursion to dynamically adapt the computation to
        the dimension of the network parameters to be estimated. 
        
        approx_func     = function that approximates gradient of net.pars[key][idx]) 
                          after changing net.pars[key][idx] by some increment h. 
                        - type : function(net, key, idx) -> float """

    nDims = len(np.shape(net.layers[layIdx].pars[key]))
    grad = np.zeros(net.layers[layIdx].pars[key].shape)

    def recFunc(idx,dim):
        thisDim = dim
        thisIdx = idx
        if thisDim < nDims:
            thisIdx.append(0)

            for i in trange(np.shape(grad)[thisDim], leave=False):
                thisIdx[thisDim] = i
                newIdx = thisIdx
                recFunc(newIdx, thisDim + 1)
            
            del thisIdx[thisDim]
        else:
            thisIdx = tuple(thisIdx)
            grad[thisIdx] = approx_func(net, key, layIdx, thisIdx)
    
    recFunc([],0)

    return grad


def finite_diff(net,X,Y,key,layer_idx,el_idx,h,c):

    net.layers[layer_idx].pars[key][el_idx] += h

    net.forward_prop(X,prediction=False)
    net.compute_loss(Y)
    net.compute_cost()
    
    c2 = net.cost["Training"][-1]

    grad_approx = (c2-c) / h

    #reset entries for next pass
    net.layers[layer_idx].pars[key][el_idx] -= h

    return grad_approx


def centered_diff(net,X,Y,key,layer_idx,el_idx,h):
    
    net.layers[layer_idx].pars[key][el_idx] -= h
        
    net.forward_prop(X,prediction=False)
    net.compute_loss(Y)
    net.compute_cost()

    c1 = net.cost["Training"][-1]

    net.layers[layer_idx].pars[key][el_idx] += 2 * h
    
    net.forward_prop(X,prediction=False)
    net.compute_loss(Y)
    net.compute_cost()

    c2 = net.cost["Training"][-1]

    grad_approx = (c2 - c1) / (2 * h)

    #reset entries for next pass
    net.layers[layer_idx].pars[key][el_idx] -= h

    return grad_approx


def relErr(Wan,Wnum,eps=1e-7):
    # Computes mean relative error between Jacobians Wan and Wnum
    assert(Wan.shape == Wnum.shape)
    
    relErr = np.mean(np.abs(Wan - Wnum)/np.maximum(eps,(np.abs(Wan) + np.abs(Wnum))))

    return relErr

