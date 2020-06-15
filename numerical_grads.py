import numpy as np
import copy
from K_NN_funcs import L2_cost , cross_entropy , cross_entropy_prime , relu
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import sys

""" Test function for numerical gradient computation """

def testGrads(X, Y, y, layerDims, lamda, h, init_func, nBatch=None, mu=0, sig=0.01, fast=True, debug=False, eta=1e-5, burnIn=10,normalize=False,alpha=0.9):
    # Compares results of analytically computed
    # gradients to numerical approximations to
    # ensure correctness.
    #
    # sampleSize is used for dimensionality reduction (how many samples to use)
    # dataDim is used for dimensionality reduction (how many dimensions to consider)

    k , N = np.shape(Y)
    d = np.shape(X)[0]
    
    if nBatch == None:
        nBatch = N
    
    print("Building anNet : \n")

    anNet = Network(normalize=normalize,alpha=alpha)
    anNet.build_layers(d,k,layerDims)
    
    print("\nBurning in %d steps...\n" % burnIn)
    
    anNet.nBatch = nBatch
    anNet.eta = [eta,eta]
    anNet.lamda = lamda
    for i in range(burnIn-1):
        print("Burn in step ", (i+1))
        anNet.forward_prop(X,debug=False)
        anNet.backward_prop(Y, anNet.eta[0])

    
    anNet.compute_loss(Y)
    anNet.compute_cost()
    
    print("Burn in done")

    anNet.eta = [0,0]

    print("Copying net for test...")

    test_net = copy.deepcopy(anNet)

    print("Computing grads using %s algorithm...\n" % ("fast but inaccurate" if fast else "slow but accurate"))
    numGrads , numNet = computeGradsNum(test_net,X,Y,y,h,fast)
    print("Numerical grads computed!\n")
    anNet.forward_prop(X,debug=False)
    anNet.backward_prop(Y, anNet.eta[0])

    anGrads = {"W":{},"b":{}}
    
    if normalize:
        anGrads["beta"] = {}
        anGrads["gamma"] = {}

        for i,layer in enumerate(anNet.layers):
            if type(layer) == FCLayer:
                anGrads["W"][i] = layer.gradW
                anGrads["b"][i] = layer.gradb
                anGrads["beta"][i] = layer.gradBeta
                anGrads["gamma"][i] = layer.gradGamma

        relErrs = {"W":{},"b":{},"beta":{},"gamma":{}}
    else:
        
        for i,layer in enumerate(anNet.layers):
            if type(layer) == FCLayer:
                anGrads["W"][i] = layer.gradW
                anGrads["b"][i] = layer.gradb

        relErrs = {"W":{},"b":{}}


    for par in anGrads.keys():
        for lay in anGrads[par].keys():
            err = relErr(anGrads[par][lay],numGrads[par][lay])
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

    return relErrs , anGrads , numGrads , anNet , numNet



def computeGradsNum(net, X, Y, y, h=1e-6, fast=False):
        """ Uses finite or centered difference approx depending on fast boolean 
            Good practice: let net burn in a few steps before computing grads"""
        
        numGrads = {"W":{},"b":{}}

        if fast:
            print("Using finite difference method")
            approx_func = lambda el,layer_idx,el_idx_i,el_idx_j,c : finite_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,c,h)
        else:
            print("Using centered difference method")
            approx_func = lambda el,layer_idx,el_idx_i,el_idx_j,c : centered_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,h)
        
        c = net.cost["Training"][-1]


        # Compute grads for all b
        biases = net.get_biases()

        for k,lay_idx in enumerate(biases.keys()):
            numGrads["b"][2*k] = np.zeros(biases[lay_idx].shape)
            #grads_b.append(np.zeros(biases[lay_idx].shape))
            
            for i in range(numGrads["b"][2*k].shape[0]):
                grad_approx = approx_func("b",lay_idx,i,0,c)
                numGrads["b"][2*k][i,0] = grad_approx

            net.layers[lay_idx].gradb = numGrads["b"][2*k]
        

        # Compute grads for all W
        weights = net.get_weights()

        for k,lay_idx in enumerate(weights.keys()):
            numGrads["W"][2*k] = np.zeros(weights[lay_idx].shape)
            #grads_W.append(np.zeros(weights[lay_idx].shape))

            for i in range(numGrads["W"][2*k].shape[0]):
                for j in range(numGrads["W"][2*k].shape[1]):
                    grad_approx = approx_func("W", lay_idx, i, j, c)
                    numGrads["W"][2*k][i,j] = grad_approx
        
            net.layers[lay_idx].gradW = numGrads["W"][2*k]

        if net.normalize:
            numGrads["gamma"] = {}
            numGrads["beta"] = {}

            # Compute grads for all gamma
            gammas = net.get_gammas()

            for k,lay_idx in enumerate(gammas.keys()):
                numGrads["gamma"][2*k] = np.zeros(gammas[lay_idx].shape)
                #grads_gamma.append(np.zeros(gammas[lay_idx].shape))
                
                for i in range(numGrads["gamma"][2*k].shape[0]):
                    grad_approx = approx_func("gamma",lay_idx,i,0,c)
                    numGrads["gamma"][2*k][i,0] = grad_approx

                net.layers[lay_idx].gradGamma = numGrads["gamma"][2*k]

            # Compute grads for all beta
            betas = net.get_betas()

            for k,lay_idx in enumerate(betas.keys()):
                numGrads["beta"][2*k] = np.zeros(betas[lay_idx].shape)
                for i in range(numGrads["beta"][2*k].shape[0]):
                    grad_approx = approx_func("beta",lay_idx,i,0,c)
                    numGrads["beta"][2*k][i,0] = grad_approx

                net.layers[lay_idx].gradBeta = numGrads["beta"][2*k]            
        
        return numGrads , net



def finite_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,c,h):

    update_el(net,el,layer_idx,el_idx_i,el_idx_j,h)

    net.forward_prop(X,prediction=False)
    net.compute_loss(Y)

    net.compute_cost()
    
    c2 = net.cost["Training"][-1]

    grad_approx = (c2-c) / h

    #reset entries for next pass
    update_el(net,el,layer_idx,el_idx_i,el_idx_j,-h)

    return grad_approx


def centered_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,h):
    
    update_el(net,el,layer_idx,el_idx_i,el_idx_j,-h)
        
    net.forward_prop(X,prediction=False)
    net.compute_loss(Y)
    net.compute_cost()

    c1 = net.cost["Training"][-1]

    update_el(net,el,layer_idx,el_idx_i,el_idx_j,2*h)
    
    net.forward_prop(X,prediction=False)
    net.compute_loss(Y)
    net.compute_cost()

    c2 = net.cost["Training"][-1]

    grad_approx = (c2 - c1) / (2 * h)

    #reset entries for next pass
    update_el(net,el,layer_idx,el_idx_i,el_idx_j,-h)

    return grad_approx

def update_el(net,el,layer_idx,el_idx_i,el_idx_j,h):
    if el == "b":
        net.layers[layer_idx].b[el_idx_i,el_idx_j] += h
    elif el == "W":
        net.layers[layer_idx].W[el_idx_i,el_idx_j] += h
    elif el == "gamma":
        net.layers[layer_idx].gamma[el_idx_i,el_idx_j] += h
    elif el == "beta":
        net.layers[layer_idx].beta[el_idx_i,el_idx_j] += h
    else:
        pass

def reset_entries(net,el,layer_idx,el_idx_i,el_idx_j,h):
    if el == "b":
        net.layers[layer_idx].b[el_idx_i,el_idx_j] -= h
    elif el == "W":
        net.layers[layer_idx].W[el_idx_i,el_idx_j] -= h
    elif el == "gamma":
        net.layers[layer_idx].gamma[el_idx_i,el_idx_j] -= h
    elif el == "beta":
        net.layers[layer_idx].beta[el_idx_i,el_idx_j] -= h
    else:
        pass

def relErr(Wan,Wnum,eps=1e-7):
    # Computes mean relative error between Jacobians Wan and Wnum
    
    if np.shape(Wan) != np.shape(Wnum):
        print("Wan and Wnum have different dimensions!")
        print("Wan: %s != Wnum: %s" % (np.shape(Wan),np.shape(Wnum)))
    
    relErr = np.mean(np.abs(Wan - Wnum)/np.maximum(np.abs(Wan),np.abs(Wnum)))
    #relErr = np.mean(np.abs(Wan - Wnum)/np.maximum(eps,(np.abs(Wan) + np.abs(Wnum))))
    return relErr

