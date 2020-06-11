import numpy as np
import copy
from K_NN_funcs import L2_cost , cross_entropy , cross_entropy_prime , relu
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer

""" Test function for numerical gradient computation """

def testGrads(X, Y, y, layerDims, lamda, h, init_func, nBatch=None, mu=0, sig=0.01, fast=True, debug=False, eta=1e-5, burnIn=10,normalize=False):
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

    anNet = Network(normalize=normalize)
    anNet.build_layers(d,k,layerDims)
    
    print("Burning in %d steps..." % burnIn)
    
    anNet.nBatch = nBatch
    anNet.eta = [eta,eta]
    anNet.lamda = lamda

    for _ in range(burnIn-1):
        anNet.forward_prop(X)
        anNet.backward_prop(Y, anNet.eta[0])

    print("number of steps after burn in: ", len(anNet.loss["Training"]))

    anNet.compute_loss(Y)
    anNet.compute_cost()

    test_net = copy.deepcopy(anNet)

    print("Computing grads using %s algorithm..." % ("fast but inaccurate" if fast else "slow but accurate"))

    numGrads , numNet = computeGradsNum(test_net,X,Y,y,h,fast,normalize)

    anNet.forward_prop(X)
    anNet.backward_prop(Y, anNet.eta[0])

    grad_names = ["W", "b"]
    Wgrads = []
    bgrads = []

    if normalize==True:
        betagrads = []
        gammagrads = []
        grad_names.append("beta")
        grad_names.append("gamma")

        for layer in anNet.layers:
            if type(layer) == FCLayer:
                Wgrads.append(layer.gradW)
                bgrads.append(layer.gradb)
                betagrads.append(layer.gradBeta)
                gammagrads.append(layer.gradGamma)
        
        anGrads = [Wgrads,bgrads,betagrads,gammagrads]

    else:
        
        for layer in anNet.layers:
            if type(layer) == FCLayer:
                print("Adding grads from layer %d" % layer.layerIdx)
                Wgrads.append(layer.gradW)
                bgrads.append(layer.gradb)
        
        anGrads = [Wgrads,bgrads]

    
    relErrs = []

    for i , (anPar , numPar) in enumerate(zip(anGrads,numGrads)):
        print("Parameter : %s" % grad_names[i])
        for j , (anGrad , numGrad) in enumerate(zip(anPar , numPar)):
            print("Layer : ", j)
            err = relErr(anGrad,numGrad)
            print("Relative error : ", err)
            relErrs.append(err)
    
    
    print("\nLargest relative error: %e" % (np.max(relErrs)))
    maxerror = np.max(relErrs)
    if maxerror > 1e-2:
        print("Probably issue with gradient")
    elif maxerror > 1e-4:
        print("You should feel uncomfortable...")
    elif maxerror > 1e-7:
        print("Ok for objectives with kinks (non-linear functions)")
    else:
        print("You should be happy!")

    return relErrs , anGrads , numGrads , anNet , numNet


def computeGradsNum(net, X, Y, y, h=1e-6, fast=False, normalize=False):
        """ Uses finite or centered difference approx depending on fast boolean 
            Good practice: let net burn in a few steps before computing grads"""

        grads_W = []
        grads_b = []
        
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
            grads_b.append(np.zeros(biases[lay_idx].shape))
            
            for i in range(grads_b[k].shape[0]):
                grad_approx = approx_func("b",lay_idx,i,0,c)
                grads_b[k][i,0] = grad_approx

            net.layers[lay_idx].gradb = grads_b[k]
        

        # Compute grads for all W
        weights = net.get_weights()

        for k,lay_idx in enumerate(weights.keys()):
            grads_W.append(np.zeros(weights[lay_idx].shape))

            for i in range(grads_W[k].shape[0]):
                for j in range(grads_W[k].shape[1]):
                    grad_approx = approx_func("W", lay_idx, i, j, c)
                    grads_W[k][i,j] = grad_approx
        
            net.layers[lay_idx].gradW = grads_W[k]

        grads = [grads_W , grads_b]

        if normalize:
            grads_gamma = []
            grads_beta = []

            # Compute grads for all gamma
            gammas = net.get_gammas()

            for k,lay_idx in enumerate(gammas.keys()):
                grads_gamma.append(np.zeros(gammas[lay_idx].shape))
                
                for i in range(grads_gamma[k].shape[0]):
                    grad_approx = approx_func("gamma",lay_idx,i,0,c)
                    grads_gamma[k][i,0] = grad_approx

                net.layers[lay_idx].gradGamma = grads_gamma[k]

            grads.append(grads_gamma)

            # Compute grads for all beta
            betas = net.get_betas()

            for k,lay_idx in enumerate(betas.keys()):
                grads_beta.append(np.zeros(betas[lay_idx].shape))
                
                for i in range(grads_beta[k].shape[0]):
                    grad_approx = approx_func("beta",lay_idx,i,0,c)
                    grads_beta[k][i,0] = grad_approx

                net.layers[lay_idx].gradBeta = grads_beta[k]

            grads.append(grads_beta)
            
        
        return grads , net



def finite_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,c,h):

    update_el(net,el,layer_idx,el_idx_i,el_idx_j,h)

    net.forward_prop(X)
    net.compute_loss(Y)

    net.compute_cost()
    
    c2 = net.cost["Training"][-1]

    grad_approx = (c2-c) / h

    #reset entries for next pass
    update_el(net,el,layer_idx,el_idx_i,el_idx_j,-h)

    return grad_approx


def centered_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,h):
    
    update_el(net,el,layer_idx,el_idx_i,el_idx_j,-h)

    net.forward_prop(X)
    net.compute_loss(Y)
    net.compute_cost()
    
    c1 = net.cost["Training"][-1]

    update_el(net,el,layer_idx,el_idx_i,el_idx_j,2*h)
    
    net.forward_prop(X)
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

def relErr(Wan,Wnum,eps=1e-10):
    # Computes mean relative error between Jacobians Wan and Wnum
    #return np.mean(np.abs(Wan - Wnum)/np.maximum(np.abs(Wan),np.abs(Wnum)))
    relErr = np.mean(np.abs(Wan - Wnum)/np.maximum(eps,(np.abs(Wan) + np.abs(Wnum))))
    return relErr