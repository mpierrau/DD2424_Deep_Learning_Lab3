import numpy as np
import copy
from K_NN_funcs import L2_cost , cross_entropy , cross_entropy_prime , relu
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer

""" Test function for numerical gradient computation """

def testGrads(X, Y, y, layerDims, lamda, h, init_func, nBatch=None, mu=0, sig=0.01, fast=True, debug=False, eta=1e-5, burnIn=10):
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

    anNet = Network()
    anNet.build_layers(d,k,layerDims,relu,init_func)
    anNet.set_cost(L2_cost)
    anNet.set_loss(cross_entropy, cross_entropy_prime)

    print("Burning in %d steps..." % burnIn)
    
    anNet.nBatch = nBatch
    anNet.eta = [eta,eta]
    anNet.lamda = lamda
    anNet.n_cycles = burnIn-1
    anNet.n_s = .5

    for _ in range(burnIn-1):
        anNet.forward_prop(X)
        anNet.backward_prop(Y, anNet.eta[0])

    print("number of steps after burn in: ", len(anNet.loss["Training"]))

    anNet.compute_loss(Y)
    anNet.compute_cost()

    print("number of steps after computing loss and loss: ", len(anNet.loss["Training"]))
    
    
    #anNet.fit([X,X],[Y,Y],[y,y],burnIn-1,.5,nBatch,[eta,eta],lamda,1,seed=1337)

    #print("number of steps taken in burn in of anNet before copy: ", len(anNet.loss["Training"]))
    test_net = copy.deepcopy(anNet)

    print("number of steps after copy in: ", len(anNet.loss["Training"]))

    print("Computing grads using %s algorithm..." % ("fast but inaccurate" if fast else "slow but accurate"))
    
    print("anNet.P before computeGradsNum ", anNet.P["Training"])

    W_grad_num , b_grad_num , numNet = computeGradsNum(test_net,X,Y,y,h,fast)

    print("number of steps taken after numerical approx: ", len(anNet.loss["Training"]))
    print("anNet.P after computeGradsNum ", anNet.P["Training"])

    anNet.forward_prop(X)
    anNet.backward_prop(Y, anNet.eta[0])

    print("anNet.P after final comp ", anNet.P["Training"])
    print("number of steps after final comp: ", len(anNet.loss["Training"]))
    # Final grad computation

    #anNet.fit([X,X],[Y,Y],[y,y],1,.5,nBatch,[eta,eta],lamda,1,seed=1337)

    gradWList = []
    gradbList = []
    
    for layer in anNet.layers:
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
    Wgrads = [gradWList, W_grad_num]
    bgrads = [gradbList, b_grad_num]
    
    
    print("Largest relative error: %e for grad[%s]." % (np.max(errs),np.argmax(errs)))
    if np.max(errs) > 1e-6: print("Large errors in gradient!")

    return errs , Wgrads , bgrads , anNet , numNet


def computeGradsNum(net, X, Y, y, h=1e-6, fast=False):
        """ Uses finite or centered difference approx depending on fast boolean 
            Good practice: let net burn in a few steps before computing grads"""

        grads_W = list()
        grads_b = list()
        
        if fast:
            print("Using finite difference method")
            approx_func = lambda el,layer_idx,el_idx_i,el_idx_j,c : finite_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,c,h)
        else:
            print("Using centered difference method")
            approx_func = lambda el,layer_idx,el_idx_i,el_idx_j,c : centered_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,h)
        
        c = net.cost["Training"][-1]

        print("Initial c in computeGradsNum : ", c)
        # Compute grads for all b

        biases = net.get_biases()

        for k,lay_idx in enumerate(biases.keys()):
            grads_b.append(np.zeros(biases[lay_idx].shape))
            
            for i in range(grads_b[k].shape[0]):
                grad_approx = approx_func("b",lay_idx,i,0,c)
                grads_b[k][i,0] = grad_approx

            net.layers[lay_idx].gradb.append(grads_b[k])
        
        # Compute grads for all W

        weights = net.get_weights()

        for k,lay_idx in enumerate(weights.keys()):
            grads_W.append(np.zeros(weights[lay_idx].shape))

            for i in range(grads_W[k].shape[0]):
                for j in range(grads_W[k].shape[1]):
                    grad_approx = approx_func("W", lay_idx, i, j, c)
                    grads_W[k][i,j] = grad_approx
        
            net.layers[lay_idx].gradW.append(grads_W[k])


        return grads_W, grads_b , net

def relErr(Wan,Wnum,eps=1e-10):
    # Computes mean relative error between Jacobians Wan and Wnum
    #return np.mean(np.abs(Wan - Wnum)/np.maximum(np.abs(Wan),np.abs(Wnum)))
    relErr = np.mean(np.abs(Wan - Wnum)/np.maximum(eps,(np.abs(Wan) + np.abs(Wnum))))
    return relErr

def maxErr(Wan,Wnum,eps=1e-10):
    # Computes mean absolute error between Jacobians Wan and Wnum
    return np.mean(np.abs(Wan-Wnum))


def finite_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,c,h):

    def update_el(el,layer_idx,el_idx_i,el_idx_j,h):
        #print("Adding h = ", h)
        if el == "b":
            net.layers[layer_idx].b[el_idx_i,el_idx_j] += h
        else:
            net.layers[layer_idx].W[el_idx_i,el_idx_j] += h

    update_el(el,layer_idx,el_idx_i,el_idx_j,h)

    net.forward_prop(X)
    net.compute_loss(Y)

    net.compute_cost()
    
    c2 = net.cost["Training"][-1]

    grad_approx = (c2-c) / h

    #reset entries for next pass
    update_el(el,layer_idx,el_idx_i,el_idx_j,-h)

    return grad_approx


def centered_diff(net,X,Y,el,layer_idx,el_idx_i,el_idx_j,h):
    
    def update_el(el,layer_idx,el_idx_i,el_idx_j,h):
        if el == "b":
            net.layers[layer_idx].b[el_idx_i,el_idx_j] += h
        else:
            net.layers[layer_idx].W[el_idx_i,el_idx_j] += h

    update_el(el,layer_idx,el_idx_i,el_idx_j,-h)

    net.forward_prop(X)
    net.compute_loss(Y)
    net.compute_cost()
    
    c1 = net.cost["Training"][-1]

    update_el(el,layer_idx,el_idx_i,el_idx_j,2*h)
    
    net.forward_prop(X)
    net.compute_loss(Y)
    net.compute_cost()

    c2 = net.cost["Training"][-1]

    grad_approx = (c2 - c1) / (2 * h)

    #reset entries for next pass
    update_el(el,layer_idx,el_idx_i,el_idx_j,-h)

    return grad_approx