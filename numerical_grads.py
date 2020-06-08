import numpy as np
import copy
from K_NN_funcs import L2_cost , cross_entropy , cross_entropy_prime , relu
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer

""" Test function for numerical gradient computation """

def testGrads(X, Y, layerDims, lamda, h, nBatch=None, mu=0, sig=0.01, fast=True, debug=False, eta=1e-5, burnIn=10):
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

    anNet = Network(lamda,eta,1,1,nBatch)
    anNet.build_layers(d,k,layerDims,relu)
    anNet.set_cost(L2_cost)
    anNet.set_loss(cross_entropy, cross_entropy_prime)

    print("Burning in %d steps..." % burnIn)
    for _ in range(burnIn-1):
        anNet.forward_prop(X)
        anNet.backward_prop(Y,eta)

    test_net = anNet.copyNet()

    # Final grad computation
    anNet.forward_prop(X)
    anNet.backward_prop(Y,eta)

    gradWList = []
    gradbList = []

    for layer in anNet.layers:
        if type(layer) == FCLayer:
            gradWList.append(layer.gradW[-1])
            gradbList.append(layer.gradb[-1])

    print("Computing grads using %s algorithm..." % ("fast but inaccurate" if fast else "slow but accurate"))
    
    W_grad_num , b_grad_num , numNet = computeGradsNum(test_net,X,Y,h,fast)

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


def computeGradsNum(net, X, Y, h=1e-6, fast=False):
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
        
        net.forward_prop(X)
        net.compute_loss(Y)
        net.compute_cost()
        c = net.cost["Training"][-1]
        
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

"""
def computeGradsNumSlow(net, X, Y):
    """ """Uses centred difference approx """ """
    
    grads_W = list()
    grads_b = list()

    test_net = Network(net.lamda,net.eta,net.n_cycles,net.n_s,net.nBatch)
    FCidx = []

    for k, layer in enumerate(net.layers):
        if type(layer) == FCLayer:
            tmpW = copy.deepcopy(layer.W)
            tmpb = copy.deepcopy(layer.b)

            test_net.add_layer(FCLayer(layer.nCols, layer.nRows, layer.mu, layer.sig, layer.lamda, tmpW, tmpb))
            FCidx.append(k)
        else:
            test_net.add_layer(layer)
    
    test_net.set_cost(net.cost_func)
    test_net.set_loss(net.loss_func,net.loss_prime_func)

    biases = test_net.get_biases()

    for j in range(len(biases)):
        grads_b.append(np.zeros((len(biases[j]),1)))
        
        for i in range(len(biases[j])):
            test_net.layers[FCidx[j]].b[i] -= h
            
            test_net.forward_prop(X)
            test_net.compute_loss(Y)
            test_net.compute_cost()

            c1 = test_net.cost["Training"][-1]

            test_net.layers[FCidx[j]].b[i] += 2*h
            
            test_net.forward_prop(X)
            test_net.compute_loss(Y)
            test_net.compute_cost()

            c2 = test_net.cost["Training"][-1]
            
            grads_b[j][i][0] = (c2-c1) / (2*h)

            #reset entries for next pass
            test_net.layers[FCidx[j]].b[i] -= h
    
    test_net.layers[FCidx[j]].gradb.append(grads_b[j])

    weights = test_net.get_weights()

    for k in range(len(weights)):
        grads_W.append(np.zeros(np.shape(weights[k])))

        for i in range(np.shape(grads_W[k])[0]):
            for j in range(np.shape(grads_W[k])[1]):
                test_net.layers[FCidx[k]].W[i,j] -= h
                
                test_net.forward_prop(X)
                test_net.compute_loss(Y)
                test_net.compute_cost()

                c1 = test_net.cost["Training"][-1]

                test_net.layers[FCidx[k]].W[i,j] += 2*h
                
                test_net.forward_prop(X)
                test_net.compute_loss(Y)
                test_net.compute_cost()

                c2 = test_net.cost["Training"][-1]

                grads_W[k][i,j] = (c2-c1) / (2*h)
                
                #reset entries for next pass
                test_net.layers[FCidx[k]].W[i,j] -= h

        test_net.layers[FCidx[k]].gradW.append(grads_W[k])

    return grads_W, grads_b , test_net
"""

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

    """
    print("Element before update")
    if el == "b":
        print(net.layers[layer_idx].b[el_idx_i,el_idx_j])
    else:
        print(net.layers[layer_idx].W[el_idx_i,el_idx_j])
   """ 
    update_el(el,layer_idx,el_idx_i,el_idx_j,h)
    """
    print("Element after update")
    if el == "b":
        print(net.layers[layer_idx].b[el_idx_i,el_idx_j])
    else:
        print(net.layers[layer_idx].W[el_idx_i,el_idx_j])
    """
    net.forward_prop(X)
    net.compute_loss(Y)

    net.compute_cost()
    
    c2 = net.cost["Training"][-1]

    grad_approx = (c2-c) / h
    
    #print("Grad approx: ", grad_approx)

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