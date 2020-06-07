import numpy as np
import copy
from K_NN_funcs import L2_cost , cross_entropy , cross_entropy_prime , relu
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer

""" Test function for numerical gradient computation """

def testGrads(X, Y, layerDims, lamda, h, nBatch=100, mu=0, sig=0.01, fast=True, debug=False, printAll=True, eta=1e-4):
    # Compares results of analytically computed
    # gradients to numerical approximations to
    # ensure correctness.
    #
    # sampleSize is used for dimensionality reduction (how many samples to use)
    # dataDim is used for dimensionality reduction (how many dimensions to consider)


    N , k = np.shape(Y)
    d = np.shape(X)[0]

    X = X[0:d,0:N]
    Y = Y[:,0:N]

    print("Building anNet : \n")

    anNet = Network(lamda,eta,1,1,nBatch)
    anNet.build_layers(d,k,layerDims,relu)
    anNet.set_cost(L2_cost)
    anNet.set_loss(cross_entropy, cross_entropy_prime)

    print("Forward prop...")
    anNet.forward_prop(X)
    print("Computing loss...")
    anNet.compute_loss(Y, anNet.P,nBatch,"Training")
    print("Computing cost...")
    anNet.compute_cost(anNet.loss["Training"][-1],lamda,"Training")
    print("Backward prop...")
    anNet.backward_prop(Y,eta)

    gradWList = []
    gradbList = []

    for layer in anNet.layers:
        if type(layer) == FCLayer:
            gradWList.append(layer.gradW[-1])
            gradbList.append(layer.gradb[-1])

    print("Computing grads for dim: %d , N: %d" % (d,N))
    print("Using %s algorithm..." % ("fast but inaccurate" if fast else "slow but accurate"))
    
    if fast:
        W_grad_num , b_grad_num , numNet = computeGradsNum(anNet,X,Y,lamda,h,nBatch)
    else:
        W_grad_num , b_grad_num , numNet = computeGradsNumSlow(anNet,X,Y,lamda,h,nBatch)

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

    if printAll:
        print("Errors: ", errs)
        print("Wgrads: ", Wgrads)
        print("bgrads: ", bgrads)

    return errs , Wgrads , bgrads , anNet , numNet

def computeGradsNum(net, X, Y, lamda, h=1e-5, nBatch=100):
        """ Uses finite difference approx """

        grads_W = list()
        grads_b = list()
        print("Copying network...")
        test_net = Network(lamda,1e-4,1,1,nBatch)
        FCidx = []

        for k, layer in enumerate(net.layers):
            if type(layer) == FCLayer:
                print("Creating layer %d : FCLayer" % k)
                tmpW = copy.deepcopy(layer.W)
                tmpb = copy.deepcopy(layer.b)
                test_net.add_layer(FCLayer(layer.nCols, layer.nRows, layer.mu, layer.sig, layer.lamda, tmpW, tmpb))
                FCidx.append(k)
            else:
                print("Creating layer %d : RELU layer" % k)
                test_net.add_layer(layer)

        test_net.set_cost(net.cost_func)
        test_net.set_loss(net.loss_func,net.loss_prime_func)

        print("Computing initial cost:")
        test_net.forward_prop(X)
        test_net.compute_loss(Y, test_net.P, nBatch,"Training")
        test_net.compute_cost(test_net.loss["Training"][-1],lamda,"Training")
        c = test_net.cost["Training"][-1]

        print("Initial cost: %s" % c)

        for j in range(len(test_net.biases)):
            print("computing grad for b[%d]" % j)
            grads_b.append(np.zeros((len(test_net.biases[j]),1)))
            
            for i in range(len(test_net.biases[j])):
                test_net.layers[FCidx[j]].b[i] += h
                test_net.forward_prop(X)
                test_net.compute_loss(Y,test_net.P,nBatch,"Training")
                test_net.compute_cost(test_net.loss["Training"][-1],lamda,"Training")
                c2 = test_net.cost["Training"][-1]

                grads_b[j][i][0] = (c2-c) / (h)
                #reset entries for next pass
                test_net.layers[FCidx[j]].b[i] -= h

            test_net.layers[FCidx[j]].gradb.append(grads_b[j])

        for k in range(len(test_net.weights)):
            print("computing grad for W[%d]" % k)

            grads_W.append(np.zeros(np.shape(test_net.weights[k])))

            for i in range(np.shape(grads_W[k])[0]):
                for j in range(np.shape(grads_W[k])[1]):
                    
                    test_net.layers[FCidx[k]].W[i,j] += h
                    
                    test_net.forward_prop(X)
                    test_net.compute_loss(Y,test_net.P,nBatch,"Training")
                    test_net.compute_cost(test_net.loss["Training"][-1],lamda,"Training")
                    
                    c2 = test_net.cost["Training"][-1]

                    grads_W[k][i,j] = (c2-c) / (h)
                    
                    #reset entries for next pass
                    test_net.layers[FCidx[k]].W[i,j] -= h
        
            test_net.layers[FCidx[k]].gradW.append(grads_W[k])

        return grads_W, grads_b , test_net

def computeGradsNumSlow(net, X, Y, lamda, h=1e-5, nBatch=100):
    """ Uses centred difference approx """
    
    grads_W = list()
    grads_b = list()

    test_net = Network(lamda,1e-4,1,1,nBatch)
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

    for j in range(len(test_net.biases)):
        grads_b.append(np.zeros((len(test_net.biases[j]),1)))
        
        for i in range(len(test_net.biases[j])):
            test_net.layers[FCidx[j]].b[i] -= h
            
            test_net.forward_prop(X)
            test_net.compute_loss(Y,test_net.P,nBatch,"Training")
            test_net.compute_cost(test_net.loss["Training"][-1],lamda,"Training")

            c1 = test_net.cost["Training"][-1]

            test_net.layers[FCidx[j]].b[i] += 2*h
            
            test_net.forward_prop(X)
            test_net.compute_loss(Y,test_net.P,nBatch,"Training")
            test_net.compute_cost(test_net.loss["Training"][-1],lamda,"Training")

            c2 = test_net.cost["Training"][-1]
            
            grads_b[j][i][0] = (c2-c1) / (2*h)

            #reset entries for next pass
            test_net.layers[FCidx[j]].b[i] -= h
    
    test_net.layers[FCidx[j]].gradb.append(grads_b[j])

    for k in range(len(test_net.weights)):
        print("computing grad for W[%d]" % k)

        grads_W.append(np.zeros(np.shape(test_net.weights[k])))

        for i in range(np.shape(grads_W[k])[0]):
            for j in range(np.shape(grads_W[k])[1]):
                test_net.layers[FCidx[k]].W[i,j] -= h
                
                test_net.forward_prop(X)
                test_net.compute_loss(Y,test_net.P,nBatch,"Training")
                test_net.compute_cost(test_net.loss["Training"][-1],lamda,"Training")

                c1 = test_net.cost["Training"][-1]

                test_net.layers[FCidx[k]].W[i,j] += 2*h
                
                test_net.forward_prop(X)
                test_net.compute_loss(Y,test_net.P,nBatch,"Training")
                test_net.compute_cost(test_net.loss["Training"][-1],lamda,"Training")

                c2 = test_net.cost["Training"][-1]

                grads_W[k][i,j] = (c2-c1) / (2*h)
                
                #reset entries for next pass
                test_net.layers[FCidx[k]].W[i,j] -= h

        test_net.layers[FCidx[k]].gradW.append(grads_W[k])

    return grads_W, grads_b , test_net

def relErr(Wan,Wnum,eps=1e-10):
    # Computes mean relative error between Jacobians Wan and Wnum
    #return np.mean(np.abs(Wan - Wnum)/np.maximum(np.abs(Wan),np.abs(Wnum)))
    relErr = np.mean(np.abs(Wan - Wnum)/np.maximum(eps,(np.abs(Wan) + np.abs(Wnum))))
    return relErr

def maxErr(Wan,Wnum,eps=1e-10):
    # Computes mean absolute error between Jacobians Wan and Wnum
    return np.mean(np.abs(Wan-Wnum))

