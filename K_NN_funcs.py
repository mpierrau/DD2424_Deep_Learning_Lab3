from K_NN_layer_class import FCLayer
import numpy as np

def relu(input_data):
    # ReLu activation function
    input_data[input_data < 0] = 0
    return input_data

def L2_cost(net,loss,lamda):
    
    N = net.input.shape[1]

    weights_sum = 0
    weights = net.get_weights()

    for W in weights:
        Wsum = np.sum(W**2)
        weights_sum += Wsum
    
    L2_reg = lamda * weights_sum / N
    
    cost = loss + L2_reg

    return cost

def cross_entropy_prime(Y,P):
    return -(Y-P)

def cross_entropy(Y,P,nBatch,oneHotEnc=True):
    # Compute Cross Entropy between Y and P

    N = np.shape(Y)[1]
    batches = int(N/nBatch)
    
    if(oneHotEnc):
        entrFunc = lambda Y,P : np.trace(-np.log(np.dot(Y.T,P)))
    else:
        entrFunc = lambda Y,P : np.trace(-np.dot(Y.T,np.log(P)))    
    
    entrSum = 0
    for i in range(batches):
        startIdx = i*nBatch
        endIdx = startIdx + nBatch
        
        Ybatch = Y[:,startIdx:endIdx]
        Pbatch = P[:,startIdx:endIdx]

        entrSum += entrFunc(Ybatch,Pbatch)
    
    entrSum /= N

    return entrSum


def softMax(X, debug=False):
    #Standard definition of the softmax function
    S = np.exp(X) / np.sum(np.exp(X), axis=0)
    
    return S


def setEta(epoch,n_s,etaMin, etaMax):
    # Cyclical learning rate
    #
    # n_s must be a positive integer
    # n_s is typically chosen to be
    # k*np.floor(N/n_batch) with k being
    # an integer between  2 and 8 and N
    # is the total number of training examples
    # and n_batch the number of examples in a 
    # batch.

    # "Normalize" the time so we don't have to
    # worry about l

    t = epoch % (2*n_s)

    if (t <= n_s):
        etat = etaMin*(1-t/n_s) + etaMax*t/n_s
    else:
        etat = etaMax*(2-t/n_s) + etaMin*(t/n_s-1)
    
    return etat
