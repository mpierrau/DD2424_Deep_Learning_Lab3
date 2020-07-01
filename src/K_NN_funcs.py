import numpy as np
import math
from scipy.linalg import fractional_matrix_power

def relu(input_data):
    """ ReLu activation function """
    input_data[input_data < 0] = 0
    return input_data

def L2_cost(net,loss):
    """ L2 regularization """
    N = net.input.shape[1]

    weights_sum = 0
    weights = net.get_pars("W")

    for W in weights["W"].values():
        Wsum = np.sum(W**2)
        weights_sum += Wsum
    
    L2_reg = net.lamda * weights_sum / N
    
    cost = loss + L2_reg

    return cost

def cross_entropy_prime(Y,P):
    """ Computes the error between Y and P. """
    return -(Y-P)

def cross_entropy(Y,P,oneHotEnc=True):
    """ Computes average Cross Entropy between Y and P batchwise """

    N = np.shape(Y)[1]
    
    if(oneHotEnc):
        entrFunc = lambda Y,P : np.trace(-np.log(np.dot(Y.T,P)))
    else:
        entrFunc = lambda Y,P : np.trace(-np.dot(Y.T,np.log(P)))    
    
    entr = entrFunc(Y,P) / N

    return entr


def softMax(X, debug=False):
    """ Standard definition of the softmax function """
    S = np.exp(X) / np.sum(np.exp(X), axis=0)
    
    return S


def setEta(epoch,ns,etaMin, etaMax):
    """ Cyclical learning rate
    
     ns must be a positive integer
     ns is typically chosen to be
     k*np.floor(N/n_batch) with k being
     an integer between  2 and 8 and N
     is the total number of training examples
     and n_batch the number of examples in a 
     batch.

     "Normalize" the time so we don't have to
     worry about l """

    t = epoch % (2*ns)

    if (t <= ns):
        etat = etaMin*(1-t/ns) + etaMax*t/ns
    else:
        etat = etaMax*(2-t/ns) + etaMin*(t/ns-1)
    
    return etat


def he_init(inDim,outDim,seed=None):
    """ He (Kaiming) initialization """
    np.random.seed(seed)
    mat = np.random.normal(0,1,(outDim,inDim))*math.sqrt(2./inDim)
    
    return mat

def xavier_init(inDim,outDim,seed=None):
    """ Xavier initalization """
    np.random.seed(seed)
    mat = np.random.uniform(-1,1,(outDim,inDim))*math.sqrt(6./(inDim + outDim))

    return mat

def regular_init(inDim,outDim,seed=None):
    """ Regular naive initialization """
    np.random.seed(seed)
    mat = np.random.normal(0,1/np.sqrt(inDim),(outDim,inDim))

    return mat


def normal_init(inDim,outDim, *args):
    """ Initialize vectors/matrices from normal dist with mu mean, sigma variance. 
        Used in experiment to determine stability of BN vs. non-BN. """

    mu = args[0]
    sig = args[1]
    seed = args[2]
    np.random.seed(seed)
    mat = np.random.normal(mu,sig,(outDim,inDim))

    return mat