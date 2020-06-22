import numpy as np
import math
from scipy.linalg import fractional_matrix_power
import csv

def relu(input_data):
    """ ReLu activation function """
    input_data[input_data < 0] = 0
    return input_data

def L2_cost(net,loss,lamda):
    """ L2 regularization """
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
    """ Computes the error between Y and P. """
    return -(Y-P)

def cross_entropy(Y,P,nBatch,oneHotEnc=True):
    """ Computes average Cross Entropy between Y and P batchwise """

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
    """ Standard definition of the softmax function """
    S = np.exp(X) / np.sum(np.exp(X), axis=0)
    
    return S


def setEta(epoch,n_s,etaMin, etaMax):
    """ Cyclical learning rate
    
     n_s must be a positive integer
     n_s is typically chosen to be
     k*np.floor(N/n_batch) with k being
     an integer between  2 and 8 and N
     is the total number of training examples
     and n_batch the number of examples in a 
     batch.

     "Normalize" the time so we don't have to
     worry about l """

    t = epoch % (2*n_s)

    if (t <= n_s):
        etat = etaMin*(1-t/n_s) + etaMax*t/n_s
    else:
        etat = etaMax*(2-t/n_s) + etaMin*(t/n_s-1)
    
    return etat


def he_init(in_dim,out_dim,seed=None):
    """ He (Kaiming) initialization """
    np.random.seed(seed)
    mat = np.random.normal(0,1,(out_dim,in_dim))*math.sqrt(2./in_dim)
    
    return mat

def xavier_init(in_dim,out_dim,seed=None):
    """ Xavier initalization """
    np.random.seed(seed)
    mat = np.random.uniform(-1,1,(out_dim,in_dim))*math.sqrt(6./(in_dim + out_dim))

    return mat

def regular_init(in_dim,out_dim,seed=None):
    """ Regular naive initialization """
    np.random.seed(seed)
    mat = np.random.normal(0,1/np.sqrt(in_dim),(out_dim,in_dim))

    return mat


def normal_init(in_dim,out_dim, *args):
    """ Initialize vectors/matrices from normal dist with mu mean, sigma variance. 
        Used in experiment to determine stability of BN vs. non-BN. """

    mu = args[0]
    sig = args[1]
    seed = args[2]
    np.random.seed(seed)
    mat = np.random.normal(mu,sig,(out_dim,in_dim))

    return mat


def write_metrics(net,fileName):
    """ Saves loss, cost and accuracy to fileName.csv """

    header = ['Step','Loss','Cost','Accuracy']

    totSteps = 2*net.n_s*net.n_cycles
    
    for key in ["Training","Validation","Test"]:
        steps = range(0,totSteps,net.rec_every)
        
        if len(steps) == 0:
            break

        thisFile = "%s_%s.csv" % (fileName,key)
        
        f = open(thisFile ,"w")
        
        with f:
            writer = csv.writer(f)
            writer.writerow(header)

        f.close()

        f = open(thisFile,"a+")
        
        with f:
            writer = csv.writer(f)
            for i in range(len(net.loss[key])):
                vals = [steps[i] , net.loss[key][i] , net.cost[key][i] , net.accuracy[key][i]]
                writer.writerow(vals)

        f.close()

        print("%s results saved in %s" % (key,thisFile))

