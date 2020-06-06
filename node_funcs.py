
import numpy as np
import copy

""" Node functions """

def softMax(X, debug=False):
#Standard definition of the softmax function
    S = np.exp(X) / np.sum(np.exp(X), axis=0)
    return S

def crossEntropy(Y,P,nBatch,oneHotEnc=True):
    # Compute Cross Entropy between Y and P

    N = np.shape(Y)[1]
    
    batches = int(N/nBatch)

    lSum = 0
    if(oneHotEnc):
        for i in range(batches):
            startIdx = i*nBatch
            endIdx = startIdx + nBatch
            Ybatch = Y[:,startIdx:endIdx]
            Pbatch = P[:,startIdx:endIdx]
            lSum += np.trace(-np.log(np.dot(Ybatch.T,Pbatch)))
    else:
        for i in range(batches):
            startIdx = i*nBatch
            endIdx = startIdx + nBatch
            Ybatch = Y[:,startIdx:endIdx]
            Pbatch = P[:,startIdx:endIdx]
            lSum += np.trace(-np.dot(Ybatch.T,np.log(Pbatch)))

    return lSum

def computeAccuracy(X, y, W, b, debug=False):
    # Computes fraction of correctly
    # classified images
    # X must be list of training and validation (and if want, also test) data


    accuracy = np.zeros(len(X))

    for i in range(len(X)):

        tmpX = X[i]
        tmpy = y[i]

        N = np.shape(tmpX)[1]
        
        P = evaluateClassifier(tmpX,W, b,debug)
        
        guess = np.argmax(P,axis=0)
        correct = sum(guess == tmpy)
        
        accuracy[i] = correct/N
        
    return accuracy

def computeGradients(X, Y, P, H, W, lamda,debug=False):
    # Computes gradients of J with respect
    # to W1,W2,b1 and b2
    
    N = np.shape(X)[1]
    G = P-Y
    
    LgradW2 = np.dot(G,H.T)/N
    Lgradb2 = np.sum(G,axis=1)/N

    G = np.dot(W[1].T,G)
    G[H<=0] = 0

    LgradW1 = np.dot(G,X.T)/N
    Lgradb1 = np.sum(G,axis=1)/N

    JgradW1 = LgradW1 + 2*lamda*W[0]
    Jgradb1 = Lgradb1
    JgradW2 = LgradW2 + 2*lamda*W[1]
    Jgradb2 = Lgradb2

    return [JgradW1, JgradW2] , [Jgradb1, Jgradb2]

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
