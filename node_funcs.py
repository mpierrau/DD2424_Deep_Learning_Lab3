
import numpy as np
import copy

""" Node functions """

def softMax(X, debug=False):
#Standard definition of the softmax function
    S = np.exp(X) / np.sum(np.exp(X), axis=0)
    return S

def evaluateClassifier(X, W, b, debug=False, returnH=False):
    # Forward pass
    # Softmax makes all entries S = WX + b 
    # run from 0 to 1 and sum to 1 over rows
    # Returns K x N matrix with probs of each 
    # image having label 0,1,...,K-1

    # P dim: K x N
    # X dim: d x N (d = (C x C x rgb))
    # 
    # W is a list containing parameters W1,W2
    # b is a list containing parameters b1, b2 
    # 
    # W1 dim: m x d    
    # W2 dim: K x m
    # b1 dim: m x 1
    # b2 dim: K x 1

    # s1 dim: m x N

    s1 = np.dot(W[0],X) + b[0]

    # h dim: m x N
    h = s1
    # Quick way of doing np.maximum(s1,0) (ReLu)
    h[h<0] = 0

    s2 = np.dot(W[1],h) + b[1]
    P = softMax(s2,debug)
    
    if returnH:
        ret = [P , h]
    else:
        ret = P

    return ret

def evaluateClassifier2(X, W, b, nBatch, debug=False, returnH=False):
    # Forward pass
    # Softmax makes all entries S = WX + b 
    # run from 0 to 1 and sum to 1 over rows
    # Returns K x N matrix with probs of each 
    # image having label 0,1,...,K-1

    # P dim: K x N
    # X dim: d x N (d = (C x C x rgb))
    # 
    # W is a list containing parameters W1,W2
    # b is a list containing parameters b1, b2 
    # 
    # W1 dim: m x d    
    # W2 dim: K x m
    # b1 dim: m x 1
    # b2 dim: K x 1

    # s1 dim: m x N

    N = np.shape(X)[1]
    K = np.shape(W[1])[0]
    m = np.shape(W[0])[0]

    batches = int(N/nBatch)
    P = np.zeros((K,N))
    h = np.zeros((m,N))
    
    for i in range(batches):
        startIdx = nBatch*i
        endIdx = startIdx + nBatch

        tmpX = X[:,startIdx:endIdx]

        s1 = np.dot(W[0],tmpX) + b[0]

        # h dim: m x N
        tmph = s1
        # Quick way of doing np.maximum(s1,0) (ReLu)
        tmph[tmph<0] = 0

        s2 = np.dot(W[1],tmph) + b[1]

        P[:,startIdx:endIdx] = softMax(s2,debug)  
        h[:,startIdx:endIdx] = tmph

    if returnH:
        ret = [P , h]
    else:
        ret = P

    return ret

def crossEntropy(Y,P,nBatch,oneHotEnc=True):
    # Compute Cross Entropy between Y and P

    N = np.shape(Y)[1]
    
    batches = int(N/nBatch)

    sum = 0
    if(oneHotEnc):
        for i in range(batches):
            startIdx = i*nBatch
            endIdx = startIdx + nBatch
            Ybatch = Y[:,startIdx:endIdx]
            Pbatch = P[:,startIdx:endIdx]
            sum += np.trace(-np.log(np.dot(Ybatch.T,Pbatch)))
    else:
        for i in range(batches):
            startIdx = i*nBatch
            endIdx = startIdx + nBatch
            Ybatch = Y[:,startIdx:endIdx]
            Pbatch = P[:,startIdx:endIdx]
            sum += np.trace(-np.dot(Ybatch.T,np.log(Pbatch)))

    return sum

def computeLoss(X, Y, W, b, nBatch, debug=False):
    # X and Y must be list of training and validation data

    if not(type(X) == list):
        thisX = [X]
        thisY = [Y]
    else:
        thisX = X
        thisY = Y

    L = np.zeros(len(thisX))

    for i in range(len(thisX)):
        tmpX = thisX[i]
        tmpY = thisY[i]

        N = np.shape(tmpX)[1]
        P = evaluateClassifier(tmpX,W, b,debug)
        H = crossEntropy(tmpY,P,nBatch,debug)

        L[i] = H/N

    return L

def computeCost(X, Y, W, b, lamda,nBatch, debug=False):
    # Cost function
    # Cost = Loss + Regularization

    L = computeLoss(X,Y,W,b,nBatch,debug)

    J = L + lamda*(np.sum(W[0]**2) + np.sum(W[1]**2))

    return J , L

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
