from K_NN_layer_class import FCLayer
import numpy as np

def relu(input_data):
    # ReLu activation function
    input_data[input_data < 0] = 0
    return input_data

def L2_cost(net,loss,lamda):
    
    N = net.input.shape[1]

    weights_sum = 0

    for W in net.weights:
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

def reduceDims(X,Y,redDim,redN):
    XbatchRedDim = X[:redDim,:redN]
    YbatchRedDim = Y[:,:redN]
    return XbatchRedDim , YbatchRedDim
