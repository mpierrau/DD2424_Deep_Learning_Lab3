from K_NN_layer_class import FCLayer
import numpy as np

def relu(input_data):
    # ReLu activation function
    input_data[input_data <= 0] = 0
    return input_data

def L2_cost(loss,lamda,net):
    weights_sum = 0
    
    for layer in net.layers:
        if type(layer) == FCLayer:
            weights_sum += np.sum(layer.weights**2)

    return loss + lamda*weights_sum

def cross_entropy_prime(Y,P):
    return P-Y

def cross_entropy(Y,P,nBatch,oneHotEnc=True):
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
