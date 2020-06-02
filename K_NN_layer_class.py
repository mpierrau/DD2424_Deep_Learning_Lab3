# Base layer class
import numpy as np 
from numpy import random

class Layer:

    # Constructor
    def __init__(self):
        
        self.input = None
        self.output = None

    def backward_pass(self,G,eta):
        raise NotImplementedError

    def forward_pass(self,input_data):
        raise NotImplementedError

    
class FCLayer(Layer):
    # input_size:
    # output_size:
    
    def __init__(self, input_size, output_size, mu=0, sig=0.01, lamda=0, W = None, b = None):
        self.lamda = lamda
        self.N = input_size
        self.m = output_size
        self.gradW = []
        self.gradb = []
        self.mu = mu
        self.sig = sig

        if W is None:
            self.weights = random.normal(loc=mu,scale=sig,size=(self.N,self.m))
        else:
            self.weights = W

        if b is None:
            self.bias = np.zeros((self.N,1))
        else:
            self.bias = b

    def forward_pass(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights,input_data) + self.bias
        return self.output
    
    def backward_pass(self, G, eta):
        dldW = np.dot(G, self.input.T) / self.N
        dldb = np.sum(G, axis=1) / self.N

        dJdW = dldW + 2 * self.lamda * self.weights
        dJdb = np.reshape(dldb,(len(dldb),1))

        self.gradW.append(dJdW)
        self.gradb.append(dJdb)

        G = np.dot(self.weights.T, G)
        
        self.weights -= eta * dJdW
        self.bias -= eta * dJdb
        
        return G

class ActLayer(Layer):

    def __init__(self, act_func):
        self.act_func = act_func

    def forward_pass(self, input_data):
        self.input = input_data 
        self.output = self.act_func(self.input)
        # Apply batch normalization here
        return self.output

    def backward_pass(self, G, eta):
        #print(np.shape(self.input))
        #print(np.shape(G))
        print("G: ", G)
        tmpG = G
        tmpG[self.input < 0] = 0
        return tmpG

class SoftMaxLayer(Layer):

    def __init__(self):

    def forward_pass(self, input_data):
        self.input = input_data
        self.output = soft_max(input_data)

    def backward_pass(self):
        # What is gradient of soft max?
