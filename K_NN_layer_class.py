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
        self.weights = []
        self.biases = []

        if W is None:
            self.W = random.normal(loc=mu,scale=sig,size=(self.N,self.m))
        else:
            self.W = W

        if b is None:
            self.b = np.zeros((self.N,1))
        else:
            self.b = b

        self.weights.append(self.W)
        self.biases.append(self.b)

    def forward_pass(self, input_data):
        self.input = input_data
        self.output = np.dot(self.W,input_data) + self.b
        return self.output
    
    def backward_pass(self, G, eta):

        """ This here pass has a packpropagated gradient for dJdW which originates both from the hidden layer
         and from the regularization term. 
         We also compute and update the W and b parameter here """
        
        self.compute_grads(G)        
        self.update_pars(eta)

        # Return gradient to pass on upwards
        return np.dot(self.W.T, G)
    
    def compute_grads(self, G):
        """ These gradients are under the assumption of mini batch gradient descent being used
            and cross entropy. Not sure that these are strict necessities """
        dldW = np.dot(G, self.input.T) / self.N
        dldb = np.sum(G, axis=1) / self.N

        dJdW = dldW + 2 * self.lamda * self.W
        dJdb = np.reshape(dldb,(len(dldb),1))

        self.gradW.append(dJdW)
        self.gradb.append(dJdb)

    def update_pars(self, eta):
        self.W -= eta * self.gradW[-1]
        self.b -= eta * self.gradb[-1]

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