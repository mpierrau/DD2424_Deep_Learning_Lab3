# Base layer class
import numpy as np 
from numpy import random

class Layer:

    # Constructor
    def __init__(self):
        
        self.input = None
        self.output = None
        self.layerIdx = None
        self.name = "LayerName"

    def backward_pass(self,G,eta):
        raise NotImplementedError

    def forward_pass(self,input_data):
        raise NotImplementedError

class FCLayer(Layer):
    # input_size:
    # output_size:
    
    def __init__(self, input_size, output_size, init_func, lamda=0, W = None, b = None, name=None):
        self.lamda = lamda
        self.batchSize = None
        self.nCols = input_size
        self.nRows = output_size
        self.gradW = None
        self.gradb = None
        self.init_func = init_func

        if name is None:
            self.name = "Fully Connected Layer (%d,%d)" % (self.nRows,self.nCols)
        else:
            self.name = name
                    
        if W is None:
            self.W = self.init_func(in_dim=self.nCols,out_dim=self.nRows)
        else:
            self.W = W

        if b is None:
            self.b = np.zeros((self.nRows,1))
        else:
            self.b = b

    def forward_pass(self, input_data):
        self.input = input_data
        self.batchSize = input_data.shape[1]
        self.output = np.dot(self.W,self.input) + self.b
        return self.output
    
    def backward_pass(self, G, eta):

        """ This here pass has a packpropagated gradient for dJdW which originates both from the hidden layer
         and from the regularization term. 
         We also compute and update the W and b parameter here """
        
        newG = np.dot(self.W.T,G)

        self.compute_grads(G)
        self.update_pars(eta)

        # Return gradient to pass on upwards
        return newG
    
    def compute_grads(self, G):
        """ These gradients are under the assumption of mini batch gradient descent being used
            and cross entropy. Not sure that these are strict necessities """

        dldW = np.dot(G, self.input.T)
        dldb = np.sum(G, axis=1).reshape((len(G),1))
        
        dJdW = (dldW / self.batchSize) + 2 * self.lamda * self.W
        dJdb = dldb / self.batchSize

        self.gradW = dJdW
        self.gradb = dJdb

    def update_pars(self, eta):
        newW = self.W - eta * self.gradW
        newb = self.b - eta * self.gradb
        
        self.W = newW
        self.b = newb

class ActLayer(Layer):

    def __init__(self, act_func,name=None):
        self.act_func = act_func

        if name is None:
            self.name = "Activation Layer"
        else:
            self.name = name

    def forward_pass(self, input_data):
        self.input = input_data
        self.output = self.act_func(self.input)
        # Apply batch normalization here
        return self.output

    def backward_pass(self, G, eta):
        
        tmpG = G
        tmpG[self.input <= 0] = 0   
        return tmpG