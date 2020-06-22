""" Layer Class for a Neural Network. Contains fully connected layer (FCLayer) and activation layer (ActLayer) objects. """

import numpy as np 
from random import Random
import copy

class Layer:

    """ Constructor """
    def __init__(self):
        
        self.input = None
        self.output = None
        self.layerIdx = None
        self.name = "LayerName"

    def backward_pass(self,G,eta):
        raise NotImplementedError

    def forward_pass(self,input_data,prediction):
        raise NotImplementedError

class FCLayer(Layer):
    """ Fully connected layer 
        
        input_size , output_size    = dimensions of this layers parameters. (dim(W) = [output_size , input_size]) 
                                    - types:    input_size : int
                                                output_size : int

        init_func   = choose function to initialize parameters. Function takes input_size, output_size as input, and an alternative seed argument for replicability. 
                    - type (function((int),(int),(double)))
        
        lamda   = regularization term parameter. 
                - type: float > 0
        
        W , b   = parameters which can be specified manually. Overrides init_func 
                - types:    W : [output_sizee , input_size] array of floats 
                            b : [output_size , 1] array of floats
        
        seed    = a random seed determining parameter initializations 
                - type: float
        
        name    = optional name for layer. If none will be initialized automatically. 
                - type: string
        
        normalize   = boolean for whether to use batch normalization or not. if false then identity type functions will be used in forward and backward prop in normalization steps. 
                    - type: boolean
        
        alpha   = weights for weighted average of mu and v estimates in batch normalization step. 
                - type: float (0 < alpha < 1)
        """
    
    def __init__(self, input_size, output_size, init_func, lamda, W = None, b = None, seed=None, name=None, normalize=True, alpha=0.9):
        self.batchSize = None
        self.nCols = input_size
        self.nRows = output_size
        
        self.lamda = lamda

        self.gradW = None
        self.gradb = None
        
        self.init_func = init_func

        self.normalize = normalize

        # Pars for batch normalization
        self.batch_mu = None
        self.batch_v = None

        self.alpha = alpha # Weight for avg_mu and avg_v in weighted average
        self.avg_mu = None
        self.avg_v = None

        self.beta = np.zeros((output_size,1))
        self.gamma = np.ones((output_size,1))
        self.gradGamma = np.ones((output_size,1))
        self.gradBeta = np.zeros((output_size,1))

        # These are used in par update of gamma and beta
        self.norm_score = None
        self.unnorm_score = None
        
        self.name = "Fully Connected Layer (%d,%d)" % (self.nRows,self.nCols) if name is None else name
        
        self.seed = seed

        self.W = self.init_func(in_dim=self.nCols,out_dim=self.nRows,seed=self.seed) if W is None else W
        self.b = np.zeros((self.nRows,1)) if b is None else b

        self.normalize_fw_func = None
        self.normalize_bw_func = None
        self.normalize_grad_func = None
        self.normalize_update_pars_func = None

        if self.normalize == True:
            self.normalize_fw_func = self.batchNormFProp
            self.normalize_bw_func = self.batchNormBProp
            self.normalize_grad_func = self.compute_norm_grads
            self.normalize_update_pars_func = self.updateParsNorm
        else:
            # If not BN then updating functions are set to identity type functions
            self.normalize_fw_func = self.batch_Non_NormFProp
            self.normalize_bw_func = self.batch_Non_NormBProp
            self.normalize_grad_func = lambda *args : None
            self.normalize_update_pars_func = lambda eta : [self.gamma,self.beta] 


    def forward_pass(self, input_data,prediction=False):
        """ Forward pass function used in forward propagation. 
            Takes previous layers output as input_data. 
            If prediction = True then normalizing parameters are not updated. """

        self.input = input_data
        self.batchSize = input_data.shape[1]
        
        self.unnorm_score = np.dot(self.W,self.input) + self.b
        
        # In first pass set avg_mu to current batch mean and avg_v to current batch variance
        if self.avg_mu is None:
            avg_mu = self.unnorm_score.mean(axis=1)
            self.avg_mu = avg_mu
        if self.avg_v is None:
            avg_v = self.unnorm_score.var(axis=1,ddof=0)
            self.avg_v = avg_v

        # If prediction then we use batch_mu and batch_v as our parameter for normalization and don't update these values.
        if prediction==True:
            mu = self.avg_mu
            v = self.avg_v
        else:
            self.batch_mu = self.unnorm_score.mean(axis=1)
            self.batch_v = self.unnorm_score.var(axis=1,ddof=0) # Biased estimator
            
            avg_mu = np.average([self.avg_mu,self.batch_mu],axis=0,weights=(self.alpha,1-self.alpha))
            avg_v = np.average([self.avg_v,self.batch_v],axis=0,weights=(self.alpha,1-self.alpha))
            
            self.avg_mu = avg_mu
            self.avg_v = avg_v

            mu = self.batch_mu
            v = self.batch_v
        
        # Forward pass normalization step
        self.normalize_fw_func(mu,v)

        s_tilde = self.gamma*self.norm_score + self.beta
        
        self.output = s_tilde

        return self.output
    
    def backward_pass(self, G, eta):
        """ G is the gradient propagated from previous layer. """
        
        self.compute_norm_grads(G)

        G *= self.gamma

        # Backward pass normalization step
        G = self.normalize_bw_func(G)
        
        newG = np.dot(self.W.T,G)

        self.compute_grads(G)

        self.update_pars(eta)
        
        return newG
    
    def compute_grads(self, G):
        """ Compute and update gradients of W and b for current layer.
            These gradients are under the assumption of mini batch gradient descent being used
            and cross entropy. Not sure that these assumptions are strict necessities """

        dldW = np.dot(G, self.input.T)
        dldb = np.sum(G, axis=1).reshape((len(G),1))
        
        dJdW = (dldW / self.batchSize) + 2 * self.lamda * self.W
        dJdb = dldb / self.batchSize

        self.gradW = dJdW
        self.gradb = dJdb


    def compute_norm_grads(self,G):
        """ Compute and update gradients of gamma and beta for current layer. """

        dJdgamma = np.sum((G * self.norm_score),axis=1) / self.batchSize
        dJdbeta = np.sum(G, axis=1) / self.batchSize

        self.gradGamma = dJdgamma.reshape((len(dJdgamma),1))
        self.gradBeta = dJdbeta.reshape((len(dJdbeta),1))


    def update_pars(self, eta):
        newW = self.W - eta * self.gradW
        newb = self.b - eta * self.gradb
        
        self.W = newW
        self.b = newb

        # Update scale and shift parameters
        newGamma , newBeta = self.normalize_update_pars_func(eta)
        self.gamma = newGamma
        self.beta = newBeta


    def updateParsNorm(self,eta):
        """ Function for updating gamma and beta if normalization is used """
        newGamma = self.gamma - eta * self.gradGamma
        newBeta = self.beta - eta * self.gradBeta
        
        return newGamma , newBeta

    
    def batchNormFProp(self,mu,v):
        """ Batch normalization for forward prop """

        eps = np.finfo(float).eps

        sigma = v**(-.5)

        norm_score = np.diag(sigma)@(self.unnorm_score - mu[:,None]) + eps
        
        self.norm_score = norm_score
    

    def batch_Non_NormFProp(self,*argv):
        """ If normalization=False then set normalized score to unnormalized score (identity) """
        self.norm_score = self.unnorm_score


    def batchNormBProp(self,G):
        """ Backward prop for batch normalization.
            The columns in returned G represent dJ/ds_i ,
            where s_i is the unnormalized score.
            The columns in input G represent dJ/d(s^)_i ,
            where (s^)_i is the normalized score """

        N = np.shape(G)[1]

        eps = np.finfo(float).eps

        sig1 = ((self.batch_v + eps)**(-.5))
        sig2 = ((self.batch_v + eps)**(-1.5))

        G1 = G*sig1[:,None]
        G2 = G*sig2[:,None]

        D = self.unnorm_score - self.batch_mu[:,None]
        c = np.sum(G2*D,axis=1)

        G = G1 - (np.sum(G1,axis=1) / N)[:,None] - (D*c[:,None] / N )

        return G
    

    def batch_Non_NormBProp(self,G):
        """ Backward prop normalization step if normalization=False. Simply return G (identity) """
        return G



class ActLayer(Layer):
    """ Activation layer class. 
        
        act_func    = Activation function, which takes data as input and returns matrix of same dimensions with some entries 0 depending on activation function used. (Throughout this assignment we use ReLu.) 
                    - type: function
        
        name    = optional name for layer. If none will be initialized automatically. 
                - type: string """

    def __init__(self, act_func,name=None):
        self.act_func = act_func
        self.name = "Activation Layer" if name is None else name

    def forward_pass(self, input_data,prediction=False):
        self.input = input_data
        self.output = self.act_func(self.input)

        return self.output

    def backward_pass(self, G, eta):
        
        tmpG = G
        
        # Entry i,j in tmpG is set to 0 if entry i,j if input from previous node is <= 0.
        tmpG[self.input <= 0] = 0   
        return tmpG