""" Layer Class for a Neural Network. Contains fully connected layer (FCLayer) and activation layer (ActLayer) objects. """

import numpy as np 
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

    def forward_pass(self,inputData,prediction):
        raise NotImplementedError

class FCLayer(Layer):
    """ Fully connected layer 
        
        input_size , output_size    = dimensions of this layers parameters. (dim(W) = [output_size , input_size]) 
                                    - types:    input_size : int
                                                output_size : int

        init_func                   = choose function to initialize parameters. Function takes input_size, output_size as input, and an alternative seed argument for replicability. 
                                    - type (function((int),(int),(double)))
        
        lamda                       = regularization term parameter. 
                                    - type: float > 0
        
        W , b                       = weight parameters which can be specified manually. Overrides init_func 
                                    - types:    W : [output_sizee , input_size] array of floats 
                                                b : [output_size , 1] array of floats
        
        seed                        = random seed determining parameter initializations 
                                    - type: float
        
        name                        = optional name for layer. If none will be initialized automatically. 
                                    - type: string
        
        normalize                   = boolean for whether to use batch normalization or not. if false then identity type functions will be used in forward and backward prop in normalization steps. 
                                    - type: boolean
        
        alpha                       = weights for weighted average of mu and v estimates in batch normalization step. 
                                    - type: float (0 < alpha < 1)
        """
    
    def __init__(self, input_size, output_size, init_func, lamda, W = None, b = None, seed = None, name = None, normalize = True, alpha = 0.9):
        self.batchSize = None
        self.nCols = input_size
        self.nRows = output_size
        
        self.lamda = lamda
        self.init_func = init_func
        self.seed = seed
        self.normalize = normalize

        self.grads = {  "W": None, 
                        "b": None, 
                        "gamma": np.ones((output_size,1)), 
                        "beta": np.zeros((output_size,1))}

        self.pars = {   "W": self.init_func(inDim=self.nCols, outDim=self.nRows, seed=self.seed) if W is None else W,
                        "b": np.zeros((self.nRows,1)) if b is None else b,
                        "gamma": np.ones((output_size,1)),
                        "beta": np.zeros((output_size,1))}

        # Pars for batch normalization
        self.batchMu = None
        self.batchV = None

        self.alpha = alpha # Weight for avgMu and avgV in weighted average
        self.avgMu = None
        self.avgV = None

        # These are used in par update of gamma and beta
        self.norm_score = None
        self.unnorm_score = None
        
        self.name = "Fully Connected Layer (%d,%d)" % (self.nRows,self.nCols) if name is None else name
        

        self.normalize_fw_func = None
        self.normalize_bw_func = None
        self.normalize_grad_func = None
        self.normalize_update_pars_func = None

        if self.normalize == True:
            self.normalize_fw_func = self.batch_norm_fwdprop
            self.normalize_bw_func = self.batch_norm_backprop
            self.normalize_grad_func = self.compute_norm_grads
            self.normalize_update_pars_func = self.update_pars_norm
        else:
            # If not BN then updating functions are set to identity like functions
            self.normalize_fw_func = self.batch_non_norm_fwdprop
            self.normalize_bw_func = self.batch_non_norm_backprop
            self.normalize_grad_func = lambda *args : None
            self.normalize_update_pars_func = lambda eta : [self.pars["gamma"],self.pars["beta"]] 


    def forward_pass(self, inputData,prediction=False):
        """ Forward pass function used in forward propagation. 
            Takes previous layers output as inputData. 
            If prediction = True then normalizing parameters are not updated. """

        self.input = inputData
        self.batchSize = inputData.shape[1]
        
        self.unnorm_score = np.dot(self.pars["W"], self.input) + self.pars["b"]
        
        # In first pass set avgMu to current batch mean and avgV to current batch variance
        if self.avgMu is None:
            avgMu = self.unnorm_score.mean(axis=1)
            self.avgMu = avgMu
        if self.avgV is None:
            avgV = self.unnorm_score.var(axis=1,ddof=0)
            self.avgV = avgV

        # If prediction then we use batchMu and batchV as our parameter for normalization and don't update these values.
        if prediction==True:
            mu = self.avgMu
            v = self.avgV
        else:
            self.batchMu = self.unnorm_score.mean(axis=1)
            self.batchV = self.unnorm_score.var(axis=1,ddof=0) # Biased estimator
            
            avgMu = np.average([self.avgMu,self.batchMu],axis=0,weights=(self.alpha,1-self.alpha))
            avgV = np.average([self.avgV,self.batchV],axis=0,weights=(self.alpha,1-self.alpha))
            
            self.avgMu = avgMu
            self.avgV = avgV

            mu = self.batchMu
            v = self.batchV
        
        # Forward pass normalization step
        self.normalize_fw_func(mu,v)

        s_tilde = self.pars["gamma"]*self.norm_score + self.pars["beta"]
        
        self.output = s_tilde

        return self.output
    
    def backward_pass(self, G, eta):
        """ G is the gradient propagated from previous layer. """
        
        self.compute_norm_grads(G)

        G *= self.pars["gamma"]

        # Backward pass normalization step
        G = self.normalize_bw_func(G)
        
        newG = np.dot(self.pars["W"].T, G)

        self.compute_grads(G)

        self.update_pars(eta)
        
        return newG
    
    def compute_grads(self, G):
        """ Compute and update gradients of W and b for current layer.
            These gradients are under the assumption of mini batch gradient descent being used
            and cross entropy. Not sure that these assumptions are strict necessities """

        dldW = np.dot(G, self.input.T)
        dldb = np.sum(G, axis=1).reshape((len(G),1))
        
        dJdW = (dldW / self.batchSize) + 2 * self.lamda * self.pars["W"]
        dJdb = dldb / self.batchSize

        self.grads["W"] = dJdW
        self.grads["b"] = dJdb


    def compute_norm_grads(self,G):
        """ Compute and update gradients of gamma and beta for current layer. """

        dJdgamma = np.sum((G * self.norm_score),axis=1) / self.batchSize
        dJdbeta = np.sum(G, axis=1) / self.batchSize

        self.grads["gamma"] = dJdgamma.reshape((len(dJdgamma),1))
        self.grads["beta"] = dJdbeta.reshape((len(dJdbeta),1))


    def update_pars(self, eta):
        self.pars["W"] -= eta * self.grads["W"]
        self.pars["b"] -= eta * self.grads["b"]
        
        # Update scale and shift parameters
        self.pars["gamma"] , self.pars["beta"] = self.normalize_update_pars_func(eta)


    def update_pars_norm(self,eta):
        """ Function for updating gamma and beta if normalization is used """
        newGamma = self.pars["gamma"] - eta * self.grads["gamma"]
        newBeta = self.pars["beta"] - eta * self.grads["beta"]
        
        return newGamma , newBeta

    
    def batch_norm_fwdprop(self,mu,v):
        """ Batch normalization for forward prop """

        eps = np.finfo(float).eps

        sigma = v**(-.5)

        norm_score = np.diag(sigma)@(self.unnorm_score - mu[:,None]) + eps
        
        self.norm_score = norm_score
    

    def batch_non_norm_fwdprop(self,*argv):
        """ If normalization=False then set normalized score to unnormalized score (identity) """
        self.norm_score = self.unnorm_score


    def batch_norm_backprop(self,G):
        """ Backward prop for batch normalization.
            The columns in returned G represent dJ/ds_i ,
            where s_i is the unnormalized score.
            The columns in input G represent dJ/d(s^)_i ,
            where (s^)_i is the normalized score """

        N = np.shape(G)[1]

        eps = np.finfo(float).eps

        sig1 = ((self.batchV + eps)**(-.5))
        sig2 = ((self.batchV + eps)**(-1.5))

        G1 = G*sig1[:, None]
        G2 = G*sig2[:, None]

        D = self.unnorm_score - self.batchMu[:, None]
        c = np.sum(G2 * D,axis=1)

        G = G1 - (np.sum(G1, axis=1) / N)[:, None] - (D * c[:,None] / N )

        return G
    

    def batch_non_norm_backprop(self,G):
        """ Backward prop normalization step if normalization=False. Simply return G (identity) """
        return G



class ActLayer(Layer):
    """ Activation layer class. 
        
        act_func    = Activation function, which takes data as input and returns matrix of same dimensions with some entries 0 depending on activation function used. (Throughout this assignment we use ReLu.) 
                    - type: function
        
        name        = optional name for layer. If none will be initialized automatically. 
                    - type: string """

    def __init__(self, act_func, name = None):
        self.act_func = act_func
        self.name = "Activation Layer" if name is None else name

    def forward_pass(self, inputData, prediction=False):
        self.input = inputData
        self.output = self.act_func(self.input)

        return self.output

    def backward_pass(self, G, eta):
        tmpG = G
        
        # Entry i,j in tmpG is set to 0 if entry i,j if input from previous node is <= 0.
        tmpG[self.input <= 0] = 0   
        return tmpG