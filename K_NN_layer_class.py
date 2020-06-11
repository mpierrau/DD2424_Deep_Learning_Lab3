# Base layer class
import numpy as np 
from numpy import random
import copy

class Layer:

    # Constructor
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
    # input_size:
    # output_size:
    
    def __init__(self, input_size, output_size, init_func, lamda=0, W = None, b = None, name=None, normalize=True, alpha=0.9):
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
                    
        self.W = self.init_func(in_dim=self.nCols,out_dim=self.nRows) if W is None else W
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
            self.normalize_fw_func = self.batch_Non_NormFProp
            self.normalize_bw_func = self.batch_Non_NormBProp
            self.normalize_grad_func = lambda *args : None
            self.normalize_update_pars_func = lambda eta : [self.gamma,self.beta]

    def forward_pass(self, input_data,prediction=False,debug=False):
        self.input = input_data
        self.batchSize = input_data.shape[1]
        self.unnorm_score = np.dot(self.W,self.input) + self.b
        
        if self.avg_mu is None:
            avg_mu = self.unnorm_score.mean(axis=1)
            self.avg_mu = avg_mu
        if self.avg_v is None:
            avg_v = self.unnorm_score.var(axis=1,ddof=0)
            self.avg_v = avg_v

        if prediction==True:
            mu = self.avg_mu
            v = self.avg_v
        else:
            self.batch_mu = self.unnorm_score.mean(axis=1)
            self.batch_v = self.unnorm_score.var(axis=1,ddof=0) # Biased estimator
            
            # For very first batch we initialize avg_mu as mu and avg_v as v, else weighted average.
            avg_mu = np.average([self.avg_mu,self.batch_mu],axis=0,weights=(self.alpha,1-self.alpha))
            avg_v = np.average([self.avg_v,self.batch_v],axis=0,weights=(self.alpha,1-self.alpha))
            
            self.avg_mu = avg_mu
            self.avg_v = avg_v

            mu = self.batch_mu
            v = self.batch_v
        
        self.normalize_fw_func(mu,v)

        s_tilde = self.gamma*self.norm_score + self.beta
        
        self.output = s_tilde

        return self.output
    
    def backward_pass(self, G, eta):

        """ This here pass has a packpropagated gradient for dJdW which originates both from the hidden layer
         and from the regularization term. 
         We also compute and update the W and b parameter here """

        self.compute_norm_grads(G)

        G *= self.gamma

        G = self.normalize_bw_func(G)
        
        newG = np.dot(self.W.T,G)

        self.compute_grads(G)

        self.update_pars(eta)

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

        # Update scale and shift parameters
        newGamma , newBeta = self.normalize_update_pars_func(eta)
        self.gamma = newGamma
        self.beta = newBeta

    def compute_norm_grads(self,G):
        dJdgamma = np.sum((G * self.norm_score),axis=1) / self.batchSize
        dJdbeta = np.sum(G, axis=1) / self.batchSize

        self.gradGamma = dJdgamma.reshape((len(dJdgamma),1))
        self.gradBeta = dJdbeta.reshape((len(dJdbeta),1))

    def batchNormFProp(self,mu,v):
        eps = np.finfo(float).eps

        sigma = v**(-.5)

        norm_score = np.diag(sigma)@(self.unnorm_score - mu[:,None]) + eps
        
        self.norm_score = norm_score
    
    def batch_Non_NormFProp(self,*argv):
        self.norm_score = self.unnorm_score

    def batchNormBProp(self,G):
        """ The columns in newG represent dJ/ds_i ,
            where s_i is the unnormalized score.
            The columns in input G represent dJ/d(s^)_i ,
            where (s^)_i is the normalized score"""

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
        return G

    def updateParsNorm(self,eta):
        newGamma = self.gamma - eta * self.gradGamma
        newBeta = self.beta - eta * self.gradBeta
        
        return newGamma , newBeta

    def get_batch_v(self):
        return self.batch_v

    def get_avg_v(self):
        return self.avg_v

class ActLayer(Layer):

    def __init__(self, act_func,name=None):
        self.act_func = act_func
        self.name = "Activation Layer" if name is None else name

    def forward_pass(self, input_data,prediction=False,debug=False):
        self.input = input_data
        self.output = self.act_func(self.input)
        # Apply batch normalization here
        return self.output

    def backward_pass(self, G, eta):
        
        tmpG = G
        tmpG[self.input <= 0] = 0   
        return tmpG