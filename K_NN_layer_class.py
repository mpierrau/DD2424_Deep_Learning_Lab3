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
        self.mu = None
        self.v = None

        self.alpha = alpha # Weight for mu_avg and v_avg in weighted average
        self.mu_avg = None
        self.v_avg = None

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

    def forward_pass(self, input_data,prediction=False):

        self.input = input_data
        self.batchSize = input_data.shape[1]

        self.unnorm_score = np.dot(self.W,self.input) + self.b

        if prediction==True:
            mu = self.mu_avg
            v = self.v_avg
        else:
            self.mu = self.unnorm_score.mean(axis=1)
            self.v = self.unnorm_score.var(axis=1,ddof=0) # Biased estimator
            
            # For very first batch we initialize mu_avg as mu and v_avg as v, else weighted average.
            self.mu_avg = self.mu if self.mu_avg is None else np.average([self.mu_avg,self.mu],axis=0,weights=(self.alpha,1-self.alpha))
            self.v_avg = self.v if self.v_avg is None else np.average([self.v_avg,self.v],axis=0,weights=(self.alpha,1-self.alpha))

            mu = self.mu
            v = self.v

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

        # Update scale and shift parameters
        newGamma , newBeta = self.normalize_update_pars_func(eta)

        self.gamma = newGamma
        self.beta = newBeta

    def compute_norm_grads(self,G):
        dJdgamma = np.sum((G * self.norm_score),axis=1) / self.batchSize
        dJdbeta = np.sum(G, axis=1) / self.batchSize

        self.gradGamma = dJdgamma
        self.gradBeta = dJdbeta

    def batchNormFProp(self,*argv):

        if len(argv) == 0:
            mu = self.mu
            v = self.v
        else:
            mu = argv[0]
            v = argv[1]
        
        eps = np.finfo(float).eps
        sigma = v
        sigma += eps
        sigma **= -.5
        norm_score = np.diag(sigma)@(self.unnorm_score - mu[:,None])
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

        sig1 = ((self.v + eps)**(-.5))
        sig2 = ((self.v + eps)**(-1.5))

        G1 = G*sig1[:,None]
        G2 = G*sig2[:,None]

        D = self.unnorm_score - self.mu[:,None]
        c = np.sum(G2*D,axis=1)

        G = G1 - (np.sum(G1,axis=1) / N)[:,None] - (D*c[:,None] / N )

        return G
    
    def batch_Non_NormBProp(self,G):
        return G

    def updateParsNorm(self,eta):
        newGamma = self.gamma - eta * self.gradGamma[:,None]
        newBeta = self.beta - eta * self.gradBeta[:,None]
        
        return newGamma , newBeta


class ActLayer(Layer):

    def __init__(self, act_func,name=None):
        self.act_func = act_func
        self.name = "Activation Layer" if name is None else name

    def forward_pass(self, input_data,prediction=False):
        self.input = input_data
        self.output = self.act_func(self.input)
        # Apply batch normalization here
        return self.output

    def backward_pass(self, G, eta):
        
        tmpG = G
        tmpG[self.input <= 0] = 0   
        return tmpG