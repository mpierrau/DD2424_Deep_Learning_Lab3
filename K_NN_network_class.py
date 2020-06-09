import numpy as np
from numpy import random
from K_NN_funcs import setEta , softMax , he_init
import copy
from K_NN_layer_class import FCLayer , ActLayer
from tqdm import trange 

# TODO: why does accuracy turn out only 0 or 0.001 ? All predictions seem to be 0.1 .

class Network:
    def __init__(self):
        
        self.lamda = None
        self.eta = None
        self.n_cycles = None
        self.n_s = None
        self.nBatch = None

        self.weights = []
        self.biases = []
        
        self.input = None
        self.layers = []

        self.loss_func = None
        self.loss_prime_func = None
        self.cost_func = None
        
        self.cost = {"Training":[],"Validation":[],"Test":[]}
        self.loss = {"Training":[],"Validation":[],"Test":[]}
        self.accuracy = {"Training":[],"Validation":[],"Test":[]}
        self.P = {"Training":None,"Validation":None}

    def build_layers(self, data_dim, nClasses, hidden_dim,act_func,init_func=he_init,W=None, b=None):

        n_layers = len(hidden_dim)
        
        if W == None:
            W = []
            for i in range(n_layers):
                W.append(None)

        if b == None:
            b = []
            for i in range(n_layers):
                b.append(None)

        self.add_layer(FCLayer( input_size=data_dim,output_size=hidden_dim[0],
                                init_func=init_func,W=W[0],b=b[0]))
        self.add_layer(ActLayer(act_func))

        for i in range(1,n_layers):
            self.add_layer(FCLayer( input_size=hidden_dim[i-1],output_size=hidden_dim[i],
                                    init_func=init_func,W=W[i],b=b[i]))
            self.add_layer(ActLayer(act_func))

        self.add_layer(FCLayer( input_size=hidden_dim[-1],output_size=nClasses,
                                init_func=init_func,W=W[-1],b=b[-1]))
    

    def add_layer(self, layer):
        layerIdx = len(self.layers)
        layer.layerIdx = layerIdx
        print("Added layer %d : %s" % (layerIdx,layer.name))
        self.layers.append(layer)
    
    def set_loss(self, loss_func, loss_prime_func):
        self.loss_func = loss_func
        self.loss_prime_func = loss_prime_func

    def set_cost(self, cost_func):
        self.cost_func = cost_func

    def forward_prop(self, input_data,key="Training"):
        """ Runs data through network and softmax 
            Returns matrix P of dimension k x N """
        
        self.input = input_data
        output = input_data
        
        for layer in self.layers:
            output = layer.forward_pass(output)
            
        output = softMax(output)

        self.P[key] = output

    def backward_prop(self, input_labels, eta):

        input_error = self.loss_prime_func(input_labels, self.P["Training"])
        for layer in reversed(self.layers):
            input_error = layer.backward_pass(input_error,eta)
            
    def fit(self, X, Y, y,n_cycles,n_s,nBatch,eta,lamda,recPerEp,seed=None):
        """ X = [Xtrain, Xval] 
            Y = [Ytrain, Yval]"""

        self.n_cycles = n_cycles
        self.n_s = n_s
        self.nBatch = nBatch
        self.eta = eta
        self.lamda = lamda

        Xtrain , Xval = X
        Ytrain , Yval = Y
        ytrain , yval = y

        eta_min = self.eta[0]
        eta_max = self.eta[1]
        tot_steps = self.n_cycles*2*self.n_s

        N = np.shape(Xtrain)[1]

        steps_per_ep = int(N/self.nBatch)
        n_epochs = int(np.ceil(tot_steps/steps_per_ep))

        rec_every = int(steps_per_ep/recPerEp)

        index = np.arange(N)
        t = 0

        for epoch in trange(n_epochs):
            # Shuffle batches in each epoch
            random.shuffle(index)

            epoch_steps = steps_per_ep if (((tot_steps - t) // steps_per_ep) > 0 ) else (tot_steps % steps_per_ep) 
            
            bar_range = trange(epoch_steps) # Gives nice progress bar

            for step in bar_range:
                t = epoch*epoch_steps + step
                step_eta = setEta(t,self.n_s,eta_min,eta_max)
                batchIdxStart = step*self.nBatch
                batchIdxEnd = batchIdxStart + self.nBatch

                tmpIdx = index[np.arange(batchIdxStart,batchIdxEnd)]
                Xbatch = Xtrain[:,tmpIdx]
                Ybatch = Ytrain[:,tmpIdx]
                ybatch = ytrain[tmpIdx]
                
                self.forward_prop(Xbatch)
                self.backward_prop(Ybatch,step_eta)
                
                if (step % rec_every) == 0:
                    self.checkpoint(Xval,Yval,yval,"Validation")
                    self.checkpoint(Xbatch,Ybatch,ybatch,"Training")
                    self.save_pars()

            bar_range.close()

    def compute_loss(self, Y, key="Training"):
        
        l = self.loss_func(Y,self.P[key],self.nBatch)

        self.loss[key].append(l)
    
    def compute_cost(self, key="Training"):
        
        J = self.cost_func(self,self.loss[key][-1],self.lamda)
        
        self.cost[key].append(J)
    
    def compute_accuracy(self, X, y, key="Training"):
        N = np.shape(X)[1]

        self.forward_prop(X,key)
        
        guess = np.argmax(self.P[key],axis=0)
        n_correct = sum(guess == y)
        
        self.accuracy[key].append(n_correct/N)

                
    def save_pars(self):
        self.weights.append(self.get_weights())
        self.biases.append(self.get_biases())

    def get_pars(self):
        return self.weights , self.biases

    
    def get_weights(self):
        weights = {}

        for layer in self.layers:
            if type(layer) == FCLayer:
                weights[layer.layerIdx] = layer.W
        
        return weights
    
    def get_biases(self):
        biases = {}

        for layer in self.layers:
            if type(layer) == FCLayer:
                biases[layer.layerIdx] = layer.b

        return biases

    def get_fcIdxs(self):
        FCidx = []
        
        for layer in self.layers:
            if type(layer) == FCLayer:
                FCidx.append(layer.layerIdx)
        
        return FCidx

    def checkpoint(self,X,Y,y,key):
        self.forward_prop(X,key)
        self.compute_loss(Y,key)
        self.compute_cost(key)
        self.compute_accuracy(X,y,key)
    

