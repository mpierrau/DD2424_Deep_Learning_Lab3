import numpy as np
from numpy import random
from K_NN_funcs import setEta , softMax , he_init , relu , cross_entropy , cross_entropy_prime , L2_cost
import copy
from K_NN_layer_class import FCLayer , ActLayer
from tqdm import trange 

class Network:
    def __init__(self,act_func=relu,loss_func=cross_entropy,loss_prime_func=cross_entropy_prime,cost_func=L2_cost,init_func=he_init,normalize=True,alpha=0.9):
        
        self.lamda = None
        self.eta = None
        self.n_cycles = None
        self.n_s = None
        self.nBatch = None

        self.alpha = alpha

        self.weights = []
        self.biases = []
        
        self.input = None
        self.layers = []
        self.layer_dims = []

        self.act_func = act_func
        self.loss_func = loss_func
        self.loss_prime_func = loss_prime_func
        self.cost_func = cost_func
        self.init_func = init_func
        self.normalize = normalize

        self.cost = {"Training":[],"Validation":[],"Test":[]}
        self.loss = {"Training":[],"Validation":[],"Test":[]}
        self.accuracy = {"Training":[],"Validation":[],"Test":[]}
        self.P = {"Training":None,"Validation":None}


    def build_layers(self, data_dim, nClasses, hidden_dim,W=None, b=None,verbose=True):

        n_layers = len(hidden_dim)
        
        self.layer_dims = hidden_dim

        if W == None:
            W = []
            for i in range(n_layers):
                W.append(None)

        if b == None:
            b = []
            for i in range(n_layers):
                b.append(None)
        
        if verbose:
            print("\n----------------------------------\n")
            print("-- Building Network with parameters --\n")
            print("- Input dimension : %d \n- Number of classes : %d\n" % (data_dim,nClasses))
            print("- Number of hidden layers: %d" % (len(hidden_dim)))
            print("\t- Dims: ", end='')
            for dim in hidden_dim:
                print(dim, end=" ")
            print("\n----------------------------------\n")


        self.add_layer(FCLayer( input_size=data_dim,output_size=hidden_dim[0],
                                init_func=self.init_func,W=W[0],b=b[0],normalize=self.normalize,alpha=self.alpha),verbose=verbose)
        self.add_layer(ActLayer(self.act_func),verbose=verbose)

        for i in range(1,n_layers):
            self.add_layer(FCLayer( input_size=hidden_dim[i-1],output_size=hidden_dim[i],
                                    init_func=self.init_func,W=W[i],b=b[i],normalize=self.normalize,alpha=self.alpha),verbose=verbose)
            self.add_layer(ActLayer(self.act_func),verbose=verbose)

        self.add_layer(FCLayer( input_size=hidden_dim[-1],output_size=nClasses,
                                init_func=self.init_func,W=W[-1],b=b[-1],normalize=False,alpha=self.alpha),verbose=verbose)
    
        if verbose:
            print("\n---------------DONE---------------\n")

    def add_layer(self, layer, verbose=True):
        layerIdx = len(self.layers)
        layer.layerIdx = layerIdx
        if verbose:
            print("Added layer %d : %s" % (layerIdx,layer.name))
        self.layers.append(layer)
    
    def forward_prop(self, input_data,key="Training",prediction=False,debug=False):
        """ Runs data through network and softmax 
            Returns matrix P of dimension k x N """
        
        self.input = input_data
        output = input_data
        
        for layer in self.layers:
            output = layer.forward_pass(output,prediction,debug=debug)
            
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
            
            bar_range = trange(epoch_steps,leave=False) # Gives nice progress bar

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

    def set_loss(self, loss_func, loss_prime_func):
        self.loss_func = loss_func
        self.loss_prime_func = loss_prime_func

    def set_cost(self, cost_func):
        self.cost_func = cost_func

    def compute_loss(self, Y, key="Training"):
        
        l = self.loss_func(Y,self.P[key],self.nBatch)

        self.loss[key].append(l)
    
    def compute_cost(self, key="Training"):
        
        J = self.cost_func(self,self.loss[key][-1],self.lamda)
        
        self.cost[key].append(J)
    
    def compute_accuracy(self, X, y, key="Training"):
        N = len(y)

        guess = self.predict(X,y,key)

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

    def get_gammas(self):
        gammas = {}

        for layer in self.layers:
            if type(layer) == FCLayer:
                gammas[layer.layerIdx] = layer.gamma

        return gammas

    def get_betas(self):
        betas = {}

        for layer in self.layers:
            if type(layer) == FCLayer:
                betas[layer.layerIdx] = layer.beta

        return betas
   
    def checkpoint(self,X,Y,y,key):
        self.forward_prop(X,key,prediction=True)
        self.compute_loss(Y,key)
        self.compute_cost(key)
        self.compute_accuracy(X,y,key)

    def predict(self,X,y,key):
        self.forward_prop(X,key,prediction=True)
        prediction = np.argmax(self.P[key],axis=0)

        return prediction