import numpy as np
from numpy import random
from node_funcs import setEta , softMax
import copy
from K_NN_layer_class import FCLayer , ActLayer

# TODO: why does accuracy turn out only 0 or 0.001 ? All predictions seem to be 0.1 .

class Network:
    def __init__(self,lamda,eta,n_cycles,n_s,nBatch):
        self.lamda = lamda
        self.eta = eta
        self.n_cycles = n_cycles
        self.n_s = n_s
        self.nBatch = nBatch
        self.eta_list = []
        self.input = None
        self.layers = []
        self.loss_func = None
        self.loss_prime_func = None
        self.cost_func = None
        self.cost = {"Training":[],"Validation":[],"Test":[]}
        self.loss = {"Training":[],"Validation":[],"Test":[]}
        self.accuracy = {"Training":[],"Validation":[],"Test":[]}
        self.weights = []
        self.biases = []
        self.P = {"Training":None,"Validation":None}

    def build_layers(self, data_dim, nClasses, hidden_dim,act_func):

        n_layers = len(hidden_dim)

        self.add_layer(FCLayer(input_size=hidden_dim[0],output_size=data_dim))
        self.add_layer(ActLayer(act_func))

        for i in range(1,n_layers-1):
            self.add_layer(FCLayer(input_size=hidden_dim[i],output_size=hidden_dim[i-1]))
            self.add_layer(ActLayer(act_func))

        self.add_layer(FCLayer(input_size=nClasses,output_size=hidden_dim[-1]))
    

    def add_layer(self, layer):
        layerIdx = len(self.layers)
        layer.layerIdx = layerIdx

        self.layers.append(layer)
        
        if type(layer) == FCLayer:
            self.weights.append(layer.W)
            self.biases.append(layer.b)
    
    def set_loss(self, loss_func, loss_prime_func):
        self.loss_func = loss_func
        self.loss_prime_func = loss_prime_func

    def set_cost(self, cost_func):
        self.cost_func = cost_func

    def forward_prop(self, input_data,key):
        """ Runs data through network and softmax 
            (should) returns matrix P of dimension k x N """
        self.input = input_data

        output = input_data
        
        for layer in self.layers:
            output = layer.forward_pass(output)
        
        output = softMax(output)

        self.P[key] = output

    def backward_prop(self, input_labels, eta):

        input_error = self.loss_prime_func(input_labels, self.P["Training"])
        #print("Initial input (error) input to last layer: ", input_error)
        for i,layer in enumerate(reversed(self.layers)):
            input_error = layer.backward_pass(input_error,eta)
            #print("Output from layer %d (input to %d):" % (i, i-1))
            #print(input_error)
            
    def fit(self, X, Y, y, recPerEp):
        """ X = [Xtrain, Xval] 
            Y = [Ytrain, Yval]"""

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

        index = np.array(range(N))
        t = 0

        for epoch in range(n_epochs):
            print("Epoch %d of %d" % (epoch,n_epochs))
            # Shuffle batches in each epoch
            random.shuffle(index)
            Xtrain = Xtrain[:,index]
            Ytrain = Ytrain[:,index]
            ytrain = ytrain[index]

            epoch_steps = steps_per_ep if (((tot_steps - t) // steps_per_ep) > 0 ) else (tot_steps % steps_per_ep) 

            for step in range(epoch_steps):
                t = epoch*epoch_steps + step
                step_eta = setEta(t,self.n_s,eta_min,eta_max)
                self.eta_list.append(step_eta)
                batchIdxStart = step*self.nBatch
                batchIdxEnd = batchIdxStart + self.nBatch

                tmpIdx = np.arange(batchIdxStart,batchIdxEnd)

                Xbatch = Xtrain[:,tmpIdx]
                Ybatch = Ytrain[:,tmpIdx]
                ybatch = ytrain[tmpIdx]

                if (step % rec_every) == 0:
                    self.checkpoint(Xval,Yval,yval,"Validation")
                    self.checkpoint(Xbatch,Ybatch,ybatch,"Training")

                self.forward_prop(Xbatch,"Training")
                self.backward_prop(Ybatch,step_eta)


    def checkpoint(self,X,Y,y,key):
        self.forward_prop(X,key)
        self.compute_loss(Y,key)
        self.compute_cost(key)
        self.compute_accuracy(X,y,key)

    def compute_loss(self, Y, key):
        
        l = self.loss_func(Y,self.P[key],self.nBatch)

        self.loss[key].append(l)
    
    def compute_cost(self, key):
        
        J = self.cost_func(self,self.loss[key][-1],self.lamda)
        
        self.cost[key].append(J)
    
    def compute_accuracy(self, X, y, key):
        N = np.shape(X)[1]

        self.forward_prop(X,key)
        
        guess = np.argmax(self.P[key],axis=0)
        n_correct = sum(guess == y)
        
        self.accuracy[key].append(n_correct/N)

    

    
    

