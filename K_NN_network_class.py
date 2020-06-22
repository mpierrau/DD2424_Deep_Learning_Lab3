""" Network Class for a neural network.
    
    Contents:
    
    build_layers = method to build network given parameters
    
    add_layers = helper function for build_layers. Can be used to manually build layer.
    
    forward_prop = method to propagate data forward through all layers and produces a final P matrix used for prediction.
    
    backward_prop = method to propagate gradient w.r.t. cost backwards through all layers to update parameters. 
    
    fit = method to train the network on some data given specific parameters. 
    
    
    Plus some surplus methods to compute, save or extract various metrics and parameters. """

import numpy as np
from random import Random
from K_NN_funcs import setEta , softMax , he_init , relu , cross_entropy , cross_entropy_prime , L2_cost , write_metrics
import copy
from K_NN_layer_class import FCLayer , ActLayer
from tqdm import trange 
import winsound

class Network:
    """ Network Class:
    
        act_func    = activation function. default : ReLu
                    - type : function
                    
        loss_func   = loss function. default : Cross Entropy 
                    - type : function
        
        loss_prime_func = derivative of act_func. default : Cross Entropy derivative
                        - type : function
                            
        cost_func   = cost function. default : L2 regularization cost using self.lamda.
                    - type : function
                    
        init_func   = function for parameter initialization. default : He Initialization.
                    - type : function
                    
        normalize   = option to use batch normalization or not. Once initialized cannot be changed easily.
                    - type : boolean """
                

    def __init__(self,act_func=relu,loss_func=cross_entropy,loss_prime_func=cross_entropy_prime,cost_func=L2_cost,init_func=he_init,normalize=True):
        
        self.lamda = None
        self.eta = None
        self.n_cycles = None
        self.n_s = None
        self.nBatch = None

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

        # Saving results for different datasets in dictionaries for easy access
        self.cost = {"Training":[],"Validation":[],"Test":[]}
        self.loss = {"Training":[],"Validation":[],"Test":[]}
        self.accuracy = {"Training":[],"Validation":[],"Test":[]}

        # Since P is a large matrix we only save one value at a time.
        self.P = {"Training":None,"Validation":None}


    def build_layers(self, data_dim, nClasses, hidden_dim,lamda,W=None, b=None,verbose=True, par_seed=None, alpha=0.9):
        """ Builds a network with layers of dims [(hidden_dim[0], data_dim), (hidden_dim[1], hidden_dim[0]), ... , (hidden_dim[k-1], hidden_dim[k])].
            Each FCLayer is followed by an ActLayer, except the last layer. 
            
            data_dim    = dimensionality of data (rows in X) 
                        - type : int > 0
                        
            nClasses    = number of classes in dataset (for CIFAR-10 nClasses=10)
                        - type : int > 0
                        
            hidden_dim  = dimensionality of hidden layers
                        - type : list or array of ints > 0
                        
            lamda   = regularization term for cost function
                    - type : float > 0
                    
            W   = option to manually set W parameter for each layer
                - type : list of matrices with appropriate dimensions
                
            b   = option to manually set b parameters for each layer
                - type : list of column vectors with appropriate dimensions
                
            verbose = print stuff along the way or not
                    - type : boolean
                    
            par_seed    = random seed for parameter initialization for replicability
                        - type : float
                        
            alpha   = value for weighted averages in batch normalization
                    - type : float 0 < alpha < 1"""

        n_layers = len(hidden_dim)

        self.alpha = alpha
        self.layer_dims = hidden_dim
        self.lamda = lamda

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
                                init_func=self.init_func,lamda=self.lamda,W=W[0],b=b[0],seed=par_seed,normalize=self.normalize,alpha=self.alpha),verbose=verbose)
        self.add_layer(ActLayer(self.act_func),verbose=verbose)

        for i in range(1,n_layers):
            self.add_layer(FCLayer( input_size=hidden_dim[i-1],output_size=hidden_dim[i],
                                    init_func=self.init_func,lamda=self.lamda,W=W[i],b=b[i],seed=par_seed,normalize=self.normalize,alpha=self.alpha),verbose=verbose)
            self.add_layer(ActLayer(self.act_func),verbose=verbose)

        self.add_layer(FCLayer( input_size=hidden_dim[-1],output_size=nClasses,
                                init_func=self.init_func,lamda=self.lamda,W=W[-1],b=b[-1],seed=par_seed,normalize=False,alpha=self.alpha),verbose=verbose)
    
        if verbose:
            print("\n---------------DONE---------------\n")

    
    def add_layer(self, layer, verbose=True):
        layerIdx = len(self.layers)
        layer.layerIdx = layerIdx
        if verbose:
            print("Added layer %d : %s" % (layerIdx,layer.name))
        self.layers.append(layer)
    

    def forward_prop(self, input_data,key="Training",prediction=False):
        """ Runs data through network and softmax 
            Returns matrix P of dimension k x N """
        
        self.input = input_data
        output = input_data
        
        for layer in self.layers:
            output = layer.forward_pass(output,prediction)
            
        output = softMax(output)

        self.P[key] = output


    def backward_prop(self, input_labels, eta):
        """ Backpropagates gradient through network """
        input_error = self.loss_prime_func(input_labels, self.P["Training"])
        for layer in reversed(self.layers):
            input_error = layer.backward_pass(input_error,eta)
            


    def fit(self, X, Y, y,n_cycles,n_s,nBatch,eta,rec_every,shuffle_seed=None,write_to_file=True,fileName="tmp_vals",sound_on_finish=False):
        """ X   = input data 
                - type : list of [training data X, validation data X]

            Y   = input data labels as one hot encoding matrix
                - type : list of [training data Y, validation data Y]

            y   = input data labels as vector of labels
                - type : list of [training data y, validation data y] 
                
            n_cycles    = number of cycles to train for
                        - type : int > 0
                        
            n_s = length of one half cycle (one cycle = 2*n_s)
                - type : int > 0

            nBatch  = size of one batch
                    - type : int > 0 > N

            eta = learning rate. Must be list of [eta_min , eta_max] which eta will vary linearly between. If eta_min = eta_max , eta will be same throughout training. 
                - type : list[float,float]

            rec_every   = record loss, cost and accuracy for training and validation set every rec_every step. The lower this number, the longer training will take.
                        - type : int >= 0

            shuffle_seed    = random seed for shuffling order of batches, for replicability
                            - type : float

            write_to_file   = option to save loss, accuracy and cost to a csv file at end of training.
                            - type : boolean

            fileName    = name of savefile to be used if write_to_file == True
                        - type : string

            sound_on_finish = plays sound when training is complete (only works on Windows OS).
                            - type : boolean
            
            """
        
        shuffleRand = Random(shuffle_seed)

        self.n_cycles = int(n_cycles)
        self.n_s = int(n_s)
        self.nBatch = int(nBatch)
        self.eta = eta
        self.rec_every = int(rec_every)
        
        Xtrain , Xval = X
        Ytrain , Yval = Y
        ytrain , yval = y

        eta_min = self.eta[0]
        eta_max = self.eta[1]
        tot_steps = self.n_cycles*2*self.n_s

        N = np.shape(Xtrain)[1]

        steps_per_ep = int(N/self.nBatch)
        n_epochs = int(np.ceil(tot_steps/steps_per_ep))

        index = np.arange(N)


        # Begin training
        t = 0
        for _ in trange(n_epochs,leave=False):
            shuffleRand.shuffle(index)                  # Shuffle batches in each epoch

            for step in trange(steps_per_ep,leave=False):
                step_eta = setEta(t,self.n_s,eta_min,eta_max)

                batchIdxStart = step*self.nBatch
                batchIdxEnd = batchIdxStart + self.nBatch
                tmpIdx = index[np.arange(batchIdxStart,batchIdxEnd)]

                Xbatch = Xtrain[:,tmpIdx]
                Ybatch = Ytrain[:,tmpIdx]
                ybatch = ytrain[tmpIdx]
                
                self.forward_prop(Xbatch)
                self.backward_prop(Ybatch,step_eta)
                
                if (t % self.rec_every) == 0:
                    self.checkpoint(Xval,Yval,yval,"Validation")
                    self.checkpoint(Xbatch,Ybatch,ybatch,"Training")

                t += 1
                
        if write_to_file:
            write_metrics(self,fileName)

        if sound_on_finish:
            winsound.MessageBeep()



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
                
    
    def predict(self,X,y,key):
        self.forward_prop(X,key,prediction=True)
        prediction = np.argmax(self.P[key],axis=0)

        return prediction
    

    def checkpoint(self,X,Y,y,key):
        self.forward_prop(X,key,prediction=True)
        self.compute_loss(Y,key)
        self.compute_cost(key)
        self.compute_accuracy(X,y,key)


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
   
 
 