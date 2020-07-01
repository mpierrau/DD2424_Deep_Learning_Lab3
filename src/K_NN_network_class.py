""" Network Class for a neural network.
    
    Contents:
    
    build_layers = method to build network given parameters
    
    add_layers = helper function for build_layers. Can be used to manually build layer.
    
    forward_prop = method to propagate data forward through all layers and produces a final P matrix used for prediction.
    
    backward_prop = method to propagate gradient w.r.t. cost backwards through all layers to update parameters. 
    
    fit = method to train the network on some data given specific parameters. 
    
    
    Plus some surplus methods to compute, save or extract various metrics and parameters. """

import numpy as np
import copy , winsound , csv , os
from random import Random
from tqdm import trange 
from datetime import datetime

from K_NN_funcs import setEta , softMax , he_init , relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_layer_class import FCLayer , ActLayer


class Network:
    """ Network Class:
    
        act_func        = activation function. default : ReLu
                        - type : function(X)
                            - type(X) : array
                    
        loss_func       = loss function. default : Cross Entropy 
                        - type : function(Y,P)
                            - type(Y) : array
                            - type(P) : array
                            - dim(Y) == dim(P)
        
        loss_prime_func = gradient of act_func. default : Cross Entropy derivative
                        - type : function(Y,P)
                            - type(Y) : array
                            - type(P) : array
                            - dim(Y) == dim(P)
                            
        cost_func       = cost function. default : L2 regularization cost using self.lamda.
                        - type : function(net,loss)
                            - type(net) : Network class object
                            - type(loss) : float
                    
        init_func       = function for parameter initialization. default : He Initialization.
                        - type : function(inDim,outDim)
                            - type(inDim) : int
                            - type(outDim) : out
                            # Produces matrix with dims (nRrows = outDim, nCols = inDim), which might be a bit counterintuitive.
                    
        normalize       = option to use batch normalization or not. Once initialized cannot be changed easily.
                        - type : boolean 
    """
                

    def __init__(self, act_func=relu, loss_func=cross_entropy, loss_prime_func=cross_entropy_prime, cost_func=L2_cost, init_func=he_init, normalize=True):
        
        self.lamda = None
        self.eta = None
        self.nCycles = None
        self.ns = None
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

        self.cost = {"Training": [],"Validation": [],"Test": []}
        self.loss = {k: [] for k in self.cost.keys()}
        self.accuracy = {k: [] for k in self.cost.keys()}

        self.P = {"Training":None, "Validation":None}


    def build_layers(self, data_dim, nClasses, hidden_dims, lamda, W=None, b=None, verbose=True, par_seed=None, alpha=0.9):
        """ Builds a network with layers of dims [(hidden_dims[0], data_dim), (hidden_dims[1], hidden_dims[0]), ... , (hidden_dims[k-1], hidden_dims[k])].
            Each FCLayer is followed by an ActLayer, except the last layer. 
            
            data_dim    = dimensionality of data (rows in X) 
                        - type : int > 0
                        
            nClasses    = number of classes in dataset (for CIFAR-10 nClasses=10)
                        - type : int > 0
                        
            hidden_dims  = dimensionality of hidden layers
                        - type : list or array of ints > 0
                        
            lamda       = regularization term for cost function
                        - type : float > 0
                    
            W           = option to manually set W parameter for each layer
                        - type : list of matrices with appropriate dimensions
                
            b           = option to manually set b parameters for each layer
                        - type : list of column vectors with appropriate dimensions
                
            verbose     = print stuff along the way or not
                        - type : boolean
                    
            par_seed    = random seed for parameter initialization for replicability
                        - type : float
                        
            alpha       = value for weighted averages in batch normalization
                        - type : float 0 < alpha < 1"""

        n_layers = len(hidden_dims)

        self.alpha = alpha
        self.layer_dims = hidden_dims
        self.lamda = lamda

        if W is None:
            W = []
            for i in range(n_layers):
                W.append(None)

        if b is None:
            b = []
            for i in range(n_layers):
                b.append(None)
        
        if verbose:
            print("\n{0}\n".format(20*"-"))
            print("-- Building Network with parameters --\n")
            print("- Input dimension : %d \n- Number of classes : %d\n" % (data_dim,nClasses))
            print("- Number of hidden layers: %d" % (len(hidden_dims)))
            print("\t- Dims: ", end='')
            for dim in hidden_dims:
                print(dim, end=" ")
            print("\n{0}\n".format(20*"-"))


        # Adding layers to network according to [dims]

        self.add_layer( FCLayer(input_size=data_dim, 
                                output_size=hidden_dims[0],
                                init_func=self.init_func, 
                                lamda=self.lamda, 
                                W=W[0], b=b[0], 
                                seed=par_seed, 
                                normalize=self.normalize, 
                                alpha=self.alpha), 
                        verbose=verbose)

        self.add_layer( ActLayer(self.act_func), 
                        verbose=verbose)

        for i in range(1,n_layers):
            self.add_layer( FCLayer(input_size=hidden_dims[i-1],
                                    output_size=hidden_dims[i],
                                    init_func=self.init_func,
                                    lamda=self.lamda,
                                    W=W[i], b=b[i],
                                    seed=par_seed,
                                    normalize=self.normalize,
                                    alpha=self.alpha),
                            verbose=verbose)

            self.add_layer( ActLayer(self.act_func),
                            verbose=verbose)

        self.add_layer( FCLayer(input_size=hidden_dims[-1],
                                output_size=nClasses,
                                init_func=self.init_func,
                                lamda=self.lamda,
                                W=W[-1], b=b[-1],
                                seed=par_seed,
                                normalize=False,
                                alpha=self.alpha),
                        verbose=verbose)
    
        if verbose:
            print("\n{0}DONE{0}\n".format(8*"-"))

    
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
            


    def fit(self, X, Y, y, nCycles, ns, nBatch, eta, recEvery, shuffleSeed=None, doWriteToFile=True, soundOnFinish=False):
        """ X               = input data 
                            - type : list of [training data X, validation data X]

            Y               = input data labels as one hot encoding matrix
                            - type : list of [training data Y, validation data Y]

            y               = input data labels as vector of labels
                            - type : list of [training data y, validation data y] 
                
            nCycles        = number of cycles to train for
                            - type : int > 0
                        
            ns             = length of one half cycle (one cycle = 2*ns)
                            - type : int > 0

            nBatch          = size of one batch
                            - type : int > 0 > N

            eta             = learning rate. Must be list of [etaMin , etaMax], which eta will vary linearly between cyclically with each cycle. If etaMin = etaMax , eta will be same throughout training. 
                            - type : list[float,float]

            recEvery        = record loss, cost and accuracy for training and validation set every recEvery step. The lower this number, the longer training will take.
                            - type : int >= 0

            shuffleSeed     = random seed for shuffling order of batches, for replicability
                            - type : float

            doWriteToFile   = option to save loss, accuracy and cost to a csv file at end of training.
                            - type : boolean

            soundOnFinish   = plays sound when training is complete (only works on Windows OS).
                            - type : boolean
            
            """
        
        shuffleRand = Random(shuffleSeed)

        self.nCycles = int(nCycles)
        self.ns = int(ns)
        self.nBatch = int(nBatch)
        self.eta = eta
        self.recEvery = int(recEvery)
        
        Xtrain , Xval = X
        Ytrain , Yval = Y
        ytrain , yval = y

        etaMin , etaMax = self.eta
        totSteps = 2 * self.nCycles * self.ns

        N = np.shape(Xtrain)[1]

        stepsPerEp = int(N / self.nBatch)
        nEps = int(np.ceil(totSteps / stepsPerEp))

        index = np.arange(N)


        # Begin training
        t = 0
    
        for _ in trange(nEps,leave=False):
            
            # Shuffle batches in each epoch
            shuffleRand.shuffle(index)                       

            for step in trange(stepsPerEp, leave=False):
        
                step_eta = setEta(t, self.ns, etaMin, etaMax)

                batchIdxStart = step * self.nBatch
                batchIdxEnd = batchIdxStart + self.nBatch
                tmpIdx = index[np.arange(batchIdxStart, batchIdxEnd)]

                Xbatch = Xtrain[:, tmpIdx]
                Ybatch = Ytrain[:, tmpIdx]
                ybatch = ytrain[tmpIdx]
                
                self.forward_prop(Xbatch)
                self.backward_prop(Ybatch, step_eta)
                
                if (t % self.recEvery) == 0:
                    self.checkpoint(Xval, Yval, yval, "Validation")
                    self.checkpoint(Xbatch, Ybatch, ybatch, "Training")

                t += 1
                
        if doWriteToFile:
            self.save_metrics()

        if soundOnFinish:  
            winsound.MessageBeep()



    def compute_loss(self, Y, key="Training"):
        
        l = self.loss_func(Y, self.P[key])

        self.loss[key].append(l)
    
    
    def compute_cost(self, key="Training"):
        
        J = self.cost_func(self,self.loss[key][-1])
        
        self.cost[key].append(J)
    

    def compute_accuracy(self, X, y, key="Training"):
        N = len(y)

        guess = self.predict(X, y, key)

        n_correct = sum(guess == y)
        
        self.accuracy[key].append(n_correct/N)
                
    
    def predict(self, X, y, key):
        self.forward_prop(X, key, prediction=True)
        prediction = np.argmax(self.P[key], axis=0)

        return prediction
    

    def checkpoint(self, X, Y, y, key):
        self.forward_prop(X, key, prediction=True)
        self.compute_loss(Y, key)
        self.compute_cost(key)
        self.compute_accuracy(X, y, key)

    
    def save_metrics(self):
        """ Saves loss, cost and accuracy, as well as all parameters to npz files. 
            When extracting dicts as npz files use .item() to recover values. """

        now = datetime.now()

        fileName = os.path.join("savefiles",now.strftime("%d%m%y%H%M%S"))
        
        np.savez(fileName + '_metrics',loss=self.loss,cost=self.cost,accuracy=self.accuracy,allow_pickle=True)

        np.savez(fileName +  '_pars', pars=self.get_pars())

        print("Metrics and parameters saved in {0}".format(fileName))


    def get_pars(self, keys = ["W", "b", "gamma", "beta"]):
        """ Returns dictionary of pars. Each dictionary value is a dictionary with layer indices as keys. """
        pars = {k : {} for k in keys}

        for k in keys:
            for layer in self.layers:
                if type(layer) == FCLayer:
                    pars[k][layer.layerIdx] = layer.pars[k]
        
        return pars
   
 
 