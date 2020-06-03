import numpy as np
from numpy import random
from node_funcs import setEta
import copy
from K_NN_layer_class import FCLayer

# TODO: why does accuracy turn out only 0 or 0.001 ? All predictions seem to be 0.1 .

class Network:
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.loss_prime_func = None
        self.cost_func = None
        self.cost = []
        self.loss = []
        self.accuracy = []
        self.weights = []
        self.biases = []
        self.P = []

    def add_layer(self, layer):
        self.layers.append(layer)
        if type(layer) == FCLayer:
            self.weights.append(layer.W)
            self.biases.append(layer.b)
        
    def set_loss(self, loss_func, loss_prime_func):
        self.loss_func = loss_func
        self.loss_prime_func = loss_prime_func

    def set_cost(self, cost_func):
        self.cost_func = cost_func

    def forward_prop(self, input_data):
        """ Runs data through network and softmax 
            (should) returns matrix P of dimension k x N """
        
        output = input_data

        for layer in self.layers:
            output = layer.forward_pass(output)
        
        output = soft_max(output)
        self.P.append(output)

    def backward_prop(self, err, eta):
        input_error = err
        for layer in reversed(self.layers):
            input_error = layer.backward_pass(input_error,eta)

    def fit(self, Xtrain, Ytrain, ytrain, nBatch, n_cycles, n_s, eta, rec_every, lamda):
        
        self.n_cycles = n_cycles
        eta_min = eta[0]
        eta_max = eta[1]
        tot_steps = n_cycles*2*n_s
        N = np.shape(Xtrain)[1]
        steps_per_ep = int(N/nBatch)
        n_epochs = int(np.ceil(tot_steps/steps_per_ep))
        #nRec = int(np.ceil((tot_steps / rec_every)))

        index = np.array(range(N))
        t = 0

        for epoch in range(n_epochs):
            print("Epoch ", epoch)
            # Shuffle batches in each epoch
            random.shuffle(index)
            Xtrain = Xtrain[:,index]
            Ytrain = Ytrain[:,index]
            ytrain = ytrain[index]

            epoch_steps = steps_per_ep if (((tot_steps - t) // steps_per_ep) > 0 ) else (tot_steps % steps_per_ep) 

            for step in range(epoch_steps):
                t = epoch*epoch_steps + step
                step_eta = setEta(t,n_s,eta_min,eta_max)
                
                batchIdxStart = step*nBatch
                batchIdxEnd = batchIdxStart + nBatch

                tmpidx = np.arange(batchIdxStart,batchIdxEnd)

                Xbatch = Xtrain[:,tmpidx]
                Ybatch = Ytrain[:,tmpidx]

                self.forward_prop(Xbatch)
                
                if (step % rec_every) == 0:
                    tmp_loss = self.compute_loss(Ybatch, self.P[-1], nBatch)
                    tmp_cost = self.compute_cost(tmp_loss,lamda)
                    self.loss.append(tmp_loss)
                    self.cost.append(tmp_cost)

                error = self.loss_prime_func(Ybatch, self.P[-1])

                self.backward_prop(error,step_eta)

    def compute_loss(self, Y, P, nBatch):

        l = self.loss_func(Y,P,nBatch)

        self.loss.append(l)
        
        return l
    
    def compute_cost(self, loss, lamda):
        
        J = self.cost_func(self,loss,lamda)
        
        self.cost.append(J)
        
        return J
    
    def compute_accuracy(self, X, y):
        tmp_accuracy = []
        
        for i in range(len(X)):
            N = np.shape(X[i])[1]
            tmpX = X[i]
            tmpy = y[i]
            
            self.forward_prop(tmpX)
            
            guess = np.argmax(self.P[-1],axis=0)
            n_correct = sum(guess == tmpy)
            
            tmp_accuracy.append(n_correct/N)
            
        self.accuracy.append(tmp_accuracy)

    

    def computeGradsNum(self, X, Y, lamda, h=1e-5, nBatch=100):
        # Here X is Xtrain (batch)
        
        grads_W = list()
        grads_b = list()
        
        print("Computing initial cost:")
        self.forward_prop(X)
        l = self.compute_loss(Y, self.P[-1], nBatch)
        c = self.compute_cost(l,lamda)
        print("Initial cost: %s" % c)
        #print("Initial P: ", self.P[-1])
        test_net = Network()
        FCidx = []

        k = 0
        for layer in self.layers:
            if type(layer) == FCLayer:
                tmpW = copy.deepcopy(layer.W)
                tmpb = copy.deepcopy(layer.b)
                test_net.add_layer(FCLayer(layer.N, layer.m, layer.mu, layer.sig, layer.lamda, tmpW, tmpb))
                FCidx.append(k)
            else:
                test_net.add_layer(layer)
        
            k += 1
        print("FCidx: ", FCidx)
        test_net.set_cost(self.cost_func)
        test_net.set_loss(self.loss_func,self.loss_prime_func)

        for j in range(len(test_net.biases)):
            print("computing grad for b[%d]" % j)
            grads_b.append(np.zeros(len(test_net.biases[j])))
            
            for i in range(len(test_net.biases[j])):
                test_net.layers[FCidx[j]].b[i] += h
                test_net.forward_prop(X)
                l2 = test_net.compute_loss(Y,test_net.P[-1],nBatch)
                c2 = test_net.compute_cost(l2,lamda)
                #print("i = %d, FCidx[%d]=%d" % (i,j,FCidx[j]))
                #print("layer[%d].b = %s" % (FCidx[j],test_net.layers[FCidx[j]].b))
                #print("c2: ", c2)

                grads_b[j][i] = (c2-c) / h
                print("New grad_b[%d][%d] = %f" % (j,i,grads_b[j][i]))
                #reset entries for next pass
                test_net.layers[FCidx[j]].b[i] -= h
        
        for k in range(len(test_net.weights)):
            print("computing grad for W[%d]" % k)

            grads_W.append(np.zeros(np.shape(test_net.weights[k])))

            for i in range(np.shape(grads_W[k])[0]):
                for j in range(np.shape(grads_W[k])[1]):
                    test_net.layers[FCidx[k]].W[i,j] += h
                    
                    test_net.forward_prop(X)
                    l2 = test_net.compute_loss(Y,test_net.P[-1],nBatch)
                    c2 = test_net.compute_cost(l2,lamda)

                    grads_W[k][i,j] = (c2-c) / h
                    
                    #reset entries for next pass
                    test_net.layers[FCidx[k]].W[i,j] -= h
        
        return [grads_W, grads_b]

    def computeGradsNumSlow(self, X, Y, lamda, h=1e-5, nBatch=100):
        # Here X is Xtrain (batch)
        
        grads_W = list()
        grads_b = list()
        
        test_net = Network()
        FCidx = []

        k = 0
        for layer in self.layers:
            if type(layer) == FCLayer:
                tmpW = copy.deepcopy(layer.W)
                tmpb = copy.deepcopy(layer.b)

                test_net.add_layer(FCLayer(layer.input_size, layer.output_size, layer.mu, layer.sig, layer.lamda, tmpW, tmpb))
                FCidx.append(k)
            else:
                test_net.add_layer(layer)
        
            k += 1
        
        test_net.set_cost(self.cost_func)
        test_net.set_loss(self.loss_func,self.loss_prime_func)

        for j in range(len(self.biases)):
            grads_b.append(np.zeros(len(self.biases[j])))
            
            for i in range(len(self.biases[j])):
                test_net.layers[FCidx[j]].b[i] -= h
                
                test_net.forward_prop(X)
                l1 = test_net.compute_loss(Y,self.P[-1],nBatch)
                c1 = test_net.compute_cost(l1,lamda)

                test_net.layers[FCidx[j]].b[i] += 2*h
                
                test_net.forward_prop(X)
                l2 = test_net.compute_loss(Y,self.P[-1],nBatch)
                c2 = test_net.compute_cost(l2,lamda)

                grads_b[j][i] = (c2-c1) / (2*h)

                #reset entries for next pass
                test_net.layers[FCidx[j]].b[i] -= h
        
        for k in range(len(self.weights)):
            print("computing grad for W[%d]" % k)

            grads_W.append(np.zeros(np.shape(self.weights[k])))

            for i in range(np.shape(grads_W[k])[0]):
                for j in range(np.shape(grads_W[k])[1]):
                    test_net.layers[FCidx[k]].W[i,j] -= h
                    
                    test_net.forward_prop(X)
                    l1 = test_net.compute_loss(Y,self.P[-1],nBatch)
                    c1 = test_net.compute_cost(l1,lamda)

                    test_net.layers[FCidx[k]].W[i,j] += 2*h
                    
                    test_net.forward_prop(X)
                    l2 = test_net.compute_loss(Y,self.P[-1],nBatch)
                    c2 = test_net.compute_cost(l2,lamda)

                    grads_W[k][i,j] = (c2-c1) / (2*h)
                    
                    #reset entries for next pass
                    test_net.layers[FCidx[k]].W[i,j] -= h
        
        return [grads_W, grads_b]
    
def soft_max(input_data):
    #Standard definition of the softmax function
    S = np.exp(input_data) / np.sum(np.exp(input_data), axis=0)
    return S
