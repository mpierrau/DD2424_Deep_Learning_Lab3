
import numpy as np
from numpy import random
from node_funcs import setEta
import copy
from K_NN_layer_class import FCLayer

# TODO: why does accuracy turn out only 0 or 0.001 ? All predictions seem to be 0.1 .

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.cost = None
        self.accuracy = []
        self.weights = []
        self.biases = []

    def add_layer(self, layer):
        self.layers.append(layer)
        if type(layer) == FCLayer:
            self.weights.append(layer.weights)
            self.biases.append(layer.bias)
        
    def set_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def set_cost(self, cost):
        self.cost = cost

    def forward_prop(self, input_data):
        output = input_data

        for layer in self.layers:
            output = layer.forward_pass(output)
        
        output = soft_max(output)

        return output

    def backward_prop(self, P, eta):
        input_error = P
        for layer in reversed(self.layers):
            input_error = layer.backward_pass(input_error,eta)

    def fit(self, Xtrain, Ytrain, ytrain, n_batch, n_cycles, n_s, eta, rec_every, lamda):
        
        self.n_cycles = n_cycles
        eta_min = eta[0]
        eta_max = eta[1]
        self.J = []
        self.l = []
        tot_steps = n_cycles*2*n_s
        N = np.shape(Xtrain)[1]
        steps_per_ep = int(N/n_batch)
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
                
                batchIdxStart = step*n_batch
                batchIdxEnd = batchIdxStart + n_batch

                tmpidx = np.arange(batchIdxStart,batchIdxEnd)

                Xbatch = Xtrain[:,tmpidx]
                Ybatch = Ytrain[:,tmpidx]

                P = self.forward_prop(Xbatch)

                if (step % rec_every) == 0:
                    step_loss = self.loss(Ybatch,P,n_batch)
                    self.l.append(step_loss)
                    self.J.append(self.cost(step_loss,self.layers,lamda))

                error = self.loss_prime(Ybatch, P)

                self.backward_prop(error,step_eta)

    
    def compute_accuracy(self, X, y):
        tmp_accuracy = []
        
        for i in range(len(X)):
            N = np.shape(X[i])[1]
            tmpX = X[i]
            tmpy = y[i]
            
            P = self.forward_prop(tmpX)
            
            guess = np.argmax(P,axis=0)
            n_correct = sum(guess == tmpy)
            
            tmp_accuracy.append(n_correct/N)
            
        self.accuracy.append(tmp_accuracy)

    def compute_cost(self, X, Y, lamda, nBatch):

        P = self.forward_prop(X)
        l = self.loss(Y,P,nBatch)
        J = self.cost(l,lamda,self)

        return J
    
    def computeGradsNum(self, X, Y, lamda, h=1e-5, nBatch=100):
        # Here X is Xtrain (batch)
        
        grads_W = list()
        grads_b = list()
        
        print("Computing initial cost:")

        c = self.compute_cost(X, Y, lamda, nBatch)

        test_net = Network()
        FCidx = []

        k = 0
        for layer in self.layers:
            if type(layer) == FCLayer:
                tmpW = copy.deepcopy(layer.weights)
                tmpb = copy.deepcopy(layer.bias)

                test_net.add_layer(FCLayer(layer.N, layer.m, layer.mu, layer.sig, layer.lamda, tmpW, tmpb))
                FCidx.append(k)
            else:
                test_net.add_layer(layer)
        
            k += 1
        
        test_net.set_cost(self.cost)
        test_net.set_loss(self.loss,self.loss_prime)

        for j in range(len(self.biases)):
            grads_b.append(np.zeros(len(self.biases[j])))
            
            for i in range(len(self.biases[j])):
                test_net.layers[FCidx[j]].bias[i] += h
                
                c2 = test_net.compute_cost(X,Y,lamda,nBatch)

                grads_b[j][i] = (c2-c) / h

                #reset entries for next pass
                test_net.layers[FCidx[j]].bias[i] -= h
        
        for k in range(len(self.weights)):
            print("computing grad for W[%d]" % k)

            grads_W.append(np.zeros(np.shape(self.weights[k])))

            for i in range(np.shape(grads_W[k])[0]):
                for j in range(np.shape(grads_W[k])[1]):
                    test_net.layers[FCidx[k]].weights[i,j] += h
                    c2 = test_net.compute_cost(X, Y, lamda, nBatch)
                    grads_W[k][i,j] = (c2-c) / h
                    
                    #reset entries for next pass
                    test_net.layers[FCidx[k]].weights[i,j] -= h
        
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
                tmpW = copy.deepcopy(layer.weights)
                tmpb = copy.deepcopy(layer.bias)

                test_net.add_layer(FCLayer(layer.input_size, layer.output_size, layer.mu, layer.sig, layer.lamda, tmpW, tmpb))
                FCidx.append(k)
            else:
                test_net.add_layer(layer)
        
            k += 1
        
        test_net.set_cost(self.cost)
        test_net.set_loss(self.loss,self.loss_prime)

        for j in range(len(self.biases)):
            grads_b.append(np.zeros(len(self.biases[j])))
            
            for i in range(len(self.biases[j])):
                test_net.layers[FCidx[j]].bias[i] -= h
                
                c1 = test_net.compute_cost(X,Y,lamda,nBatch)

                test_net.layers[FCidx[j]].bias[i] += 2*h

                c2 = test_net.compute_cost(X,Y,lamda,nBatch)

                grads_b[j][i] = (c2-c1) / (2*h)

                #reset entries for next pass
                test_net.layers[FCidx[j]].bias[i] -= h
        
        for k in range(len(self.weights)):
            print("computing grad for W[%d]" % k)

            grads_W.append(np.zeros(np.shape(self.weights[k])))

            for i in range(np.shape(grads_W[k])[0]):
                for j in range(np.shape(grads_W[k])[1]):
                    test_net.layers[FCidx[k]].weights[i,j] -= h
                    c1 = test_net.compute_cost(X, Y, lamda, nBatch)

                    test_net.layers[FCidx[k]].weights[i,j] += 2*h
                    c2 = test_net.compute_cost(X, Y, lamda, nBatch)

                    grads_W[k][i,j] = (c2-c1) / (2*h)
                    
                    #reset entries for next pass
                    test_net.layers[FCidx[k]].weights[i,j] -= h
        
        return [grads_W, grads_b]
    
def soft_max(input_data):
    #Standard definition of the softmax function
    print("Softmax input_data shape: ", np.shape(input_data))
    S = np.exp(input_data) / np.sum(np.exp(input_data), axis=0)
    return S
