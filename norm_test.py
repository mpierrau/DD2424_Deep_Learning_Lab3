
import numpy as np
from data_handling import loadPreProcData , loadBatch , loadAllFiles , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost
from K_NN_network_class import Network
import matplotlib.pyplot as plt

X,Y,y = loadPreProcData("data_batch_1","data_batch_2","data_batch_3")

Xtrain = X[0]
Ytrain = Y[0]
ytrain = y[0]

Xval = X[1]
Yval = Y[1]
yval = y[1]

Xtest = X[2]
Ytest = Y[2]
ytest = y[2]

Xin = [Xtrain,Xval]
Yin = [Ytrain,Yval]
yin = [ytrain,yval]

d , N = Xtrain.shape
k = np.shape(Ytrain)[0]

dims = [50,20]

net = Network(normalize=False)
net.build_layers(d,k,dims)
net.fit(Xin,Yin,yin,2,50,100,[1e-5,1e-1],.001,10,seed=1337)


netNorm = Network(normalize=True)
netNorm.build_layers(d,k,dims)
netNorm.fit(Xin,Yin,yin,2,50,100,[1e-5,1e-1],.001,10,seed=1337)


#net.layers[0].norm_score
#net.layers[0].unnorm_score

net.compute_accuracy(Xtest,ytest,key="Test")
net.accuracy["Test"]


netNorm.compute_accuracy(Xtest,ytest,key="Test")
netNorm.accuracy["Test"]

numbers = [[1, 2, 3],[4,5,6]]
letters = [["A", "B", "C"],["D","E","F"]]

for numbers_list, letters_list in zip(numbers, letters):
    for numbers_value , letters_value in zip(numbers_list, letters_list):
        print(numbers_value, letters_value)