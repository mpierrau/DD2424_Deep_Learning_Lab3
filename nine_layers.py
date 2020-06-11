""" Running 9 layer NN with parameters from lab instructions. Test acc. 41.6% """

import numpy as np
from data_handling import loadPreProcData , loadBatch , loadAllFiles , loadTestFiles , reduceDims
from numerical_grads import testGrads , relErr
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost , he_init , xavier_init
from K_NN_network_class import Network
from K_NN_layer_class import FCLayer , ActLayer
import matplotlib.pyplot as plt

#X , Y , y = loadPreProcData('data_batch_1','data_batch_2','data_batch_3')

training_data , validation_data = loadAllFiles(valSize=5000)
test_data = loadTestFiles()

Xin = [training_data[0],validation_data[0]]
Yin = [training_data[1],validation_data[1]]
yin = [training_data[2],validation_data[2]]

Xtest = test_data[0]
ytest = test_data[2]

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

recPerEp = 10

nBatch = 100
cycles = 2
eta = [1e-5, 1e-1]
lamda = .005
#lamda = 1.73e-5
n_s = 5*45000/nBatch

#n_s = int(2*np.floor(N/nBatch))

layerDims = [50, 30, 20, 20, 10, 10, 10, 10]

net = Network()
net.build_layers(d,k,layerDims)

net.fit(Xin,Yin,yin,cycles,n_s,nBatch,eta,lamda,recPerEp)

steps_train = np.linspace(0,2*n_s*cycles,len(net.cost["Training"]))
steps_val = np.linspace(0,2*n_s*cycles,len(net.cost["Validation"]))
plt.plot(steps_train,net.cost["Training"],label="Training")
plt.plot(steps_val,net.cost["Validation"],label="Validation")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss for %d-layer network with hidden dimensions %s" % (len(layerDims) + 1 , layerDims))
plt.legend()
plt.show()


plt.plot(np.arange(len(net.accuracy["Training"])),net.accuracy["Training"],label="Training")
plt.plot(np.arange(len(net.accuracy["Validation"])),net.accuracy["Validation"],label="Validation")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for %d-layer network with hidden dimensions %s" % (len(layerDims) + 1 , layerDims))
plt.legend()
plt.show()

net.compute_accuracy(Xtest, ytest , key="Test")
print(net.accuracy["Test"])
pars = net.get_pars()