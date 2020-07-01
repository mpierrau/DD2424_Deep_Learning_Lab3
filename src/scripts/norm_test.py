
import numpy as np
from data_handling import loadPreProcData , reduceDims
from K_NN_network_class import Network

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

XredTr , YredTr , yredTr = reduceDims(Xtrain,Ytrain,ytrain,100,100)
XredVal , YredVal , yredVal = reduceDims(Xval,Yval,yval,100,100)


d , N = XredTr.shape
k = YredTr.shape[0]

"""
d , N = Xtrain.shape
k = np.shape(Ytrain)[0]
"""

dims = [50,20]

net = Network(normalize=True)
net.build_layers(d,k,dims,.003)
net.fit([XredTr,XredVal],[YredTr,YredVal],[yredTr,yredVal],2,50,20,[1e-5,1e-1],5,shuffle_seed=1337,write_to_file=True,fileName="testing")
