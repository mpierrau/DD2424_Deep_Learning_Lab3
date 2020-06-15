""" 1. Do a coarse grid search for good lambda parameter values 
    2. Do a fine search around the best value from coarse search"""

import numpy as np
from lambda_search_funcs import lambdaSearch
from data_handling import loadAllFiles , loadTestFiles

training_data , validation_data = loadAllFiles(valSize=5000)
test_data = loadTestFiles()

Xtrain = training_data[0]
Ytrain = training_data[1]
ytrain = training_data[2]

Xval = validation_data[0]
Yval = validation_data[1]
yval = validation_data[2]

Xtest = test_data[0]
Ytest = test_data[1]
ytest = test_data[2]

Xin = [Xtrain,Xval]
Yin = [Ytrain,Yval]
yin = [ytrain,yval]

N = np.shape(Xtrain)[1]
nBatch = 100
dims = [50,50]
cycles = 2
n_s = 2*np.floor(N/nBatch)
eta = [1e-5,1e-1]
lambdaMin = -5
lambdaMax = -1
nLambda = 10

lambdaSearch(   Xin, Yin, yin, dims, cycles, n_s, nBatch, eta, lambdaMin, lambdaMax, nLambda, 
                randomLambda=False, recPerEp = 50, fileName = "lab3_coardse_lambda_search")

