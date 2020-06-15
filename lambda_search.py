""" 1. Do a coarse grid search for good lambda parameter values """

import numpy as np
from lambda_search_funcs import lambdaSearch
from data_handling import loadAllFiles2 , loadTestFiles

Xin , Yin , yin = loadAllFiles2(valSize=5000)

N = np.shape(Xin[0])[1]
nBatch = 100
dims = [50,50]
cycles = 1
n_s = 2*np.floor(N/nBatch)
eta = [1e-5,1e-1]
lambdaMin = 1e-4
lambdaMax = 1e-2
nLambda = 10

lambdaSearch(   Xin, Yin, yin, dims, cycles, n_s, nBatch, eta, lambdaMin, lambdaMax, nLambda, 
                randomLambda=False, logScale=False, recPerEp = 50, fileName = "lab3_coarse_lambda_search3",seed=1337)
