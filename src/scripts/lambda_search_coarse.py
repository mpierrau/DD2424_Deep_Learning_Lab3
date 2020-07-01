""" 1. Do a coarse grid search for good lambda parameter values """

import numpy as np
from data_handling import loadAllFiles2 , loadTestFiles
from importlib import reload
import lambda_search_funcs
#reload(lambda_search_funcs)

Xin , Yin , yin = loadAllFiles2(valSize=5000)

N = np.shape(Xin[0])[1]
nBatch = 100
dims = [50,50]
cycles = 1
n_s = 2*np.floor(N/nBatch)
eta = [1e-5,1e-1]
lambdaMin = 1e-4
lambdaMax = 1e-1
nLambda = 10

lambda_search_funcs.lambdaSearch(   Xin, Yin, yin, dims, cycles, n_s, nBatch, eta, lambdaMin, lambdaMax, nLambda, 
                randomLambda=False, logScale=False, recPerEp = 10, fileName = "lab3_coarse_lambda_search_2",seed=1337)
