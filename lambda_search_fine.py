""" 2. Do a fine search around the best value from coarse search"""

import numpy as np
from lambda_search_funcs import lambdaSearch
from data_handling import loadAllFiles2 , loadTestFiles

Xin , Yin , yin = loadAllFiles2(valSize=5000)

N = np.shape(Xin[0])[1]
nBatch = 100
dims = [50,50]
cycles = 2
n_s = 2*np.floor(N/nBatch)
eta = [1e-5,1e-1]
nLambda = 5

lambdaMinFine = 2e-3
lambdaMaxFine = 3e-3

lambdaSearch(   Xin, Yin, yin, dims, cycles, n_s, nBatch, eta, lambdaMinFine, lambdaMaxFine, nLambda, 
                randomLambda=False, logScale=False, recPerEp = 50, fileName = "lab3_fine_lambda_search_nonrandom")
