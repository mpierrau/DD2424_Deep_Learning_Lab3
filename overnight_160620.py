""" Running some overnight experiments """


import numpy as np
from data_handling import loadAllFiles2 , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost , normal_init
from K_NN_network_class import Network
import matplotlib.pyplot as plt
import plot_funcs
import tqdm
from lambda_search_funcs import lambdaSearch

Xin , Yin , yin = loadAllFiles2(valSize=5000)
test_data = loadTestFiles()

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

dims = [50,50]

# FIND BEST LAMBDA

cycles = 2
nBatch = 100
n_s = 5*N/nBatch
eta_min = 1e-5
eta_max = 1e-1
eta = [eta_min , eta_max]
rec = 100
lambdaMinFine = .004
lambdaMaxFine = .007
nLambda = 10

new_lamda , acc = lambdaSearch( Xin, Yin, yin, dims, cycles, n_s, nBatch, eta, lambdaMinFine, lambdaMaxFine, nLambda, 
                                randomLambda=False, logScale=False, recPerEp = rec, fileName = "overnight_very_fine_lambda_search_nonrandom_1337", seed=1337)

if acc < .5232:
    lamda = .006733
    foundNewLambda = False
else:
    lamda = new_lamda
    foundNewLambda = True

print("Found new lambda : ", foundNewLambda)
print("Using lambda : ", lamda)

# RUN 3-LAYER WITH BEST LAMBDA

print("Running 3-layer for 3 cycles with new lambda")

cycles = 3
nBatch = 100
n_s = 5*N/nBatch
eta_min = 1e-5
eta_max = 1e-1
rec = 100

net = Network(normalize=True)
net.build_layers(d,k,dims,lamda,par_seed=1337)
net.fit(Xin,Yin,yin,cycles,n_s,nBatch,[eta_min,eta_max],rec,shuffle_seed=1337,fileName="overnight_bestLambda")

# Doing sens_to_init experiment
print("Starting init experiment")

cycles = 2
nBatch = 100
n_s = 5*N/nBatch
eta_min = 1e-5
eta_max = 1e-1
lamda = .005
rec = 5
sigmas = [1e-1,1e-3,1e-5]

for i,sig in tqdm.tqdm(enumerate(sigmas)):
    W1 = normal_init(d,dims[0],0,sig)
    W2 = normal_init(dims[0],dims[1],0,sig)
    W3 = normal_init(dims[1],k,0,sig)

    W = [W1,W2,W3]


    # Fit model with normalization with specified sigma
    normNet = Network(normalize=True)

    normNet.build_layers(d,k,dims,lamda=lamda, W=W)

    normNet.fit(Xin, Yin, yin, n_cycles = cycles, n_s = n_s, nBatch = nBatch, eta = [eta_min, eta_max] , rec_every = rec, shuffle_seed = 1337, fileName="overnight_sens_to_init_%.e_BN" % sig)

    plot_funcs.plotCost(   net = normNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Norm_cost.png" % sig, 
                foldername = "Imgs/Sigma_sens/",
                title = "Cost for %d-layer network with hidden dimensions\n %s, with BN, sig=%.e" % (len(normNet.layer_dims) + 1, normNet.layer_dims, sig))
    plot_funcs.plotAcc(    net = normNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Norm_acc.png" % sig, 
                foldername = "Imgs/Sigma_sens/",
                title = "Accuracy for %d-layer network with hidden dimensions\n %s, with BN, sig=%.e" % (len(normNet.layer_dims) + 1, normNet.layer_dims, sig))

    # Fit model without normalization with specified sigma
    nonNormNet = Network(normalize=False)

    nonNormNet.build_layers(d,k,dims,lamda=lamda,W=W)

    nonNormNet.fit(Xin, Yin, yin, n_cycles = cycles, n_s = n_s, nBatch = nBatch, eta = [eta_min, eta_max] , rec_every = rec, shuffle_seed = 1337, fileName="overnight_sens_to_init_%.e_Non_BN" % sig)

    plot_funcs.plotCost(   net = nonNormNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Unnorm_cost.png" % sig, 
                foldername = "Imgs/Sigma_sens/",
                title = "Cost for %d-layer network with hidden dimensions\n %s, with BN, sig=%.e" % (len(nonNormNet.layer_dims) + 1, nonNormNet.layer_dims, sig))    

    plot_funcs.plotAcc(    net = nonNormNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Unnorm_acc.png" % sig, 
                foldername = "Imgs/Sigma_sens/",
                title = "Accuracy for %d-layer network with hidden dimensions\n %s, with BN, sig=%.e" % (len(nonNormNet.layer_dims) + 1, nonNormNet.layer_dims, sig))

print("Done with all experiments! ... That went better than expected!")