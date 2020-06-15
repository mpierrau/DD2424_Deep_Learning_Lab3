""" Training 2-layer network with best lambda for 2 cycles to see accuracy """


import numpy as np
from data_handling import loadAllFiles2 , loadTestFiles , reduceDims
from K_NN_funcs import  relu , cross_entropy , cross_entropy_prime , L2_cost , normal_init
from K_NN_network_class import Network
import matplotlib.pyplot as plt
import plot_funcs
import tqdm
from importlib import reload
reload(plot_funcs)

Xin , Yin , yin = loadAllFiles2(valSize=5000)
test_data = loadTestFiles()

d , N = np.shape(Xin[0])
k = np.shape(Yin[0])[0]

dims = [50,50]

# Parameters

cycles = 2
nBatch = 100
n_s = 5*N/nBatch
eta_min = 1e-5
eta_max = 1e-1
lamda = .005
rec = 5
sigmas = [1e-1,1e-3,1e-5]

cost = {"Norm":{},"Unnorm":{}}
acc = {"Norm":{},"Unnorm":{}}

for i,sig in tqdm.tqdm(enumerate(sigmas)):
    W1 = normal_init(d,dims[0],0,sig)
    W2 = normal_init(dims[0],dims[1],0,sig)
    W3 = normal_init(dims[1],k,0,sig)

    W = [W1,W2,W3]


    # Fit model with normalization with specified sigma
    normNet = Network(normalize=True)

    normNet.build_layers(d,k,dims,W=W)

    normNet.fit(Xin, Yin, yin, n_cycles = cycles, n_s = n_s, nBatch = nBatch, eta = [eta_min, eta_max], lamda = lamda, recPerEp = rec)

    plot_funcs.plotCost(   net = normNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Norm_cost.png" % sig, 
                foldername = "Imgs/Sigma_sens/")
    plot_funcs.plotAcc(    net = normNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Norm_acc.png" % sig, 
                foldername = "Imgs/Sigma_sens/")

    cost["Norm"][i] = normNet.cost["Validation"]
    acc["Norm"][i] = normNet.accuracy["Validation"]

    # Fit model without normalization with specified sigma
    nonNormNet = Network(normalize=False)

    nonNormNet.build_layers(d,k,dims,W=W)

    nonNormNet.fit(Xin,Yin,yin,n_cycles=cycles,n_s=n_s,nBatch=nBatch,eta=[eta_min,eta_max],lamda=lamda,recPerEp=rec)

    plot_funcs.plotCost(   net = nonNormNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Unnorm_cost.png" % sig, 
                foldername = "Imgs/Sigma_sens/",
                title = "Cost for %d-layer network with hidden dimensions %s \n without BN" % (len(nonNormNet.layer_dims) + 1 , nonNormNet.layer_dims))
    plot_funcs.plotAcc(    net = nonNormNet, keys = ["Training","Validation"], 
                savename = "Lab3_Sensitivity_%f_Unnorm_acc.png" % sig, 
                foldername = "Imgs/Sigma_sens/",
                title = "Accuracy for %d-layer network with hidden dimensions %s \n without BN" % (len(nonNormNet.layer_dims) + 1 , nonNormNet.layer_dims))

    cost["Unnorm"][i] = nonNormNet.cost["Validation"]
    acc["Unnorm"][i] = nonNormNet.accuracy["Validation"]

for key in cost.keys():
    print(key)
    for layer in cost[key].keys():
        print(layer)
        if type(cost[key][layer]) is list:
            cost[key][layer] = cost[key][layer][-1]
        if type(acc[key][layer]) is list:
            acc[key][layer] = acc[key][layer][-1]

"""
 acc
{'Norm': {0: 0.5106, 1: 0.5408, 2: 0.5442}, 'Unnorm': {0: 0.4908, 1: 0.3354, 2: 0.0958}}
 cost
{'Norm': {0: 1.38699839517635, 1: 1.3222448158226383, 2: 1.325424823186113}, 'Unnorm': {0: 1.442628096039757, 1: 1.8195557548677783, 2: 2.3027599857585073}} 
"""