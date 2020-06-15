import numpy as np
from K_NN_network_class import Network
import csv
import tqdm

def lambdaSearch(X, Y, y, dims, cycles, n_s, nBatch, eta, lambdaMin, lambdaMax, nLambda, randomLambda, logScale, recPerEp, fileName,seed=None):
    # Do a coarse search on nLambda random lambda values between lambdaMin and lambdaMax
    # to find ballbark of good lambdas for our model.
    
    fileName = fileName + '.csv'

    if randomLambda:
        l = lambdaMin + (lambdaMax - lambdaMin)*np.random.rand(nLambda)
    else:
        l = np.linspace(lambdaMin,lambdaMax,nLambda)

    if logScale:
        lamda = 10**l
    else:
        lamda = l

    print("Lambdas generated: ", lamda)

    f = open(fileName,"w")
    
    header = ['Lambda','Index','Accuracy','Cost','Loss',"EtaMin","EtaMax","Cycles","n_s","Batchsize"]
    
    with f:
        writer = csv.writer(f)
        writer.writerow(header)

    f.close()



    d = X[0].shape[0]
    k = Y[0].shape[0]
    
    for tmp_lamda in tqdm.tqdm(lamda):

        net = Network()
        net.build_layers(d,k,dims,verbose=False)
        net.fit(X,Y,y,cycles,n_s,nBatch,eta,tmp_lamda,recPerEp,seed=seed)

        valAcc = net.accuracy["Validation"]
        valCost = net.cost["Validation"]
        valLoss = net.loss["Validation"]

        bestIdx = np.argmax(valAcc)

        eta_min , eta_max = eta

        pars = [tmp_lamda,bestIdx,valAcc[bestIdx],valCost[bestIdx],valLoss[bestIdx],eta_min,eta_max,cycles,n_s,nBatch]

        f = open(fileName,"a+")

        with f:
            writer = csv.writer(f)
            writer.writerow(pars)

        f.close()