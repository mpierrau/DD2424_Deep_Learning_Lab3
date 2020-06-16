
import matplotlib.pyplot as plt
import numpy as np

""" Plotting functions """
def montage(W,savename,foldername):
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[5*i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    fig.suptitle('Visualization of $W$')
    plt.savefig(foldername + savename)
    fig.clf()

def plotCost(net,keys,savename,foldername,size=(7,5),showPlots=False,savePlots=True,title=None):
    
    plt.figure(figsize=size)
    
    if title is None:
        with_out = "with" if net.normalize else "without"
        title = "Cost for %d-layer network with hidden dimensions \n %s %s BN" % (len(net.layer_dims) + 1 , net.layer_dims , with_out)

    for key in keys:
        steps = np.linspace(0,2*net.n_s*net.n_cycles,len(net.cost[key]))
        plt.plot(steps,net.cost[key],label=key)
    plt.xlabel("Steps")
    plt.ylabel("Cost")
    plt.title(title)
    plt.legend()
    
    if savePlots:
        plt.savefig(foldername + savename)
    if showPlots:
        plt.show()
    
    plt.clf()


def plotAcc(net,keys,savename,foldername,size=(7,5),showPlots=False,savePlots=True,title=None):

    plt.figure(figsize=size)

    if title is None:
        with_out = "with" if net.normalize else "without"
        title = "Accuracy for %d-layer network with hidden dimensions \n %s %s BN" % (len(net.layer_dims) + 1 , net.layer_dims , with_out)

    for key in keys:
        steps = np.linspace(0,2*net.n_s*net.n_cycles,len(net.accuracy[key]))
        plt.plot(steps,net.accuracy[key],label=key)
    
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    
    if savePlots:
        plt.savefig(foldername + savename)
    if showPlots:
        plt.show()

    plt.clf()


def plotAcc2(net,keys,savename,foldername,size=(7,5),showPlots=False,savePlots=True,title=None,skipSteps=1):

    plt.figure(figsize=size)

    for key in keys:
        N = len(net.accuracy[key])
        rem = N % skipSteps
        newN = int((N - rem)/skipSteps)
        steps = np.linspace(0,2*net.n_s*net.n_cycles,newN)
        newAcc = net.accuracy[key][0:N:skipSteps]
        plt.plot(steps,newAcc,label=key)
    
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    
    if savePlots:
        plt.savefig(foldername + savename)
    if showPlots:
        plt.show()

    plt.clf()

def plotCost2(net,keys,savename,foldername,size=(7,5),showPlots=False,savePlots=True,title=None,skipSteps=1):

    plt.figure(figsize=size)

    for key in keys:
        N = len(net.cost[key])
        rem = N % skipSteps
        newN = int((N - rem)/skipSteps)
        steps = np.linspace(0,2*net.n_s*net.n_cycles,newN)
        newAcc = net.cost[key][0:N:skipSteps]
        plt.plot(steps,newAcc,label=key)
    
    plt.xlabel("Steps")
    plt.ylabel("Cost")
    plt.title(title)
    plt.legend()
    
    if savePlots:
        plt.savefig(foldername + savename)
    if showPlots:
        plt.show()

    plt.clf()


def custPlot(data1,data2,title,xlab,ylab,n_s,cycles):

    steps1 = np.linspace(0,2*n_s*cycles,len(data1))
    steps2 = np.linspace(0,2*n_s*cycles,len(data2))

    plt.plot(steps1,data1)
    plt.plot(steps2,data2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()

    return plt