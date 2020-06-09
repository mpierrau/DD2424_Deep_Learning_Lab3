
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

def plotData(cost,labels,title,xlab,ylab,shift,savename,foldername,showPlots=False,savePlots=True):
    x = np.linspace(0,len(cost[0])*shift,len(cost[0]))
    fig , axs = plt.subplots()
    for i in range(len(cost)):
        axs.plot(x,cost[i],label=labels[i])
    axs.set_xlabel(xlab)
    axs.set_ylabel(ylab)
    axs.set_title(title)
    axs.legend()
    if savePlots:
        fig.savefig(foldername + savename)
    if showPlots:
        plt.show()

def custPlot(data1,data2,title,xlab,ylab,n_s,cycles)

    steps1 = np.linspace(0,2*n_s*cycles,len(data1))
    steps2 = np.linspace(0,2*n_s*cycles,len(data2))

    plt.plot(steps1,data1)
    plt.plot(steps2,data2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()

    return plt
