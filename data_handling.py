import numpy as np
import pickle


""" Data processing / helpers """

def loadBatch(filename):
    with open('Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    X = np.array(dict['data'].reshape((len(dict['data']),32*32*3)),dtype=int).T
    y = np.array(dict['labels'])
    K = 10
    n = len(y)
    Y = np.zeros((K,n))
    for i in range(n):
            Y[y[i],i] = 1 
    return X, Y, y

def loadFiles(fileNameTraining,fileNameValidation,fileNameTest):
    X1,Y1,y1 = loadBatch(fileNameTraining)
    training_data = list([X1,Y1,y1])

    X2,Y2,y2 = loadBatch(fileNameValidation)
    validation_data = list([X2, Y2, y2])

    X3,Y3,y3 = loadBatch(fileNameTest)
    test_data = list([X3, Y3, y3])
    
    return training_data, validation_data, test_data

def loadPreProcData(fileNameTraining,fileNameValidation,fileNameTest):
    training_data , validation_data , test_data = loadFiles(fileNameTraining,fileNameValidation,fileNameTest)

    X_train, Y_train, y_train = training_data
    X_val, Y_val, y_val = validation_data
    X_test, Y_test, y_test = test_data

    # Centers data around 0 with standard deviance 1
    X_train = preProc(X_train)
    X_val = preProc(X_val)
    X_test = preProc(X_test)

    X = [X_train, X_val, X_test]
    Y = [Y_train, Y_val, Y_test]
    y = [y_train, y_val, y_test]

    return X , Y , y

def loadAllFiles(valSize):
    X1,Y1,y1 = loadBatch('data_batch_1')

    X2,Y2,y2 = loadBatch('data_batch_2')

    X3,Y3,y3 = loadBatch('data_batch_3')

    X4,Y4,y4 = loadBatch('data_batch_4')

    X5,Y5,y5 = loadBatch('data_batch_5')

    N5 = np.shape(X5)[1]
    
    X5new = X5[:,:N5-valSize]
    Y5new = Y5[:,:N5-valSize]
    y5new = y5[:N5-valSize]

    Xval = X5[:,N5-valSize:N5]
    Yval = Y5[:,N5-valSize:N5]
    yval = y5[N5-valSize:N5]

    X = np.concatenate((X1,X2,X3,X4,X5new),axis=1)
    Y = np.concatenate((Y1,Y2,Y3,Y4,Y5new),axis=1)
    y = np.concatenate((y1,y2,y3,y4,y5new))
    
    XProc = preProc(X)
    XvalProc = preProc(Xval)

    training_data = list([XProc,Y,y])
    validation_data = list([XvalProc,Yval,yval])

    return training_data , validation_data


def loadTestFiles():
    Xtest , Ytest , ytest = loadBatch('test_batch')

    XTestProc = preProc(Xtest)

    test_data = list([XTestProc,Ytest,ytest])
    
    return test_data


def preProc(X):
    #from sklearn import preprocessing
    #X = preprocessing.normalize(X)
    X = X/255
    #Xmean = np.mean(X,1)
    #Xstd = np.std(X,1)
    #X = X - Xmean[:,None]
    #X = X / Xstd[:,None]
    #X = X + np.abs(np.min(X,1))[:,None]
    #X = X / np.max(X,1)[:,None]"""
    return X

def reduceDims(X,Y,y,redDim,redN):
    XbatchRedDim = X[:redDim,:redN]
    YbatchRedDim = Y[:,:redN]
    ybatchRedDim = y[:redN]
    return XbatchRedDim , YbatchRedDim , ybatchRedDim
