#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


def addOneToBeginningOfRow(x):
    """Adds one to the beginning of the row vector x
    Inputs: 
        x - Row Vector 
    Outputs:
        x - Row Vector
    """
    
    
    if x.ndim == 1: 
        x = x.reshape([1,len(x)])
    elif x.shape[0] > x.shape[1]:
        x = x.T
    
    #Adds 1 to row vector
    xTemp = np.ones([1,x.shape[1]+1])
    xTemp[:,1:] = x
    x = xTemp
    return x


# In[3]:


def addOneToBeginningOfCol(x):
    """Adds one to the beginning of the column vector x
    Inputs: 
        x - Column Vector 
    Outputs:
        x - Column Vector
    """
    
    if x.ndim == 1: 
        x = x.reshape([len(x),1])
    elif x.shape[1] > x.shape[0]:
        x = x.T
    
    #Adds 1 to column vector
    xTemp = np.ones([x.shape[0]+1,1])
    xTemp[1:,:] = x
    x = xTemp
    return x


# In[4]:


def removeFirstColumn(w):
    """Removes the first column from the matrix w
    Inputs:
        w - Matrix
    Outputs:
        w - Matrix
    """
    
    #Removes the first column
    w = w[:,1:]
    return w


# In[5]:


def computeMisclassificationRate(y, t):
    """Computes the misclassification rate
    Inputs:
        y - prediction
        t - target value
    Outputs:
        MisclassificationRate - The misclassification rate
    """
    
    #Computes the misclassification rate
    MisclassificationRate = np.sum(np.absolute(np.subtract(y,t)))/y.shape[0]
    return MisclassificationRate


# In[6]:


def ReLU(z):
    """Passes outputs into activation function ReLU(z)
    Inputs:
        z - outputs from previous hidden layer
    Outputs:
        h - current hidden layer
    """
    
    if z.ndim == 2:
        #Computes the relu element-wise
        h = np.zeros([z.shape[0],z.shape[1]])
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                h[i,j] = max(z[i,j],0.0)
    else:
        #Computes the relu element-wise
        h = np.zeros(len(z))
        for i in range(len(z)):
            h[i] = max(z[i],0.0)
    return h


# In[7]:


def ReLU_Prime(z):
    """Computes the derivative of ReLU(z)
    Inputs:
        z - outputs from previous hidden layer
    Outputs:
        g_z_Prime - derivative of ReLU(z)
    """
    
    #Computes relu'(z) element-wise
    g_z_Prime = np.zeros([z.shape[0],z.shape[1]])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i,j] >= 0:
                g_z_Prime[i,j] = 1
            else:
                g_z_Prime[i,j] = 0
    return g_z_Prime


# In[8]:


def sigmoid(z):
    """Passes outputs into activation function sigmoid(z)
    Inputs:
        z - outputs from previous hidden layer
    Outputs:
        y - FNN output
    """
    
    if z.ndim == 2:
        #Computes the sigmoid element-wise
        y = np.zeros([z.shape[0],z.shape[1]])
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                y[i,j] = 1/(1+np.exp(-z))
    else:
        #Computes the sigmoid element-wise
        y = np.zeros(len(z))
        for i in range(len(z)):
            y[i] = 1/(1+np.exp(-z))
    return y


# In[9]:


def InitializeParameters(nX,n1,n2,nY):
    """Initializes the parameter matrices based on the number of inputs, neurons and outputs
    Inputs:
        nX - number of inputs
        n1 - number of neurons in hidden layer 1
        n2 - number of layers in hidden layer 2
        nY - number of outputs
    Outputs:
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
    """
    
    #Initializing w1
    w1 = np.random.randn(n1,nX+1)
    
    #Initializing w2
    w2 = np.random.randn(n2,n1+1)
    
    #Initializing w3
    w3 = np.random.randn(nY,n2+1)
    
    return w1,w2,w3


# In[10]:


def ForwardPropogation(x,w1,w2,w3):
    """Calculates the prediction by passing the input x through the FNN
    Inputs:
        x - inputs
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
    Outputs:
        z1 - outputs from inputs
        h1 - layer 1
        z2 - outputs from layer 1
        h2 - layer 2
        z3 - outputs from layer 2
        y - prediction
    """
    
    #Computes Layer 1
    z1 = np.dot(w1,addOneToBeginningOfCol(x))
    h1 = ReLU(z1)

    #Computes Layer 2
    z2 = np.dot(w2,addOneToBeginningOfCol(h1))
    h2 = ReLU(z2)
    
    #Computes Layer 3
    z3 = np.dot(w3,addOneToBeginningOfCol(h2))
    y = sigmoid(z3)
    return z1,h1,z2,h2,z3,y


# In[11]:


def BackwardsPropogation(x,w1,w2,w3,z1,h1,z2,h2,z3,t):
    """Computes the change in parameter matrices using backwards propogation
    Inputs:
        x - inputs
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        z1 - outputs from inputs
        h1 - layer 1
        z2 - outputs from layer 1
        h2 - layer 2
        z3 - outputs from layer 2
        t - target value
    Outputs:
        delta_w1_J - change in parameter matrix 1
        delta_w2_J - change in parameter matrix 2
        delta_w3_J - change in parameter matrix 3
    """
    
    #Computes from Layer 3
    dJ_dz3 = -t + sigmoid(z3)
    delta_w3_J = np.dot(dJ_dz3,addOneToBeginningOfRow(h2.T)) 
    delta_z2_J =  np.multiply(ReLU_Prime(z2),np.dot(removeFirstColumn(w3).T,dJ_dz3))
    
    #Computes from Layer 2
    delta_w2_J = np.dot(delta_z2_J,addOneToBeginningOfRow(h1.T))
    delta_z1_J = np.multiply(ReLU_Prime(z1),np.dot(removeFirstColumn(w2).T,delta_z2_J))

    #Computes from Layer 1
    delta_w1_J = np.dot(delta_z1_J,addOneToBeginningOfRow(x.T))
    return delta_w1_J,delta_w2_J,delta_w3_J


# In[12]:


def StochasticGradientDescent(learningRate,w1,w2,w3,delta_w1_J,delta_w2_J,delta_w3_J):
    """Updates the parameter matrices
    Inputs:
        learningRate - the learning rate
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        delta_w1_J - change in parameter matrix 1
        delta_w2_J - change in parameter matrix 2
        delta_w3_J - change in parameter matrix 3
    Outputs:
        new_w1 - updated parameter matrix 1
        new_w2 - updated parameter matrix 2
        new_w3 - updated parameter matrix 3
    """
    
    #Updates the parameter matrices
    new_w1 = w1 - (learningRate*delta_w1_J)
    new_w2 = w2 - (learningRate*delta_w2_J)
    new_w3 = w3 - (learningRate*delta_w3_J)
    return new_w1,new_w2,new_w3


# In[13]:


def CrossEntropyCost(y, t):
    """Computes the cross-entropy error
    Inputs:
        y - predictors
        t - targets
    
    Outputs:
        cost - the cross-entropy error
    """
    
    if y.ndim == 1:
        y = y.reshape(len(y),1)
    if t.ndim == 1:
        t = t.reshape(len(t),1)

    #Computes cross-entropy error
    cost = np.sum(np.subtract(np.multiply(((-1)*t),np.log(y)),np.multiply((1-t),np.log(1-y))))/t.shape[0]
    return cost


# In[14]:


def ForwardNeuralNetwork(x,t,w1,w2,w3,learningRate):
    """Computes the necessary parameters to update the parameter matrices
    Inputs:
        x - inputs
        t - targets
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        learningRate - the learning rate
    Outputs:
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
    """
    
    #Pass x through neural network model
    z1,h1,z2,h2,z3,y = ForwardPropogation(x,w1,w2,w3)
    #Compute change in w
    delta_w1_J,delta_w2_J,delta_w3_J = BackwardsPropogation(x,w1,w2,w3,z1,h1,z2,h2,z3,t)
    #Update w1,w2,w3
    w1,w2,w3 = StochasticGradientDescent(learningRate,w1,w2,w3,delta_w1_J,delta_w2_J,delta_w3_J)
    return w1,w2,w3


# In[15]:


def TrainEntireDataSet(X_train,t_train,w1,w2,w3,learningRate):
    """Passes each point from the training set to train the model
    Inputs:
        X_train - the set of training points
        t_train - the training targets
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        learningRate - the learning rate
    Outputs:
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
    """
    
    #Passes each invdividual point through neural network and updates parameter matrices accordingly
    for pointI in range(X_train.shape[0]):
        x = X_train[pointI,:]
        t = t_train[pointI]
        w1,w2,w3 = ForwardNeuralNetwork(x.T,t,w1,w2,w3,learningRate)
    return w1,w2,w3


# In[16]:


def TrainEachEpoch(numEpochs,X_And_t_train,X_valid,t_valid,w1,w2,w3,learningRate,m):
    """Train entire dataset numEpochs amount of times to ind the best w1, w2, w3
    Inputs:
        numEpochs
        X_And_t_train - training inputs and targets
        X_train - the set of training points
        t_train - the training targets
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        learningRate - the learning rate
        m - the amount of epochs to try before solidifying choice as smallest cross validation
    Outputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid - minimal validation cost
        minMisclassificationRate - the missclassification rate associated with the best w1, w2, w3
        costTrainList - the list of training cross-entropy
        costValidList - the list of validation cross-entropy
    """
    
    #Initializing variable
    tempCostTrainList = np.zeros([1,numEpochs]) 
    tempCostValidList = np.zeros([1,numEpochs])
    numTurns = 0
    
    #Passes training set for each epoch
    for epochI in range(numEpochs):
        #Randomly shuffle X and t training points
        np.random.shuffle(X_And_t_train)
        #Seperate into x and t points
        X_train = X_And_t_train[:,:-1]
        t_train = X_And_t_train[:,-1]
        #Compute parameter matrices after training each point
        w1,w2,w3 = TrainEntireDataSet(X_train,t_train,w1,w2,w3,learningRate)
        #Compute training and validation predictions
        y_train = ComputePredictions(X_train,w1,w2,w3)
        y_valid = ComputePredictions(X_valid,w1,w2,w3)
        #Classify validation
        y_validClassified = ClassifyPredictions(y_valid,threshold=0.5)
        #Training cost
        costTrain = CrossEntropyCost(y_train, t_train)
        tempCostTrainList[0,epochI] = costTrain
        #Validation cost
        costValid = CrossEntropyCost(y_valid, t_valid)
        tempCostValidList[0,epochI] = costValid
        #Compute misclassification rate
        MisclassificationRate = computeMisclassificationRate(y_validClassified, t_valid)
        if (epochI == 0) or (costValid < minCostValid):
            #Updates minimum validation cross-entropy cost
            minCostValid = costValid
            #Updates misclassiication rate
            minMisclassificationRate = MisclassificationRate 
            #Updates best parameters
            best_w1 = w1
            best_w2 = w2
            best_w3 = w3
            numTurns = 0
        else:
            numTurns += 1
        if (numTurns >= m) or (epochI == numEpochs-1):
            #Slices the training and valid cross-entropy cost
            costTrainList = tempCostTrainList[:,:epochI+1]
            costValidList = tempCostValidList[:,:epochI+1]
            break
    return best_w1,best_w2,best_w3,minCostValid,minMisclassificationRate,costTrainList,costValidList


# In[17]:


def RunEachIteration(numIterations,numEpochs,nX,n1,n2,nY,X_And_t_train,X_valid,t_valid,learningRate,m):
    """Train entire dataset numEpochs amount of times to find the best w1, w2, w3, with numInterations number of iterations
    Inputs:
        numIterations - the number of iterations
        numEpochs - the number of epochs
        nX - number of inputs
        n1 - number of neurons in hidden layer 1
        n2 - number of layers in hidden layer 2
        nY - number of outputs
        X_And_t_train - training inputs and targets
        X_valid - the set of validation points
        t_valid - the validation targets
        learningRate - the learning rate
        m - the amount of epochs to try before solidifying choice as smallest cross validation
    Outputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid - minimal validation cost
        minMisclassificationRate - the missclassification rate associated with the best w1, w2, w3
        ListOfCostTrainLists - list of training cross-entropy lists
        ListOfCostValidLists - list of validation cross-entropy lists
    """
    
    ListOfCostTrainLists = []
    ListOfCostValidLists = []
    #Goes through all the traning points, through all the epochs, through all the iterations
    for itI in range(numIterations):
        w1,w2,w3 = InitializeParameters(nX,n1,n2,nY)
        w1,w2,w3,costValid,MisclassificationRate,CostTrainList,CostValidList = TrainEachEpoch(numEpochs,X_And_t_train,X_valid,t_valid,w1,w2,w3,learningRate,m)
        #Adds the training and validation cost list to the lost of cost lists
        ListOfCostTrainLists.append(CostTrainList)
        ListOfCostValidLists.append(CostValidList)
        if (itI == 0) or costValid < minCostValid:
            #Updates minimum validation cross-entropy cost
            minCostValid = costValid
            #Updates misclassiication rate
            minMisclassificationRate = MisclassificationRate
            #Updates best parameters
            best_w1 = w1
            best_w2 = w2
            best_w3 = w3
    return best_w1,best_w2,best_w3,minCostValid,minMisclassificationRate,ListOfCostTrainLists,ListOfCostValidLists


# In[18]:


def ChooseBestNumHiddenUnits(numIterations,numEpochs,nX,nY,X_train,t_train,X_valid,t_valid,learningRate,m):
    """Find the optimal number of neurons for each hidden layer
    Inputs:
        numIterations - the number of iterations
        numEpochs - the number of epochs
        nX - number of inputs
        nY - number of outputs
        X_train - the set of training points
        t_train - the training targets
        X_valid - the set of validation points
        t_valid - the validation targets
        learningRate - the learning rate
        m - the amount of epochs to try before solidifying choice as smallest cross validation
    Outputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid -
        minCostValidListForEachN1N2 -
        misclassificationListForEachN1N2 -
        best_n1 - best n1 value
        best_n2 - best n2 value
        BestListOfCostTrainLists - best list of training cross-entropy lists
        BestListOfCostValidLists - best list of validation cross-entropy lists
    """
    
    #Initializes parameters
    X_And_t_train = np.zeros([X_train.shape[0],X_train.shape[1]+1]) 
    X_And_t_train[:,:-1] = X_train
    X_And_t_train[:,-1] = t_train
    minCostValidListForEachN1N2 = np.zeros([2*nX,2*nX])
    misclassificationListForEachN1N2 = np.zeros([2*nX,2*nX])
    #Goes through each possible combination of n1 and n2 values
    for n1 in range(1,2*nX+1):
        for n2 in range(1,2*nX+1):
            print("n1: ",n1,"n2: ",n2)
            #Computes the parameter matrices, misclassification rate, the list of training cost lists and the list of validation cost lists
            w1,w2,w3,costValid,minMisclassificationRate,ListOfCostTrainLists,ListOfCostValidLists = RunEachIteration(numIterations,numEpochs,nX,n1,n2,nY,X_And_t_train,X_valid,t_valid,learningRate,m)
            #Creates matrix holding minimal validation cost for each n1 and n2
            minCostValidListForEachN1N2[n1-1,n2-1] = costValid
            #Creates matrix holding misclassification rate for each n1 and n2
            misclassificationListForEachN1N2[n1-1,n2-1] = minMisclassificationRate
            if ((n1 == 1) and (n2 == 1)) or (costValid < minCostValid):
                #Updates minimum validation cost 
                minCostValid = costValid
                #Updates best parameter matrices
                best_w1 = w1
                best_w2 = w2
                best_w3 = w3
                #Updates best n1 and n2 values
                best_n1 = n1
                best_n2 = n2
                #Updates the best list of training/validation cost lists
                BestListOfCostTrainLists = ListOfCostTrainLists
                BestListOfCostValidLists = ListOfCostValidLists
    return best_w1,best_w2,best_w3,minCostValid,minCostValidListForEachN1N2,misclassificationListForEachN1N2,best_n1,best_n2,BestListOfCostTrainLists,BestListOfCostValidLists


# In[19]:


def ComputePredictions(x,w1,w2,w3):
    """Compute the predictions
    Inputs:
        x - inputs
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
    Outputs:
        y - the predictions
    """
    
    #Initializes y
    y = np.zeros([x.shape[0],1])
    #Computes predictions for each input
    for pointI in range(x.shape[0]):
        y[pointI,0] = ForwardPropogation(x[pointI,:],w1,w2,w3)[-1] 
    return y


# In[20]:


def ClassifyPredictions(yPred,threshold):
    """Classifies the predictions
    Inputs:
        yPred - predictions as probabilities
    Outputs:
        y - predictions classified as 1 or 0
    """
    
    #Initializes y
    y = np.zeros(yPred.shape[0])
    #Classifies predictions
    for i in range(yPred.shape[0]):
        if yPred[i] >= threshold:
            y[i] = 1
        else:
            y[i] = 0
    return y


# In[ ]:





# In[21]:


def CrossEntropyCostWithRegularization(y, t, w1, w2, w3, lamda):
    """Computes the cross-entropy error with regularization
    Inputs:
        y - predictors
        w1 - paramater matrix 1
        w2 - paramater matrix 2
        w3 - paramater matrix 3
        t - targets
        lambda - lambda value for regularization
    
    Outputs:
        cost - the cross-entropy error
    """
    
    #Calculating cost
    crossEntropyCost = CrossEntropyCost(y, t)
    L2RegularizationCost = lamda*(np.sum(np.dot(w1.T,w1))+np.sum(np.dot(w2.T,w2))+np.sum(np.dot(w3.T,w3)))
    cost = crossEntropyCost + L2RegularizationCost
    return cost


# In[22]:


def BackwardsPropogationWithRegularization(x,w1,w2,w3,z1,h1,z2,h2,z3,t,lamda):
    """Computes the change in parameter matrices using backwards propogation
    Inputs:
        x - inputs
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        z1 - outputs from inputs
        h1 - layer 1
        z2 - outputs from layer 1
        h2 - layer 2
        z3 - outputs from layer 2
        t - target value
        lambda - lambda value for regularization
        
    Outputs:
        delta_w1_J - change in parameter matrix 1
        delta_w2_J - change in parameter matrix 2
        delta_w3_J - change in parameter matrix 3
    """
    
    #Computes from Layer 3
    dJ_dz3 = -t + sigmoid(z3)
    delta_w3_J = np.dot(dJ_dz3,addOneToBeginningOfRow(h2.T))
    delta_w3_J[:,1:] = delta_w3_J[:,1:] + 2*lamda*w3[:,1:]
    delta_z2_J =  np.multiply(ReLU_Prime(z2),np.dot(removeFirstColumn(w3).T,dJ_dz3))
    
    #Computes from Layer 2
    delta_w2_J = np.dot(delta_z2_J,addOneToBeginningOfRow(h1.T))
    delta_w2_J[:,1:] = delta_w2_J[:,1:] + 2*lamda*w2[:,1:]
    delta_z1_J = np.multiply(ReLU_Prime(z1),np.dot(removeFirstColumn(w2).T,delta_z2_J))

    #Computes from Layer 1
    delta_w1_J = np.dot(delta_z1_J,addOneToBeginningOfRow(x.T))
    delta_w1_J[:,1:] = delta_w1_J[:,1:] + 2*lamda*w1[:,1:]
    return delta_w1_J,delta_w2_J,delta_w3_J


# In[23]:


def ForwardNeuralNetworkWithRegularization(x,t,w1,w2,w3,learningRate,lamda):
    """Computes the necessary parameters to update the parameter matrices
    Inputs:
        x - inputs
        t - targets
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        learningRate - the learning rate
        lambda - lambda value for regularization
        
    Outputs:
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
    """
    
    #Pass x through neural network model
    z1,h1,z2,h2,z3,y = ForwardPropogation(x,w1,w2,w3)
    #Compute change in w
    delta_w1_J,delta_w2_J,delta_w3_J = BackwardsPropogationWithRegularization(x,w1,w2,w3,z1,h1,z2,h2,z3,t,lamda)
    #Update w1,w2,w3
    w1,w2,w3 = StochasticGradientDescent(learningRate,w1,w2,w3,delta_w1_J,delta_w2_J,delta_w3_J)
    return w1,w2,w3


# In[24]:


def TrainEntireDataSetWithRegularization(X_train,t_train,w1,w2,w3,learningRate,lamda):
    """Passes each point from the training set to train the model
    Inputs:
        X_train - the set of training points
        t_train - the training targets
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        learningRate - the learning rate
        lambda - lambda value for regularization
        
    Outputs:
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
    """
    
    #Passes each invdividual point through neural network and updates parameter matrices accordingly
    for pointI in range(X_train.shape[0]):
        x = X_train[pointI,:]
        t = t_train[pointI]
        w1,w2,w3 = ForwardNeuralNetworkWithRegularization(x.T,t,w1,w2,w3,learningRate,lamda)
    return w1,w2,w3


# In[25]:


def TrainEachEpochWithRegularization(numEpochs,X_And_t_train,X_valid,t_valid,w1,w2,w3,learningRate,m,lamda):
    """Train entire dataset numEpochs amount of times to ind the best w1, w2, w3
    Inputs:
        numEpochs
        X_And_t_train - training inputs and targets
        X_train - the set of training points
        t_train - the training targets
        w1 - parameter matrix 1
        w2 - parameter matrix 2
        w3 - parameter matrix 3
        learningRate - the learning rate
        m - the amount of epochs to try before solidifying choice as smallest cross validation
    Outputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid - minimal validation cost
        minMisclassificationRate - the missclassification rate associated with the best w1, w2, w3
        costTrainList - the list of training cross-entropy
        costValidList - the list of validation cross-entropy
    """
    
    #Initializing variable
    tempCostTrainList = np.zeros([1,numEpochs]) 
    tempCostValidList = np.zeros([1,numEpochs])
    numTurns = 0
    
    #Passes training set for each epoch
    for epochI in range(numEpochs):
        #Randomly shuffle X and t training points
        np.random.shuffle(X_And_t_train)
        #Seperate into x and t points
        X_train = X_And_t_train[:,:-1]
        t_train = X_And_t_train[:,-1]
        #Compute parameter matrices after training each point
        w1,w2,w3 = TrainEntireDataSetWithRegularization(X_train,t_train,w1,w2,w3,learningRate,lamda)
        #Compute training and validation predictions
        y_train = ComputePredictions(X_train,w1,w2,w3)
        y_valid = ComputePredictions(X_valid,w1,w2,w3)
        #Classify validation
        y_validClassified = ClassifyPredictions(y_valid,threshold=0.5)
        #Training cost
        costTrain = CrossEntropyCostWithRegularization(y_train, t_train, w1,w2,w3, lamda)
        tempCostTrainList[0,epochI] = costTrain
        #Validation cost
        costValid = CrossEntropyCostWithRegularization(y_valid, t_valid, w1,w2,w3, lamda)
        tempCostValidList[0,epochI] = costValid
        #Compute misclassification rate
        MisclassificationRate = computeMisclassificationRate(y_validClassified, t_valid)
        if (epochI == 0) or (costValid < minCostValid):
            #Updates minimum validation cross-entropy cost
            minCostValid = costValid
            #Updates misclassiication rate
            minMisclassificationRate = MisclassificationRate 
            #Updates best parameters
            best_w1 = w1
            best_w2 = w2
            best_w3 = w3
            numTurns = 0
        else:
            numTurns += 1
        if (numTurns >= m) or (epochI == numEpochs-1):
            #Slices the training and valid cross-entropy cost
            costTrainList = tempCostTrainList[:,:epochI+1]
            costValidList = tempCostValidList[:,:epochI+1]
            break
    return best_w1,best_w2,best_w3,minCostValid,minMisclassificationRate,costTrainList,costValidList


# In[26]:


def RunEachIterationWithRegularization(numIterations,numEpochs,nX,n1,n2,nY,X_And_t_train,X_valid,t_valid,learningRate,m,lamda):
    """Train entire dataset numEpochs amount of times to find the best w1, w2, w3, with numInterations number of iterations
    Inputs:
        numIterations - the number of iterations
        numEpochs - the number of epochs
        nX - number of inputs
        n1 - number of neurons in hidden layer 1
        n2 - number of layers in hidden layer 2
        nY - number of outputs
        X_And_t_train - training inputs and targets
        X_valid - the set of validation points
        t_valid - the validation targets
        learningRate - the learning rate
        m - the amount of epochs to try before solidifying choice as smallest cross validation
    Outputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid - minimal validation cost
        minMisclassificationRate - the missclassification rate associated with the best w1, w2, w3
        ListOfCostTrainLists - list of training cross-entropy lists
        ListOfCostValidLists - list of validation cross-entropy lists
    """
    
    ListOfCostTrainLists = []
    ListOfCostValidLists = []
    #Goes through all the traning points, through all the epochs, through all the iterations
    for itI in range(numIterations):
        w1,w2,w3 = InitializeParameters(nX,n1,n2,nY)
        w1,w2,w3,costValid,MisclassificationRate,CostTrainList,CostValidList = TrainEachEpochWithRegularization(numEpochs,X_And_t_train,X_valid,t_valid,w1,w2,w3,learningRate,m,lamda)
        #Adds the training and validation cost list to the lost of cost lists
        ListOfCostTrainLists.append(CostTrainList)
        ListOfCostValidLists.append(CostValidList)
        if (itI == 0) or costValid < minCostValid:
            #Updates minimum validation cross-entropy cost
            minCostValid = costValid
            #Updates misclassiication rate
            minMisclassificationRate = MisclassificationRate
            #Updates best parameters
            best_w1 = w1
            best_w2 = w2
            best_w3 = w3
    return best_w1,best_w2,best_w3,minCostValid,minMisclassificationRate,ListOfCostTrainLists,ListOfCostValidLists


# In[27]:


def ChooseBestNumHiddenUnitsAndLambdaValueWithRegularization(numIterations,numEpochs,nX,nY,X_train,t_train,X_valid,t_valid,learningRate,m,startlnLamdaValue,StoplnLamdaValue,SteplnLamdaValue):
    """Find the optimal number of neurons for each hidden layer
    Inputs:
        numIterations - the number of iterations
        numEpochs - the number of epochs
        nX - number of inputs
        nY - number of outputs
        X_train - the set of training points
        t_train - the training targets
        X_valid - the set of validation points
        t_valid - the validation targets
        learningRate - the learning rate
        m - the amount of epochs to try before solidifying choice as smallest cross validation
        lambda - lambda value for regularization
    Outputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid - minimum validation cost
        minCostValidListForEachN1N2 -
        misclassificationListForEachN1N2 -
        best_n1 - best n1 value
        best_n2 - best n2 value
        BestListOfCostTrainLists - best list of training cross-entropy lists
        BestListOfCostValidLists - best list of validation cross-entropy lists
    """
    
    #Initializes parameters
    X_And_t_train = np.zeros([X_train.shape[0],X_train.shape[1]+1]) 
    X_And_t_train[:,:-1] = X_train
    X_And_t_train[:,-1] = t_train
    
    listOfminCostValidListForEachN1N2 = []
    listOfmisclassificationListForEachN1N2 = []
    i = 0
    #Goes through each possible valud for lambda
    for lnlamda in range(startlnLamdaValue,StoplnLamdaValue+SteplnLamdaValue,SteplnLamdaValue):
        i += 1
        lamda = np.exp(lnlamda)
        #Goes through each possible combination of n1 and n2 values
        minCostValidListForEachN1N2 = np.zeros([2*nX,2*nX])
        misclassificationListForEachN1N2 = np.zeros([2*nX,2*nX])
        for n1 in range(1,2*nX+1):
            for n2 in range(1,2*nX+1):
                print("ln(lambda):", lnlamda," n1:",n1," n2:",n2)
                #Computes the parameter matrices, misclassification rate, the list of training cost lists and the list of validation cost lists
                w1,w2,w3,costValid,minMisclassificationRate,ListOfCostTrainLists,ListOfCostValidLists = RunEachIterationWithRegularization(numIterations,numEpochs,nX,n1,n2,nY,X_And_t_train,X_valid,t_valid,learningRate,m,lamda)
                #Creates matrix holding minimal validation cost for each n1 and n2
                minCostValidListForEachN1N2[n1-1,n2-1] = costValid
                #Creates matrix holding misclassification rate for each n1 and n2
                misclassificationListForEachN1N2[n1-1,n2-1] = minMisclassificationRate
                if ((lnlamda == startlnLamdaValue) and (n1 == 1) and (n2 == 1)) or (costValid < minCostValid):
                    #Updates minimum validation cost 
                    minCostValid = costValid
                    #Updates best parameter matrices
                    best_w1 = w1
                    best_w2 = w2
                    best_w3 = w3
                    #Updates best n1 and n2 values
                    best_n1 = n1
                    best_n2 = n2
                    best_lamda = lamda
                    #Updates the best list of training/validation cost lists
                    BestListOfCostTrainLists = ListOfCostTrainLists
                    BestListOfCostValidLists = ListOfCostValidLists
        listOfminCostValidListForEachN1N2.append(minCostValidListForEachN1N2)
        listOfmisclassificationListForEachN1N2.append(misclassificationListForEachN1N2)  
    listOfLnLamdaValues = np.zeros([1,i])
    i = 0
    for lnlamda in range(startlnLamdaValue,StoplnLamdaValue+SteplnLamdaValue,SteplnLamdaValue):
        listOfLnLamdaValues[0,i] = lnlamda 
        i += 1
    return best_w1,best_w2,best_w3,minCostValid,listOfminCostValidListForEachN1N2,listOfmisclassificationListForEachN1N2,listOfLnLamdaValues,best_n1,best_n2,best_lamda,BestListOfCostTrainLists,BestListOfCostValidLists


# In[ ]:





# In[28]:


def plotOutputs(BestListOfCostTrainLists,BestListOfCostValidLists,best_n1,best_n2):
    """Plots the learning curves
    Inputs:
        BestListOfCostTrainLists - best list of training cross-entropy lists
        BestListOfCostValidLists - best list of validation cross-entropy lists
        best_n1 - best n1 value
        best_n2 - best n2 value
    """
    
    #Creates number of iterations matrix
    iterations = [i for i in range(len(BestListOfCostTrainLists))]
    
    #Plots each learning curve
    for numIterations in range(len(iterations)):
        epochs = [j for j in range(1,BestListOfCostTrainLists[numIterations].shape[1]+1)]
        plt.scatter(epochs, BestListOfCostTrainLists[numIterations], color = 'green', label = 'Training Cross-Entropy Cost')
        plt.scatter(epochs, BestListOfCostValidLists[numIterations], color = 'red', label = 'Validation Cross-Entropy Cost')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Cross-Entropy Cost vs Number of Epochs For Iteration #"+str(numIterations+1)+" For N1 = "+str(best_n1)+" and N2 = "+str(best_n2))
        plt.xlabel("Number of Epochs")
        plt.ylabel("Cross-Entropy Cost")
        plt.show()


# In[29]:


def printAllOutputs(best_w1,best_w2,best_w3,minCostValid,minCostValidListForEachN1N2,misclassificationListForEachN1N2,best_n1,best_n2,TestMisclassificationRate):
    """Prints all the outputs
    Inputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid - minimal validation cost
        minCostValidListForEachN1N2 - minimal validation cost for each N1 and N2
        misclassificationListForEachN1N2 - list of misclassification rates for each n1 and n2
        best_n1 - best n1 value
        best_n2 - best n2 value
        TestMisclassificationRate - misclassification rate for testing data
    """
    
    #Prints all parameter matrices
    print("Best w1 vector:")
    print(best_w1)
    print()
    print("Best w2 vector:")
    print(best_w2)
    print()
    print("Best w3 vector:")
    print(best_w3)
    print()
    #Princes the smallest validation cross-entropy costs and misclassification rate for each n1 and n2
    for i in range(minCostValidListForEachN1N2.shape[0]):
        for j in range(minCostValidListForEachN1N2.shape[1]):
            print("N1: ",i+1, " N2: ",j+1)
            print("Smallest Validation Cross-Entropy Cost: ",minCostValidListForEachN1N2[i,j])
            print()
    #Prints the validation cost and test misclassification rate for the best n1 and n2
    print("Best N1 = ",best_n1," and Best N2 = ",best_n2)
    print("Validation Cross-Entropy Cost: ",minCostValid)
    print()
    print("Test Missclassification Rate: ",TestMisclassificationRate)


# In[ ]:





# In[30]:


def plotOutputsRegularization(listOfminCostValidListForEachN1N2,listOfmisclassificationListForEachN1N2,listOfLnLamdaValues,best_n1,best_n2,best_lambda):
    """Plots outputs
    Inputs:
        listOfminCostValidListForEachN1N2 - list of minimum validation cost lists for each n1 and n2
        listOfmisclassificationListForEachN1N2 - list of misclassification lists for each n1 and n2
        listOfLnLamdaValues - listof ln(lamda) values
        best_n1 - best n1 value
        best_n2 - best n2 value
        best_lambda - best lambda value
    """
    
    print("len(listOfLnLamdaValues[0]):",len(listOfLnLamdaValues[0]))
    listOfminCostValid = np.zeros([1,len(listOfLnLamdaValues[0])])
    for i in range(len(listOfLnLamdaValues[0])): 
        listOfminCostValid[0,i] = listOfminCostValidListForEachN1N2[i][best_n1-1][best_n2-1]
    print(listOfminCostValid)
    plt.scatter(listOfLnLamdaValues, listOfminCostValid, color = 'green')
    plt.title("Validation Cross-Entropy for n1 = "+str(best_n1)+" and n2 = "+str(best_n2)+" vs ln(lamdba)")
    plt.xlabel("ln(lamdba)")
    plt.ylabel("Validation Cross-Entropy for n1 = "+str(best_n1)+" and n2 = "+str(best_n2))
    plt.show()


# In[31]:


def printOutputsRegularization(best_w1,best_w2,best_w3,minCostValid,listOfminCostValidListForEachN1N2,listOfmisclassificationListForEachN1N2,listOfLnLamdaValues,best_n1,best_n2,best_lambda,BestListOfCostTrainLists,BestListOfCostValidLists,TestMisclassificationRate):
    """Prints the outputs
    Inputs:
        best_w1 - best parameter matrix 1
        best_w2 - best parameter matrix 2
        best_w3 - best parameter matrix 3
        minCostValid - minimum validation cost
        listOfminCostValidListForEachN1N2 - list of minimum validation cost lists for each n1 and n2
        listOfmisclassificationListForEachN1N2 - list of misclassification lists for each n1 and n2
        listOfLnLamdaValues - listof ln(lamda) values
        best_n1 - best n1 value
        best_n2 - best n2 value
        best_lambda - best lambda value
        BestListOfCostTrainLists - best list of training cost list
        BestListOfCostValidLists - best list of training cost list
        TestMisclassificationRate - misclassification rate for testing data
    """
    
    #Prints all parameter matrices
    print("Best w1 vector:")
    print(best_w1)
    print()
    print("Best w2 vector:")
    print(best_w2)
    print()
    print("Best w3 vector:")
    print(best_w3)
    print()
    
    #Print validcation cross-entropy cost for each ln(lamda) value
    for i in range(len(listOfLnLamdaValues[0])): 
        print("ln(lamda) = ",listOfLnLamdaValues[0][i]," Validation Cross-Entropy Cost",listOfminCostValidListForEachN1N2[i][best_n1-1][best_n2-1])
    print()
    
    #Prints the validation cost and test misclassification rate for the best n1 and n2
    print("ln(best_lambda) = ",np.log(best_lambda)," Best N1 = ",best_n1," and Best N2 = ",best_n2)
    print("Validation Cross-Entropy Cost: ",minCostValid)
    print()
    print("Test Missclassification Rate: ",TestMisclassificationRate)


# In[ ]:





# In[32]:


#load data set
dataset = pd.read_csv('data_banknote_authentication.txt', header=None)
X_data = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values

#split the data into training, validation and test data 
X_train, X_valid_test, t_train, t_valid_test = train_test_split(X_data, t, test_size = 0.2, random_state = 7878)
X_valid, X_test, t_valid, t_test = train_test_split(X_valid_test, t_valid_test, test_size = 0.5, random_state = 7878)

#normalize training and testing data 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)

#Initializing inputs
numIterations = 1
numEpochs = 1
#numIterations = 5
#numEpochs = 100
nX = X_train.shape[1]
nY = 1
learningRate = 0.005
m = 10


# In[33]:


#Computing best w1 w2 and w3
#best_w1,best_w2,best_w3,minCostValid,minCostValidListForEachN1N2,misclassificationListForEachN1N2,best_n1,best_n2,BestListOfCostTrainLists,BestListOfCostValidLists = ChooseBestNumHiddenUnits(numIterations,numEpochs,nX,nY,X_train,t_train,X_valid,t_valid,learningRate,m)


# In[34]:


#Computes the misclassification rate associated with the best prediction
#TestMisclassificationRate = computeMisclassificationRate(ClassifyPredictions(ComputePredictions(X_test,best_w1,best_w2,best_w3),threshold=0.5), t_test)


# In[35]:


#Plots the output graphs
#plotOutputs(BestListOfCostTrainLists,BestListOfCostValidLists,best_n1,best_n2)


# In[36]:


#Prints the outputs
#printAllOutputs(best_w1,best_w2,best_w3,minCostValid,minCostValidListForEachN1N2,misclassificationListForEachN1N2,best_n1,best_n2,TestMisclassificationRate)


# In[ ]:





# In[37]:


#Regularization
startlnLamdaValue = -20
stoplnLamdaValue = 0
steplnLamdaValue = 10
#Computing best w1 w2 and w3
best_w1,best_w2,best_w3,minCostValid,listOfminCostValidListForEachN1N2,listOfmisclassificationListForEachN1N2,listOfLnLamdaValues,best_n1,best_n2,best_lambda,BestListOfCostTrainLists,BestListOfCostValidLists = ChooseBestNumHiddenUnitsAndLambdaValueWithRegularization(numIterations,numEpochs,nX,nY,X_train,t_train,X_valid,t_valid,learningRate,m,startlnLamdaValue,stoplnLamdaValue,steplnLamdaValue)


# In[38]:


#Computes the misclassification rate associated with the best prediction
TestMisclassificationRate = computeMisclassificationRate(ClassifyPredictions(ComputePredictions(X_test,best_w1,best_w2,best_w3),threshold=0.5), t_test)


# In[39]:


#Plot the outputs
plotOutputsRegularization(listOfminCostValidListForEachN1N2,listOfmisclassificationListForEachN1N2,listOfLnLamdaValues,best_n1,best_n2,best_lambda)


# In[40]:


#Prints the outputs
printOutputsRegularization(best_w1,best_w2,best_w3,minCostValid,listOfminCostValidListForEachN1N2,listOfmisclassificationListForEachN1N2,listOfLnLamdaValues,best_n1,best_n2,best_lambda,BestListOfCostTrainLists,BestListOfCostValidLists,TestMisclassificationRate)


# In[ ]:




