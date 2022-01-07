#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


def ComputeW(X_train, t_train, numIterations, alpha):
    """Computes the parameter vector
    Inputs: 
        X_train - the training features
        t_train - the training targets
        numIterations - the number of iterations to improve the parameter vector values
        alpha - impacts how much the parameter vector values change by per iteration
    Outputs:
        W - the parameter vector
    """
    
    #Adding X_train to a column of ones
    xMat = np.ones((X_train.shape[0],X_train.shape[1]+1))
    xMat[:,1:] = X_train
    #Initializing matrices for W
    W = np.ones(X_train.shape[1]+1)
    
    #Performing gradient descent
    for n in range(numIterations):
        z = np.dot(xMat,W)
        y = 1/(1+np.exp(-z))
        diff = y - t_train
        gr = np.dot(xMat.T, diff)/X_train.shape[0]

        W = W - alpha * gr
    return W


# In[3]:


def computeY(X,W,threshold):
    """Computes the predictor
    Inputs: 
        X - the set of features
        W - the parameter vector
        threshold - the threhold used to select a class
    Outputs:
        y - the predictor
    """
    
    #Adding X_train to a column of ones
    xMat = np.ones((X.shape[0],X.shape[1]+1))
    xMat[:,1:] = X
    
    #Initializing z and y matrices
    z = np.dot(xMat,W)
    y = np.zeros(z.shape[0])
    
    #Creating the predictor
    for i in range(y.shape[0]):
        if z[i] >= threshold:
            y[i] = 1
        else:
            y[i] = 0
    return y


# In[4]:


def computePeformanceParamaters(y, t):
    """Computes the missclassification rate, the precision, the recall and the F1 score
    Inputs: 
        y - the predictor
        t - the target values
    Outputs:
        MissclassificationRate - the misclassification rate
        P - the precision
        R - the recall
        F1 - the F1 score
    """
    
    #Computing performance paramaters
    MissclassificationRate = np.sum(np.absolute(np.subtract(y,t)))/y.shape[0]    
    NumEstimatedPos = np.count_nonzero(y == 1)
    NumTrulyPosExFound = np.count_nonzero(np.multiply((y == 1),(t == 1)))
    NumTrulyPos = np.count_nonzero(t == 1)
    try: 
        P = NumTrulyPosExFound/NumEstimatedPos
    except ZeroDivisionError:
        P = 0
    try:
        R = NumTrulyPosExFound/NumTrulyPos
    except ZeroDivisionError:
        R = 0
    try:
        F1 = (2*P*R)/(P+R)
    except ZeroDivisionError:
        F1 = 0
    return MissclassificationRate, P, R, F1


# In[5]:


def ComputeThresholdList(X,W):
    """Computes the list of thresholds
    Inputs: 
        X - the set of features
        W - the parameter vector
    Outputs:
        thresholdList - the list of threshold values
    """
    
    #Adding X_train to a column of ones
    xMat = np.ones((X.shape[0],X.shape[1]+1))
    xMat[:,1:] = X
    
    z = np.dot(xMat,W) #Computing z
    thresholdList = np.sort(z) #Computing list of threshold values
    return thresholdList


# In[6]:


def ComputePeformanceParamatersList(X,t,W,thresholdList):
    """Computes the list of performance parameters
    Inputs: 
        X - the set of features
        t - the target values
        W - the parameter vector
        thresholdList - the list of threshold values
    Outputs:
        MissclassificationRateList - the list of missclassification rates
        PList - the list of precision values
        RList - the list of recall values
        F1List - the list of F1 scores
        BestThreshold_MissclassificationRate - the best threshold associated with the lowest missclassification rate
        LowestMissclassificationRate - the lowest missclassification rate
        BestThreshold_F1Score - the best threshold associated with the highest F1 score
        HighestF1Score - the highest F1 score
    """
    
    #Initializing lists for performance parameters
    MissclassificationRateList = np.zeros(thresholdList.shape[0])
    PList = np.zeros(thresholdList.shape[0])
    RList = np.zeros(thresholdList.shape[0])
    F1List = np.zeros(thresholdList.shape[0])
    
    #Computes list of performance parameter associated with each threshold
    for i in range(thresholdList.shape[0]):
        y = computeY(X,W,thresholdList[i])
        MissclassificationRateList[i], PList[i], RList[i], F1List[i] = computePeformanceParamaters(y, t)
    
    #Computing the minimal error and threshold associated
    LowestMissclassificationRateInd = np.argmin(MissclassificationRateList)
    BestThreshold_MissclassificationRate = thresholdList[LowestMissclassificationRateInd]
    LowestMissclassificationRate = MissclassificationRateList[LowestMissclassificationRateInd]
    HighestF1ScoreInd = np.argmax(F1List)
    BestThreshold_F1Score = thresholdList[HighestF1ScoreInd]
    HighestF1Score = F1List[HighestF1ScoreInd]
    return MissclassificationRateList, PList, RList, F1List, BestThreshold_MissclassificationRate, LowestMissclassificationRate, BestThreshold_F1Score, HighestF1Score


# In[7]:


def PlotPerfomanceCurves(thresholdList,MissclassificationRateList, PList, RList, F1List):
    """Plots the performance curves
    Inputs: 
        thresholdList - the list of threshold values
        MissclassificationRateList - the list of missclassification rates
        PList - the list of precision values
        RList - the list of recall values
        F1List - the list of F1 scores
    """
    
    #Plotting graphs
    plt.title("Missclassification Rate Vs Threshold Curve")
    plt.xlabel("Threshold")
    plt.ylabel("Missclassification Rate")
    plt.scatter(thresholdList, MissclassificationRateList)
    plt.show()
    
    plt.title("F1 Score Vs Threshold Curve")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.scatter(thresholdList, F1List)
    plt.show()
    
    plt.title("Precision Vs Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.scatter(RList, PList)
    plt.show()


# In[8]:


def calcDistance(X_train,X_test):
    """Calculates the distance between the test and training features
    Inputs: 
        X_train - the training features
        X_test - the test features
    Outputs:
        dist - the distance between the test and training features
    """
    
    #Initializing parameters
    numTrainPts = X_train.shape[0]
    numTestPts = X_test.shape[0]
    dist = np.zeros((numTestPts,numTrainPts))
    
    #Calculating the distance between the test and training features
    for test_I in range(numTestPts):
        for train_I in range(numTrainPts):
            dist[test_I,train_I] = np.sqrt(np.sum(np.power(np.subtract(X_test[test_I,:],X_train[train_I,:]),2)))
    return dist


# In[9]:


def CalcKNN(K,X_train,X_test,t_train,t_test):
    """Computes the predictor for each k value in k-nearest neighbour from 1 to K using cross-validation
    Inputs: 
        K - the maximum K value for K-nearest neighbour 
        X_train - the training features
        X_test - the test features
        t_train - the training targets
        t_test - the test targets
    Outputs:
        KNNMatrix_of_yCols - the predictor for each k value in k-nearest neighbour from 1 to K
    """
    
    #Reshaping training points
    t_train = np.reshape(t_train, (t_train.shape[0], 1))
    t_test = np.reshape(t_test, (t_test.shape[0], 1))
    
    #Initializing parameters
    numTrainPts = X_train.shape[0]
    numTestPts = X_test.shape[0]
    dist = calcDistance(X_train,X_test)
    ind = np.argsort(dist, axis=1)
    KNNMatrix_of_yCols = np.zeros((numTestPts,K))
    
    #Performing KNN
    for k in range(K): # (k+1)-NN
        for test_I in range(numTestPts):
            predictorForTest_I_List = np.array([])
            for s in range(k+1):
                predictorForTest_I_List = np.append(predictorForTest_I_List, t_train[ind[test_I,s],0])
            if (np.count_nonzero(predictorForTest_I_List == 1) > np.count_nonzero(predictorForTest_I_List == 0)):
                predictorForTest_I = 1
            elif (np.count_nonzero(predictorForTest_I_List == 1) < np.count_nonzero(predictorForTest_I_List == 0)):
                predictorForTest_I = 0
            else:
                predictorForTest_I = t_train[ind[test_I,s+1],0]
            KNNMatrix_of_yCols[test_I,k] = predictorForTest_I
    return KNNMatrix_of_yCols


# In[10]:


def calculateKNNTestError_MissclassificationRate(K,X_train,X_test,t_train,t_test,BestKNN):
    """Computes the test error using missclassification rate for the best k value for k-nearest neighbour as the classifier
    Inputs:
        K - the maximum K value for K-nearest neighbour 
        X_train - the training features
        X_test - the test features
        t_train - the training targets
        t_test - the test targets
        BestKNN - the k value for k-nearest neighbour which gives the smallest average cross-validation error
    
    Outputs:
        MissclassificationRate - the missclassification rate
    """
    
    #Obtains the predicted values using knn and cross-validation
    KNNMatrix_of_yCols = CalcKNN(K,X_train,X_test,t_train,t_test) 
    y = KNNMatrix_of_yCols[:,BestKNN-1]
    
    #Calculates missclassification rate
    MissclassificationRate = computePeformanceParamaters(y, t_test)[0]
    return MissclassificationRate


# In[11]:


def getMatSubSet(Mat,i,KFold,numPoints):
    """Returns all rows of Mat and 1/KFold of the total rows
    Inputs: 
        Mat - the full matrix
        i - the ith group of rows of the matrix
        KFold - number of groups to split the total rows into
        numPoints - number of rows
        
    Outputs:
        MatSubSet - Piece of the matrix, mat
    """
    
    if i == KFold: #Sets MatSubSet to the rest of the rows for the last piece of the matrix, Mat
        MatSubSet = Mat[numPoints*(i-1):,:]
    else: #Sets MatSubSet to the numPoints number of rows and all columns from the matrix, Mat
        MatSubSet = Mat[numPoints*(i-1):(numPoints*i),:]
    return MatSubSet


# In[12]:


def splitMat(Mat,KFold):
    """Splits mat into all combinations of training sets and validation sets for K-fold cross-validation
    Inputs: 
        Mat - the full matrix
        KFold - number of groups to split the total rows into
        
    Outputs:
        ListOfTrainMatsCombinations - list of training matrices
        ListOfValidMats - list of validation matrices
    """
    
    numPoints = round(Mat.shape[0]/KFold) #Number of rows to have in each sub matrix
    ListOfSplitMats = [getMatSubSet(Mat,i,KFold,numPoints) for i in range(1,KFold+1)] #Creates a list of each sub-matrix
    ListOfTrainMatsCombinations = []
    ListOfValidMats = []
    
    
    for i in range(len(ListOfSplitMats)):
        trainMat = 'null'    
        #Combines 4 sub matrices to form the training set and sets the other as a validation set
        for j in range(len(ListOfSplitMats)): 
            if i == j:
                ListOfValidMats.append(ListOfSplitMats[j]) #Adds the validation set to the list of validation sets
                continue
            if type(trainMat) == str: 
                trainMat = ListOfSplitMats[j]
            else: 
                trainMat = np.concatenate((trainMat, ListOfSplitMats[j]))
        ListOfTrainMatsCombinations.append(trainMat) #Adds the training set to the list of training set combinations
    return ListOfTrainMatsCombinations, ListOfValidMats  


# In[13]:


def kFoldCrossValidation_MissclassificationRate(k,xMat,t_Points):
    """Performs K-fold cross-validation on a matrix with missclassification rate as the error
    Inputs: 
        k - the maximum K value for K-nearest neighbour
        xMat - the matrix of features
        t_Points - target values
        
    Outputs:
        ListOfKNNMatrix_of_yCols - the list of KNN matrices for each fold
        listOfCrossValidErrorList - the list of cross-validation error lists for each k-nearest neighbour classifier
        listOfAverageCrossValidError - the list of average cross-validation errors for each k-nearest neighbour classifier
        BestKNN - the k value for k-nearest neighbour which gives the smallest average cross-validation error
    """
    
    t_Points = t_Points.reshape((len(t_Points),1))
    
    #Obtain list of K-fold training and validation sets
    ListOfX_trainCombinations, ListOfX_valid = splitMat(xMat,5)
    ListOft_trainCombinations, ListOft_valid = splitMat(t_Points,5)
    ListOfKNNMatrix_of_yCols = []
    
    #Fixes format of validation targets
    for i in range(len(ListOft_valid)): 
        ListOft_valid[i] = ListOft_valid[i].flatten()
    
    #Computes prediction values with knn using cross-validation 
    for X_train,t_train,X_test,t_test in zip(ListOfX_trainCombinations,ListOft_trainCombinations,ListOfX_valid,ListOft_valid): 
        ListOfKNNMatrix_of_yCols.append(CalcKNN(k,X_train,X_test,t_train,t_test))
    listOfCrossValidErrorList = []
    tempListOft_valid = []
    listOfAverageCrossValidError = []
    
    #Generates list of errors for each individual K-fold cross-validation test
    for K_NN in range(ListOfKNNMatrix_of_yCols[0].shape[1]):
        CrossValidErrorList = []
        for Dataset in range(len(ListOfKNNMatrix_of_yCols)):
            MissclassificationRate = computePeformanceParamaters(ListOfKNNMatrix_of_yCols[Dataset][:,K_NN], ListOft_valid[Dataset])[0]
            CrossValidErrorList.append(MissclassificationRate)
        listOfAverageCrossValidError.append(np.mean(CrossValidErrorList))
        listOfCrossValidErrorList.append(CrossValidErrorList)
        BestKNN = np.argmin(listOfAverageCrossValidError)+1
    return ListOfKNNMatrix_of_yCols,listOfCrossValidErrorList,listOfAverageCrossValidError,BestKNN


# In[14]:


def kFoldCrossValidation_F1Score(k,xMat,t_Points):
    """Performs K-fold cross-validation on a matrix with the F1 score as the error
    Inputs: 
        k - the maximum K value for K-nearest neighbour
        xMat - the matrix of features
        t_Points - target values
        
    Outputs:
        ListOfKNNMatrix_of_yCols - the list of KNN matrices for each fold
        listOfCrossValidErrorList - the list of cross-validation error lists for each k-nearest neighbour classifier
        listOfAverageCrossValidError - the list of average cross-validation errors for each k-nearest neighbour classifier
        BestKNN - the k value for k-nearest neighbour which gives the smallest average cross-validation error
    """
    
    t_Points = t_Points.reshape((len(t_Points),1))
    
    #Obtain list of K-fold training and validation sets
    ListOfX_trainCombinations, ListOfX_valid = splitMat(xMat,5)
    ListOft_trainCombinations, ListOft_valid = splitMat(t_Points,5)
    ListOfKNNMatrix_of_yCols = []

    #Fixes format of validation targets
    for i in range(len(ListOft_valid)): 
        ListOft_valid[i] = ListOft_valid[i].flatten()
    
    #Computes prediction values with knn using cross-validation 
    for X_train,t_train,X_test,t_test in zip(ListOfX_trainCombinations,ListOft_trainCombinations,ListOfX_valid,ListOft_valid): 
        ListOfKNNMatrix_of_yCols.append(CalcKNN(k,X_train,X_test,t_train,t_test))
    listOfCrossValidErrorList = []
    tempListOft_valid = []
    listOfAverageCrossValidError = []
    
    #Generates list of errors for each individual K-fold cross-validation test
    for K_NN in range(ListOfKNNMatrix_of_yCols[0].shape[1]):
        CrossValidErrorList = []
        for Dataset in range(len(ListOfKNNMatrix_of_yCols)):
            F1Score = computePeformanceParamaters(ListOfKNNMatrix_of_yCols[Dataset][:,K_NN], ListOft_valid[Dataset])[3]
            CrossValidErrorList.append(F1Score)
        listOfAverageCrossValidError.append(np.mean(CrossValidErrorList))
        listOfCrossValidErrorList.append(CrossValidErrorList)
        BestKNN = np.argmax(listOfAverageCrossValidError)+1
    return ListOfKNNMatrix_of_yCols,listOfCrossValidErrorList,listOfAverageCrossValidError,BestKNN


# In[15]:


def calculateKNNTestError_F1Score(K,X_train,X_test,t_train,t_test,BestKNN):
    """Computes the test error using the F1 score for the best k value for k-nearest neighbour as the classifier
    Inputs:
        K - the maximum K value for K-nearest neighbour 
        X_train - the training features
        X_test - the test features
        t_train - the training targets
        t_test - the test targets
        BestKNN - the k value for k-nearest neighbour which gives the smallest average cross-validation error
    
    Outputs:
        F1Score - the F1 score
    """
    
    #Obtains the predicted values using knn and cross-validation
    KNNMatrix_of_yCols = CalcKNN(K,X_train,X_test,t_train,t_test)
    y = KNNMatrix_of_yCols[:,BestKNN-1]
    
    #Calculates F1 score
    F1Score = computePeformanceParamaters(y, t_test)[3]
    return F1Score


# In[16]:


def scikit_ComputeW(X_train, t_train):
    """Computes the parameter vector using scikit-learn
    Inputs: 
        X_train - the training features
        t_train - the training targets
        numIterations - the number of iterations to improve the parameter vector values
        alpha - impacts how much the parameter vector values change by per iteration
    Outputs:
        W - the parameter vector
    """
    
    #Add training features to a column of ones
    xMat = np.ones((X_train.shape[0],X_train.shape[1]+1))
    xMat[:,1:] = X_train
    
    #Computes parameter vector
    classifier = LogisticRegression().fit(xMat, t_train)
    W = classifier.coef_[0]
    return W


# In[17]:


def scikit_kFoldCrossValidation_MissclassificationRate(k,xMat,t_Points):
    """Performs K-fold cross-validation on a matrix using scikit-learn, with misclassification rate as the error
    Inputs: 
        k - the maximum K value for K-nearest neighbour
        xMat - the matrix of features
        t_Points - target values
        
    Outputs:
        ListOfKNNMatrix_of_yCols - the list of KNN matrices for each fold
        listOfCrossValidErrorList - the list of cross-validation error lists for each k-nearest neighbour classifier
        listOfAverageCrossValidError - the list of average cross-validation errors for each k-nearest neighbour classifier
        BestKNN - the k value for k-nearest neighbour which gives the smallest average cross-validation error
    """
    
    #Reshape target points
    t_Points = t_Points.reshape((len(t_Points),1))
    
    #Obtain list of K-fold training and validation sets
    ListOfX_trainCombinations, ListOfX_valid = splitMat(xMat,5)
    ListOft_trainCombinations, ListOft_valid = splitMat(t_Points,5)
    ListOfKNNMatrix_of_yCols = []
    
    #Fixes format of validation targets
    for i in range(len(ListOft_valid)): 
        ListOft_valid[i] = ListOft_valid[i].flatten()
        
    #Generates list of errors for each individual K-fold cross-validation test
    for X_train,t_train,X_test,t_test in zip(ListOfX_trainCombinations,ListOft_trainCombinations,ListOfX_valid,ListOft_valid): 
        KNNMatrix_of_yCols = np.zeros([X_test.shape[0],k])
        for knn in range(k):
            t_train = t_train.reshape((t_train.shape[0],)) 
            classifier = KNeighborsClassifier(n_neighbors = knn+1)
            classifier.fit(X_train,t_train)
            KNNMatrix_of_yCols[:,knn] = classifier.predict(X_test)
        ListOfKNNMatrix_of_yCols.append(KNNMatrix_of_yCols)
    listOfCrossValidErrorList = []
    tempListOft_valid = []
    listOfAverageCrossValidError = []
    for K_NN in range(ListOfKNNMatrix_of_yCols[0].shape[1]):
        CrossValidErrorList = []
        for Dataset in range(len(ListOfKNNMatrix_of_yCols)):
            MissclassificationRate = computePeformanceParamaters(ListOfKNNMatrix_of_yCols[Dataset][:,K_NN], ListOft_valid[Dataset])[0]
            CrossValidErrorList.append(MissclassificationRate)
        listOfAverageCrossValidError.append(np.mean(CrossValidErrorList))
        listOfCrossValidErrorList.append(CrossValidErrorList)
        BestKNN = np.argmin(listOfAverageCrossValidError)+1
    return ListOfKNNMatrix_of_yCols,listOfCrossValidErrorList,listOfAverageCrossValidError,BestKNN


# In[18]:


def scikit_kFoldCrossValidation_F1Score(k,xMat,t_Points):
    """Performs K-fold cross-validation on a matrix using scikit-learn, with the F1 score as the error
    Inputs: 
        k - the maximum K value for K-nearest neighbour
        xMat - the matrix of features
        t_Points - target values
        
    Outputs:
        ListOfKNNMatrix_of_yCols - the list of KNN matrices for each fold
        listOfCrossValidErrorList - the list of cross-validation error lists for each k-nearest neighbour classifier
        listOfAverageCrossValidError - the list of average cross-validation errors for each k-nearest neighbour classifier
        BestKNN - the k value for k-nearest neighbour which gives the smallest average cross-validation error
    """
    
    #Reshape training points
    t_Points = t_Points.reshape((len(t_Points),1))
    
    #Get all folds for cross-validation
    ListOfX_trainCombinations, ListOfX_valid = splitMat(xMat,5)
    ListOft_trainCombinations, ListOft_valid = splitMat(t_Points,5)
    ListOfKNNMatrix_of_yCols = []
    
    #Fix format of validation targets
    for i in range(len(ListOft_valid)): 
        ListOft_valid[i] = ListOft_valid[i].flatten()
        
    #Performs 1 to k nearest neighbour for each fold
    for X_train,t_train,X_test,t_test in zip(ListOfX_trainCombinations,ListOft_trainCombinations,ListOfX_valid,ListOft_valid): 
        KNNMatrix_of_yCols = np.zeros([X_test.shape[0],k])
        for knn in range(k):
            t_train = t_train.reshape((t_train.shape[0],)) 
            classifier = KNeighborsClassifier(n_neighbors = knn+1)
            classifier.fit(X_train,t_train)
            KNNMatrix_of_yCols[:,knn] = classifier.predict(X_test)
        ListOfKNNMatrix_of_yCols.append(KNNMatrix_of_yCols)
    
    #Computes cross-validation errors
    listOfCrossValidErrorList = []
    tempListOft_valid = []
    listOfAverageCrossValidError = []
    for K_NN in range(ListOfKNNMatrix_of_yCols[0].shape[1]):
        CrossValidErrorList = []
        for Dataset in range(len(ListOfKNNMatrix_of_yCols)):
            F1Score = computePeformanceParamaters(ListOfKNNMatrix_of_yCols[Dataset][:,K_NN], ListOft_valid[Dataset])[3]
            CrossValidErrorList.append(F1Score)
        listOfAverageCrossValidError.append(np.mean(CrossValidErrorList))
        listOfCrossValidErrorList.append(CrossValidErrorList)
        BestKNN = np.argmax(listOfAverageCrossValidError)+1
    return ListOfKNNMatrix_of_yCols,listOfCrossValidErrorList,listOfAverageCrossValidError,BestKNN


# In[19]:


def printLogisticRegressionOutputs(BestThreshold_MissclassificationRate,LowestMissclassificationRate,BestThreshold_F1Score,HighestF1Score,W):
    """Prints the desired variables from logistic regression
    Inputs: 
        BestThreshold_MissclassificationRate - the threshold associated with the lowest misclassification rate
        LowestMissclassificationRate - the lowest misclassification rate
        BestThreshold_F1Score - the threshold associated with the highest F1 score
        HighestF1Score - the highest F1 score 
        W - the parameter vector
    """
    
    #Prints Logistic Regression Outputs
    print("Best Threshold For Missclassification Rate: ",BestThreshold_MissclassificationRate)
    print("Lowest Missclassification Rate: ",LowestMissclassificationRate)
    print()
    print("Best Threshold For F1 Score: ",BestThreshold_F1Score)
    print("Highest F1 Score: ",HighestF1Score)
    print()
    for i in range(W.shape[0]):
        print("W"+str(i)+": "+str(W[i]))


# In[20]:


def printKNNOutputs(listOfAverageCrossValidError_MissclassificationRate,BestKNN_For_MissclassificationRate,BestKNN_MissclassificationRate,listOfAverageCrossValidError_F1Score,BestKNN_For_F1Score,BestKNN_F1Score):
    """Prints the desired variables from KNN
    Inputs: 
        listOfAverageCrossValidError_MissclassificationRate - the list of average cross-validation errors (misclassification rate) for each k-nearest neighbour classifier
        BestKNN_For_MissclassificationRate - the k value for k-nearest neighbour which gives the smallest average cross-validation error (misclassification rate)
        BestKNN_MissclassificationRate - the test misclassification rate
        listOfAverageCrossValidError_F1Score - the list of average cross-validation errors (F1 score) for each k-nearest neighbour classifier
        BestKNN_For_F1Score - the k value for k-nearest neighbour which gives the smallest average cross-validation error (F1 score)
        BestKNN_F1Score - the test F1 score
    """
    
    #Prints KNN outputs
    for i in range(len(listOfAverageCrossValidError_MissclassificationRate)):
        print(str((i+1))+"-Nearest Neighbours, Average Cross-validation Error (Misclassification Rate): "+str(listOfAverageCrossValidError_MissclassificationRate[i]))
    print("Best KNN (Misclassification Rate): "+str(BestKNN_For_MissclassificationRate))
    print("Test Misclassification Rate: "+str(BestKNN_MissclassificationRate))
    print()
    for i in range(len(listOfAverageCrossValidError_F1Score)):
        print(str((i+1))+"-Nearest Neighbours, Average Cross-validation Error (F1 Score): "+str(listOfAverageCrossValidError_F1Score[i]))
    print("Best KNN (F1 Score): "+str(BestKNN_For_F1Score))
    print("Test F1 Score: "+str(BestKNN_F1Score))


# In[21]:


#load data set
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#store breast cancer data into X_data matrix
X_data, t = load_breast_cancer(return_X_y=True)

#split data into training set and test set
from sklearn.model_selection import train_test_split

#split the data into training and test data 
X_train, X_test, t_train, t_test = train_test_split(X_data, t, test_size = 0.2, random_state = 7878)

#normalize training and testing data 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#suppress scientific notation
np.set_printoptions(suppress=True)

#Extra variables
numIterations = 200
alpha = 0.1


# In[22]:


#Logistic Regression My Implementation
W = ComputeW(X_train, t_train, numIterations, alpha)
thresholdList = ComputeThresholdList(X_train,W)
MissclassificationRateList, PList, RList, F1List, BestThreshold_MissclassificationRate, LowestMissclassificationRate, BestThreshold_F1Score, HighestF1Score = ComputePeformanceParamatersList(X_test,t_test,W,thresholdList)
PlotPerfomanceCurves(thresholdList,MissclassificationRateList, PList, RList, F1List)
printLogisticRegressionOutputs(BestThreshold_MissclassificationRate,LowestMissclassificationRate,BestThreshold_F1Score,HighestF1Score,W)


# In[23]:


#Logistic Regression Scikit
W_Scikit = scikit_ComputeW(X_train, t_train)
thresholdList_Scikit = ComputeThresholdList(X_train,W_Scikit)
MissclassificationRateList_Scikit, PList_Scikit, RList_Scikit, F1List_Scikit, BestThreshold_MissclassificationRate_Scikit, LowestMissclassificationRate_Scikit, BestThreshold_F1Score_Scikit, HighestF1Score_Scikit = ComputePeformanceParamatersList(X_test,t_test,W_Scikit,thresholdList_Scikit)
PlotPerfomanceCurves(thresholdList_Scikit,MissclassificationRateList_Scikit, PList_Scikit, RList_Scikit, F1List_Scikit)
printLogisticRegressionOutputs(BestThreshold_MissclassificationRate_Scikit,LowestMissclassificationRate_Scikit,BestThreshold_F1Score_Scikit,HighestF1Score_Scikit,W_Scikit)


# In[24]:


#KNN Nearest Neighbours My Implementation
K = 5
ListOfKNNMatrix_of_yCols_MissclassificationRate,listOfCrossValidErrorList_MissclassificationRate,listOfAverageCrossValidError_MissclassificationRate,BestKNN_For_MissclassificationRate = kFoldCrossValidation_MissclassificationRate(K,X_train,t_train)
BestKNN_MissclassificationRate = calculateKNNTestError_MissclassificationRate(K,X_train,X_test,t_train,t_test,BestKNN_For_MissclassificationRate)
ListOfKNNMatrix_of_yCols_F1Score,listOfCrossValidErrorList_F1Score,listOfAverageCrossValidError_F1Score,BestKNN_For_F1Score = kFoldCrossValidation_F1Score(K,X_train,t_train)
BestKNN_F1Score = calculateKNNTestError_F1Score(K,X_train,X_test,t_train,t_test,BestKNN_For_F1Score)
printKNNOutputs(listOfAverageCrossValidError_MissclassificationRate,BestKNN_For_MissclassificationRate,BestKNN_MissclassificationRate,listOfAverageCrossValidError_F1Score,BestKNN_For_F1Score,BestKNN_F1Score)


# In[25]:


#KNN Nearest Neighbours Scikit
K_Scikit = 5
ListOfKNNMatrix_of_yCols_Scikit_MissclassificationRate,listOfCrossValidErrorList_Scikit_MissclassificationRate,listOfAverageCrossValidError_Scikit_MissclassificationRate,BestKNN_Scikit_For_MissclassificationRate = scikit_kFoldCrossValidation_MissclassificationRate(K_Scikit,X_train,t_train)
BestKNN_Scikit_MissclassificationRate = calculateKNNTestError_MissclassificationRate(K_Scikit,X_train,X_test,t_train,t_test,BestKNN_Scikit_For_MissclassificationRate)
ListOfKNNMatrix_of_yCols_Scikit_F1Score,listOfCrossValidErrorList_Scikit_F1Score,listOfAverageCrossValidError_Scikit_F1Score,BestKNN_Scikit_For_F1Score = scikit_kFoldCrossValidation_F1Score(K_Scikit,X_train,t_train)
BestKNN_Scikit_F1Score = calculateKNNTestError_F1Score(K_Scikit,X_train,X_test,t_train,t_test,BestKNN_Scikit_For_F1Score)
printKNNOutputs(listOfAverageCrossValidError_Scikit_MissclassificationRate,BestKNN_Scikit_For_MissclassificationRate,BestKNN_Scikit_MissclassificationRate,listOfAverageCrossValidError_Scikit_F1Score,BestKNN_Scikit_For_F1Score,BestKNN_Scikit_F1Score)


# In[ ]:





# In[ ]:





# In[ ]:




