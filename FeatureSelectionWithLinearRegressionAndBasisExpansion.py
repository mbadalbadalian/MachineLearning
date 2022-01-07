#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[2]:


def createXMat(X_set,deg):
    """Creates the x matrix
    Inputs: 
        X_set - the input values of the points
        deg - the degree of the polynomial
        
    Outputs:
        xMat - x in matrix form
    """
        
    xMat = np.zeros((X_set.size,deg+1)) #Initializes zero xMat
    for exponent in range(deg+1):
        xMat[:,exponent] = np.power(X_set, exponent) #Sets each column according to its polynomial degree 
    return xMat


# In[3]:


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


# In[4]:


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


# In[5]:


def kFoldCrossValidation(SMat,t_Points):
    """Performs K-fold cross-validation on a matrix
    Inputs: 
        SMat - the matrix of features ordered from best to worst
        t_Points - target values
        
    Outputs:
        subsetErrorList - list of error for each individual K-fold cross-validation test
        averageError - average of all K-fold cross-validation tests
    """
    
    t_Points = t_Points.reshape((len(t_Points),1))
    
    #Obtain list of K-fold training and validation sets
    ListOfS_trainCombinations, ListOfS_valid = splitMat(SMat,5)
    ListOft_trainCombinations, ListOft_valid = splitMat(t_Points,5)
    subsetErrorList = []
    i = 0
    
    #Generates list of errors for each individual K-fold cross-validation test
    for x_trainMat,t_train,x_validMat,t_valid in zip(ListOfS_trainCombinations,ListOft_trainCombinations,ListOfS_valid,ListOft_valid): 
        Error = GenerateWVectAndError(x_trainMat,t_train,x_validMat,t_valid)[1];
        subsetErrorList.append(Error)
        i += 1
        
    averageError = sum(subsetErrorList)/len(subsetErrorList) #Computes average of all K-fold cross-validation tests
    return subsetErrorList, averageError


# In[6]:


def reorder(X_test,XFeatNumList):
    """Creates the testing matrix based upon the order of best to worst feature
    Inputs: 
        X_test - test set
        XFeatNumList - ordered list containing number of best feature to worst going from left to right
        
    Outputs:
        SMat_test - the ordered testing matrix
    """
    
    SMat_test = np.ones((X_test.shape[0],(X_test.shape[1]+1))) #Initializes the ordered testing matrix
    
    #Creates the testing matrix based upon the order of best to worst feature
    for i in range(1,SMat_test.shape[1]):
        SMat_test[:,i] = X_test[:,(XFeatNumList[i-1]-1)]
    return SMat_test


# In[7]:


def basisExpandValid(X_testMatReordered,bestDegreeList):
    """Applies basis expansion to each feature based on the ideal degree 
    Inputs: 
        X_testMatReordered - the ordered testing matrix
        bestDegreeList - list of ideal degrees for each feature going from the best to worst feature
        
    Outputs:
        X_testMatBasisExpansion - the testing matrix with basis expansion
    """
    
    X_testMatBasisExpansion = np.ones((X_testMatReordered.shape[0],X_testMatReordered.shape[1])) #Initializes the testing matrix with basis expansion
    
    #Applies basis expansion to each feature based on the ideal degree and adds it to X_testMatBasisExpansion
    for i in range(1,X_testMatBasisExpansion.shape[1]):
        X_testMatBasisExpansion[:,i] = np.power(X_testMatReordered[:,i],bestDegreeList[i-1])
    return X_testMatBasisExpansion    


# In[8]:


def findMinAverageErrors(listofAverageErrorLists):
    """Computes the list of cross-validation error for the next best feature added to SMat
    Inputs: 
        listofAverageErrorLists - each element in this list contains a list for k = 1 to k = 13, where a given list contains the average cross-validation error for every feature that was tested 
        
    Outputs:
        listofMinAverageErrors - list of average cross-validation error for the next best feature added to SMat
    """
    
    listofMinAverageErrors = []
    
    #Computes the list of cross-validation error for the next best feature added to SMat
    for AverageErrorList in listofAverageErrorLists:
        listofMinAverageErrors.append(min(AverageErrorList))
    return listofMinAverageErrors


# In[9]:


def GenerateWVectAndError(x_trainMat,t_train,x_validMat,t_valid):
    """Computes the parameters using linear regression, as well as the validation error
    Inputs: 
        x_trainMat - the training matrix
        t_train - the training targets
        x_validMat - the validation matrix
        t_valid - the validation targets
        
    Outputs:
        WVect - the vector of parameters
        error - the validation error 
    """
    
    WVect = np.dot(np.linalg.inv(np.dot(x_trainMat.T, x_trainMat)), np.dot(x_trainMat.T,t_train)) #Computes the vector of parameters
    error = np.dot(np.subtract(t_valid,np.dot(x_validMat,WVect)).T,np.subtract(t_valid,np.dot(x_validMat,WVect)))/len(t_valid) #Computes the validation error
    
    #Ensures that error returned is of type float
    if isinstance(error, np.float64):
        return WVect, error
    else:
        return WVect, error[0][0]


# In[10]:


def XnFeatKFoldCrossValidation(SMat,XFeat,t_Points):
    """Adds a feature to SMat and computes the K-fold cross-validation error
    Inputs: 
        SMat - the matrix of features ordered from best to worst
        XFeat - the values of a given feature
        t_Points - the target points
        
    Outputs:
        subsetErrorList - list of error for each individual K-fold cross-validation test
        averageError - average of all K-fold cross-validation tests
    """
    
    XFeat = XFeat.reshape((len(XFeat),1))
    SMat = np.concatenate((SMat, XFeat), axis = 1) #Adds a feature to SMat
    subsetErrorList,averageError = kFoldCrossValidation(SMat,t_Points) #Computes the K-fold cross-validation error
    return subsetErrorList,averageError


# In[11]:


def FindFeatureWithSmallestAverageError(SMat,k,XFeatMat,t_Points,listofSubsetErrorMat,listofAverageErrorLists):
    """Creates the S matrix by doing k-fold cross-validation for every feature 
    Inputs: 
        SMat - the matrix of features ordered from best to worst
        k - the number of features wanted to be added to SMat
        XFeatMat - the matrix containing the features
        t_Points - the target values
        listofSubsetErrorMat - list of matricies containing the error for each individual K-fold cross-validation test, for each feature that was tested
        listofAverageErrorLists - each element in this list contains a list for k = 1 to k = 13, where a given list contains the average cross-validation error for every feature that was tested
        
    Outputs:
        SMat - the matrix of features ordered from best to worst
        listofSubsetErrorMat - list of matricies containing the error for each individual K-fold cross-validation test, for each feature that was tested
        listofAverageErrorLists - each element in this list contains a list for k = 1 to k = 13, where a given list contains the average cross-validation error for every feature that was tested
    """
    
    #Ends the recursion once the SMatrix has been filled by the desired number of parameters
    if (XFeatMat.shape[1] == 0) or (k == 0):
        return SMat, listofSubsetErrorMat, listofAverageErrorLists  
    subsetErrorMat = []
    averageErrorList = []
    
    #Computes the average cross-validation error for each feature
    for i in range(XFeatMat.shape[1]):
        subsetErrorList,averageError = XnFeatKFoldCrossValidation(SMat,XFeatMat[:,i],t_Points)
        subsetErrorMat.append(subsetErrorList)
        averageErrorList.append(averageError)
    
    #Finds the index of the feature which produced the smaller average cross-validation error and adds it to the S Matrix
    minAverageErrorIndex = averageErrorList.index(min(averageErrorList)) 
    SMat = np.concatenate((SMat, XFeatMat[:,minAverageErrorIndex].reshape((len(XFeatMat[:,minAverageErrorIndex]),1))), axis = 1)
    
    #Adds the matrix of cross-validation error, as well as the average cross-validation error
    listofSubsetErrorMat.append(subsetErrorMat)
    listofAverageErrorLists.append(averageErrorList)
    
    #Deletes the feature that was inputted from XFeatMat
    XFeatMat = np.delete(XFeatMat, minAverageErrorIndex, axis=1)
    k -= 1
    return FindFeatureWithSmallestAverageError(SMat,k,XFeatMat,t_Points,listofSubsetErrorMat,listofAverageErrorLists)


# In[12]:


def GenerateW_vectors(SMat_train,t_train,SMat_valid,t_valid):
    """Computes the parameters and validation error for each training set
    Inputs: 
        SMat_train - the training matrix
        t_train - the target values
        SMat_valid - the validation matrix
        t_valid - the validation matrix
        
    Outputs:
        WVectMat - the matrix containing the parameter list for each training set
        ErrorList - the list containing the validation error for each validation set
    """
    
    WVectMat = []
    ErrorList = []
    
    #Computes the parameters and validation error for each training set and adds it to the matrix/list
    for i in range(2,SMat_train.shape[1]+1):
        WVect, Error = GenerateWVectAndError(SMat_train[:,:i],t_train,SMat_valid[:,:i],t_valid)
        WVectMat.append(WVect)
        ErrorList.append(Error)
    return WVectMat, ErrorList


# In[13]:


def FindOrderOfXFeat(SMat,xFeatMat):
    """Returns an array with the first feature's number to the last feature's number that was added to the S matrix
    Inputs: 
        SMat - the matrix of features ordered from best to worst
        xFeatMat - the matrix of features ordered from x1 to xn
        
    Outputs:
        XFeatNumList - array with the first feature's number to the last feature's number that was added to the S matrix
    """
    
    XFeatNumList = []
    #Computes an array with the first feature's number to the last feature's number that was added to the S matrix
    for i in range(1,SMat.shape[1]):
        SList = SMat[:,i]
        for j in range(xFeatMat.shape[1]):
            xFeat = xFeatMat[:,j]
            if np.array_equal(xFeat,SList):
                XFeatNum = j + 1;
                XFeatNumList.append(XFeatNum)
                break
    return XFeatNumList


# In[14]:


def FindXnFeatBestBasisExpansion(SMatWithBasisExpansion,XFeat,t_Points,deg):
    """Finds the best degree to raise the XFeat vector to and adds it to SMatWithBasisExpansion
    Inputs: 
        SMatWithBasisExpansion - the matrix of features ordered from best to worst with basis expansion
        XFeat - the values of a given feature
        t_Points - the target values
        deg - the maximum degree of xFeat to test out
        
    Outputs:
        SMatWithBasisExpansion - the matrix of features ordered from best to worst with basis expansion
        subsetErrorMat - matrix containing the error for each individual K-fold cross-validation test, for each feature that was tested
        averageErrorList - the list of average cross-validation errors for each feature
        bestDegree - the best degree used for basis expansion on the given XFeat vector
    """
    
    XFeatExpandedMat = createXMat(XFeat,deg) #creates XFeatExpandedMat by raising the column vector XFeat to the power of 1 to deg
    subsetErrorMat = []
    averageErrorList = []
    
    #Temporarily adds the XFeat raised to the degree n to SMatWithBasisExpansion to compute the k-fold cross-validation and computes cross-validation error
    for i in range(1,deg+1):
        subsetErrorList,averageError = XnFeatKFoldCrossValidation(SMatWithBasisExpansion,XFeatExpandedMat[:,i],t_Points)
        subsetErrorMat.append(subsetErrorList)
        averageErrorList.append(averageError)
        
    #Finds the best degree for the XFeat to be raised to and adds it to SMatWithBasisExpansion
    bestDegree = averageErrorList.index(min(averageErrorList))+1
    SMatWithBasisExpansion = np.concatenate((SMatWithBasisExpansion, XFeatExpandedMat[:,bestDegree].reshape((len(XFeatExpandedMat[:,bestDegree]),1))), axis = 1)
    return SMatWithBasisExpansion, subsetErrorMat, averageErrorList, bestDegree


# In[15]:


def GenerateSMatBasisExpansion(SMat,t_Points,deg):
    """Finds the best degree to raise each XFeat vector in SMAT to and adds it to SMatWithBasisExpansion
    Inputs: 
        SMat - the matrix of features ordered from best to worst
        t_Points - the target values
        deg - the maximum degree of xFeat to test out
        
    Outputs:
        SMatWithBasisExpansion - the matrix of features ordered from best to worst with basis expansion
        listofSubsetErrorMat - list of matricies containing the error for each individual K-fold cross-validation test, for each feature that was tested
        listofAverageErrorLists - each element in this list contains a list for k = 1 to k = 13, where a given list contains the average cross-validation error for every feature that was tested
        bestDegreeList - the list of the best degree used for basis expansion for each XFeat vector
    """
    
    listofSubsetErrorMat = []
    listofAverageErrorLists = []
    bestDegreeList = []
    SMatWithBasisExpansion = np.ones((SMat.shape[0],1)) #Initializing SMatWithBasisExpansion
    #Finds the best degree to raise each XFeat vector in SMAT to and adds it to SMatWithBasisExpansion
    for i in range(1,SMat.shape[1]):
        SMatWithBasisExpansion, subsetErrorMat, averageErrorList, bestDegree = FindXnFeatBestBasisExpansion(SMatWithBasisExpansion,SMat[:,i],t_Points,deg)
        listofSubsetErrorMat.append(subsetErrorMat)
        listofAverageErrorLists.append(averageErrorList)
        bestDegreeList.append(bestDegree)
    return SMatWithBasisExpansion, listofSubsetErrorMat, listofAverageErrorLists, bestDegreeList


# In[16]:


def plotAllErrors(TestErrorList,TestErrorListBasisExpansion,listofMinAverageCrossValidErrors,listofMinAverageCrossValidErrorsBasisExpansion,k):
    """
    Inputs: 
        TestErrorList - the list of test errors
        TestErrorListBasisExpansion - the list of test errors with basis expansion
        listofMinAverageCrossValidErrors - the list of average cross-valid errors for each feature that was added to SMat 
        listofMinAverageCrossValidErrorsBasisExpansion -  the list of average cross-valid errors with basis expansion for each feature that was added to SMatWithBasisExpansion 
        k - the number of features wanted to be added to SMat
    """
    
    #Creates the range of k values going from 1 to k
    kList = list(range(1,k+1))
    #Plotting the curves
    plt.scatter(kList, listofMinAverageCrossValidErrors, color = 'blue', label = '5-fold cross-validation error')
    plt.scatter(kList, TestErrorList, color = 'magenta', label = 'Test error') 
    plt.scatter(kList, listofMinAverageCrossValidErrorsBasisExpansion, color = 'green', label = '5-fold cross-validation error with basis expansion')
    plt.scatter(kList, TestErrorListBasisExpansion, color = 'red', label = 'Test error with basis expansion')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Errors for k = 1 to k = 13")
    plt.xlabel("Number of features, k")
    plt.ylabel("Error")
    plt.show()


# In[17]:


def printOutputs(XFeatNumList,WVectMat,bestDegreeList,WVectMatBasisExpansion):
    """Prints out useful information
    Inputs: 
        XFeatNumList -
    """
    
    #Prints out the order of x features added to SMat
    string = "Order of X features added to SMat: ["
    for i in range(len(XFeatNumList)-1):
        string += "x"+str(XFeatNumList[i])+", "
    string += "x"+str(XFeatNumList[len(XFeatNumList)-1])+"]"
    print(string)
    
    #Printing out the linear regression equations without basis expansion
    for i in range(len(WVectMat)):
        WVect = WVectMat[i]
        string1 = "Linear regression for k = "+str(i+1)+": \nY = "+str(WVect[0])+" + "
        k = 0
        for j in range(1,len(WVect)-1):
            string1 += str(WVect[j])+"*x"+str(XFeatNumList[k])+" + "
            k += 1
        string1 += str(WVect[len(WVect)-1])+"*x"+str(XFeatNumList[len(WVect)-2])
        print()
        print(string1)
    
    #Printing out the degrees raised for each feat for basis expansion
    print()
    string2 = "Degrees raised for each feat for basis expansion: ["
    for i in range(len(bestDegreeList)-1):
        if bestDegreeList[i] > 1:
            string2 += "x"+str(XFeatNumList[i])+"^"+str(bestDegreeList[i])+", "
        else:
            string2 += "x"+str(XFeatNumList[i])+", "
    if bestDegreeList[i] > 1:
        string2 += "x"+str(XFeatNumList[len(bestDegreeList)-1])+"^"+str(bestDegreeList[len(bestDegreeList)-1])+"]"
    else:
        string2 += "x"+str(XFeatNumList[len(bestDegreeList)-1])+"]"
    print(string2)
    
    #Printing out the linear regression equations with basis expansion
    for i in range(len(WVectMatBasisExpansion)):
        WVectBasisExpansion = WVectMatBasisExpansion[i]
        string3 = "Linear regression with basis expansion for k = "+str(i+1)+": \nY = "+str(WVectBasisExpansion[0])+" + "
        k = 0
        for j in range(1,len(WVectBasisExpansion)-1):
            if bestDegreeList[k] > 1:
                string3 += str(WVectBasisExpansion[j])+"*x"+str(XFeatNumList[k])+"^"+str(bestDegreeList[k])+" + "
            else:
                string3 += str(WVectBasisExpansion[j])+"*x"+str(XFeatNumList[k])+" + "
            k += 1
        if bestDegreeList[k] > 1:
            string3 += str(WVectBasisExpansion[len(WVectBasisExpansion)-1])+"*x"+str(XFeatNumList[len(WVectBasisExpansion)-2])+"^"+str(bestDegreeList[len(WVectBasisExpansion)-2])
        else:
            string3 += str(WVectBasisExpansion[len(WVectBasisExpansion)-1])+"*x"+str(XFeatNumList[len(WVectBasisExpansion)-2])
        print()
        print(string3)


# In[ ]:





# In[18]:


#load data set
from sklearn.datasets import load_boston
housing = load_boston()
X_data, t = load_boston(return_X_y=True)
#split data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X_data, t, test_size = 0.2, random_state = 7878)
#Initializing extra variables
SMat = np.ones((X_train.shape[0],1))
k = 13
listofSubsetErrorMat = []
listofAverageErrorLists = []
degree = 10


# In[19]:


#Get linear model for each function
SMat, listofSubsetErrorMat, listofAverageCrossValidErrorLists = FindFeatureWithSmallestAverageError(SMat,k,X_train,t_train,listofSubsetErrorMat,listofAverageErrorLists)
XFeatNumList = FindOrderOfXFeat(SMat,X_train)
SMat_test = reorder(X_test,XFeatNumList)
WVectMat, TestErrorList = GenerateW_vectors(SMat,t_train,SMat_test,t_test)
listofMinAverageCrossValidErrors = findMinAverageErrors(listofAverageCrossValidErrorLists) 

#Get linear model for each function with basis expansion
SMatWithBasisExpansion, listofSubsetErrorMat, listofAverageCrossValidErrorListsBasisExpansion, bestDegreeList = GenerateSMatBasisExpansion(SMat,t_train,degree)
SMat_testBasisExpansion = basisExpandValid(SMat_test,bestDegreeList)
WVectMatBasisExpansion, TestErrorListBasisExpansion = GenerateW_vectors(SMatWithBasisExpansion,t_train,SMat_testBasisExpansion,t_test)
listofMinAverageCrossValidErrorsBasisExpansion = findMinAverageErrors(listofAverageCrossValidErrorListsBasisExpansion)

plotAllErrors(TestErrorList,TestErrorListBasisExpansion,listofMinAverageCrossValidErrors,listofMinAverageCrossValidErrorsBasisExpansion,k)


# In[20]:


printOutputs(XFeatNumList,WVectMat,bestDegreeList,WVectMatBasisExpansion)


# In[ ]:




