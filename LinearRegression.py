#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[2]:


def createXMat(X_set,deg):
    """Creates the x matrix.
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


def generateX_matrices(X_set):
    """Generates list of n x_matrices of degree 0 to 9.
    Inputs: 
        X_set - the input values or features.
        
    Outputs:
        X_matrices - list of n x_matrices to degree n.
    """
    
    X_matrices = [createXMat(X_set, degree) for degree in range(9+1)]
    return X_matrices


# In[4]:


def createWVector(xMat,t_Points):
    """Generates the W vector of parameters.
    Inputs: 
        xMat - the x matrix for a given degree.
        t_Points - the output values of the points.
        
    Outputs:
        WVect - the w vector. 
    """
    
    WVect = np.dot(np.linalg.inv(np.dot(xMat.T, xMat)), np.dot(xMat.T,t_Points))
    return WVect


# In[5]:


def generateW_vectors(X_matrices,t_Points):
    """Generates list of n Wvects of a given degree.
    Inputs: 
        X_matrices - list of n x_matrices to degree n.
        t_Points - the output values of the points.
        
    Outputs:
        W_vectors - list of n W_vectors of degree n. 
    """

    W_vectors = [createWVector(x,t_Points) for x in X_matrices]
    return W_vectors


# In[6]:


def generateError(WVect, xMat, T):
    """Generates error between predictor of degree n and set of output data.
    Inputs: 
        WVect - the w vector.
        xMat - x in matrix form.
        T - set of output data.
        
    Outputs:
        error- error between predictor of degree n and set of data.
    """
    
    error = np.dot(np.subtract(T,np.dot(xMat,WVect)).T,np.subtract(T,np.dot(xMat,WVect)))/len(T)
    return error 


# In[7]:


def generateError_list(W_vectors, X_matrices, T):
    """Generates list of errors for each degree.
    Inputs: 
        W_vectors - list of n WVects of each degree. 
        X_matrices - list of n xMats of each degree n.
        T - set of output data.
        
    Outputs:
        error_list - list of errors for each degree.
    """
    
    error_list = np.array([generateError(WVect,xMat,T) for WVect, xMat in zip(W_vectors, X_matrices)])
    return error_list.tolist()


# In[8]:


def generateAverageError(x_Vect, t_Vect):
    """Generates the average error between the true function and the targets.
    Inputs: 
        x_Vect - the input values or features.
        t_Vect - the target values or sample outputs.
        
    Outputs:
        average_error - list of errors for each degree.
    """
    
    y_true = np.sin(4*np.pi*x_Vect)
    average_error = np.sqrt(np.dot(np.subtract(y_true,t_Vect).T,np.subtract(y_true,t_Vect))/len(t_Vect))
    return average_error 


# In[9]:


def generateNdegPlot(xMat_train, xMat_valid, X_train, X_valid, t_train, t_valid, WVect, deg):
    """Generates a plot for an xMat of degree n.
    Inputs: 
        xMat_train - the X matrix of degree n formed from the training set.
        xMat_valid - the X matrix of degree n formed from the validation set.
        X_train - the training set.
        X_valid - the validation set.
        t_train - the targets or outputs from the training set.
        t_valid - the targets or outputs from the validation set.
        WVect - the vector of parameters
        deg - n degree
    """
    
    x_true = np.linspace(0.0,1.0,1000) #Generates 1000 points uniforly distributed over the range of 0 to 1
    y_train = np.dot(xMat_train,WVect) #Generates predictor from training inputs
    y_valid = np.dot(xMat_valid,WVect) #Generates predictor from validation inputs
    y_true = np.sin(4*np.pi*x_true) #Generates true function curve
    
    #Plotting the curves
    plt.scatter(x_true, y_true, color = 'green', label = 'True function') 
    plt.scatter(X_valid, y_valid, color = 'red', label = 'Prediction with validation')
    plt.scatter(X_train, y_train, color = 'yellow', label = 'Prediction with training')
    plt.scatter(X_valid, t_valid, color = 'blue', label = 'Validation points')
    plt.scatter(X_train, t_train, color = 'magenta', label = 'Training points')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Predicting f(x) = sin(4*pi*x) for degree "+str(deg))
    plt.xlabel("Inputs x")
    plt.ylabel("Outputs, f(x)")
    plt.show()


# In[10]:


def generateAllDegreePlots(X_train_matrices, X_valid_matrices, X_train, X_valid, t_train, t_valid, W_vectors):
    """Generates plots for all xMat of all degrees.
    Inputs: 
        X_train_matrices - list of training X matrices.
        X_valid_matrices - list of validation X matrices.
        X_train - the training set.
        X_valid - the validation set.
        t_train - the targets or outputs from the training set.
        t_valid - the targets or outputs from the validation set.
        W_vectors - the list of wVects.
    """
    
    for deg in range(10):
        generateNdegPlot(X_train_matrices[deg], X_valid_matrices[deg], X_train, X_valid, t_train, t_valid, W_vectors[deg], deg)


# In[11]:


def generateErrorPlots(train_error_list,valid_error_list,average_target_true_error,train_error_Fit,valid_error_Fit,train_error_Underfit,valid_error_Underfit,lamdaFit,lamdaUnderfit):
    """Generates a plot of the error.
    Inputs: 
        train_error_list - the list of training errors for each degree.
        valid_error_list - the list of validation errors for each degree.
        average_target_true_error - the average error between the targets and the true function
        train_error_Fit - the training error for the fitted 9th degree predictor.
        valid_error_Fit - the validation error for the fitted 9th degree predictor.
        train_error_Underfit - the training error for the underfitted 9th degree predictor.
        valid_error_Underfit - the training error for the underfitted 9th degree predictor.
        lamdaFit - the lamda value for the fit 9th degree predictor.
        lamdaUnderfit - the lamda value for the underfit 9th degree predictor.
    """

    deg = list(range(10)) #Generates a list representing each degree from 0 to 9
    average_target_true_error_list = np.zeros((1,10)) #Generates an array of size 10 filled wth zeros
    average_target_true_error_list[:] = average_target_true_error #Sets each value of the array to the average_target_true_error
    
    #Plotting the curves
    plt.scatter(deg, train_error_list, color = 'magenta', label = 'Training error') 
    plt.scatter(deg, valid_error_list, color = 'blue', label = 'Validation error')
    plt.scatter(deg, average_target_true_error_list, color = 'red', label = 'Average error between true function and targets')
    plt.scatter(9,train_error_Fit, color = 'black', label = 'Training Error With Regularization, ln(lamda) = '+str(np.log(lamdaFit)))
    plt.scatter(9,valid_error_Fit, color = 'cyan', label = 'Validation Error With Regularization, ln(lamda) = '+str(np.log(lamdaFit)))
    plt.scatter(9,train_error_Underfit, color = 'orange', label = 'Training Error With Regularization, lamda = '+str(lamdaUnderfit))
    plt.scatter(9,valid_error_Underfit, color = 'green', label = 'Validation Error With Regularization, lamda = '+str(lamdaUnderfit))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Error for degrees 0 to 9")
    plt.xlabel("Degree, M")
    plt.ylabel("Error")
    plt.show()


# In[12]:


def standardizeParametersMatrices(XMat_train,XMat_valid):
    """Standardizes the X vectors.
    Inputs: 
        X_train - the training set.
        X_valid - the validation set.
        
    Outputs:
        X_train_standardized_Mat - the X matrix of degree n formed from the standardized training set.
        X_valid_standardized_Mat - the X matrix of degree n formed from the standardized validation set.
    """

    X_train_standardized_Mat = np.ones(XMat_train.shape)
    X_valid_standardized_Mat = np.ones(XMat_valid.shape)

    sc = StandardScaler()
    XMax_train = sc.fit_transform(XMat_train[:,1:])
    XMax_valid = sc.transform(XMat_valid[:,1:])
    X_train_standardized_Mat[:,1:] = XMax_train
    X_valid_standardized_Mat[:,1:] = XMax_valid
    
    return X_train_standardized_Mat, X_valid_standardized_Mat


# In[13]:


def createBMatrix(lamda):
    """Creates the B Matrix.
    Inputs: 
        lamda - the lamda value chosen for regularization.
        
    Outputs:
        BMat - the B matrix generated from the lamda value.
    """
    
    BMat = np.zeros((10,10)) #Creates a 10x10 matrix filled with zeros
    
    #Fills each diagonal except for the first element (row = 0, col = 0) with 2*lamda
    for i in range(1,10): 
        BMat[i,i] = 2*lamda
    return BMat 


# In[14]:


def regularizedW(xMat,BMat,t_Vect):
    """Regularizes the W vector.
    Inputs: 
        xMat - the standardized x matrix.
        BMat - the B matrix generated from the lamda value.
        t_Vect - the vector of target values.
        
    Outputs:
        w_Vec - the vector of predictors.
    """

    w_Vec = np.dot(np.linalg.inv(np.add(np.dot(xMat.T, xMat),(t_Vect.size/2)*BMat)),np.dot(xMat.T,t_Vect))
    return w_Vec


# In[15]:


def generateRegularizationPlots(xMat_train, xMat_valid, X_train, X_valid, t_train, t_valid, WVectFit, WVectUnderfit, lamdaFit, lamdaUnderfit):
    """Generates plots for the regularized matrices.
    Inputs: 
        xMat_train - the X matrix of degree n formed from the training set.
        xMat_valid - the X matrix of degree n formed from the validation set.
        X_train - the training set.
        X_valid - the validation set.
        t_train - the targets or outputs from the training set.
        t_valid - the targets or outputs from the validation set.
        WVectFit - the vector of parameters representing the fitted 9th degree predictor.
        WVectUnderfit - the vector of parameters representing the underfitted 9th degree predictor.
        lamdaFit - the lamda value for the fitted 9th degree predictor.
        lamdaUnderfit - the lamda value for the underfitted 9th degree predictor. 
    """
    
    x_true = np.linspace(0.0,1.0,1000) #Generates 1000 points uniforly distributed over the range of 0 to 1
    y_train = np.dot(xMat_train, WVectFit) #Generates fitted predictor from training inputs
    y_valid = np.dot(xMat_valid, WVectFit) #Generates fitted predictor from validation inputs
    y_true = np.sin(4*np.pi*x_true) #Generates true function curve
    
    #Plotting the predictor representing the 9th degree with fitting 
    plt.scatter(x_true, y_true, color = 'green', label = 'True function')
    plt.scatter(X_valid, y_valid, color = 'red', label = 'Prediction with validation')
    plt.scatter(X_train, y_train, color = 'yellow', label = 'Prediction with training')
    plt.scatter(X_valid, t_valid, color = 'blue', label = 'Validation points')
    plt.scatter(X_train, t_train, color = 'magenta', label = 'Training points')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Predicting f(x) = sin(4*pi*x) for degree 9 with regularization, ln(lambda) = "+str(np.log(lamdaFit)))
    plt.xlabel("Inputs x")
    plt.ylabel("Outputs, f(x)")
    plt.show()
    
    
    y_train_Underfit = np.dot(xMat_train, WVectUnderfit) #Generates underfitted predictor from training inputs  
    y_valid_Underfit = np.dot(xMat_valid, WVectUnderfit) #Generates underfitted predictor from training inputs
    
    #Plotting the predictor representing the 9th degree with underfitting
    plt.scatter(x_true, y_true, color = 'green', label = 'True function')
    plt.scatter(X_valid, y_valid_Underfit, color = 'red', label = 'Prediction with validation')
    plt.scatter(X_train, y_train_Underfit, color = 'yellow', label = 'Prediction with training')
    plt.scatter(X_valid, t_valid, color = 'blue', label = 'Validation points')
    plt.scatter(X_train, t_train, color = 'magenta', label = 'Training points')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Predicting f(x) = sin(4*pi*x) for degree 9 with regularization, lambda = "+str(lamdaUnderfit))
    plt.xlabel("Inputs x")
    plt.ylabel("Outputs, f(x)")
    plt.show()


# In[16]:


#Generating sample points
X_train = np.linspace(0.0,1.0,10) # training set
X_valid = np.linspace(0.0,1.0,100) # validation set
np.random.seed(7878)
t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)
t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
lamdaFit = np.exp(-18);
lamdaUnderfit = 20;


# In[17]:


#Main Code
X_train_matrices = generateX_matrices(X_train)
W_vectors = generateW_vectors(X_train_matrices,t_train)
train_error_list = generateError_list(W_vectors, X_train_matrices, t_train)

X_valid_matrices = generateX_matrices(X_valid)
valid_error_list = generateError_list(W_vectors, X_valid_matrices, t_valid)
average_target_true_error = generateAverageError(X_valid, t_valid)

generateAllDegreePlots(X_train_matrices, X_valid_matrices, X_train, X_valid, t_train, t_valid, W_vectors)

#Regularization
X_train_standardized_Mat, X_valid_standardized_Mat = standardizeParametersMatrices(X_train_matrices[9],X_valid_matrices[9])
BMat_Fit = createBMatrix(lamdaFit)
w_Fit = regularizedW(X_train_standardized_Mat,BMat_Fit,t_train)
train_error_Fit = generateError(w_Fit, X_train_standardized_Mat, t_train)
valid_error_Fit = generateError(w_Fit, X_valid_standardized_Mat, t_valid)
BMat_Underfit = createBMatrix(lamdaUnderfit)
w_Underfit = regularizedW(X_train_standardized_Mat,BMat_Underfit,t_train)
train_error_Underfit = generateError(w_Underfit, X_train_standardized_Mat, t_train)
valid_error_Underfit = generateError(w_Underfit, X_valid_standardized_Mat, t_valid)
    
generateRegularizationPlots(X_train_standardized_Mat, X_valid_standardized_Mat, X_train, X_valid, t_train, t_valid, w_Fit, w_Underfit, lamdaFit, lamdaUnderfit)
generateErrorPlots(train_error_list,valid_error_list,average_target_true_error,train_error_Fit,valid_error_Fit,train_error_Underfit,valid_error_Underfit,lamdaFit, lamdaUnderfit)


# In[ ]:




