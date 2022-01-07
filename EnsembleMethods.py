#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[2]:


def kFoldCrossValidation(classifier,X_train,t_train):
    """Performs 5-fold cross-validation on a classifier
    Inputs: 
        classifier - base classifier
        X_train - the training features
        t_train - the training targets 
        
    Outputs:
        AverageCrossValidError - the average cross-validation error for the decision tree
    """
    
    #Performs 5-fold cross-validation on a classifier
    AverageCrossValidError = 1-cross_val_score(classifier,X_train,t_train).mean() #Average of 5-Fold Cross-Validation
    return AverageCrossValidError


# In[3]:


def computeDecisionTree2ToNMaxLeafNodes(largestMaxLeafNodes,X_train,t_train):
    """Computes the average cross-validation error for each decision with 2 to largestMaxLeafNodes as max_leaf_nodes
    Inputs: 
        largestMaxLeafNodes - largest maximimum number of leaf nodes
        X_train - the training features
        t_train - the training targets 
        
    Outputs:
        maxLeafNodesList - the list of the number of maximum leaf nodes
        bestAverageCrossValidError - the best average cross-validation error for the decision tree
        bestMaxLeafNodes - best maximum leaf nodes
        AverageCrossValidErrorList - list of average cross-validation errors corresponding to each decision tree
    """
    
    #Computes the average cross-validation error for each decision with 2 to largestMaxLeafNodes as max_leaf_nodes
    AverageCrossValidErrorList = [kFoldCrossValidation(DecisionTreeClassifier(max_leaf_nodes=maxLeafNodes,random_state=7878),X_train,t_train) for maxLeafNodes in range(2,largestMaxLeafNodes+1)]
    #Creates list of number of max_leaf_nodes used
    maxLeafNodesList = [maxLeafNodes for maxLeafNodes in range(2,largestMaxLeafNodes+1)]
    #Finds the optimal number of maximum leaf nodes
    bestMaxLeafNodes = np.argmin(AverageCrossValidErrorList)+2
    #Finds the best average cross-validation error
    bestAverageCrossValidError = np.amin(AverageCrossValidErrorList)
    return maxLeafNodesList,bestMaxLeafNodes,bestAverageCrossValidError,AverageCrossValidErrorList


# In[4]:


def computeBaggingClassifier(baseClassifier,numEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the error using bagging classifiers
    Inputs:
        baseClassifier - the base classifier
        numEstimators - the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        baggingError - the error obtained from the bagging classifier
    """
    
    #Creating the classifier
    bagging_classifier = BaggingClassifier(base_estimator=baseClassifier,max_samples=0.5,n_estimators=numEstimators,random_state=7878).fit(X_train,t_train)
    #Calculating error
    baggingError = 1-bagging_classifier.score(X_valid,t_valid)
    return baggingError


# In[5]:


def computeNBaggingClassifiers(baseClassifier,startNumEstimators,maxNumEstimators,incrementNumEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the bagging error for each bagging classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    Inputs:
        baseClassifier - the base classifier
        startNumEstimators - the start number of estimators
        maxNumEstimators - the maximum number of estimators
        incrementNumEstimators - the increment for the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        numEstimatorsList - list of the number of estimators used
        baggingErrorList - list of the bagging errors
    """
    
    #Initializing lists
    numEstimatorsList = np.zeros([1,int((maxNumEstimators-startNumEstimators)/incrementNumEstimators)+1])
    baggingErrorList = np.zeros([1,numEstimatorsList.shape[1]])
    i = 0
    
    #Computes the bagging error for each bagging classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    for numEstimators in range(startNumEstimators,maxNumEstimators+incrementNumEstimators,incrementNumEstimators):
        numEstimatorsList[0,i] = numEstimators
        baggingErrorList[0,i] = computeBaggingClassifier(baseClassifier,numEstimators,X_train,t_train,X_valid,t_valid)
        i += 1
    return numEstimatorsList,baggingErrorList


# In[6]:


def computeRandomForestClassifier(numEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the error using random forests
    Inputs:
        numEstimators - the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        randomForestError - the error obtained from the random forests classifier
    """
    
    #Creating the classifier
    randomForest_classifier = RandomForestClassifier(n_estimators=numEstimators,random_state=7878).fit(X_train,t_train)
    #Calculating error
    randomForestError = 1-randomForest_classifier.score(X_valid,t_valid)
    return randomForestError


# In[7]:


def computeNRandomForestClassifiers(startNumEstimators,maxNumEstimators,incrementNumEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the random forest error for each random forest classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    Inputs:
        startNumEstimators - the start number of estimators
        maxNumEstimators - the maximum number of estimators
        incrementNumEstimators - the increment for the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        numEstimatorsList - list of the number of estimators used
        randomForestErrorList - list of the random forest errors
    """
    
    #Initializing lists
    numEstimatorsList = np.zeros([1,int((maxNumEstimators-startNumEstimators)/incrementNumEstimators)+1])
    randomForestErrorList = np.zeros([1,numEstimatorsList.shape[1]])
    i = 0
    
    #Computes the random forest error for each random forest classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    for numEstimators in range(startNumEstimators,maxNumEstimators+incrementNumEstimators,incrementNumEstimators):
        numEstimatorsList[0,i] = numEstimators
        randomForestErrorList[0,i] = computeRandomForestClassifier(numEstimators,X_train,t_train,X_valid,t_valid)
        i += 1
    return numEstimatorsList,randomForestErrorList


# In[8]:


def computeAdaboostDecisionStumpsClassifier(numEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the error using Adaboost with decision stumps
    Inputs:
        numEstimators - the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        AdaboostDecisionStumpsError - the error obtained from the Adaboost with decision stumps classifier
    """
    
    #Creating the classifier
    AdaboostDecisionStumps_classifier = AdaBoostClassifier(n_estimators=numEstimators,random_state=7878).fit(X_train,t_train)
    #Calculating error
    AdaboostDecisionStumpsError = 1-AdaboostDecisionStumps_classifier.score(X_valid,t_valid)
    return AdaboostDecisionStumpsError


# In[9]:


def computeNAdaboostDecisionStumpsClassifiers(startNumEstimators,maxNumEstimators,incrementNumEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the Adaboost with decision stumps error for each Adaboost with decision stumps classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    Inputs:
        startNumEstimators - the start number of estimators
        maxNumEstimators - the maximum number of estimators
        incrementNumEstimators - the increment for the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        numEstimatorsList - list of the number of estimators used
        adaboostDecisionStumpsErrorList - list of the Adaboost with decision stumps errors
    """
    
    #Initializing lists
    numEstimatorsList = np.zeros([1,int((maxNumEstimators-startNumEstimators)/incrementNumEstimators)+1])
    adaboostDecisionStumpsErrorList = np.zeros([1,numEstimatorsList.shape[1]])
    i = 0
    
    #Computes the Adaboost with decision stumps error for each Adaboost with decision stumps classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    for numEstimators in range(startNumEstimators,maxNumEstimators+incrementNumEstimators,incrementNumEstimators):
        numEstimatorsList[0,i] = numEstimators
        adaboostDecisionStumpsErrorList[0,i] = computeAdaboostDecisionStumpsClassifier(numEstimators,X_train,t_train,X_valid,t_valid)
        i += 1
    return numEstimatorsList,adaboostDecisionStumpsErrorList


# In[10]:


def computeAdaboostClassifier(baseClassifier,numEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the error using Adaboost with the baseClassifier as the base classifier
    Inputs:
        baseClassifier - the base classifier
        numEstimators - the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        AdaboostError - the error obtained from the Adaboost with the baseClassifier as the base classifier
    """
    
    #Creating the classifier
    Adaboost_classifier = AdaBoostClassifier(base_estimator=baseClassifier,n_estimators=numEstimators,random_state=7878).fit(X_train,t_train)
    #Calculating error
    AdaboostError = 1-Adaboost_classifier.score(X_valid,t_valid)
    return AdaboostError


# In[11]:


def computeNAdaboostClassifiers(baseClassifier,startNumEstimators,maxNumEstimators,incrementNumEstimators,X_train,t_train,X_valid,t_valid):
    """Computes the Adaboost with baseClassifier error for each Adaboost with baseClassifier classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    Inputs:
        baseClassifier - the base classifier
        startNumEstimators - the start number of estimators
        maxNumEstimators - the maximum number of estimators
        incrementNumEstimators - the increment for the number of estimators
        X_train - the training features
        t_train - the training targets
        X_valid - the validation features
        t_valid - the validation targets
    
    Outputs:
        numEstimatorsList - list of the number of estimators used
        adaboostErrorList - list of the Adaboost with baseClassifier errors
    """
    
    #Initializing lists
    numEstimatorsList = np.zeros([1,int((maxNumEstimators-startNumEstimators)/incrementNumEstimators)+1])
    adaboostErrorList = np.zeros([1,numEstimatorsList.shape[1]])
    i = 0
    
    #Computes the Adaboost with baseClassifier error for each Adaboost with baseClassifier classifier where the number of predictors increments by incrementNumEstimators from startNumEstimators to maxNumEstimators
    for numEstimators in range(startNumEstimators,maxNumEstimators+incrementNumEstimators,incrementNumEstimators):
        numEstimatorsList[0,i] = numEstimators
        adaboostErrorList[0,i] = computeAdaboostClassifier(baseClassifier,numEstimators,X_train,t_train,X_valid,t_valid)
        i += 1
    return numEstimatorsList,adaboostErrorList


# In[12]:


def plotAllEnsembleMethods(maxLeafNodesList,bestMaxLeafNodes,bestAverageCrossValidError,AverageCrossValidErrorList,numEstimatorsList,baggingErrorList,randomForestErrorList,adaboostDecisionStumpsErrorList,adaboostErrorList_10MaxLeafTree,adaboostErrorList_NoMaxTree):
    """Plots all the emsemble methods
    Inputs:
        maxLeafNodesList - the list of the number of maximum leaf nodes
        bestMaxLeafNodes - best maximum leaf nodes
        bestAverageCrossValidError - the best average cross-validation error for the decision tree
        AverageCrossValidErrorList - list of average cross-validation errors corresponding to each decision tree
        numEstimatorsList - list of the number of estimators used
        baggingErrorList - list of the bagging errors
        randomForestErrorList - list of the random forest errors
        adaboostDecisionStumpsErrorList - list of the Adaboost with decision stumps errors
        adaboostErrorList_10MaxLeafTree - list of the Adaboost with maximum 10 leaves decision tree errors
        adaboostErrorList_NoMaxTree - list of the Adaboost with no maximum depth decision tree errors
    """    
    
    #Copying Best Average Cross Validation Error to match dimensions
    bestAverageCrossValidErrorList = [bestAverageCrossValidError for i in range(numEstimatorsList.shape[1])]
    
    #Plotting Classifier Test Errors vs Number of Predictors
    plt.scatter(numEstimatorsList, bestAverageCrossValidErrorList, color = 'green', label = 'Maximum '+str(bestMaxLeafNodes)+' Leaves Decision Tree')
    plt.scatter(numEstimatorsList, baggingErrorList, color = 'red', label = 'Bagging')
    plt.scatter(numEstimatorsList, randomForestErrorList, color = 'yellow', label = 'Random Forest')
    plt.scatter(numEstimatorsList, adaboostDecisionStumpsErrorList, color = 'blue', label = 'Adaboost With Decision Stumps')
    plt.scatter(numEstimatorsList, adaboostErrorList_10MaxLeafTree, color = 'magenta', label = 'Adaboost With Maximum 10 Leaves Decision Tree')
    plt.scatter(numEstimatorsList, adaboostErrorList_NoMaxTree, color = 'black', label = 'Adaboost With No Maximum Depth Decision Tree')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Classifier Test Errors vs Number of Predictors")
    plt.xlabel("Number of Predictors")
    plt.ylabel("Test Error")
    plt.show()
    
    #Plotting Decision Trees Average 5-Fold Cross-Validation Error vs Maximum Number Of Leaf Nodes
    plt.scatter(maxLeafNodesList, AverageCrossValidErrorList, color = 'green', label = '')
    plt.title("Decision Trees Average 5-Fold Cross-Validation Error vs Maximum Number Of Leaf Nodes")
    plt.xlabel("Maximum Number Of Leaf Nodes")
    plt.ylabel("Average 5-Fold Cross-Validation Error")
    plt.show()


# In[13]:


def printAllOutputs(maxLeafNodesList,bestMaxLeafNodes,bestAverageCrossValidError,AverageCrossValidErrorList,numEstimatorsList,baggingErrorList,randomForestErrorList,adaboostDecisionStumpsErrorList,adaboostErrorList_10MaxLeafTree,adaboostErrorList_NoMaxTree):
    """Prints all the outputs
    Inputs:
        maxLeafNodesList - the list of the number of maximum leaf nodes
        bestMaxLeafNodes - best maximum leaf nodes
        bestAverageCrossValidError - the best average cross-validation error for the decision tree
        AverageCrossValidErrorList - list of average cross-validation errors corresponding to each decision tree
        numEstimatorsList - list of the number of estimators used
        baggingErrorList - list of the bagging errors
        randomForestErrorList - list of the random forest errors
        adaboostDecisionStumpsErrorList - list of the Adaboost with decision stumps errors
        adaboostErrorList_10MaxLeafTree - list of the Adaboost with maximum 10 leaves decision tree errors
        adaboostErrorList_NoMaxTree - list of the Adaboost with no maximum depth decision tree errors
    """ 

    print("Decision Tree Best Number Of Max Leaf Nodes: ",bestMaxLeafNodes," Average Cross-Valid Error = ",bestAverageCrossValidError)
    print()
    print("Average Bagging Error: ", np.mean(baggingErrorList))
    print("Average Random Forest Error: ", np.mean(randomForestErrorList))
    print("Average Adaboost Decision Stumps Error: ", np.mean(adaboostDecisionStumpsErrorList))
    print("Average Adaboost With Maximum 10 Leaves Decision Tree Error: ", np.mean(adaboostErrorList_10MaxLeafTree))
    print("Average Adaboost With No Maximum Depth Decision Tree Error: ", np.mean(adaboostErrorList_NoMaxTree))
    print()
    print("Minimum Bagging Error: ", np.amin(baggingErrorList))
    print("Minimum Random Forest Error: ", np.amin(randomForestErrorList))
    print("Minimum Adaboost Decision Stumps Error: ", np.amin(adaboostDecisionStumpsErrorList))
    print("Minimum Adaboost With Maximum 10 Leaves Decision Tree Error: ", np.amin(adaboostErrorList_10MaxLeafTree))
    print("Minimum Adaboost With No Maximum Depth Decision Tree Error: ", np.amin(adaboostErrorList_NoMaxTree))
    print()
    print("Maximum Bagging Error: ", np.amax(baggingErrorList))
    print("Maximum Random Forest Error: ", np.amax(randomForestErrorList))
    print("Maximum Adaboost Decision Stumps Error: ", np.amax(adaboostDecisionStumpsErrorList))
    print("Maximum Adaboost With Maximum 10 Leaves Decision Tree Error: ", np.amax(adaboostErrorList_10MaxLeafTree))
    print("Maximum Adaboost With No Maximum Depth Decision Tree Error: ", np.amax(adaboostErrorList_NoMaxTree))
    print()
    print("Bagging Variance Error: ", np.var(baggingErrorList))
    print("Random Forest Variance Error: ", np.var(randomForestErrorList))
    print("Adaboost Decision Stumps Variance Error: ", np.var(adaboostDecisionStumpsErrorList))
    print("Adaboost With Maximum 10 Leave Variances Decision Tree Variance Error: ", np.var(adaboostErrorList_10MaxLeafTree))
    print("Adaboost With No Maximum Depth Decision Tree Variance Error: ", np.var(adaboostErrorList_NoMaxTree))
    print()
    for i in range(len(maxLeafNodesList)):
        print("Decision Tree Number Of Max Leaf Nodes: ",maxLeafNodesList[i]," Average Cross-Valid Error = ",AverageCrossValidErrorList[i])
    print()
    for i in range(numEstimatorsList.shape[1]):
        print("Number of Estimators: ",numEstimatorsList[0,i])
        print("Bagging Error: ", baggingErrorList[0,i])
        print("Random Forest Error: ", randomForestErrorList[0,i])
        print("Adaboost Decision Stumps Error: ", adaboostDecisionStumpsErrorList[0,i])
        print("Adaboost With Maximum 10 Leaves Decision Tree Error: ", adaboostErrorList_10MaxLeafTree[0,i])
        print("Adaboost With No Maximum Depth Decision Tree Error: ", adaboostErrorList_NoMaxTree[0,i])
        print()


# In[14]:


#load data set
dataset = pd.read_csv('spambase.data', header=None)
X_data = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values

#split the data into training and test data 
X_train, X_valid, t_train, t_valid = train_test_split(X_data, t, test_size = 1/3, random_state = 7878)

highestNumMaxLeafNodes = 400
startNumPredictors = 50
stopNumPredictors = 2500
incrementNumPredictors = 50


# In[15]:


#Computing Decision Trees Errors
maxLeafNodesList,bestMaxLeafNodes,bestAverageCrossValidError,AverageCrossValidErrorList = computeDecisionTree2ToNMaxLeafNodes(highestNumMaxLeafNodes,X_train,t_train)


# In[16]:


#Computing Bagging Classifiers Errors
numEstimatorsList,baggingErrorList = computeNBaggingClassifiers(DecisionTreeClassifier(random_state=7878),startNumPredictors,stopNumPredictors,incrementNumPredictors,X_train,t_train,X_valid,t_valid)


# In[17]:


#Computing Random Forest Errors
numEstimatorsList,randomForestErrorList = computeNRandomForestClassifiers(startNumPredictors,stopNumPredictors,incrementNumPredictors,X_train,t_train,X_valid,t_valid)


# In[18]:


#Computing Adaboost Decision Stump Errors
numEstimatorsList,adaboostDecisionStumpsErrorList = computeNAdaboostDecisionStumpsClassifiers(startNumPredictors,stopNumPredictors,incrementNumPredictors,X_train,t_train,X_valid,t_valid)


# In[19]:


#Computing Adaboost Maximum 10 Leaves Decision Trees
numEstimatorsList,adaboostErrorList_10MaxLeafTree = computeNAdaboostClassifiers(DecisionTreeClassifier(max_leaf_nodes=10,random_state=7878),startNumPredictors,stopNumPredictors,incrementNumPredictors,X_train,t_train,X_valid,t_valid)


# In[20]:


#Computing Adaboost Unlimited Depth Decision Trees
numEstimatorsList,adaboostErrorList_NoMaxTree = computeNAdaboostClassifiers(DecisionTreeClassifier(random_state=7878),startNumPredictors,stopNumPredictors,incrementNumPredictors,X_train,t_train,X_valid,t_valid)


# In[21]:


#Plotting Performance Graphs
plotAllEnsembleMethods(maxLeafNodesList,bestMaxLeafNodes,bestAverageCrossValidError,AverageCrossValidErrorList,numEstimatorsList,baggingErrorList,randomForestErrorList,adaboostDecisionStumpsErrorList,adaboostErrorList_10MaxLeafTree,adaboostErrorList_NoMaxTree)


# In[22]:


#Printing Outputs
printAllOutputs(maxLeafNodesList,bestMaxLeafNodes,bestAverageCrossValidError,AverageCrossValidErrorList,numEstimatorsList,baggingErrorList,randomForestErrorList,adaboostDecisionStumpsErrorList,adaboostErrorList_10MaxLeafTree,adaboostErrorList_NoMaxTree)


# In[ ]:





# In[ ]:




