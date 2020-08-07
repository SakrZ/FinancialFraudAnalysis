

#!/usr/bin/python

import math
import sys
import pickle
sys.path.append("C:/Users/SakrZeyad/Desktop/Nanodegree/IntroToMachineLearning/Udacity_ML_library/tools")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary', 'director_fees'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

LofEmployees=data_dict.keys()
LofFeatures=data_dict["LAY KENNETH L"].keys()
#print ListofEmployees

###############################################
######Investigating structure of dataset#######
###############################################
LayDict=data_dict["LAY KENNETH L"]
print LayDict



###############################################
########Investigating NaNs
###############################################

#############
def isNaN(xx):
    """
    Checks for string NaNs
    """
    if xx=='NaN':
        return True
    else:
        return False
#############

NanCount=dict() #dictionary that shows the percentage of nans for each feature
NanCount=NanCount.fromkeys(LofFeatures)
NofEmp=len(LofEmployees)


for ff in LofFeatures:
    counter=0.0
    for emp in LofEmployees:
        xx=isNaN(data_dict[emp][ff])
        #print xx
        if xx is True:
            #data_dict[emp][ff]=nan
            counter+=1
            pass
    NanCount[ff]=counter/NofEmp

print NanCount


#Make a list of features that have a low nan count
#email address is the only string features so it is removed
NonFdFeatures=['poi']
for cc in NanCount:
    if cc not in 'email_address':
        if NanCount[cc]<0.40 and cc!='poi':
            NonFdFeatures.append(cc)
print(NonFdFeatures)
features_list =NonFdFeatures


###############################################
########Removing outliers
###############################################
#top and bottom two

#secMax was ignored as it resulted in higher flase negatives
def secMax(fList, maxInd):
    """
    Second Max
    """
    secmax=0.0
    for idx, val in enumerate(fList):
        if val>secmax and idx!=maxInd:
            secmaxval=val
            secmaxInd=idx

    
    return secmaxInd



for ff in LofFeatures:
    fList=[]
    for emp in LofEmployees:
        fList.append(data_dict[emp][ff])
    
    maxInd=fList.index(max(fList))
    minInd=fList.index(min(fList))
    data_dict[LofEmployees[maxInd]][ff]='NaN'
    data_dict[LofEmployees[minInd]][ff]='NaN'

    secmaxInd=secMax(fList, maxInd)
    #data_dict[LofEmployees[secmaxInd]][ff]='NaN'



###############################################
########Shaping Features and using PCA to reduce no.of features
###############################################

#####Shape features into a machine digestable format
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#print features


#######Split Data
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42) #random state set to maintain reproducible output



###############################################
########Implement PCA
###############################################

from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


#tried scaling pre PCA but performance was lower (based on confusion matrix and f1 score)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)


n_components = 4 #4 was also found to be optimal for performance
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print pca.explained_variance_ratio_
print pca.components_

print(NonFdFeatures)
print sum(pca.explained_variance_ratio_)

t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))



###############################################
########Scaling Data to mitigate effects for non-linear algorithm (SVM)
###############################################

scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)



###############################################
########Classifier Functions
###############################################

from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn import tree


def runSVMgrid():
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
              'kernel':('rbf', 'poly', 'sigmoid')}
    clf = GridSearchCV(
    SVC(class_weight='balanced'), param_grid)
    
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    return clf


def runDTgrid():
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'min_samples_leaf': [1,2,3,4,5],
              'min_samples_split': [2,3,4], 
              'splitter':('best','random')}
    
    clf = GridSearchCV(
    tree.DecisionTreeClassifier(random_state=23), param_grid)

    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    return clf


def evaluate(clf):
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    #print("done in %0.3fs" % (time() - t0))
    mat=confusion_matrix(y_test, y_pred, labels=[0,1])
    falseNeg=float(mat[1][0])
    falsePos=float(mat[0][1])
    truePos=float(mat[1][1])
    print('Confusion Matrix')
    print(mat) #0 is not, poi 1 is poi
    print('F1 Score')
    print(f1_score(y_test, y_pred))
    print('Accuracy')
    print(clf.score(X_test_pca,y_test))
    print('Recall')
    print((truePos)/(truePos+falseNeg))
    print('Precision')
    print((truePos)/(truePos+falsePos))


###############################################
########Running Classifiers & Evaluating
###############################################

"""
#SVM
clf=runSVMgrid()
print('************SVM solution************')
evaluate(clf)

#Decision Trees
clf=runDTgrid()
print('************Decision Trees Solution************')
evaluate(clf)
"""

#RBF Solution priortising F1 Score
clf = make_pipeline(StandardScaler(), SVC(C=50000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))
clf.fit(X_train_pca, y_train)

#Manual Tuning
print('************SVM RBF************')
evaluate(clf)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

