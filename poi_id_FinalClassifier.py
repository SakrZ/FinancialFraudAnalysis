

import math
import sys
import pickle
sys.path.append("C:/Users/SakrZeyad/Desktop/Nanodegree/IntroToMachineLearning/Udacity_ML_library/tools")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###############################################
########Removing outliers
###############################################
#Only outlier found was the Total field which has no true meaning
data_dict.pop("TOTAL")

#Get a list of Employees
LofEmployees=data_dict.keys()
#Get list of Features
LofFeatures=data_dict["LAY KENNETH L"].keys()


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
#print NanCount

#Make a list of features that have a low nan count
#email address is the only string features so it is removed
NonFdFeatures=['poi']
for cc in NanCount:
    if cc not in 'email_address':
        if NanCount[cc]<0.40 and cc!='poi':
            NonFdFeatures.append(cc)
#print(NonFdFeatures)
#features_list =NonFdFeatures





###############################################
########Adding New Feature
###############################################


#bonus to salary ratio feature
for emp in LofEmployees:
    #print type(data_dict[emp]['bonus'])

    if type(data_dict[emp]['bonus'])==type('horus') or type(data_dict[emp]['salary'])==type('horus'):
        data_dict[emp]['bs']='NaN'
    else:
        data_dict[emp]['bs']=float(data_dict[emp]['bonus'])/float(data_dict[emp]['salary'])
    
    #print data_dict[emp]['bs']


LofFeatures=data_dict["LAY KENNETH L"].keys()



###############################################
########Features Selected
###############################################

###########Based on Principal Component 1 Vector
features_list=['poi', 'total_payments', 'exercised_stock_options', 'total_stock_value', 'bs']

#####Shape features into a machine digestable format
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)





###############################################
########Split Data
###############################################

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=0) #random state set to maintain reproducible output



###############################################
########Import Libraries
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
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn import tree


###############################################
########Scale
###############################################

def scaleInputs(X, X_test):
    #Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    return X, X_test

###############################################
########SVM Grid Search
###############################################

def runSVMgrid(X, y, X_test):
    print("Fitting the classifier to the training set")
    
    t0 = time()
    #"kernel":["linear", "poly", "rbf"]
    param_grid = {'C': [0.1,1,2,3, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.001, 0.01, 0.1,1]}
    clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid, scoring='f1', cv=3)
    
    clf = clf.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    return clf.best_estimator_

###############################################
########Evaluation Function
###############################################

def evaluate(clf):
    t0 = time()
    y_pred = clf.predict(X_test)
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
    print(clf.score(X_test,y_test))
    print('Recall')
    print((truePos)/(truePos+falseNeg))
    print('Precision')
    print((truePos)/(truePos+falsePos))



###############################################
########Chosen Classifier
###############################################

"""
#This works (values are the rbf tuning from the PCA code) Precision is a bit lower
clf = make_pipeline(StandardScaler(), SVC(C=50000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))

#Better precision than one above but doesn't meet precision benchmark
  clf = make_pipeline(StandardScaler(), SVC(C=10000000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))
"""



"""
#The one that meets the benchmark
clf = make_pipeline(StandardScaler(), SVC(C=100000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))

#model with the best recall using gridsearch but still lower on precision benchmark
clf = make_pipeline(StandardScaler(), SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))

clf.fit(X_train, y_train)
"""

#X_train, X_test= scaleInputs(X_train, X_test)
#clf=runSVMgrid(X_train, y_train, X_test)

#The model that meets benchmark
clf = make_pipeline(StandardScaler(), SVC(C=100000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))

clf.fit(X_train, y_train)

evaluate(clf)

dump_classifier_and_data(clf, my_dataset, features_list)
    

