# Classifiaction-Tree-from-Scratch

Classification Tree from Scratch: Build and train a Decision Tree for classifiaction tasks.

## Classes:
Classification Tree: binary classification tree object
Node: node object

## Attributes:
1. max_depth: max depth of the tree (stop criterion)
2. min_samples_leaf: minimum number of samples per lieaf  (stop criterion)
3. min_samples_split: minimum number of samples to split a node (not leaves, also a stop criterion)
4. n_classes_: number of classes in training set 
5. n_features_: number of features in training set and test set
6. n_samples_: number of samples
7. criterion: criterion to split a node. Chosen from "gini", "cross_entropy" and "isclassification_error"
8. tree_: tree class

## Methods:
1. __init__(max_depth, min_samples_leaf, min_samples_split, criterion = "gini"): constructor, build a Classifiactiopn Tree class
2. fit(X,y): fit the training set data
3. predict(X): predict classes for test data using trained parameters
4. predict_probability(X): predict probabilities for test data
5. acc_score: print the accuracy score of the tree model

## Example
# import libraries
from sklearn import datasets
iris = datasets.load_iris()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# import data
X = iris.data
y = iris.target

# split train and test samples
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 33)

# create a classification tree
cdt = ClassificationTree(max_depth = 3, min_samples_leaf = 3, criterion = "misclassification_error")

# fit the tree
cdt.fit(X_train, y_train)

# make prediction on test set
y_pred = cdt.predict(X_test)

# calculate accuracy score
cdt.acc_score(y_pred, y_test)

# print classification report
print(classification_report(y_test, y_pred))

# predict probabilities of each class for each sample
prob_pred = cdt.predict_probability(X_test)
