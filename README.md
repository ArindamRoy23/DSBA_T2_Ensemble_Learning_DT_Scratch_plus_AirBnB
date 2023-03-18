# Classifiaction-Tree-from-Scratch

Classification Tree from Scratch: Build and train a Decision Tree for classifiaction tasks.

Classes:
Classification Tree: binary classification tree object
Node: node object

Attributes:
    max_depth: max depth of the tree (stop criterion)
    min_samples_leaf: minimum number of samples per lieaf  (stop criterion)
    min_samples_split: minimum number of samples to split a node (not leaves, also a stop criterion)
    n_classes_: number of classes in training set 
    n_features_: number of features in training set and test set
    n_samples_: number of samples
    criterion: criterion to split a node. Chosen from "gini", "cross_entropy" and "isclassification_error"
    tree_: tree class

Methods:
    __init__: constructor, build a Classifiactiopn Tree class
    fit(X,y): fit the training set data
    predict(X): predict classes for test data using trained parameters
    predict_probability(X): predict probabilities for test data
    acc_score: print the accuracy score of the tree model
