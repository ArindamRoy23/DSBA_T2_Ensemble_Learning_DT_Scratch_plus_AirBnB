# Classifiaction-Tree-from-Scratch

Classification Tree from Scratch: Build and train a Decision Tree for classifiaction tasks.

Classes:
Classification Tree: binary classification tree object
Node: node object

Attributes:
1. max_depth: max depth of the tree (stop criterion)
2. min_samples_leaf: minimum number of samples per lieaf  (stop criterion)
3. min_samples_split: minimum number of samples to split a node (not leaves, also a stop criterion)
4. n_classes_: number of classes in training set 
5. n_features_: number of features in training set and test set
6. n_samples_: number of samples
7. criterion: criterion to split a node. Chosen from "gini", "cross_entropy" and "isclassification_error"
8. tree_: tree class

Methods:
1. __init__: constructor, build a Classifiactiopn Tree class
2. fit(X,y): fit the training set data
3. predict(X): predict classes for test data using trained parameters
4. predict_probability(X): predict probabilities for test data
5. acc_score: print the accuracy score of the tree model
