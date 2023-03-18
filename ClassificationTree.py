# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 22:27:55 2023

@author: lenovo
"""

import numpy as np 

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, label_freq = None):
        # Initialize a node with feature index, threshold, left and right children, and value (for leaf nodes)
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.label_freq = label_freq
        
class ClassificationTree:
    
    """
        Classification Tree: Build and train a Decision Tree for classifiaction tasks.
        
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
    """
    
    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2, criterion = "gini"):
        # Initialize the decision tree model with the specified parameters
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        
    def fit(self, X, y):
        """
            Build the decision tree recursively using the training data
            Syntax: ClassificationTree.fit(X,y)
            Inputs: X: n_samples_ by n_features_ numpy array; y: n_samples_ numpy array
            Output: build a ClassificationTree object
        """
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.n_samples_ = len(y)
        self.tree_ = self._build_tree(X, y)
        
        
    def predict(self, X):
        """
            Predict the class labels of new data using the trained decision tree
            Syntax: ClassificationTree.predict(X)
            Input: X: n_samples_ by n_features_ numpy array
            Output: a numpy array with n_samples_ rows, indicating the predicted classes of that sample
        """
        return [self._predict(inputs) for inputs in X]
    
    def predict_probability(self, X):
        """
            Predict the class probabilities of new data using the trained decision tree
            Syntax: ClassificationTree.predict_probability(X)
            Input: X: n_samples_ by n_features_ numpy array
            Output: n_samples_ by n_classes_ numpy array, each column indicates the probability of that class
        """
        return [self._predict_probability(inputs) for inputs in X]
    
    def _build_tree(self, X, y, depth=0):
        # Recursively build the decision tree
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if depth == self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            # If the maximum depth is reached, or if the number of samples is less than the minimum split,
            # or if there is only one label in the current node, return a leaf node with the most common class label
            leaf_value = self._most_common_label(y)
            label_freq = self._label_frequency(y)
            return Node(value=leaf_value, label_freq = label_freq)
        
        # Find the best feature to split the data
        best_feature_idx, best_threshold = self._best_split(X, y)
        
        if best_feature_idx is None:
            # If no feature can be found to split the data, return a leaf node with the most common class label
            leaf_value = self._most_common_label(y)
            label_freq = self._label_frequency(y)
            return Node(value=leaf_value, label_freq = label_freq)
        
        # Split the data into two subsets using the best feature and threshold
        left_indices = X[:, best_feature_idx] < best_threshold
        right_indices = X[:, best_feature_idx] >= best_threshold
        left_subset, left_labels = X[left_indices], y[left_indices]
        right_subset, right_labels = X[right_indices], y[right_indices]
        
        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(left_subset, left_labels, depth+1)
        right_subtree = self._build_tree(right_subset, right_labels, depth+1)
        
        # Return the node with the best split
        return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left_subtree, right=right_subtree)
    
    def _best_split(self, X, y):
        # Find the best feature and threshold to split the data
        best_gain = -1
        best_feature_idx, best_threshold = None, None
        for feature_idx in range(self.n_features_):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        if best_gain == -1:
            # If no feature can be found to split the data, return None for feature index and threshold
            return None, None
        
        return best_feature_idx, best_threshold
    
    def _information_gain(self, X, y, feature_idx, threshold):
        # Calculate the information gain of splitting the data by the specified feature and threshold
        parent_entropy = self._entropy(y)
        
        left_indices = X[:, feature_idx] < threshold
        right_indices = X[:, feature_idx] >= threshold
        left_labels, right_labels = y[left_indices], y[right_indices]
        
        if len(left_labels) == 0 or len(right_labels) == 0:
            return 0
        
        left_entropy = self._entropy(left_labels)
        right_entropy = self._entropy(right_labels)
        child_entropy = (len(left_labels) / len(y)) * left_entropy + (len(right_labels) / len(y)) * right_entropy
        
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _entropy(self, y):
        # Calculate the entropy of a set of labels
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)  
        if self.criterion == 'gini':
            return 1 - np.sum(p ** 2)
        elif self.criterion == 'crossentropy':
            return -np.sum(p * np.log2(p))
        # elif self.criterion == 'misclassifcation_error'
        else:
            raise ValueError('Invalid criterion, Select from "gini", "crossentropy"')
    
    def _most_common_label(self, y):
        # Find the most common label in a set of labels
        labels, counts = np.unique(y, return_counts=True)
        most_common_label = labels[np.argmax(counts)]
        return most_common_label
    
    def _label_frequency(self, y):
        # Store labels of samples in the nodes
        label_freq = y  
        return label_freq
    
    def _predict(self, inputs):
        # Traverse the decision tree to predict the class label of new data
        node = self.tree_
        while node.left:
            if inputs[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def _predict_probability(self, inputs):
        # Traverse the decision tree to predict the class probabilities of new data
        node = self.tree_
        while node.left:
            if inputs[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
                
        # Calculate the class probabilities based on the number of samples in each class in the leaf node
        proba = np.zeros(self.n_classes_)
        total_samples = len(node.label_freq)
        for class_idx in range(self.n_classes_):
            class_count = sum(node.label_freq==class_idx)
            proba[class_idx] = class_count / total_samples
        
        return proba
    
    def acc_score(self, y_true, y_pred):
        """
            Calculate the accuracy of predicted values
            Syntax: ClassificatonTree.acc_score(X,y)
            Inputs: X: n_samples_ by n_features_ numpy array; y: n_samples_ numpy array
            Output: float, accuracy score
        """
        
        if hasattr(self, 'tree_'):  # if the tree is fit
            # Create a confusion matrix
            n_classes = self.n_classes_
            conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
            for i in range(len(y_true)):
                conf_matrix[y_true[i]][y_pred[i]] += 1

            # Calculate the accuracy
            total = sum(sum(conf_matrix))
            correct = sum([conf_matrix[i][i] for i in range(n_classes)])
            acc = correct / total
            print('The accuracy socre is: % .3f' % acc)
            return round(acc,3)
        else:
            raise ValueError('Please fit the classification tree first')