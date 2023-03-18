import numpy as np
import pandas as pd


class Node:
    """
    This class defines a node of the tree.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # index of feature to split on
        self.threshold = threshold  # threshold value to split on
        self.left = left  # left subtree
        self.right = right  # right subtree
        self.value = value  # predicted value (for leaf nodes)


class DecisionTreeRegressor:
    """
    This class has methods to build a decision tree.
    Each function has its description inside it.
    Please note, this implementation does not deal with categorical train data.
    This can be a future scope of improvement
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None  # root node of the decision tree

    def fit(self, X, y):
        """
        This function is used to fit the data to the decision tree.
                Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target labels.

        Returns
        -------
        root : Trained tree saved to class instance
        """

        self.root = self.tree_build(X, y)

    def tree_build(self, X, y, depth=0):
        """
        Recursively builds a decision tree using the input data X and labels y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target labels.

        depth : int, optional (default=0)
            The current depth of the tree.

        Returns
        -------
        node : Node object
            The root node of the decision tree.

        Notes
        -----
        This function builds a decision tree using a recursive approach. It first checks whether
        the stopping criteria have been met, which can be either a maximum depth or a minimum number
        of samples required to split. If neither stopping criteria is met, it finds the best feature
        and threshold to split on using the best_split method, and splits the data into left
        and right subsets. It then recursively builds the left and right subtrees by calling itself
        with the left and right subsets. Finally, it returns a Node object that stores the feature,
        threshold, and left and right child nodes.
        """

        n_samples, n_features = X.shape

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return Node(value=np.mean(y, axis=0))

        # Find best split
        best_feature, best_threshold = self.best_split(X, y)

        # Split data
        left_indices = X[:, best_feature] < best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # Recursive call to build left and right subtrees
        left = self.tree_build(X_left, y_left, depth + 1)
        right = self.tree_build(X_right, y_right, depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def best_split(self, X, y):
        """
        Finds the best feature and threshold for splitting the data X and labels y
        based on the mean squared error (MSE) cost function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target labels.

        Returns
        -------
        best_feature : int or None
            The index of the best feature to split on, or None if no good split is found.

        best_threshold : float or None
            The threshold value to use for the best split, or None if no good split is found.

        Notes
        -----
        This function loops over all features and thresholds in X to find the best split
        that minimizes the sum of the MSE of the left and right subsets. If no good split
        is found, best_feature and best_threshold are set to None.
        """
        n_samples, n_features = X.shape
        best_cost = np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                y_left = y[left_indices]
                y_right = y[~left_indices]

                # Calculate cost of split
                if len(y_left) > 0 and len(y_right) > 0:
                    cost = self.meansqerror(y_left) + self.meansqerror(y_right)
                    if cost < best_cost:
                        best_cost = cost
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def meansqerror(self, y):
        """
        Finds the best feature and threshold for splitting the data X and labels y
        based on the mean squared error (MSE) cost function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target labels.

        Returns
        -------
        best_feature : int or None
            The index of the best feature to split on, or None if no good split is found.

        best_threshold : float or None
            The threshold value to use for the best split, or None if no good split is found.

        Notes
        -----
        This function loops over all features and thresholds in X to find the best split
        that minimizes the sum of the MSE of the left and right subsets. If no good split
        is found, best_feature and best_threshold are set to None.
        """
        return np.mean((y - np.mean(y, axis=0)) ** 2)

    def predict(self, X):
        """
        Predicts the target labels for a given input data X using the trained decision tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to predict on.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target labels.

        Notes
        -----
        This function predicts the target labels for each sample in X by recursively traversing
        the decision tree, starting from the root node. At each node, it checks whether the sample
        belongs to the left or right subtree based on the feature and threshold stored in the node.
        It then recursively traverses the left or right subtree until it reaches a leaf node, which
        contains the predicted target label for the sample. Finally, it returns an array of predicted
        target labels for all samples in X.
        """

        return np.array([self.predict_distinct(x, self.root) for x in X])

    def predict_distinct(self, x, node):
        """
        Predicts the target label for a given input sample x using the trained decision tree.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            The input sample to predict on.

        node : Node
            The current node to start the prediction from.

        Returns
        -------
        y_pred : float
            The predicted target label for the input sample x.

        Notes
        -----
        This function predicts the target label for the input sample x by recursively traversing
        the decision tree, starting from the given node. At each node, it checks whether the input
        sample belongs to the left or right subtree based on the feature and threshold stored in
        the node. It then recursively traverses the left or right subtree until it reaches a leaf
        node, which contains the predicted target label for the sample. Finally, it returns the
        predicted target label for the input sample x.
        """

        if node.value is not None:
            return node.value
        elif x[node.feature] < node.threshold:
            return self.predict_distinct(x, node.left)
        else:
            return self.predict_distinct(x, node.right)


if __name__ == '__main__':
    # Unit test
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Generate a sample regression dataset
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create decision tree regressor and fit on training data
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    tree.fit(X_train, y_train)

    # Predict on test data
    y_pred = tree.predict(X_test)

    # Calculate mean squared error (MSE) on test data
    mse = np.mean((y_pred - y_test) ** 2)
    print("Mean squared error on test data:", mse)
