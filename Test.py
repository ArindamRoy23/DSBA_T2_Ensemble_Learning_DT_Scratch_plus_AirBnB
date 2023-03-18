import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from RegressorTree import DecisionTreeRegressor
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ClassificationTree import ClassificationTree

try:
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print("Test-1 passed: Regressor working ok")

except:
    print('Test-1 failed: Regressor not working')

try:
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    cdt = ClassificationTree(max_depth=3, min_samples_leaf=3, criterion="misclassification_error")
    cdt.fit(X_train, y_train)
    y_pred = cdt.predict(X_test)
    cdt.acc_score(y_pred, y_test)
    prob_pred = cdt.predict_probability(X_test)
    print('Test-2 passed: Classifier working ok')

except:
    print('Test-2 failed: Classifier not working')
