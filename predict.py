# Make a prediction with a decision tree
# This function takes in two arguments: a decision tree node and a row of data
def predict(node, row):
    # If the value of the feature at the current node is less than the threshold
    # for that feature, follow the left subtree
    if row[node['index']] < node['value']:
        # If the left subtree is another decision node, recursively call the
        # `predict()` function with the left node and the same row of data
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        # Otherwise, return the class label assigned to the left leaf node
        else:
            return node['left']
    # If the value of the feature at the current node is greater than or equal to
    # the threshold for that feature, follow the right subtree
    else:
        # If the right subtree is another decision node, recursively call the
        # `predict()` function with the right node and the same row of data
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        # Otherwise, return the class label assigned to the right leaf node
        else:
            return node['right']


# Unit test
if __name__ == "__main__":

    dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.81281357, 0],
               [3.961043357, 2.61995032, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.00220326, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]
    #  predict with a stump
    stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
    for row in dataset:
        prediction = predict(stump, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))
