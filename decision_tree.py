from build_tree import build_tree
from predict import predict
from  print_tree import print_tree


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


# Unit test
if __name__ == "__main__":
    train = [[2.771244718, 1.784783929, 0],
             [1.728571309, 1.169761413, 0],
             [3.678319846, 2.81281357, 0],
             [3.961043357, 2.61995032, 0],
             [2.999208922, 2.209014212, 1],
             [7.497545867, 3.162953546, 1],
             [9.00220326, 3.339047188, 1],
             [7.444542326, 0.476683375, 2],
             [10.12493903, 3.234550982, 2],
             [6.642287351, 3.319983761, 2]]
    test = [[1, 1.784783929],
            [7, 3.234550982]]
    tree = build_tree(train, 4, 1)
    print_tree(tree)
    print(decision_tree(train, test, max_depth=4, min_size=1))
