from split import split
from get_split import get_split
from print_tree import print_tree

def build_tree(data, max_depth, min_size):
    # Obtain the root split using the get_split() function
    root = get_split(data)
    # Split the tree recursively until the maximum depth or minimum size is reached
    split(root, max_depth, min_size, 1)
    # Return the root node of the built decision tree
    return root


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
    tree = build_tree(dataset, 4, 1)
    print_tree(tree)
