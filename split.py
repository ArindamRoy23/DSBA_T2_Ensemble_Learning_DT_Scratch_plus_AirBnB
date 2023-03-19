from test_split import test_split
from gini_index import gini_index
from terminal_node import terminal_node
from to_terminal import to_terminal
from get_split import get_split


# This function splits a node into left and right child nodes based on the best split point
# for the data at the current node.
# The function takes in the following arguments:
# - node: the current node in the decision tree
# - max_depth: the maximum depth of the decision tree
# - min_size: the minimum number of samples required to split a node
# - depth: the current depth of the decision tree
def split(node, max_depth, min_size, depth):
    # Separate the data for the left and right child nodes and remove the 'groups' attribute
    left_data, right_data = node['groups']
    del node['groups']

    # If either the left or right data is empty, assign both child nodes to the same terminal node
    if not left_data or not right_data:
        node['left'] = node['right'] = to_terminal(left_data + right_data)
        return

    # If the maximum depth has been reached, assign the left and right child nodes to terminal nodes
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left_data), to_terminal(right_data)
        return

    # Split the left child node
    if len(left_data) <= min_size:
        # If the number of samples in the left data is less than or equal to the minimum size,
        # assign the left child node to a terminal node
        node['left'] = to_terminal(left_data)
    else:
        # Otherwise, get the best split point for the left data and create a new child node for it
        node['left'] = get_split(left_data)
        # Recursively split the left child node
        split(node['left'], max_depth, min_size, depth + 1)

    # Split the right child node
    if len(right_data) <= min_size:
        # If the number of samples in the right data is less than or equal to the minimum size,
        # assign the right child node to a terminal node
        node['right'] = to_terminal(right_data)
    else:
        # Otherwise, get the best split point for the right data and create a new child node for it
        node['right'] = get_split(right_data)
        # Recursively split the right child node
        split(node['right'], max_depth, min_size, depth + 1)


# Unit test
# if __name__ == "__main__":
#     print(gini_index([[[1, 1], [1, 1]], [[1, 1], [1, 0]]], [0, 1]))
