def print_tree(node, depth=0):
    # Check if the current node is a dictionary, indicating a decision node
    if isinstance(node, dict):
        # Print the decision rule at the current node
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        # Recursively print the left and right subtrees
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    # If the current node is not a dictionary, it is a terminal node
    else:
        # Print the terminal value at the current node
        print('%s[%s]' % ((depth*' ', node)))
