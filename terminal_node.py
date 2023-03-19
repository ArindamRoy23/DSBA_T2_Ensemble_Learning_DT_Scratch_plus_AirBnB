def terminal_node(group):
    # Extract the output values from the group of data instances.
    results = [row[-1] for row in group]
    # Determine the most common output value in the results list using the set() and count() functions.
    # This will be the output value for the leaf node.
    return max(set(results), key=results.count)

# Unit test
# if __name__ == "__main__":
#     print(gini_index([[[1, 1], [1, 1]], [[1, 1], [1, 0]]], [0, 1]))