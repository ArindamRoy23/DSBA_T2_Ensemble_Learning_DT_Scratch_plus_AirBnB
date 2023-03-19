# Split a dataset based on an attribute and an attribute value
# The function takes an index of an attribute, a value of the attribute and the dataset itself
def test_split(index, value, dataset):
    # Create two empty lists to hold the split data
    left, right = list(), list()
    # For each row (data point) in the dataset
    for row in dataset:
        # If the value of the specified attribute is less than the split value
        if row[index] < value:
            # Append the row to the left list
            left.append(row)
        else:
            # Otherwise, append the row to the right list
            right.append(row)
    # Return the two lists, which now contain the split data
    return left, right


# Unit test
# if __name__ == "__main__":
#     print(gini_index([[[1, 1], [1, 1]], [[1, 1], [1, 0]]], [0, 1]))
