# Create a terminal node value
def to_terminal(data):
    # Extract the outcome values from the input data
    outcomes = [row[-1] for row in data]
    # Determine the most common outcome and return it as the terminal value
    return max(set(outcomes), key=outcomes.count)


# Unit test
# if __name__ == "__main__":
#     print(gini_index([[[1, 1], [1, 1]], [[1, 1], [1, 0]]], [0, 1]))