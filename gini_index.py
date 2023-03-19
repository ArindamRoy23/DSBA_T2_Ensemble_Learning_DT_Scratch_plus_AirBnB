# This function calculates the Gini index for a split dataset.
def gini_index(data_groups, class_labels):
    # Count the total number of instances at the split point.
    total_instances = float(sum([len(group) for group in data_groups]))
    # Calculate the weighted Gini index for each group and sum them together.
    gini = 0.0
    for group in data_groups:
        # Calculate the size of the group.
        group_size = float(len(group))
        # If the group size is zero, skip it to avoid a divide-by-zero error.
        if group_size == 0:
            continue
        # Calculate the score for the group based on the proportion of instances belonging to each class.
        group_score = 0.0
        for class_val in class_labels:
            class_proportion = [row[-1] for row in group].count(class_val) / group_size
            group_score += class_proportion * class_proportion
        # Weight the group score by its relative size in the overall dataset.
        gini += (1.0 - group_score) * (group_size / total_instances)
    # Return the total Gini index for the split.
    return gini


# Unit test
if __name__ == "__main__":
    print(gini_index([[[1, 1], [1, 1]], [[1, 1], [1, 0]]], [0, 1]))
