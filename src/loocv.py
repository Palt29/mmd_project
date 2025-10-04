import numpy as np


def loocv_split(bin_utility_matrix: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """
    Performs Leave-One-Out (LOOCV) splitting of the dataset for Top-N evaluation.
    For each user, selects one rated item and moves it to the test set,
    setting its value to 0 in the training matrix.

    Args:
        bin_utility_matrix (np.ndarray): The complete binary matrix (num_users x num_items).

    Returns:
        out (tuple[np.ndarray, dict[int, int]]):
            - Training matrix (train_matrix): Matrix with the test item removed (set to 0).
            - Test items (test_items): Mapping {user_index: test_item_index}.
    """
    num_users, _ = bin_utility_matrix.shape
    test_items = {}
    train_matrix = bin_utility_matrix.copy()

    # Randomly select one rated item from each user and place it in the test set
    for user_idx in range(num_users):
        # Items with interaction (value 1)
        rated_items = np.where(bin_utility_matrix[user_idx, :] == 1)[0]

        if len(rated_items) > 0:
            # Randomly choose one rated item to leave out (test set)
            test_item = np.random.choice(rated_items, 1)[0]
            test_items[user_idx] = test_item

            # Remove the test item from the training matrix
            train_matrix[user_idx, test_item] = 0

    # The train_matrix (remaining data) comprises the training set
    return train_matrix, test_items
