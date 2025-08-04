"""Functions for Recommender System."""

import logging

import numpy as np

# Logger setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(filename: str) -> tuple[list[list[float]], int, int]:
    """Loads and processes rating data from a CSV file into indexed format.

    Args:
        filename: Path to the input CSV file. The file should contain
            rows in the format: user_id, item_id, rating (with a header row).

    Returns:
        A tuple containing:
            - A list of [user_idx, item_idx, rating] entries.
            - Total number of unique users.
            - Total number of unique items.
    """
    input_lines = []
    users = {}
    num_users = 0
    items = {}
    num_items = 0
    raw_lines = open(filename, "r").read().splitlines()
    del raw_lines[0]
    for line in raw_lines:
        line_content = line.split(",")
        user_id = int(line_content[0])
        item_id = int(line_content[1])
        rating = float(line_content[2])
        if user_id not in users:
            users[user_id] = num_users
            num_users += 1
        if item_id not in items:
            items[item_id] = num_items
            num_items += 1
        input_lines.append([users[user_id], items[item_id], rating])
    return input_lines, num_users, num_items


def create_utility_matrix(
    input_ratings: list[list[float]], num_users: int, num_items: int
) -> np.ndarray:
    """Builds the user-item utility matrix and logs its sparsity.

    This function is designed to work directly with the output of `load_data()`.
    It takes the remapped user/item indices and constructs a dense matrix where
    each cell [i, j] contains the rating assigned by user i to item j.

    Args:
        input_ratings: List of [user_idx, item_idx, rating] triples,
            as returned by `load_data()`.
        num_users: Total number of users (also from `load_data()`).
        num_items: Total number of items (also from `load_data()`).

    Returns:
        A 2D NumPy array representing the utility matrix.
    """
    ratings = np.zeros((num_users, num_items))

    for row in input_ratings:
        ratings[int(row[0]), int(row[1])] = row[2]

    sparsity = 100 * float(np.count_nonzero(ratings)) / float(num_users * num_items)
    logging.info("Sparsity: %.2f%%", sparsity)

    return ratings
