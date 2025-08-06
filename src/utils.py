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
        out:
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
    # Remove the first line
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
    # Create an NxI matrix of zeros,
    # where N = number of users
    # and I = number of items
    ratings = np.zeros((num_users, num_items))

    # Fill the matrix with the ratings
    # NOTE: input_ratings: list[list]
    for row in input_ratings:
        ratings[int(row[0]), int(row[1])] = row[2]

    # Compute the "sparsity", i.e., percentage of non-zero cells
    sparsity = 100 * float(np.count_nonzero(ratings)) / float(num_users * num_items)
    logging.info("Sparsity: %.2f%%", sparsity)

    return ratings


def count_and_avg_rating_per_user(utility_matrix: np.ndarray) -> np.ndarray:
    """Computes the number of rated items and average rating per user.

    Args:
        utility_matrix: A 2D NumPy array of shape (num_users, num_items),
            where each cell [i, j] contains the rating from user i for item j,
            or 0 if no rating was given.

    Returns:
        out:
        A NumPy array of shape (num_users, 2), where:
            - [:, 0] contains the count of rated items per user.
            - [:, 1] contains the average rating per user (0 if no ratings).
    """
    counts = np.count_nonzero(utility_matrix, axis=1)
    sums = utility_matrix.sum(axis=1)
    averages = np.zeros_like(sums, dtype=float)
    mask = counts > 0
    averages[mask] = sums[mask] / counts[mask]
    return np.vstack((counts, averages)).T


def top_users_by_count(user_info: np.ndarray, top_k: int) -> np.ndarray:
    """Returns the top-k users by number of rated items, including their indices.

    Args:
        user_info: NumPy array of shape (num_users, 2), where:
            - [:, 0] contains the count of rated items per user.
            - [:, 1] contains the average rating per user.
        top_k: Number of top users to return.

    Returns:
        out:
        A NumPy array of shape (top_k, 3), where each row contains:
            [user_index, count, average_rating].
    """
    indices = np.argsort(user_info[:, 0])[::-1][:top_k]
    return np.column_stack((indices, user_info[indices]))


def count_and_avg_ratings_per_item(utility_matrix: np.ndarray) -> np.ndarray:
    """Computes the number of ratings and average rating per item.

    Args:
        utility_matrix: A 2D NumPy array of shape (num_users, num_items),
            where each cell [i, j] contains the rating from user i for item j,
            or 0 if no rating was given.

    Returns:
        out:
        A NumPy array of shape (num_items, 2), where:
            - [:, 0] contains the count of ratings per item.
            - [:, 1] contains the average rating per item (0 if unrated).
    """
    counts = np.count_nonzero(utility_matrix, axis=0)
    sums = utility_matrix.sum(axis=0)
    averages = np.zeros_like(sums, dtype=float)
    mask = counts > 0
    averages[mask] = sums[mask] / counts[mask]
    return np.vstack((counts, averages)).T


def top_items_by_avg_and_count(
    item_info: np.ndarray, min_count: int, top_k: int
) -> np.ndarray:
    """Returns the top-k items with the highest average rating among those with sufficient ratings.

    Args:
        item_info: NumPy array of shape (num_items, 2), where:
            - [:, 0] contains the count of ratings per item.
            - [:, 1] contains the average rating per item.
        min_count: Minimum number of ratings required to include an item.
        top_k: Number of top items to return (default is 10).

    Returns:
        out:
        A NumPy array of shape (top_k, 3), where each row contains:
            [item_index, num_ratings, avg_rating].
    """
    mask = item_info[:, 0] >= min_count
    filtered = item_info[mask]
    indices = np.where(mask)[0]

    sorted_idx = np.argsort(filtered[:, 1])[::-1][:top_k]
    return np.column_stack((indices[sorted_idx], filtered[sorted_idx]))


def normalize_utility_matrix(utility_matrix: np.ndarray) -> np.ndarray:
    """Normalizes the utility matrix by subtracting each user's average rating.

    Each nonzero entry is replaced with (rating - user_avg). Unrated entries remain 0.

    Args:
        utility_matrix: A 2D NumPy array of shape (num_users, num_items),
            where each cell [i, j] contains the rating from user i for item j,
            or 0 if no rating was given.

    Returns:
        A normalized utility matrix of the same shape as the input,
        with zero-mean rows (for nonzero entries).
    """
    user_info = count_and_avg_rating_per_user(utility_matrix)
    avg_ratings = user_info[:, 1][:, None]
    return np.where(utility_matrix != 0, utility_matrix - avg_ratings, 0)


def train_test_split_v2(
    ratings: np.ndarray, sample_per_user: int = 10, seed: int = 123425536
) -> tuple[np.ndarray, np.ndarray]:
    """Splits the utility matrix into train and test sets for evaluation.

    For each user, a percentage of their ratings is randomly selected and moved to the test set.
    The resulting train and test matrices are disjoint.

    Args:
        ratings: A 2D NumPy array of shape (num_users, num_items),
            containing the full utility matrix with ratings.
        sample_per_user: Percentage of each user's ratings to sample for the test set.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of two NumPy arrays (train, test), each of shape (num_users, num_items).
    """
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    np.random.seed(seed)
    for user in range(ratings.shape[0]):
        num_ratings = len(ratings[user, :].nonzero()[0])
        if num_ratings == 0:
            continue
        actual_sample = int(num_ratings * sample_per_user / 100)
        test_ratings = np.random.choice(
            ratings[user, :].nonzero()[0],
            size=actual_sample,
            replace=False,
        )
        train[user, test_ratings] = 0.0
        test[user, test_ratings] = ratings[user, test_ratings]

    assert np.all((train * test) == 0)
    return train, test


# Cosine similarity (alternatively, Pearson correlation) is used in memory-based collaborative filtering
# (e.g., user-user or item-item CF).
# It is not applicable for rGLSVD and sGLSVD, which require clustering methods to assign users into fixed subsets.
def compute_similarity(
    ratings: np.ndarray,
    kind: str = "user",
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Returns a cosine-similarity matrix for users or items.

    Args:
        ratings: Utility matrix of shape (num_users, num_items), where each
            non-zero entry denotes user-item interaction strength.
        kind: Either `"user"` for a user-user similarity matrix or `"item"` for
            an item-item similarity matrix.
        epsilon: Small constant added to the dot-product matrix to prevent
            division-by-zero when normalizing.

    Returns:
        out:
        A square NumPy array:
            * (num_users, num_users) if `kind == "user"`,
            * (num_items, num_items) if `kind == "item"`,
        where each entry ∈ [0, 1] is the cosine similarity between the
        corresponding user or item vectors.
    """
    # We compute the dot product between ratings
    # epsilon -> small number for handling divide-by-zero errors
    if kind == "user":
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == "item":
        sim = ratings.T.dot(ratings) + epsilon
    else:
        raise ValueError("kind must be 'user' or 'item'.")

    # From the diagonal of the dot product matrix we extract the norms
    # (diagonal contains squared magnitudes: v·v = ||v||²)
    norms = np.array([np.sqrt(np.diagonal(sim))])

    # The double division allows broadcasting
    result: np.ndarray = sim / norms / norms.T
    return result


# The below function is crucial for rGLSVD and sGLSVD implementations
# because they do not use explicit ratings,
# but only binary indicators of interaction (implicit feedback)
def binarize_ratings(utility_matrix: np.ndarray) -> np.ndarray:
    """
    Converts a user-item rating matrix into a binary implicit feedback matrix.

    All positive ratings are set to 1.0 (indicating user interaction),
    and all zero or missing ratings are set to 0.0.

    Args:
        utility_matrix: A 2D NumPy array of shape (num_users, num_items),
            where each cell [i, j] contains the rating from user i for item j,
            or 0 if no rating was given

    Returns:
        A binary matrix of the same shape as input, where
        entries are 1.0 if rating > 0, and 0.0 otherwise.
    """
    return (utility_matrix > 0).astype(float)
