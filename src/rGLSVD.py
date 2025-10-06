import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def rGLSVD(
    bin_utility_matrix: np.ndarray,
    clusters: np.ndarray,
    num_clusters: int,  # TODO: We could extract this value from `clusters` variable
    f_g: int,
    f_c: dict[int, int],
) -> tuple[
    np.ndarray,  # gu_vector
    np.ndarray,  # user_global (P)
    np.ndarray,  # sigma_global_matrix (Σ_g)
    np.ndarray,  # item_global (Q)
    list[np.ndarray],  # user_local (P^c)
    list[np.ndarray],  # sigma_local_matrices (Σ_c)
    list[np.ndarray],  # item_local (Q^c)
    list[np.ndarray],  # cluster_global_indices
]:
    """Applies the rGLSVD algorithm to compute user global weights.

    This function iteratively calculates the global weight vector `gu_vector`
    for each user by combining global and local SVD decompositions
    of a binary rating matrix.

    Args:
        clusters (np.ndarray): 1D array of cluster assignments for each user.
        bin_utility_matrix (np.ndarray): 2D binary utility matrix (num_users x num_items).
        num_clusters (int, optional): Number of user clusters.
        f_g (int, optional): Rank for the global SVD.
        f_c (dict[int, int], optional): Dictionary mapping cluster index to rank for local SVD.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final user global weight vector (`gu_vector`) with values in [0, 1].
            - np.ndarray: Global user latent matrix (`user_global`, P).
            - np.ndarray: Global singular values matrix (`sigma_global_matrix`, Σ_g).
            - np.ndarray: Global item latent matrix (`item_global`, Q).
            - list[np.ndarray]: Local user latent matrices (`user_local`, P^c for each cluster).
            - list[np.ndarray]: Local singular value matrices (`sigma_local_matrices`, Σ_c).
            - list[np.ndarray]: Local item latent matrices (`item_local`, Q^c for each cluster).
            - list[np.ndarray]: Indices of users belonging to each cluster (`cluster_global_indices`).
    """
    if num_clusters != np.unique(clusters).size:
        raise ValueError("Incorrect number of clusters specified.")

    # num_users, num_items -> the latter variable is not accessed
    num_users, _ = bin_utility_matrix.shape
    # Initial weights vector (weights set to 0.5 for each user)
    gu_vector: np.ndarray = np.full(num_users, 0.5)

    # cluster_global_indices will be populated with arrays containing indices
    # of users belonging to each cluster
    cluster_global_indices: list[np.ndarray] = [
        np.empty((0,), dtype=int) for _ in range(num_clusters)
    ]
    # Initialize empty containers for local SVD components (one per cluster):
    # user_local, sigma_local_matrices, item_local
    # np.empty((0, 0)) creates an empty array
    user_local: list[np.ndarray] = [np.empty((0, 0)) for _ in range(num_clusters)]
    sigma_local_matrices: list[np.ndarray] = [
        np.empty((0, 0)) for _ in range(num_clusters)
    ]
    item_local: list[np.ndarray] = [np.empty((0, 0)) for _ in range(num_clusters)]

    eps = 1e-12  # Guard against division by zero
    # We assign num_change_users = num_users just to start the while loop
    num_changed_users = num_users

    while num_changed_users / num_users > 0.01:
        # Global matrix initialization
        R_global = gu_vector[:, np.newaxis] * bin_utility_matrix
        # Local matrices initialization (None as placeholder)
        R_local: list[np.ndarray | None] = [None] * num_clusters

        # Global decomposition
        # On the right, notation according to the paper (P, Σ, Q),
        # while the shapes reported are based on svds from scipy:
        # user_global = P, shape = num_user x f_g
        # sigma_global = singular values matrix (Σ), shape f_g x f_g
        # item_global = Q, shape = f_g x num_items

        # sp.csr_matrix allows us to treat the matrix as sparse
        R_global_sparse = sp.csr_matrix(R_global)

        user_global, sigma_global, item_global = svds(R_global_sparse, k=f_g)

        for cluster_idx in range(num_clusters):
            user_indices_in_cluster = np.where(clusters == cluster_idx)[0]
            # For each cluster, we assign the corresponding users' indices
            cluster_global_indices[cluster_idx] = user_indices_in_cluster
            local_gu_vector = 1.0 - gu_vector[user_indices_in_cluster]
            bin_cluster_utility_matrix = bin_utility_matrix[user_indices_in_cluster, :]
            # Same thing we did for the globlal matrix, just on a cluster level
            R_local[cluster_idx] = (
                local_gu_vector[:, np.newaxis] * bin_cluster_utility_matrix
            )

            # Local decompositions
            R_local_sparse = sp.csr_matrix(
                R_local[cluster_idx]
            )  # use sparse format to speed up svds on binary matrices

            U_loc, sigma_loc_diag, Vt_loc = svds(R_local_sparse, k=f_c[cluster_idx])

            user_local[cluster_idx] = U_loc
            sigma_local_matrices[cluster_idx] = np.diag(sigma_loc_diag)
            item_local[cluster_idx] = Vt_loc

        # -------------------- Eq. (3) from the paper, Section 4.3 --------------------
        gu_new = np.zeros(num_users)  # new gu vector
        sigma_global_matrix = np.diag(sigma_global)

        # gu_vector shape = num_users x 1
        for user_idx in range(num_users):
            user_vector = bin_utility_matrix[user_idx, :]

            # -------------------- GLOBAL PREDICTIONS --------------------
            p_user_global = user_global[
                user_idx, :
            ]  # p_user_global = p_u on the paper, shape = 1 x f_g
            predictions_global = (
                (p_user_global).dot(sigma_global_matrix).dot(item_global)
            )  # (1 x f_g) @ (f_g x f_g) @ (f_g x num_items) = 1 x num_items

            # -------------------- LOCAL PREDICTIONS --------------------
            cluster_id = int(clusters[user_idx])
            U_loc = user_local[cluster_id]
            sigma_loc = sigma_local_matrices[cluster_id]
            Vt_loc = item_local[cluster_id]

            user_indices_in_cluster = np.where(clusters == cluster_id)[0]
            # Assumes user u belongs to this cluster and is present
            local_u_index = np.where(user_indices_in_cluster == user_idx)[0][0]

            p_user_local = U_loc[local_u_index, :]
            predictions_local = p_user_local.dot(sigma_loc).dot(Vt_loc)

            gu = gu_vector[user_idx]
            # eps variable allows us to guard against division by zero
            a = predictions_global / max(gu, eps)
            b = predictions_local / max(1.0 - gu, eps)
            diff_pred = a - b

            numerator = float(np.sum(diff_pred * (user_vector - b)))
            denominator = float(np.sum(diff_pred**2))

            # Assumption: if the computation is not possible,
            # the user weight is considered global (= 1)
            if denominator == 0.0:
                gu_new[user_idx] = 1.0
            else:
                gu_new[user_idx] = float(np.clip(numerator / denominator, 0.0, 1.0))

        diff_vector = np.abs(gu_new - gu_vector)
        num_changed_users = int(np.sum(diff_vector > 0.01))

        gu_vector = gu_new

    return (
        gu_vector,
        user_global,  # P
        sigma_global_matrix,  # fg matrix
        item_global,  # Q
        user_local,  # P^c
        sigma_local_matrices,  # fc matrices
        item_local,  # Q^c
        cluster_global_indices,
    )
