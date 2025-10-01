import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def rGLSVD(
    clusters: np.ndarray,
    bin_utiliy_matrix: np.ndarray,
    num_clust: int = 7,
    f_g: int = 10,
    f_c: dict[int, int] = {
        0: 7,
        1: 10,
        2: 15,
        3: 12,
        4: 22,
        5: 15,
        6: 20,
    },  # test numbers
) -> np.ndarray:
    """Applies the rGLSVD algorithm to compute user global weights.

    This function iteratively calculates the global weight vector `gu_vector`
    for each user by combining global and local SVD decompositions
    of a binary rating matrix.

    Args:
        clusters (np.ndarray): 1D array of cluster assignments for each user.
        bin_utiliy_matrix (np.ndarray): 2D binary utility matrix (num_users x num_items).
        num_clust (int, optional): Number of user clusters. Defaults to 7.
        f_g (int, optional): Rank for the global SVD.
        f_c (dict[int, int], optional): Dictionary mapping cluster index to rank for local SVD.

    Returns:
        np.ndarray: Final user global weight vector (`gu_vector`) with values in [0, 1].
    """
    num_users, num_items = bin_utiliy_matrix.shape
    gu_vector = np.full(num_users, 0.5)  # initial weights vector
    num_changed_users = num_users
    eps = 1e-12  # guard against division by zero
    cluster_global_indices: list[np.ndarray | None] = [None] * num_clust

    # we assign num_change_users = num_users
    # just to start the while loop
    while num_changed_users / num_users > 0.01:
        # Global matrix initialization
        R_global = gu_vector[:, np.newaxis] * bin_utiliy_matrix
        # Local matrices initialization (None as placeholder)
        R_local: list[np.ndarray | None] = [None] * num_clust

        # Global decomposition
        # Notation in the paper:
        # user_global = P, shape = num_user x f_g
        # sigma_global = singular values matrix, shape f_g x f_g
        # item_global = Q, shape = num_items x f_g
        R_global_sparse = sp.csr_matrix(R_global)

        user_global, sigma_global, item_global = svds(R_global_sparse, k=f_g)

        # Local decompositions (None as placeholder, later filled with arrays)
        user_local: list[np.ndarray | None] = [None] * num_clust
        sigma_local_matrices: list[np.ndarray | None] = [None] * num_clust
        item_local: list[np.ndarray | None] = [None] * num_clust

        for i in range(num_clust):
            user_indices_in_cluster = np.where(clusters == i)[0]
            cluster_global_indices[i] = user_indices_in_cluster
            local_gu_vector = 1.0 - gu_vector[user_indices_in_cluster]
            bin_cluster_utility_matrix = bin_utiliy_matrix[user_indices_in_cluster, :]
            R_local[i] = local_gu_vector[:, np.newaxis] * bin_cluster_utility_matrix

            # Local decompositions
            R_local_sparse = sp.csr_matrix(
                R_local[i]
            )  # use sparse format to speed up svds on binary matrices

            U_loc, sigma_loc_diag, Vt_loc = svds(R_local_sparse, k=f_c[i])

            user_local[i] = U_loc
            item_local[i] = Vt_loc
            sigma_local_matrices[i] = np.diag(sigma_loc_diag)

        # Eq. (3) from the paper (Section 4.3)
        gu_new = np.zeros(num_users)  # new gu vector
        sigma_global_matrix = np.diag(sigma_global)

        # gu_vector shape = num_users x 1
        for u in range(num_users):
            user_vector = bin_utiliy_matrix[u, :]

            # Global predictions
            p_user_global = user_global[u, :]  # p_user_global = p_u, shape = 1 x f_g
            predictions_global = (
                (p_user_global.T).dot(sigma_global_matrix).dot(item_global)
            )  # (1 x f_g) @ (f_g x f_g) @ (f_g x num_items) = 1 x num_items

            # Local predictions
            cluster_id = int(clusters[u])
            U_loc = user_local[cluster_id]
            Vt_loc = item_local[cluster_id]
            sigma_loc = sigma_local_matrices[cluster_id]

            user_indices_in_cluster = np.where(clusters == cluster_id)[0]
            # Assumes user u belongs to this cluster and is present
            local_u_index = np.where(user_indices_in_cluster == u)[0][0]

            p_user_local = U_loc[local_u_index, :]
            predictions_local = p_user_local.dot(sigma_loc).dot(Vt_loc)

            gu = gu_vector[u]
            # eps allows us to guard against division by zero
            a = predictions_global / max(gu, eps)
            b = predictions_local / max(1.0 - gu, eps)
            diff_pred = a - b

            numerator = float(np.sum(diff_pred * (user_vector - b)))
            denominator = float(np.sum(diff_pred**2))

            if denominator == 0.0:
                gu_new[u] = 1.0
            else:
                gu_new[u] = float(np.clip(numerator / denominator, 0.0, 1.0))

        diff_vector = np.abs(gu_new - gu_vector)
        num_changed_users = int(np.sum(diff_vector > 0.01))

        gu_vector = gu_new

    return (gu_vector, 
        user_global,  # P
        sigma_global_matrix, #fg matrix
        item_global,       # Q
        user_local,   # P^c
        sigma_local_matrices, #fc matrices
        item_local,  #Q^c
        cluster_global_indices 
        )
