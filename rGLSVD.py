import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def rGLSVD(
    clusters: np.ndarray,
    R_bin: np.ndarray,
    num_clust: int = 7,
    fg: int = 10,
    fc: dict[int, int] = {0: 7, 1: 10, 2: 15, 3: 12, 4: 22, 5: 15, 6: 20},
) -> np.ndarray:
    """Applies the rGLSVD algorithm to compute user global weights.

    This function iteratively calculates the global weight vector `gu_vector`
    for each user by combining global and local SVD decompositions
    of a binary rating matrix.

    Args:
        clusters (np.ndarray): 1D array of cluster assignments for each user.
        R_bin (np.ndarray): 2D user-item binary rating matrix (num_users x num_items).
        num_clust (int, optional): Number of user clusters. Defaults to 7.
        fg (int, optional): Rank for the global SVD. Defaults to 10.
        fc (dict[int, int], optional): Dictionary mapping cluster index to rank for local SVD.
            Defaults to {0: 7, 1: 10, 2: 15, 3: 12, 4: 22, 5: 15, 6: 20}.

    Returns:
        np.ndarray: Final user global weight vector (`gu_vector`) with values in [0, 1].
    """
    num_users, num_items = R_bin.shape
    gu_vector = np.full(num_users, 0.5)  # vector of initial weights
    num_changed_users = 50
    eps = 1e-12  # guard against division by zero

    while num_changed_users / num_users > 0.01:
        # Global Matrix initialization
        R_global = gu_vector[:, np.newaxis] * R_bin

        # Local matrices: explicit ndarray lists to satisfy mypy
        R_local: list[np.ndarray] = [np.empty((0, 0)) for _ in range(num_clust)]
        for i in range(num_clust):  # Local matrix initialization
            user_indices_in_cluster = np.where(clusters == i)[0]
            local_gu_vector = 1.0 - gu_vector[user_indices_in_cluster]
            cluster_R_bin = R_bin[user_indices_in_cluster, :]
            R_local[i] = local_gu_vector[:, np.newaxis] * cluster_R_bin

        # Global decomposition
        U_global, sigma_global, Item_global = svds(R_global, k=fg)

        # Local decompositions (typed lists)
        U_local: list[np.ndarray] = [np.empty((0, 0)) for _ in range(num_clust)]
        sigma_local: list[np.ndarray] = [np.empty((0, 0)) for _ in range(num_clust)]
        Vt_local: list[np.ndarray] = [
            np.empty((0, num_items)) for _ in range(num_clust)
        ]

        for i in range(num_clust):  # SVD on local matrices
            R_local_sparse = sp.csr_matrix(R_local[i])
            # Note: assumes fc[i] is valid for given cluster size; keep as-is per your code
            U_loc, sigma_loc_diag, Vt_loc = svds(R_local_sparse, k=fc[i])

            U_local[i] = U_loc
            Vt_local[i] = Vt_loc
            sigma_local[i] = np.diag(sigma_loc_diag)

        gu_new = np.zeros(num_users)  # new gu vector
        Q_global = Item_global.T
        Sigma_global_mat = np.diag(sigma_global)

        for u in range(num_users):
            r_u_actual = R_bin[u, :]

            # Global predictions
            p_u_global = U_global[u, :]
            predictions_global = p_u_global.dot(Sigma_global_mat).dot(Q_global.T)

            # Local predictions
            cluster_id = int(clusters[u])
            U_loc = U_local[cluster_id]
            Vt_loc = Vt_local[cluster_id]
            sigma_loc = sigma_local[cluster_id]

            user_indices_in_cluster = np.where(clusters == cluster_id)[0]
            # Assumes user u belongs to this cluster and is present
            local_u_index = np.where(user_indices_in_cluster == u)[0][0]

            p_u_local = U_loc[local_u_index, :]
            predictions_local = p_u_local.dot(sigma_loc).dot(Vt_loc)

            # --- Eq. (3) correction with safeguards ---
            gu = gu_vector[u]
            a = predictions_global / max(gu, eps)
            b = predictions_local / max(1.0 - gu, eps)
            diff_pred = a - b

            numerator = float(np.sum(diff_pred * (r_u_actual - b)))
            denominator = float(np.sum(diff_pred**2))

            if denominator == 0.0:
                gu_new[u] = 1.0
            else:
                gu_new[u] = float(np.clip(numerator / denominator, 0.0, 1.0))

        diff_vector = np.abs(gu_new - gu_vector)
        num_changed_users = int(np.sum(diff_vector > 0.01))

        gu_vector = gu_new

    return gu_vector
