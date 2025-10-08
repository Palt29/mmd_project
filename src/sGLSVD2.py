import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from typing import Optional


def sGLSVD(
    initial_clusters: np.ndarray,
    bin_utility_matrix: np.ndarray,
    num_clusters: int,
    f_g: int,
    f_c: int,
    min_error_improvement: float = 0.01,
) -> tuple[
    tuple[
        np.ndarray,  # gu_vector
        np.ndarray,  # user_global
        np.ndarray,  # sigma_global_matrix
        np.ndarray,  # item_global
        list[Optional[np.ndarray]],  # user_local_list
        list[Optional[np.ndarray]],  # sigma_local_list
        list[Optional[np.ndarray]],  # item_local_list
        list[np.ndarray],  # cluster_global_indices
    ],
    np.ndarray,  # clusters
]:
    """Implements the sGLSVD algorithm with cluster reassignment and adaptive gu update."""

    num_users, _ = bin_utility_matrix.shape
    gu_vector = np.full(num_users, 0.5)
    clusters = initial_clusters.copy()

    eps = 1e-12
    iteration = 0
    prev_total_error = np.inf
    num_changed_users = num_users

    cluster_global_indices: list[Optional[np.ndarray]] = [None] * num_clusters

    # Initialize lists for local factors
    user_local: list[Optional[np.ndarray]] = [None] * num_clusters
    sigma_local_matrices: list[Optional[np.ndarray]] = [None] * num_clusters
    item_local: list[Optional[np.ndarray]] = [None] * num_clusters

    while num_changed_users / num_users > 0.01:
        iteration += 1

        # GLOBAL FACTORS
        R_global = sp.csr_matrix(gu_vector[:, np.newaxis] * bin_utility_matrix) #sparse format
        user_global, sigma_global, item_global = svds(R_global, k=f_g)
        sigma_global_matrix = np.diag(sigma_global)

        # LOCAL FACTORS
        for cluster_idx in range(num_clusters):
            user_indices = np.where(clusters == cluster_idx)[0]
            cluster_global_indices[cluster_idx] = user_indices

            if user_indices.size == 0:
                continue

            local_gu = 1.0 - gu_vector[user_indices]
            R_local = sp.csr_matrix(local_gu[:, np.newaxis] * bin_utility_matrix[user_indices, :]) #sparse format
            U_loc, sigma_loc_diag, Vt_loc = svds(R_local, k=f_c)

            user_local[cluster_idx] = U_loc
            sigma_local_matrices[cluster_idx] = np.diag(sigma_loc_diag)
            item_local[cluster_idx] = Vt_loc

        # CLUSTER REASSIGNMENT
        gu_new = gu_vector.copy()
        clusters_new = clusters.copy()
        total_error = 0.0
        users_switched = 0

        for u in range(num_users):
            r_u = bin_utility_matrix[u, :]
            current_cluster = int(clusters[u])

            min_error = np.inf
            best_cluster = current_cluster
            best_gu = gu_vector[u]

            # Evaluate current cluster
            if user_local[current_cluster] is not None:
                user_indices = cluster_global_indices[current_cluster]
                local_pos = np.where(user_indices == u)[0]
                if local_pos.size > 0:
                    local_u = int(local_pos[0])

                    U_loc = user_local[current_cluster]
                    Sigma_loc = sigma_local_matrices[current_cluster]
                    Vt_loc = item_local[current_cluster]

                    if U_loc is not None and Sigma_loc is not None and Vt_loc is not None:
                        p_u_global = user_global[u, :]
                        p_u_local = U_loc[local_u, :]

                        # Equation 3
                        pred_global = p_u_global.dot(sigma_global_matrix).dot(item_global)
                        pred_local = p_u_local.dot(Sigma_loc).dot(Vt_loc)

                        diff_pred = pred_global - pred_local
                        num = float(np.sum(diff_pred * (r_u - pred_local)))
                        den = float(np.sum(diff_pred**2))

                        g_u_opt = gu_vector[u] if den < eps else float(np.clip(num / den, 0.0, 1.0))

                        # Equation 5
                        final_pred = g_u_opt * pred_global + (1.0 - g_u_opt) * pred_local

                        min_error = float(np.sum((r_u - final_pred) ** 2))
                        best_gu = g_u_opt

            # Evaluate alternative clusters
            for c_test in range(num_clusters):
                if c_test == current_cluster or user_local[c_test] is None:
                    continue

                Vt_loc = item_local[c_test]
                Sigma_loc = sigma_local_matrices[c_test]
                sigma_diag = np.diag(Sigma_loc) if Sigma_loc is not None else None
                if Vt_loc is None or sigma_diag is None:
                    continue
                
                # Equation 4
                inv_sigma = np.diag(1.0 / np.where(sigma_diag > eps, sigma_diag, eps))
                p_u_local_test = r_u.dot(Vt_loc.T.dot(inv_sigma))

                # Equation 3
                p_u_global = user_global[u, :]
                pred_global = p_u_global.dot(sigma_global_matrix).dot(item_global)
                pred_local = p_u_local_test.dot(Sigma_loc).dot(Vt_loc)

                diff_pred = pred_global - pred_local
                num = float(np.sum(diff_pred * (r_u - pred_local)))
                den = float(np.sum(diff_pred**2))
                g_u_test = 0.5 if den < eps else float(np.clip(num / den, 0.0, 1.0))

                #Equation 5
                error = float(np.sum((r_u - (g_u_test * pred_global + (1.0 - g_u_test) * pred_local)) ** 2))
                
                if error < min_error * (1.0 - min_error_improvement):
                    min_error = error
                    best_cluster = c_test
                    best_gu = g_u_test

            if best_cluster != current_cluster:
                users_switched += 1

            clusters_new[u] = best_cluster
            gu_new[u] = best_gu
            total_error += min_error

        # CONVERGENCE
        num_changed_users = users_switched
        improvement = (prev_total_error - total_error) / prev_total_error if prev_total_error != np.inf else 1.0

        print(
            f"Iter {iteration}: Switch={users_switched} ({users_switched / num_users:.1%}), "
            f"Error={total_error:.6f}, Improvement={improvement:.6f}"
        )

        prev_total_error = total_error
        gu_vector = gu_new
        clusters = clusters_new

    # FINAL LOCAL FACTORS
    for cluster_idx in range(num_clusters):
        user_indices = np.where(clusters == cluster_idx)[0]
        cluster_global_indices[cluster_idx] = user_indices
        if user_indices.size == 0:
            continue

        local_gu = 1.0 - gu_vector[user_indices]
        R_local = sp.csr_matrix(local_gu[:, np.newaxis] * bin_utility_matrix[user_indices, :])
        U_loc, sigma_loc_diag, Vt_loc = svds(R_local, k=f_c)
        user_local[cluster_idx] = U_loc
        sigma_local_matrices[cluster_idx] = np.diag(sigma_loc_diag)
        item_local[cluster_idx] = Vt_loc

    return (
        (
            gu_vector,
            user_global,
            sigma_global_matrix,
            item_global,
            user_local,
            sigma_local_matrices,
            item_local,
            cluster_global_indices,
        ),
        clusters,
    )
