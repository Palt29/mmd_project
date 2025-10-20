import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# ------------ Logger setup ------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SGLSVDRecommender:
    """sGLSVD Recommender System with cluster reassignment and adaptive gu update,
    including methods for LOOCV and evaluation.
    """

    def __init__(
        self,
        R_bin: np.ndarray,
        initial_clusters: np.ndarray,
        num_clusters: int,
        f_g: int,
        f_c: int,
    ) -> None:
        """Initialize sGLSVD configuration and data.

        Args:
            R_bin (np.ndarray): Binary utility matrix (num_users x num_items).
            initial_clusters (np.ndarray): 1D array assigning each user to an initial cluster id in [0, num_clusters).
            num_clusters (int): Total number of user clusters.
            f_g (int): Global SVD rank.
            f_c (int): Local SVD rank (shared by all clusters).
        """
        self.R_bin = R_bin
        self.clusters = initial_clusters.copy()
        self.num_clusters = num_clusters
        self.f_g = f_g
        self.f_c = f_c

        # Track model state
        self.is_fitted: bool = False
        self.trained_factors: (
            tuple[
                np.ndarray,  # gu_vector
                np.ndarray,  # user_global (P)
                np.ndarray,  # sigma_global_matrix (Σ_g)
                np.ndarray,  # item_global (Q)
                list[np.ndarray],  # user_local (P^c)
                list[np.ndarray],  # sigma_local_matrices (Σ_c)
                list[np.ndarray],  # item_local (Q^c)
                list[np.ndarray],  # cluster_global_indices
            ]
            | None
        ) = None
        self.num_iterations: int = 0  # Track convergence iterations

    def loocv_split(self) -> tuple[np.ndarray, dict[int, int]]:
        """
        Perform Leave-One-Out (LOOCV) splitting of the dataset for Top-N evaluation.

        For each user, selects one interacted item (entry equal to 1) to hold out for testing,
        setting its corresponding entry to 0 in the training matrix.

        Returns:
            out (tuple[np.ndarray, dict[int, int]]):
                - train_matrix: Copy of self.R_bin with one held-out item per user set to 0.
                - test_items: Mapping {user_index: held_out_item_index}.
        """
        bin_utility_matrix = self.R_bin
        num_users, _ = bin_utility_matrix.shape
        test_items: dict[int, int] = {}
        train_matrix = bin_utility_matrix.copy()

        # Randomly select one rated item from each user and place it in the test set
        for user_idx in range(num_users):
            # Indices of items that the user has interacted with (entries equal to 1)
            rated_items = np.where(bin_utility_matrix[user_idx, :] == 1)[0]

            if len(rated_items) > 0:
                # Randomly choose one rated item to leave out (test set)
                test_item = np.random.choice(rated_items, 1)[0]
                test_items[user_idx] = int(test_item)

                # Set the held-out test item to 0 in the training matrix
                train_matrix[user_idx, test_item] = 0

        return train_matrix, test_items

    def fit(
        self,
        min_error_improvement: float = 0.01,
        convergence_fraction_threshold: float = 0.005,
        max_iterations: int | None = 20,
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
        """
        Run sGLSVD algorithm on the binary utility matrix.

        The sGLSVD algorithm alternates between global and local truncated SVD
        decompositions, updating both the user-specific global weights *g_u*
        and the cluster assignments for all users. The global SVD is weighted
        by *g_u*, while the local SVDs are computed per cluster and weighted
        by *(1 - g_u)*. At each iteration, users are reassigned to the cluster
        that minimizes their reconstruction error, provided that the improvement
        exceeds a minimum threshold.

        Args:
            min_error_improvement (float): Minimum relative improvement in the reconstruction
                error required to trigger a cluster reassignment. Default: 0.01.
            convergence_fraction_threshold (float): Algorithm stops when the fraction of users
                changing clusters between iterations falls below this threshold. Default: 0.005.
            max_iterations (int | None): Maximum number of iterations before forcing stop.
                If None, runs until convergence. Default: 20.

        Returns:
            out: A tuple containing:
                - np.ndarray: Final user global weight vector (`gu_vector`) with values in [0, 1].
                - np.ndarray: Global user latent matrix (`user_global`, P).
                - np.ndarray: Global singular values matrix (`sigma_global_matrix`, Σ_g).
                - np.ndarray: Global item latent matrix (`item_global`, Q).
                - list[np.ndarray]: Local user latent matrices (`user_local`, P^c for each cluster).
                - list[np.ndarray]: Local singular value matrices (`sigma_local_matrices`, Σ_c).
                - list[np.ndarray]: Local item latent matrices (`item_local`, Q^c for each cluster).
                - list[np.ndarray]: Indices of users belonging to each cluster (`cluster_global_indices`).
        """
        bin_utility_matrix = self.R_bin
        clusters = self.clusters.copy()
        num_clusters = self.num_clusters
        f_g = self.f_g
        f_c = self.f_c

        if num_clusters != np.unique(clusters).size:
            raise ValueError("Incorrect number of clusters specified.")

        # num_users, num_items -> the latter variable is not accessed
        num_users, _ = bin_utility_matrix.shape
        # Initial weights vector (weights set to 0.5 for each user)
        gu_vector = np.full(num_users, 0.5)

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

        # iterations_count tracking and convergence setup
        eps = 1e-12  # Guard against division by zero
        iterations_count = 0
        # We assign num_change_users = num_users just to start the while loop
        num_changed_users = num_users
        prev_total_error = np.inf

        # Continue iterations until the fraction of changed users falls below the threshold or until max_iterations is reached
        while (num_changed_users / num_users) > convergence_fraction_threshold:
            iterations_count += 1
            if max_iterations is not None and iterations_count > max_iterations:
                logger.info(f"Stopping: reached max_iterations={max_iterations}")
                break

            # Global matrix initialization
            R_global = sp.csr_matrix(gu_vector[:, np.newaxis] * bin_utility_matrix)
            # Ensure k for svds() is valid: must be smaller than min(matrix dimensions)
            k_g = min(f_g, min(R_global.shape) - 1) if min(R_global.shape) > 1 else 1

            try:
                user_global, sigma_global, item_global = svds(R_global, k=k_g)
            except Exception:
                # Fallback: if svds() fails (e.g., matrix too small or ill-conditioned),
                # perform a full SVD using NumPy instead (less efficient)
                U, s, Vt = np.linalg.svd(R_global.toarray(), full_matrices=False)
                user_global = U[:, :f_g]
                sigma_global = s[:f_g]
                item_global = Vt[:f_g, :]
            sigma_global_matrix = np.diag(sigma_global)

            # --- Local factor computation for each cluster ---
            for cluster_idx in range(num_clusters):
                user_indices = np.where(clusters == cluster_idx)[0]
                cluster_global_indices[cluster_idx] = user_indices

                if user_indices.size == 0:
                    user_local[cluster_idx] = np.empty((0, 0))
                    sigma_local_matrices[cluster_idx] = np.empty((0, 0))
                    item_local[cluster_idx] = np.empty((0, 0))
                    continue

                local_gu = 1.0 - gu_vector[user_indices]
                R_local = sp.csr_matrix(
                    local_gu[:, np.newaxis] * bin_utility_matrix[user_indices, :]
                )

                # k for local svds(): must not exceed min(matrix dimensions) - 1
                min_dim = min(R_local.shape)
                if min_dim <= 1:
                    user_local[cluster_idx] = np.empty((0, 0))
                    sigma_local_matrices[cluster_idx] = np.empty((0, 0))
                    item_local[cluster_idx] = np.empty((0, 0))
                    continue

                k_c = min(f_c, min_dim - 1)
                try:
                    U_loc, sigma_loc_diag, Vt_loc = svds(R_local, k=k_c)
                    user_local[cluster_idx] = U_loc
                    sigma_local_matrices[cluster_idx] = np.diag(sigma_loc_diag)
                    item_local[cluster_idx] = Vt_loc
                except Exception:
                    # Simple fallback: skip the SVD for this cluster if it fails
                    user_local[cluster_idx] = np.empty((0, 0))
                    sigma_local_matrices[cluster_idx] = np.empty((0, 0))
                    item_local[cluster_idx] = np.empty((0, 0))

            # --- Cluster reassignment and g_u update ---
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

                # Evaluate current cluster (if available)
                if (
                    user_local[current_cluster].size != 0
                    and sigma_local_matrices[current_cluster].size != 0
                    and item_local[current_cluster].size != 0
                ):
                    user_indices = cluster_global_indices[current_cluster]
                    local_pos = np.where(user_indices == u)[0]
                    if local_pos.size > 0:
                        local_u = int(local_pos[0])

                        U_loc = user_local[current_cluster]
                        Sigma_loc = sigma_local_matrices[current_cluster]
                        Vt_loc = item_local[current_cluster]

                        p_u_global = user_global[u, :]
                        p_u_local = U_loc[local_u, :]

                        pred_global = p_u_global.dot(sigma_global_matrix).dot(
                            item_global
                        )
                        pred_local = p_u_local.dot(Sigma_loc).dot(Vt_loc)

                        diff_pred = pred_global - pred_local
                        num = float(np.sum(diff_pred * (r_u - pred_local)))
                        den = float(np.sum(diff_pred**2))

                        g_u_opt = (
                            gu_vector[u]
                            if den < eps
                            else float(np.clip(num / den, 0.0, 1.0))
                        )
                        final_pred = (
                            g_u_opt * pred_global + (1.0 - g_u_opt) * pred_local
                        )
                        min_error = float(np.sum((r_u - final_pred) ** 2))
                        best_gu = g_u_opt

                # Evaluate other clusters as candidates
                for c_test in range(num_clusters):
                    if c_test == current_cluster:
                        continue
                    if (
                        item_local[c_test].size == 0
                        or sigma_local_matrices[c_test].size == 0
                    ):
                        continue

                    Vt_loc = item_local[c_test]
                    Sigma_loc = sigma_local_matrices[c_test]

                    # stable inverse
                    sigma_diag = np.diag(Sigma_loc)
                    inv_sigma = np.diag(
                        1.0 / np.where(sigma_diag > eps, sigma_diag, eps)
                    )

                    # Eq. 4: estimate p_u_local for candidate cluster
                    p_u_local_test = r_u.dot(Vt_loc.T.dot(inv_sigma))

                    # Predictions (Eq. 3)
                    p_u_global = user_global[u, :]
                    pred_global = p_u_global.dot(sigma_global_matrix).dot(item_global)
                    pred_local = p_u_local_test.dot(Sigma_loc).dot(Vt_loc)

                    diff_pred = pred_global - pred_local
                    num = float(np.sum(diff_pred * (r_u - pred_local)))
                    den = float(np.sum(diff_pred**2))
                    g_u_test = 0.5 if den < eps else float(np.clip(num / den, 0.0, 1.0))

                    final_pred = g_u_test * pred_global + (1.0 - g_u_test) * pred_local
                    error = float(np.sum((r_u - final_pred) ** 2))

                    if error < min_error * (1.0 - min_error_improvement):
                        min_error = error
                        best_cluster = c_test
                        best_gu = g_u_test

                if best_cluster != current_cluster:
                    users_switched += 1
                clusters_new[u] = best_cluster
                gu_new[u] = best_gu
                total_error += min_error

            # Convergence and logging
            num_changed_users = users_switched
            improvement = (
                (prev_total_error - total_error) / prev_total_error
                if prev_total_error != np.inf
                else 1.0
            )

            logger.info(
                f"Iter {iterations_count}: Switch={users_switched} "
                f"({users_switched / num_users:.1%}), "
                f"Error={total_error:.6f}, Improvement={improvement:.6f}"
            )

            prev_total_error = total_error
            gu_vector = gu_new
            clusters = clusters_new

        # --- Final recomputation of local factors for the final clusters ---
        for cluster_idx in range(num_clusters):
            user_indices = np.where(clusters == cluster_idx)[0]
            cluster_global_indices[cluster_idx] = user_indices

            if user_indices.size == 0:
                user_local[cluster_idx] = np.empty((0, 0))
                sigma_local_matrices[cluster_idx] = np.empty((0, 0))
                item_local[cluster_idx] = np.empty((0, 0))
                continue

            local_gu = 1.0 - gu_vector[user_indices]
            R_local = sp.csr_matrix(
                local_gu[:, np.newaxis]
                * bin_utility_matrix[user_indices, :]  # NOTE: see typo remark below
            )
            min_dim = min(R_local.shape)
            if min_dim <= 1:
                user_local[cluster_idx] = np.empty((0, 0))
                sigma_local_matrices[cluster_idx] = np.empty((0, 0))
                item_local[cluster_idx] = np.empty((0, 0))
                continue

            k_c = min(f_c, min_dim - 1)
            try:
                U_loc, sigma_loc_diag, Vt_loc = svds(R_local, k=k_c)
                user_local[cluster_idx] = U_loc
                sigma_local_matrices[cluster_idx] = np.diag(sigma_loc_diag)
                item_local[cluster_idx] = Vt_loc
            except Exception:
                user_local[cluster_idx] = np.empty((0, 0))
                sigma_local_matrices[cluster_idx] = np.empty((0, 0))
                item_local[cluster_idx] = np.empty((0, 0))

        # Save state in the model
        self.trained_factors = (
            gu_vector,
            user_global,
            sigma_global_matrix,
            item_global,
            user_local,
            sigma_local_matrices,
            item_local,
            cluster_global_indices,
        )
        self.clusters = clusters
        self.is_fitted = True
        self.num_iterations = iterations_count

        return self.trained_factors

    def evaluate_metrics(
        self,
        test_items: dict[int, int],
        train_matrix: np.ndarray,
        N: int,
    ) -> tuple[float, float]:
        """Compute HR@N and ARHR under LOOCV using pre-trained latent factors.

        Each user has exactly one **held-out** item *i* in ``test_items``
        (the single item removed from training during LOOCV).
        We score all items for each user, rank only the **candidate** items
        (those with value equal to 0 in ``train_matrix``), and measure:
        * **HR@N**: 1 if *i* is in the top-N, else 0 (averaged over evaluated users).
        * **ARHR**: Reciprocal rank of *i* if it appears in the top list, else 0
            (averaged over evaluated users).

        Args:
            test_items (dict[int, int]): Mapping ``{user_index: held_out_item_index}``.
            train_matrix (np.ndarray): Training user-item **binary** matrix (user x I) with the
                held-out item set to 0 for each evaluated user.
            N (int): Number of top recommendations to consider when computing HR and ARHR.

        Returns:
            out (tuple[float, float]):
                - ``hr``: Hit-Rate@N in [0, 1].
                - ``arhr``: Average Reciprocal Hit-Rank in [0, 1].
        """
        if not self.is_fitted or self.trained_factors is None:
            raise RuntimeError(
                "Model must be fitted before evaluation. Call fit() first."
            )

        # Unpack from self.trained_factors
        (
            _,  # gu_vector
            user_global,
            sigma_global_matrix,
            item_global,
            user_local,
            sigma_local_matrices,
            item_local,
            cluster_global_indices,
        ) = self.trained_factors

        users_to_evaluate = list(test_items.keys())

        # Check if users_to_evaluate is empty or None
        if not users_to_evaluate:
            return 0.0, 0.0

        hit_count = 0
        reciprocal_ranks: list[float] = []

        for user in users_to_evaluate:
            test_item = test_items[user]
            cluster_id = int(self.clusters[user])

            # 1) Compute prediction scores for all items for user u (shape: (I,))
            all_item_scores = self._predict_scores(
                user,
                cluster_id,
                user_global,
                sigma_global_matrix,
                item_global,
                user_local,
                sigma_local_matrices,
                item_local,
                cluster_global_indices,
            )

            # 2) Define recommendation candidates = items not seen in training
            unrated_indices = np.where(train_matrix[user, :] == 0)[0]

            # 3) Rank candidates by descending score
            candidate_scores = all_item_scores[unrated_indices]
            ranked_indices_in_candidates = np.argsort(candidate_scores)[::-1]

            # 4) Find the rank of the held-out item within candidates
            test_item_idx_array = np.where(unrated_indices == test_item)[0]
            if len(test_item_idx_array) == 0:
                # Held-out item not in the candidate set; skip this user
                continue

            # We flatten the obtained test_item_idx_array
            test_item_idx = int(test_item_idx_array[0])
            rank = (
                int(np.where(ranked_indices_in_candidates == test_item_idx)[0][0]) + 1
            )  # 1-based

            # 5) Update metrics
            if rank <= N:
                hit_count += 1
                # The reciprocal rank assigns higher weight to top positions.
                # Example: rank = 1 → 1/1 = 1.0, rank = 10 → 1/10 = 0.1
                reciprocal_ranks.append(1.0 / rank)

        # Final HR and ARHR
        num_eval = len(users_to_evaluate)
        hr = hit_count / num_eval
        arhr = (float(np.sum(reciprocal_ranks)) / num_eval) if num_eval > 0 else 0.0

        return hr, arhr

    @staticmethod
    def _predict_scores(
        u_idx: int,
        cluster_id: int,
        user_global: np.ndarray,
        sigma_global_matrix: np.ndarray,
        item_global: np.ndarray,
        user_local: list[np.ndarray],
        sigma_local_matrices: list[np.ndarray],
        item_local: list[np.ndarray],
        cluster_global_indices: list[np.ndarray],
    ) -> np.ndarray:
        """Compute predicted scores for a single user across all items
        (Eq. 5 from the original rGLSVD formulation).

        Used internally by ``evaluate_metrics`` to obtain user-item relevance scores
        combining global and local latent components.

        Args:
            u_idx (int): Global user index.
            cluster_id (int): ID of the cluster to which the user belongs.
            user_global (np.ndarray): Global user latent matrix (P).
            sigma_global_matrix (np.ndarray): Global singular values (Σ_g).
            item_global (np.ndarray): Global item latent matrix (Q).
            user_local (list[np.ndarray]): List of local user latent matrices (P^c).
            sigma_local_matrices (list[np.ndarray]): List of local singular value matrices (Σ_c).
            item_local (list[np.ndarray]): List of local item latent matrices (Q^c).
            cluster_global_indices (list[np.ndarray]): Mapping of cluster IDs to user indices.

        Returns:
            out (np.ndarray):
                Array of predicted scores for all items for the given user.
        """
        # Global component (p_u * Sigma_fg * Q^T)
        p_u = user_global[u_idx, :]
        global_scores = p_u.dot(sigma_global_matrix).dot(item_global)

        # Local component (p_u^c * Sigma_fc * Q^cT)
        global_indices = cluster_global_indices[cluster_id]
        local_u_index = np.where(global_indices == u_idx)[0][0]

        U_loc = user_local[cluster_id]
        p_u_c = U_loc[local_u_index, :]

        sigma_loc = sigma_local_matrices[cluster_id]
        Vt_loc = item_local[cluster_id]
        local_scores = p_u_c.dot(sigma_loc).dot(Vt_loc)

        total_score: np.ndarray = global_scores + local_scores

        return total_score
