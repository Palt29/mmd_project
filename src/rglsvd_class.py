import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class RGLSVDRecommender:
    """rGLSVD Recommender System."""

    def __init__(
        self,
        R_bin: np.ndarray,
        clusters: np.ndarray,
        num_clusters: int,
        f_g: int,
        f_c: dict[int, int],
    ) -> None:
        """Initialize rGLSVD configuration and data.

        Args:
            R_bin (np.ndarray): Binary utility matrix (num_users x num_items).
            clusters (np.ndarray): 1D array assigning each user to a cluster id in [0, num_clusters).
            num_clusters (int): Total number of user clusters.
            f_g (int): Global SVD rank.
            f_c (dict[int, int]): Mapping {cluster_id: local SVD rank}.
        """
        self.R_bin = R_bin
        self.clusters = clusters
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
                test_items[user_idx] = test_item

                # Set the held-out test item to 0 in the training matrix
                train_matrix[user_idx, test_item] = 0

        return train_matrix, test_items

    def fit(
        self,
        convergence_threshold: float = 0.01,
        weight_change_threshold: float = 0.01,
        max_iterations: int | None = None,
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
        Run rGLSVD algorithm on the binary utility matrix.

        Uses the model configuration defined at initialization (clusters, global and
        local ranks, convergence threshold).
        Iteratively alternates between global and local truncated SVDs,
        and updates the user-specific weights *g_u* according to Eq. (3) from the
        original rGLSVD formulation. The process continues until convergence.

        Args:
            convergence_threshold (float): Algorithm stops when the fraction of changed users
                falls below this value. Default: 0.01 (1%).
            weight_change_threshold (float): Minimum absolute change in *g_u* to consider a user
                as "changed". Default: 0.01.
            max_iterations (int | None): Maximum number of iterations before forcing stop.
                If None, runs until convergence. Default: None.

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
        clusters = self.clusters
        num_clusters = self.num_clusters
        f_g = self.f_g
        f_c = self.f_c

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
        iterations_count = 0
        # We assign num_change_users = num_users just to start the while loop
        num_changed_users = num_users

        # Build convergence condition
        def should_continue() -> bool:
            if max_iterations is None:
                return bool(num_changed_users / num_users > convergence_threshold)
            return bool(
                num_changed_users / num_users > convergence_threshold
                and iterations_count < max_iterations
            )

        while should_continue():
            iterations_count += 1
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
                bin_cluster_utility_matrix = bin_utility_matrix[
                    user_indices_in_cluster, :
                ]
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
            num_changed_users = int(np.sum(diff_vector > weight_change_threshold))

            gu_vector = gu_new

        # We assign the trained factors
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
        self.is_fitted = True
        self.num_iterations = iterations_count

        return (
            gu_vector,
            user_global,
            sigma_global_matrix,
            item_global,
            user_local,
            sigma_local_matrices,
            item_local,
            cluster_global_indices,
        )

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
