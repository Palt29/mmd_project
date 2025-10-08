import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from typing import Optional, Tuple, List, Dict


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
        self.f_c = f_c  # Local rank is a single integer

        # Track model state
        self.is_fitted: bool = False
        self.trained_factors: Optional[
            Tuple[
                np.ndarray,  # gu_vector
                np.ndarray,  # user_global (P)
                np.ndarray,  # sigma_global_matrix (Σ_g)
                np.ndarray,  # item_global (Q)
                List[Optional[np.ndarray]],  # user_local (P^c)
                List[Optional[np.ndarray]],  # sigma_local_matrices (Σ_c)
                List[Optional[np.ndarray]],  # item_local (Q^c)
                List[np.ndarray],  # cluster_global_indices
            ]
        ] = None
        self.num_iterations: int = 0  # Track convergence iterations

        # Constants
        self._eps: float = 1e-12

    def loocv_split(self) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Performs Leave-One-Out (LOOCV) splitting for Top-N evaluation.

        Randomly selects one interacted item (entry == 1) per user to hold out for testing,
        setting its entry to 0 in the training matrix.

        Returns:
            out (Tuple[np.ndarray, Dict[int, int]]):
                - train_matrix: Copy of self.R_bin with one held-out item per user set to 0.
                - test_items: Mapping {user_index: held_out_item_index}.
        """
        bin_utility_matrix = self.R_bin
        num_users, _ = bin_utility_matrix.shape
        test_items: Dict[int, int] = {}
        train_matrix = bin_utility_matrix.copy()

        for user_idx in range(num_users):
            rated_items = np.where(bin_utility_matrix[user_idx, :] == 1)[0]

            if len(rated_items) > 0:
                # Randomly choose one rated item to leave out (test set)
                test_item = np.random.choice(rated_items, 1)[0]
                test_items[user_idx] = int(test_item)

                # Set the held-out test item to 0 in the training matrix
                train_matrix[user_idx, test_item] = 0

        return train_matrix, test_items

    @staticmethod
    def _predict_scores(
        u_idx: int,
        cluster_id: int,
        gu_vector: np.ndarray,
        user_global: np.ndarray,
        sigma_global_matrix: np.ndarray,
        item_global: np.ndarray,
        user_local: List[Optional[np.ndarray]],
        sigma_local_matrices: List[Optional[np.ndarray]],
        item_local: List[Optional[np.ndarray]],
        cluster_global_indices: List[np.ndarray],
    ) -> np.ndarray:
        """Compute predicted scores for a single user across all items
        (Eq. 5 from the original rGLSVD formulation).

        Combines global and local latent components using the user-specific weight g_u.
        
        Args:
            u_idx (int): Global user index.
            cluster_id (int): ID of the cluster to which the user belongs.
            gu_vector (np.ndarray): Final user global weight vector (g_u).
            user_global (np.ndarray): Global user latent matrix (P).
            sigma_global_matrix (np.ndarray): Global singular values (Σ_g).
            item_global (np.ndarray): Global item latent matrix (Q).
            user_local (list[Optional[np.ndarray]]): List of local user latent matrices (P^c).
            sigma_local_matrices (list[Optional[np.ndarray]]): List of local singular value matrices (Σ_c).
            item_local (list[Optional[np.ndarray]]): List of local item latent matrices (Q^c).
            cluster_global_indices (list[np.ndarray]): Mapping of cluster IDs to user indices.

        Returns:
            out (np.ndarray): Array of predicted scores for all items for the given user.
        """
        g_u = gu_vector[u_idx]

        # 1. Global component (p_u * Sigma_g * Q^T)
        p_u = user_global[u_idx, :]
        global_scores = p_u.dot(sigma_global_matrix).dot(item_global)

        # 2. Local component (p_u^c * Sigma_c * Q^cT)
        U_loc = user_local[cluster_id]
        Sigma_loc = sigma_local_matrices[cluster_id]
        Vt_loc = item_local[cluster_id]
        
        if U_loc is None or Sigma_loc is None or Vt_loc is None:
            # If the cluster is empty or failed SVD, the local component is zero
            local_scores = np.zeros_like(global_scores)
        else:
            global_indices = cluster_global_indices[cluster_id]
            # Assumes the user is in the cluster (which they are by definition in fit)
            local_u_index = np.where(global_indices == u_idx)[0][0]
            p_u_c = U_loc[local_u_index, :]
            local_scores = p_u_c.dot(Sigma_loc).dot(Vt_loc)

        # 3. Combined score (Eq. 5)
        # Prediction(r_u) = g_u * Prediction_g + (1 - g_u) * Prediction_c
        total_score: np.ndarray = g_u * global_scores + (1.0 - g_u) * local_scores

        return total_score

    def evaluate_metrics(
        self,
        test_items: Dict[int, int],
        train_matrix: np.ndarray,
        N: int,
    ) -> Tuple[float, float]:
        """Compute HR@N and ARHR under LOOCV using pre-trained latent factors.

        Args:
            test_items (Dict[int, int]): Mapping ``{user_index: held_out_item_index}``.
            train_matrix (np.ndarray): Training user-item **binary** matrix with the
                held-out item set to 0 for each evaluated user.
            N (int): Number of top recommendations to consider.

        Returns:
            out (Tuple[float, float]):
                - ``hr``: Hit-Rate@N in [0, 1].
                - ``arhr``: Average Reciprocal Hit-Rank in [0, 1].
        """
        if not self.is_fitted or self.trained_factors is None:
            raise RuntimeError(
                "Model must be fitted before evaluation. Call fit() first."
            )

        # Unpack from self.trained_factors
        (
            gu_vector,
            user_global,
            sigma_global_matrix,
            item_global,
            user_local,
            sigma_local_matrices,
            item_local,
            cluster_global_indices,
        ) = self.trained_factors

        users_to_evaluate = list(test_items.keys())

        if not users_to_evaluate:
            return 0.0, 0.0

        hit_count = 0
        reciprocal_ranks: List[float] = []

        for user in users_to_evaluate:
            test_item = test_items[user]
            cluster_id = int(self.clusters[user])

            # 1) Compute prediction scores for all items
            all_item_scores = self._predict_scores(
                user,
                cluster_id,
                gu_vector,
                user_global,
                sigma_global_matrix,
                item_global,
                user_local,
                sigma_local_matrices,
                item_local,
                cluster_global_indices,
            )

            # 2) Define recommendation candidates = items not rated in training (value == 0)
            # The held-out item is guaranteed to be here.
            unrated_indices = np.where(train_matrix[user, :] == 0)[0]

            # 3) Map scores to candidates and rank by descending score
            candidate_scores = all_item_scores[unrated_indices]
            # Indices of candidates sorted by score (highest score first)
            ranked_indices_in_candidates = np.argsort(candidate_scores)[::-1]

            # 4) Find the rank of the held-out item within candidates
            # Find the position of test_item within the *unrated_indices* array
            test_item_pos_in_unrated = np.where(unrated_indices == test_item)[0]
            
            if len(test_item_pos_in_unrated) == 0:
                continue # Should not happen if LOOCV was performed correctly

            test_item_idx = int(test_item_pos_in_unrated[0])
            
            # Find the rank (1-based) of the item at test_item_idx in the ranked list
            rank = (
                int(np.where(ranked_indices_in_candidates == test_item_idx)[0][0]) + 1
            )

            # 5) Update metrics
            if rank <= N:
                hit_count += 1
                reciprocal_ranks.append(1.0 / rank)

        # Final HR and ARHR
        num_eval = len(users_to_evaluate)
        hr = hit_count / num_eval
        arhr = (float(np.sum(reciprocal_ranks)) / num_eval) if num_eval > 0 else 0.0

        return hr, arhr

    # ==================================================================================
    # Il metodo 'fit' è stato fornito nell'output precedente, ma lo includo qui per
    # completezza, rimuovendo il suo corpo e lasciando solo la sua definizione
    # per evitare ridondanze. L'utente ha già la logica di 'fit'.
    # ==================================================================================
    def fit(
        self,
        min_error_improvement: float = 0.01,
        convergence_fraction_threshold: float = 0.005,
        max_iterations: Optional[int] = 20,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[np.ndarray]],
        List[Optional[np.ndarray]],
        List[Optional[np.ndarray]],
        List[np.ndarray],
    ]:
        """
        Esegue il training sGLSVD con l'algoritmo fornito:
        - SVD globale pesata da gu
        - SVD locali per ogni cluster pesata da (1-gu)
        - riassegnamento cluster + aggiornamento gu ottimale per ciascun utente
        """

        R = self.R_bin
        num_users, num_items = R.shape
        num_clusters = self.num_clusters
        f_g, f_c = self.f_g, self.f_c

        # inizializzazioni
        gu_vector = np.full(num_users, 0.5)
        clusters = self.clusters.copy()
        eps = self._eps
        iteration = 0
        prev_total_error = np.inf
        num_changed_users = num_users

        # strutture locali
        cluster_global_indices: List[np.ndarray] = [np.empty((0,), dtype=int) for _ in range(num_clusters)]
        user_local: List[Optional[np.ndarray]] = [None] * num_clusters
        sigma_local_matrices: List[Optional[np.ndarray]] = [None] * num_clusters
        item_local: List[Optional[np.ndarray]] = [None] * num_clusters

        # ciclo finché la frazione di utenti cambiati è > threshold (o fino a max_iterations)
        while (num_changed_users / num_users) > convergence_fraction_threshold:
            iteration += 1
            if max_iterations is not None and iteration > max_iterations:
                print(f"Stopping: reached max_iterations={max_iterations}")
                break

            # --- GLOBAL FACTORS (SVD pesata da gu) ---
            # costruisco la matrice pesata: ogni riga u moltiplicata per gu_u
            R_global = sp.csr_matrix(gu_vector[:, np.newaxis] * R)
            # controllo k per svds: deve essere < min(dim)
            k_g = min(f_g, min(R_global.shape) - 1) if min(R_global.shape) > 1 else 1
            try:
                user_global, sigma_global, item_global = svds(R_global, k=k_g)
            except Exception:
                # fallback: se svds fallisce (es. dati troppo piccoli), uso SVD piena via numpy (meno efficiente)
                U, s, Vt = np.linalg.svd(R_global.toarray(), full_matrices=False)
                user_global = U[:, :f_g]
                sigma_global = s[:f_g]
                item_global = Vt[:f_g, :]
            sigma_global_matrix = np.diag(sigma_global)

            # --- LOCAL FACTORS per cluster ---
            for cluster_idx in range(num_clusters):
                user_indices = np.where(clusters == cluster_idx)[0]
                cluster_global_indices[cluster_idx] = user_indices

                if user_indices.size == 0:
                    user_local[cluster_idx] = None
                    sigma_local_matrices[cluster_idx] = None
                    item_local[cluster_idx] = None
                    continue

                local_gu = 1.0 - gu_vector[user_indices]
                R_local = sp.csr_matrix(local_gu[:, np.newaxis] * R[user_indices, :])

                # k per svds locale: non superare min(shape)-1
                min_dim = min(R_local.shape)
                if min_dim <= 1:
                    # dati troppo ridotti -> saltare SVD locale
                    user_local[cluster_idx] = None
                    sigma_local_matrices[cluster_idx] = None
                    item_local[cluster_idx] = None
                    continue

                k_c = min(f_c, min_dim - 1)
                try:
                    U_loc, sigma_loc_diag, Vt_loc = svds(R_local, k=k_c)
                    user_local[cluster_idx] = U_loc
                    sigma_local_matrices[cluster_idx] = np.diag(sigma_loc_diag)
                    item_local[cluster_idx] = Vt_loc
                except Exception:
                    # fallback semplice: salta la SVD per il cluster se fallisce
                    user_local[cluster_idx] = None
                    sigma_local_matrices[cluster_idx] = None
                    item_local[cluster_idx] = None

            # --- CLUSTER REASSIGNMENT & aggiornamento g_u ---
            gu_new = gu_vector.copy()
            clusters_new = clusters.copy()
            total_error = 0.0
            users_switched = 0

            for u in range(num_users):
                r_u = R[u, :]
                current_cluster = int(clusters[u])

                min_error = np.inf
                best_cluster = current_cluster
                best_gu = gu_vector[u]

                # === Valuta cluster corrente (se disponibile) ===
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

                            pred_global = p_u_global.dot(sigma_global_matrix).dot(item_global)
                            pred_local = p_u_local.dot(Sigma_loc).dot(Vt_loc)

                            diff_pred = pred_global - pred_local
                            num = float(np.sum(diff_pred * (r_u - pred_local)))
                            den = float(np.sum(diff_pred ** 2))

                            g_u_opt = gu_vector[u] if den < eps else float(np.clip(num / den, 0.0, 1.0))
                            final_pred = g_u_opt * pred_global + (1.0 - g_u_opt) * pred_local
                            min_error = float(np.sum((r_u - final_pred) ** 2))
                            best_gu = g_u_opt

                # === Valuta altri cluster come candidati ===
                for c_test in range(num_clusters):
                    if c_test == current_cluster or user_local[c_test] is None:
                        continue

                    Vt_loc = item_local[c_test]
                    Sigma_loc = sigma_local_matrices[c_test]
                    if Vt_loc is None or Sigma_loc is None:
                        continue

                    sigma_diag = np.diag(Sigma_loc)
                    # stable inverse
                    inv_sigma = np.diag(1.0 / np.where(sigma_diag > eps, sigma_diag, eps))

                    # Equation 4: stima p_u_local per cluster candidato
                    # p_u_local_test shape (f_c,)
                    p_u_local_test = r_u.dot(Vt_loc.T.dot(inv_sigma))

                    # Equation 3: predizioni
                    p_u_global = user_global[u, :]
                    pred_global = p_u_global.dot(sigma_global_matrix).dot(item_global)
                    pred_local = p_u_local_test.dot(Sigma_loc).dot(Vt_loc)

                    diff_pred = pred_global - pred_local
                    num = float(np.sum(diff_pred * (r_u - pred_local)))
                    den = float(np.sum(diff_pred ** 2))
                    g_u_test = 0.5 if den < eps else float(np.clip(num / den, 0.0, 1.0))

                    final_pred = g_u_test * pred_global + (1.0 - g_u_test) * pred_local
                    error = float(np.sum((r_u - final_pred) ** 2))

                    # accetta solo se migliora in misura significativa
                    if error < min_error * (1.0 - min_error_improvement):
                        min_error = error
                        best_cluster = c_test
                        best_gu = g_u_test

                # aggiorna assegnazione/gu e contatori
                if best_cluster != current_cluster:
                    users_switched += 1
                clusters_new[u] = best_cluster
                gu_new[u] = best_gu
                total_error += min_error

            # convergenza / logging
            num_changed_users = users_switched
            (prev_total_error - total_error) / prev_total_error if prev_total_error != np.inf else 1.0

            """print(
                f"Iter {iteration}: Switch={users_switched} ({users_switched / num_users:.1%}), "
                f"Error={total_error:.6f}, Improvement={improvement:.6f}"
            )"""

            prev_total_error = total_error
            gu_vector = gu_new
            clusters = clusters_new

        # --- Ricalcolo fattori locali finali per i cluster finali ---
        for cluster_idx in range(num_clusters):
            user_indices = np.where(clusters == cluster_idx)[0]
            cluster_global_indices[cluster_idx] = user_indices
            if user_indices.size == 0:
                user_local[cluster_idx] = None
                sigma_local_matrices[cluster_idx] = None
                item_local[cluster_idx] = None
                continue

            local_gu = 1.0 - gu_vector[user_indices]
            R_local = sp.csr_matrix(local_gu[:, np.newaxis] * R[user_indices, :])
            min_dim = min(R_local.shape)
            if min_dim <= 1:
                user_local[cluster_idx] = None
                sigma_local_matrices[cluster_idx] = None
                item_local[cluster_idx] = None
                continue

            k_c = min(f_c, min_dim - 1)
            try:
                U_loc, sigma_loc_diag, Vt_loc = svds(R_local, k=k_c)
                user_local[cluster_idx] = U_loc
                sigma_local_matrices[cluster_idx] = np.diag(sigma_loc_diag)
                item_local[cluster_idx] = Vt_loc
            except Exception:
                user_local[cluster_idx] = None
                sigma_local_matrices[cluster_idx] = None
                item_local[cluster_idx] = None

        # salva stato nel modello
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
        self.num_iterations = iteration

        return self.trained_factors
