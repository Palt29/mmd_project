import numpy as np


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
    """Compute predicted scores for a single user across all items (Eq. 5).

    Used internally by ``evaluate_metrics`` to obtain user-item relevance scores
    combining global and local latent components.

    Args:
        u_idx: Global user index.
        cluster_id: ID of the cluster to which the user belongs.
        user_global: Global user latent matrix (P).
        sigma_global_matrix: Global singular values (Σ_g).
        item_global: Global item latent matrix (Q).
        user_local: List of local user latent matrices (P^c).
        sigma_local_matrices: List of local singular value matrices (Σ_c).
        item_local: List of local item latent matrices (Q^c).
        cluster_global_indices: Mapping of cluster IDs to user indices.

    Returns:
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


def evaluate_metrics(
    trained_factors: tuple[
        np.ndarray,  # gu_vector
        np.ndarray,  # user_global (P)
        np.ndarray,  # sigma_global_matrix (Σ_g)
        np.ndarray,  # item_global (Q)
        list[np.ndarray],  # user_local (P^c)
        list[np.ndarray],  # sigma_local_matrices (Σ_c)
        list[np.ndarray],  # item_local (Q^c)
        list[np.ndarray],  # cluster_global_indices
    ],
    test_items: dict[int, int],
    train_matrix: np.ndarray,
    clusters: np.ndarray,
    N: int,
) -> tuple[float, float]:
    """Compute HR@N and ARHR under LOOCV using pre-trained latent factors.

    Each user *user* has exactly one **held-out** item *i* in ``test_items``
    (the single item removed from training during LOOCV).
    We score all items for *user*, rank only the **candidate** items
    (those with value equal to 0 in ``train_matrix``), and measure:
      * **HR@N**: 1 if *i* is in the top-N, else 0 (averaged over evaluated users).
      * **ARHR**: Reciprocal rank of *i* if it appears in the top list, else 0
        (averaged over evaluated users).

    Args:
      trained_factors: Tuple returned by ``rGLSVD``:
        (gu_vector, P, Σ_g, Q, {P^c}, {Σ_c}, {Q^c}, cluster_global_indices).
      test_items: Mapping ``{user_index: held_out_item_index}``.
      train_matrix: Training user-item **binary** matrix (user x I) with the
        held-out item set to 0 for each evaluated user.
      clusters: Cluster assignments for each users.
      N: Number of top recommendations to consider when computing HR and ARHR.

    Returns:
      Tuple ``(hr, arhr)``:
        - ``hr``: Hit-Rate@N in [0, 1].
        - ``arhr``: Average Reciprocal Hit-Rank in [0, 1].
    """
    # Unpack rGLSVD outputs (trained_factors) into explicit names
    (
        _,  # gu_vector
        user_global,
        sigma_global_matrix,
        item_global,
        user_local,
        sigma_local_matrices,
        item_local,
        cluster_global_indices,
    ) = trained_factors

    users_to_evaluate = list(test_items.keys())

    # Check if users_to_evaluate is empty or None
    if not users_to_evaluate:
        return 0.0, 0.0

    hit_count = 0
    reciprocal_ranks: list[float] = []

    for user in users_to_evaluate:
        test_item = test_items[user]
        cluster_id = int(clusters[user])

        # 1) Compute prediction scores for all items for user u (shape: (I,))
        all_item_scores = _predict_scores(
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
