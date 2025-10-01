import numpy as np
from typing import Any

def predict_scores(
    u_idx: int, 
    cluster_id: int,
    user_global: np.ndarray, 
    sigma_global_matrix: np.ndarray, 
    item_global: np.ndarray, 
    user_local: list[np.ndarray], 
    sigma_local_matrices: list[np.ndarray], 
    item_local: list[np.ndarray],
    cluster_global_indices: list[np.ndarray]
) -> np.ndarray:
    """
    (Equation 5) prediction of a single user on all items.
    """
    
    # Global component (p_u * Sigma_fg * Q^T) ---
    p_u = user_global[u_idx, :] # (1 x f_g)
    global_scores = p_u.dot(sigma_global_matrix).dot(item_global) # (1 x m)
    
    # Local component (p_u^c * Sigma_fc * Q^cT) ---
    
    # Index in U_loc
    global_indices = cluster_global_indices[cluster_id]
    local_u_index = np.where(global_indices == u_idx)[0][0]
    
    U_loc = user_local[cluster_id]
    p_u_c = U_loc[local_u_index, :] # (1 x f_c)
    
    sigma_loc = sigma_local_matrices[cluster_id]
    Vt_loc = item_local[cluster_id]
    local_scores = p_u_c.dot(sigma_loc).dot(Vt_loc) # (1 x m)
    
    return global_scores + local_scores

def evaluate_metrics_loocv(
    trained_factors: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    test_items: dict[int, int],
    train_matrix: np.ndarray,
    clusters: np.ndarray,
    N_recommend: int,
) -> tuple[float, float]:
    """
    Calcola Hit-Rate (HR) e ARHR utilizzando i fattori latenti addestrati 
    e i dati di test LOOCV.

    Args:
        trained_factors (tuple): Output completo della funzione rGLSVD.
        test_items (dict): Mappa {indice utente: indice item di test (left-out)}.
        train_matrix (np.ndarray): Matrice di training (item di test a 0).
        clusters (np.ndarray): Assegnazioni dei cluster per ogni utente.
        N_recommend (int): Dimensione della lista Top-N.

    Returns:
        tuple[float, float]: Hit-Rate (HR) e Average Reciprocal Hit Rank (ARHR).
    """
    # Decomponi i fattori addestrati per l'uso
    (gu_vector, user_global, sigma_global_matrix, item_global, 
     user_local, sigma_local_matrices, item_local, cluster_global_indices) = trained_factors
     
    users_to_evaluate = list(test_items.keys())
    if not users_to_evaluate:
        return 0.0, 0.0

    hit_count = 0
    reciprocal_ranks = []
    
    for u in users_to_evaluate:
        test_item = test_items[u]
        cluster_id = int(clusters[u])
        # 1. Calcola i punteggi di predizione per tutti gli item (1 x m)
        all_item_scores = predict_scores(
            u, cluster_id, user_global, sigma_global_matrix, item_global,
            user_local, sigma_local_matrices, item_local, cluster_global_indices
        )
        
        # 2. Definisci i candidati per la raccomandazione
        # I candidati sono tutti gli item non interagiti nel training set (valore 0)
        unrated_indices = np.where(train_matrix[u, :] == 0)[0]
        
        # 3. Classifica i candidati
        candidate_scores = all_item_scores[unrated_indices]
        
        # Ottieni gli indici degli item candidati ordinati per punteggio
        ranked_indices_in_candidates = np.argsort(candidate_scores)[::-1]
        
        # 4. Trova il Rank del test_item (Ground Truth)
        
        # Posizione del test_item nell'array non ordinato 'unrated_indices'
        test_item_pos_in_candidates = np.where(unrated_indices == test_item)[0][0]
        
        # Rank: la posizione nell'array ordinato, +1 perché il rank parte da 1
        rank = np.where(ranked_indices_in_candidates == test_item_pos_in_candidates)[0][0] + 1
        
        # 5. Aggiorna le Metriche
        
        if rank <= N_recommend:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
            
    # Calcolo finale di HR e ARHR
    hr = hit_count / len(users_to_evaluate)
    arhr = np.sum(reciprocal_ranks) / len(users_to_evaluate)
    
    return hr, arhr