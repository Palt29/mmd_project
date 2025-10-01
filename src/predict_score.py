import numpy as np

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
