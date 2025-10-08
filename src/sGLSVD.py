import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from typing import List, Tuple, Optional, cast

LocalFactors = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]

def sGLSVD(
    initial_clusters: np.ndarray,
    bin_utiliy_matrix: np.ndarray,
    num_clust: int,
    f_g: int,
    f_c: int,
    min_error_improvement: float = 0.01  # Minimo miglioramento relativo richiesto
) -> Tuple[
    Tuple[
        np.ndarray,               # gu_vector
        np.ndarray,               # user_global
        np.ndarray,               # sigma_global_matrix
        np.ndarray,               # item_global
        List[Optional[np.ndarray]],  # user_local_list (some clusters may be None)
        List[Optional[np.ndarray]],  # sigma_local_list
        List[Optional[np.ndarray]],  # item_local_list
        List[Optional[np.ndarray]]   # cluster_global_indices
    ],
    np.ndarray
]:
    
    num_users, _ = bin_utiliy_matrix.shape
    gu_vector = np.full(num_users, 0.5)
    clusters = initial_clusters.copy()
    eps = 1e-12
    # Lista di indici per cluster; ogni elemento può essere un np.ndarray o None
    cluster_global_indices: List[Optional[np.ndarray]] = [None] * num_clust
    
    iteration = 0
    prev_total_error = np.inf
    num_users_switching = num_users  # Per avviare il loop
    
    # Preallocazioni per i fattori locali (possono rimanere None per alcuni cluster)
    user_local: List[Optional[np.ndarray]] = [None] * num_clust
    sigma_local_matrices: List[Optional[np.ndarray]] = [None] * num_clust
    sigma_local: List[Optional[np.ndarray]] = [None] * num_clust
    item_local: List[Optional[np.ndarray]] = [None] * num_clust#ripetizione inutile

    while num_users_switching / num_users > 0.01:
        iteration += 1

        # Global Factor
        R_global = gu_vector[:, np.newaxis] * bin_utiliy_matrix
        R_global_sparse = sp.csr_matrix(R_global)
        user_global, sigma_global, item_global = svds(R_global_sparse, k=f_g)
        sigma_global_matrix = np.diag(sigma_global)

        # Local Factors: (ricreo localmente le liste perché possono essere ricalcolate ad ogni iter)
        user_local = [None] * num_clust
        sigma_local_matrices = [None] * num_clust
        sigma_local = [None] * num_clust
        item_local = [None] * num_clust
        
        for i in range(num_clust):
            user_indices_in_cluster = np.where(clusters == i)[0]
            cluster_global_indices[i] = user_indices_in_cluster  # Global index
            
            if len(user_indices_in_cluster) == 0: #senso?
                continue
                
            local_gu_vector = 1.0 - gu_vector[user_indices_in_cluster]
            bin_cluster_utility_matrix = bin_utiliy_matrix[user_indices_in_cluster, :]
            R_local_sparse = sp.csr_matrix(local_gu_vector[:, np.newaxis] * bin_cluster_utility_matrix)
            
            U_loc, sigma_loc_diag, Vt_loc = svds(R_local_sparse, k=f_c)
            user_local[i] = U_loc
            item_local[i] = Vt_loc
            sigma_local[i] = sigma_loc_diag
            sigma_local_matrices[i] = np.diag(sigma_loc_diag)
        
        # Reassignment of clusters
        gu_new = gu_vector.copy()
        clusters_new = clusters.copy()
        users_switched_count = 0
        total_error = 0.0
        
        for u in range(num_users):
            r_u_vector = bin_utiliy_matrix[u, :]
            current_cluster_id = int(clusters[u])
            
            # Calcola errore per cluster corrente
            current_error = np.inf
            # prendiamo il valore temporaneo per permettere al type checker di inferire None-check
            local_indices_opt = cluster_global_indices[current_cluster_id]
            if local_indices_opt is not None:
                local_indices = local_indices_opt  # non sono certo sia necessario questo controllo
                # verifica membership
                if u in local_indices:
                    # trova posizione locale dell'utente
                    local_u_pos = int(np.where(local_indices == u)[0][0])
                    # verifico che i fattori locali esistano per questo cluster
                    if user_local[current_cluster_id] is not None and \
                       sigma_local_matrices[current_cluster_id] is not None and \
                       item_local[current_cluster_id] is not None:
                        
                        # cast per tranquillizzare mypy
                        U_loc_cluster = cast(np.ndarray, user_local[current_cluster_id])
                        Sigma_loc_cluster = cast(np.ndarray, sigma_local_matrices[current_cluster_id])
                        Vt_loc_cluster = cast(np.ndarray, item_local[current_cluster_id])

                        p_u_c = U_loc_cluster[local_u_pos, :]
                        p_user_global = user_global[u, :]
                        
                        pred_global = p_user_global.dot(sigma_global_matrix).dot(item_global)
                        pred_local = p_u_c.dot(Sigma_loc_cluster).dot(Vt_loc_cluster)
                        
                        # Ottimizza g_u per cluster corrente
                        diff_pred = pred_global - pred_local
                        numerator = float(np.sum(diff_pred * (r_u_vector - pred_local)))
                        denominator = float(np.sum(diff_pred**2))
                        
                        if denominator < eps:
                            g_u_current = gu_vector[u]
                        else:
                            g_u_current = float(np.clip(numerator / denominator, 0.0, 1.0))
                        
                        final_pred = g_u_current * pred_global + (1.0 - g_u_current) * pred_local
                        current_error = float(np.sum((r_u_vector - final_pred)**2))
            
            # Cerca cluster migliore
            min_error = current_error
            best_cluster_id = current_cluster_id
            best_gu = gu_vector[u] if current_error != np.inf else 0.5
            
            for c_test in range(num_clust):
                if c_test == current_cluster_id:
                    continue
                        
                cluster_idx_opt = cluster_global_indices[c_test]
                if cluster_idx_opt is None:
                    continue
                if len(cluster_idx_opt) == 0:
                    continue
                
                # Assicuriamoci anche che i fattori locali per c_test esistano
                if item_local[c_test] is None or sigma_local[c_test] is None or sigma_local_matrices[c_test] is None:
                    continue
                
                Vt_loc_test = cast(np.ndarray, item_local[c_test])
                sigma_diag_test = cast(np.ndarray, sigma_local[c_test])
                Sigma_loc_test = cast(np.ndarray, sigma_local_matrices[c_test])
                
                # Per numeri piccoli, evitiamo divisioni per zero
                inv_sigma_diag_test = np.diag(1.0 / np.where(sigma_diag_test > eps, sigma_diag_test, eps))
                projection_matrix = Vt_loc_test.T.dot(inv_sigma_diag_test)
                p_u_c_test = r_u_vector.dot(projection_matrix)
                
                # Calcola predizioni
                p_user_global = user_global[u, :]
                pred_global = p_user_global.dot(sigma_global_matrix).dot(item_global)
                pred_local = p_u_c_test.dot(Sigma_loc_test).dot(Vt_loc_test)
                
                # Ottimizza g_u
                diff_pred = pred_global - pred_local
                numerator = float(np.sum(diff_pred * (r_u_vector - pred_local)))
                denominator = float(np.sum(diff_pred**2))
                
                if denominator < eps:
                    g_u_test = 0.5
                else:
                    g_u_test = float(np.clip(numerator / denominator, 0.0, 1.0))
                
                # Calcola errore
                final_pred = g_u_test * pred_global + (1.0 - g_u_test) * pred_local
                error = float(np.sum((r_u_vector - final_pred)**2))
                
                # ISTERESI: richiedi miglioramento minimo per switchare
                improvement_threshold = min_error * (1.0 - min_error_improvement)
                
                if error < improvement_threshold:
                    min_error = error
                    best_cluster_id = c_test
                    best_gu = g_u_test
            
            # Applica assignment
            if best_cluster_id != current_cluster_id:
                users_switched_count += 1
            
            clusters_new[u] = best_cluster_id
            gu_new[u] = best_gu
            total_error += min_error
        
        # ===== FASE 3: Check Convergenza =====
        num_users_switching = users_switched_count
        switching_ratio = num_users_switching / num_users
        
        # Calcola miglioramento dell'errore totale
        if prev_total_error != np.inf:
            error_improvement = (prev_total_error - total_error) / prev_total_error
        else:
            error_improvement = 1.0
        
        print(f"Iter {iteration}: Switch={num_users_switching} ({switching_ratio:.1%}), "
              f"Error={total_error:.6f}, Improvement={error_improvement:.6f}")
        
        # Aggiorna stato
        prev_total_error = total_error
        gu_vector = gu_new
        clusters = clusters_new
    
    # ===== RICALCOLA FATTORI LOCALI con i cluster finali =====
    for i in range(num_clust):
        user_indices_in_cluster = np.where(clusters == i)[0]
        cluster_global_indices[i] = user_indices_in_cluster
        
        if len(user_indices_in_cluster) == 0:
            continue
            
        local_gu_vector = 1.0 - gu_vector[user_indices_in_cluster]
        bin_cluster_utility_matrix = bin_utiliy_matrix[user_indices_in_cluster, :]
        R_local_sparse = sp.csr_matrix(local_gu_vector[:, np.newaxis] * bin_cluster_utility_matrix)
        
        U_loc, sigma_loc_diag, Vt_loc = svds(R_local_sparse, k=f_c)
        user_local[i] = U_loc
        item_local[i] = Vt_loc
        sigma_local[i] = sigma_loc_diag
        sigma_local_matrices[i] = np.diag(sigma_loc_diag)
    
    #Output
    user_local_list: List[Optional[np.ndarray]] = []
    sigma_local_list: List[Optional[np.ndarray]] = []
    item_local_list: List[Optional[np.ndarray]] = []
    
    for i in range(num_clust):
        user_local_list.append(user_local[i])
        sigma_local_list.append(sigma_local_matrices[i])
        item_local_list.append(item_local[i])

    return (
        (gu_vector, user_global, sigma_global_matrix, item_global, 
         user_local_list, sigma_local_list, item_local_list, cluster_global_indices), 
        clusters
    )

