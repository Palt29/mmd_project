import numpy as np

def loocv_split(bin_utility_matrix: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """
    Esegue lo split Leave-One-Out (LOOCV) del dataset per la valutazione Top-N.
    Per ogni utente, seleziona un item valutato e lo sposta nel test set,
    impostando il suo valore a 0 nella matrice di training.

    Args:
        bin_utility_matrix (np.ndarray): La matrice binaria completa (U x I).

    Returns:
        tuple[np.ndarray, dict[int, int]]: 
            - Matrice di training (train_matrix): Matrice con l'item di test rimosso (posto a 0).
            - Item di test (test_items): Mappa {indice utente: indice item di test}.
    """
    num_users, _ = bin_utility_matrix.shape
    test_items = {}
    train_matrix = bin_utility_matrix.copy()
    
    # [cite_start]"Selezioniamo casualmente un item valutato da ogni utente, e lo poniamo nel test set" [cite: 186]
    for u in range(num_users):
        # Items con interazione (valore 1)
        rated_items = np.where(bin_utility_matrix[u, :] == 1)[0]
        
        if len(rated_items) > 0:
            # Scegli casualmente l'item valutato da lasciare fuori (test set)
            test_item = np.random.choice(rated_items, 1)[0]
            test_items[u] = test_item
            
            # [cite_start]Rimuovi l'item di test dalla matrice di training [cite: 187]
            train_matrix[u, test_item] = 0
            
    # [cite_start]La train_matrix (resto dei dati) comprende il training set [cite: 187]
    return train_matrix, test_items