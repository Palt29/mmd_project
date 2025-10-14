import numpy as np
from typing import Literal


class CollaborativeFilteringRecommender:
    """Collaborative Filtering Recommender User-User."""

    def __init__(
        self,
        R_bin: np.ndarray,
        similarity: Literal["jaccard", "cosine"] = "jaccard",
        k_neighbors: int = 20
    ) -> None:
        self.R_bin = R_bin
        self.similarity_metric = similarity
        self.k_neighbors = k_neighbors
        self.num_users, self.num_items = R_bin.shape
        self.similarity_matrix: np.ndarray | None = None
        self.is_fitted: bool = False

    def jaccard_similarity(self, matrix: np.ndarray) -> np.ndarray:
        
        matrix_bool = matrix.astype(bool)
    
        intersection = matrix_bool @ matrix_bool.T
        
        cardinalities = matrix_bool.sum(axis=1, keepdims=True)
        union = cardinalities + cardinalities.T - intersection
       
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = np.where(union > 0, intersection / union, 0.0)
        
        return similarity.astype(float)

    def cosine_similarity(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = matrix / norms
        similarity = normalized @ normalized.T
        return similarity

    def fit(self, train_matrix: np.ndarray | None = None) -> np.ndarray:
    
        matrix_to_use = train_matrix if train_matrix is not None else self.R_bin
        
    
        if self.similarity_metric == "jaccard":
            self.similarity_matrix = self.jaccard_similarity(matrix_to_use)
        else:  
            self.similarity_matrix = self.cosine_similarity(matrix_to_use)
                
        self.is_fitted = True
        return self.similarity_matrix

    def predict_scores(self, user_idx: int) -> np.ndarray:
        if not self.is_fitted or self.similarity_matrix is None:
            raise RuntimeError("Call fit() before.")
        

        user_similarities = self.similarity_matrix[user_idx].copy()
        user_similarities[user_idx] = -1  
        
        top_k_indices = np.argsort(user_similarities)[-self.k_neighbors:]
        top_k_sims = user_similarities[top_k_indices]
        
        if np.sum(top_k_sims) > 0:
            weights = top_k_sims / np.sum(top_k_sims)
            scores = weights @ self.R_bin[top_k_indices]
        else:
            scores = np.zeros(self.num_items)
        
        return scores

    def loocv_split(self) -> tuple[np.ndarray, dict[int, int]]:
        num_users = self.num_users
        test_items: dict[int, int] = {}
        train_matrix = self.R_bin.copy()

        for user_idx in range(num_users):
            rated_items = np.where(self.R_bin[user_idx, :] == 1)[0]
            if len(rated_items) > 0:
                test_item = np.random.choice(rated_items, 1)[0]
                test_items[user_idx] = test_item
                train_matrix[user_idx, test_item] = 0

        return train_matrix, test_items

    def evaluate_metrics(
        self,
        test_items: dict[int, int],
        train_matrix: np.ndarray,
        N: int,
    ) -> tuple[float, float]:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before.")

        users_to_evaluate = list(test_items.keys())
        if not users_to_evaluate:
            return 0.0, 0.0

        hit_count = 0
        reciprocal_ranks: list[float] = []

        for user in users_to_evaluate:
            test_item = test_items[user]
            all_item_scores = self.predict_scores(user)
            unrated_indices = np.where(train_matrix[user, :] == 0)[0]
            candidate_scores = all_item_scores[unrated_indices]
            ranked_indices_in_candidates = np.argsort(candidate_scores)[::-1]

            test_item_idx_array = np.where(unrated_indices == test_item)[0]
            if len(test_item_idx_array) == 0:
                continue

            test_item_idx = int(test_item_idx_array[0])
            rank = int(np.where(ranked_indices_in_candidates == test_item_idx)[0][0]) + 1

            if rank <= N:
                hit_count += 1
                reciprocal_ranks.append(1.0 / rank)

        num_eval = len(users_to_evaluate)
        hr = hit_count / num_eval if num_eval > 0 else 0.0
        arhr = float(np.sum(reciprocal_ranks)) / num_eval if num_eval > 0 else 0.0

        return hr, arhr
