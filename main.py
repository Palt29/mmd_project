"""Run the full rGLSVD evaluation pipeline."""

import logging

import numpy as np

from src.rglsvd_class import RGLSVDRecommender


def main() -> None:
    # ------------ Logger setup ------------
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # ------------ Load data ------------
    R_bin = np.load("utility_matrix.npy")
    clusters = np.load("clusters.npy")
    num_clusters = 7

    # Global and local ranks for SVD
    f_g = 10
    f_c_dict = {0: 7, 1: 10, 2: 15, 3: 12, 4: 22, 5: 15, 6: 20}

    # ------------ Initialize rglsvd_model ------------
    rglsvd_model = RGLSVDRecommender(
        R_bin=R_bin,
        clusters=clusters,
        num_clusters=num_clusters,
        f_g=f_g,
        f_c=f_c_dict,
    )
    logger.info("Initialized RGLSVDRecommender.")

    # ------------ 1. LOOCV SPLIT ------------
    train_matrix, test_items = rglsvd_model.loocv_split()
    logger.info("Performed LOOCV split.")

    # ------------ 2. TRAINING ------------
    rglsvd_model.fit(
        convergence_threshold=0.01,
        weight_change_threshold=0.01,
        max_iterations=None,
    )
    logger.info(
        f"rglsvd_model trained with rGLSVD (converged in {rglsvd_model.num_iterations} iterations)."
    )

    # ------------ 3. METRICS EVALUATION ------------
    hr, arhr = rglsvd_model.evaluate_metrics(
        test_items=test_items,
        train_matrix=train_matrix,
        N=10,
    )

    logger.info(f"Hit-Rate (HR@10): {hr:.4f}")
    logger.info(f"ARHR: {arhr:.4f}")


if __name__ == "__main__":
    main()
