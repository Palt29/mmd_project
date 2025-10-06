"""Run the full rGLSVD evaluation pipeline."""

import logging

import numpy as np

from src.loocv import loocv_split
from src.loocv2 import evaluate_metrics
from src.rGLSVD import rGLSVD


def main() -> None:
    # ------------ Logger setup ------------
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)  # get a named logger

    # ------------ rGLSVD algorithm pipeline ------------
    # Assumption: clusters, num_clusters, f_g, and f_c previously defined
    R_bin = np.load("utility_matrix.npy")
    clusters = np.load("clusters.npy")
    num_clusters = 7

    # Global and local ranks for SVD
    f_g = 18
    f_c_dict = {0: 7, 1: 5, 2: 10, 3: 8, 4: 6, 5: 7, 6: 6}

    # 1. LOOCV SPLIT
    train_matrix, test_items = loocv_split(R_bin)
    logger.info("Performed LOOCV split.")

    # 2. TRAINING
    trained_factors = rGLSVD(train_matrix, clusters, num_clusters, f_g, f_c_dict)
    logger.info("Model trained with rGLSVD.")

    # 3. METRICS EVALUATION
    hr, arhr = evaluate_metrics(trained_factors, test_items, train_matrix, clusters, 10)

    logger.info(f"Hit-Rate (HR@10): {hr:.4f}")
    logger.info(f"ARHR: {arhr:.4f}")


if __name__ == "__main__":
    main()
