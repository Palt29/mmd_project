"""Run rGLSVD, sGLSVD, and CF (Jaccard/Cosine) evaluation pipelines."""

import logging

import numpy as np

from src.utils import run_cf, run_rglsvd, run_sglsvd


def main() -> None:
    # ------------ Logger setup ------------
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # Reproducibility for LOOCV sampling
    np.random.seed(42)

    # ------------ Load data ------------
    R_bin = np.load("utility_matrix.npy")  # binary implicit matrix (users × items)
    clusters = np.load("clusters.npy")  # initial/fixed user clusters
    num_clusters = int(np.max(clusters)) + 1

    # Global and local ranks
    f_g = 10
    f_c_dict = {0: 7, 1: 10, 2: 15, 3: 12, 4: 22, 5: 15, 6: 20}
    f_c_shared = 10  # sGLSVD uses a single local rank across clusters
    topN = 10

    # ------------ rGLSVD ------------
    logger.info("=== rGLSVD ===")
    hr_r, arhr_r = run_rglsvd(
        R_bin=R_bin,
        clusters=clusters,
        num_clusters=num_clusters,
        f_g=f_g,
        f_c_dict=f_c_dict,
        topN=topN,
        logger=logger,
    )
    logger.info("rGLSVD  HR@%d: %.4f | ARHR: %.4f", topN, hr_r, arhr_r)

    # ------------ sGLSVD ------------
    logger.info("=== sGLSVD ===")
    hr_s, arhr_s = run_sglsvd(
        R_bin=R_bin,
        initial_clusters=clusters,
        num_clusters=num_clusters,
        f_g=f_g,
        f_c_shared=f_c_shared,
        topN=topN,
        logger=logger,
    )
    logger.info("sGLSVD  HR@%d: %.4f | ARHR: %.4f", topN, hr_s, arhr_s)

    # ------------ CF (Jaccard) ------------
    logger.info("=== CF (user-user, Jaccard, k=20) ===")
    hr_cj, arhr_cj = run_cf(
        R_bin=R_bin,
        similarity="jaccard",
        k_neighbors=20,
        topN=topN,
        logger=logger,
    )
    logger.info("CF-Jaccard  HR@%d: %.4f | ARHR: %.4f", topN, hr_cj, arhr_cj)

    # ------------ CF (Cosine) ------------
    logger.info("=== CF (user-user, Cosine, k=20) ===")
    hr_cc, arhr_cc = run_cf(
        R_bin=R_bin,
        similarity="cosine",
        k_neighbors=20,
        topN=topN,
        logger=logger,
    )
    logger.info("CF-Cosine   HR@%d: %.4f | ARHR: %.4f", topN, hr_cc, arhr_cc)


if __name__ == "__main__":
    main()
