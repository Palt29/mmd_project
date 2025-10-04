def main() -> None:
    import logging

    import numpy as np

    from src.loocv import loocv_split

    # Logger setup
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger(__name__)

    # rGLSVD algorithm pipeline
    R_bin = np.load("utility_matrix.npy")
    clusters = np.load("clusters.npy")

    num_clusters = 7
    # Global rank for SVD
    f_g = 18
    # Local ranks (one for each cluster) for local SVD
    f_c_dict = (
        {
            0: 7,
            1: 5,
            2: 10,
            3: 8,
            4: 6,
            5: 7,
            6: 6,
        },
    )  # test numbers

    # 1. LOOCV SPLIT
    train_matrix, test_items = loocv_split(R_bin)

    print(clusters, num_clusters, f_g, f_c_dict, train_matrix, test_items)


if __name__ == "__main__":
    main()
