def main() -> None:
    import logging

    # Logger setup
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger(__name__)

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

    print(num_clusters, f_g, f_c_dict)


if __name__ == "__main__":
    main()
