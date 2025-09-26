from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def rGLSVD(
    clusters: np.ndarray,
    R_bin: np.ndarray,
    num_clust: int = 7,
    fg: int = 10,
    fc: dict[int, int] = {0: 7, 1: 10, 2: 15, 3: 12, 4: 22, 5: 15, 6: 20},
    *,
    max_iter: int = 50,
    tol: float = 1e-3,
    frac_tol: float = 0.01,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate global user weights g via rGLSVD on implicit feedback.

    Args:
        clusters: (num_users,) cluster id in [0, num_clust-1] per user.
        R_bin: (num_users, num_items) binary implicit-feedback matrix.
        num_clust: Number of clusters.
        fg: Global SVD rank.
        fc: Per-cluster SVD rank mapping.
        max_iter: Max outer iterations.
        tol: Infinity-norm tolerance on g update (per-user).
        frac_tol: Stop if fraction of users with |Δg|>tol <= frac_tol.
        eps: Small constant to guard divisions by zero.

    Returns:
        (num_users,) vector g with values in [0, 1].
    """
    num_users, num_items = R_bin.shape
    if clusters.shape[0] != num_users:
        raise ValueError("clusters length must match number of users in R_bin")

    # Keep a sparse master copy
    R_sparse = sp.csr_matrix(R_bin, dtype=np.float64)

    # Initialize g
    g = np.full(num_users, 0.5, dtype=np.float64)

    def _safe_svds(
        X: sp.csr_matrix, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run svds with stable (descending) singular values and safe k."""
        m, n = X.shape
        k_safe = max(1, min(m, n) - 1)
        k_use = min(k, k_safe)
        if k_use < 1:
            # Degenerate case: return zeros with consistent shapes
            return (
                np.zeros((m, 1), dtype=np.float64),
                np.zeros((1,), dtype=np.float64),
                np.zeros((1, n), dtype=np.float64),
            )
        U, s, Vt = svds(X, k=k_use)  # s is ascending
        idx = np.argsort(s)[::-1]
        return U[:, idx], s[idx], Vt[idx, :]

    for _ in range(max_iter):
        # ---- Global weighted matrix and SVD ----
        Rg = sp.diags(g) @ R_sparse
        Ug, sg, Vtg = _safe_svds(Rg, fg)

        # ---- Local SVDs per cluster ----
        Uc_list: list[np.ndarray | None] = [None] * num_clust
        sc_list: list[np.ndarray | None] = [None] * num_clust
        Vtc_list: list[np.ndarray | None] = [None] * num_clust
        cluster_user_indices: list[np.ndarray | None] = [None] * num_clust

        for c in range(num_clust):
            idx_u = np.where(clusters == c)[0]
            cluster_user_indices[c] = idx_u
            if idx_u.size == 0:
                continue  # keep Nones; handled later
            Rc = R_sparse[idx_u, :]
            Rc_weighted = sp.diags(1.0 - g[idx_u]) @ Rc
            kc = fc.get(c, min(fg, 10))
            Uc, sc, Vtc = _safe_svds(Rc_weighted, kc)
            Uc_list[c], sc_list[c], Vtc_list[c] = Uc, sc, Vtc

        # ---- Update g user-wise ----
        g_new = np.empty_like(g)

        for u in range(num_users):
            # Global prediction for user u
            if sg.size:
                pu_g = Ug[u, :]
                pred_g = (pu_g * sg) @ Vtg  # shape: (num_items,)
            else:
                pred_g = np.zeros((num_items,), dtype=np.float64)

            # Local prediction for user u
            c = int(clusters[u])

            idx_u_cluster_opt: np.ndarray | None = cluster_user_indices[c]
            Uc_opt: np.ndarray | None = Uc_list[c]
            sc_opt: np.ndarray | None = sc_list[c]
            Vtc_opt: np.ndarray | None = Vtc_list[c]

            if (
                idx_u_cluster_opt is None
                or idx_u_cluster_opt.size == 0
                or Uc_opt is None
                or sc_opt is None
                or Vtc_opt is None
                or sc_opt.size == 0
            ):
                pred_c = np.zeros((num_items,), dtype=np.float64)
            else:
                local_pos = np.where(idx_u_cluster_opt == u)[0]
                if local_pos.size == 0:
                    pred_c = np.zeros((num_items,), dtype=np.float64)
                else:
                    pu_c = Uc_opt[local_pos[0], :]
                    pred_c = (pu_c * sc_opt) @ Vtc_opt

            # Closed-form update with a = pred_g / g_u, b = pred_c / (1-g_u)
            gu = g[u]
            a = pred_g / max(gu, eps)
            b = pred_c / max(1.0 - gu, eps)
            diff = a - b

            ru = R_bin[u, :]
            num = float(np.sum(diff * (ru - b)))
            den = float(np.sum(diff * diff))

            g_new[u] = 1.0 if den <= eps else float(np.clip(num / den, 0.0, 1.0))

        # ---- Stopping criteria ----
        delta = np.abs(g_new - g)
        if float(np.max(delta)) <= tol or float(np.mean(delta > tol)) <= frac_tol:
            g = g_new
            break

        g = g_new

    return g
