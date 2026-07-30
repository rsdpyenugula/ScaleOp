"""Representation alignment: CKA (and, later, ridge / Procrustes maps). [M2]

Before comparing weights across two model sizes we first ask how their
residual-stream *representations* relate. Linear CKA is a basis-independent
similarity: it ignores rotations/scalings of the feature axes, so it can compare
a 2048-wide model to a 1024-wide one and tell us whether layer i of one lines up
with layer i of the other.
"""
from __future__ import annotations

import torch


def _center(x: torch.Tensor) -> torch.Tensor:
    """Subtract each feature's mean across samples (CKA needs centered features)."""
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA similarity between two activation matrices.

    X: (n, d1), Y: (n, d2) — the SAME n samples (e.g. token positions) seen by two
    models of possibly different width. Returns a scalar in [0, 1]: 1 means the
    two representations are the same up to a linear transform, 0 means unrelated.
    """
    X, Y = _center(X), _center(Y)
    xty = (X.T @ Y).pow(2).sum()          # ||X^T Y||_F^2
    xtx = (X.T @ X).pow(2).sum().sqrt()   # ||X^T X||_F
    yty = (Y.T @ Y).pow(2).sum().sqrt()   # ||Y^T Y||_F
    return (xty / (xtx * yty)).item()


@torch.no_grad()
def cka_matrix(A: torch.Tensor, B: torch.Tensor, *, device=None) -> torch.Tensor:
    """All-pairs linear CKA between the layers of two models.

    A: (La, N, dA), B: (Lb, N, dB) — per-layer activations on the SAME N samples.
    Returns a (La, Lb) CKA matrix. For a same-depth pair, a dominant diagonal
    means layer i of one model corresponds to layer i of the other. Self-norms are
    precomputed per layer so only the La·Lb cross terms are recomputed.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    A = [_center(A[i].to(device, torch.float32)) for i in range(A.shape[0])]
    B = [_center(B[j].to(device, torch.float32)) for j in range(B.shape[0])]
    a_norm = [(X.T @ X).pow(2).sum().sqrt() for X in A]
    b_norm = [(Y.T @ Y).pow(2).sum().sqrt() for Y in B]

    out = torch.empty(len(A), len(B))
    for i, X in enumerate(A):
        for j, Y in enumerate(B):
            out[i, j] = ((X.T @ Y).pow(2).sum() / (a_norm[i] * b_norm[j])).item()
    return out


# --- Width-conversion maps: how well does large-model activation space map onto
#     the small model's? R^2 on held-out data is the discriminative per-layer
#     measure CKA couldn't give (CKA saturates in the middle band). ------------

def _split_center(X, Y, n_train):
    """Split into train/eval and center both by the TRAIN means (no test leakage)."""
    xm, ym = X[:n_train].mean(0), Y[:n_train].mean(0)
    return X[:n_train] - xm, X[n_train:] - xm, Y[:n_train] - ym, Y[n_train:] - ym


def _r2(y_true, y_pred):
    """Coefficient of determination; inputs already centered by the train mean,
    so the baseline (predict-the-mean) sum-of-squares is just ||y_true||^2."""
    return (1 - (y_true - y_pred).pow(2).sum() / y_true.pow(2).sum()).item()


def ridge_r2(X, Y, n_train, *, lam: float = 1e-3) -> float:
    """Held-out R^2 of an UNCONSTRAINED linear map Y ≈ X·W (ridge-regularized).

    X: (N, dA) large-model activations, Y: (N, dB) small-model. Fit W on the first
    `n_train` rows, score R^2 on the rest. `lam` is scaled by the mean diagonal of
    XᵀX so it's invariant to feature scale. This is the most permissive alignment
    (a full 2048→1024 linear map).
    """
    Xtr, Xte, Ytr, Yte = _split_center(X, Y, n_train)
    XtX = Xtr.T @ Xtr
    reg = lam * XtX.diagonal().mean()
    W = torch.linalg.solve(XtX + reg * torch.eye(XtX.shape[0], device=X.device), Xtr.T @ Ytr)
    return _r2(Yte, Xte @ W)


def procrustes_r2(X, Y, n_train, *, k: int | None = None) -> float:
    """Held-out R^2 of a CONSTRAINED map: SVD-reduce X to k dims, then a scaled
    orthogonal (rotation-only) fit to Y.

    Much less free than ridge — only a rotation + one global scale — so the gap
    between this and `ridge_r2` measures how much of the cross-width relationship
    is *just* a change of basis vs. a genuine linear reshaping. `k` defaults to the
    small model's width.
    """
    Xtr, Xte, Ytr, Yte = _split_center(X, Y, n_train)
    k = k or Ytr.shape[1]
    V = torch.linalg.svd(Xtr, full_matrices=False)[2][:k].T   # (dA, k) top right-singular vecs
    Xtr_r, Xte_r = Xtr @ V, Xte @ V
    U, S, Vh = torch.linalg.svd(Xtr_r.T @ Ytr)                # orthogonal fit
    R = U @ Vh
    scale = S.sum() / Xtr_r.pow(2).sum()                     # optimal isotropic scale
    return _r2(Yte, scale * (Xte_r @ R))
