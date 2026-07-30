"""Weight projection: reduce large-model weights to small-model shape. [M3]

A weight matrix touches two spaces (one per axis). To project it to the small
shape we reduce each axis with a basis P (shape: small_dim x large_dim), giving
    W_hat = P_out @ W @ P_in.T
The bases come from the *size relationship*, NOT from the target weights:
  - residual-stream axis -> `pca_basis` of the large model's residual activations
    (one global basis, so the residual stream stays consistent across the model)
  - internal axes (attention head space, MLP hidden) -> `svd_axis_basis`, the
    weight's own top singular directions on that axis
The leftover W_small - W_hat is the residual M4 analyzes.

`fit_operator` is separate: it *fits* a shared (A,B) to reproduce the target
weights (it peeks at W_small), so it is an M4-style "is there a linear operator
relating the sizes?" probe, not a target-free projection.
"""
from __future__ import annotations

import torch


def relative_error(target: torch.Tensor, approx: torch.Tensor) -> float:
    """‖target − approx‖_F / ‖target‖_F for one matrix."""
    return (torch.linalg.norm(target - approx) / torch.linalg.norm(target)).item()


def pca_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k principal directions of X (n, d) as rows -> P (k, d), orthonormal.

    Use as a width-reduction map for the residual stream: a residual vector v
    (large width d) reduces to `P @ v` (small width k). Via the (d, d) covariance
    eigendecomposition — X is very tall (n >> d), so a direct SVD would allocate a
    huge left-singular matrix.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc                                  # (d, d)
    evecs = torch.linalg.eigh(cov).eigenvectors      # ascending eigenvalues
    return evecs[:, -k:].flip(1).T                   # top-k directions as rows


def svd_axis_basis(W: torch.Tensor, k: int, axis: int) -> torch.Tensor:
    """Top-k singular directions of W along one axis -> P (k, dim_axis).

    axis=0 reduces the output (rows) via left singular vectors; axis=1 reduces the
    input (cols) via right singular vectors. Used for the internal (head / MLP)
    axes, which have no activation-based correspondence. W must have rank >= k on
    that axis (stack every weight touching the axis if one alone is too thin).
    """
    assert k <= min(W.shape), f"rank {min(W.shape)} < k={k}; stack more weights"
    U, _, Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k].T if axis == 0 else Vh[:k]


def project_weight(W: torch.Tensor, P_out: torch.Tensor | None, P_in: torch.Tensor | None) -> torch.Tensor:
    """W_hat = P_out @ W @ P_in.T. A None basis leaves that axis unchanged
    (e.g. the shared vocabulary axis of the embeddings)."""
    if P_out is not None:
        W = P_out @ W
    if P_in is not None:
        W = W @ P_in.T
    return W


def fit_operator(WL: torch.Tensor, WS: torch.Tensor, *, steps: int = 1500, lr: float = 5e-3):
    """Fit ONE shared (A, B) minimizing ‖WS − A·WL·Bᵀ‖ over all layers of a type.

    Peeks at the target WS, so this is a diagnostic for M4's "single linear
    operator across layers?" question, not a target-free projection. Returns
    (projected_stack, mean_relative_error). One (A,B) for all L layers is what
    makes it constrained — a per-layer fit of this form is degenerate (the form
    A·W·B with W full-rank can reproduce any target, so it trivially hits 0).
    """
    out_S, in_S = WS.shape[1], WS.shape[2]
    out_L, in_L = WL.shape[1], WL.shape[2]
    rank = min(out_L, in_L)
    if out_S <= rank and in_S <= rank:                              # SVD-of-mean warm start
        U, _, Vh = torch.linalg.svd(WL.mean(0), full_matrices=False)
        A, B = U[:, :out_S].T.clone(), Vh[:in_S].clone()
    else:                                                           # rank too low (e.g. MLP axes)
        A = torch.linalg.qr(torch.randn(out_L, out_S, device=WL.device))[0].T
        B = torch.linalg.qr(torch.randn(in_L, in_S, device=WL.device))[0].T
    A.requires_grad_(True)
    B.requires_grad_(True)
    denom = WS.pow(2).sum()
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = torch.matmul(torch.matmul(A, WL), B.T)
        (WS - pred).pow(2).sum().div(denom).backward()
        opt.step()
    with torch.no_grad():
        pred = torch.matmul(torch.matmul(A, WL), B.T)
    errs = [relative_error(WS[l], pred[l]) for l in range(WL.shape[0])]
    return pred, sum(errs) / len(errs)
