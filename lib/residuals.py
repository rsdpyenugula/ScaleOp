"""Residual analysis: is Δ = W_small − Project(W_large) structured or noise? [M4]

Every test compares the real residual against a matched control with the same
entries but destroyed structure (a random permutation of the elements), and
against a Gaussian with matched mean/std. Structure shows up as: lower effective
rank (energy concentrated in few directions) and higher cross-layer similarity
than the controls.
"""
from __future__ import annotations

import torch


def effective_rank(M: torch.Tensor) -> float:
    """Entropy-based effective rank: exp(H(p)) where p = singular values / sum.

    A matrix with all its energy in one direction scores 1; an isotropic noise
    matrix scores near min(shape). Lower than control = spectrally structured.
    """
    s = torch.linalg.svdvals(M)
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()


def shuffle_control(M: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Same entries, randomly permuted — kills structure, keeps the value histogram."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(M.numel(), generator=g).to(M.device)
    return M.flatten()[perm].view_as(M)


def gaussian_control(M: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Gaussian noise with the same mean/std/shape as M."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(M.shape, generator=g).to(M.device) * M.std() + M.mean()


def cross_layer_cosine(deltas: torch.Tensor) -> float:
    """Mean pairwise cosine similarity between vectorized per-layer residuals.

    deltas: (L, out, in), one residual per layer of the same weight type. If the
    residuals share a direction across layers, this exceeds the shuffle control
    (which is ~0 for anything high-dimensional).
    """
    V = torch.nn.functional.normalize(deltas.flatten(1), dim=1)
    C = V @ V.T
    L = C.shape[0]
    return ((C.sum() - L) / (L * (L - 1))).item()


# --- Patch predictor R(Ŵ) → Δ (M4 tests 3–5). Patches are P×P so the same
#     predictor applies to any weight shape / model pair (transfer test). -----

PATCH = 16
TYPES = ["Q", "K", "V", "O", "MLP_UP", "MLP_DOWN"]


def patches(M: torch.Tensor):
    """(L, O, I) -> (L, nP, PATCH*PATCH) patch vectors + per-patch (row, col) fracs."""
    L, O, I = M.shape
    x = (M.view(L, O // PATCH, PATCH, I // PATCH, PATCH)
          .permute(0, 1, 3, 2, 4).reshape(L, -1, PATCH * PATCH))
    r = torch.arange(O // PATCH).repeat_interleave(I // PATCH) / (O // PATCH)
    c = torch.arange(I // PATCH).repeat(O // PATCH) / (I // PATCH)
    return x, r, c


def unpatch(x: torch.Tensor, O: int, I: int) -> torch.Tensor:
    """Inverse of `patches` for one layer: (nP, PATCH*PATCH) -> (O, I)."""
    return (x.view(O // PATCH, I // PATCH, PATCH, PATCH)
             .permute(0, 2, 1, 3).reshape(O, I))


def patch_dataset(What, Delta, t_idx, device, shuffle_layers=None):
    """Per-type (X, Y, layer_idx): X = [Ŵ patch | type one-hot | layer, row, col].

    `shuffle_layers` (a layer permutation) builds the control pairing: patches of
    Ŵ from layer l paired with Δ from layer perm[l] — same marginals, broken
    layer-specific correspondence.
    """
    L = What.shape[0]
    src = Delta[shuffle_layers] if shuffle_layers is not None else Delta
    xw, r, c = patches(What)
    yd, _, _ = patches(src)
    n = xw.shape[1]
    onehot = torch.zeros(len(TYPES), device=device)
    onehot[t_idx] = 1
    feats = [torch.cat([xw[l], onehot.expand(n, -1),
                        torch.full((n, 1), l / (L - 1), device=device),
                        r.to(device).unsqueeze(1), c.to(device).unsqueeze(1)], dim=1)
             for l in range(L)]
    layer_idx = torch.arange(L, device=device).repeat_interleave(n)
    return torch.cat(feats), yd.reshape(-1, PATCH * PATCH), layer_idx


def train_patch_predictor(X, Y, hidden, *, epochs=5, bs=16384, lr=1e-3, seed=0):
    """Train R on the given rows. hidden=0 -> linear; else one-hidden-layer MLP."""
    from torch import nn

    torch.manual_seed(seed)
    model = (nn.Linear(X.shape[1], Y.shape[1]) if hidden == 0 else
             nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                           nn.Linear(hidden, Y.shape[1]))).to(X.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0], device=X.device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(X[idx]), Y[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def predict_patches(model, X, bs=16384):
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


@torch.no_grad()
def eval_patches(model, X, Y, bs=16384):
    """Per-patch squared errors of the prediction and of predicting zero."""
    se_pred = (Y - predict_patches(model, X, bs)).pow(2).sum(1)
    return se_pred, Y.pow(2).sum(1)


def err_reduction(se_pred, se_zero) -> float:
    """1 − ‖Δ − R‖/‖Δ‖ over the given patches; 0 = no better than predicting zero."""
    return (1 - (se_pred.sum() / se_zero.sum()).sqrt()).item()
