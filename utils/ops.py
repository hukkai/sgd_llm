from __future__ import annotations

import torch


@torch.no_grad()
def polar(
    a: torch.Tensor,
    tolerance: float = 1e-5,
    eps: float = 1e-10,
) -> torch.Tensor:
    if a.ndim != 3:
        raise ValueError(f"expected a to have shape (b, n, m), got {tuple(a.shape)}")

    _, n, m = a.shape
    if n > m:
        raise ValueError(f"expected n <= m, got shape {tuple(a.shape)}")

    screen_dtype = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    a_screen = a if a.dtype == screen_dtype else a.to(screen_dtype)

    ident = torch.eye(n, device=a.device, dtype=screen_dtype)
    aat = a_screen @ a_screen.transpose(-1, -2)
    err = torch.linalg.matrix_norm(aat - ident, ord="fro", dim=(-2, -1))
    mask = err > tolerance

    if not mask.any():
        return a

    a_bad = a[mask].to(torch.float64)
    aat_bad = a_bad @ a_bad.transpose(-1, -2)

    try:
        eigvals, eigvecs = torch.linalg.eigh(aat_bad)
        inv_sqrt = eigvals.clamp_min(eps).rsqrt()
        aat_inv_sqrt = (eigvecs * inv_sqrt.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        q_bad = aat_inv_sqrt @ a_bad
    except RuntimeError:
        u, _, vh = torch.linalg.svd(a_bad, full_matrices=False)
        q_bad = u @ vh

    out = a.clone()
    out[mask] = q_bad.to(a.dtype)
    return out