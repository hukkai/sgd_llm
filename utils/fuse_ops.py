from __future__ import annotations

import math

import torch


ORDER_2_CUTOFF = 0.05
ORDER_3_CUTOFF = 0.25
ORDER_4_CUTOFF = 1.0
LARGE_STEP_TAYLOR_ORDER = 6


def asym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x - x.mT)


def build_transition(
    x: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    G = x @ x.mT
    M = delta @ x.mT
    N = x @ delta.mT
    H = delta @ delta.mT
    K = asym(M)

    top = torch.cat((-(G @ K + N), G), dim=-1)
    bottom = torch.cat((-(M @ K + H), M), dim=-1)
    transition = torch.cat((top, bottom), dim=-2)
    basis = torch.cat((x, delta), dim=-2)
    return transition, basis, G, M, N, H, K


def generator_fro_norm(
    G: torch.Tensor,
    M: torch.Tensor,
    N: torch.Tensor,
    H: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    work_dtype = torch.float32 if G.dtype in (torch.float16, torch.bfloat16) else G.dtype
    G = G.to(dtype=work_dtype)
    M = M.to(dtype=work_dtype)
    N = N.to(dtype=work_dtype)
    H = H.to(dtype=work_dtype)
    K = K.to(dtype=work_dtype)

    rr_t = H + M @ K - K @ N - K @ G @ K
    rx_t = K @ G - M
    xr_t = -(N + G @ K)

    u_gram = torch.cat(
        (
            torch.cat((G, N), dim=-1),
            torch.cat((M, H), dim=-1),
        ),
        dim=-2,
    )
    v_gram = torch.cat(
        (
            torch.cat((rr_t, rx_t), dim=-1),
            torch.cat((xr_t, G), dim=-1),
        ),
        dim=-2,
    )

    fro_sq = (u_gram * v_gram.mT).sum(dim=(-2, -1))
    return fro_sq.clamp_min_(0).sqrt()


def taylor_coeff_exp(T: torch.Tensor, order: int) -> torch.Tensor:
    batch, dim, _ = T.shape
    eye = torch.eye(dim, device=T.device, dtype=T.dtype).expand(batch, dim, dim)
    coeff = eye.clone()
    term = eye
    for degree in range(1, order + 1):
        term = (term @ T) / float(degree)
        coeff = coeff + term
    return coeff


def scaled_taylor_coeff_exp(
    T: torch.Tensor, norm: float, base_order: int = LARGE_STEP_TAYLOR_ORDER
) -> torch.Tensor:
    if not math.isfinite(norm):
        raise ValueError(f"Expected a finite generator norm, got {norm}.")

    scale_steps = 0
    if norm > 0:
        scale_steps = max(0, math.ceil(math.log2(norm)))

    if scale_steps > 0:
        T = T * math.ldexp(1.0, -scale_steps)

    coeff = taylor_coeff_exp(T, base_order)
    for _ in range(scale_steps):
        coeff = coeff @ coeff
    return coeff


@torch.no_grad()
def update_fused(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or delta.ndim != 3:
        raise ValueError(f"Expected rank-3 inputs, got x.ndim={x.ndim}, delta.ndim={delta.ndim}.")
    if x.shape != delta.shape:
        raise ValueError(f"Expected matching shapes, got x.shape={x.shape}, delta.shape={delta.shape}.")

    transition, basis, G, M, N, H, K = build_transition(x, delta)
    norm = float(generator_fro_norm(G, M, N, H, K).max().item())

    if norm < ORDER_2_CUTOFF:
        coeff = taylor_coeff_exp(transition, 2)
    elif norm < ORDER_3_CUTOFF:
        coeff = taylor_coeff_exp(transition, 3)
    elif norm < ORDER_4_CUTOFF:
        coeff = taylor_coeff_exp(transition, 4)
    else:
        coeff = scaled_taylor_coeff_exp(transition, norm)

    n = x.size(-2)
    return coeff.narrow(-2, 0, n) @ basis
