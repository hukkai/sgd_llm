import torch

from .ops import polar as exact_polar


TAYLOR2_MAX_ERR = 0.06
TAYLOR3_MAX_ERR = 0.14
TAYLOR4_MAX_ERR = 0.28

_COEFFS2 = (-0.5, 0.375)
_COEFFS3 = (-0.5, 0.375, -0.3125)
_COEFFS4 = (-0.5, 0.375, -0.3125, 0.2734375)


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def stiefel_project(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return grad - _symmetrize(grad @ x.transpose(-1, -2)) @ x


def _screen_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _validate_shape(a: torch.Tensor) -> None:
    if a.ndim != 3:
        raise ValueError(f"expected a to have shape (b, n, m), got {tuple(a.shape)}")
    if a.shape[-2] > a.shape[-1]:
        raise ValueError(f"expected n <= m, got shape {tuple(a.shape)}")


def _gram_error(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gram_error = a @ a.transpose(-1, -2)
    gram_error.diagonal(dim1=-2, dim2=-1).sub_(1)
    err = torch.linalg.matrix_norm(gram_error, ord="fro", dim=(-2, -1))
    return gram_error, err


def _apply_series(
    a: torch.Tensor,
    gram_error: torch.Tensor,
    coeffs: tuple[float, ...],
) -> torch.Tensor:
    term = gram_error @ a
    q = a + coeffs[0] * term
    for coeff in coeffs[1:]:
        term = gram_error @ term
        q = q + coeff * term
    return q


@torch.no_grad()
def polar_taylor2(a: torch.Tensor) -> torch.Tensor:
    _validate_shape(a)
    work = a.to(_screen_dtype(a.dtype))
    gram_error, _ = _gram_error(work)
    return _apply_series(work, gram_error, _COEFFS2).to(a.dtype)


@torch.no_grad()
def polar_taylor3(a: torch.Tensor) -> torch.Tensor:
    _validate_shape(a)
    work = a.to(_screen_dtype(a.dtype))
    gram_error, _ = _gram_error(work)
    return _apply_series(work, gram_error, _COEFFS3).to(a.dtype)


@torch.no_grad()
def polar_taylor4(a: torch.Tensor) -> torch.Tensor:
    _validate_shape(a)
    work = a.to(_screen_dtype(a.dtype))
    gram_error, _ = _gram_error(work)
    return _apply_series(work, gram_error, _COEFFS4).to(a.dtype)


@torch.no_grad()
def fast_polar(
    a: torch.Tensor,
    tolerance: float = 1e-5,
    eps: float = 1e-10,
    taylor2_max_err: float = TAYLOR2_MAX_ERR,
    taylor3_max_err: float = TAYLOR3_MAX_ERR,
    taylor4_max_err: float = TAYLOR4_MAX_ERR,
) -> torch.Tensor:
    """
    Approximate Q = (A A^T)^(-1/2) A with the binomial series around A A^T = I.

    For E = A A^T - I:
        (I + E)^(-1/2) = I - 1/2 E + 3/8 E^2 - 5/16 E^3 + 35/128 E^4 + ...
    """
    _validate_shape(a)

    work = a.to(_screen_dtype(a.dtype))
    gram_error, err = _gram_error(work)
    max_err = err.max().item()

    if max_err <= tolerance:
        return a

    if max_err < taylor2_max_err:
        return _apply_series(work, gram_error, _COEFFS2).to(a.dtype)

    if max_err < taylor3_max_err:
        return _apply_series(work, gram_error, _COEFFS3).to(a.dtype)

    if max_err < taylor4_max_err:
        return _apply_series(work, gram_error, _COEFFS4).to(a.dtype)

    return exact_polar(a, tolerance=0.0, eps=eps)


def stiefel_update_taylor(
    x: torch.Tensor,
    update: torch.Tensor,
    tolerance: float = 1e-5,
    eps: float = 1e-10,
    taylor2_max_err: float = TAYLOR2_MAX_ERR,
    taylor3_max_err: float = TAYLOR3_MAX_ERR,
    taylor4_max_err: float = TAYLOR4_MAX_ERR,
    projected: bool = False,
) -> torch.Tensor:
    if not projected:
        update = stiefel_project(x, update)
    return fast_polar(
        x + update,
        tolerance=tolerance,
        eps=eps,
        taylor2_max_err=taylor2_max_err,
        taylor3_max_err=taylor3_max_err,
        taylor4_max_err=taylor4_max_err,
    )

