"""Residual-Quantization Compatibility (RQC) scoring utilities.

RQC measures the compatibility between the current query residual and a
candidate RQ codeword. It is algebraically equal to the decrease in squared
residual norm produced by appending that codeword. The implementation is
decoder-independent so it can be tested without loading model checkpoints.
"""
from __future__ import annotations

import torch


def residual_reconstruction_gain(
    query_embedding: torch.Tensor,
    prefix_reconstruction: torch.Tensor,
    codebook: torch.Tensor,
    *,
    normalize: bool = False,
    mode: str = "gain",
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return the residual reconstruction gain for every candidate codeword.

    Let rho = q - r_prefix. The score is the exact decrease in squared
    residual norm after appending one codeword:

        Delta(c | rho) = ||rho||^2 - ||rho - c||^2
                       = 2 rho^T c - ||c||^2

    Args:
        query_embedding: ``[N, D]`` or ``[D]`` query/RQ embedding.
        prefix_reconstruction: ``[N, D]`` or ``[D]`` sum of committed codewords.
        codebook: ``[K, D]`` codebook for the level being expanded.
        normalize: z-normalize the K biases independently for each row.
        mode: ``gain`` for exact squared-error reduction, ``alignment`` for
            residual--codeword dot product, or ``cosine`` for scale-free
            residual--codeword alignment.
        eps: minimum standard deviation used by normalization.

    Returns:
        A float32 tensor of shape ``[N, K]`` (or ``[K]`` for 1-D input).
    """
    squeeze = query_embedding.dim() == 1
    if squeeze:
        query_embedding = query_embedding.unsqueeze(0)
    if prefix_reconstruction.dim() == 1:
        prefix_reconstruction = prefix_reconstruction.unsqueeze(0)

    if query_embedding.dim() != 2 or prefix_reconstruction.dim() != 2:
        raise ValueError("query_embedding and prefix_reconstruction must be 1-D or 2-D")
    if query_embedding.shape != prefix_reconstruction.shape:
        raise ValueError(
            "query_embedding and prefix_reconstruction must have identical shape, "
            f"got {tuple(query_embedding.shape)} and {tuple(prefix_reconstruction.shape)}"
        )
    if codebook.dim() != 2 or codebook.size(1) != query_embedding.size(1):
        raise ValueError(
            f"codebook must be [K,{query_embedding.size(1)}], got {tuple(codebook.shape)}"
        )

    query = query_embedding.float()
    prefix = prefix_reconstruction.float()
    codebook = codebook.float()
    residual = query - prefix
    mode = str(mode).lower()
    if mode == "gain":
        bias = 2.0 * residual @ codebook.t() - codebook.square().sum(dim=-1).unsqueeze(0)
    elif mode == "alignment":
        bias = residual @ codebook.t()
    elif mode == "cosine":
        bias = torch.nn.functional.normalize(residual, dim=-1) @ torch.nn.functional.normalize(
            codebook, dim=-1
        ).t()
    else:
        raise ValueError(f"unknown residual compatibility mode: {mode}")

    if normalize:
        bias = bias - bias.mean(dim=-1, keepdim=True)
        bias = bias / bias.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))

    return bias.squeeze(0) if squeeze else bias


def selected_residual_reconstruction_gain(
    query_embedding: torch.Tensor,
    prefix_reconstruction: torch.Tensor,
    codebook: torch.Tensor,
    selected_code_ids: torch.Tensor,
    *,
    normalize: bool = False,
    mode: str = "gain",
    eps: float = 1.0e-6,
    moments: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Score one selected codeword per row without materializing all K scores.

    With normalization enabled, the mean and variance over the full codebook
    are computed analytically from codebook moments. This is mathematically
    equivalent to selecting from ``residual_reconstruction_gain(..., True)``.
    """
    query = query_embedding.float()
    prefix = prefix_reconstruction.float()
    codes = codebook.float()
    selected = selected_code_ids.long()
    if query.dim() != 2 or prefix.shape != query.shape:
        raise ValueError("query and prefix reconstruction must have identical [N,D] shape")
    if selected.shape != (query.size(0),):
        raise ValueError("selected_code_ids must have shape [N]")
    mode = str(mode).lower()
    if mode != "gain":
        raise ValueError("selected analytic RQC currently supports mode='gain' only")
    residual = query - prefix
    selected_codes = codes.index_select(0, selected)
    raw = 2.0 * (residual * selected_codes).sum(-1) - selected_codes.square().sum(-1)
    if not normalize:
        return raw

    if moments is None:
        norms = codes.square().sum(-1)
        moments = (
            codes.mean(0),
            norms.mean(),
            codes.t().matmul(codes) / float(codes.size(0)),
            (codes * norms[:, None]).mean(0),
            norms.square().mean(),
        )
    mean_code, mean_norm, second_code, code_norm_cross, mean_norm_sq = moments
    mean = 2.0 * (residual * mean_code).sum(-1) - mean_norm
    second = (
        4.0 * ((residual.matmul(second_code)) * residual).sum(-1)
        - 4.0 * (residual * code_norm_cross).sum(-1)
        + mean_norm_sq
    )
    variance = (second - mean.square()).clamp_min(float(eps) ** 2)
    return (raw - mean) / variance.sqrt()


# Canonical public terminology. The legacy names above remain aliases used by
# existing checkpoints/configuration plumbing; both paths are exactly
# equivalent and return the same tensors.
residual_quantization_compatibility = residual_reconstruction_gain
selected_residual_quantization_compatibility = selected_residual_reconstruction_gain
