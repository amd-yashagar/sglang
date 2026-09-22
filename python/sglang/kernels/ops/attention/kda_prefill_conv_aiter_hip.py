"""Fail-closed gfx950 FlyDSL causal-conv for Kimi-K3 KDA prefill.

Not dispatched from serving: on gfx950 K3 shapes (T=1024, dim=4608) this
kernel measured ~4.6x slower than Triton ``causal_conv1d_fn``. Keep using
Triton for prefill conv. This adapter exists so the AITER stride / FP32-weight
fixes stay correctness-gated.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.utils import is_hip

_FAILED = False
_WARMED: set[tuple[int, int, int, int]] = set()


def enabled() -> bool:
    return os.environ.get("SGLANG_K3_KDA_FUSED_BACKEND", "").lower() == "aiter"


def failed() -> bool:
    return _FAILED


def _fn():
    try:
        from aiter.ops.flydsl.causal_conv1d_flydsl import (
            causal_conv1d_split_qkv_flydsl_fn,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    return causal_conv1d_split_qkv_flydsl_fn


def available() -> bool:
    return bool(is_hip() and enabled() and torch.cuda.is_available() and _fn())


def covered(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
) -> bool:
    if _FAILED or not available() or x.ndim != 2 or weight.ndim != 2:
        return False
    dim = x.shape[0]
    width = int(weight.shape[1])
    return (
        q_dim == k_dim
        and q_dim > 0
        and v_dim > 0
        and dim == q_dim + k_dim + v_dim
        and width in (2, 3, 4)
        and weight.shape[0] == dim
        and x.dtype == torch.bfloat16
        and weight.dtype in (torch.float32, torch.bfloat16)
        and conv_states.ndim == 3
        and conv_states.shape[1:] == (dim, width - 1)
        and conv_states.dtype == torch.bfloat16
        and query_start_loc.dtype == torch.int32
        and cache_indices.dtype == torch.int32
        and has_initial_state.dtype == torch.bool
        and cache_indices.shape[0] == has_initial_state.shape[0]
        and cache_indices.shape[0] == query_start_loc.numel() - 1
    )


def run(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _FAILED
    fn = _fn()
    if fn is None:
        raise RuntimeError("AITER FlyDSL causal conv is unavailable")
    try:
        return fn(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            k_dim_size=k_dim,
            v_dim_size=v_dim,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
        )
    except Exception:
        _FAILED = True
        raise


def warmup(
    *,
    weight: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
) -> None:
    if not available():
        return
    dim = q_dim + k_dim + v_dim
    device_index = -1 if weight.device.index is None else weight.device.index
    key = (device_index, dim, int(weight.shape[1]), int(weight.dtype != torch.float32))
    if key in _WARMED:
        return
    device = weight.device
    tokens = 64
    x = torch.zeros(dim, tokens, dtype=torch.bfloat16, device=device)
    conv_states = torch.zeros(
        1, dim, int(weight.shape[1]) - 1, dtype=torch.bfloat16, device=device
    )
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    cache_indices = torch.zeros(1, dtype=torch.int32, device=device)
    has_initial_state = torch.zeros(1, dtype=torch.bool, device=device)
    try:
        run(
            x=x,
            weight=weight,
            bias=None,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
        )
        torch.cuda.synchronize(device)
        _WARMED.add(key)
    except Exception:
        pass
