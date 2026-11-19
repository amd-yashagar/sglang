"""Fused KDA chain verify with ReplaySSM or per-step snapshots on gfx950.

The bias-free conv and SSM state share ``cache_indices``; intermediate windows
and snapshots use ``intermediate_state_indices``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(
    bv: int, t: int, lower_bound: bool, snap: bool, optimized: bool = False
) -> Module:
    args = make_cpp_args(bv, t, lower_bound, snap, optimized)
    return load_jit(
        "kda_verify_gfx950",
        *args,
        cuda_files=["attention/kda_verify_gfx950.cuh"],
        cuda_wrappers=[("run", f"KdaVerifyGfx950Kernel<{args}>::run")],
    )


def _octet_rows(t: torch.Tensor) -> torch.Tensor:
    # the kernel reads 8 bf16 channels per 16-byte load, so rows must be 16-byte aligned
    if t.stride(1) == 1 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0:
        return t
    return t.contiguous()


@cache_once
def _no_ring(device: torch.device) -> tuple[torch.Tensor, ...]:
    # Zero-length ring views for the snapshot variant: ring_len 0 disables every ring store.
    rows_bf16 = torch.empty((1, 1, 0, 128), dtype=torch.bfloat16, device=device)
    rows_f32 = torch.empty((1, 1, 0, 128), dtype=torch.float32, device=device)
    beta = torch.empty((1, 1, 0), dtype=torch.float32, device=device)
    return rows_bf16, rows_bf16, rows_f32, beta


def supports_kda_verify_gfx950(
    *,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    T: int,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    replayssm_rawv: Optional[torch.Tensor],
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_conv_window: Optional[torch.Tensor],
) -> bool:
    ring = (
        replayssm_rawv is not None
        and intermediate_states_buffer is None
        and replayssm_rawv.dtype == torch.bfloat16
    )
    snapshots = (
        replayssm_rawv is None
        and intermediate_states_buffer is not None
        and intermediate_states_buffer.dtype == torch.float32
        and intermediate_states_buffer.is_contiguous()
        and intermediate_states_buffer.shape[1] >= T
    )
    return (
        (ring or snapshots)
        and intermediate_conv_window is not None
        and num_q_heads == num_v_heads
        and head_k_dim == head_v_dim == 128
        and 2 <= T <= 8  # the block maps 8 token groups; groups past T idle
        and mixed_qkv.dtype == torch.bfloat16
        and a.dtype == b.dtype == torch.bfloat16
        and conv_weight.dtype == torch.float32
        and conv_weight.dim() == 2
        and conv_weight.shape[1] == 4
        # the kernel reads the conv state and window as BF16 without checking
        and conv_state.dtype == intermediate_conv_window.dtype == torch.bfloat16
        and ssm_states.dtype == torch.float32
    )


def use_full_head_verify(
    *, batch_size: int, T: int, heads: int, snap: bool, with_norm: bool
) -> bool:
    # Cold-state, 69-layer graphs: a full head underfills gfx950 at c1; at c8
    # it only wins when it also consumes the norm. Preserve those fallbacks.
    return (
        envs.SGLANG_OPT_KDA_VERIFY_GFX950.get()
        and T == 5
        and heads == 12
        and not snap
        and (batch_size >= 16 or (with_norm and batch_size >= 8))
    )


def kda_verify_gfx950(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_states_buffer: Optional[torch.Tensor],
    scale: float,
    T: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: Optional[float] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    norm_gate: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Returns ``o`` of shape [1, seq_len, HV, V]."""
    snap = intermediate_states_buffer is not None
    assert use_qk_l2norm_in_kernel, "the kernel always l2-normalizes q and k"
    assert cache_ring != snap, "needs exactly one of the ReplaySSM ring and snapshots"
    if snap:
        replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta = _no_ring(
            mixed_qkv.device
        )
    seq_len = mixed_qkv.shape[0]
    HV, V = num_v_heads, head_v_dim
    batch_size = seq_len // T
    # Baseline T=8 mapping is unchanged. For the new T=5 full-head layout,
    # 16 waves (BV=8) favour latency, 8 waves (BV=16) favour c64 throughput.
    optimized = use_full_head_verify(
        batch_size=batch_size, T=T, heads=HV, snap=snap, with_norm=norm_gate is not None
    )
    bv = (8 if batch_size <= 16 else 16) if optimized else (4 if batch_size <= 2 else 8)
    if norm_gate is not None:
        assert optimized, "fused norm requires the T=5 full-head ring kernel"
        assert norm_gate.dtype == torch.bfloat16 and norm_gate.shape == (
            seq_len,
            HV * V,
        )
        assert norm_gate.stride(1) == 1
        assert norm_weight is not None and norm_weight.shape == (V,)
        assert (
            norm_weight.dtype in (torch.float32, torch.bfloat16)
            and norm_weight.is_contiguous()
        )
        assert norm_gate.device == norm_weight.device == mixed_qkv.device
    o = mixed_qkv.new_empty(seq_len, HV, V)
    module = _jit_module(bv, T, lower_bound is not None, snap, optimized)
    module.run(
        _octet_rows(mixed_qkv),
        conv_weight,
        conv_state,
        cache_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        _octet_rows(a.reshape(seq_len, HV * head_k_dim)),
        _octet_rows(b.reshape(seq_len, HV)),
        A_log,
        dt_bias,
        ssm_states,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        intermediate_states_buffer,
        norm_gate,
        norm_weight,
        float(norm_eps),
        o,
        num_q_heads,
        float(scale),
        float(lower_bound) if lower_bound is not None else 0.0,
        float(softplus_beta),
        float(softplus_threshold),
    )
    return o.view(1, seq_len, HV, V)
