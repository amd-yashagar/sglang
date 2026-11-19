# SPDX-License-Identifier: Apache-2.0
"""K3 routing glue with coalesced, packed membership stable ranks.

Launch A runs routing and independent quant chunks side by side. Launch B reuses
PR46656 sorting/scale scattering, but ranks the TOPK pairs with one bit lookup
per 32 preceding tokens instead of TOPK expert-id comparisons per token.
M=1 uses a barrier-free single-launch route/sort/quant/zero path.
The serving model is deliberately not switched to this experimental API.
"""

import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe import k3_route_sort_mid as mid
from sglang.kernels.ops.moe.k3_route_sort import _mx_e8m0, _mx_scale_col, _mx_scale_row
from sglang.kernels.ops.moe.k3_route_sort_mid import _aiter_biased_topk

_BITMAPS = {}


@triton.jit(do_not_specialize=["M", "stride_lg", "stride_x"])
def _route_quant_parallel(
    logits,
    bias,
    x,
    ids_out,
    weights_out,
    counts,
    bitmap,
    qout,
    exps,
    M,
    stride_lg,
    stride_x,
    N: tl.constexpr,
    MP: tl.constexpr,
    CHUNK: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    if pid < M:
        t = pid
        e = tl.arange(0, 1024)
        ids, w = _aiter_biased_topk(
            logits + t * stride_lg, bias, e, e < 896, 16, True, 1.0
        )
        k = tl.arange(0, 16)
        tl.store(ids_out + t * 16 + k, ids)
        tl.store(weights_out + t * 16 + k, w)
        tl.atomic_or(
            bitmap + ids * MP + t // 32, (1 << (t % 32)).to(tl.uint32), sem="relaxed"
        )
        tl.atomic_add(counts + ids, 1, sem="relaxed")
    else:
        qp = pid - M
        t = qp // tl.cdiv(N, CHUNK)
        c = (qp % tl.cdiv(N, CHUNK)) * CHUNK + tl.arange(0, CHUNK)
        v = tl.load(x + t * stride_x + c, c < N, other=0).to(tl.float32)
        v2 = tl.reshape(v, (CHUNK // 32, 32))
        exp, scale = _mx_e8m0(v2)
        q = tl.clamp(v2 / scale[:, None], -448.0, 448.0)
        tl.store(
            qout + t * N + c, tl.reshape(q, (CHUNK,)).to(qout.dtype.element_ty), c < N
        )
        g = (qp % tl.cdiv(N, CHUNK)) * (CHUNK // 32) + tl.arange(0, CHUNK // 32)
        tl.store(exps + t * (N // 32) + g, exp.to(tl.uint8), g < N // 32)


@triton.jit
def _single_route_sort_quant(
    logits,
    bias,
    x,
    ids_out,
    w_out,
    sids,
    sw,
    se,
    nv,
    buf,
    qout,
    qs,
    N: tl.constexpr,
    SP: tl.constexpr,
    CHUNK: tl.constexpr = 512,
):
    e = tl.arange(0, 1024)
    ids, w = _aiter_biased_topk(logits, bias, e, e < 896, 16, True, 1.0)
    k = tl.arange(0, 16)
    # With one token, the expert order alone defines every destination.
    block = tl.sum((ids[None, :] < ids[:, None]).to(tl.int32), axis=1)
    dest = block * 32
    if tl.program_id(0) == 0:
        tl.store(ids_out + k, ids)
        tl.store(w_out + k, w)
        b = tl.arange(0, 32)
        at = dest[:, None] + b[None, :]
        tl.store(sids + at, tl.where(b[None, :] == 0, k[:, None] << 24, (16 << 24) | 1))
        tl.store(sw + at, tl.where(b[None, :] == 0, w[:, None], 0.0))
        tl.store(se + block, ids)
        two = tl.arange(0, 2)
        tl.store(nv + two, tl.where(two == 0, 512, 1))
    c = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    v = tl.load(x + c, c < N, other=0).to(tl.float32)
    v2 = tl.reshape(v, (CHUNK // 32, 32))
    exp, scale = _mx_e8m0(v2)
    q = tl.clamp(v2 / scale[:, None], -448.0, 448.0)
    tl.store(qout + c, tl.reshape(q, (CHUNK,)).to(qout.dtype.element_ty), c < N)
    tl.store(buf + c, 0, c < N)
    g = tl.program_id(0) * (CHUNK // 32) + tl.arange(0, CHUNK // 32)
    addr = _mx_scale_row(dest, SP)[:, None] + _mx_scale_col(g)[None, :]
    tl.store(
        qs + addr,
        tl.broadcast_to(exp[None, :], (16, CHUNK // 32)).to(tl.uint8),
        g[None, :] < N // 32,
    )


def route_sort_quant(
    logits, bias, x, ids, weights, out, block_size=32, *, workspace_key="fast"
):
    """Experimental opt-in standalone glue; production dispatch is unchanged.

    Set SGLANG_K3_ROUTE_SORT_FAST=1. Use distinct workspace_key values for
    overlapping streams/graphs. Calls sharing a key must execute serially.
    Finite K3 sigmoid+bias logits, top-16, renormalize=True, scale=1, no EP mask.
    out is the existing aiter five-tensor sort tuple with a full [M,N] moe_buf.
    """
    if os.environ.get("SGLANG_K3_ROUTE_SORT_FAST") != "1":
        raise RuntimeError("Enable experimental glue with SGLANG_K3_ROUTE_SORT_FAST=1")
    m, n = x.shape
    assert 0 < m <= 320 and n in (3584, 7168) and block_size == 32
    assert (
        logits.dtype in (torch.bfloat16, torch.float32) and bias.dtype == logits.dtype
    )
    assert x.dtype == torch.bfloat16 and x.stride(1) == logits.stride(1) == 1
    assert bias.is_contiguous() and bias.shape == (896,)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert (
        weights.shape == ids.shape
        and weights.device == ids.device == x.device == logits.device == bias.device
    )
    assert ids.is_contiguous() and weights.is_contiguous()
    assert out[4].shape == (m, n) and out[4].is_contiguous()
    assert logits.shape == (m, 896) and ids.shape == (m, 16)
    if m == 1:
        q = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
        sp = triton.cdiv(n // 32, 8) * 8
        scale = torch.empty((out[0].numel(), sp), device=x.device, dtype=torch.uint8)
        _single_route_sort_quant[(triton.cdiv(n, 512),)](
            logits, bias, x, ids, weights, *out, q, scale, N=n, SP=sp, num_warps=4
        )
        return q, scale.view(torch.float8_e8m0fnu)
    counts = mid.counts_buffer(x.device, workspace_key, 896)
    qout = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
    exps = torch.empty((m, n // 32), device=x.device, dtype=torch.uint8)
    words = triton.next_power_of_2(triton.cdiv(m, 32))
    key = (str(x.device), workspace_key, words)
    bitmap = _BITMAPS.get(key)
    if bitmap is None:
        bitmap = torch.zeros((896, words), device=x.device, dtype=torch.int32)
        _BITMAPS[key] = bitmap
    _route_quant_parallel[(m * (1 + triton.cdiv(n, 512)),)](
        logits,
        bias,
        x,
        ids,
        weights,
        counts,
        bitmap,
        qout,
        exps,
        m,
        logits.stride(0),
        x.stride(0),
        N=n,
        MP=words,
        num_warps=4,
    )
    state = mid.RouteState(ids, weights, x, qout, exps, counts, bitmap=bitmap)
    return mid.sort_scatter(state, *out, block_size, 896)
