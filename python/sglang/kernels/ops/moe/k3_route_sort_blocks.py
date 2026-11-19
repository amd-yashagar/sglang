# SPDX-License-Identifier: Apache-2.0
"""K3 V2: bounded-contention routing and block-coalesced scale sorting.

Launch A routes tokens and quantizes independent chunks in parallel. Expert
membership uses one bit per token (at most 32 writers per bitmap word), without
the contended expert counters. Launch B computes the expert prefix once, saves
membership and clears the bitmap. Launch C scatters one coalesced packed scale
block per CTA, rather than recomputing the expert prefix per token.

This standalone API retains V1 for M<=128 (M<192 at N=3584).
No serving dispatch change.
SGLANG_K3_ROUTE_SORT_FAST=1 is required. Distinct workspace_key values are
required for overlapping streams/graphs. FP8 codes remain per-token.
"""

import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.k3_route_sort import _mx_e8m0
from sglang.kernels.ops.moe.k3_route_sort_fast import route_sort_quant as _v1
from sglang.kernels.ops.moe.k3_route_sort_mid import _aiter_biased_topk

_BITMAPS = {}


@triton.jit(do_not_specialize=["M", "stride_lg", "stride_x"])
def _route_quant_membership(
    logits,
    bias,
    x,
    ids_out,
    weights_out,
    bitmap,
    slots,
    qout,
    exps,
    buf,
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
        tl.store(slots + ids * M + t, k.to(tl.uint8))
        tl.atomic_or(
            bitmap + (t // 32) * 896 + ids, (1 << (t % 32)).to(tl.uint32), sem="relaxed"
        )
    else:
        qp = pid - M
        t = qp // tl.cdiv(N, CHUNK)
        c0 = (qp % tl.cdiv(N, CHUNK)) * CHUNK
        c = c0 + tl.arange(0, CHUNK)
        v = tl.load(x + t * stride_x + c, c < N, other=0).to(tl.float32)
        v2 = tl.reshape(v, (CHUNK // 32, 32))
        exp, scale = _mx_e8m0(v2)
        q = tl.clamp(v2 / scale[:, None], -448.0, 448.0)
        tl.store(
            qout + t * N + c, tl.reshape(q, (CHUNK,)).to(qout.dtype.element_ty), c < N
        )
        tl.store(buf + t * N + c, 0, c < N)
        g = c0 // 32 + tl.arange(0, CHUNK // 32)
        tl.store(exps + t * (N // 32) + g, exp.to(tl.uint8), g < N // 32)


@triton.jit
def _popc(x):
    return tl.inline_asm_elementwise(
        "v_bcnt_u32_b32 $0, $1, 0",
        "=v,v",
        [x.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.int32)


@triton.jit(do_not_specialize=["M"])
def _expert_prefix(bitmap, snapshot, meta, se, nv, M, MP: tl.constexpr):
    e = tl.arange(0, 1024)
    word = tl.arange(0, MP)
    ptr = word[:, None] * 896 + e[None, :]
    mask = (e[None, :] < 896) & (word[:, None] < tl.cdiv(M, 32))
    bits = tl.load(bitmap + ptr, mask, other=0)
    cnt = tl.sum(_popc(bits), axis=0)
    tl.store(snapshot + ptr, bits, mask)
    # This is the sole source reader/resetter. A CTA barrier finishes reads
    # before reset; stream ordering keeps the next routing call after reset.
    tl.debug_barrier()
    tl.store(bitmap + ptr, 0, mask)
    nb = (cnt + 31) // 32
    bb = tl.cumsum(nb, axis=0) - nb
    tl.store(meta + e, cnt, e < 896)
    tl.store(meta + 896 + e, bb, e < 896)
    # Stream metadata: avoid materializing a 1024xMAXB tensor.
    for j in range(tl.max(nb, axis=0)):
        tl.store(se + bb + j, e, (e < 896) & (j < nb))
    two = tl.arange(0, 2)
    tl.store(nv + two, tl.where(two == 0, tl.sum(nb, axis=0) * 32, M))


@triton.jit(do_not_specialize=["M"])
def _scatter_blocks(
    weights,
    slots,
    exps,
    snapshot,
    meta,
    sids,
    sw,
    se,
    nv,
    scales,
    M,
    G: tl.constexpr,
    GP: tl.constexpr,
    MP: tl.constexpr,
):
    block = tl.program_id(0)
    if block < tl.load(nv) // 32:
        e = tl.load(se + block)
        cnt = tl.load(meta + e)
        base = tl.load(meta + 896 + e)
        r = (block - base) * 32 + tl.arange(0, 32)
        live = r < cnt
        word = tl.arange(0, MP)
        bits = tl.load(snapshot + word * 896 + e, word < tl.cdiv(M, 32), other=0).to(
            tl.uint32
        )
        count = _popc(bits)
        pre = tl.cumsum(count, axis=0) - count
        owns = (r[:, None] >= pre[None, :]) & (r[:, None] < (pre + count)[None, :])
        wi = tl.sum(tl.where(owns, word[None, :], 0), axis=1)
        value = tl.gather(bits, wi, axis=0)
        nth = r - tl.gather(pre, wi, axis=0)
        bit = tl.zeros((32,), tl.int32)
        for shift in tl.static_range(4, -1, -1):
            width = 1 << shift
            lower = _popc(value & ((1 << width) - 1))
            upper = nth >= lower
            nth -= tl.where(upper, lower, 0)
            value = tl.where(upper, value >> width, value)
            bit += tl.where(upper, width, 0)
        token = wi * 32 + bit
        slot = tl.load(slots + e * M + token, live, other=0).to(tl.int32)
        w = tl.load(weights + token * 16 + slot, live, other=0.0)
        row = tl.arange(0, 32)
        tl.store(
            sids + block * 32 + row,
            tl.where(live, (slot << 24) | token, (16 << 24) | M),
        )
        tl.store(sw + block * 32 + row, w)

        # Four adjacent bytes: (row,g), (row+16,g), (row,g+4), (row+16,g+4).
        # Pack them into aligned words, including arbitrary zero padding bytes.
        rr = tl.arange(0, 16)
        lo = tl.gather(token, rr, axis=0)
        hi = tl.gather(token, rr + 16, axis=0)
        ml = tl.gather(live, rr, axis=0)
        mh = tl.gather(live, rr + 16, axis=0)
        gp = tl.arange(0, GP)
        g = (gp // 4) * 8 + gp % 4
        a = tl.load(
            exps + lo[None, :] * G + g[:, None], ml[None, :] & (g[:, None] < G), other=0
        ).to(tl.uint32)
        b = tl.load(
            exps + hi[None, :] * G + g[:, None], mh[None, :] & (g[:, None] < G), other=0
        ).to(tl.uint32)
        c = tl.load(
            exps + lo[None, :] * G + g[:, None] + 4,
            ml[None, :] & (g[:, None] + 4 < G),
            other=0,
        ).to(tl.uint32)
        d = tl.load(
            exps + hi[None, :] * G + g[:, None] + 4,
            mh[None, :] & (g[:, None] + 4 < G),
            other=0,
        ).to(tl.uint32)
        packed = a | (b << 8) | (c << 16) | (d << 24)
        addr = (
            block * (G * 8)
            + (g // 8)[:, None] * 64
            + (g % 4)[:, None] * 16
            + rr[None, :]
        )
        tl.store(scales.to(tl.pointer_type(tl.uint32)) + addr, packed, g[:, None] < G)


def route_sort_quant(
    logits,
    bias,
    x,
    ids,
    weights,
    out,
    block_size=32,
    *,
    workspace_key="hist",
):
    """Opt-in standalone V2. Same K3 numerical/layout contract as V1."""
    if os.environ.get("SGLANG_K3_ROUTE_SORT_FAST") != "1":
        raise RuntimeError("Enable experimental glue with SGLANG_K3_ROUTE_SORT_FAST=1")
    m, n = x.shape
    if m <= 128 or (n == 3584 and m < 192):
        return _v1(
            logits, bias, x, ids, weights, out, block_size, workspace_key=workspace_key
        )
    assert 0 < m <= 320 and n in (3584, 7168) and block_size == 32
    assert (
        logits.dtype in (torch.bfloat16, torch.float32) and logits.dtype == bias.dtype
    )
    assert logits.shape == (m, 896) and bias.shape == (896,) and bias.is_contiguous()
    assert x.dtype == torch.bfloat16 and x.stride(1) == logits.stride(1) == 1
    assert (
        ids.shape == weights.shape == (m, 16)
        and ids.dtype == torch.int32
        and weights.dtype == torch.float32
    )
    assert ids.is_contiguous() and weights.is_contiguous()
    assert all(t.device == x.device for t in (logits, bias, ids, weights, *out))
    assert (
        out[4].shape == (m, n)
        and out[4].dtype == torch.bfloat16
        and out[4].is_contiguous()
    )
    assert out[0].dtype == out[2].dtype == out[3].dtype == torch.int32
    assert out[1].dtype == torch.float32 and all(t.is_contiguous() for t in out)
    assert out[0].numel() >= ((m * 16 + 896 * 32 - 16 + 31) // 32) * 32
    assert (
        out[1].numel() == out[0].numel()
        and out[2].numel() >= out[0].numel() // 32
        and out[3].numel() >= 2
    )
    mp = triton.next_power_of_2(triton.cdiv(m, 32))
    key = (str(x.device), workspace_key, mp)
    bitmap = _BITMAPS.get(key)
    if bitmap is None:
        bitmap = torch.zeros((mp, 896), dtype=torch.int32, device=x.device)
        _BITMAPS[key] = bitmap
    snapshot = torch.empty_like(bitmap)
    slots = torch.empty((896, m), dtype=torch.uint8, device=x.device)
    meta = torch.empty(1792, dtype=torch.int32, device=x.device)
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    exps = torch.empty((m, n // 32), dtype=torch.uint8, device=x.device)
    scale = torch.empty((out[0].numel(), n // 32), dtype=torch.uint8, device=x.device)
    _route_quant_membership[(m * (1 + triton.cdiv(n, 512)),)](
        logits,
        bias,
        x,
        ids,
        weights,
        bitmap,
        slots,
        q,
        exps,
        out[4],
        m,
        logits.stride(0),
        x.stride(0),
        N=n,
        MP=mp,
        num_warps=4,
    )
    _expert_prefix[(1,)](
        bitmap,
        snapshot,
        meta,
        out[2],
        out[3],
        m,
        MP=mp,
        num_warps=4,
    )
    _scatter_blocks[(out[2].numel(),)](
        weights,
        slots,
        exps,
        snapshot,
        meta,
        *out[:4],
        scale,
        m,
        G=n // 32,
        GP=triton.next_power_of_2(n // 64),
        MP=mp,
        num_warps=4,
    )
    return q, scale.view(torch.float8_e8m0fnu)
