"""MLA's v absorb (``attn_output @ w_vc``, a head at a time) for a decode-sized
batch: one program a (head, 32-column block) over every token, fp32 accumulate,
bf16 out into [tokens, heads, v_head_dim] -- what ``torch.bmm`` writes there,
without hipBLASLt's tile for a 6-row problem."""

import torch
import triton
import triton.language as tl

MAX_TOKENS = 64


@triton.jit
def _v_absorb_kernel(
    X, W, Y, T,
    sx_t, sx_h,
    sw_h,
    sy_t, sy_h,
    K: tl.constexpr, N: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, G, FUSE_GATE: tl.constexpr, sw_k: tl.constexpr, sw_n: tl.constexpr, sg_t: tl.constexpr,
):  # fmt: skip
    h = tl.program_id(0)
    nb = tl.program_id(1)
    t = tl.program_id(2) * BLOCK_T + tl.arange(0, BLOCK_T)
    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X + t[:, None] * sx_t + h * sx_h + k[None, :],
            mask=t[:, None] < T,
            other=0.0,
        )
        w = tl.load(W + h * sw_h + k[:, None] * sw_k + n[None, :] * sw_n)
        acc = tl.dot(x, w, acc)
    value = acc.to(tl.bfloat16)
    if FUSE_GATE:
        gate = tl.load(
            G + t[:, None] * sg_t + h * sy_h + n[None, :], mask=t[:, None] < T, other=0
        ).to(tl.float32)
        sigmoid = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
        value = (value.to(tl.float32) * sigmoid.to(tl.float32)).to(tl.bfloat16)
    tl.store(
        Y + t[:, None] * sy_t + h * sy_h + n[None, :],
        value,
        mask=t[:, None] < T,
    )


def covered(attn_output: torch.Tensor, w_vc: torch.Tensor) -> bool:
    return (
        attn_output.dim() == 3
        and attn_output.dtype == torch.bfloat16
        and attn_output.stride(-1) == 1
        and 0 < attn_output.shape[0] <= MAX_TOKENS
        and w_vc.dtype == torch.bfloat16
        and w_vc.is_contiguous()
        and w_vc.shape[0] == attn_output.shape[1]
        and w_vc.shape[1] == attn_output.shape[2]
        and w_vc.shape[1] % 64 == 0
        and w_vc.shape[2] % 32 == 0
    )


def mla_v_absorb(
    attn_output: torch.Tensor, w_vc: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """attn_output [T, H, K] bf16 (last dim contiguous), w_vc [H, K, N] bf16,
    out [T, H, N] bf16 (last dim contiguous)."""
    T, H, K = attn_output.shape
    N = w_vc.shape[2]
    block_t = max(16, triton.next_power_of_2(T))
    _v_absorb_kernel[(H, N // 32)](
        attn_output, w_vc, out, T,
        attn_output.stride(0), attn_output.stride(1),
        w_vc.stride(0),
        out.stride(0), out.stride(1),
        K=K, N=N, BLOCK_T=block_t, BLOCK_N=32, BLOCK_K=64,
        G=out, FUSE_GATE=False, sw_k=w_vc.stride(1), sw_n=w_vc.stride(2), sg_t=out.stride(0),
    )  # fmt: skip
    return out


def mla_v_absorb_gate(
    attn_output: torch.Tensor, w_vc: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """V projection + BF16-rounded sigmoid/mul, directly token-major.

    Extends the decode V-absorb kernel to tiled M (including target verify).
    Preserves both BF16 rounding boundaries of bmm -> sigmoid -> mul.
    """
    t, h, k = attn_output.shape
    n = w_vc.shape[2]
    assert attn_output.dtype == w_vc.dtype == gate.dtype == torch.bfloat16
    assert w_vc.shape[:2] == (h, k) and k % 64 == 0 and n % 32 == 0
    assert attn_output.stride(-1) == 1
    assert gate.shape == (t, h * n) and gate.stride(-1) == 1
    out = torch.empty((t, h, n), dtype=attn_output.dtype, device=attn_output.device)
    if t:
        _v_absorb_kernel[(h, n // 32, triton.cdiv(t, 16))](
            attn_output,
            w_vc,
            out,
            t,
            attn_output.stride(0),
            attn_output.stride(1),
            w_vc.stride(0),
            out.stride(0),
            out.stride(1),
            K=k,
            N=n,
            BLOCK_T=16,
            BLOCK_N=32,
            BLOCK_K=64,
            G=gate,
            FUSE_GATE=True,
            sw_k=w_vc.stride(1),
            sw_n=w_vc.stride(2),
            sg_t=gate.stride(0),
        )
    return out.flatten(1)


@triton.jit
def _q_absorb_cat(
    Q,
    W,
    Y,
    M,
    SQ: tl.constexpr,
    SH: tl.constexpr,
    WH: tl.constexpr,
    WK: tl.constexpr,
    WN: tl.constexpr,
):
    h = tl.program_id(0)
    nb = tl.program_id(1)
    m = tl.program_id(2) * 16 + tl.arange(0, 16)
    n = nb * 32 + tl.arange(0, 32)
    k = tl.arange(0, 64)
    acc = tl.zeros((16, 32), tl.float32)
    for ki in range(2):
        kk = ki * 64 + k
        q = tl.load(Q + m[:, None] * SQ + h * SH + kk[None, :], m[:, None] < M, 0)
        w = tl.load(W + h * WH + kk[:, None] * WK + n[None, :] * WN)
        acc = tl.dot(q, w, acc)
    tl.store(
        Y + m[:, None] * (12 * 576) + h * 576 + n[None, :],
        acc.to(tl.bfloat16),
        m[:, None] < M,
    )
    if nb == 0:
        r = tl.arange(0, 64)
        tail = tl.load(
            Q + m[:, None] * SQ + h * SH + 128 + r[None, :], m[:, None] < M, 0
        )
        tl.store(
            Y + m[:, None] * (12 * 576) + h * 576 + 512 + r[None, :],
            tail,
            m[:, None] < M,
        )


def mla_q_absorb_cat(q: torch.Tensor, w_kc: torch.Tensor) -> torch.Tensor:
    """K3.1 Q absorb and passthrough tail in one token-major [M,12,576] store."""
    m = q.shape[0]
    assert q.shape == (m, 12, 192) and w_kc.shape == (12, 128, 512)
    assert q.dtype == w_kc.dtype == torch.bfloat16 and q.stride(-1) == 1
    out = q.new_empty((m, 12, 576))
    if m:
        _q_absorb_cat[(12, 16, triton.cdiv(m, 16))](
            q, w_kc, out, m, q.stride(0), q.stride(1), *w_kc.stride(), num_warps=4
        )
    return out
