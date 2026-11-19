"""Opt-in K3.1 BF16 MLA front: independent Q/KV RMSNorm and KV scatter.

K3 sets skip_rope=True: the 64 tail channels are copied, not rotated.
The cache write rounds normalized KV to BF16 *before* an optional raw FP8 cast.
Packed/scaled FP8, sharded, remapped and window pools retain their normal writer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_cache(
    Q,
    K,
    QW,
    KW,
    QO,
    KO,
    CACHE,
    LOC,
    qs: tl.constexpr,
    ks: tl.constexpr,
    cs: tl.constexpr,
    QE: tl.constexpr,
    KE: tl.constexpr,
    CAP: tl.constexpr,
):
    row = tl.program_id(0)
    branch = tl.program_id(1)
    if branch == 0:
        c = tl.arange(0, 2048)
        x = tl.load(Q + row * qs + c, c < 1536, 0).to(tl.float32)
        w = tl.load(QW + c, c < 1536, 0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 0) / 1536 + QE) * w
        tl.store(QO + row * 1536 + c, y, c < 1536)
    else:
        kc = tl.arange(0, 512)
        kx = tl.load(K + row * ks + kc).to(tl.float32)
        kw = tl.load(KW + kc).to(tl.float32)
        ky = (kx * tl.rsqrt(tl.sum(kx * kx, 0) / 512 + KE) * kw).to(tl.bfloat16)
        tl.store(KO + row * 576 + kc, ky)
        loc = tl.load(LOC + row)
        # Slot zero is graph padding. Never race padding rows against each other.
        valid = (loc > 0) & (loc < CAP)
        tl.store(CACHE + loc * cs + kc, ky, valid)
        r = tl.arange(0, 64)
        tail = tl.load(K + row * ks + 512 + r)
        tl.store(KO + row * 576 + 512 + r, tail)
        tl.store(CACHE + loc * cs + 512 + r, tail, valid)


def norm_cache(q, kv, qw, kw, cache, loc, qeps=1e-5, keps=1e-5):
    """Return normalized Q [M,1536], packed KV [M,576]; scatter live rows.

    loc contains physical, unique positive live slots; zero/negative means padding.
    Caller owns pool-layout eligibility. No allocation proportional to cache size.
    """
    m = q.shape[0]
    assert q.shape == (m, 1536) and kv.shape == (m, 576)
    assert q.dtype == kv.dtype == qw.dtype == kw.dtype == torch.bfloat16
    assert qw.shape == (1536,) and kw.shape == (512,)
    assert q.stride(-1) == kv.stride(-1) == 1
    assert qw.is_contiguous() and kw.is_contiguous()
    assert cache.shape[-1] == 576 and cache.is_contiguous()
    assert cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert loc.shape == (m,) and loc.is_contiguous()
    assert loc.dtype in (torch.int32, torch.int64)
    qo = q.new_empty((m, 1536))
    ko = kv.new_empty((m, 576))
    if m:
        _norm_cache[(m, 2)](
            q,
            kv,
            qw,
            kw,
            qo,
            ko,
            cache,
            loc,
            q.stride(0),
            kv.stride(0),
            576,
            qeps,
            keps,
            cache.numel() // 576,
            num_warps=4,
        )
    return qo, ko


def try_norm_cache(attn, q, latent, batch):
    """Conservative serving guard; None preserves every unsupported writer."""
    if not getattr(attn, "_k31_mla_fuse", False):
        return None
    if not (
        batch.forward_mode.is_decode_or_idle() or batch.forward_mode.is_target_verify()
    ):
        return None
    if not (
        0 < q.shape[0] <= 320
        and attn.dcp_size == 1
        and attn.current_attention_backend in ("dsa", "aiter")
        and attn.rotary_emb is None
        and attn.q_lora_rank == 1536
        and attn.kv_lora_rank == 512
        and attn.qk_rope_head_dim == 64
        and q.dtype == torch.bfloat16
        and attn.q_b_proj.weight.dtype == torch.bfloat16
    ):
        return None
    if not (
        q.shape == (q.shape[0], 1536)
        and latent.shape == (q.shape[0], 576)
        and latent.dtype == torch.bfloat16
        and q.stride(-1) == latent.stride(-1) == 1
        and batch.out_cache_loc.shape == (q.shape[0],)
        and batch.out_cache_loc.is_contiguous()
        and batch.out_cache_loc.dtype in (torch.int32, torch.int64)
        and attn.q_a_layernorm.weight.dtype
        == attn.kv_a_layernorm.weight.dtype
        == torch.bfloat16
        and attn.q_a_layernorm.weight.is_contiguous()
        and attn.kv_a_layernorm.weight.is_contiguous()
    ):
        return None
    from sglang.srt.mem_cache.kv.layout.mla import LayerMajorMLAView, MLAView
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
    from sglang.srt.model_executor.forward_context import get_kv_layers

    handle = get_kv_layers().layer(attn.attn_mqa.layer_id)
    pool = handle.view
    # Legacy plain pools resolve their own write ids; with DCP=1 these are
    # already physical. Native owner views also take physical ids, but only
    # when the iteration explicitly marked them so. Never bypass a window,
    # draft, remapped or composite view whose ids need a different translation.
    legacy = (
        handle.plain
        and type(pool) in (DSATokenToKVPool, MLATokenToKVPool)
        and not pool.write_loc_is_dcp_resolved
    )
    native = (
        not handle.plain
        and handle.owner
        and type(pool) in (MLAView, LayerMajorMLAView)
        and batch.out_cache_loc_is_physical
    )
    if (
        not (legacy or native)
        or pool.dsa_kv_cache_store_fp8
        or pool.dtype != torch.bfloat16
    ):
        return None
    cache = handle.key()
    if not (
        cache.is_contiguous()
        and cache.shape[-1] == 576
        and cache.dtype == torch.bfloat16
    ):
        return None
    return norm_cache(
        q,
        latent,
        attn.q_a_layernorm.weight,
        attn.kv_a_layernorm.weight,
        cache,
        batch.out_cache_loc,
        attn.q_a_layernorm.variance_epsilon,
        attn.kv_a_layernorm.variance_epsilon,
    )
