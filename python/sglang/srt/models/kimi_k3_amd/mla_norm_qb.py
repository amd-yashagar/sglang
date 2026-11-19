# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""M5 q-RMSNorm + q_b, KV-RMSNorm/tail/cache in one gfx950 launch.

The wave-private DMA ring and MFMA fragment layout are adapted from AITERs
MIT-licensed gemm_a16w16_skinny_gfx950. Here each wave first normalizes rows,
then applies gamma/RMS and the BF16 boundary as fragments leave LDS. An extra
workgroup handles KV; no grid barrier, atomic counter, or semaphore is used.
Only the real TP8 BF16 M5 geometry is supported. Other shapes use v1.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    _run_compiled,
)
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import ReductionOp, T

WAVE_SIZE = 64
# bf16 elements in one 16-byte lane of a global->LDS DMA
DMA_VEC = 8
# MFMA 16x16x32: K elements consumed per step
MFMA_K = 32
GFX950_NUM_XCDS = 8
# fp32 row pad of the row-major partial reduce: puts the four MFMA lane groups
# in distinct LDS banks
REDUCE_ROW_PAD = 4


@functools.lru_cache(maxsize=256)
def _compile_norm_qb(a_stride: int, loc64: bool = True):
    n, k = 2304, 1536
    block_m, block_n, block_k, stages, k_waves = 8, 16, 128, 3, 4
    TA, TN, KC, K_WAVES = block_m, block_n, block_k, k_waves
    # TA = A rows staged; TM = MFMA rows (TA rounded up to 16). MFMA rows in
    # [TA, TM) read other LDS rows, but they only feed C rows >= m, which the
    # C descriptor bounds drop.
    MF = (TA + 15) // 16
    NF = (TN + 15) // 16
    N_BLOCKS = n // TN
    KS = k // K_WAVES
    CHUNKS = KS // KC
    ST = min(stages, CHUNKS)
    CPR = KC // DMA_VEC  # 16-byte chunks per staged row
    ROWS_PER_INST = WAVE_SIZE // CPR
    R = TA + TN  # rows per staged chunk: A rows, then W rows
    IPC = R // ROWS_PER_INST  # DMA instructions per chunk
    A_INSTS = TA // ROWS_PER_INST
    STEPS = KC // MFMA_K
    STAGE_ELEMS = R * KC
    WAVE_ELEMS = ST * STAGE_ELEMS
    BLOCK_THREADS = K_WAVES * WAVE_SIZE
    LDS_ELEMS = K_WAVES * WAVE_ELEMS
    # Operand-swapped MFMA (C^T tile) with accumulator-order partials: one MFMA
    # row tile only.
    SWAPPED = TA <= 16
    # (mf, 4-column group, row) output slots of the swapped epilogue, one per thread
    EPI_SLOTS = MF * (TN // 4) * 16
    ROW_GROUP = max(1, 16 // CPR)  # rows sharing one 256-byte bank line
    XCD = GFX950_NUM_XCDS if N_BLOCKS % GFX950_NUM_XCDS == 0 else 0

    def swz(r, c):
        return c ^ ((r // ROW_GROUP) % CPR)

    @fx.struct
    class SkinnyLdsShared:
        buf: fx.Array[fx.BFloat16, LDS_ELEMS, 16]
        rms: fx.Array[fx.Float32, TA, 16]

    KERNEL_NAME = f"k31_norm_qb_t{TA}x{TN}x{KC}x{ST}_w1x1x{K_WAVES}"

    def mfma(a_fr, b_fr, acc):
        # raw intrinsic: the fragments come straight out of the swizzled LDS
        # ring with no tensor/partition form for fx.make_mma_atom to take
        return rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_fr, b_fr, acc, 0, 0, 0])

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def norm_qb_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        QW: fx.Pointer,
        KW: fx.Pointer,
        QO: fx.Pointer,
        KO: fx.Pointer,
        CACHE: fx.Pointer,
        LOC: fx.Pointer,
        cap: fx.Int32,
        m: fx.Int32,
    ):
        if fx.block_idx.x < N_BLOCKS:
            acc_init = fx.Vector.filled(4, 0.0, fx.Float32)
            a_bytes = (fx.Int64(m) - 1) * (a_stride * 2) + k * 2
            c_bytes = fx.Int64(m) * (n * 2)
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(A)), num_records_bytes=a_bytes
            )
            c_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(C)), num_records_bytes=c_bytes
            )
            lds = fx.SharedAllocator().allocate(SkinnyLdsShared).peek()
            buf_ptr = lds.buf.ptr
            red_ptr = fx.recast_iter(fx.Float32, buf_ptr)
            rms_ptr = lds.rms.ptr
            qw_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(QW)), num_records_bytes=k * 2
            )
            qo_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(QO)), num_records_bytes=fx.Int64(m) * (k * 2)
            )

            tid = fx.Index(fx.thread_idx.x)
            wid = tid // WAVE_SIZE
            lane = tid % WAVE_SIZE
            ln = lane % 16
            lg = lane // 16
            if const_expr(XCD > 0):
                # XCD-contiguous slabs: workgroups are dealt round-robin over the
                # XCDs, so give each XCD a dense run of N slabs.
                bid = fx.Index(fx.block_idx.x)
                n0 = ((bid % XCD) * (N_BLOCKS // XCD) + bid // XCD) * TN
            else:
                n0 = fx.Index(fx.block_idx.x) * TN
            k0 = wid * KS

            for rr in range_constexpr(TA // K_WAVES):
                row = wid + rr * K_WAVES
                sq = fx.Float32(0.0)
                zs = []
                for j in range_constexpr(k // (WAVE_SIZE * DMA_VEC)):
                    kk = lane * DMA_VEC + j * WAVE_SIZE * DMA_VEC
                    z = buffer_ops.buffer_load(
                        a_rsrc,
                        fx.Int32(row * a_stride + kk),
                        vec_width=DMA_VEC,
                        dtype=fx.BFloat16,
                    ).to(fx.Float32)
                    zs.append(z)
                    sq = sq + (z * z).reduce(ReductionOp.ADD)
                for off in range_constexpr(6):
                    sq = sq + sq.shuffle_xor(32 >> off, 64)
                rs = fmath.rsqrt(sq * (1.0 / k) + 1.0e-5)
                if lane == 0:
                    fx.ptr_store(rs, rms_ptr + fx.Int64(row))
                if fx.block_idx.x == 0:
                    for j in range_constexpr(k // (WAVE_SIZE * DMA_VEC)):
                        kk = lane * DMA_VEC + j * WAVE_SIZE * DMA_VEC
                        gam = buffer_ops.buffer_load(
                            qw_rsrc, fx.Int32(kk), vec_width=DMA_VEC, dtype=fx.BFloat16
                        ).to(fx.Float32)
                        qn = (zs[j] * rs * gam).to(fx.BFloat16)
                        buffer_ops.buffer_store(qn, qo_rsrc, fx.Int32(row * k + kk))
            gpu.barrier()

            # B is based at this workgroup's N slab, not at the tensor: the DMA's
            # voffset is one 32-bit VGPR, so a tensor-based descriptor wraps as soon
            # as n0 * k * 2 passes 2 GiB (the offset is cast to a SIGNED i32) and the
            # load reads whatever happens to be mapped there. Folding n0 into the
            # 64-bit base leaves the voffset at most TN * k * 2 bytes to span, which
            # no supported config comes near, and num_records then bounds the slab
            # exactly -- a future offset bug returns zeros instead of faulting.
            b_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(B)) + fx.Int64(n0) * (k * 2),
                num_records_bytes=TN * k * 2,
            )

            buf_i64 = fx.Int64(fx.ptrtoint(buf_ptr))
            wave_off = rocdl.readfirstlane(
                T.i64, fx.Int64(wid * (WAVE_ELEMS * 2)).ir_value()
            )
            wave_lds = buffer_ops.get_element_ptr(
                create_llvm_ptr(buf_i64, fx.AddressSpace.Shared), wave_off
            )

            # per-lane global byte offset of chunk 0, one per DMA instruction
            def chunk0_offset(i):
                p = lane + i * WAVE_SIZE
                r = p // CPR
                c = swz(r, p % CPR)
                if const_expr(i < A_INSTS):
                    g = (r * a_stride + k0 + c * DMA_VEC) * 2
                else:
                    # relative to b_rsrc, which is already based at slab n0
                    g = ((r - TA) * k + k0 + c * DMA_VEC) * 2
                return fx.Int32(g)

            vofs = [None] * IPC
            chunk_soff = [fx.Int32(j * KC * 2) for j in range_constexpr(CHUNKS)]

            def issue_chunk(j):
                stage = j % ST
                for i in range_constexpr(IPC):
                    lds_p = buffer_ops.get_element_ptr(
                        wave_lds,
                        static_byte_offset=stage * STAGE_ELEMS * 2 + i * WAVE_SIZE * 16,
                    )
                    rsrc = a_rsrc if const_expr(i < A_INSTS) else b_rsrc
                    cpol = "" if const_expr(i < A_INSTS) else " nt"
                    if const_expr(vofs[i] is None):
                        # each chunk-0 DMA goes out as soon as its own offset is
                        # ready; the sched_barrier below keeps the remaining
                        # address math from being hoisted in front of it
                        vofs[i] = chunk0_offset(i)
                    # the chunk's K offset goes in soffset: an instruction offset
                    # would also shift the LDS destination of the DMA.
                    # llvm.inline_asm is a raw dialect op, so the typed operands
                    # have to be unwrapped here.
                    llvm.InlineAsmOp(
                        None,
                        [lds_p, vofs[i].ir_value(), rsrc, chunk_soff[j].ir_value()],
                        "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, $3 offen"  # codespell:ignore offen
                        + cpol
                        + " lds",
                        "s,v,s,s",
                        has_side_effects=True,
                    )
                    if const_expr(j == 0):
                        rocdl.sched_barrier(0)

            def read_frag(stage, row, s):
                c = s * 4 + lg
                slot = row * CPR + swz(row, c)
                return fx.ptr_load(
                    buf_ptr
                    + fx.Int64(
                        wid * WAVE_ELEMS + (stage * STAGE_ELEMS) + slot * DMA_VEC
                    ),
                    result_type=fx.Vector.make_type(DMA_VEC, fx.BFloat16),
                )

            for j in range_constexpr(ST):
                issue_chunk(j)
            rocdl.sched_barrier(0)

            accs = [[acc_init] * NF for _ in range_constexpr(MF)]
            for j in range_constexpr(CHUNKS):
                stage = j % ST
                younger = min(CHUNKS, j + ST) - 1 - j
                llvm.InlineAsmOp(
                    None,
                    [],
                    f"s_waitcnt vmcnt({younger * IPC})",
                    "",
                    has_side_effects=True,
                )
                rocdl.sched_barrier(0)
                frags = []
                for s in range_constexpr(STEPS):
                    a_fr = []
                    for mf in range_constexpr(MF):
                        row = ln + mf * 16
                        aa = read_frag(stage, row, s).to(fx.Float32)
                        rs = fx.ptr_load(rms_ptr + fx.Int64(row % TA))
                        kk = k0 + j * KC + s * MFMA_K + lg * DMA_VEC
                        gam = buffer_ops.buffer_load(
                            qw_rsrc, fx.Int32(kk), vec_width=DMA_VEC, dtype=fx.BFloat16
                        ).to(fx.Float32)
                        a_fr.append((aa * rs * gam).to(fx.BFloat16))
                    b_fr = [
                        read_frag(stage, ln + (TA + nf * 16), s)
                        for nf in range_constexpr(NF)
                    ]
                    frags.append((a_fr, b_fr))
                if const_expr(j + ST < CHUNKS):
                    # refill the stage just read once its ds_reads have landed
                    llvm.InlineAsmOp(
                        None, [], "s_waitcnt lgkmcnt(0)", "", has_side_effects=True
                    )
                    issue_chunk(j + ST)
                rocdl.sched_barrier(0)
                for s in range_constexpr(STEPS):
                    a_fr, b_fr = frags[s]
                    for mf in range_constexpr(MF):
                        for nf in range_constexpr(NF):
                            if const_expr(SWAPPED):
                                # C^T tile: each lane holds 4 consecutive N columns of one M row
                                accs[mf][nf] = mfma(b_fr[nf], a_fr[mf], accs[mf][nf])
                            else:
                                accs[mf][nf] = mfma(a_fr[mf], b_fr[nf], accs[mf][nf])
                rocdl.sched_barrier(0)

            red_wave0 = wid * (WAVE_ELEMS // 2)
            if const_expr(SWAPPED):
                # partials in accumulator order, one 16-byte LDS write per fragment:
                # red[w][mf*NF+nf][lane][i] = C[mf*16 + lane%16][nf*16 + 4*(lane//16) + i]
                for mf in range_constexpr(MF):
                    for nf in range_constexpr(NF):
                        if const_expr(nf * 16 < TN):
                            fx.ptr_store(
                                accs[mf][nf],
                                red_ptr
                                + fx.Int64(
                                    red_wave0
                                    + (mf * NF + nf) * (WAVE_SIZE * 4)
                                    + lane * 4
                                ),
                            )
                gpu.barrier()
                for base in range_constexpr(0, EPI_SLOTS, BLOCK_THREADS):
                    sl = tid + base
                    mf_s = sl // ((TN // 4) * 16)
                    rem = sl % ((TN // 4) * 16)
                    cg = rem // 16  # 4-column group
                    rr = rem % 16

                    def reduce_store(mf_s, cg, rr):
                        src = (mf_s * NF + cg // 4) * (WAVE_SIZE * 4) + (
                            (cg % 4) * 16 + rr
                        ) * 4
                        tot = None
                        for w in range_constexpr(K_WAVES):
                            v = fx.ptr_load(
                                red_ptr + fx.Int64(src + w * (WAVE_ELEMS // 2)),
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            )
                            tot = v if const_expr(w == 0) else tot + v
                        out_v = tot.to(fx.BFloat16)
                        c_off = fx.Int32((mf_s * 16 + rr) * n + n0 + cg * 4)
                        buffer_ops.buffer_store(out_v, c_rsrc, c_off)

                    if const_expr(base + BLOCK_THREADS <= EPI_SLOTS):
                        reduce_store(mf_s, cg, rr)
                    else:
                        if sl < EPI_SLOTS:
                            reduce_store(mf_s, cg, rr)
                return

        else:
            tid = fx.Index(fx.thread_idx.x)
            wid = tid // 64
            lane = tid % 64
            for rr in range_constexpr(TA // K_WAVES):
                row = wid + rr * K_WAVES
                if row < m:
                    ap = fx.recast_iter(fx.BFloat16, A)
                    kwp = fx.recast_iter(fx.BFloat16, KW)
                    kp = fx.recast_iter(fx.BFloat16, KO)
                    cp = fx.recast_iter(fx.BFloat16, CACHE)
                    lp = fx.recast_iter(fx.Int64 if loc64 else fx.Int32, LOC)
                    kk = lane * 8
                    v = fx.ptr_load(
                        ap + fx.Int64(row * a_stride + 1536 + kk),
                        result_type=fx.Vector.make_type(8, fx.BFloat16),
                    ).to(fx.Float32)
                    gam = fx.ptr_load(
                        kwp + fx.Int64(kk),
                        result_type=fx.Vector.make_type(8, fx.BFloat16),
                    ).to(fx.Float32)
                    sq = (v * v).reduce(ReductionOp.ADD)
                    for off in range_constexpr(6):
                        sq = sq + sq.shuffle_xor(32 >> off, 64)
                    val = (v * fmath.rsqrt(sq * (1.0 / 512) + 1.0e-5) * gam).to(
                        fx.BFloat16
                    )
                    fx.ptr_store(val, kp + fx.Int64(row * 576 + kk))
                    tail = fx.ptr_load(ap + fx.Int64(row * a_stride + 2048 + lane))
                    fx.ptr_store(tail, kp + fx.Int64(row * 576 + 512 + lane))
                    loc = fx.ptr_load(lp + fx.Int64(row))
                    if (loc > 0) & (loc < cap):
                        fx.ptr_store(val, cp + fx.Int64(loc) * 576 + fx.Int64(kk))
                        fx.ptr_store(
                            tail, cp + fx.Int64(loc) * 576 + 512 + fx.Int64(lane)
                        )

    @flyc.jit
    def launch_skinny_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        QW: fx.Pointer,
        KW: fx.Pointer,
        QO: fx.Pointer,
        KO: fx.Pointer,
        CACHE: fx.Pointer,
        LOC: fx.Pointer,
        cap: fx.Int32,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        norm_qb_kernel._func.__name__ = KERNEL_NAME
        norm_qb_kernel(C, A, B, QW, KW, QO, KO, CACHE, LOC, cap, m).launch(
            grid=(N_BLOCKS + 1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    # the ring-fill DMAs need the kernel args; preloaded into SGPRs they no
    # longer wait on the kernarg s_load round trip
    launch_skinny_kernel.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_skinny_kernel


def covered(q, kv, qw, kw, weight, cache, loc):
    tensors = (q, kv, qw, kw, weight, cache, loc)
    return (
        all(
            isinstance(t, torch.Tensor) and t.is_cuda and t.device == q.device
            for t in tensors
        )
        and q.shape == (5, 1536)
        and kv.shape == (5, 576)
        and qw.shape == (1536,)
        and kw.shape == (512,)
        and weight.shape == (2304, 1536)
        and q.stride() == (3648, 1)
        and kv.stride() == q.stride()
        and kv.data_ptr() == q.data_ptr() + 1536 * 2
        and all(t.dtype == torch.bfloat16 for t in tensors[:-1])
        and all(t.is_contiguous() for t in (qw, kw, weight, cache, loc))
        and cache.shape[-1] == 576
        and cache.numel() % 576 == 0
        and loc.shape == (5,)
        and loc.dtype in (torch.int32, torch.int64)
        and all(t.data_ptr() % 16 == 0 for t in tensors)
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
    )


def norm_qb_cache(q, kv, qw, kw, weight, cache, loc):
    """Return normalized Q, packed KV, projected Q; zero is a padding slot."""
    assert covered(q, kv, qw, kw, weight, cache, loc)
    qo = q.new_empty((5, 1536))
    ko = q.new_empty((5, 576))
    out = q.new_empty((5, 2304))
    exe = _compile_norm_qb(q.stride(0), loc.dtype == torch.int64)
    args = (out, q, weight, qw, kw, qo, ko, cache, loc)
    pointers = [
        flyc.from_c_void_p(fx.Uint8, t.data_ptr(), assumed_align=16) for t in args
    ]
    _run_compiled(
        exe,
        *pointers,
        cache.numel() // 576,
        5,
        fx.Stream(torch.cuda.current_stream(q.device)),
    )
    return qo, ko, out
