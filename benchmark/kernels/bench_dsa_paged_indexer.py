#!/usr/bin/env python3
"""Cold-cache K3.1 ROCm paged MQA + existing paged top-k microbench.
Reuses aiter/op_tests/test_paged_mqa_logits.py input builder, FP32 reference,
and its calc_diff < 1e-3 criterion. Geometry is explicit, not inferred from K3 config.
"""

import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path

ROOT = Path(
    os.environ.get("SGLANG_DSA_INDEXER_BENCH_DIR", Path(__file__).resolve().parent)
)
if "TMPDIR" not in os.environ:
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str((ROOT / "tmp").resolve())
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "triton-cache"))
os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", str(ROOT / "flydsl-cache"))
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "torch-extensions"))
os.environ.setdefault("MAX_JOBS", str(os.cpu_count()))
os.environ.setdefault("SGLANG_JIT_CACHE_DIR", str(ROOT / "sglang-jit"))

import aiter.ops.triton.attention.pa_mqa_logits as pa
import torch

from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    aiter_paged_mqa_logits,
    aiter_paged_mqa_logits_persistent_schedule,
    aiter_paged_mqa_logits_verify,
)
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2


def load_helpers():
    import aiter

    spec = importlib.util.spec_from_file_location(
        "aiter_existing_mqa_test",
        Path(aiter.__file__).resolve().parent.parent
        / "op_tests/test_paged_mqa_logits.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compiled_schedule(batch):
    # Explicit target, no driver/device discovery. Identical AITER metadata body;
    # integer scalar specialization is intentionally disabled (not timed).
    import triton
    from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits import (
        _deepgemm_fp8_paged_mqa_logits_persistent_schedule as kernel,
    )
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    return triton.compile(
        ASTSource(
            kernel,
            signature={
                "context_len_ptr": "*i32",
                "cta_info_ptr": "*i32",
                "batch_size": "i32",
                "slots": "i32",
                "max_model_len": "i32",
            },
            constexprs={
                "ChunkK": 256,
                "NEXT_N": 1,
                "BLOCK_B": triton.next_power_of_2(batch),
                "BLOCK_S": 16,
            },
        ),
        target=GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 1},
    )


def install_schedule():
    class Launcher:
        def __getitem__(self, grid):
            def run(lens, out, batch, slots, width, **kwargs):
                compiled_schedule(batch)[tuple(grid) + (1,) * (3 - len(grid))](
                    lens, out, batch, slots, width
                )

            return run

    pa._deepgemm_fp8_paged_mqa_logits_persistent_schedule = Launcher()


def build_only():
    import subprocess

    from sglang.kernels.ops.attention.dsa import gfx950_mqa_logits

    assert all(
        os.environ.get(key) == ""
        for key in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
    ), "Hide GPUs for CPU compilation"
    from aiter.ops.triton.utils._triton import arch_info

    arch_info._CACHED_ARCH = "gfx950"  # explicit target, do not probe a driver
    for batch in (1, 8, 16, 64):
        compiled_schedule(batch)
    for persistent, waves in ((True, 2), (False, 4)):
        pa._compile_deepgemm_fp8_paged_mqa_logits(
            32, 256, True, 64, 128, True, WavePerEU=waves, Persistent=persistent
        )
    source = Path(gfx950_mqa_logits.__file__).parent / "hip_gfx950/grouped_mqa.cu"
    lib = os.environ["SGLANG_DSA_INDEXER_HIP_LIB"]
    subprocess.run(
        [
            "/opt/rocm/bin/hipcc",
            "--offload-arch=gfx950",
            "-O3",
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-mllvm",
            "-amdgpu-mfma-vgpr-form",
            str(source),
            "-o",
            lib,
        ],
        check=True,
    )
    # The normal JIT loader discovers its target from a GPU. Supply the known
    # gfx950 target explicitly instead, including its feature fingerprint, and
    # reuse exactly the existing top-k source/wrappers/flags (no kernel changes).
    from contextlib import ExitStack
    from unittest.mock import patch

    from sglang.kernels.jit.utils.arch import CompileArch
    from sglang.kernels.jit.utils.compile import cache, loader, toolchain

    arch = os.environ.get("SGLANG_DSA_INDEXER_BUILD_ARCH", "gfx950:sramecc+:xnack-")
    with ExitStack() as stack:
        for module in (cache, loader, toolchain):
            stack.enter_context(patch.object(module, "is_hip_runtime", lambda: True))
            stack.enter_context(patch.object(module, "is_sail_runtime", lambda: False))
            stack.enter_context(
                patch.object(
                    module, "get_compile_arch", lambda *a: CompileArch(9, 5, "")
                )
            )
        stack.enter_context(patch.object(toolchain, "gpu_arch_name", lambda: arch))
        stack.enter_context(
            patch.object(
                loader,
                "get_default_target_flags",
                lambda *a: ["-DUSE_ROCM", "-std=c++20", "-O3", "-DHIP_FP8_TYPE_E4M3=1"],
            )
        )
        exports = (
            "transform_paged",
            "transform_ragged",
            "transform_packed",
            "transform_candidates",
            "transform_candidates_paged",
            "plan",
            "transform_kpool",
            "transform_kpool_ragged",
        )
        loader.load_jit(
            "dpsk_v4_topk_v2",
            "false",
            "mode1",
            extra_cuda_cflags=["-DSGL_TOPK_MODE=1"],
            cuda_files=["deepseek_v4/topk_v2.cuh"],
            cuda_wrappers=[
                ("topk_" + name, "TopKKernel<false>::" + name) for name in exports
            ],
        )
    # Build the existing AITER grouped FlyDSL body at the real query count.
    # Meta tensors and explicit gfx950 target; no live-device discovery.
    import flydsl.compiler as flyc
    from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx950 import _compile

    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_GPU_ARCH"] = "gfx950"
    try:
        for n in range(1, 9):

            def meta(shape, dtype):
                return torch.empty(shape, dtype=dtype, device="meta")

            b, width = 16, 131072
            q = meta((b, n, 32, 128), torch.float8_e4m3fn)
            kv = meta((b * (width // 64) * 8448,), torch.uint8)
            w = meta((b * n, 32), torch.float32)
            out = meta((b * n, width), torch.float32)
            lens = meta((b,), torch.int32)
            table = meta((b, width // 64), torch.int32)
            for ctx_stride in {1, n}:
                ctx = lens if ctx_stride == 1 else meta((b, n), torch.int32)
                flyc.compile(
                    _compile(ctx_stride, n),
                    q,
                    kv,
                    w,
                    out,
                    ctx,
                    table,
                    lens,
                    1024,
                    b,
                    64,
                    n,
                    0,
                    n * 4096,
                    width // 64,
                    width,
                )
    finally:
        os.environ.pop("COMPILE_ONLY", None)
    print("CPU builds complete", flush=True)


def reject_runtime_compilation():
    from triton.backends.amd.compiler import HIPBackend

    def reject(*args, **kwargs):
        raise RuntimeError(
            "Unexpected Triton cache miss: compile with --build-only outside gpuq"
        )

    HIPBackend.add_stages = reject
    from sglang.kernels.jit.utils.compile import ninja

    ninja.build = reject
    os.environ["SGLANG_CRASH_ON_JIT_COMPILE"] = "1"
    os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"


def cold_graph_us(fn, flush, samples=9):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(samples)
    ]
    with torch.cuda.graph(graph):
        for start, end in events:
            flush.add_(1)  # 512 MiB read + write, outside measured events
            start.record()
            fn()
            end.record()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    values = [s.elapsed_time(e) * 1000 for s, e in events]
    return {
        "us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def topk_check(scores, indices, lens, label, strict=True):
    below = 0
    missing = 0
    max_below = 0.0
    # Membership equality modulo exact cutoff ties, with uniqueness and padding.
    for row, length in enumerate(lens.tolist()):
        k = min(2048, length)
        idx = indices[row]
        assert (idx[idx < 0] == -1).all(), (label, row, "padding")
        valid = idx[idx >= 0].long()
        assert valid.numel() == k and valid.unique().numel() == k, (
            label,
            row,
            "count/duplicates",
        )
        assert (valid < length).all(), (label, row, "range")
        if k:
            threshold = scores[row, :length].topk(k).values[-1]
            chosen = scores[row, valid]
            below += int((chosen < threshold).sum())
            strict_count = int((scores[row, :length] > threshold).sum())
            chosen_strict_count = int((chosen > threshold).sum())
            missing += strict_count - chosen_strict_count
            max_below = max(max_below, float((threshold - chosen).max()))
            if strict:
                assert (chosen >= threshold).all(), (
                    label,
                    row,
                    "below cutoff",
                    max_below,
                )
                assert int((chosen > threshold).sum()) == int(
                    (scores[row, :length] > threshold).sum()
                ), (label, row, "missing strict winner")
    return {
        "below_cutoff_count": below,
        "max_cutoff_delta": max_below,
        "missing_strict_winners": missing,
        "exact_mod_ties": below == 0 and missing == 0,
    }


def fp64_reference(inp, length):
    """Independent dequantized reference; optional and never timed."""
    q = inp["q_fp8"].double()
    b, n, _, d = q.shape
    raw = inp["kv_raw"].reshape(inp["kv_raw"].shape[0], -1)
    keys = raw[:, : 64 * d].contiguous().view(torch.float8_e4m3fn).double()
    scales = raw[:, 64 * d :].contiguous().view(torch.float32).double()
    kv = keys.view(-1, 64, d) * scales.view(-1, 64, 1)
    pos = torch.arange(length, device=q.device)
    table = inp["kv_indices"][:, pos // 64]
    result = torch.empty((b * n, length), dtype=torch.float64, device=q.device)
    for batch in range(b):
        dots = torch.einsum("nhd,pd->nhp", q[batch], kv[table[batch], pos % 64]).relu()
        weights = inp["weights"][batch * n : (batch + 1) * n].double()
        result[batch * n : (batch + 1) * n] = (dots * weights[:, :, None]).sum(1)
    return result


def run_case(batch, next_n, length, args, flush):
    inp = helpers.build_inputs(
        (batch, next_n, 32, 128, 64, True),
        length,
        ((length + 255) // 256) * 256,
        seed=args.seed,
    )
    width = inp["max_model_len"]
    q, cache, w = inp["q_fp8"], inp["kv_cache"], inp["weights"]
    req_lens, tables = inp["context_lens"], inp["kv_indices"]
    if args.ragged:
        req_lens.sub_(torch.arange(batch, device="cuda", dtype=torch.int32) % 61)
    lens = (
        req_lens[:, None]
        - next_n
        + 1
        + torch.arange(next_n, device="cuda", dtype=torch.int32)
    ).flatten()
    row_tables = tables.repeat_interleave(next_n, dim=0)
    plan = plan_topk_v2(lens)
    schedule = (
        aiter_paged_mqa_logits_persistent_schedule(lens, width) if next_n == 1 else None
    )
    if next_n == 1:

        def compute_logits():
            return aiter_paged_mqa_logits(
                q.flatten(0, 1),
                cache,
                w,
                lens,
                row_tables,
                width,
                preshuffle=True,
                kv_block_size=64,
                persistent_schedule=schedule,
            )

    else:

        def compute_logits():
            return aiter_paged_mqa_logits_verify(
                q.flatten(0, 1),
                cache,
                w,
                req_lens,
                tables,
                width,
                q_offset=batch * next_n,
                B=batch,
                next_n=next_n,
                kv_block_size=64,
            )

    def with_flag(value, flydsl="0"):
        names = ("SGLANG_DSA_GFX950_GROUPED_MQA", "SGLANG_DSA_GFX950_FLYDSL_MQA")
        previous = {name: os.environ.get(name) for name in names}
        os.environ[names[0]], os.environ[names[1]] = value, flydsl
        try:
            return compute_logits()
        finally:
            for name, old in previous.items():
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    variants = {"baseline": lambda: with_flag("0")}
    if args.candidate:
        variants["candidate"] = lambda: with_flag("1")
    if args.flydsl:
        variants["v2"] = lambda: with_flag("1", "1")
    ref = helpers.run_torch(inp, length)
    # The existing helper assumes uniform causal lengths; apply ragged bounds.
    ref.masked_fill_(
        torch.arange(length, device="cuda")[None, :] >= lens[:, None], -float("inf")
    )
    ref64 = fp64_reference(inp, length) if args.fp64_reference else None
    if ref64 is not None:
        ref64.masked_fill_(
            torch.arange(length, device="cuda")[None, :] >= lens[:, None], -float("inf")
        )
    result = {
        "batch": batch,
        "next_n": next_n,
        "context": length,
        "ragged": args.ragged,
        "layers": args.layers,
        "heads": 32,
        "head_dim": 128,
        "page": 64,
        "cache_flush_MiB": 512,
        "index_k_bytes": batch * length * 132,
    }
    # Optimistic K-read-only roof and conservative traffic accounting (one K read/request,
    # Q/weights once, logits store + top-k read, selected index store). Decimal 6 TB/s.
    kbytes = batch * length * 132
    total_bytes = (
        kbytes
        + q.numel()
        + w.numel() * 4
        + batch * next_n * length * 8
        + batch * next_n * 2048 * 4
    )
    result["k_roof_us"] = kbytes / 6e6
    result["pipeline_roof_us"] = total_bytes / 6e6
    baseline_scores = None
    baseline_error = None
    for name, fn in variants.items():
        logits = fn()
        valid = torch.arange(length, device="cuda")[None, :] < lens[:, None]
        got = logits[:, :length]
        a, b = got.masked_fill(~valid, 0), ref.masked_fill(~valid, 0)
        if baseline_scores is None:
            baseline_scores = a.clone()
        baseline_max_diff = float((a - baseline_scores).abs().max())
        max_error = float((a - b).abs().max())
        if baseline_error is None:
            baseline_error = max_error
        assert max_error <= baseline_error + 1e-4, (
            name,
            "Exceeds existing kernel error",
            max_error,
            baseline_error,
        )
        diff = helpers.calc_diff(a, b)
        assert diff < 1e-3, (name, "calc_diff", diff)
        # Same per-element bound as the existing aiter benchmark.
        torch.testing.assert_close(a, b, atol=5.0, rtol=1e-2)
        out = torch.empty((batch * next_n, 2048), device="cuda", dtype=torch.int32)

        def topk():
            topk_transform_paged_v2(logits, lens, None, out, 64, plan)

        topk()
        topk_check(logits, out, lens, name + ":own_logits")
        baseline_membership = topk_check(
            baseline_scores, out, lens, name + ":baseline", strict=False
        )
        fp32_membership = topk_check(
            ref, out, lens, name + ":fp32_reference", strict=False
        )
        fp64_membership = (
            topk_check(ref64, out, lens, name + ":fp64_reference", strict=False)
            if ref64 is not None
            else None
        )
        # Exercise the actual physical page-table transform as well as raw membership.
        physical = torch.empty_like(out)
        topk_transform_paged_v2(
            logits, lens, row_tables, physical, 64, plan, out_raw_indices=out
        )
        expected = (
            row_tables.gather(1, out.clamp_min(0).long() // 64) * 64
            + out.clamp_min(0) % 64
        )
        expected.masked_fill_(out < 0, -1)
        assert torch.equal(physical, expected), (name, "page transform")

        def pipeline():
            scores = fn()
            topk_transform_paged_v2(scores, lens, row_tables, physical, 64, plan)

        logit_time = cold_graph_us(fn, flush, args.samples)
        topk_time = cold_graph_us(topk, flush, args.samples)
        pipe_time = cold_graph_us(pipeline, flush, args.samples)
        result[name] = {
            "logits": logit_time,
            "topk": topk_time,
            "pipeline": pipe_time,
            "ms_iter": pipe_time["us"] * args.layers / 1000,
            "calc_diff": diff,
            "max_abs_error": float((a - b).abs().max()),
            "max_diff_from_baseline": baseline_max_diff,
            "topk_exact_mod_ties": True,
            "fp32_membership": fp32_membership,
            "fp64_membership": fp64_membership,
            "max_abs_fp64_error": (
                float((a - ref64.masked_fill(~valid, 0)).abs().max())
                if ref64 is not None
                else None
            ),
            "baseline_membership": baseline_membership,
            "pipeline_kind": (
                "flydsl_or_retained_v1_logits_topk_v2"
                if name == "v2"
                else "grouped_logits_topk_v2"
                if name == "candidate"
                and next_n == 5
                and width >= 32768
                and batch >= 16
                else "aiter_logits_topk_v2"
            ),
            "pipeline_over_roof": pipe_time["us"] / result["pipeline_roof_us"],
        }
    print(json.dumps(result), flush=True)
    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="*", help="batch:next_n:context; default full requested matrix"
    )
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument(
        "--flydsl",
        action="store_true",
        help="Compare v2 FlyDSL plus retained v1 regimes",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument(
        "--fp64-reference",
        action="store_true",
        help="Also audit membership against independent float64 math",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=24,
        help="ASSUMED 24 full-attention layers in supplied K3 config; K3.1 DSA config unavailable",
    )
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.build_only:
        build_only()
        return
    reject_runtime_compilation()
    install_schedule()
    global helpers
    helpers = load_helpers()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "aiter": str(Path(pa.__file__)),
                "gpu": str(torch.cuda.get_device_properties(0)),
                "assumptions": "H32 D128 page64 topk2048 DCP1; 24 layers assumed, NOT confirmed K3.1 config",
            }
        ),
        flush=True,
    )
    flush = torch.zeros(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    cases = (
        [tuple(map(int, c.split(":"))) for c in args.cases]
        if args.cases
        else [(b, n, l) for l in (1024, 32768) for b in (1, 8, 16, 64) for n in (1, 5)]
    )
    with torch.inference_mode():
        for case in cases:
            run_case(*case, args, flush)


if __name__ == "__main__":
    main()
