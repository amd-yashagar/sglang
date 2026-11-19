"""K3 route/sort/MXFP8 microbench. Run only through gpuq; no MoE GEMMs timed.
Builds on test_moe_sorting_small._reference and aiter sorting/quant APIs.
PR routing uses the exact extracted top-k body; shared-GEMM overlap is NOT credited in this isolated glue benchmark.
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import aiter
import aiter.fused_moe as fm
import torch
import triton
from sglang.kernels.ops.moe import k3_route_sort as small
from sglang.kernels.ops.moe import k3_route_sort_mid as mid
from sglang.kernels.ops.moe import moe_route_radix4 as router

ROOT = Path(__file__).resolve().parent
MS = [1, 5, 8, 40, 64, 320]
E, K, BS = 896, 16, 32


def inputs(m, n, pattern, dtype=torch.bfloat16):
    # Correlated groups of five model verify tokens. Synthetic, not owner logits.
    gen = torch.Generator(device="cuda").manual_seed(92000 + m)
    base = torch.randn((triton.cdiv(m, 5), E), generator=gen, device="cuda")
    logits = (
        base.repeat_interleave(5, 0)[:m]
        + 0.2 * torch.randn((m, E), generator=gen, device="cuda")
    ).to(dtype)
    bias = (0.03 * torch.randn(E, generator=gen, device="cuda")).to(dtype)
    if pattern == "skew":
        bias[:32] += 2
    elif pattern == "one_rank":
        # TP8 is not EP8, but also stress all choices in one 112-expert range.
        bias[784:] += 4
    elif pattern == "ties":
        logits.zero_()
        bias.zero_()
    x = torch.randn((m, n), generator=gen, dtype=torch.bfloat16, device="cuda")
    return logits, bias, x


def buffers(m, n):
    blocks = triton.cdiv(m * K + E * BS - K, BS)
    return (
        torch.empty(blocks * BS, dtype=torch.int32, device="cuda"),
        torch.empty(blocks * BS, dtype=torch.float32, device="cuda"),
        torch.empty(blocks, dtype=torch.int32, device="cuda"),
        torch.empty(2, dtype=torch.int32, device="cuda"),
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    )


def paths(logits, bias, x):
    m, n = x.shape
    out = buffers(m, n)
    ids = torch.empty((m, K), dtype=torch.int32, device="cuda")
    w = torch.empty((m, K), dtype=torch.float32, device="cuda")
    counts = torch.zeros(1056, dtype=torch.int32, device="cuda")
    qo = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qs = torch.empty(
        (out[0].numel(), triton.cdiv(n // 32, 8) * 8), dtype=torch.uint8, device="cuda"
    )

    def route():
        router.build().run(logits, bias, w, ids, K, 1.0, True)

    def main():
        route()
        s = fm._moe_sorting_impl(ids, w, E, n, torch.bfloat16, BS, None, None, 0, True)
        q = fm.fused_dynamic_mxfp8_quant_moe_sort(
            x, s[0], s[3], m, K, BS, sorted_weights=s[1], num_experts_upper_bound=E
        )
        return ids, w, s, q

    def route_pr():
        mid._k3_mid_route_kernel[(m,)](
            logits,
            logits.stride(0),
            bias,
            ids,
            w,
            1.0,
            counts,
            x,
            x.stride(0),
            qo,
            qs,
            E=E,
            E_POW2=1024,
            TOPK=K,
            RENORM=True,
            DO_TOPK=True,
            N_COLS=n,
            QCHUNK=1024 if n == 7168 else 512,
            COUNT_QUANT=False,
            num_warps=4,
        )

    def pr():
        if m <= 16:
            route_pr()
            small.sort_quant(ids, w, *out, BS, x, qo, qs)
            return ids, w, out, (qo, qs)
        if m <= 128:
            # At M<=64 the PR routes alongside shared down; count/quant is separate.
            if m <= 64:
                route_pr()
                st = mid.route_quant(
                    x, counts, topk_ids=ids, topk_weights=w, num_experts=E
                )
            else:
                st = mid.route_quant(
                    x,
                    counts,
                    topk_ids=ids,
                    topk_weights=w,
                    logits=logits,
                    correction_bias=bias,
                )
            q = mid.sort_scatter(st, *out, BS, E)
            return ids, w, out, q
        return main()

    result = {"main": main, "pr46656": pr}
    try:
        from sglang.kernels.ops.moe.k3_route_sort_fast import route_sort_quant

        def candidate():
            os.environ["SGLANG_K3_ROUTE_SORT_FAST"] = "1"
            q = route_sort_quant(logits, bias, x, ids, w, out, BS)
            return ids, w, out, q

        result["candidate"] = candidate
    except ImportError:
        pass
    return result


def graph_us(fn, trials=7, repeats=64):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeats):
            fn()
    for _ in range(3):
        g.replay()
    times = []
    for _ in range(trials):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / repeats)
    return statistics.median(times), min(times), max(times)


def kernels(fn):
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        {"name": e.name, "us": e.device_time_total}
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=int, nargs="+", default=MS)
    p.add_argument("--dims", type=int, nargs="+", default=[7168])
    p.add_argument(
        "--patterns", nargs="+", default=["correlated", "skew", "one_rank", "ties"]
    )
    p.add_argument("--profile", action="store_true")
    p.add_argument("--logits-dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print("device", torch.cuda.get_device_properties(0), flush=True)
    print(
        "Isolated routing glue; shared GEMM overlap excluded. Roof 250/92 = 2.717 us/layer.",
        flush=True,
    )
    rows = []
    for n in a.dims:
        for pattern in a.patterns:
            for m in a.ms:
                fs = paths(
                    *inputs(
                        m,
                        n,
                        pattern,
                        torch.bfloat16 if a.logits_dtype == "bf16" else torch.float32,
                    )
                )
                for name, fn in fs.items():
                    med, lo, hi = graph_us(fn)
                    row = {
                        "m": m,
                        "n": n,
                        "pattern": pattern,
                        "logits_dtype": a.logits_dtype,
                        "path": name,
                        "us": med,
                        "min_us": lo,
                        "max_us": hi,
                        "ms_iter": med * 92 / 1000,
                        "roof_us": 250 / 92,
                    }
                    if a.profile:
                        row["kernels"] = kernels(fn)
                    rows.append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "kernels"}),
                        flush=True,
                    )
    a.output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "gpu": str(torch.cuda.get_device_properties(0)),
                "sglang_revision": subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(Path(router.__file__).parent),
                        "rev-parse",
                        "HEAD",
                    ],
                    text=True,
                ).strip(),
                "aiter_revision": subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(Path(aiter.__file__).parent),
                        "rev-parse",
                        "HEAD",
                    ],
                    text=True,
                ).strip(),
                "main_baseline": "ba9c86bb7e",
                "pr_baseline": "0761dfa0f9",
                "trials": 7,
                "graph_repeats": 64,
                "scope": "Isolated glue. PR exact extracted topk body; no shared-GEMM overlap credited. Synthetic correlated logits, not model traces.",
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
