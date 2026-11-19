"""Sweep benchmark for the KDA chain-verify kernels (one layer, in-graph).

Compares the four target-verify variants the KDA backend can dispatch:

    unfused        causal_conv1d_update + recurrence, per-step ssm snapshots
    unfused+ring   causal_conv1d_update + recurrence, ReplaySSM CACHE_RING
    fused          fused_kda_conv_gating_verify, per-step ssm snapshots
    fused+ring     fused_kda_conv_gating_verify, ReplaySSM CACHE_RING

Timing replays a CUDA graph capturing GRAPH_BATCH calls, matching how the
production verify runs (in-graph; bare launches would drown these ~10us
kernels in launch overhead). Imports only sglang.kernels.*, so it runs on
boxes where the sglang.srt/test import chain is broken.

    PYTHONPATH=python python3 benchmark/kernels/bench_kda_verify_sweep.py
    ... --batch-sizes 1 4 16 64 --modes fused fused+ring
    ... --sweep-bv          # re-tune KDA_VERIFY_BLOCK_V per mode/batch
    ... --hv-heads 16       # GQA shape (HV != H)
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify as fused_mod
import torch
from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

_DEVICE = "cuda"
_DTYPE = torch.bfloat16
_W = 4
# Ring length: power of two >= 2 * draft tokens (memory_pool.py invariant).
_RING_LEN = 16
_MODES = (
    "unfused",
    "unfused+ring",
    "fused",
    "fused+ring",
    "hip",
    "hip+ring",
    "hip+norm",
    "hip+ring+norm",
)
GRAPH_BATCH = 10


def make_inputs(B, T, H, HV, K, V, seed=0):
    torch.manual_seed(seed)
    dim = 2 * H * K + HV * V
    seq_len = B * T
    lines = slots = B + 1
    rnd = lambda *s, dt=_DTYPE: torch.randn(*s, device=_DEVICE, dtype=dt)
    return {
        "mixed": rnd(seq_len, dim) * 0.5,
        "w": rnd(dim, _W) * 0.3,
        "bias": rnd(dim) * 0.1,
        "a": rnd(seq_len, HV * K) * 0.5,
        "b": rnd(seq_len, HV),
        "A_log": rnd(HV, dt=torch.float32) * 0.5,
        "dt_bias": rnd(HV * K, dt=torch.float32) * 0.5,
        "conv_pool": rnd(lines, _W - 1, dim),
        "ssm": rnd(slots, HV, V, K, dt=torch.float32) * 0.2,
        "win_pool": torch.zeros(lines, T, _W - 1, dim, device=_DEVICE, dtype=_DTYPE),
        "inter_ssm": torch.zeros(
            lines, T, HV, V, K, device=_DEVICE, dtype=torch.float32
        ),
        "rawv": rnd(slots, HV, _RING_LEN, V),
        "rawk": rnd(slots, H, _RING_LEN, K),
        "g": rnd(slots, HV, _RING_LEN, K, dt=torch.float32),
        "beta": rnd(slots, HV, _RING_LEN, dt=torch.float32),
        "cache_indices": torch.arange(B, device=_DEVICE, dtype=torch.int32),
        "inter_indices": torch.arange(B, device=_DEVICE, dtype=torch.int32),
        "norm_gate": rnd(seq_len, HV * V + 8)[:, : HV * V],
        "norm_weight": rnd(V) * 0.1 + 1,
        "norm_rstd": torch.empty(seq_len * HV, device=_DEVICE),
        "cu": torch.arange(0, B + 1, device=_DEVICE, dtype=torch.int32) * T,
    }


def _ring_kwargs(inp, on):
    return dict(
        cache_ring=on,
        replayssm_rawv=inp["rawv"] if on else None,
        replayssm_rawk=inp["rawk"] if on else None,
        replayssm_g=inp["g"] if on else None,
        replayssm_beta=inp["beta"] if on else None,
    )


@lru_cache(None)
def _load_norm_binary(path):
    # Optional CPU-precompiled copy of the EXISTING production norm kernel.
    # Loading a CompiledKernel does not compile on a GPU lease.
    from sglang.kernels.ops.attention.fla.fused_norm_gate import (
        layer_norm_gated_fwd_kernel,
    )
    from triton.compiler import ASTSource, CompiledKernel

    spec = json.loads(Path(path).read_text())
    src = ASTSource(layer_norm_gated_fwd_kernel, spec["signature"], spec["constants"])
    return CompiledKernel(src, spec["metadata_group"], spec["hash"]), src


def _norm_runner(inp, heads, dim):
    if not inp.get("norm_binary"):
        from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated

        return lambda out: rms_norm_gated(
            out,
            inp["norm_gate"].unflatten(-1, (heads, dim)),
            inp["norm_weight"],
            None,
            activation="sigmoid",
            eps=1e-5,
        )
    kernel, src = _load_norm_binary(inp["norm_binary"])
    constants = {src.fn.arg_names[k[0]]: v for k, v in src.constants.items()}
    if constants.get("D") != dim or constants.get("G_HEADS") != heads:
        raise ValueError("norm binary was compiled for different head dimensions")
    rows = inp["norm_gate"].shape[0] * heads

    def fn(out):
        params = dict(
            x=out,
            g=inp["norm_gate"],
            y=out,
            w=inp["norm_weight"],
            b=None,
            residual=None,
            residual_out=None,
            mean=None,
            rstd=inp["norm_rstd"],
            eps=1e-5,
            T=rows,
            stride_g_tok=inp["norm_gate"].stride(0),
            stride_g_head=dim,
            **{src.fn.arg_names[k[0]]: v for k, v in src.constants.items()},
        )
        kernel[((rows + 31) // 32, 1, 1)](*(params[name] for name in src.fn.arg_names))
        return out

    return fn


def make_runner(mode, inp, B, T, H, HV, K, V, lower_bound=None):
    dim = 2 * H * K + HV * V
    seq_len = B * T
    ring = "+ring" in mode
    scale = K**-0.5

    if mode.startswith("hip"):
        if H != HV or K != 128 or V != 128:
            raise ValueError("gfx950 HIP verify requires equal heads and K=V=128")
        from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import (
            kda_verify_gfx950,
            use_full_head_verify,
        )

        # Kimi-K3 has bias-free conv with FP32 weights (not the generic BF16 case).
        inp["w"] = inp["w"].float()

        with_norm = "+norm" in mode
        fused_norm = with_norm and use_full_head_verify(
            batch_size=B, T=T, heads=HV, snap=not ring, with_norm=True
        )
        norm = _norm_runner(inp, HV, V) if with_norm and not fused_norm else None

        def fn():
            out = kda_verify_gfx950(
                mixed_qkv=inp["mixed"],
                conv_weight=inp["w"],
                conv_state=inp["conv_pool"].transpose(-1, -2),
                intermediate_conv_window=inp["win_pool"].transpose(-1, -2),
                intermediate_state_indices=inp["inter_indices"],
                a=inp["a"],
                b=inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                ssm_states=inp["ssm"],
                cache_indices=inp["cache_indices"],
                intermediate_states_buffer=None if ring else inp["inter_ssm"],
                scale=scale,
                T=T,
                num_q_heads=H,
                num_v_heads=HV,
                head_k_dim=K,
                head_v_dim=V,
                lower_bound=lower_bound,
                norm_gate=inp["norm_gate"] if fused_norm else None,
                norm_weight=inp["norm_weight"] if fused_norm else None,
                **_ring_kwargs(inp, ring),
            )
            return norm(out) if norm is not None else out

        return fn

    if mode.startswith("fused"):

        def fn():
            fused_kda_conv_gating_verify(
                mixed_qkv=inp["mixed"],
                conv_weight=inp["w"],
                conv_bias=inp["bias"],
                conv_state=inp["conv_pool"].transpose(-1, -2),
                conv_state_indices=inp["cache_indices"],
                intermediate_conv_window=inp["win_pool"].transpose(-1, -2),
                intermediate_state_indices=inp["inter_indices"],
                a=inp["a"],
                b=inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                ssm_states=inp["ssm"],
                cache_indices=inp["cache_indices"],
                intermediate_states_buffer=None if ring else inp["inter_ssm"],
                scale=scale,
                T=T,
                num_q_heads=H,
                num_v_heads=HV,
                head_k_dim=K,
                head_v_dim=V,
                lower_bound=lower_bound,
                **_ring_kwargs(inp, ring),
            )

        return fn

    def fn():
        x3 = inp["mixed"].reshape(B, T, dim).transpose(1, 2)
        out3 = causal_conv1d_update(
            x3,
            inp["conv_pool"].transpose(-1, -2),
            inp["w"],
            inp["bias"],
            activation="silu",
            conv_state_indices=inp["cache_indices"],
            intermediate_conv_window=inp["win_pool"].transpose(-1, -2),
            intermediate_state_indices=inp["inter_indices"],
        )
        mixed_out = out3.transpose(1, 2).reshape(seq_len, dim)
        q, k, v = mixed_out.split([H * K, H * K, HV * V], dim=-1)
        fused_sigmoid_gating_delta_rule_update(
            A_log=inp["A_log"],
            a=inp["a"],
            dt_bias=inp["dt_bias"],
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q.unflatten(-1, (H, K)).unsqueeze(0),
            k=k.unflatten(-1, (H, K)).unsqueeze(0),
            v=v.unflatten(-1, (HV, V)).unsqueeze(0),
            b=inp["b"],
            initial_state_source=inp["ssm"],
            initial_state_indices=inp["cache_indices"],
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inp["cu"],
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=None if ring else inp["inter_ssm"],
            intermediate_state_indices=None if ring else inp["inter_indices"],
            cache_steps=T,
            retrieve_parent_token=None,
            lower_bound=lower_bound,
            **_ring_kwargs(inp, ring),
        )

    return fn


def bench_graph(fn, iters=200, graph_batch=GRAPH_BATCH, repeats=1):
    """us per call; capture once, then time independent graph-replay samples."""
    for _ in range(3):  # callers must precompile outside the lease
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_batch):
            fn()
    samples = []
    for _ in range(repeats):
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / (iters * graph_batch))
    graph.reset()
    return samples[0] if repeats == 1 else samples


def check_ring_bitwise(B, T, H, HV, K, V, lower_bound=None):
    """One-shot guard: fused+ring must fill the same ring bytes as unfused+ring."""
    ref, fus = (make_inputs(B, T, H, HV, K, V, seed=7) for _ in range(2))
    make_runner("unfused+ring", ref, B, T, H, HV, K, V, lower_bound)()
    make_runner("fused+ring", fus, B, T, H, HV, K, V, lower_bound)()
    torch.cuda.synchronize()
    for name in ("rawv", "rawk", "g", "beta"):
        assert torch.equal(ref[name], fus[name]), f"ring mismatch: {name}"


def run_modes(args, label_extra=""):
    print(
        f"H={args.heads} HV={args.hv_heads} K={args.head_k_dim} V={args.head_v_dim} "
        f"T={args.draft_tokens} gate={'safe' if args.lower_bound is not None else 'std'} "
        f"BV={fused_mod.KDA_VERIFY_BLOCK_V}{label_extra}"
    )
    header = f"{'B':>4}  " + "".join(f"{m:>14}" for m in args.modes)
    print(header)
    records = []
    for B in args.batch_sizes:
        times = []
        for mode in args.modes:
            runners = []
            for layer in range(args.layers):
                inp = make_inputs(
                    B,
                    args.draft_tokens,
                    args.heads,
                    args.hv_heads,
                    args.head_k_dim,
                    args.head_v_dim,
                    seed=layer,
                )
                inp["norm_binary"] = args.norm_binary
                runners.append(
                    make_runner(
                        mode,
                        inp,
                        B,
                        args.draft_tokens,
                        args.heads,
                        args.hv_heads,
                        args.head_k_dim,
                        args.head_v_dim,
                        args.lower_bound,
                    )
                )

            def fn():
                for runner in runners:
                    runner()

            print(f"Timing B={B} {mode}, {args.layers} distinct layers", flush=True)
            measured = bench_graph(
                fn,
                iters=args.iters,
                graph_batch=max(1, GRAPH_BATCH // args.layers),
                repeats=args.repeats,
            )
            if args.repeats == 1:
                measured = [measured]
            samples = [us / args.layers for us in measured]
            us = sorted(samples)[len(samples) // 2]
            times.append(us)
            # State traffic only, NOT total HBM traffic: ring verify does not write state;
            # snapshot verify writes T complete states. Acceptance is a separate operation.
            state_bytes = (
                B
                * args.hv_heads
                * args.head_k_dim
                * args.head_v_dim
                * 4
                * (1 if "+ring" in mode else 1 + args.draft_tokens)
            )
            gbps = state_bytes / us / 1e3
            records.append(
                dict(
                    batch=B,
                    layers=args.layers,
                    mode=mode,
                    us=us,
                    samples_us=samples,
                    state_bytes=state_bytes,
                    state_GBs=gbps,
                    roof_fraction=gbps / 6000,
                )
            )
        row = f"{B:>4}  " + "".join(f"{t:>11.2f} us" for t in times)
        print(row)
    for r in records:
        print(
            "B={batch} {mode}: {state_GBs:.1f} state GB/s ({roof_fraction:.1%} of 6 TB/s)".format(
                **r
            )
        )
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2) + "\n")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=8)  # ling-v3 TP4 KDA shape
    parser.add_argument("--hv-heads", type=int, default=None)
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--draft-tokens", type=int, default=4)
    # ling-v3 runs the safe gate: --lower-bound -5.0 (kda_lower_bound).
    parser.add_argument("--lower-bound", type=float, default=None)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--modes", nargs="+", default=list(_MODES[:4]), choices=_MODES)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Distinct per-layer states per timed call; 69 avoids hot-state cache bias",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--norm-binary",
        help="CPU-precompiled production RMSNorm manifest (HIP +norm modes)",
    )
    parser.add_argument("--json", help="Write timing and state-traffic records")
    parser.add_argument(
        "--sweep-bv",
        action="store_true",
        help="re-run the fused modes across KDA_VERIFY_BLOCK_V candidates; "
        "BLOCK_V was tuned with snapshot writes on, so ring mode may move it",
    )
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    if min(args.layers, args.iters, args.repeats) < 1:
        parser.error("--layers, --iters and --repeats must be positive")
    if args.hv_heads is None:
        args.hv_heads = args.heads
    if args.draft_tokens * 2 > _RING_LEN:
        raise ValueError(f"--draft-tokens > {_RING_LEN // 2} exceeds the bench ring")

    if not args.skip_check:
        check_ring_bitwise(
            4,
            args.draft_tokens,
            args.heads,
            args.hv_heads,
            args.head_k_dim,
            args.head_v_dim,
            args.lower_bound,
        )
        print("ring bitwise check: OK\n")

    run_modes(args)

    if args.sweep_bv:
        args.modes = [m for m in args.modes if m.startswith("fused")] or [
            "fused",
            "fused+ring",
        ]
        default_bv = fused_mod.KDA_VERIFY_BLOCK_V
        try:
            for bv in (2, 4, 8, 16, 32):
                fused_mod.KDA_VERIFY_BLOCK_V = bv
                run_modes(args, label_extra=" (BV sweep)")
        finally:
            fused_mod.KDA_VERIFY_BLOCK_V = default_bv


if __name__ == "__main__":
    try:
        main()
    finally:
        _load_norm_binary.cache_clear()
