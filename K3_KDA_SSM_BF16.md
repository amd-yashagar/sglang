# Kimi-K3 fused KDA decode + bf16 SSM

Branch: `k3-kda-ssm-bf16`  
Pair with: `amd-mvarjoka/aiter` `k3-kda-ssm-bf16` (causal-conv stride/FP32-weight fix; **not** required for decode serving)  
Target: gfx950 / MI355X, SGLang Kimi-K3 TP8. One commit on `sgl-project/sglang` main.

## What changed

HIP fused FlyDSL KDA decode (`SGLANG_K3_KDA_FUSED_BACKEND=aiter`) now accepts `--mamba-ssm-dtype bfloat16` as well as `float32`.

- Recurrent math stays **FP32**.
- SSM pool `[H,128,128]` is loaded/stored as **bf16x8** (16 B, gfx950 buffer-copy cap) or **f32x4**.
- `extf` load / round-nearest `truncf` store, matching unfused FLA.
- Warmup compiles **both** SSM dtypes × batch `{1,2}` **before** HIP graph capture.

Prefill conv stays **Triton**. `kda_prefill_conv_aiter_hip.py` exists only to gate the AITER conv fix; it is not dispatched (FlyDSL measured ~4.6× slower than Triton at T=1024, dim=4608).

## Why

Two independent HIP-graph bugs, both concurrency-dependent (GSM8K 0% / ~89%):

1. **Fused FlyDSL** (`SGLANG_K3_KDA_FUSED_BACKEND=aiter`): JIT inside graph capture loaded a bf16 pool as f32×4. Warmup now compiles both SSM dtypes × batch `{1,2}` and uses bf16x8 / f32x4.
2. **Unfused Triton** (fusion env unset; this is the traced `fused_sigmoid_gating` path): Kimi-K3 always sets `gate_lower_bound`, and decode used to skip packed T=1 and take varlen `fused_sigmoid`. Graph replay fills padded `query_start_loc` so those rows have **T=0**, which left KDA outputs as `new_empty` garbage that then entered MoE (decode MoE GEMM can look 4–8× “faster” from empty expert tiles). Packed decode already implements the safe gate and writes zeros for `idx == -1`. Decode now uses packed even with `lower_bound`. Varlen fused_sigmoid also stores zeros when T≤0.

Colleague flags (`--disable-cuda-graph`, huge chunked-prefill, int8 mamba ckpt) were workarounds, not the fix.

`--mamba-ssm-dtype` is the KDA recurrent pool, **not** KV cache.

## How to use (serving)

Keep HIP graphs **on**. bf16 SSM works on both:

```bash
# unfused Triton packed decode (default; no fusion env)
python -m sglang.launch_server \
  --model <kimi-k3> \
  --tp 8 \
  --mamba-ssm-dtype bfloat16

# optional fused FlyDSL decode
export SGLANG_K3_KDA_FUSED_BACKEND=aiter
python -m sglang.launch_server \
  --model <kimi-k3> \
  --tp 8 \
  --mamba-ssm-dtype bfloat16
```

Do **not** add `--disable-cuda-graph` for either path. Fusion is off unless `SGLANG_K3_KDA_FUSED_BACKEND=aiter`.

## Expected

| Check | Expect |
|---|---|
| Kernel tests | `pytest test/registered/kernels/ops/kimi_k3/flydsl_ops/test_kimi_k3_kda_decode.py -v` on gfx950. Covers fp32+bf16 and graph replay at BS 1/2/8/16. |
| Unfused packed + graph pad | `pytest test/registered/attention/test_kda_kernels.py -k "graph_replay_bf16 or t0_padded" -v` |
| Prefill conv (optional, not served) | `pytest test/registered/kernels/ops/kimi_k3/flydsl_ops/test_kimi_k3_kda_prefill_conv.py -v` needs the AITER branch. |
| Microbench | `python test/registered/kernels/ops/kimi_k3/flydsl_ops/bench_kimi_k3_kda_decode.py --state-dtype bfloat16 --batch 16` |
| Serving quality | Fused+bf16 should match **Triton-bf16** GSM8K (~95). It does **not** restore fp32-pool NIAH 64k. |
| Serving speed | Decode C16 kernel ~7% vs fp32 SSM in isolation. Prefill unchanged. |

Unfused log: decode kernel `fused_recurrent_kda_packed_decode` (not `fused_sigmoid_gating_delta_rule`). Fused log: HIP fused KDA accepted; rocprof `kimi_k3_kda_decode_fb_*_ssmbf16`.

## Not this change

A 6–8× decode MoE1/MoE2 duration drop with **the same kernel names** is not an SSM-bf16 win. Prefill MoE should be unchanged. If decode MoE GEMM collapses while `grouped_topk` time stays identical, treat it as possible empty expert tiles from bad activations, not a faster MoE binary.
