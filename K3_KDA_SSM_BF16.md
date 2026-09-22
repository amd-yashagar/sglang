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

`--mamba-ssm-dtype bfloat16` + fused decode previously JIT'd the wrong-width kernel **inside** graph capture (bf16 buffer read as f32×4). That produced GSM8K 0%/89% by concurrency. Colleague flags (`--disable-cuda-graph`, huge chunked-prefill, int8 mamba ckpt) were workarounds, not the fix.

`--mamba-ssm-dtype` is the KDA recurrent pool, **not** KV cache.

## How to use (serving)

Keep HIP graphs **on**. Opt in to fused decode, use bf16 SSM:

```bash
export SGLANG_K3_KDA_FUSED_BACKEND=aiter
python -m sglang.launch_server \
  --model <kimi-k3> \
  --tp 8 \
  --mamba-ssm-dtype bfloat16
```

Do **not** add `--disable-cuda-graph` for this path. Fusion is off unless `SGLANG_K3_KDA_FUSED_BACKEND=aiter`.

## Expected

| Check | Expect |
|---|---|
| Kernel tests | `pytest test/registered/kernels/ops/kimi_k3/flydsl_ops/test_kimi_k3_kda_decode.py -v` on gfx950. Covers fp32+bf16 and graph replay at BS 1/2/8/16. |
| Prefill conv (optional, not served) | `pytest test/registered/kernels/ops/kimi_k3/flydsl_ops/test_kimi_k3_kda_prefill_conv.py -v` needs the AITER branch. |
| Microbench | `python test/registered/kernels/ops/kimi_k3/flydsl_ops/bench_kimi_k3_kda_decode.py --state-dtype bfloat16 --batch 16` |
| Serving quality | Fused+bf16 should match **Triton-bf16** GSM8K (~95). It does **not** restore fp32-pool NIAH 64k. |
| Serving speed | Decode C16 kernel ~7% vs fp32 SSM in isolation. Prefill unchanged. |

Log should show HIP fused KDA accepted, not `K3 HIP fused KDA rejected`. rocprof kernel name: `kimi_k3_kda_decode_fb_*_ssmbf16`, not `fused_sigmoid_gating_delta_rule`.

## Not this change

A 6–8× decode MoE1/MoE2 duration drop with **the same kernel names** is not an SSM-bf16 win. Prefill MoE should be unchanged. If decode MoE GEMM collapses while `grouped_topk` time stays identical, treat it as possible empty expert tiles from bad activations, not a faster MoE binary.
