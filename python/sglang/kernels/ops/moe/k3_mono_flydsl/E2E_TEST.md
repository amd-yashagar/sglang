# Kimi-K3 decode monokernel — e2e test instructions

**Repo:** SGLang only (`amd-yashagar/sglang`).  
**Branch:** `k3-monokernel-fuse-opt` (this work on top of the [#43481](https://github.com/sgl-project/sglang/pull/43481) integrate).  
**AITER:** no fork branch required — use your normal ROCm/AITER install (`SGLANG_USE_AITER=1`). This path does **not** depend on MegaMoE / Aiter #6259.

Target: **MI355X / gfx950, TP=8**, decode rows **≤ 64** (LDS tile 8; host
tiling for 16/32/64), weekly **A4W4** (`AITER_SITUV2_A4W4=1`).

## What this branch adds (vs staged serve)

One FlyDSL launch covers MoE (and optionally K2: o_proj + Attn-TP AR + AttnRes seam):

| Env | Effect |
|---|---|
| `SGLANG_K3_MONO_DECODE=1` | MoE monokernel (router, experts, LN, up, in-kernel ARs) |
| `SGLANG_K3_MONO_LAYER=1` | + o_proj + attn AR + AttnRes into the same launch |
| `SGLANG_K3_MONO_K1=0` | **Keep off** for e2e (graph-unsafe host index reads) |

Also on this branch: `fuse_resid` (residual in-kernel), buffer prealloc, A4W4 native messaging, latq/route pipeline overlap.

## Checkout

```bash
git clone git@github.com:amd-yashagar/sglang.git
cd sglang
git fetch origin k3-monokernel-fuse-opt
git checkout k3-monokernel-fuse-opt
# install / PYTHONPATH as you usually do for this image
export PYTHONPATH=$PWD/python:${PYTHONPATH}
```

Optional baseline without the fuse/perf commits: branch `k3-monokernel` (PR integrate only), same fork.

## Microbench (no weights; 8 GPUs)

Confirms accuracy + MoE-layer speedup before serve:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /path/to/sglang

PYTHONPATH=python python3 test/registered/amd/test_kimi_k3_mono_decode.py
PYTHONPATH=python python3 test/registered/amd/test_kimi_k3_mono_decode.py --guint
PYTHONPATH=python python3 test/registered/amd/test_kimi_k3_mono_kda_pre.py
```

Expect MoE cos ≥ 0.999 vs staged AITER; K2 seam cos ~1.0.  
On gfx950 TP8, MoE-layer HIP timing vs staged+2×RCCL: **~3–5.5×** at rows
1/4/8; host-tiled **~1.9×** at 16; **~0.9×** at 32 (tile tax — improve next);
**~4×** at 64 vs tile-staged (full AITER sorter broken at 64 here). Not e2e
tokens/s. Wide sweep: `kimi-k3/monokernel/timed_mono_vs_staged.py`.

## Serve A/B (needs Kimi-K3 checkpoint)

Use the **same image, same MODEL, same conc**, only change mono flags / port.

### Shared env (both arms)

```bash
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_FLYDSL_FORCE=1
export AITER_SITUV2_A4W4=1          # weekly path; mono's native expert layout
unset AITER_SITUV2_A8W4
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1
export PYTHONPATH=/path/to/sglang/python:${PYTHONPATH}
MODEL=/path/to/Kimi-K3               # required
```

### Baseline (staged)

```bash
export SGLANG_K3_MONO_DECODE=0
export SGLANG_K3_MONO_LAYER=0
export SGLANG_K3_MONO_K1=0

python3 -m sglang.launch_server \
  --model "$MODEL" \
  --port 30000 \
  --tensor-parallel-size 8 \
  --prefill-attention-backend aiter \
  --decode-attention-backend aiter \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --max-running-requests 256 \
  --cuda-graph-max-bs-decode 256 \
  --mamba-full-memory-ratio 0.9 \
  --mamba-ssm-dtype bfloat16 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --trust-remote-code \
  --disable-radix-cache
```

### Mono treatment (MoE + K2)

```bash
export SGLANG_K3_MONO_DECODE=1
export SGLANG_K3_MONO_LAYER=1
export SGLANG_K3_MONO_K1=0

python3 -m sglang.launch_server \
  --model "$MODEL" \
  --port 30001 \
  # ... same flags as baseline ...
```

Helper script (if you have the notes tree):  
`QUANT=a4w4 MODE=baseline|mono MODEL=... ./run_serve_ab.sh`

### Conc / fallback

- Keep decode batch so **rows ≤ 8**; above that the host falls back to staged kernels.
- `--moe-a2a-backend none`, no DP attention.
- Log once per process: `Kimi-K3 mono decode engaged on a decode step` when mono runs.

## What to measure

| Check | How |
|---|---|
| Correctness | GSM8K (or your usual K3 suite) baseline vs mono, same seeds/prompts |
| Perf | TPOT / step time at low conc (rows 1–8); do **not** expect MoE microbench × to equal e2e |
| Trace | Mono: fused FlyDSL `k3_mono_*` launches; fewer standalone `cross_device_reduce` / `_agg_kernel` on the K2 seam for decode |
| Sanity | With mono flags **0**, behavior matches your weekly A4W4 baseline |

## Out of scope on this branch

- Prefill / FlyTape whole-request fusion  
- EP / MegaMoE  
- `SGLANG_K3_MONO_K1=1` under CUDA graphs  
- Claiming a fixed e2e % without a weights-present A/B on your box  

## Contact / provenance

- Upstream integrate: SGLang PR #43481 / vLLM-inspired fuse (host differs).  
- This branch: fuse_resid + host prealloc + MoE latq/route pipeline on gfx950 TP8.
