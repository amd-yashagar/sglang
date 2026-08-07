# SILOTIGER-785: native MiniMax-M3 MXFP8 AITER MoE

## Scope

This commit enables the native AITER/FlyDSL per-1x32 MoE path for
MiniMax-M3 on ROCm `gfx950`.

It provides:

- backend reachability for `--moe-runner-backend aiter`;
- native MXFP8 weight/scale preparation and gate/up interleaving;
- clamped SwiGLU activation mapping;
- bypass of the Triton-oriented MXFP8-to-block conversion;
- standard-EP top-k signaling to the paired AITER implementation.

## Atomic AITER requirement

Build SGLang and AITER as one paired image. The AITER source must contain:

- `fused_moe(..., has_fake_topk_slot=...)`;
- the SILOTIGER-785 EP4 configuration contract;
- FlyDSL `0.3.0`.

Do not use the Dockerfile's stock `AITER_COMMIT_DEFAULT`. Supply the paired
AITER repository/ref explicitly:

```bash
docker build \
  -f docker/rocm.Dockerfile \
  --build-arg BRANCH_TYPE=local \
  --build-arg GPU_ARCH=gfx950-rocm720 \
  --build-arg AITER_REPO=https://github.com/<owner>/aiter.git \
  --build-arg AITER_COMMIT=<paired-aiter-ref> \
  -t <immutable-image-tag> .
```

Record the resulting image digest and both source SHAs before benchmarking.

## Runtime arguments

```bash
export SGLANG_USE_AITER=1
export AITER_JIT_DIR=/tmp/aiter-minimax-m3-mxfp8

python -m sglang.launch_server \
  --model-path <MiniMax-M3-MXFP8-snapshot> \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --tp 4 \
  --ep-size 4 \
  --attention-backend aiter \
  --moe-runner-backend aiter
```

Use an explicit backend; do not rely on `auto` while comparing treatments.
Restart the server when changing AITER revisions or MoE backends because model
weights are transformed during loading.

No `SGLANG_USE_AITER_MOE_GU_ITLV` override is needed for native MiniMax MXFP8;
the runner forces the required interleaved gate/up layout.

## Validation

```bash
pytest -q test/registered/unit/layers/moe/test_aiter_runner.py
pytest -q test/registered/unit/test_model_overrides.py -k mxfp8
```

Deployment still requires TP4/EP4 serving, fixed-seed GSM8K, and inspection of
the log to confirm top-k 4 tuned-row selection rather than heuristic fallback.
