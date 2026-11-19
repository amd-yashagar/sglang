// SPDX-License-Identifier: Apache-2.0
// MFMA / wave reduction adapted from the existing
// dsa_gfx950/paged_mqa_logits.cuh. Each wave scores one 64-token page against
// five causal query rows. Hoist Q/weights, load index-K once, and prefetch the
// next page into registers. The existing exact top-k consumes the output; no
// histogram/LDS workspace.
#include <cstdint>
#include <hip/hip_runtime.h>
typedef __attribute__((__vector_size__(8 * sizeof(int)))) int i32x8;
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int i32x4;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float f32x4;

#define PAGE_TOK 64
#define HD 128
#define HEADS 32
#define TOK_STRIDE 132
#define PAGE_BYTES (PAGE_TOK * TOK_STRIDE) // 8448
#define K_BYTES (PAGE_TOK * HD)            // 8192

union V8 {
  i32x8 v;
  i32x4 h[2];
};

__device__ __forceinline__ f32x4 mfma128(i32x8 a, i32x8 b, f32x4 c) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127,
                                                          0, 127);
}

// wave64 lane exchanges on the VALU instead of through LDS:
//   xor32 <- v_permlane64_b32 (lane l reads lane l^32)
//   xor16 <- v_permlanex16_b32 with identity selectors (lane l reads lane l^16)
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int i32x2;
__device__ __forceinline__ float xor32(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane32_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 32) ? r[0] : r[1]);
}
__device__ __forceinline__ float xor16(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane16_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 16) ? r[0] : r[1]);
}

template <int N, int WARPS = 4>
__global__ __launch_bounds__(WARPS * 64) void grouped_logits(
    const uint8_t *__restrict__ q, const uint8_t *__restrict__ kv,
    const float *__restrict__ weights, const int *__restrict__ lens,
    const int *__restrict__ tables, float *__restrict__ out, int pages,
    int width) {
  const int req = blockIdx.y, lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
  const int g = lane >> 4, c = lane & 15;
  const int length = max(0, min(lens[req], width));
  const int npages = min((length + 63) / 64, pages);
  // Two 16-head MFMA groups cover H32; each lane carries 32 dimensions.
  V8 qa0[N], qa1[N];
  f32x4 w0[N], w1[N];
#pragma unroll
  for (int n = 0; n < N; ++n) {
    const uint8_t *qp = q + ((size_t)req * N + n) * 4096 + c * 128 + g * 32;
    qa0[n].h[0] = *(const i32x4 *)qp;
    qa0[n].h[1] = *(const i32x4 *)(qp + 16);
    qa1[n].h[0] = *(const i32x4 *)(qp + 2048);
    qa1[n].h[1] = *(const i32x4 *)(qp + 2064);
    const float *wp = weights + ((size_t)req * N + n) * 32 + g * 4;
    w0[n] = *(const f32x4 *)wp;
    w1[n] = *(const f32x4 *)(wp + 16);
  }
  const int step = gridDim.x * WARPS;
  const int first = blockIdx.x * WARPS + warp;
  if (first >= npages)
    return;
  const int koff = g * 512 + c * 16;
  const uint8_t *firstbase = kv + (size_t)tables[req * pages + first] * 8448;
  V8 nextb[4];
#pragma unroll
  for (int tt = 0; tt < 4; ++tt) {
    nextb[tt].h[0] = *(const i32x4 *)(firstbase + koff + tt * 2048);
    nextb[tt].h[1] = *(const i32x4 *)(firstbase + koff + tt * 2048 + 256);
  }
  float nextscale = ((const float *)(firstbase + 8192))[lane];
  for (int p = first; p < npages; p += step) {
    V8 b[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt)
      b[tt] = nextb[tt];
    const float scale = nextscale;
    if (p + step < npages) {
      const uint8_t *base = kv + (size_t)tables[req * pages + p + step] * 8448;
#pragma unroll
      for (int tt = 0; tt < 4; ++tt) {
        nextb[tt].h[0] = *(const i32x4 *)(base + koff + tt * 2048);
        nextb[tt].h[1] = *(const i32x4 *)(base + koff + tt * 2048 + 256);
      }
      nextscale = ((const float *)(base + 8192))[lane];
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int n = 0; n < N; ++n) {
      float s[4];
#pragma unroll
      for (int tt = 0; tt < 4; ++tt) {
        f32x4 z = {0, 0, 0, 0};
        f32x4 a0 = mfma128(qa0[n].v, b[tt].v, z),
              a1 = mfma128(qa1[n].v, b[tt].v, z);
        float acc = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          acc = fmaf(fmaxf(a0[i], 0.f), w0[n][i], acc);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          acc = fmaf(fmaxf(a1[i], 0.f), w1[n][i], acc);
        s[tt] = acc;
      }
      // Reduce head contributions in registers; lane l writes key p*64+l.
      const bool hi = (g & 2) != 0, odd = (g & 1) != 0;
      float a0 = (hi ? s[2] : s[0]) + xor32(hi ? s[0] : s[2], lane);
      float a1 = (hi ? s[3] : s[1]) + xor32(hi ? s[1] : s[3], lane);
      float val = ((odd ? a1 : a0) + xor16(odd ? a0 : a1, lane)) * scale;
      int pos = p * 64 + lane;
      if (pos < width)
        out[((size_t)req * N + n) * width + pos] =
            pos < length - N + 1 + n ? val : -INFINITY;
    }
  }
}
// ABI deliberately has no build/JIT or device-discovery side effects.
extern "C" int launch_grouped_mqa(const void *q, const void *kv, const void *w,
                                  const void *lens, const void *tables,
                                  void *out, int batch, int next_n, int pages,
                                  int width, int splits, void *stream) {
  if (batch == 0)
    return (int)hipSuccess;
  if (batch < 0 || next_n != 5 || pages < 1 || width < 1 || width % 64 ||
      pages != width / 64 || splits < 1)
    return (int)hipErrorInvalidValue;
  dim3 grid(splits, batch), block(256);
  hipLaunchKernelGGL((grouped_logits<5>), grid, block, 0, (hipStream_t)stream,
                     (const uint8_t *)q, (const uint8_t *)kv, (const float *)w,
                     (const int *)lens, (const int *)tables, (float *)out,
                     pages, width);
  return (int)hipGetLastError();
}
