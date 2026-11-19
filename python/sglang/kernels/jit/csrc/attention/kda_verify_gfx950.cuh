// KDA chain spec-verify for gfx950 with the ReplaySSM ring: causal conv1d (width 4) on the packed
// q|k|v input, the gated delta-rule recurrence over the T verify tokens and the ring / conv-window
// writes, in one wave per (V tile, request, head). Same contract as the Triton
// fused_kda_conv_gating_verify for a bias-free conv (Kimi-K3's): with kSnap, no ring and a per-step
// state snapshot instead.
//
// Phase 1 computes everything that does not depend on the recurrent state for all T tokens at
// once (conv outputs rounded to bf16, q/k l2 norms, gate decays, beta) into LDS and writes the
// ring and conv-window rows. Phase 2 runs the T serial steps on a [K, BV] state tile: lane
// (c, j) holds K rows [j * KR, (j + 1) * KR) of column c, and the two per-step K reductions stay
// inside a DPP row (LPC = 64 / BV lanes per column).

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace kda_verify_detail {

constexpr int kKdaK = 128;
constexpr int kKdaV = 128;
constexpr int kWave = 64;
constexpr int kWaves = 4;
constexpr int kBlock = kWave * kWaves;

struct KdaVerifyParams {
  const uint16_t* __restrict__ x;              // [seq_len, dim] bf16, pre-conv q | k | v
  const float* __restrict__ conv_w;            // [dim, 4]
  const uint16_t* __restrict__ conv_state;     // [lines, dim, 3] bf16, dim contiguous
  const int32_t* __restrict__ cache_indices;   // [B] conv / ssm / ring slot, < 0 = padded row
  uint16_t* __restrict__ window;               // [lines, steps, dim, 3] bf16, dim contiguous
  const int32_t* __restrict__ window_indices;  // [B]
  const uint16_t* __restrict__ a;              // [seq_len, HV * K] bf16
  const uint16_t* __restrict__ b;              // [seq_len, HV] bf16
  const float* __restrict__ A_log;             // [HV]
  const float* __restrict__ dt_bias;           // [HV * K]
  const float* __restrict__ state;             // [slots, HV, V, K]
  uint16_t* __restrict__ rawv;                 // [slots, HV, L, V] bf16
  uint16_t* __restrict__ rawk;                 // [slots, H, L, K] bf16
  float* __restrict__ ring_g;                  // [slots, HV, L, K]
  float* __restrict__ ring_beta;               // [slots, HV, L]
  float* __restrict__ snap;                    // [rows, steps, HV, V, K]; row = window_indices[i_n]
  const uint16_t* __restrict__ norm_gate;       // optional [seq_len, H * V] bf16
  const void* __restrict__ norm_weight;         // optional [V], bf16 or fp32
  int64_t stride_norm_gate;
  float norm_eps;
  bool norm_weight_f32;
  uint16_t* __restrict__ o;                    // [seq_len, HV, V] bf16
  int64_t stride_x_tok;
  int64_t stride_cs_line;
  int64_t stride_cs_win;
  int64_t stride_iw_line;
  int64_t stride_iw_step;
  int64_t stride_iw_win;
  int64_t stride_a_tok;
  int64_t stride_b_tok;
  int64_t stride_state_slot;
  int64_t stride_rawv_slot;
  int64_t stride_rawk_slot;
  int64_t stride_g_slot;
  int64_t stride_beta_slot;
  int64_t stride_snap_row;
  int64_t stride_snap_step;
  float scale;
  float lower_bound;
  float softplus_beta;
  float softplus_threshold;
  int32_t num_heads;
  int32_t ring_len;
};

SGL_DEVICE float bf16_to_f32(uint16_t v) {
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

// Round to nearest even, as torch / Triton fp32 -> bf16 (inputs are finite).
SGL_DEVICE uint16_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

template <int kCtrl>
SGL_DEVICE float dpp_add(float x) {
  const int moved = __builtin_amdgcn_update_dpp(0, __float_as_int(x), kCtrl, 0xf, 0xf, false);
  return x + __int_as_float(moved);
}

// Butterfly sum over aligned groups of kLanes (4, 8 or 16) lanes inside a DPP row; every lane of
// the group ends with the group total.
template <int kLanes>
SGL_DEVICE float group_sum(float x) {
  x = dpp_add<0xb1>(x);                               // quad_perm [1, 0, 3, 2]
  x = dpp_add<0x4e>(x);                               // quad_perm [2, 3, 0, 1]
  if constexpr (kLanes >= 8) x = dpp_add<0x141>(x);   // row_half_mirror
  if constexpr (kLanes >= 16) x = dpp_add<0x140>(x);  // row_mirror
  return x;
}

SGL_DEVICE float sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// c0 * w0 + c1 * w1 + c2 * w2 + x * w3 with separately rounded products, summed from 0 as the
// unfused bias-free conv does, then SiLU and the bf16 round trip of its output.
SGL_DEVICE float conv_silu(float c0, float c1, float c2, float xt, const float (&w)[4]) {
  float acc = 0.0f;
  acc = __fadd_rn(acc, __fmul_rn(c0, w[0]));
  acc = __fadd_rn(acc, __fmul_rn(c1, w[1]));
  acc = __fadd_rn(acc, __fmul_rn(c2, w[2]));
  acc = __fadd_rn(acc, __fmul_rn(xt, w[3]));
  return acc / (1.0f + __expf(-acc));
}

template <bool kLowerBound>
SGL_DEVICE float kda_gate(float a, float dt_bias, float exp_a_log, const KdaVerifyParams& p) {
  const float x = a + dt_bias;
  if constexpr (kLowerBound) {
    return p.lower_bound * sigmoid(exp_a_log * x);
  } else {
    const float bx = p.softplus_beta * x;
    const float sp = bx <= p.softplus_threshold ? (1.0f / p.softplus_beta) * __logf(1.0f + __expf(bx)) : x;
    return -exp_a_log * sp;
  }
}

SGL_DEVICE float bf16_lane(const uint4& v, int i) {
  const uint32_t w = (&v.x)[i >> 1];
  return __uint_as_float((i & 1) ? (w & 0xffff0000u) : (w << 16));
}

SGL_DEVICE uint4 pack_bf16x8(const float (&f)[8]) {
  uint4 r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    (&r.x)[i] = static_cast<uint32_t>(f32_to_bf16(f[2 * i])) | (static_cast<uint32_t>(f32_to_bf16(f[2 * i + 1])) << 16);
  }
  return r;
}

// Raw bf16 input of channel `ch` at position `pos` of the conv sequence [cs0, cs1, cs2, x0, x1, ...]:
// the conv state for pos < 3, else token pos - 3.
SGL_DEVICE const uint16_t* seq_ptr(const KdaVerifyParams& p, const uint16_t* cs, int64_t bos, int pos, int64_t ch) {
  return pos < 3 ? cs + ch + pos * p.stride_cs_win : p.x + (bos + pos - 3) * p.stride_x_tok + ch;
}

// Conv + SiLU + bf16 round trip over 8 channels for one token: hist = sequence positions t .. t + 3.
SGL_DEVICE void conv_octet(const uint4 (&hist)[4], const float4 (&w)[8], float (&y)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float wi[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
    y[i] = bf16_to_f32(f32_to_bf16(
        conv_silu(bf16_lane(hist[0], i), bf16_lane(hist[1], i), bf16_lane(hist[2], i), bf16_lane(hist[3], i), wi)));
  }
}

// Baseline: 4 waves sharing one head's state-independent work (phase 1) for 4 * BV value columns;
// each wave then runs the serial recurrence on its own [K, BV] state tile (phase 2).
// kFullHead: 8/16 waves own all 128 V columns, prepare the head just once,
// use interleaved float4 state tiles, and optionally consume the output gate.
template <int kBV, int kT, bool kLowerBound, bool kSnap, bool kFullHead = false>
__global__ __launch_bounds__(kFullHead ? 8192 / kBV : kBlock) void
kda_verify_gfx950_kernel(const __grid_constant__ KdaVerifyParams p) {
  constexpr int kWaves = kFullHead ? 128 / kBV : 4;
  constexpr int kBlock = kWaves * kWave;
  constexpr int K = kKdaK;
  constexpr int V = kKdaV;
  constexpr int LPC = kWave / kBV;
  constexpr int KR = K / LPC;
  constexpr int kCols = kWaves * kBV;  // value columns per block
  constexpr int kVIters = (kCols * kT + kBlock - 1) / kBlock;
  // The gate and the q / k conv map the block onto 8 token groups; groups past kT idle.
  constexpr int kTokenGroups = 8;
  constexpr int kGateLanes = 256 / kTokenGroups;  // lanes per token for the gate, kGateRows rows each
  constexpr int kGateRows = K / kGateLanes;
  static_assert(LPC == 4 || LPC == 8 || LPC == 16);
  static_assert(!kFullHead || (!kSnap && kCols == V), "full-head variant requires the ring");
  static_assert(kT >= 1 && kT <= kTokenGroups, "T must be 1..8");
  static_assert(kGateRows * kGateLanes == K && kGateRows % 2 == 0, "unsupported gate mapping");

  __shared__ float s_q[kT][K];
  __shared__ float s_k[kT][K];
  __shared__ float s_decay[kT][K];
  __shared__ float s_v[kT][kCols];
  __shared__ float s_beta[kT];
  __shared__ float s_out[kFullHead ? kT : 1][kFullHead ? V : 1];

  const int tid = threadIdx.x;
  const int wave = tid / kWave;
  const int lane = tid % kWave;
  const int i_v = blockIdx.x;
  const int H = p.num_heads;
  const int i_n = blockIdx.y / H;
  const int hv = blockIdx.y % H;
  const int slot = p.cache_indices[i_n];
  if (slot < 0) return;  // padded graph row: outputs are never read
  const int64_t bos = static_cast<int64_t>(i_n) * kT;
  const bool owner = i_v == 0;
  const int widx = p.window_indices[i_n];
  const uint16_t* cs = p.conv_state + static_cast<int64_t>(slot) * p.stride_cs_line;

  // Baseline prefetches state before the first global store (CDNA vmcnt also
  // counts stores). Full-head preparation instead separates register lifetimes
  // and defers the state load until after conv/gate preparation below.
  const int c = lane / LPC;
  const int j = lane % LPC;
  const int col = i_v * kCols + wave * kBV + c;
  const int r0 = j * KR;
  // Interleave float4 chunks across the K lanes: adjacent lanes now fetch
  // adjacent 16-byte segments and address distinct LDS banks.
  const auto row_at = [=] __device__(int i) {
    return kFullHead ? j * 4 + (i / 4) * LPC * 4 + i % 4 : r0 + i;
  };
  float h[KR];
  const float* hp =
      p.state + static_cast<int64_t>(slot) * p.stride_state_slot + (static_cast<int64_t>(hv) * V + col) * K;
  if constexpr (!kFullHead) {
#pragma unroll
    for (int i = 0; i < KR; i += 4) {
      const float4 v4 = device::load_as<float4>(hp + row_at(i));
      h[i] = v4.x;
      h[i + 1] = v4.y;
      h[i + 2] = v4.z;
      h[i + 3] = v4.w;
    }
  }

  if constexpr (kFullHead) {
    // Each preparation phase has a separate register lifetime. Only the first
    // four waves prepare Q/K; remaining waves do not replicate that work.
    if (tid < 256 && (tid / 16 % 8) < kT) {
      const int t = tid / 16 % 8;
      const bool is_k = tid >= 128;
      const int c8 = tid % 16 * 8;
      const int64_t ch = (is_k ? static_cast<int64_t>(H) * K : 0) + hv * K + c8;
      uint4 hist[4];
      float4 w[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) hist[i] = device::load_as<uint4>(seq_ptr(p, cs, bos, t + i, ch));
#pragma unroll
      for (int i = 0; i < 8; ++i) w[i] = device::load_as<float4>(p.conv_w + (ch + i) * 4);
      float y[8];
      conv_octet(hist, w, y);
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += y[i] * y[i];
      const float rn = (1.0f / sqrtf(group_sum<16>(ss) + 1e-6f)) * (is_k ? 1.0f : p.scale);
      float* dst = is_k ? &s_k[t][c8] : &s_q[t][c8];
#pragma unroll
      for (int i = 0; i < 8; ++i) dst[i] = y[i] * rn;
      if (widx >= 0) {
        uint16_t* wp = p.window + widx * p.stride_iw_line + t * p.stride_iw_step + ch;
#pragma unroll
        for (int i = 0; i < 3; ++i) device::store_as<uint4>(wp + i * p.stride_iw_win, hist[i + 1]);
      }
      if (is_k && t < p.ring_len)
        device::store_as<uint4>(p.rawk + slot * p.stride_rawk_slot + (hv * p.ring_len + t) * K + c8, pack_bf16x8(y));
    }
    for (int idx = tid; idx < kT * K; idx += kBlock) {
      const int t = idx / K, r = idx % K;
      const float a = bf16_to_f32(p.a[(bos + t) * p.stride_a_tok + hv * K + r]);
      const float g = kda_gate<kLowerBound>(a, p.dt_bias[hv * K + r], __expf(p.A_log[hv]), p);
      s_decay[t][r] = __expf(g);
      if (t < p.ring_len) p.ring_g[slot * p.stride_g_slot + (hv * p.ring_len + t) * K + r] = g;
    }
    if (tid < kT) {
      const float beta = sigmoid(bf16_to_f32(p.b[(bos + tid) * p.stride_b_tok + hv]));
      s_beta[tid] = beta;
      if (tid < p.ring_len) p.ring_beta[slot * p.stride_beta_slot + hv * p.ring_len + tid] = beta;
    }
    if (tid < kT * (V / 8)) {
      const int t = tid / (V / 8), v = tid % (V / 8) * 8;
      const int64_t ch = 2LL * H * K + hv * V + v;
      uint4 hist[4];
      float4 w[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) hist[i] = device::load_as<uint4>(seq_ptr(p, cs, bos, t + i, ch));
#pragma unroll
      for (int i = 0; i < 8; ++i) w[i] = device::load_as<float4>(p.conv_w + (ch + i) * 4);
      float y[8];
      conv_octet(hist, w, y);
#pragma unroll
      for (int i = 0; i < 8; ++i) s_v[t][v + i] = y[i];
      if (widx >= 0) {
        uint16_t* wp = p.window + widx * p.stride_iw_line + t * p.stride_iw_step + ch;
#pragma unroll
        for (int i = 0; i < 3; ++i) device::store_as<uint4>(wp + i * p.stride_iw_win, hist[i + 1]);
      }
      if (t < p.ring_len)
        device::store_as<uint4>(p.rawv + slot * p.stride_rawv_slot + (hv * p.ring_len + t) * V + v, pack_bf16x8(y));
    }
    __syncthreads();
  } else {
    // q / k conv: DPP row r (16 lanes x 8 channels = one head) takes token r % 8 of q (r < 8) or k.
    const int row = tid >> 4;
    const int conv_t = min(row & 7, kT - 1);
    const bool conv_valid = tid < 256 && (row & 7) < kT;
    const bool is_k = row >= 8;
    const int c8 = (tid & 15) * 8;
    const int64_t cch = (is_k ? static_cast<int64_t>(H) * K : 0) + static_cast<int64_t>(hv) * K + c8;
    uint4 hist[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      hist[i] = device::load_as<uint4>(seq_ptr(p, cs, bos, conv_t + i, cch));
    float4 w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      w[i] = device::load_as<float4>(p.conv_w + (cch + i) * 4);

    // gate: token tid / kGateLanes, K rows [g0, g0 + kGateRows)
    const int gate_t = min(tid / kGateLanes, kT - 1);
    const bool gate_valid = tid / kGateLanes < kT;
    const int g0 = (tid % kGateLanes) * kGateRows;
    uint32_t a_raw[kGateRows / 2];
    float dt[kGateRows];
#pragma unroll
    for (int i = 0; i < kGateRows; i += 2) {
      a_raw[i / 2] = device::load_as<uint32_t>(p.a + (bos + gate_t) * p.stride_a_tok + hv * K + g0 + i);
      const float2 d = device::load_as<float2>(p.dt_bias + hv * K + g0 + i);
      dt[i] = d.x;
      dt[i + 1] = d.y;
    }
    const float a_log = p.A_log[hv];
    const float b_raw = bf16_to_f32(p.b[(bos + min(tid, kT - 1)) * p.stride_b_tok + hv]);

    // v conv: one (column, token) per lane and iteration
    uint16_t v_hist[kVIters][4];
    float4 wv[kVIters];
#pragma unroll
    for (int it = 0; it < kVIters; ++it) {
      const int idx = min(tid + it * kBlock, kCols * kT - 1);
      const int64_t vc = 2LL * H * K + static_cast<int64_t>(hv) * V + i_v * kCols + idx % kCols;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        v_hist[it][i] = *seq_ptr(p, cs, bos, idx / kCols + i, vc);
      wv[it] = device::load_as<float4>(p.conv_w + vc * 4);
    }
    asm volatile("" ::: "memory");  // keep every load above ahead of the compute and stores below

    // ---- phase 1: state-independent work for all T tokens ----
    float y[8];
    conv_octet(hist, w, y);
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      ss += y[i] * y[i];
    const float rn = (1.0f / sqrtf(group_sum<16>(ss) + 1e-6f)) * (is_k ? 1.0f : p.scale);
    if (conv_valid) {
      float* dst = is_k ? &s_k[conv_t][c8] : &s_q[conv_t][c8];
      device::store_as<float4>(dst, make_float4(y[0] * rn, y[1] * rn, y[2] * rn, y[3] * rn));
      device::store_as<float4>(dst + 4, make_float4(y[4] * rn, y[5] * rn, y[6] * rn, y[7] * rn));
    }
    const float exp_a_log = __expf(a_log);
    float g[kGateRows];
#pragma unroll
    for (int i = 0; i < kGateRows; ++i) {
      const uint32_t pair = a_raw[i / 2];
      const float a_i = __uint_as_float((i & 1) ? (pair & 0xffff0000u) : (pair << 16));
      g[i] = kda_gate<kLowerBound>(a_i, dt[i], exp_a_log, p);
      if (gate_valid) s_decay[gate_t][g0 + i] = __expf(g[i]);
    }
    const float beta = sigmoid(b_raw);
    if (tid < kT) s_beta[tid] = beta;
    uint16_t v_out[kVIters];
#pragma unroll
    for (int it = 0; it < kVIters; ++it) {
      const int idx = tid + it * kBlock;
      const float wi[4] = {wv[it].x, wv[it].y, wv[it].z, wv[it].w};
      v_out[it] = f32_to_bf16(conv_silu(
          bf16_to_f32(v_hist[it][0]),
          bf16_to_f32(v_hist[it][1]),
          bf16_to_f32(v_hist[it][2]),
          bf16_to_f32(v_hist[it][3]),
          wi));
      if (idx < kCols * kT) s_v[idx / kCols][idx % kCols] = bf16_to_f32(v_out[it]);
    }

    // ---- global stores: conv windows and ReplaySSM ring rows ----
    uint16_t* iw = p.window + static_cast<int64_t>(widx) * p.stride_iw_line;
    const int64_t ring_head = static_cast<int64_t>(hv) * p.ring_len;
    if (owner) {
      if (conv_valid) {
        if (widx >= 0) {
          uint16_t* wp = iw + conv_t * p.stride_iw_step + cch;
#pragma unroll
          for (int i = 0; i < 3; ++i)
            device::store_as<uint4>(wp + i * p.stride_iw_win, hist[i + 1]);
        }
        if (is_k && conv_t < p.ring_len) {
          device::store_as<uint4>(p.rawk + slot * p.stride_rawk_slot + (ring_head + conv_t) * K + c8, pack_bf16x8(y));
        }
      }
      if (gate_valid && gate_t < p.ring_len) {
        float* gp = p.ring_g + slot * p.stride_g_slot + (ring_head + gate_t) * K + g0;
#pragma unroll
        for (int i = 0; i < kGateRows; i += 2)
          device::store_as<float2>(gp + i, make_float2(g[i], g[i + 1]));
      }
      if (tid < kT && tid < p.ring_len) p.ring_beta[slot * p.stride_beta_slot + ring_head + tid] = beta;
    }
#pragma unroll
    for (int it = 0; it < kVIters; ++it) {
      const int idx = tid + it * kBlock;
      const int t = idx / kCols;
      const int vcol = i_v * kCols + idx % kCols;
      if (idx < kCols * kT) {
        if (widx >= 0) {
          uint16_t* wp = iw + t * p.stride_iw_step + 2LL * H * K + static_cast<int64_t>(hv) * V + vcol;
#pragma unroll
          for (int i = 0; i < 3; ++i)
            wp[i * p.stride_iw_win] = v_hist[it][i + 1];
        }
        if (t < p.ring_len) p.rawv[slot * p.stride_rawv_slot + (ring_head + t) * V + vcol] = v_out[it];
      }
    }
    // LDS visibility only needs lgkmcnt; __syncthreads() would also wait for the ring / window
    // stores above to be acknowledged.
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
  }

  // Keep the large state tile out of the conv/gate register live ranges.
  if constexpr (kFullHead) {
#pragma unroll
    for (int i = 0; i < KR; i += 4) {
      const float4 v4 = device::load_as<float4>(hp + row_at(i));
      h[i] = v4.x;
      h[i + 1] = v4.y;
      h[i + 2] = v4.z;
      h[i + 3] = v4.w;
    }
  }

  // ---- phase 2: serial recurrence on this wave's [K, BV] tile ----
  const int vcol_local = wave * kBV + c;
#pragma unroll
  for (int t = 0; t < kT; ++t) {
    float u0 = 0.0f, u1 = 0.0f;
#pragma unroll
    for (int i = 0; i < KR; i += 2) {
      h[i] *= s_decay[t][row_at(i)];
      h[i + 1] *= s_decay[t][row_at(i + 1)];
      u0 += h[i] * s_k[t][row_at(i)];
      u1 += h[i + 1] * s_k[t][row_at(i + 1)];
    }
    const float u = group_sum<LPC>(u0 + u1);
    const float vn = (s_v[t][vcol_local] - u) * s_beta[t];
    float o0 = 0.0f, o1 = 0.0f;
#pragma unroll
    for (int i = 0; i < KR; i += 2) {
      h[i] += s_k[t][row_at(i)] * vn;
      h[i + 1] += s_k[t][row_at(i + 1)] * vn;
      o0 += h[i] * s_q[t][row_at(i)];
      o1 += h[i + 1] * s_q[t][row_at(i + 1)];
    }
    const float o_sum = group_sum<LPC>(o0 + o1);
    if (j == 0) {
      if constexpr (kFullHead) s_out[t][col] = bf16_to_f32(f32_to_bf16(o_sum));
      else p.o[((bos + t) * H + hv) * V + col] = f32_to_bf16(o_sum);
    }
    if constexpr (kSnap) {
      if (widx >= 0) {
        float* sp =
            p.snap + widx * p.stride_snap_row + t * p.stride_snap_step + (static_cast<int64_t>(hv) * V + col) * K + r0;
#pragma unroll
        for (int i = 0; i < KR; i += 4)
          device::store_as<float4>(sp + i, make_float4(h[i], h[i + 1], h[i + 2], h[i + 3]));
      }
    }
  }
  if constexpr (kFullHead) {
    __syncthreads();
    // Each wave owns one token and two V channels/lane. Full-head ownership
    // makes the output norm local, with no global scratch or extra launch.
    for (int t = wave; t < kT; t += kWaves) {
      const float x0 = s_out[t][lane], x1 = s_out[t][lane + 64];
      float inv = 1.0f;
      if (p.norm_gate) {
        float ss = x0 * x0 + x1 * x1;
#pragma unroll
        for (int mask = 32; mask > 0; mask >>= 1) ss += __shfl_xor(ss, mask, 64);
        inv = rsqrtf(ss / V + p.norm_eps);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int v = lane + i * 64;
        float out = i ? x1 : x0;
        if (p.norm_gate) {
          const float w = p.norm_weight_f32 ? static_cast<const float*>(p.norm_weight)[v]
              : bf16_to_f32(static_cast<const uint16_t*>(p.norm_weight)[v]);
          const float gate = bf16_to_f32(p.norm_gate[(bos + t) * p.stride_norm_gate + hv * V + v]);
          out = (out * inv * w) * sigmoid(gate);
        }
        p.o[((bos + t) * H + hv) * V + v] = f32_to_bf16(out);
      }
    }
  }
}

}  // namespace kda_verify_detail

template <int kBV, int kT, bool kLowerBound, bool kSnap, bool kFullHead = false>
struct KdaVerifyGfx950Kernel {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView conv_w,
      const tvm::ffi::TensorView conv_state,
      const tvm::ffi::TensorView cache_indices,
      const tvm::ffi::TensorView window,
      const tvm::ffi::TensorView window_indices,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView A_log,
      const tvm::ffi::TensorView dt_bias,
      const tvm::ffi::TensorView state,
      const tvm::ffi::TensorView rawv,
      const tvm::ffi::TensorView rawk,
      const tvm::ffi::TensorView ring_g,
      const tvm::ffi::TensorView ring_beta,
      const tvm::ffi::Optional<tvm::ffi::TensorView> snap,
      const tvm::ffi::Optional<tvm::ffi::TensorView> norm_gate,
      const tvm::ffi::Optional<tvm::ffi::TensorView> norm_weight,
      const double norm_eps,
      const tvm::ffi::TensorView o,
      const int64_t num_heads,
      const double scale,
      const double lower_bound,
      const double softplus_beta,
      const double softplus_threshold) {
    using namespace host;
    using namespace kda_verify_detail;
    const int64_t seq_len = x.size(0);
    const int64_t batch = seq_len / kT;
    RuntimeCheck(batch * kT == seq_len, "seq_len must be a multiple of T");
    RuntimeCheck(x.size(1) == 3 * num_heads * kKdaK && x.stride(1) == 1, "x must be [seq_len, 3 * H * 128]");
    RuntimeCheck(conv_state.stride(1) == 1 && conv_state.size(2) == 3, "conv_state must be [lines, dim, 3]");
    RuntimeCheck(window.stride(2) == 1 && window.size(3) == 3, "window must be [lines, steps, dim, 3]");
    RuntimeCheck(window.size(1) >= kT, "window has fewer steps than T");
    RuntimeCheck(
        conv_w.stride(0) == 4 && conv_w.stride(1) == 1 && conv_w.size(1) == 4, "conv weight must be a dense [dim, 4]");
    RuntimeCheck(
        x.stride(0) % 8 == 0 && conv_state.stride(2) % 8 == 0 && window.stride(3) % 8 == 0 && a.stride(0) % 8 == 0,
        "channel octets must be 16-byte aligned");
    RuntimeCheck(a.stride(1) == 1 && b.stride(1) == 1, "a and b must be row-contiguous");
    RuntimeCheck(
        state.stride(3) == 1 && state.stride(2) == kKdaK && state.stride(1) == kKdaK * kKdaV,
        "state must be [slots, HV, V, K] with contiguous heads");
    RuntimeCheck(
        rawv.stride(2) == kKdaV && rawk.stride(2) == kKdaK && ring_g.stride(2) == kKdaK,
        "ring rows must be contiguous");
    if constexpr (kSnap) {
      RuntimeCheck(snap.has_value(), "the snapshot variant needs the intermediate state buffer");
      const tvm::ffi::TensorView sv = snap.value();
      RuntimeCheck(
          sv.stride(4) == 1 && sv.stride(3) == kKdaK && sv.stride(2) == kKdaK * kKdaV && sv.size(1) >= kT,
          "snapshots must be [rows, steps >= T, HV, V, K] with contiguous heads");
      RuntimeCheck(rawv.size(2) == 0, "the snapshot variant writes no ring");
    }
    RuntimeCheck(norm_gate.has_value() == norm_weight.has_value(), "norm requires both gate and weight");
    if (norm_gate.has_value()) {
      RuntimeCheck(kFullHead, "norm requires the full-head kernel");
      const auto ng = norm_gate.value();
      const auto nw = norm_weight.value();
      RuntimeCheck(ng.ndim() == 2 && ng.size(0) == seq_len && ng.size(1) == num_heads * kKdaV && ng.stride(1) == 1,
                   "norm gate must be row-contiguous [seq_len, H * V]");
      RuntimeCheck(ng.dtype().code == kDLBfloat && ng.dtype().bits == 16, "norm gate must be bf16");
      RuntimeCheck(nw.ndim() == 1 && nw.size(0) == kKdaV && nw.stride(0) == 1,
                   "norm weight must be contiguous [V]");
      RuntimeCheck((nw.dtype().code == kDLBfloat && nw.dtype().bits == 16) ||
                   (nw.dtype().code == kDLFloat && nw.dtype().bits == 32), "norm weight must be bf16 or fp32");
    }
    if (batch == 0) return;
    const KdaVerifyParams params{
        .x = static_cast<const uint16_t*>(x.data_ptr()),
        .conv_w = static_cast<const float*>(conv_w.data_ptr()),
        .conv_state = static_cast<const uint16_t*>(conv_state.data_ptr()),
        .cache_indices = static_cast<const int32_t*>(cache_indices.data_ptr()),
        .window = static_cast<uint16_t*>(window.data_ptr()),
        .window_indices = static_cast<const int32_t*>(window_indices.data_ptr()),
        .a = static_cast<const uint16_t*>(a.data_ptr()),
        .b = static_cast<const uint16_t*>(b.data_ptr()),
        .A_log = static_cast<const float*>(A_log.data_ptr()),
        .dt_bias = static_cast<const float*>(dt_bias.data_ptr()),
        .state = static_cast<const float*>(state.data_ptr()),
        .rawv = static_cast<uint16_t*>(rawv.data_ptr()),
        .rawk = static_cast<uint16_t*>(rawk.data_ptr()),
        .ring_g = static_cast<float*>(ring_g.data_ptr()),
        .ring_beta = static_cast<float*>(ring_beta.data_ptr()),
        .snap = kSnap ? static_cast<float*>(snap.value().data_ptr()) : nullptr,
        .norm_gate = norm_gate.has_value() ? static_cast<const uint16_t*>(norm_gate.value().data_ptr()) : nullptr,
        .norm_weight = norm_weight.has_value() ? norm_weight.value().data_ptr() : nullptr,
        .stride_norm_gate = norm_gate.has_value() ? norm_gate.value().stride(0) : 0,
        .norm_eps = static_cast<float>(norm_eps),
        .norm_weight_f32 = norm_weight.has_value() && norm_weight.value().dtype().bits == 32,
        .o = static_cast<uint16_t*>(o.data_ptr()),
        .stride_x_tok = x.stride(0),
        .stride_cs_line = conv_state.stride(0),
        .stride_cs_win = conv_state.stride(2),
        .stride_iw_line = window.stride(0),
        .stride_iw_step = window.stride(1),
        .stride_iw_win = window.stride(3),
        .stride_a_tok = a.stride(0),
        .stride_b_tok = b.stride(0),
        .stride_state_slot = state.stride(0),
        .stride_rawv_slot = rawv.stride(0),
        .stride_rawk_slot = rawk.stride(0),
        .stride_g_slot = ring_g.stride(0),
        .stride_beta_slot = ring_beta.stride(0),
        .stride_snap_row = kSnap ? snap.value().stride(0) : 0,
        .stride_snap_step = kSnap ? snap.value().stride(1) : 0,
        .scale = static_cast<float>(scale),
        .lower_bound = static_cast<float>(lower_bound),
        .softplus_beta = static_cast<float>(softplus_beta),
        .softplus_threshold = static_cast<float>(softplus_threshold),
        .num_heads = static_cast<int32_t>(num_heads),
        .ring_len = static_cast<int32_t>(rawv.size(2)),
    };
    constexpr int kWaves = kFullHead ? 128 / kBV : 4;
    constexpr int kBlock = kWaves * kWave;
    const dim3 grid(kKdaV / (kWaves * kBV), static_cast<uint32_t>(batch * num_heads));
    LaunchKernel(grid, kBlock, x.device())(kda_verify_gfx950_kernel<kBV, kT, kLowerBound, kSnap, kFullHead>, params);
  }
};

}  // namespace sglang
