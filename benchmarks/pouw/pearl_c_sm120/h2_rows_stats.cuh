// -h2's rows form `s`: A's row leaves (hash_rows_b3s) and stats_s5's outputs, bit for bit, in one pass over A.
// pearl_c_sm120.cu includes this after stats_s5, whose S5_LINE and S5_FLOOR it reads; simt_twin.py builds it on the host.
#pragma once

// A CTA a row over the stats' i4 rows (m), hashing the first i0 (m_real); thread j's segment is FP32 words 64j .. 64j +
// 63, whose max |x| and every 8th word it keeps as it hashes them (rows_b3s).  Then warp 0 is stats_s5's warp: lane c
// runs chain c (x[256 i + 8 c], i in order, one FMA each), s is the row's max |x| (a max, so the same in any order), and
// the butterfly and alpha, beta, 1/alpha, (s, rho) are stats_s5's.  p0 .. p4 and i0 .. i3 are hash_rows_b3s's (i1 = L =
// 4k bytes, k a multiple of 256; p0 16-byte aligned), p5 alpha, p6 beta, p7 1/alpha, p8 (s, rho), i4 the stats' rows; 32
// ⌈L / 8,192⌉ threads, and b3s_stats_smem(L) bytes of dynamic shared memory: the segment tree, then the samples
constexpr uint32_t b3s_stats_smem(uint32_t L) { return (L / B3S_SEG) * (32 + 33); }
extern "C" __global__ void __launch_bounds__(512) hash_rows_b3s_stats(const Args a) {
  extern __shared__ uint32_t sh[];
  __shared__ uint32_t lks[B3S_MAX_LEVELS][8];
  __shared__ float wmax[B3S_MAX_SEGS / 32];
  if (blockIdx.x >= (uint32_t)a.i[4]) return;
  const uint32_t L = (uint32_t)a.i[1], nseg = L / B3S_SEG, k = L / 4;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  uint32_t(*lv)[8] = reinterpret_cast<uint32_t(*)[8]>(sh);
  float* samp = reinterpret_cast<float*>(sh + 8 * nseg);
  float s = rows_b3s<true>(a, lv, lks, samp);
#pragma unroll
  for (int o = 16; o; o >>= 1) s = fmaxf(s, __shfl_xor_sync(~0u, s, o));
  if (lane == 0) wmax[warp] = s;
  __syncthreads();
  if (warp) return;
  s = lane < (int)(blockDim.x / 32) ? wmax[lane] : 0.0f;
  float part = 0.0f;
  for (uint32_t i = 0; i < nseg / 4; i++) {
    const float e = samp[33 * i + lane];
    part = __fmaf_rn(e, e, part);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { s = fmaxf(s, __shfl_xor_sync(~0u, s, o)); part = __fadd_rn(part, __shfl_xor_sync(~0u, part, o)); }
  if (lane) return;
  const uint64_t row = blockIdx.x;
  const float rho = __fsqrt_rn(__fdiv_rn(part, (float)((k + 7) / 8)));
  const float alpha = __fdiv_rn(448.0f, fmaxf(__fadd_rn(s, __fmul_rn(4.0f, rho)), S5_FLOOR));
  reinterpret_cast<float*>(a.p[5])[row] = alpha;
  reinterpret_cast<float*>(a.p[6])[row] = __fmul_rn(__fmul_rn(rho, alpha), __uint_as_float(S5_LINE));
  reinterpret_cast<float*>(a.p[7])[row] = __frcp_rn(alpha);
  reinterpret_cast<float2*>(a.p[8])[row] = make_float2(s, rho);
}
