// pouw_hash.cuh's vectors from the device and from the host (the same run_checks):
//   pouw_hash_host check     (g++ -x c++ pouw_hash_check.cu, no CUDA) the host build's lines
//   device_checks()          the device's: pouw_hash_bench check (pouw_hash_bench.cu) launches these kernels
// Here Keccak is not inlined (PH_KECCAK_INLINE): the check instantiates about fifty sponges, and inlined they take cicc over ten
// minutes. The timed kernels inline it, and `epi`'s gate checks the fused path's digests in the timed kernel itself.
#ifdef __CUDACC__
#define PH_KECCAK_INLINE __host__ __device__ __noinline__
#endif
#include "vectors.cuh"

#ifndef __CUDACC__
// ---------------------------------------------------------------- the host build
int main(int argc, char** argv) {
  if (argc < 2 || strcmp(argv[1], "check")) {
    fprintf(stderr, "usage: pouw_hash_host check\n");
    return 2;
  }
  static uint8_t buf[4096];
  static CheckOut o;
  run_checks<0>(buf, &o), run_checks<1>(buf, &o), run_checks<2>(buf, &o), run_checks<3>(buf, &o), run_checks<4>(buf, &o);
  run_checks<5>(buf, &o), run_checks<6>(buf, &o);
  // the warp and block trees are the device's; on the host they are the one-thread tree they must equal
  for (int i = 0; i < 8; ++i) {
    o.b3tree_warp[0][i] = o.b3tree[0][i], o.b3tree_warp[1][i] = o.b3tree[2][i];
    o.b3fine_warp[i] = o.b3fine[0][i], o.b3fine_block[i] = o.b3fine[2][i];
  }
  uint32_t key[8];
  b3_test_key(key);
  for (int lane = 0; lane < 32; ++lane) {
    uint8_t msg[256], out[32];
    for (int a = 0; a < 2; ++a)
      for (int j = 0; j < 32; ++j) {
        const uint32_t w = quad_word(a, ph::quad_row(lane), 32 * ph::quad_chunk(lane) + j);
        for (int b = 0; b < 4; ++b) msg[128 * a + 4 * j + b] = (uint8_t)(w >> (8 * b));
      }
    ph::blake3_chunk(key, msg, 256, out);
    for (int i = 0; i < 8; ++i) o.quad_b3[lane][i] = (uint32_t)out[4 * i] | (uint32_t)out[4 * i + 1] << 8 | (uint32_t)out[4 * i + 2] << 16 | (uint32_t)out[4 * i + 3] << 24;
  }
  ph::quad_msg_digest_warp<2>([](int lane, int a, int j, int i) { return quad_frag(lane, a, j, i); }, 0x21, o.quad);
  uint64_t dg[4 * ph::pearlc::DIGESTS], d[32][4];
  for (int region = 0; region < 4; ++region) {
    ph::quad_msg_digest_warp<2>([&](int lane, int a, int j, int i) { return tile_frag(region, lane, a, j, i); }, 0x21, d);
    for (int lane = 0; lane < 32; ++lane)
      for (int i = 0; i < 4; ++i) dg[4 * ph::pearlc::digest_index(region, lane) + i] = d[lane][i];
  }
  ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o.pc_leaf_quad);
  for (int chunk = 0; chunk < 2; ++chunk)
    for (int p = 0; p < 2; ++p) {
      ph::quad_msg_digest_warp<2>([&](int lane, int a, int j, int i) { return narrow_frag(chunk, p, lane, a, j, i); }, 0x21, d);
      for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i) dg[4 * (2 * ph::pearlc::narrow_row(p, lane) + chunk) + i] = d[lane][i];
    }
  ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o.pc_leaf_narrow);
  print_checks(o, "host");
  return 0;
}
#else
#include <cuda_runtime.h>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

template <int PART>
__global__ void k_check(uint8_t* buf, CheckOut* o) {
  run_checks<PART>(buf, o);
}
__global__ void k_check_quad(CheckOut* o, uint64_t* dg) {
  __shared__ uint64_t scratch[128];
  const int lane = threadIdx.x;
  uint64_t d[4];
  ph::quad_msg_digest<2>([&](int a, int j, int i) { return quad_frag(lane, a, j, i); }, 0x21, scratch, d);
  for (int i = 0; i < 4; ++i) o->quad[lane][i] = d[i];
  for (int region = 0; region < 4; ++region) {
    ph::quad_msg_digest<2>([&](int a, int j, int i) { return tile_frag(region, lane, a, j, i); }, 0x21, scratch, d);
    for (int i = 0; i < 4; ++i) dg[4 * ph::pearlc::digest_index(region, lane) + i] = d[i];
  }
  __syncwarp();
  if (lane == 0) ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o->pc_leaf_quad);
  __syncwarp();
  for (int chunk = 0; chunk < 2; ++chunk)
    for (int p = 0; p < 2; ++p) {
      ph::quad_msg_digest<2>([&](int a, int j, int i) { return narrow_frag(chunk, p, lane, a, j, i); }, 0x21, scratch, d);
      for (int i = 0; i < 4; ++i) dg[4 * (2 * ph::pearlc::narrow_row(p, lane) + chunk) + i] = d[i];
    }
  __syncwarp();
  if (lane == 0) ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o->pc_leaf_narrow);
  uint32_t key[8], h[8];
  b3_test_key(key);
  ph::quad_msg_digest_b3<2>([&](int a, int j, int i) { return quad_frag(lane, a, j, i); }, key, scratch, h);
  for (int i = 0; i < 8; ++i) o->quad_b3[lane][i] = h[i];
  ph::b3_iv(key);
  ph::warp_blake3_tree<2>(key, 0, 16, [](int c, int i) { return ptn_le_word(256 * c + i); }, h);
  if (lane == 0)
    for (int i = 0; i < 8; ++i) o->b3tree_warp[0][i] = h[i];
  ph::warp_blake3_tree<32>(key, 0, 16, [](int c, int i) { return ptn_le_word(256 * c + i); }, h);
  if (lane == 0)
    for (int i = 0; i < 8; ++i) o->b3tree_warp[1][i] = h[i];
  ph::warp_blake3_tree<32>(key, 0, 1, [](int c, int i) { return ptn_le_word(16 * c + i); }, h);
  if (lane == 0)
    for (int i = 0; i < 8; ++i) o->b3fine_warp[i] = h[i];
}
__global__ void k_check_fine(CheckOut* o) {
  __shared__ uint32_t sh[16 * 8];
  uint32_t iv[8], h[8];
  ph::b3_iv(iv);
  ph::block_blake3_fine<512>(iv, 0, [](int t, int i) { return ptn_le_word(16 * t + i); }, sh, h);
  if (threadIdx.x == 0)
    for (int i = 0; i < 8; ++i) o->b3fine_block[i] = h[i];
}

void device_checks(CheckOut* h) {
  uint8_t* buf;
  CheckOut* o;
  uint64_t* dg;
  CK(cudaMalloc(&buf, 4096)); CK(cudaMalloc(&o, sizeof(CheckOut))); CK(cudaMalloc(&dg, 4096));
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 32768));
  k_check<0><<<1, 1>>>(buf, o), k_check<1><<<1, 1>>>(buf, o), k_check<2><<<1, 1>>>(buf, o);
  k_check<3><<<1, 1>>>(buf, o), k_check<4><<<1, 1>>>(buf, o), k_check<5><<<1, 1>>>(buf, o), k_check<6><<<1, 1>>>(buf, o);
  CK(cudaDeviceSynchronize());
  k_check_quad<<<1, 32>>>(o, dg);
  k_check_fine<<<1, 512>>>(o);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, o, sizeof *h, cudaMemcpyDeviceToHost));
  cudaFree(buf), cudaFree(o), cudaFree(dg);
}
#endif
