// H1's per-unit noise (seeds.noise_block: block t of a seed's stream is SHA-512(NOISE_TAG || seed || u64be t), one
// compression of a 96-byte message) on sm_90a / sm_100a, the floor of any H1 + fx1 kernel on these cards: a unit derives
// h1tile.Layout's `blocks` from seed_j and `lanes` from sigma_R (576 + 32 at K = 4096, 1856 + 32 at K = 14336).
//
//   ca     compute accounting's benchmarks/pouw/served_h1/noise.cu `h1_noise`, copied verbatim but for the compression:
//          one thread a noise chunk, sx::compress<0, 2> (verity/ml/kernels/sha512.cuh, K from constant memory)
//   mid    one warp an (S, chunk) item, J blocks a lane (t = chunk 32 J + 32 j + lane): the message's words 0..10 are the
//          seed's, so rounds 0..10 (the midstate) and every schedule term free of word 11 (t) are computed once an item,
//          then each block runs rounds 11..79 on them (h1sha.cuh's noise_mid and noise_block, h1u.cu's own)
// Both write `out` as h1_noise does (S x nb x 16 big-endian halves), or with sink one word a thread: h1_noise's per-block
// XOR of its halves; mid's per (item, lane) XOR over its J blocks.
//
//   kn_ca(seeds, S, nb, out, sink, stream)            kn_mid(seeds, S, nb, J, out, sink, stream)
//   kn_mid_items(S, nb, J)                            items (warps) of a mid launch (sink words = 32 x items)
#include "h1sha.cuh"

// ---- ca: served_h1/noise.cu's kernel, verbatim ----------------------------------------------------------------------
extern "C" __global__ void h1_noise(const u64* __restrict__ seeds, u64 n, u32 nb, u64 tag0, u64 tag1, u64 tag2,
                                    u32* __restrict__ out, int sink) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nb) return;
  const u64 u = i / nb;
  const u64 t = i % nb;
  u64 w[16];
  w[0] = tag0;
  w[1] = tag1;
  w[2] = tag2;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[3 + j] = seeds[8 * u + j];
  w[11] = t;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 96 * 8;
  u64 s[8];
  sx::iv(s);
  sx::compress<0, 2>(s, w);
  if (sink) {
    u32 x = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) x ^= (u32)(s[j] >> 32) ^ (u32)s[j];
    out[i] = x;
    return;
  }
  u32* o = out + 16 * i;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    o[2 * j] = (u32)(s[j] >> 32);
    o[2 * j + 1] = (u32)s[j];
  }
}

// ---- mid ----------------------------------------------------------------------------------------------------------
template <bool SINK>
__global__ void __launch_bounds__(256) h1_noise_mid(const u64* __restrict__ seeds, u32 S, u32 nb, u32 J,
                                                    u32* __restrict__ out) {
  const u32 lane = threadIdx.x & 31;
  const u32 per = 32 * J, chunks = (nb + per - 1) / per;
  const u64 item = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (item >= (u64)S * chunks) return;
  const u32 u = (u32)(item / chunks), chunk = (u32)(item % chunks);
  u64 seed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) seed[j] = __ldg(seeds + 8 * (u64)u + j);
  NoiseMid nm;
  noise_mid(seed, nm);
  u32 acc = 0;
#pragma unroll 1
  for (u32 j = 0; j < J; ++j) {
    const u32 t = chunk * per + 32 * j + lane;
    if (t >= nb) break;
    u64 s[8];
    noise_block(nm, t, s);
    if (SINK) {
#pragma unroll
      for (int q = 0; q < 8; ++q) acc ^= (u32)(s[q] >> 32) ^ (u32)s[q];
    } else {
      uint4* o = reinterpret_cast<uint4*>(out + 16 * ((u64)u * nb + t));
#pragma unroll
      for (int q = 0; q < 4; ++q)
        o[q] = make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]);
    }
  }
  if (SINK) out[item * 32 + lane] = acc;
}
#undef RND
#undef dep

extern "C" int kn_ca(const void* seeds, unsigned long long S, unsigned nb, void* out, int sink, void* stream) {
  const unsigned long long total = S * nb;
  if (!total) return 0;
  h1_noise<<<(unsigned)((total + 127) / 128), 128, 0, (cudaStream_t)stream>>>((const u64*)seeds, S, nb, TAG0, TAG1, TAG2,
                                                                              (u32*)out, sink);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" unsigned long long kn_mid_items(unsigned S, unsigned nb, unsigned J) {
  return (unsigned long long)S * ((nb + 32 * J - 1) / (32 * J));
}

extern "C" int kn_mid(const void* seeds, unsigned S, unsigned nb, unsigned J, void* out, int sink, void* stream) {
  if (!J) return 2;
  const unsigned long long items = kn_mid_items(S, nb, J);
  if (!items) return 0;
  const unsigned grid = (unsigned)((items * 32 + 255) / 256);
  if (sink) h1_noise_mid<true><<<grid, 256, 0, (cudaStream_t)stream>>>((const u64*)seeds, S, nb, J, (u32*)out);
  else h1_noise_mid<false><<<grid, 256, 0, (cudaStream_t)stream>>>((const u64*)seeds, S, nb, J, (u32*)out);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// The tile-width price (measured only): h1u.cu's h1s_noise and h1j_noise with the layout's counts and the seeds'
// row stride (u64 words) as arguments, so an r x c tile's noise (nb = r rb + c cb blocks, its rows_nb = r rb row tiles first,
// h1tile.Layout's order) runs at any c. Unit u = ri ncb + cj hashes its blocks t >= t0: t0 = 0 for a band leader
// (cj % band == 0), rows_nb otherwise (J10: a non-leader reads its leader's rows); band 1 is per-unit noise.
__global__ void __launch_bounds__(256) h1_noise_band(const u64* __restrict__ seeds, u32 stride, u32 S, u32 ncb, u32 nb,
                                                     u32 rows_nb, u32 band, u32 J, u32* __restrict__ out) {
  const u32 lane = threadIdx.x & 31;
  const u32 per = 32 * J, chunks = (nb + per - 1) / per;
  const u64 item = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (item >= (u64)S * chunks) return;
  const u32 u = (u32)(item / chunks), chunk = (u32)(item % chunks);
  const u32 t0 = (u % ncb) % band ? rows_nb : 0;
  if ((chunk + 1) * per <= t0) return;
  u64 seed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) seed[j] = __ldg(seeds + (u64)stride * u + j);
  NoiseMid nm;
  noise_mid(seed, nm);
#pragma unroll 1
  for (u32 j = 0; j < J; ++j) {
    const u32 t = chunk * per + 32 * j + lane;
    if (t >= nb) break;
    if (t < t0) continue;
    u64 s[8];
    noise_block(nm, t, s);
    uint4* o = reinterpret_cast<uint4*>(out + 16 * ((u64)u * nb + t));
#pragma unroll
    for (int q = 0; q < 4; ++q)
      o[q] = make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]);
  }
}

extern "C" int kn_band(const void* seeds, unsigned stride, unsigned S, unsigned ncb, unsigned nb, unsigned rows_nb,
                       unsigned band, unsigned J, void* out, void* stream) {
  if (!J || !band || !ncb || rows_nb > nb || stride < 8) return 2;
  const unsigned long long items = kn_mid_items(S, nb, J);
  if (!items) return 0;
  h1_noise_band<<<(unsigned)((items * 32 + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
      (const u64*)seeds, stride, S, ncb, nb, rows_nb, band, J, (u32*)out);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
