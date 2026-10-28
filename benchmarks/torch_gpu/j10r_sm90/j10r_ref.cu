// J10-R's proof block, `PouwH1Tile_v5` (h1tile.j10r_reference), slow but exact on sm_90a (the fallback): every value in
// the Definition's own order, integers only, no tensor cores, the noise and the fold keys hashed by each proof block's CTA itself.
//
//   jr_seeds   one thread a listed proof block: sigma_R = seeds.receipt_seed(salt, base + u, roots); one more thread seed_c =
//              seeds.call_seed(salt, base, roots) (SHA-512 over 224 bytes, two compressions each)
//   jr_units   one CTA a listed proof block, one thread a (lane, chain) of h1tile.TileShape(32, 32, K): lane l = tid / 8 holds the
//              tile's rows g + 8 s and columns 2 q + 8 (s >> 1) + (s & 1) (g = l / 4, q = l % 4), and chain c = tid % 8
//              absorbs the lane's entries e' = 4 t + (c % 4) for t = c / 4, + 2, + 4, + 6 (segment.row_slot / col_slot of
//              each), so the thread keeps those four entries' sums T_s (dp4a over each segment's 32 products) and its
//              chain's fx1 state (y_c, z_c), from its lane's key block (block l of sigma_R's stream). The proof block's noise is
//              seed_c's call-scale stream at h1tile.j10r_blocks: row i's block t of a chunk of 16 segments at
//              (32 ri + i) RB + t, column j's at M RB + (32 cj + j) CB + t, its tail (the rows' digit coins, the columns'
//              U bytes) at t = K / 512. After each set the receipt's words 2 s and 2 s + 1 are the XOR over the CTA's
//              256 chains of y and of z; after the KS segments the residue sums Y, the digits A_d (segment.sum_digits),
//              then O' = T_last + A_0 (-U)^T and D_d = A_d U^T entry by entry, absorbed as sets KS .. KS + m - 1.
//   jr_z       one thread an output word: z[i][j] = sum_k A[i][k] B[j][k] mod 2^32 (segment.seg_dot), v5's z
//
// A, B: s8 row-major [M, K] and [N, K]; units: u32 proof-block indices u = ri (N / 32) + cj; data: kh_seeds_j10r's words
// (j10r_units.seed_words: CALL_SEED_TAG, RECEIPT_TAG, salt, root_a, root_b, big-endian 64-bit) and NOISE_TAG's 3 words
// (33 in all); seed_c: 8 u64, sigma: n x 8 u64 (the digests' big-endian words); rec: n x jr_words(K) u32; z: n x 1024 u32.
// Requires M % 32 == N % 32 == 0, K % 512 == 0 (16 segments a noise block) and n_digits(K / 32) <= 4.
#include <cstdint>
#include <cuda_runtime.h>

#include "../sha512_rows/sha512x.cuh"

typedef unsigned long long u64;
typedef unsigned int u32;

constexpr int TILE = 32, THREADS = 256, CHUNK = 16, MAXM = 4, WARPS = THREADS / 32;

static int n_digits(long long ks) {
  long long p = 128;
  for (int m = 1; m < 16; ++m, p *= 128)
    if (-p <= -128 * ks && 127 * ks / (p / 128) <= 127) return m;
  return 0;
}

__device__ __forceinline__ u32 mix32(u32 x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}

__device__ __forceinline__ u32 rotl32(u32 x, int r) { return (x << r) | (x >> (32 - r)); }

__device__ __forceinline__ int sx8(u32 v) { return (int)(signed char)(v & 0xFF); }

// seeds.noise_block(seed, t) as 16 big-endian halves: SHA-512(NOISE_TAG || seed || u64be t), one compression
__device__ void noise_block(const u64* tag, const u64* seed, u64 t, u32* out) {
  u64 w[16] = {tag[0], tag[1], tag[2], seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7], t,
               0x8000000000000000ULL, 0, 0, 96 * 8};
  u64 s[8];
  sx::iv(s);
  sx::compress<0, 2>(s, w);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[2 * j] = (u32)(s[j] >> 32);
    out[2 * j + 1] = (u32)s[j];
  }
}

__global__ void __launch_bounds__(128) k_seeds(const u64* __restrict__ data, u64 base, const u32* __restrict__ units,
                                               u32 n, u64* __restrict__ seedc, u64* __restrict__ sigma) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) return;
  const int kind = i == n ? 0 : 1;  // 0: seed_c at the call's first position, 1: the proof block's sigma_R
  const u64 pos = i == n ? base : base + units[i];
  u64 msg[32];
  msg[0] = pos;
  for (int w = 0; w < 27; ++w) msg[1 + w] = (w < 3 ? data[3 * kind + w] : data[3 + w]) ^ pos;
  msg[28] = 0x8000000000000000ULL;
  msg[29] = msg[30] = 0;
  msg[31] = 224 * 8;
  u64 s[8];
  sx::iv(s);
  sx::compress<0, 2>(s, msg);
  sx::compress<0, 2>(s, msg + 16);
  u64* o = i == n ? seedc : sigma + 8 * (u64)i;
  for (int j = 0; j < 8; ++j) o[j] = s[j];
}

struct Shared {
  u32 nb[2 * TILE][16];          // the chunk's noise blocks: rows 0..31, then columns
  u32 x[TILE][CHUNK * 8];        // X = A + b over the chunk, 4 bytes a word (after the segments: Y, int [32][32])
  u32 w[TILE][CHUNK * 8];        // W' = B + c + U over the chunk (after the segments: A_d, bytes [MAXM][32][32])
  unsigned char u[TILE][TILE];   // U's int8 encodings, column by residue
  u32 coin[TILE][MAXM - 1];      // each row's digit coins: its tail block's halves 0 .. m - 2
  u32 keys[TILE][16];            // each lane's fold keys
  u32 red[WARPS][CHUNK][2];      // each warp's XOR of y and of z after each set of the chunk
};

__device__ __forceinline__ void absorb(u32& y, u32& z, const int (&T)[4], const u32 (&fk)[4], const u32 (&ka)[4],
                                       const u32 (&kk)[4], const int (&rot)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const u32 uu = y ^ ((u32)T[i] * fk[i]) ^ kk[i];
    y = uu * ka[i] + z;
    z = rotl32(uu, rot[i]) + y;
  }
}

__global__ void __launch_bounds__(THREADS) k_units(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M,
                                                   u32 N, u32 K, int m, const u64* __restrict__ data,
                                                   const u64* __restrict__ seedc, const u64* __restrict__ sigma,
                                                   const u32* __restrict__ units, u32* __restrict__ rec) {
  __shared__ Shared sh;
  const u32 tid = threadIdx.x, unit = blockIdx.x, u = units[unit];
  const u32 ncb = N / TILE, ri = u / ncb, cj = u % ncb, ks = K / 32, W = 2 * (ks + m);
  const u64 RB = (K + 32 * (m - 1) + 511) / 512, CB = (K + 256 + 511) / 512, tail = K / 512;
  const u64 rt0 = (u64)(TILE * ri) * RB, ct0 = (u64)M * RB + (u64)(TILE * cj) * CB;
  const u64* tag = data + 30;
  const u32 ln = tid >> 3, ch = tid & 7, p = ch >> 2, e = ch & 3, g = ln >> 2, qq = ln & 3;
  int row[4], col[4], rot[4];
  u32 fk[4], ka[4], kk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const u32 t = p + 2 * i, q = 4 * t + e, rs = 2 * (t / 4) + (e >> 1), cs = 2 * (t % 4) + (e & 1);
    row[i] = g + 8 * rs;
    col[i] = 2 * qq + 8 * (cs >> 1) + (cs & 1);
    fk[i] = mix32(0xA0000000u + 2 * (16 + 32 * q)) | 1u;
    ka[i] = mix32(0xA0000000u + 2 * t) | 1u;
    kk[i] = mix32(0xB0000000u + 2 * t + 1);
    rot[i] = (int)((t * 7 + 5) % 31) + 1;
  }
  u32 blk[16];
  if (tid < TILE) {
    noise_block(tag, seedc, rt0 + tid * RB + tail, blk);
    for (int d = 0; d < m - 1; ++d) sh.coin[tid][d] = blk[d];
  } else if (tid < 2 * TILE) {
    const u32 j = tid - TILE;
    noise_block(tag, seedc, ct0 + j * CB + tail, blk);
    for (int r = 0; r < TILE; ++r) sh.u[j][r] = (unsigned char)((-64 + (int)((blk[r >> 2] >> (8 * (r & 3))) & 0xFF) % 129) & 0xFF);
  } else if (tid < 3 * TILE) {
    const u32 l = tid - 2 * TILE;
    noise_block(tag, sigma + 8 * (u64)unit, l, blk);
    for (int w = 0; w < 16; ++w) sh.keys[l][w] = blk[w];
  }
  __syncthreads();
  u32 y = sh.keys[ln][ch], z = sh.keys[ln][8 + ch];
  int T[4] = {0, 0, 0, 0};
  int Y[4] = {0, 0, 0, 0};  // residue sums of row tid / 8 at residues 4 (tid % 8) ..
  const u32 yr = tid >> 3, yg = tid & 7;
  u32* out = rec + (u64)unit * W;
  for (u32 c = 0; c < tail; ++c) {
    if (tid < 2 * TILE)
      noise_block(tag, seedc, (tid < TILE ? rt0 + tid * RB : ct0 + (tid - TILE) * CB) + c, sh.nb[tid]);
    __syncthreads();
    for (u32 idx = tid; idx < TILE * CHUNK * 8; idx += THREADS) {
      const u32 r = idx / (CHUNK * 8), wd = idx % (CHUNK * 8), k0 = 512 * c + 4 * wd;
      const u32 a4 = *reinterpret_cast<const u32*>(A + (u64)(TILE * ri + r) * K + k0);
      const u32 b4 = *reinterpret_cast<const u32*>(B + (u64)(TILE * cj + r) * K + k0);
      const u32 db = sh.nb[r][wd >> 3] >> (4 * (wd & 7)), cb = sh.nb[TILE + r][wd >> 3] >> (4 * (wd & 7));
      u32 xv = 0, wv = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        xv |= (((a4 >> (8 * b)) + ((db >> b) & 1)) & 0xFF) << (8 * b);
        wv |= (((b4 >> (8 * b)) + ((cb >> b) & 1) + sh.u[r][4 * (wd & 7) + b]) & 0xFF) << (8 * b);
      }
      sh.x[r][wd] = xv;
      sh.w[r][wd] = wv;
    }
    __syncthreads();
    for (int s = 0; s < CHUNK; ++s) {
      const u32 xw = sh.x[yr][8 * s + yg];
#pragma unroll
      for (int b = 0; b < 4; ++b) Y[b] += sx8(xw >> (8 * b));
    }
    for (int s = 0; s < CHUNK; ++s) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int acc = T[i];
#pragma unroll
        for (int wd = 0; wd < 8; ++wd) acc = __dp4a((int)sh.x[row[i]][8 * s + wd], (int)sh.w[col[i]][8 * s + wd], acc);
        T[i] = acc;
      }
      absorb(y, z, T, fk, ka, kk, rot);
      const u32 xy = __reduce_xor_sync(0xFFFFFFFFu, y), xz = __reduce_xor_sync(0xFFFFFFFFu, z);
      if ((tid & 31) == 0) {
        sh.red[tid >> 5][s][0] = xy;
        sh.red[tid >> 5][s][1] = xz;
      }
    }
    __syncthreads();
    if (tid < 2 * CHUNK) {
      u32 v = 0;
      for (int wp = 0; wp < WARPS; ++wp) v ^= sh.red[wp][tid >> 1][tid & 1];
      out[2 * (CHUNK * c + (tid >> 1)) + (tid & 1)] = v;
    }
  }
  int (*Ys)[TILE] = reinterpret_cast<int (*)[TILE]>(&sh.x[0][0]);
  unsigned char (*Ad)[TILE][TILE] = reinterpret_cast<unsigned char (*)[TILE][TILE]>(&sh.w[0][0]);
#pragma unroll
  for (int b = 0; b < 4; ++b) Ys[yr][4 * yg + b] = Y[b];
  __syncthreads();
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int rho = 4 * yg + b;
    int R = Ys[yr][rho];
    for (int d = 0; d < m - 1; ++d) {
      const int qv = R >> 7;
      const int zd = (d == m - 2 && qv == 127) ? 0 : (int)((sh.coin[yr][d] >> rho) & 1);
      Ad[d][yr][rho] = (unsigned char)((R & 127) | (zd << 7));
      R = qv + zd;
    }
    Ad[m - 1][yr][rho] = (unsigned char)(R & 0xFF);
  }
  __syncthreads();
  for (int d = 0; d < m; ++d) {
    int v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int acc = d == 0 ? T[i] : 0;
      for (int rho = 0; rho < TILE; ++rho) {
        const u32 uv = sh.u[col[i]][rho];
        acc += sx8(d == 0 ? (0u - uv) : uv) * sx8(Ad[d][row[i]][rho]);
      }
      v[i] = acc;
    }
    absorb(y, z, v, fk, ka, kk, rot);
    const u32 xy = __reduce_xor_sync(0xFFFFFFFFu, y), xz = __reduce_xor_sync(0xFFFFFFFFu, z);
    if ((tid & 31) == 0) {
      sh.red[tid >> 5][d][0] = xy;
      sh.red[tid >> 5][d][1] = xz;
    }
  }
  __syncthreads();
  if (tid < 2u * m) {
    u32 v = 0;
    for (int wp = 0; wp < WARPS; ++wp) v ^= sh.red[wp][tid >> 1][tid & 1];
    out[2 * (ks + (tid >> 1)) + (tid & 1)] = v;
  }
}

__global__ void __launch_bounds__(TILE * TILE) k_z(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 N,
                                                   u32 K, const u32* __restrict__ units, u32* __restrict__ zo) {
  const u32 unit = blockIdx.x, u = units[unit], ncb = N / TILE, ri = u / ncb, cj = u % ncb;
  const u32 i = threadIdx.x / TILE, j = threadIdx.x % TILE;
  const u32* a = reinterpret_cast<const u32*>(A + (u64)(TILE * ri + i) * K);
  const u32* b = reinterpret_cast<const u32*>(B + (u64)(TILE * cj + j) * K);
  int acc = 0;
  for (u32 k = 0; k < K / 4; ++k) acc = __dp4a((int)__ldg(a + k), (int)__ldg(b + k), acc);
  zo[(u64)unit * TILE * TILE + threadIdx.x] = (u32)acc;
}

static int shape_ok(unsigned M, unsigned N, unsigned K) {
  if (M % TILE || N % TILE || !M || !N) return 1;
  if (K % 512 || !K) return 2;
  const int m = n_digits(K / 32);
  if (m < 1 || m > MAXM) return 3;
  return 0;
}

extern "C" int jr_digits(unsigned K) { return n_digits(K / 32); }
extern "C" unsigned jr_words(unsigned K) { return 2 * (K / 32 + n_digits(K / 32)); }
extern "C" unsigned jr_data_words() { return 33; }

extern "C" int jr_seeds(const void* data, unsigned long long base, const void* units, unsigned n, void* seedc,
                        void* sigma, void* stream) {
  k_seeds<<<(n + 1 + 127) / 128, 128, 0, (cudaStream_t)stream>>>((const u64*)data, base, (const u32*)units, n,
                                                                  (u64*)seedc, (u64*)sigma);
  return cudaGetLastError() == cudaSuccess ? 0 : 10;
}

extern "C" int jr_units(const void* A, const void* B, unsigned M, unsigned N, unsigned K, const void* data,
                        const void* seedc, const void* sigma, const void* units, unsigned n, void* rec, void* stream) {
  if (int bad = shape_ok(M, N, K)) return bad;
  if (!n) return 0;
  k_units<<<n, THREADS, 0, (cudaStream_t)stream>>>((const int8_t*)A, (const int8_t*)B, M, N, K, n_digits(K / 32),
                                                   (const u64*)data, (const u64*)seedc, (const u64*)sigma,
                                                   (const u32*)units, (u32*)rec);
  return cudaGetLastError() == cudaSuccess ? 0 : 10;
}

extern "C" int jr_z(const void* A, const void* B, unsigned M, unsigned N, unsigned K, const void* units, unsigned n,
                    void* z, void* stream) {
  if (int bad = shape_ok(M, N, K)) return bad;
  if (!n) return 0;
  k_z<<<n, TILE * TILE, 0, (cudaStream_t)stream>>>((const int8_t*)A, (const int8_t*)B, N, K, (const u32*)units, (u32*)z);
  return cudaGetLastError() == cudaSuccess ? 0 : 10;
}
