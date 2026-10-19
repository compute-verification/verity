// H1 + fx1's unit (pouw.circuit.h1tile, PouwH1Tile_v2) fused on sm_90a / sm_100a, bit for bit served_h1/receipts.py's
// `call_receipts` (receipts_ca.py here): a served call's A [M, K] and B [N, K] s8, its units the full 32 x 32 tiles in
// row-block-major order, unit u at position base + u.
//
//   h1_seeds   one thread a unit: seed_j and sigma_R (seeds.seed / receipt_seed, SHA-512 over 224 bytes, two
//              compressions each), the digests' big-endian words, 16 a unit
//   h1u        one warp a unit. The lane's fold keys are block `lane` of sigma_R's stream. seed_j's midstate goes to
//              shared memory once; then per 16-segment chunk c each lane hashes its row's block lane RB + c and its
//              column's block 32 RB + lane CB + c into shared memory (the rows' dither bits b and the columns' coin bits c
//              of 16 segments), after one first pass over block K / 512 (the rows' digit coins, the columns' U bytes).
//              A segment: X = A + b and W' = B + c + U (mod 2^8) built in registers as mma.sync m16n8k32 s8 fragments
//              (2 m-tiles x 4 n-tiles: the lane's 32 entries are exactly h1tile.Layout's at segment.row_slot / col_slot),
//              T += X W'^T, and Y += X P (P the identity under the fragments' K order, so Y is the residue sums); then
//              the set T_s folded: entry q times fx1_key(q), absorbed by duo chain q % 8 at key 2 (q / 8) + (q % 8 >= 4);
//              each set's XOR over the lane's chains to a ring, reduced across lanes once a chunk. After the KS sets: the
//              digits A_d from Y and the coins, O' = T_last + A_0 (-U)^T and D_d = A_d U^T on the tensor cores (A_d
//              packed from the accumulator layout in registers, no shuffle), folded as sets KS .. KS + m - 1.
// The K order inside a segment is a bijection both operands share (lane q's 8 bytes 8q..8q+7 are its A/B fragment's
// k = 4q..4q+3 and 16+4q..16+4q+3), so every product is the segment's; the tail's order is the accumulator columns'.
// A dead row block (live[ri] == 0, PouwH1Tile_v2's all-zero tile) writes the zero receipt and derives no noise.
// Requires K % 512 == 0 (KS % 16: each noise block is 16 segments' bits) and m in {2, 3} (K <= 16384).
//
//   kh_seeds(data, base, U, seeds, stream)   data: SEED_TAG, RECEIPT_TAG (3 words each), salt, root_a, root_b (8 each)
//   kh_unit(A, B, M, N, K, seeds, live, out, stream)   out: U x kh_words(K) u32 (live may be null: every unit live)
//   kh_words(K), kh_digits(K)
#include "h1sha.cuh"

constexpr int WARPS = 4;
// CTAs an SM the register allocation must leave room for (h1u_b3.so: 3, at most 168 registers a thread)
#ifndef H1U_MINB
#define H1U_MINB 1
#endif

constexpr u32 mix32c(u32 x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
// segment.KA / KK / ROT (duo's multiplier, xor key and rotation at tile position t) and fx1_key(q)
struct KeysT {
  u32 ka[8], kk[8], rot[8], fx1[32];
  constexpr KeysT() : ka(), kk(), rot(), fx1() {
    for (u32 t = 0; t < 8; ++t) {
      ka[t] = mix32c(0xA0000000u + 2 * t) | 1u;
      kk[t] = mix32c(0xB0000000u + 2 * t + 1);
      rot[t] = ((t * 7 + 5) % 31) + 1;
    }
    for (u32 q = 0; q < 32; ++q) fx1[q] = mix32c(0xA0000000u + 2 * (16 + 32 * q)) | 1u;
  }
};
__device__ constexpr KeysT KEYS{};

__host__ __device__ constexpr int n_digits(int ks) {
  // segment.n_digits: the least m with -128^m <= -128 KS and 127 KS // 128^(m - 1) <= 127
  long long p = 128;
  for (int m = 1; m < 8; ++m, p *= 128)
    if (-p <= -128LL * ks && 127LL * ks / (p / 128) <= 127) return m;
  return 0;
}

struct alignas(16) WarpSmem {
  NoiseMid nm;          // seed_j's stream
  u32 rowbits[32][17];  // row i's block: 16 halves (one segment's 32 dither bits each), padded against bank conflicts
  u32 colbits[32][17];
  u32 coin[32][2];      // row i's digit coins: half d's bit rho is z_d(i, rho)
  u32 utab[32][9];      // column j's U + 128 bytes, rho = 0..31
  u32 ring[32][33];     // [2 j + (0 y | 1 z)][lane], a chunk's 16 sets
};

__device__ __forceinline__ u32 expand4(u32 n) { return (n * 0x00204081u) & 0x01010101u; }
// bytewise a + bb mod 2^8 for bb's bytes in {0, 1}
__device__ __forceinline__ u32 add01(u32 a, u32 bb) { return ((a & 0x7F7F7F7Fu) + bb) ^ (a & 0x80808080u); }
// bytewise (b + 128) + v mod 2^8, v's bytes in [64, 193] (U + 128 + c): B + c + U
__device__ __forceinline__ u32 addw(u32 b, u32 v) {
  return ((b & 0x7F7F7F7Fu) + (v & 0x7F7F7F7Fu)) ^ (~(b ^ v) & 0x80808080u);
}

__device__ __forceinline__ void mma_s8(int (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// one set: the lane's entry q = 4 t + e is acc[t][e]; chain c absorbs entry 8 i + c at step i, key t = 2 i + (c >= 4)
__device__ __forceinline__ void fold(const int (&acc)[8][4], u32 (&y)[8], u32 (&z)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int t = 2 * i + (c >> 2), e = c & 3;
      const u32 p = (u32)acc[t][e] * KEYS.fx1[8 * i + c];
      const u32 uu = y[c] ^ p ^ KEYS.kk[t];
      y[c] = uu * KEYS.ka[t] + z[c];
#if H1_PIPE & 64
      // rotl(uu, r) + y as two IMADs: hi(uu 2^r) + (uu 2^r + y), the rotation's halves having disjoint bits
      const u32 m = PW2[KEYS.rot[t]];
      asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(z[c]) : "r"(uu), "r"(m), "r"(uu * m + y[c]));
#else
      z[c] = __funnelshift_l(uu, uu, KEYS.rot[t]) + y[c];
#endif
    }
  }
}

__device__ __forceinline__ void ring_put(WarpSmem& S, int slot, u32 lane, const u32 (&y)[8], const u32 (&z)[8]) {
  S.ring[2 * slot][lane] = y[0] ^ y[1] ^ y[2] ^ y[3] ^ y[4] ^ y[5] ^ y[6] ^ y[7];
  S.ring[2 * slot + 1][lane] = z[0] ^ z[1] ^ z[2] ^ z[3] ^ z[4] ^ z[5] ^ z[6] ^ z[7];
}

__device__ __forceinline__ u32 ring_xor(const WarpSmem& S, u32 row) {
  u32 x = 0;
#pragma unroll
  for (int l = 0; l < 32; ++l) x ^= S.ring[row][l];
  return x;
}

__global__ void __launch_bounds__(128) h1_seeds(const u64* __restrict__ data, u64 base, u32 U, u64* __restrict__ out) {
  const u32 u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= U) return;
  const u64 pos = base + u;
#pragma unroll 1
  for (int tag = 0; tag < 2; ++tag) {
    u64 msg[32];
    msg[0] = pos;
#pragma unroll
    for (int w = 0; w < 27; ++w) msg[1 + w] = (w < 3 ? data[3 * tag + w] : data[3 + w]) ^ pos;
    msg[28] = 0x8000000000000000ULL;
    msg[29] = 0;
    msg[30] = 0;
    msg[31] = 224 * 8;
    u64 s[8];
    sha512_words(msg, 2, s);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[16 * (u64)u + 8 * tag + j] = s[j];
  }
}

template <int MD>
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1u(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M, u32 N,
                                                  u32 K, const u64* __restrict__ seeds, const uint8_t* __restrict__ live,
                                                  u32* __restrict__ out) {
  __shared__ WarpSmem smem[WARPS];
  const u32 lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const u32 g = lane >> 2, q = lane & 3;
  const u32 ncb = N / 32, units = (M / 32) * ncb;
  const u32 u = blockIdx.x * WARPS + wp;
  if (u >= units) return;
  const u32 ri = u / ncb, cj = u % ncb;
  const u32 ks = K / 32, chunks = ks / 16, rb = chunks + 1, cb = chunks + 1;
  const u32 nwords = 2 * (ks + MD);
  u32* o = out + (u64)u * nwords;
  if (live && !live[ri]) {
    for (u32 w = lane; w < nwords; w += 32) o[w] = 0;
    return;
  }
  WarpSmem& S = smem[wp];
  const u64* sd = seeds + 16 * (u64)u;

  u32 y[8], z[8];
  {
    u64 msg[16] = {TAG0, TAG1, TAG2};
#pragma unroll
    for (int j = 0; j < 8; ++j) msg[3 + j] = __ldg(sd + 8 + j);
    msg[11] = lane;
    msg[12] = 0x8000000000000000ULL;
    msg[13] = 0;
    msg[14] = 0;
    msg[15] = 96 * 8;
    u64 s[8];
    sha512_words(msg, 1, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      y[2 * j] = (u32)(s[j] >> 32);
      y[2 * j + 1] = (u32)s[j];
      z[2 * j] = (u32)(s[4 + j] >> 32);
      z[2 * j + 1] = (u32)s[4 + j];
    }
  }
  {
    u64 seed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) seed[j] = __ldg(sd + j);
    NoiseMid nm;
    noise_mid(seed, nm);
    if (lane == 0) S.nm = nm;
  }
  __syncwarp();

  // P: column n of n-tile nt is rho = 8 nt + n; fragment k = 4 q' + i is segment byte 8 q' + i, k = 16 + 4 q' + i is
  // 8 q' + 4 + i
  u32 idr[4][2];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    idr[nt][0] = (q == (u32)nt && g < 4) ? 1u << (8 * g) : 0u;
    idr[nt][1] = (q == (u32)nt && g >= 4) ? 1u << (8 * (g - 4)) : 0u;
  }
  int T[8][4], Y[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) T[t][e] = Y[t][e] = 0;
  u32 ulo[4], uhi[4];

  const int8_t* Ab = A + (u64)(32 * ri) * K;
  const int8_t* Bb = B + (u64)(32 * cj) * K;
#pragma unroll 1
  for (u32 it = 0; it <= chunks; ++it) {
    const u32 blk = it == 0 ? chunks : it - 1;
#pragma unroll 1
    for (int jb = 0; jb < 2; ++jb) {
      const u64 t = jb == 0 ? (u64)lane * rb + blk : 32ull * rb + (u64)lane * cb + blk;
      u32* dst = jb == 0 ? S.rowbits[lane] : S.colbits[lane];
      u64 s[8];
      noise_block(S.nm, t, s);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[2 * j] = (u32)(s[j] >> 32);
        dst[2 * j + 1] = (u32)s[j];
      }
    }
    __syncwarp();
    if (it == 0) {
#pragma unroll
      for (int d = 0; d < MD - 1; ++d) S.coin[lane][d] = S.rowbits[lane][d];
#pragma unroll
      for (int h = 0; h < 8; ++h) {
        const u32 hv = S.colbits[lane][h];
        u32 w = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const u32 x = (hv >> (8 * b)) & 0xFFu;
          w |= ((x >= 129 ? x - 129 : x) + 64) << (8 * b);
        }
        S.utab[lane][h] = w;
      }
      __syncwarp();
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        ulo[nt] = S.utab[8 * nt + g][2 * q];
        uhi[nt] = S.utab[8 * nt + g][2 * q + 1];
      }
      continue;
    }
    const uint8_t* rbits = reinterpret_cast<const uint8_t*>(&S.rowbits[0][0]);
    const uint8_t* cbits = reinterpret_cast<const uint8_t*>(&S.colbits[0][0]);
#pragma unroll 1
    for (u32 j = 0; j < 16; ++j) {
      const u32 kofs = 32 * (16 * blk + j) + 8 * q;
      u32 af[2][4], bf[4][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const u32 row = 16 * mt + g + 8 * h;
          const uint2 a8 = __ldg(reinterpret_cast<const uint2*>(Ab + (u64)row * K + kofs));
          const u32 nb8 = rbits[row * 68 + 4 * j + q];
          af[mt][h] = add01(a8.x, expand4(nb8 & 15u));
          af[mt][2 + h] = add01(a8.y, expand4(nb8 >> 4));
        }
      }
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const u32 col = 8 * nt + g;
        const uint2 b8 = __ldg(reinterpret_cast<const uint2*>(Bb + (u64)col * K + kofs));
        const u32 cb8 = cbits[col * 68 + 4 * j + q];
        bf[nt][0] = addw(b8.x, ulo[nt] + expand4(cb8 & 15u));
        bf[nt][1] = addw(b8.y, uhi[nt] + expand4(cb8 >> 4));
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          mma_s8(T[4 * mt + nt], af[mt], bf[nt]);
          mma_s8(Y[4 * mt + nt], af[mt], idr[nt]);
        }
      fold(T, y, z);
      ring_put(S, j, lane, y, z);
    }
    __syncwarp();
    o[32 * blk + lane] = ring_xor(S, lane);
    __syncwarp();
  }

  // the tail: A_d packed as A fragments in the accumulators' column order (fragment k 4 q' .. of rows g / g + 8 is
  // columns 2 q', 2 q' + 1, 8 + 2 q', 9 + 2 q' of n-tiles 0, 1; k 16 + 4 q' .. those of n-tiles 2, 3)
  u32 ad[MD][2][4];
#pragma unroll
  for (int d = 0; d < MD; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int r = 0; r < 4; ++r) ad[d][mt][r] = 0;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    const int mt = t >> 2, nt = t & 3;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const u32 row = 16 * mt + g + 8 * (e >> 1), rho = 8 * nt + 2 * q + (e & 1);
      int yv = Y[t][e];
      const int reg = (e >> 1) + 2 * (nt >> 1), sh8 = 8 * (2 * (nt & 1) + (e & 1));
#pragma unroll
      for (int d = 0; d < MD - 1; ++d) {
        const int qv = yv >> 7;
        u32 zd = (S.coin[row][d] >> rho) & 1u;
        if (d == MD - 2 && qv == 127) zd = 0;
        ad[d][mt][reg] |= (((u32)yv & 127u) | (zd << 7)) << sh8;
        yv = qv + (int)zd;
      }
      ad[MD - 1][mt][reg] |= ((u32)yv & 0xFFu) << sh8;
    }
  }
  u32 uf[4][2], nf[4][2];
  {
    const uint8_t* ub = reinterpret_cast<const uint8_t*>(&S.utab[0][0]);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const uint8_t* r = ub + (8 * nt + g) * 36;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const u32 c0 = 16 * hf + 2 * q;
        const u32 by[4] = {r[c0], r[c0 + 1], r[c0 + 8], r[c0 + 9]};
        u32 pu = 0, pn = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          pu |= ((by[b] - 128u) & 0xFFu) << (8 * b);
          pn |= ((128u - by[b]) & 0xFFu) << (8 * b);
        }
        uf[nt][hf] = pu;
        nf[nt][hf] = pn;
      }
    }
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mma_s8(T[4 * mt + nt], ad[0][mt], nf[nt]);
  fold(T, y, z);
  ring_put(S, 0, lane, y, z);
#pragma unroll
  for (int d = 1; d < MD; ++d) {
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) Y[t][e] = 0;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) mma_s8(Y[4 * mt + nt], ad[d][mt], uf[nt]);
    fold(Y, y, z);
    ring_put(S, d, lane, y, z);
  }
  __syncwarp();
  if (lane < 2 * MD) o[2 * ks + lane] = ring_xor(S, lane);
}

extern "C" int kh_digits(unsigned K) { return n_digits((int)(K / 32)); }
extern "C" unsigned kh_words(unsigned K) { return 2 * (K / 32 + n_digits((int)(K / 32))); }

extern "C" int kh_seeds(const void* data, unsigned long long base, unsigned U, void* seeds, void* stream) {
  if (!U) return 0;
  h1_seeds<<<(U + 127) / 128, 128, 0, (cudaStream_t)stream>>>((const u64*)data, base, U, (u64*)seeds);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int kh_unit(const void* A, const void* B, unsigned M, unsigned N, unsigned K, const void* seeds, const void* live,
                       void* out, void* stream) {
  if (K % 512) return 2;
  const int m = n_digits((int)(K / 32));
  const unsigned units = (M / 32) * (N / 32);
  if (!units) return 0;
  const unsigned grid = (units + WARPS - 1) / WARPS;
  cudaStream_t s = (cudaStream_t)stream;
  if (m == 2)
    h1u<2><<<grid, 32 * WARPS, 0, s>>>((const int8_t*)A, (const int8_t*)B, M, N, K, (const u64*)seeds,
                                        (const uint8_t*)live, (u32*)out);
  else if (m == 3)
    h1u<3><<<grid, 32 * WARPS, 0, s>>>((const int8_t*)A, (const int8_t*)B, M, N, K, (const u64*)seeds,
                                        (const uint8_t*)live, (u32*)out);
  else
    return 2;
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
