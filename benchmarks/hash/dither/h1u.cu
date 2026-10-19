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
// kh_unit_ws is the same receipts in two launches, for calls too small to fill the GPU one warp a unit:
//   h1s_noise  one warp an (unit, chunk) item, J blocks a lane: every block of seed_j's stream the unit reads (the rows'
//              blocks t = lane RB + c, the columns' 32 RB + lane CB + c), its 16 words to the caller's workspace
//   h1u<WS>    h1u reading those blocks instead of hashing them
// Both forms load the A and B words H1U_PD segments ahead while the current segment is multiplied and folded (WS always,
// the fused form with H1U_PF), in a ring of H1U_PD register sets; H1U_L2PF also prefetches into L2 the A and B lines 16
// segments ahead and (WS) the next chunk's workspace blocks. H1U_WARPS is the warps a CTA (one a unit).
//
//   kh_seeds(data, base, U, seeds, stream)   data: SEED_TAG, RECEIPT_TAG (3 words each), salt, root_a, root_b (8 each)
//   kh_unit(A, B, M, N, K, seeds, live, out, stream)   out: U x kh_words(K) u32 (live may be null: every unit live)
//   kh_unit_ws(A, B, M, N, K, seeds, live, out, ws, J, stream)   ws: kh_ws_bytes(M, N, K) bytes, the caller's
//   kh_words(K), kh_digits(K), kh_ws_bytes(M, N, K), kh_noise_blocks(K)
#include "h1sha.cuh"

#ifndef H1U_WARPS
#define H1U_WARPS 4
#endif
constexpr int WARPS = H1U_WARPS;
// CTAs an SM the register allocation must leave room for (h1u_b3.so: 3, at most 168 registers a thread)
#ifndef H1U_MINB
#define H1U_MINB 1
#endif
#ifndef H1U_PF
#define H1U_PF 0
#endif
#ifndef H1U_PD
#define H1U_PD 1
#endif
#ifndef H1U_L2PF
#define H1U_L2PF 0
#endif
static_assert(16 % H1U_PD == 0, "a chunk's 16 segments are whole rings of H1U_PD");

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

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

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
      z[c] = __funnelshift_l(uu, uu, KEYS.rot[t]) + y[c];
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

__global__ void __launch_bounds__(256) h1s_noise(const u64* __restrict__ seeds, u32 units, u32 ncb, u32 nb, u32 J,
                                                 const uint8_t* __restrict__ live, u32* __restrict__ ws) {
  const u32 lane = threadIdx.x & 31;
  const u32 per = 32 * J, chunks = (nb + per - 1) / per;
  const u64 item = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (item >= (u64)units * chunks) return;
  const u32 u = (u32)(item / chunks), chunk = (u32)(item % chunks);
  if (live && !live[u / ncb]) return;
  u64 seed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) seed[j] = __ldg(seeds + 16 * (u64)u + j);
  NoiseMid nm;
  noise_mid(seed, nm);
#pragma unroll 1
  for (u32 j = 0; j < J; ++j) {
    const u32 t = chunk * per + 32 * j + lane;
    if (t >= nb) break;
    u64 s[8];
    noise_block(nm, t, s);
    uint4* o = reinterpret_cast<uint4*>(ws + 16 * ((u64)u * nb + t));
#pragma unroll
    for (int q = 0; q < 4; ++q)
      o[q] = make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]);
  }
}

template <int MD, bool WS>
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1u(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M, u32 N,
                                                  u32 K, const u64* __restrict__ seeds, const uint8_t* __restrict__ live,
                                                  u32* __restrict__ out, const u32* __restrict__ ws) {
  constexpr bool PF = WS || H1U_PF;
  constexpr int PD = H1U_PD;
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
  if (!WS) {
    u64 seed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) seed[j] = __ldg(sd + j);
    NoiseMid nm;
    noise_mid(seed, nm);
    if (lane == 0) S.nm = nm;
  }
  __syncwarp();
  const u32* wsu = WS ? ws + 16ull * 32 * (rb + cb) * u : nullptr;

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
  // segment sg's A rows 16 mt + g + 8 h and B columns 8 nt + g, 8 bytes each at k = 32 sg + 8 q, in ring slot sg % PD
  uint2 pa[PD][4], pb[PD][4];
  const auto load_seg = [&](u32 sg, uint2(&a)[4], uint2(&b)[4]) {
    const u32 kofs = 32 * sg + 8 * q;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      a[r] = __ldg(reinterpret_cast<const uint2*>(Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + kofs));
      b[r] = __ldg(reinterpret_cast<const uint2*>(Bb + (u64)(8 * r + g) * K + kofs));
    }
  };
  if (PF) {
#pragma unroll
    for (int d = 0; d < PD; ++d) load_seg(d, pa[d], pb[d]);
  }
#pragma unroll 1
  for (u32 it = 0; it <= chunks; ++it) {
    const u32 blk = it == 0 ? chunks : it - 1;
#pragma unroll 1
    for (int jb = 0; jb < 2; ++jb) {
      const u64 t = jb == 0 ? (u64)lane * rb + blk : 32ull * rb + (u64)lane * cb + blk;
      u32* dst = jb == 0 ? S.rowbits[lane] : S.colbits[lane];
      if (WS) {
        const uint4* src = reinterpret_cast<const uint4*>(wsu + 16 * t);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const uint4 v = __ldg(src + w);
          dst[4 * w] = v.x;
          dst[4 * w + 1] = v.y;
          dst[4 * w + 2] = v.z;
          dst[4 * w + 3] = v.w;
        }
      } else {
        u64 s[8];
        noise_block(S.nm, t, s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          dst[2 * j] = (u32)(s[j] >> 32);
          dst[2 * j + 1] = (u32)s[j];
        }
      }
    }
    if (WS && H1U_L2PF && it < chunks) {
      prefetch_l2(wsu + 16 * ((u64)lane * rb + it));
      prefetch_l2(wsu + 16 * (32ull * rb + (u64)lane * cb + it));
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
    for (u32 j0 = 0; j0 < 16; j0 += PD) {
#pragma unroll
      for (int jj = 0; jj < PD; ++jj) {
        const u32 j = j0 + jj, sg = 16 * blk + j;
        uint2 ca[4], cw[4];
        if (H1U_L2PF && sg % 4 == 0 && sg + 16 < ks) {
          prefetch_l2(Ab + (u64)lane * K + 32 * (sg + 16));
          prefetch_l2(Bb + (u64)lane * K + 32 * (sg + 16));
        }
        if (PF) {
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            ca[r] = pa[jj][r];
            cw[r] = pb[jj][r];
          }
          if (sg + PD < ks) load_seg(sg + PD, pa[jj], pb[jj]);
        } else {
          const u32 kofs = 32 * sg + 8 * q;
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            ca[r] = __ldg(reinterpret_cast<const uint2*>(Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + kofs));
            cw[r] = __ldg(reinterpret_cast<const uint2*>(Bb + (u64)(8 * r + g) * K + kofs));
          }
        }
        u32 af[2][4], bf[4][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const u32 row = 16 * mt + g + 8 * h;
            const uint2 a8 = ca[2 * mt + h];
            const u32 nb8 = rbits[row * 68 + 4 * j + q];
            af[mt][h] = add01(a8.x, expand4(nb8 & 15u));
            af[mt][2 + h] = add01(a8.y, expand4(nb8 >> 4));
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const u32 col = 8 * nt + g;
          const uint2 b8 = cw[nt];
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
    h1u<2, false><<<grid, 32 * WARPS, 0, s>>>((const int8_t*)A, (const int8_t*)B, M, N, K, (const u64*)seeds,
                                               (const uint8_t*)live, (u32*)out, nullptr);
  else if (m == 3)
    h1u<3, false><<<grid, 32 * WARPS, 0, s>>>((const int8_t*)A, (const int8_t*)B, M, N, K, (const u64*)seeds,
                                               (const uint8_t*)live, (u32*)out, nullptr);
  else
    return 2;
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// seed_j's blocks a unit reads: rows' and columns' (K / 512 + 1) each, 32 apiece
extern "C" unsigned kh_noise_blocks(unsigned K) { return 64 * (K / 512 + 1); }
extern "C" unsigned long long kh_ws_bytes(unsigned M, unsigned N, unsigned K) {
  return 64ull * kh_noise_blocks(K) * ((M / 32) * (N / 32));
}

extern "C" int kh_unit_ws(const void* A, const void* B, unsigned M, unsigned N, unsigned K, const void* seeds,
                          const void* live, void* out, void* ws, unsigned J, void* stream) {
  if (K % 512 || !J || !ws) return 2;
  const int m = n_digits((int)(K / 32));
  if (m != 2 && m != 3) return 2;
  const unsigned units = (M / 32) * (N / 32);
  if (!units) return 0;
  cudaStream_t s = (cudaStream_t)stream;
  const unsigned nb = kh_noise_blocks(K);
  const unsigned long long items = (unsigned long long)units * ((nb + 32 * J - 1) / (32 * J));
  h1s_noise<<<(unsigned)((items * 32 + 255) / 256), 256, 0, s>>>((const u64*)seeds, units, N / 32, nb, J,
                                                                  (const uint8_t*)live, (u32*)ws);
  const unsigned grid = (units + WARPS - 1) / WARPS;
  if (m == 2)
    h1u<2, true><<<grid, 32 * WARPS, 0, s>>>((const int8_t*)A, (const int8_t*)B, M, N, K, (const u64*)seeds,
                                              (const uint8_t*)live, (u32*)out, (const u32*)ws);
  else
    h1u<3, true><<<grid, 32 * WARPS, 0, s>>>((const int8_t*)A, (const int8_t*)B, M, N, K, (const u64*)seeds,
                                              (const uint8_t*)live, (u32*)out, (const u32*)ws);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
