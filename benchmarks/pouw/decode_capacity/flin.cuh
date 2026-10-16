// ---------------------------------------------------------------- flin: the decode linears as a warp-specialized GEMM with filler warps
// (FILL, round 2; included by dcap.cu after the dfill section, whose dseed/dword/dcval/drec/dstep/dfill_ref it reuses)
//
// D (M x N, row-major s32) = X (M x K s8) W^T (W N x K s8), computed as D^T = W X^T: the weights are the MMA's A operand
// (16 output features a fragment), the tokens its B operand (8 a fragment). Weights and activations are repacked once into
// tiles that are contiguous in memory and already in mma.sync m16n8k32's fragment order:
//   W tile (nt, kb): BN features x BK of k, as [i < BN/16][c < BK/32][lane][a0 a1 a2 a3] (512 B a fragment);
//   X tile kb: M tokens x BK of k, as [j < M/8][p < BK/64][lane][b0(2p) b1(2p) b0(2p+1) b1(2p+1)];
// so one bulk async copy (cp.async.bulk, the TMA engine) moves a stage's W or X tile, and a lane's fragment is one 16- or
// 8-byte shared load with no bank conflict. One CTA a SM, three roles:
//   producer warp: lane 0 streams the CTA's units (W tile + X tile) into an ST-stage ring guarded by full/empty mbarriers;
//   consumer warps (CWN x CWM): the served MMAs, a warp BN/CWN features x M/CWM tokens, partial sums to D by red.add.s32;
//   filler warps (FW): round 1's chain (dfill_k's tile, from the coins and the warp's slot) until every consumer is done,
//     then the tile's receipt and its step count; with thr set, a filler warp steps only while every consumer waits on a
//     stage (it sleeps otherwise).
// Work is split stream-K: the (N/BN) x (K/BK) units in tile-major order, a contiguous run of them a CTA, so every SM streams
// the same bytes. Partial tiles meet in D by integer red.add, exact in any order; D must start at zero, so each CTA also
// zeroes its share of the next linear's output (Z, zn ints, a multiple of 4).
constexpr int FLIN_BN = 128;

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, unsigned n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(sa(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint64_t* b, unsigned bytes) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.expect_tx.shared.b64 s, [%0], %1; }" ::"r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.shared.b64 s, [%0]; }" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ bool mb_try(uint64_t* b, unsigned par) {
  unsigned ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok)
               : "r"(sa(b)), "r"(par)
               : "memory");
  return ok;
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void mma_acc(int (&d)[4], const uint4& a, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// One filler warp: dfill_k's tile `tile` (NC chains), stepping 8 at a time while *done < ndone (and, with thr, only while
// *wait == ndone), then its receipt to rec and its step count to cnt. Every lane reads the flags through lane 0.
template <int NC>
__device__ __forceinline__ void filler_warp(uint64_t c0, uint64_t c1, unsigned long long tile, const int* done, const int* wait,
                                            int ndone, int thr, unsigned long long* rec, int* cnt) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const uint64_t seed = dseed(c0, c1, tile);
  uint32_t b[2][2], c[4], st[NC][4];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    b[j][0] = dword(seed, 1 + j, g, t);
    b[j][1] = dword(seed, 1 + j, g, 4 + t);
  }
  c[0] = (uint32_t)dcval(seed, g, 2 * t);
  c[1] = (uint32_t)dcval(seed, g, 2 * t + 1);
  c[2] = (uint32_t)dcval(seed, g + 8, 2 * t);
  c[3] = (uint32_t)dcval(seed, g + 8, 2 * t + 1);
#pragma unroll
  for (int ch = 0; ch < NC; ch++) {
    st[ch][0] = dword(seed, 16 + ch, g, t);
    st[ch][1] = dword(seed, 16 + ch, g + 8, t);
    st[ch][2] = dword(seed, 16 + ch, g, 4 + t);
    st[ch][3] = dword(seed, 16 + ch, g + 8, 4 + t);
  }
  const volatile int* vd = done;
  const volatile int* vw = wait;
  int steps = 0;
  for (;;) {
    const int d = __shfl_sync(0xffffffffu, lane == 0 ? *vd : 0, 0);
    if (d >= ndone) break;
    if (thr) {
      const int w = __shfl_sync(0xffffffffu, lane == 0 ? *vw : 0, 0);
      if (w < ndone) {
        __nanosleep(64);
        continue;
      }
    }
#pragma unroll
    for (int u = 0; u < 8; u++)
#pragma unroll
      for (int ch = 0; ch < NC; ch++) dstep(st[ch], b[u & 1], c);
    steps += 8;
  }
  uint64_t h = 0;
#pragma unroll
  for (int ch = 0; ch < NC; ch++)
    h ^= drec(seed, ch, g, t, st[ch][0]) ^ drec(seed, ch, g + 8, t, st[ch][1]) ^ drec(seed, ch, g, 4 + t, st[ch][2]) ^
         drec(seed, ch, g + 8, 4 + t, st[ch][3]);
#pragma unroll
  for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  if (lane == 0) {
    *rec = h;
    *cnt = steps;
  }
}

struct FlinArgs {
  const int8_t* W;  // tiled, [N/BN][K/BK][BN·BK]
  const int8_t* X;  // tiled, [K/BK][M·BK]
  int* D;           // M x N, zero on entry
  int* Z;           // the next linear's output, zeroed here
  long long zn;
  int N, K;
  uint64_t c0, c1;
  unsigned long long tile0;  // filler warp `slot` (blockIdx.x · FW + fw) runs dfill tile tile0 + slot
  unsigned long long* rec;   // [gridDim.x · FW]
  int* cnt;
  int thr;
};

template <int M, int BK, int ST, int CWN, int CWM, int FW, int NC>
__global__ void __launch_bounds__(32 * (CWN * CWM + 1 + FW), 1) flin_k(const FlinArgs a) {
  constexpr int BN = FLIN_BN, CW = CWN * CWM, FM = BN / CWN / 16, TN = M / CWM / 8, KC = BK / 32;
  constexpr int WB = BN * BK, XB = M * BK, SB = WB + XB;
  static_assert(FM >= 1 && TN >= 1 && KC % 2 == 0, "flin tile shape");
  extern __shared__ __align__(128) uint8_t sm[];
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + ST * SB);
  uint64_t* empty = full + ST;
  int* flag = reinterpret_cast<int*>(empty + ST);  // [0] consumer warps done, [1] consumer warps waiting on a stage
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int KB = a.K / BK, U = a.N / BN * KB;
  const int u0 = (int)((long long)blockIdx.x * U / gridDim.x), u1 = (int)((long long)(blockIdx.x + 1) * U / gridDim.x);
  if (threadIdx.x == 0) {
    for (int s = 0; s < ST; s++) {
      mb_init(&full[s], 1);
      mb_init(&empty[s], CW);
    }
    flag[0] = flag[1] = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == CW) {  // producer
    if (lane == 0) {
      uint64_t pw, px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
      for (int u = u0, i = 0; u < u1; u++, i++) {
        const int s = i % ST;
        if (i >= ST)
          while (!mb_try(&empty[s], ((i / ST) - 1) & 1)) {
          }
        mb_expect(&full[s], SB);
        const int nt = u / KB, kb = u % KB;
        bulk_g2s(sm + s * SB, a.W + ((size_t)nt * KB + kb) * WB, WB, &full[s], pw);
        bulk_g2s(sm + s * SB + WB, a.X + (size_t)kb * XB, XB, &full[s], px);
      }
    }
    return;
  }
  if (warp > CW) {
    if constexpr (FW > 0) {
      const int slot = blockIdx.x * FW + (warp - CW - 1);
      filler_warp<NC>(a.c0, a.c1, a.tile0 + slot, &flag[0], &flag[1], CW, a.thr, a.rec + slot, a.cnt + slot);
    }
    return;
  }
  // consumer
  const int wn = warp % CWN, wm = warp / CWN, g = lane >> 2, t = lane & 3;
  {
    const long long n4 = a.zn / 4, z0 = (long long)blockIdx.x * n4 / gridDim.x, z1 = (long long)(blockIdx.x + 1) * n4 / gridDim.x;
    int4* z = reinterpret_cast<int4*>(a.Z);
    for (long long e = z0 + threadIdx.x; e < z1; e += 32 * CW) z[e] = make_int4(0, 0, 0, 0);
  }
  int acc[FM][TN][4];
  auto clear = [&] {
#pragma unroll
    for (int f = 0; f < FM; f++)
#pragma unroll
      for (int j = 0; j < TN; j++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[f][j][r] = 0;
  };
  auto flush = [&](int tile) {
#pragma unroll
    for (int f = 0; f < FM; f++)
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int n = tile * BN + (wn * FM + f) * 16 + g, m = (wm * TN + j) * 8 + 2 * t;
        int* d = a.D + (size_t)m * a.N + n;
        atomicAdd(d, acc[f][j][0]);
        atomicAdd(d + a.N, acc[f][j][1]);
        atomicAdd(d + 8, acc[f][j][2]);
        atomicAdd(d + a.N + 8, acc[f][j][3]);
      }
  };
  clear();
  int nt = u0 / KB;
  for (int u = u0, i = 0; u < u1; u++, i++) {
    if (u / KB != nt) {
      flush(nt);
      clear();
      nt = u / KB;
    }
    const int s = i % ST;
    const unsigned par = (i / ST) & 1;
    if (!mb_try(&full[s], par)) {
      if (lane == 0) atomicAdd(&flag[1], 1);
      while (!mb_try(&full[s], par)) {
      }
      if (lane == 0) atomicSub(&flag[1], 1);
    }
    __syncwarp();  // lanes leave the wait at different polls; mma.sync.aligned needs the warp converged
    const uint8_t* ws = sm + s * SB + wn * FM * KC * 512 + lane * 16;
    const uint8_t* xs = sm + s * SB + WB + wm * TN * (KC / 2) * 512 + lane * 16;
    // one k pair's fragments live at a time: the whole stage's would not fit beside the accumulators and the filler warps
#pragma unroll 1
    for (int p = 0; p < KC / 2; p++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint4 aw[FM];
#pragma unroll
        for (int f = 0; f < FM; f++) aw[f] = *reinterpret_cast<const uint4*>(ws + (f * KC + 2 * p + h) * 512);
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const uint2 bx = *reinterpret_cast<const uint2*>(xs + (j * (KC / 2) + p) * 512 + h * 8);
#pragma unroll
          for (int f = 0; f < FM; f++) mma_acc(acc[f][j], aw[f], bx.x, bx.y);
        }
      }
    __syncwarp();
    if (lane == 0) mb_arrive(&empty[s]);
  }
  flush(nt);
  __syncwarp();
  if (lane == 0) atomicAdd(&flag[0], 1);
}

// Wt[nt][kb][i][c][lane][4 words] from W (N x K row-major): word r of lane (g, t) is W[row][col..col+3] with
// row = nt·BN + 16i + g + 8·(r & 1), col = kb·BK + 32c + 16·(r >> 1) + 4t (a0 a1 a2 a3 of m16n8k32's A fragment)
__global__ void repack_w(const int8_t* __restrict__ W, int8_t* __restrict__ Wt, int N, int K, int BK) {
  const size_t words = (size_t)N * K / 4;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < words; e += (size_t)gridDim.x * blockDim.x) {
    const int r = e & 3, lane = (e >> 2) & 31, g = lane >> 2, t = lane & 3;
    size_t rest = e >> 7;
    const int KC = BK / 32, IB = FLIN_BN / 16, KB = K / BK;
    const int c = rest % KC; rest /= KC;
    const int i = rest % IB; rest /= IB;
    const int kb = rest % KB;
    const int nt = (int)(rest / KB);
    const int row = nt * FLIN_BN + 16 * i + g + 8 * (r & 1), col = kb * BK + 32 * c + 16 * (r >> 1) + 4 * t;
    reinterpret_cast<int*>(Wt)[e] = *reinterpret_cast<const int*>(W + (size_t)row * K + col);
  }
}
// Xt[kb][j][p][lane][4 words] from X (M x K row-major): words b0(2p) b1(2p) b0(2p+1) b1(2p+1) of lane (g, t), where
// b0(c) = X[8j + g][kb·BK + 32c + 4t ..] and b1(c) the same 16 further along k (m16n8k32's B fragment, col layout)
__global__ void repack_x(const int8_t* __restrict__ X, int8_t* __restrict__ Xt, int M, int K, int BK) {
  const size_t words = (size_t)M * K / 4;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < words; e += (size_t)gridDim.x * blockDim.x) {
    const int r = e & 3, lane = (e >> 2) & 31, g = lane >> 2, t = lane & 3;
    size_t rest = e >> 7;
    const int KP = BK / 64, JB = M / 8;
    const int p = rest % KP; rest /= KP;
    const int j = rest % JB;
    const int kb = (int)(rest / JB);
    const int c = 2 * p + (r >> 1);
    const int row = 8 * j + g, col = kb * BK + 32 * c + 16 * (r & 1) + 4 * t;
    reinterpret_cast<int*>(Xt)[e] = *reinterpret_cast<const int*>(X + (size_t)row * K + col);
  }
}

// ---------------------------------------------------------------- host side
struct FlinCfg {
  int BK, ST, CW;
};
static bool flin_m_ok(int M) { return M == 32 || M == 64 || M == 128 || M == 256; }
static FlinCfg flin_cfg(int M) {
  if (M == 32 || M == 64) return {128, 4, 4};
  if (M == 128) return {128, 3, 8};
  return {64, 4, 8};
}
static int flin_smem(int M) {
  const FlinCfg c = flin_cfg(M);
  return c.ST * (FLIN_BN + M) * c.BK + 2 * c.ST * 8 + 16;
}
template <int M, int BK, int ST, int CWN, int CWM, int FW, int NC>
static void flin_go(const FlinArgs& a, int grid, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(flin_k<M, BK, ST, CWN, CWM, FW, NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, flin_smem(M)));
    attr = true;
  }
  flin_k<M, BK, ST, CWN, CWM, FW, NC><<<grid, 32 * (CWN * CWM + 1 + FW), flin_smem(M), st>>>(a);
}
// the (FW, NC) pairs built for each M: FW = 0 is the GEMM alone
template <int M, int BK, int ST, int CWN, int CWM>
static bool flin_fw(const FlinArgs& a, int grid, int FW, int NC, cudaStream_t st) {
  const int k = FW == 0 ? 0 : FW * 10 + NC;
  if (k == 0) { flin_go<M, BK, ST, CWN, CWM, 0, 2>(a, grid, st); return true; }
  if constexpr (M <= 64) {
    switch (k) {
      case 42: flin_go<M, BK, ST, CWN, CWM, 4, 2>(a, grid, st); return true;
      case 44: flin_go<M, BK, ST, CWN, CWM, 4, 4>(a, grid, st); return true;
      case 82: flin_go<M, BK, ST, CWN, CWM, 8, 2>(a, grid, st); return true;
      case 84: flin_go<M, BK, ST, CWN, CWM, 8, 4>(a, grid, st); return true;
    }
  } else if constexpr (M == 128) {
    switch (k) {
      case 42: flin_go<M, BK, ST, CWN, CWM, 4, 2>(a, grid, st); return true;
      case 44: flin_go<M, BK, ST, CWN, CWM, 4, 4>(a, grid, st); return true;
    }
  } else {
    switch (k) {
      case 22: flin_go<M, BK, ST, CWN, CWM, 2, 2>(a, grid, st); return true;
      case 24: flin_go<M, BK, ST, CWN, CWM, 2, 4>(a, grid, st); return true;
    }
  }
  return false;
}
static bool flin_launch(int M, const FlinArgs& a, int grid, int FW, int NC, cudaStream_t st) {
  switch (M) {
    case 32: return flin_fw<32, 128, 4, 4, 1>(a, grid, FW, NC, st);
    case 64: return flin_fw<64, 128, 4, 4, 1>(a, grid, FW, NC, st);
    case 128: return flin_fw<128, 128, 3, 4, 2>(a, grid, FW, NC, st);
    case 256: return flin_fw<256, 64, 4, 4, 2>(a, grid, FW, NC, st);
  }
  return false;
}
static const char* flin_built(int M) {
  return M <= 64 ? "0, 4:2, 4:4, 8:2, 8:4" : M == 128 ? "0, 4:2, 4:4" : "0, 2:2, 2:4";
}

// The step's linears through flin: every layer's weights tiled for M, X tiled for K = HID and K = FF, two output buffers
// (linear i writes Db[i & 1] and zeroes Db[(i + 1) & 1] for linear i + 1), and the filler's receipts a launch.
struct Flin {
  int M = 0, BK = 0, FW = 0, NC = 2, thr = 0, grid = 0;
  int8_t* Wt[LAYERS][4] = {};
  int8_t *XtH = nullptr, *XtF = nullptr;
  int* Db[2] = {};
  unsigned long long* rec = nullptr;
  int* cnt = nullptr;
  int slots = 0;  // a launch's filler slots, grid · FW
  static constexpr int LAUNCHES = LAYERS * 4;
};
static void flin_tile_x(const Dev& d, Flin& f) {
  const size_t h = (size_t)f.M * HID, ff = (size_t)f.M * FF;
  if (!f.XtH) CK(cudaMalloc(&f.XtH, h));
  if (!f.XtF) CK(cudaMalloc(&f.XtF, ff));
  repack_x<<<1024, 256, 0, d.st>>>(d.X, f.XtH, f.M, HID, f.BK);
  repack_x<<<1024, 256, 0, d.st>>>(d.X, f.XtF, f.M, FF, f.BK);
}
static void flin_prepare(Dev& d, Flin& f, int M, bool all_layers) {
  f.M = M;
  f.BK = flin_cfg(M).BK;
  f.grid = d.sms;
  for (int l = 0; l < (all_layers ? LAYERS : 1); l++)
    for (int j = 0; j < 4; j++) {
      const size_t n = (size_t)LN[j] * LK[j];
      if (!f.Wt[l][j]) CK(cudaMalloc(&f.Wt[l][j], n));
      repack_w<<<4096, 256, 0, d.st>>>(d.W[l][j], f.Wt[l][j], LN[j], LK[j], f.BK);
    }
  flin_tile_x(d, f);
  f.Db[0] = reinterpret_cast<int*>(reinterpret_cast<char*>(d.D) + ((size_t)256 << 20));
  f.Db[1] = reinterpret_cast<int*>(reinterpret_cast<char*>(d.D) + ((size_t)512 << 20));
  if (!f.rec) {
    const size_t mx = (size_t)Flin::LAUNCHES * d.sms * 16;
    CK(cudaMalloc(&f.rec, mx * sizeof(unsigned long long)));
    CK(cudaMalloc(&f.cnt, mx * sizeof(int)));
  }
  CK(cudaMemsetAsync(f.Db[0], 0, (size_t)M * 2 * FF * sizeof(int), d.st));
  CK(cudaMemsetAsync(f.Db[1], 0, (size_t)M * 2 * FF * sizeof(int), d.st));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
}
// launch i (0 ≤ i < 128, layer i / 4, linear i % 4) of the step's sequence; `seq` the linears that run (all four, or one j)
static bool flin_one(Dev& d, const Flin& f, int i, int next_j) {
  const int l = i / 4, j = i % 4;
  FlinArgs a;
  a.W = f.Wt[l][j];
  a.X = j == 3 ? f.XtF : f.XtH;
  a.D = f.Db[i & 1];
  a.Z = f.Db[(i + 1) & 1];
  a.zn = (long long)f.M * LN[next_j];
  a.N = LN[j];
  a.K = LK[j];
  a.c0 = g_c0;
  a.c1 = g_c1;
  a.tile0 = (0x5ull << 56) | ((unsigned long long)i << 24);
  a.rec = f.rec + (size_t)i * f.grid * 16;
  a.cnt = f.cnt + (size_t)i * f.grid * 16;
  a.thr = f.thr;
  return flin_launch(f.M, a, f.grid, f.FW, f.NC, d.st);
}
// the filler's MACs over the launches `ls` (counts from the last run of them), and up to n receipts checked on the host
struct FlinTally {
  double macs = 0;
  long long steps = 0, slots = 0, idle = 0;
  int checked = 0, bad = 0, ctl_equal = 0;
};
static FlinTally flin_tally(const Flin& f, const std::vector<int>& ls, int n, uint64_t salt) {
  FlinTally r;
  if (f.FW == 0) return r;
  const size_t per = (size_t)f.grid * 16;
  std::vector<unsigned long long> rec(per);
  std::vector<int> cnt(per);
  std::vector<std::pair<int, int>> cand;
  std::vector<std::vector<unsigned long long>> recs(ls.size());
  std::vector<std::vector<int>> cnts(ls.size());
  for (size_t q = 0; q < ls.size(); q++) {
    CK(cudaMemcpy(rec.data(), f.rec + (size_t)ls[q] * per, per * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cnt.data(), f.cnt + (size_t)ls[q] * per, per * 4, cudaMemcpyDeviceToHost));
    for (int s = 0; s < f.grid * f.FW; s++) {
      r.steps += cnt[s];
      r.slots++;
      r.idle += cnt[s] == 0;
      if (cnt[s] > 0) cand.push_back({(int)q, s});
    }
    recs[q] = rec;
    cnts[q] = cnt;
  }
  r.macs = (double)r.steps * f.NC * 4096.0;
  std::mt19937_64 rng(salt);
  for (int k = 0; k < n && !cand.empty(); k++) {
    const auto [q, s] = cand[rng() % cand.size()];
    const unsigned long long tile = (0x5ull << 56) | ((unsigned long long)ls[q] << 24) | (unsigned long long)s;
    const int steps = cnts[q][s];
    r.bad += recs[q][s] != dfill_ref(g_c0, g_c1, tile, f.NC, steps);
    if (steps >= 8) r.ctl_equal += recs[q][s] == dfill_ref(g_c0, g_c1, tile, f.NC, steps - 8);
    r.checked++;
  }
  return r;
}

// fgate: flin bit for bit against ref_rows (every row, M ≤ 256) for each linear's shape at M = 32, 64, 128, 256, alone and
// with filler warps (whose receipts are checked against dfill_ref, with the steps − 8 control), from poisoned outputs.
static int flin_gate(Dev& d) {
  int fails = 0;
  for (int M : {32, 64, 128, 256}) {
    Flin f;
    flin_prepare(d, f, M, false);
    for (int j = 0; j < 4; j++)
      for (int fw : {0, M == 256 ? 2 : 4}) {
        f.FW = fw;
        f.NC = 2;
        const int i = j;  // layer 0's linear j
        CK(cudaMemsetAsync(f.Db[i & 1], 0, (size_t)M * LN[j] * 4, d.st));
        CK(cudaMemsetAsync(f.rec + (size_t)i * f.grid * 16, 0xA5, (size_t)f.grid * 16 * 8, d.st));
        if (!flin_one(d, f, i, (j + 1) % 4)) { fprintf(stderr, "flin: M %d FW %d not built\n", M, fw); return 1; }
        ref_rows<<<(unsigned)(((size_t)M * LN[j] + 255) / 256), 256, 0, d.st>>>(d.X, d.W[0][j], g_ref, M, 1, LN[j], LK[j]);
        CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, d.st));
        count_neq_rows<<<(unsigned)(((size_t)M * LN[j] + 255) / 256), 256, 0, d.st>>>(f.Db[i & 1], g_ref, M, 1, LN[j], g_cnt);
        unsigned long long bad = 0;
        CK(cudaMemcpyAsync(&bad, g_cnt, sizeof bad, cudaMemcpyDeviceToHost, d.st));
        CK(cudaStreamSynchronize(d.st));
        CK(cudaGetLastError());
        const FlinTally t = flin_tally(f, {i}, 8, 0x6A7E + M * 16 + j);
        const bool ok = bad == 0 && t.bad == 0 && t.ctl_equal == 0 && (fw == 0 || t.checked > 0);
        printf("{\"mode\": \"fgate\", \"M\": %d, \"linear\": \"%s\", \"N\": %d, \"K\": %d, \"FW\": %d, \"NC\": %d, "
               "\"mismatches\": %llu, \"rec_checked\": %d, \"rec_bad\": %d, \"ctl_equal\": %d, \"filler_steps\": %lld, "
               "\"idle_slots\": %lld, \"ok\": %s}\n",
               M, LNAME[j], LN[j], LK[j], fw, f.NC, bad, t.checked, t.bad, t.ctl_equal, t.steps, t.idle, ok ? "true" : "false");
        fflush(stdout);
        fails += !ok;
      }
  }
  return fails;
}

// flin M FW NC THR: each linear's 32 layers (distinct weights) as one graph: cuBLASLt, flin alone, flin with filler warps
static int run_flin(Dev& d, int M, int FW, int NC, int thr, int reps) {
  Flin f;
  flin_prepare(d, f, M, true);
  f.thr = thr;
  std::vector<Lt> p(4);
  for (int j = 0; j < 4; j++) p[j] = lt_make(M, LN[j], LK[j], d.X, d.W[0][j], d.D, d.st);
  for (int j = 0; j < 4; j++) {
    std::vector<int> ls;
    for (int l = 0; l < LAYERS; l++) ls.push_back(l * 4 + j);
    auto fl = [&](int fw) {
      f.FW = fw;
      f.NC = NC;
      for (int i : ls)
        if (!flin_one(d, f, i, j)) { fprintf(stderr, "flin: M %d FW %d NC %d not built (%s)\n", M, fw, NC, flin_built(M)); exit(1); }
    };
    fl(FW);  // sets the kernels' attributes outside capture
    fl(0);
    CK(cudaStreamSynchronize(d.st));
    const auto vb = time_work(d, [&] { for (int l = 0; l < LAYERS; l++) lt_run(p[j], d.X, d.W[l][j], d.D, d.st); }, reps);
    const auto v0 = time_work(d, [&] { fl(0); }, reps);
    const auto vf = time_work(d, [&] { fl(FW); }, reps);
    const auto vb2 = time_work(d, [&] { for (int l = 0; l < LAYERS; l++) lt_run(p[j], d.X, d.W[l][j], d.D, d.st); }, reps);
    f.FW = FW;
    const FlinTally t = flin_tally(f, ls, 8, 0xF11 + M * 4 + j);
    const Rep mb = median(vb), m0 = median(v0), mf = median(vf), mb2 = median(vb2);
    const double base = (mb.ms + mb2.ms) / 2, cap = 1024.0 * d.sms * mf.mhz * 1e6;
    const double macs = (double)LAYERS * M * LN[j] * LK[j], bytes = (double)LAYERS * LN[j] * LK[j];
    printf("{\"mode\": \"flin\", \"M\": %d, \"linear\": \"%s\", \"N\": %d, \"K\": %d, \"FW\": %d, \"NC\": %d, \"thr\": %d, "
           "\"cublas_ms\": %.4f, \"cublas2_ms\": %.4f, \"flin_ms\": %.4f, \"fill_ms\": %.4f, \"flin_vs_cublas\": %.4f, "
           "\"slowdown\": %.4f, \"cublas_gb_s\": %.1f, \"flin_gb_s\": %.1f, \"useful_share\": %.4f, \"fill_share\": %.4f, "
           "\"mhz\": %.0f, \"watt\": %.1f, \"filler_steps\": %lld, \"idle_slots\": %lld, \"rec_checked\": %d, \"rec_bad\": %d, "
           "\"ctl_equal\": %d}\n",
           M, LNAME[j], LN[j], LK[j], FW, NC, thr, mb.ms, mb2.ms, m0.ms, mf.ms, m0.ms / base, mf.ms / base,
           bytes / (base * 1e-3) / 1e9, bytes / (m0.ms * 1e-3) / 1e9, macs / (mf.ms * 1e-3) / cap,
           t.macs / (mf.ms * 1e-3) / cap, mf.mhz, mf.watt, t.steps, t.idle, t.checked, t.bad, t.ctl_equal);
    fflush(stdout);
    if (t.bad || t.ctl_equal) return 1;
  }
  return 0;
}

// fstep B L FW NC THR: the decode step (enqueue_step's order) with its linears through flin; lm_head stays on cuBLASLt and
// attention is unchanged. Arms: the cuBLASLt step, flin alone, flin with filler warps, the cuBLASLt step again. The
// filler's MACs are its step counts over the 128 launches of the last rep; the last linear's output is checked on every row.
static void enqueue_fstep(Dev& d, const Flin& f, const std::vector<Lt>& p, int B, int L, bool* wrapped) {
  for (int l = 0; l < LAYERS; l++) {
    flin_one(d, f, 4 * l, 1);
    enqueue_att(d, B, L, l, wrapped);
    for (int j = 1; j < 4; j++) flin_one(d, f, 4 * l + j, (j + 1) % 4);
  }
  if (B <= 256 && p[4].algo_id != -2) lt_run(p[4], d.X, d.Wlm, d.D, d.st);
}
static int run_fstep(Dev& d, int B, int L, int FW, int NC, int thr, int reps) {
  const std::vector<Lt> p = lin_plans(d, B);
  Flin f;
  flin_prepare(d, f, B, true);
  f.thr = thr;
  f.NC = NC;
  bool wrapped = false;
  for (int fw : {FW, 0}) {  // the kernels' attributes, outside capture
    f.FW = fw;
    if (!flin_one(d, f, 0, 1)) { fprintf(stderr, "flin: M %d FW %d NC %d not built (%s)\n", B, fw, NC, flin_built(B)); exit(1); }
    for (int i = 1; i < Flin::LAUNCHES; i++) flin_one(d, f, i, (i + 1) % 4);
  }
  CK(cudaStreamSynchronize(d.st));
  auto base = [&] { enqueue_step(d, p, 2, B, L, &wrapped); };
  auto fused = [&](int fw) { f.FW = fw; enqueue_fstep(d, f, p, B, L, &wrapped); };
  const auto vb = time_work(d, base, reps);
  const auto v0 = time_work(d, [&] { fused(0); }, reps);
  const auto vf = time_work(d, [&] { fused(FW); }, reps);
  const auto vb2 = time_work(d, base, reps);
  f.FW = FW;
  std::vector<int> all(Flin::LAUNCHES);
  for (int i = 0; i < Flin::LAUNCHES; i++) all[i] = i;
  const FlinTally t = flin_tally(f, all, 16, 0xF57E + B * 8 + L);
  const int N = LN[3], K = LK[3];
  ref_rows<<<(unsigned)(((size_t)B * N + 255) / 256), 256, 0, d.st>>>(d.X, d.W[LAYERS - 1][3], g_ref, B, 1, N, K);
  CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, d.st));
  count_neq_rows<<<(unsigned)(((size_t)B * N + 255) / 256), 256, 0, d.st>>>(f.Db[(Flin::LAUNCHES - 1) & 1], g_ref, B, 1, N, g_cnt);
  unsigned long long out_bad = 0;
  CK(cudaMemcpyAsync(&out_bad, g_cnt, sizeof out_bad, cudaMemcpyDeviceToHost, d.st));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  double lin_macs = 0;
  for (int j = 0; j < 4; j++) lin_macs += (double)LAYERS * B * LN[j] * LK[j];
  const double step_macs = lin_macs + (double)B * VOCAB * HID + 2.0 * LAYERS * B * HQ * HD * (double)L;
  const Rep mb = median(vb), m0 = median(v0), mf = median(vf), mb2 = median(vb2);
  const double bms = (mb.ms + mb2.ms) / 2, cap = 1024.0 * d.sms * mf.mhz * 1e6;
  printf("{\"mode\": \"fstep\", \"B\": %d, \"L\": %d, \"FW\": %d, \"NC\": %d, \"thr\": %d, \"coins\": [\"%016llx\", \"%016llx\"], "
         "\"kv_wrapped\": %s, \"step_macs\": %.6e, \"lin_macs\": %.6e, \"base_ms\": %.4f, \"base2_ms\": %.4f, \"fused0_ms\": %.4f, "
         "\"fused_ms\": %.4f, \"fused0_vs_base\": %.4f, \"slowdown\": %.4f, \"step_share_base\": %.5f, \"step_share_fused\": %.5f, "
         "\"fill_macs\": %.6e, \"fill_share\": %.5f, \"mhz\": %.0f, \"watt\": %.1f, \"filler_steps\": %lld, \"slots\": %lld, "
         "\"idle_slots\": %lld, \"rec_checked\": %d, \"rec_bad\": %d, \"ctl_equal\": %d, \"out_mismatches\": %llu, \"arms\": {",
         B, L, FW, NC, thr, (unsigned long long)g_c0, (unsigned long long)g_c1, wrapped ? "true" : "false", step_macs, lin_macs,
         mb.ms, mb2.ms, m0.ms, mf.ms, m0.ms / bms, mf.ms / bms, step_macs / (bms * 1e-3) / (1024.0 * d.sms * mb.mhz * 1e6),
         step_macs / (mf.ms * 1e-3) / cap, t.macs, t.macs / (mf.ms * 1e-3) / cap, mf.mhz, mf.watt, t.steps, t.slots, t.idle,
         t.checked, t.bad, t.ctl_equal, out_bad);
  print_arm("base", vb, true);
  print_arm("fused0", v0, true);
  print_arm("fused", vf, true);
  print_arm("base2", vb2, false);
  printf("}}\n");
  fflush(stdout);
  return t.bad || t.ctl_equal || out_bad ? 8 : 0;
}
