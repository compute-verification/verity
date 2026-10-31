// What an overflow correction costs inside an s8 Strassen leaf, in W1 writes (32 s8 MACs at the card's peak) per
// (noise row, nonzero): hardness_premise's timing (natural.py's `--alu`).
//
//   leaf_corr FIXTURE_DIR OUT_DIR [--m M] [--reps R] [--peak MAC_PER_SM_CLK]
//
// FIXTURE_DIR is leaf_fixture.py's: the leaf's B operand lo (n × K int8, a natural down projection's Strassen W-form's low
// digit) and its overflow digit hi as per-column (k, hi) lists. E, the noise side (m × K int8), is drawn here. Arms:
//   leaf      E · lo on mma.sync m16n8k32 s8: 128 × 128 tiles of 8 warps (64 × 32 each), a 3-stage cp.async ring of
//             64-byte k-tiles, swizzled ldmatrix; the epilogue stores M^T (n × m int32, a column's rows contiguous).
//   leaf+imad the same leaf, and in its epilogue 256 · E · hi for every overflowing entry of the tile's columns: a warp
//             a column, a lane 4 rows, one 32-bit load of E^T (K × m) per (k, lane) and per row a PRMT and an IMAD.
//   leaf+pack the same, with E^T stored biased (E + 128 as bytes) and summed two rows a register in 16-bit lanes:
//             per (k, lane) a load, two PRMTs and two IMADs (signs kept apart, the bias taken off once a column).
//   corr+imad / corr+pack   the correction alone, in place on M^T, over the columns that have overflow.
//   empty     a kernel that does nothing on corr's grid (its launch cost).
//   peak      mma.sync m16n8k32 s8 from registers: the card's s8 rate, the W1 unit (--peak overrides it, e.g. the H200's
//             wgmma rate when this runs there on mma.sync).
// The gate runs before any timing: every arm's output, written over 0xA5, matches the CPU's E · (lo + 256 hi) on 64
// rows × all columns (E · lo for `leaf`), and the poison alone fails it.
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../../gpu_kit/bench.cuh"

typedef uint32_t u32;

constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3, THREADS = 256;
constexpr int TILE_BYTES = BM * BK;
constexpr int SMEM_MAIN = STAGES * 2 * TILE_BYTES;
constexpr int TP = BM + 4;  // the epilogue's int32 [64 columns][TP]: fragment stores and 16-byte reads both conflict-free
constexpr int SMEM_EPI = 64 * TP * 4;
constexpr int SMEM = SMEM_MAIN > SMEM_EPI ? SMEM_MAIN : SMEM_EPI;
constexpr int PEAK_CHAINS = 8;

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma(int* c, const u32* a, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// byte b of x sign-extended (PTX prmt's default mode: a selector nibble's bit 3 replicates the byte's sign)
template <u32 SEL>
__device__ __forceinline__ int sext(u32 x) {
  u32 r;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(x), "n"(SEL));
  return (int)r;
}
// a 64-byte k-tile row's 16-byte chunk, XOR-swizzled so ldmatrix's 8 rows land on 8 distinct 16-byte bank groups
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void load_stage(u32 sa, u32 sb, const int8_t* __restrict__ E, const int8_t* __restrict__ L,
                                           int m0, int n0, int k0, int K, int tid) {
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int idx = tid + i * THREADS, row = idx >> 2, ch = idx & 3;
    cp_async16(sa + swz(row, ch), E + (size_t)(m0 + row) * K + k0 + ch * 16);
    cp_async16(sb + swz(row, ch), L + (size_t)(n0 + row) * K + k0 + ch * 16);
  }
}

// hi is public (W's form), so everything about it is a table: per entry E^T's row offset in bytes, and per column pack's
// two constants
struct Hi {
  const u32* Et;             // E^T, K × m bytes as words (CORR 2: biased, E + 128)
  const int* colptr;         // n + 1
  const longlong2* ent;      // per overflowing (k, j): {k · m, hi}
  const int2* colc;          // per column: {B, B + 128 Σ hi}, B = 255 Σ_{hi < 0} |hi|
};

// v (rows m0 + 4 lane .. + 3 of column j) += 256 · Σ_k E[row, k] hi[k, j]
template <int CORR>
__device__ __forceinline__ void correct(int4& v, int j, int m0, int lane, const Hi& h) {
  const int beg = __ldg(h.colptr + j), end = __ldg(h.colptr + j + 1);
  if (beg == end) return;
  const char* col = reinterpret_cast<const char*>(h.Et + (m0 >> 2) + lane);
  if constexpr (CORR == 1) {
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll 2
    for (int p = beg; p < end; p++) {
      const longlong2 e = __ldg(h.ent + p);
      const int d = (int)e.y;
      const u32 x = __ldg(reinterpret_cast<const u32*>(col + e.x));
      a0 += d * sext<0x8880>(x);
      a1 += d * sext<0x9991>(x);
      a2 += d * sext<0xAAA2>(x);
      a3 += d * sext<0xBBB3>(x);
    }
    v.x += a0 * 256;
    v.y += a1 * 256;
    v.z += a2 * 256;
    v.w += a3 * 256;
  } else {
    // Two rows a register in 16-bit lanes: a lane ends at B + Σ hi·(E + 128) = Σ_{hi>0} hi·u + Σ_{hi<0} |hi|·(255 − u),
    // in [0, 255 Σ|hi|] ⊂ [0, 65536) (checked on the host), so the 32-bit sum, exact mod 2^32 whatever it borrows on
    // the way, holds both lanes exactly; one IMAD a register a nonzero, no branch on hi's sign
    const int2 c = __ldg(h.colc + j);
    const u32 b2 = (u32)c.x * 0x10001u;
    u32 p01 = b2, p23 = b2;
#pragma unroll 2
    for (int p = beg; p < end; p++) {
      const longlong2 e = __ldg(h.ent + p);
      const u32 d = (u32)e.y;
      const u32 x = __ldg(reinterpret_cast<const u32*>(col + e.x));
      p01 += d * __byte_perm(x, 0, 0x4140);
      p23 += d * __byte_perm(x, 0, 0x4342);
    }
    v.x += ((int)(p01 & 0xffff) - c.y) * 256;
    v.y += ((int)(p01 >> 16) - c.y) * 256;
    v.z += ((int)(p23 & 0xffff) - c.y) * 256;
    v.w += ((int)(p23 >> 16) - c.y) * 256;
  }
}

template <int CORR>
__global__ void __launch_bounds__(THREADS, 2)
    leaf_kernel(const int8_t* __restrict__ E, const int8_t* __restrict__ L, int m, int K, int* __restrict__ Mt, Hi h) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const u32 base = smem_u32(smem);
  int acc[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0;
  const int KT = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load_stage(base + s * 2 * TILE_BYTES, base + s * 2 * TILE_BYTES + TILE_BYTES, E, L, m0, n0, s * BK, K, tid);
    cp_commit();
  }
  const int a_row = wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8, a_ch = lane >> 4;
  const int b_row = wn * 32 + (lane & 7) + (lane >> 4) * 8, b_ch = (lane >> 3) & 1;
#pragma unroll 1
  for (int kt = 0; kt < KT; kt++) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nk = kt + STAGES - 1;
    if (nk < KT) {
      const int s = nk % STAGES;
      load_stage(base + s * 2 * TILE_BYTES, base + s * 2 * TILE_BYTES + TILE_BYTES, E, L, m0, n0, nk * BK, K, tid);
    }
    cp_commit();
    const u32 sa = base + (kt % STAGES) * 2 * TILE_BYTES, sb = sa + TILE_BYTES;
#pragma unroll
    for (int s = 0; s < 2; s++) {
      u32 a[4][4], b[4][2];
#pragma unroll
      for (int mt = 0; mt < 4; mt++) ldsm4(sa + swz(a_row + mt * 16, a_ch + 2 * s), a[mt][0], a[mt][1], a[mt][2], a[mt][3]);
#pragma unroll
      for (int p = 0; p < 2; p++)
        ldsm4(sb + swz(b_row + p * 16, b_ch + 2 * s), b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1]);
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++) mma(acc[mt][nt], a[mt], b[nt][0], b[nt][1]);
    }
  }
  cp_wait<0>();
  __syncthreads();
  int* T = reinterpret_cast<int*>(smem);
  const int g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int half = 0; half < 2; half++) {
    if ((wn >> 1) == half) {
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          const int r = wm * 64 + mt * 16 + g, c = (wn & 1) * 32 + nt * 8 + 2 * q;
          T[c * TP + r] = acc[mt][nt][0];
          T[(c + 1) * TP + r] = acc[mt][nt][1];
          T[c * TP + r + 8] = acc[mt][nt][2];
          T[(c + 1) * TP + r + 8] = acc[mt][nt][3];
        }
    }
    __syncthreads();
#pragma unroll 1
    for (int cc = 0; cc < 8; cc++) {
      const int c = warp * 8 + cc, j = n0 + half * 64 + c;
      int4 v = *reinterpret_cast<const int4*>(&T[c * TP + 4 * lane]);
      if constexpr (CORR != 0) correct<CORR>(v, j, m0, lane, h);
      *reinterpret_cast<int4*>(&Mt[(size_t)j * m + m0 + 4 * lane]) = v;
    }
    __syncthreads();
  }
}

// the correction alone: block (row tile, 8 overflowing columns), a warp a column
template <int CORR>
__global__ void __launch_bounds__(THREADS) corr_kernel(int* __restrict__ Mt, int m, const int* __restrict__ cols, int ncols,
                                                       Hi h) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int m0 = blockIdx.x * BM, ci = blockIdx.y * 8 + warp;
  if (ci >= ncols) return;
  const int j = __ldg(cols + ci);
  int4* p = reinterpret_cast<int4*>(&Mt[(size_t)j * m + m0 + 4 * lane]);
  int4 v = *p;
  correct<CORR>(v, j, m0, lane, h);
  *p = v;
}

__global__ void empty_kernel(int* out) {
  if (out && threadIdx.x == 1u << 30) out[0] = 0;
}

__global__ void __launch_bounds__(THREADS) peak_kernel(int iters, u32 seed, int* __restrict__ out) {
  const u32 t = seed + threadIdx.x;
  const u32 a[4] = {t * 0x9E3779B1u, t * 0x85EBCA77u, t * 0xC2B2AE3Du, t * 0x27D4EB2Fu};
  const u32 b0 = t * 0x165667B1u, b1 = t * 0xD3A2646Cu;
  int acc[PEAK_CHAINS][4];
#pragma unroll
  for (int c = 0; c < PEAK_CHAINS; c++) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0;
#pragma unroll 1
  for (int i = 0; i < iters; i++)
#pragma unroll
    for (int c = 0; c < PEAK_CHAINS; c++) mma(acc[c], a, b0, b1);
  int s = 0;
#pragma unroll
  for (int c = 0; c < PEAK_CHAINS; c++) s += acc[c][0] ^ acc[c][1] ^ acc[c][2] ^ acc[c][3];
  out[blockIdx.x * THREADS + threadIdx.x] = s;
}

// ---------------------------------------------------------------- host

template <class T>
static std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    fprintf(stderr, "cannot read %s\n", path.c_str());
    exit(2);
  }
  const size_t n = (size_t)f.tellg();
  std::vector<T> v(n / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)n);
  return v;
}

template <class T>
static T* dput(const std::vector<T>& h) {
  T* d = nullptr;
  GK_CK(cudaMalloc(&d, h.size() * sizeof(T)));
  GK_CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static double median_of(const gk::Timing& t, int arm, bool mhz) {
  std::vector<double> v;
  for (const gk::Item& it : t.items)
    if (it.arm == arm) v.push_back(mhz ? (double)it.sm_mhz() : it.ms / it.calls);
  return gk::median(v);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: leaf_corr FIXTURE_DIR OUT_DIR [--m M] [--reps R] [--peak MAC_PER_SM_CLK]\n");
    return 2;
  }
  const std::string fx = argv[1], out = argv[2];
  int m_arg = 0, reps = 15;
  double peak_override = 0;
  for (int i = 3; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--m")) m_arg = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--peak")) peak_override = atof(argv[i + 1]);
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  std::vector<int8_t> lo = read_bin<int8_t>(fx + "/lo.bin");
  std::vector<int> colptr = read_bin<int>(fx + "/colptr.bin"), ent = read_bin<int>(fx + "/ent.bin");
  const int n = (int)colptr.size() - 1, K = (int)(lo.size() / n);
  const long long nnz = (long long)ent.size();
  if ((size_t)n * K != lo.size() || n % BN || K % BK || colptr[n] != nnz) {
    fprintf(stderr, "bad fixture: n %d K %d nnz %lld\n", n, K, nnz);
    return 2;
  }
  std::vector<int> cols;
  long long max_abs = 0;
  for (int j = 0; j < n; j++) {
    long long s = 0;
    for (int p = colptr[j]; p < colptr[j + 1]; p++) s += std::abs((int)(int8_t)(ent[p] & 255));
    max_abs = std::max(max_abs, s);
    if (colptr[j + 1] > colptr[j]) cols.push_back(j);
  }
  if (max_abs * 255 >= 65536) {
    fprintf(stderr, "a column's sum of |hi| (%lld) overflows pack's 16-bit lanes\n", max_abs);
    return 2;
  }

  int dev = 0;
  GK_CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  GK_CK(cudaGetDeviceProperties(&prop, dev));
  const int sms = prop.multiProcessorCount;
  GK_CK(cudaFuncSetAttribute(leaf_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  int occ = 0;
  GK_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, leaf_kernel<1>, THREADS, SMEM));
  // m: two full waves of 128 × 128 tiles over the n / 128 column tiles
  const int m = m_arg ? m_arg : BM * std::max(1, (2 * sms * occ + n / BN - 1) / (n / BN));
  if (m % BM) {
    fprintf(stderr, "m %d is not a multiple of %d\n", m, BM);
    return 2;
  }

  std::vector<int8_t> E((size_t)m * K), Et((size_t)K * m), Etb((size_t)K * m);
  uint64_t st = 0x2545F4914F6CDD1Dull;
  for (auto& x : E) {
    st ^= st << 13, st ^= st >> 7, st ^= st << 17;
    x = (int8_t)(st >> 24);
  }
  for (int i = 0; i < m; i++)
    for (int k = 0; k < K; k++) {
      Et[(size_t)k * m + i] = E[(size_t)i * K + k];
      Etb[(size_t)k * m + i] = (int8_t)(E[(size_t)i * K + k] ^ 0x80);
    }
  int8_t *dE = dput(E), *dL = dput(lo), *dEt = dput(Et), *dEtb = dput(Etb);
  std::vector<longlong2> ent2(ent.size());
  for (size_t p = 0; p < ent.size(); p++)
    ent2[p] = make_longlong2((long long)(ent[p] >> 8) * m, (long long)(int8_t)(ent[p] & 255));
  std::vector<int2> colc(n);
  for (int j = 0; j < n; j++) {
    int neg = 0, sum = 0;
    for (int p = colptr[j]; p < colptr[j + 1]; p++) {
      const int d = (int8_t)(ent[p] & 255);
      sum += d;
      if (d < 0) neg -= d;
    }
    colc[j] = make_int2(255 * neg, 255 * neg + 128 * sum);
  }
  int *dcolptr = dput(colptr), *dcols = dput(cols);
  longlong2* dent = dput(ent2);
  int2* dcolc = dput(colc);
  int *dM = nullptr, *dS = nullptr, *dpeak = nullptr;
  const size_t mbytes = (size_t)n * m * 4;
  GK_CK(cudaMalloc(&dM, mbytes));
  GK_CK(cudaMalloc(&dS, mbytes));
  const Hi h1{reinterpret_cast<const u32*>(dEt), dcolptr, dent, dcolc},
      h2{reinterpret_cast<const u32*>(dEtb), dcolptr, dent, dcolc};
  cudaStream_t s;
  GK_CK(cudaStreamCreate(&s));
  const dim3 lgrid(n / BN, m / BM), cgrid(m / BM, (unsigned)((cols.size() + 7) / 8));
  const int peak_blocks = sms * 4, peak_iters = 4096;
  GK_CK(cudaMalloc(&dpeak, (size_t)peak_blocks * THREADS * 4));

  // the CPU's reference on 64 rows (all residues mod 128 that 131·t reaches), every column
  const int NR = 64;
  std::vector<int> rows(NR);
  for (int t = 0; t < NR; t++) rows[t] = (int)((131LL * t + 7) % m);
  std::vector<long long> ref_lo((size_t)NR * n), ref_full((size_t)NR * n);
  for (int t = 0; t < NR; t++) {
    const int8_t* e = &E[(size_t)rows[t] * K];
    for (int j = 0; j < n; j++) {
      const int8_t* l = &lo[(size_t)j * K];
      long long a = 0;
      for (int k = 0; k < K; k++) a += (int)e[k] * (int)l[k];
      long long c = 0;
      for (int p = colptr[j]; p < colptr[j + 1]; p++) c += (long long)(int8_t)(ent[p] & 255) * e[ent[p] >> 8];
      ref_lo[(size_t)t * n + j] = a;
      ref_full[(size_t)t * n + j] = a + 256 * c;
    }
  }
  auto matches = [&](const std::vector<long long>& ref) {
    return [&, ref_p = &ref](const void* hptr, size_t) {
      const int* Mh = static_cast<const int*>(hptr);
      for (int t = 0; t < NR; t++)
        for (int j = 0; j < n; j++)
          if ((long long)Mh[(size_t)j * m + rows[t]] != (*ref_p)[(size_t)t * n + j]) return false;
      return true;
    };
  };
  auto leaf = [&](int corr, int* M, cudaStream_t st_) {
    if (corr == 0) leaf_kernel<0><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h1);
    if (corr == 1) leaf_kernel<1><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h1);
    if (corr == 2) leaf_kernel<2><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h2);
  };
  auto corr = [&](int c, int* M, cudaStream_t st_) {
    if (c == 1) corr_kernel<1><<<cgrid, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), h1);
    if (c == 2) corr_kernel<2><<<cgrid, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), h2);
  };

  bool ok = true;
  struct GateRow { const char* name; gk::PoisonGate g; };
  std::vector<GateRow> gates;
  gates.push_back({"leaf", gk::poison_gate(dM, mbytes, s, [&](cudaStream_t st_) { leaf(0, dM, st_); }, matches(ref_lo))});
  gates.push_back({"leaf+imad", gk::poison_gate(dM, mbytes, s, [&](cudaStream_t st_) { leaf(1, dM, st_); }, matches(ref_full))});
  gates.push_back({"leaf+pack", gk::poison_gate(dM, mbytes, s, [&](cudaStream_t st_) { leaf(2, dM, st_); }, matches(ref_full))});
  for (int c = 1; c <= 2; c++) {
    // the correction applied to the uncorrected leaf's output; the control writes the leaf's output only
    auto arm = [&, c](cudaStream_t st_) { leaf(0, dM, st_); corr(c, dM, st_); };
    auto ctl = [&](cudaStream_t st_) { leaf(0, dM, st_); };
    gates.push_back({c == 1 ? "corr+imad" : "corr+pack", gk::poison_gate(dM, mbytes, s, arm, matches(ref_full), ctl)});
  }
  for (auto& r : gates) {
    printf("{\"gate\": \"%s\", \"accepted\": %s, \"control_rejected\": %s, \"poison_left\": %zu}\n", r.name,
           r.g.accepted ? "true" : "false", r.g.control_rejected ? "true" : "false", r.g.left);
    ok &= r.g.ok();
  }
  fflush(stdout);
  if (!ok) {
    fprintf(stderr, "gate failed: no timing\n");
    return 3;
  }

  GK_CK(cudaMemcpy(dS, dM, mbytes, cudaMemcpyDeviceToDevice));
  gk::Nvml nv(dev);
  std::vector<gk::Arm> arms = {
      {"leaf", [&](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) leaf(0, dM, st_); }},
      {"leaf+imad", [&](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) leaf(1, dM, st_); }},
      {"leaf+pack", [&](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) leaf(2, dM, st_); }},
      {"corr+imad", [&](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) corr(1, dS, st_); }},
      {"corr+pack", [&](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) corr(2, dS, st_); }},
      {"empty", [&](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) empty_kernel<<<cgrid, THREADS, 0, st_>>>(nullptr); }},
      {"peak", [&](cudaStream_t st_, int k) {
         for (int i = 0; i < k; i++) peak_kernel<<<peak_blocks, THREADS, 0, st_>>>(peak_iters, 1u + i, dpeak);
       }},
  };
  gk::Timing t = gk::timed_items(arms, reps, s, nv);
  GK_CK(cudaGetLastError());
  gk::summary(t, arms);
  {
    FILE* f = fopen((out + "/items.jsonl").c_str(), "w");
    if (f) {
      gk::items_jsonl(t, arms, f);
      fclose(f);
    }
  }
  // per arm: median ms and SM clock; cycles = ms · MHz · 1e3 (per SM)
  std::vector<double> ms(arms.size()), mhz(arms.size()), cyc(arms.size());
  for (size_t a = 0; a < arms.size(); a++) {
    ms[a] = median_of(t, (int)a, false);
    mhz[a] = median_of(t, (int)a, true);
    cyc[a] = ms[a] * mhz[a] * 1e3;
  }
  const double peak_macs = (double)peak_blocks * (THREADS / 32) * peak_iters * PEAK_CHAINS * 16 * 8 * 32;
  const double peak_measured = peak_macs / (cyc[6] * sms);
  const double peak = peak_override > 0 ? peak_override : peak_measured;  // MAC per SM-clock: W1 per SM-clock
  const double leaf_macs = (double)m * n * K, pairs = (double)m * nnz;
  auto writes = [&](double cycles) { return cycles * sms * peak / (32.0 * pairs); };
  printf(
      "{\"result\": \"leaf_corr\", \"gpu\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"occupancy\": %d, \"m\": %d, \"n\": %d, "
      "\"K\": %d, \"nnz\": %lld, \"cols\": %zu, \"pairs\": %.0f, \"steady\": %s, \"peak_mac_per_sm_clk_measured\": %.2f, "
      "\"peak_mac_per_sm_clk\": %.2f",
      prop.name, prop.major, prop.minor, sms, occ, m, n, K, nnz, cols.size(), pairs, t.steady ? "true" : "false",
      peak_measured, peak);
  for (size_t a = 0; a < arms.size(); a++)
    printf(", \"%s\": {\"ms\": %.6f, \"sm_mhz\": %.0f, \"cycles\": %.1f}", arms[a].name.c_str(), ms[a], mhz[a], cyc[a]);
  const double eff = leaf_macs / (cyc[0] * sms * peak);
  printf(
      ", \"leaf_efficiency\": %.4f, \"writes\": {\"leaf+imad_marginal\": %.4f, \"leaf+pack_marginal\": %.4f, "
      "\"corr+imad\": %.4f, \"corr+pack\": %.4f, \"corr+imad_less_launch\": %.4f, \"corr+pack_less_launch\": %.4f, "
      "\"empty\": %.4f}}\n",
      eff, writes(cyc[1] - cyc[0]), writes(cyc[2] - cyc[0]), writes(cyc[3]), writes(cyc[4]), writes(cyc[3] - cyc[5]),
      writes(cyc[4] - cyc[5]), writes(cyc[5]));
  // the same by time alone, each arm at its own clock, against the peak arm's MAC a ms (no NVML needed)
  auto writes_t = [&](double dms) { return dms * (peak_macs / ms[6]) / (32.0 * pairs); };
  printf(
      "{\"result\": \"leaf_corr_by_time\", \"peak_mac_per_ms\": %.4e, \"leaf_efficiency\": %.4f, \"writes\": "
      "{\"leaf+imad_marginal\": %.4f, \"leaf+pack_marginal\": %.4f, \"corr+imad\": %.4f, \"corr+pack\": %.4f, "
      "\"corr+imad_less_launch\": %.4f, \"corr+pack_less_launch\": %.4f, \"empty\": %.4f}}\n",
      peak_macs / ms[6], leaf_macs / (ms[0] * peak_macs / ms[6]), writes_t(ms[1] - ms[0]), writes_t(ms[2] - ms[0]),
      writes_t(ms[3]), writes_t(ms[4]), writes_t(ms[3] - ms[5]), writes_t(ms[4] - ms[5]), writes_t(ms[5]));
  return 0;
}
