// h_falsify: can an adversary that knows A* = B* = 0 produce the zero instance's tiled transcript faster than the honest
// dense int8 kernel? Every arm folds every 32-step prefix word T_ℓ[i, j] into one
// duo receipt (receipt_folds.cuh rf_duo_word, a chain per (i, j) over ℓ), so the output path is identical and the time
// ratio isolates the compute. ref.py is the reference each arm's receipt is checked against, bit for bit.
//
//   honest     generic int8 Ã (m×k), B̃ (k×n): one m16n8k32 IMMA a segment, B̃ reloaded a segment (the baseline).
//   zero_imma  the zero instance R (0/1), U resident: the same IMMA, but B = U is shared by every segment (route a).
//   four_r     Four Russians: tables of U-row sums by 8-bit groups of R in shared memory, four LDS a word (route c).
//
// gpu_kit carries the timing protocol: 0xA5 poison with a no-write control (--gate), interleaved items with the SM clock
// (--time), and the dump format. The reference verifier runs on the dumped receipt after the lease.
#include "../gpu_kit/bench.cuh"
#include "../receipts/receipt_folds.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK GK_CK
static const uint32_t SEED = 0x48464C53u;  // ref.SEED
constexpr int SEG = 32;

// ---- the shared duo fold: segment ℓ's public keys, and a chain's start, matching ref.py -------------------------------
__host__ __device__ inline uint32_t key_a(int ell) {
  for (int i = 0;; i++) {
    uint32_t c = rf_mix32(0xA0000000u ^ ((uint32_t)ell << 8) ^ (uint32_t)i) | 1u;
    if (rf_key_ok(c)) return c;
  }
}
__host__ __device__ inline uint32_t key_k(int ell) { return rf_mix32(0xB0000000u + (uint32_t)ell); }
__host__ __device__ inline int key_r(int ell) { return (ell * 7 + 5) % 31 + 1; }
__host__ __device__ inline uint32_t chain_y0(uint32_t idx) { return rf_mix32(SEED ^ rf_mix32(idx)); }

// ---- the m16n8k32 s8 MMA (IMMA.16832.S8.S8), accumulator c0..c3 the committed s32 words --------------------------------
__device__ __forceinline__ void imma(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Each warp owns 16 rows and WN column-tiles of 8; 8 warps a block share one bi. A fragment of (bi, ℓ) is staged once in
// shared memory and read by all warps. zero_imma keeps U resident (one B fragment a tile for every segment); honest
// reloads B̃'s fragment a segment. acc is the running prefix word; the duo chain folds it each segment.
template <int WN, bool U_RESIDENT>
__global__ void imma_kernel(const uint32_t* __restrict__ Ap,     // [MB][L][128]
                            const uint32_t* __restrict__ Bp,     // honest: [NB][L][64]; zero: [NB][64]
                            uint32_t* __restrict__ out, int L, int n, int tj) {
  __shared__ uint32_t sa[128];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int bi = blockIdx.x;                      // 16-row block
  const int base_nt = blockIdx.y * (8 * WN) + warp * WN;  // this warp's first 8-column tile
  const int row0 = 16 * bi + g;

  int acc[WN][4];
  uint32_t y[WN][4], z[WN][4];
  uint32_t bf[WN][2];
#pragma unroll
  for (int s = 0; s < WN; s++) {
    const int nt = base_nt + s;
    if (U_RESIDENT) {
      bf[s][0] = Bp[nt * 2 + 0];
      bf[s][1] = Bp[nt * 2 + 1];
    }
    const int col0 = 8 * nt + 2 * t;
#pragma unroll
    for (int c = 0; c < 4; c++) {
      acc[s][c] = 0;
      const int ii = row0 + (c >> 1) * 8, jj = col0 + (c & 1);
      y[s][c] = chain_y0((uint32_t)ii * n + jj);
      z[s][c] = rf_mix32(y[s][c] + 0x9E3779B9u);
    }
  }

  for (int ell = 0; ell < L; ell++) {
    sa[threadIdx.x & 127] = Ap[((size_t)bi * L + ell) * 128 + (threadIdx.x & 127)];
    __syncthreads();
    const uint32_t a0 = sa[lane * 4 + 0], a1 = sa[lane * 4 + 1], a2 = sa[lane * 4 + 2], a3 = sa[lane * 4 + 3];
    const uint32_t a_k = key_a(ell), k_k = key_k(ell);
    const int r = key_r(ell);
#pragma unroll
    for (int s = 0; s < WN; s++) {
      const int nt = base_nt + s;
      uint32_t b0, b1;
      if (U_RESIDENT) {
        b0 = bf[s][0];
        b1 = bf[s][1];
      } else {
        b0 = Bp[((size_t)nt * L + ell) * 2 + 0];
        b1 = Bp[((size_t)nt * L + ell) * 2 + 1];
      }
      int c[4] = {0, 0, 0, 0};
      imma(c, a0, a1, a2, a3, b0, b1);
#pragma unroll
      for (int w = 0; w < 4; w++) {
        acc[s][w] += c[w];
        rf_duo_word(y[s][w], z[s][w], (uint32_t)acc[s][w], k_k, a_k, r);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int s = 0; s < WN; s++) {
    const int nt = base_nt + s, col0 = 8 * nt + 2 * t;
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const int ii = row0 + (c >> 1) * 8, jj = col0 + (c & 1);
      out[(size_t)ii * n + jj] = rf_duo_final(y[s][c], z[s][c]);
    }
  }
}

// ---- Four Russians: one CTA a column tile of BN, tables tab[g][p][col] = Σ_{m∈group g, bit p} U[col, m], in shared memory
// (4·256·BN int32); each word is four LDS of a residue byte of R plus three adds. A CTA loops its rows. -----------------
template <int BN>
__global__ void four_russians(const uint8_t* __restrict__ Rb,    // [m][L][4] residue bytes of R
                              const int8_t* __restrict__ U,       // [n][32]
                              uint32_t* __restrict__ out, int L, int m, int n, int row_split) {
  extern __shared__ int tab[];  // [4][256][BN]
  const int col0 = blockIdx.x * BN;
  for (int e = threadIdx.x; e < 4 * 256 * BN; e += blockDim.x) {
    const int col = e % BN, p = (e / BN) & 255, grp = e / (256 * BN);
    int acc = 0;
#pragma unroll
    for (int b = 0; b < 8; b++)
      if (p & (1 << b)) acc += U[(col0 + col) * SEG + grp * 8 + b];
    tab[e] = acc;
  }
  __syncthreads();

  const int rows = (m + row_split - 1) / row_split;
  const int i0 = blockIdx.y * rows, i1 = min(i0 + rows, m);
  for (int e = i0 * BN + threadIdx.x; e < i1 * BN; e += blockDim.x) {
    const int i = e / BN, col = e % BN, j = col0 + col;
    int acc = 0;
    uint32_t y = chain_y0((uint32_t)i * n + j), z = rf_mix32(y + 0x9E3779B9u);
    for (int ell = 0; ell < L; ell++) {
      const uint8_t* rb = &Rb[((size_t)i * L + ell) * 4];
      acc += tab[rb[0] * BN + col] + tab[(256 + rb[1]) * BN + col] + tab[(512 + rb[2]) * BN + col] +
             tab[(768 + rb[3]) * BN + col];
      rf_duo_word(y, z, (uint32_t)acc, key_k(ell), key_a(ell), key_r(ell));
    }
    out[(size_t)i * n + j] = rf_duo_final(y, z);
  }
}

// ---- host: inputs, the packed fragments, and the reference receipt for the gate ---------------------------------------
struct Inputs {
  int m, k, n, L;
  std::vector<int8_t> At, Bt;  // honest generic, m×k and n×k
  std::vector<int8_t> R, U;    // zero instance, m×k (0/1) and n×32
};

static Inputs make_inputs(int m, int k, int n, unsigned seed) {
  Inputs in{m, k, n, k / SEG};
  std::mt19937 g(seed);
  in.At.resize((size_t)m * k);
  in.Bt.resize((size_t)n * k);
  in.R.resize((size_t)m * k);
  in.U.resize((size_t)n * SEG);
  std::uniform_int_distribution<int> s8(-128, 127), bit(0, 1), mag(1, 64), sign(0, 1);
  for (auto& v : in.At) v = (int8_t)s8(g);
  for (auto& v : in.Bt) v = (int8_t)s8(g);
  for (auto& v : in.R) v = (int8_t)bit(g);
  for (auto& v : in.U) v = (int8_t)(sign(g) ? mag(g) : -mag(g));
  return in;
}

// Pack an int8 row-major (rows×k) into m16n8k32 A fragments [rows/16][L][128] (uint32). rows is 16 or n-as-8 handled below.
static std::vector<uint32_t> pack_a(const std::vector<int8_t>& X, int rows, int k) {
  const int L = k / SEG, MB = rows / 16;
  std::vector<uint32_t> out((size_t)MB * L * 128);
  for (int mb = 0; mb < MB; mb++)
    for (int ell = 0; ell < L; ell++)
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane / 4, t = lane % 4;
        const int rr[4] = {g, g + 8, g, g + 8}, cc[4] = {4 * t, 4 * t, 16 + 4 * t, 16 + 4 * t};
        for (int reg = 0; reg < 4; reg++) {
          uint32_t w = 0;
          for (int b = 0; b < 4; b++) {
            const int row = 16 * mb + rr[reg], c = ell * SEG + cc[reg] + b;
            w |= (uint32_t)(uint8_t)X[(size_t)row * k + c] << (8 * b);
          }
          out[(((size_t)mb * L + ell) * 32 + lane) * 4 + reg] = w;
        }
      }
  return out;
}

// Pack an int8 (n×k) into m16n8k32 B fragments [n/8][L][64] (uint32); if resident==true k is 32 and L is 1.
static std::vector<uint32_t> pack_b(const std::vector<int8_t>& Y, int n, int k) {
  const int L = k / SEG, NB = n / 8;
  std::vector<uint32_t> out((size_t)NB * L * 64);
  for (int nb = 0; nb < NB; nb++)
    for (int ell = 0; ell < L; ell++)
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane / 4, t = lane % 4;
        const int cc[2] = {4 * t, 16 + 4 * t};
        for (int reg = 0; reg < 2; reg++) {
          uint32_t w = 0;
          for (int b = 0; b < 4; b++) {
            const int row = 8 * nb + g, c = ell * SEG + cc[reg] + b;
            w |= (uint32_t)(uint8_t)Y[(size_t)row * k + c] << (8 * b);
          }
          out[(((size_t)nb * L + ell) * 32 + lane) * 2 + reg] = w;
        }
      }
  return out;
}

static std::vector<uint8_t> pack_rb(const std::vector<int8_t>& R, int m, int k) {
  const int L = k / SEG;
  std::vector<uint8_t> out((size_t)m * L * 4);
  for (int i = 0; i < m; i++)
    for (int ell = 0; ell < L; ell++)
      for (int grp = 0; grp < 4; grp++) {
        uint8_t b = 0;
        for (int bit = 0; bit < 8; bit++)
          if (R[(size_t)i * k + ell * SEG + grp * 8 + bit]) b |= (uint8_t)(1 << bit);
        out[((size_t)i * L + ell) * 4 + grp] = b;
      }
  return out;
}

// The reference receipt for a set of (i, j): the plain transcript and the stdlib duo, to gate the device output.
static std::vector<uint32_t> ref_receipt(const int8_t* A, const int8_t* B, int k, int n,
                                         const std::vector<std::pair<int, int>>& pairs) {
  const int L = k / SEG;
  std::vector<uint32_t> out(pairs.size());
  for (size_t p = 0; p < pairs.size(); p++) {
    const int i = pairs[p].first, j = pairs[p].second;
    uint32_t y = chain_y0((uint32_t)i * n + j), z = rf_mix32(y + 0x9E3779B9u);
    int acc = 0;
    for (int ell = 0; ell < L; ell++) {
      for (int mm = 0; mm < SEG; mm++) acc += (int)A[(size_t)i * k + ell * SEG + mm] * (int)B[(size_t)j * k + ell * SEG + mm];
      rf_duo_word(y, z, (uint32_t)acc, key_k(ell), key_a(ell), key_r(ell));
    }
    out[p] = rf_duo_final(y, z);
  }
  return out;
}

// B̃ of the zero instance as an (n×k) int8, S[t,j] = U[j, t mod 32], so the generic reference checks the zero arms too.
static std::vector<int8_t> expand_u(const std::vector<int8_t>& U, int n, int k) {
  std::vector<int8_t> out((size_t)n * k);
  for (int j = 0; j < n; j++)
    for (int t = 0; t < k; t++) out[(size_t)j * k + t] = U[(size_t)j * SEG + (t % SEG)];
  return out;
}

template <class T>
static T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

enum Arm { HONEST, ZERO_IMMA, FOUR_R };
static const char* arm_name(Arm a) { return a == HONEST ? "honest" : a == ZERO_IMMA ? "zero_imma" : "four_r"; }

static constexpr int WN = 4, NW = 8, BN_FR = 32;

static void launch(Arm a, int m, int k, int n, int L, const uint32_t* Ap, const uint32_t* Bp, const uint8_t* Rb,
                   const int8_t* Ud, uint32_t* out, cudaStream_t s) {
  if (a == FOUR_R) {
    const int row_split = 8;
    dim3 grid(n / BN_FR, row_split);
    const size_t sm = (size_t)4 * 256 * BN_FR * sizeof(int);
    static bool set = false;
    if (!set) { CK(cudaFuncSetAttribute(four_russians<BN_FR>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm)); set = true; }
    four_russians<BN_FR><<<grid, 256, sm, s>>>(Rb, Ud, out, L, m, n, row_split);
  } else {
    dim3 grid(m / 16, n / (8 * WN * NW));
    if (a == HONEST) imma_kernel<WN, false><<<grid, 32 * NW, 0, s>>>(Ap, Bp, out, L, n, n / (8 * WN));
    else imma_kernel<WN, true><<<grid, 32 * NW, 0, s>>>(Ap, Bp, out, L, n, n / (8 * WN));
  }
}

int main(int argc, char** argv) {
  int m = 4096, k = 14336, n = 4096, reps = 20;
  std::string mode = "time", dumpdir = "out";
  unsigned seed = 1;
  std::vector<Arm> arms = {HONEST, ZERO_IMMA, FOUR_R};
  for (int i = 1; i < argc; i++) {
    std::string f = argv[i];
    auto nx = [&] { return std::atoi(argv[++i]); };
    if (f == "--m") m = nx();
    else if (f == "--k") k = nx();
    else if (f == "--n") n = nx();
    else if (f == "--reps") reps = nx();
    else if (f == "--seed") seed = nx();
    else if (f == "--mode") mode = argv[++i];
    else if (f == "--dump") dumpdir = argv[++i];
    else if (f == "--arms") {
      arms.clear();
      for (std::string t; i + 1 < argc && argv[i + 1][0] != '-';) {
        t = argv[++i];
        arms.push_back(t == "honest" ? HONEST : t == "zero_imma" ? ZERO_IMMA : FOUR_R);
      }
    }
  }
  const int L = k / SEG;
  fprintf(stderr, "m=%d k=%d n=%d L=%d mode=%s\n", m, k, n, L, mode.c_str());

  Inputs in = make_inputs(m, k, n, seed);
  uint32_t *Ap_h = upload(pack_a(in.At, m, k)), *Ap_z = upload(pack_a(in.R, m, k));
  uint32_t *Bp_h = upload(pack_b(in.Bt, n, k)), *Uf = upload(pack_b(in.U, n, SEG));
  uint8_t* Rb = upload(pack_rb(in.R, m, k));
  int8_t* Ud = upload(in.U);
  uint32_t* out = nullptr;
  CK(cudaMalloc(&out, (size_t)m * n * sizeof(uint32_t)));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));

  auto inputs_of = [&](Arm a) -> std::pair<const uint32_t*, const uint32_t*> {
    return a == HONEST ? std::make_pair((const uint32_t*)Ap_h, (const uint32_t*)Bp_h)
                       : std::make_pair((const uint32_t*)Ap_z, (const uint32_t*)Uf);
  };
  // the (n×k) B the reference uses: honest's generic B̃, the zero arms' expanded U
  std::vector<int8_t> Su = expand_u(in.U, n, k);

  if (mode == "gate") {
    std::vector<std::pair<int, int>> pairs;
    std::mt19937 pg(99);
    for (int t = 0; t < 4096 && t < m * n; t++) pairs.push_back({(int)(pg() % m), (int)(pg() % n)});
    bool all = true;
    for (Arm a : arms) {
      auto [Ad, Bd] = inputs_of(a);
      const int8_t* Bref = a == HONEST ? in.Bt.data() : Su.data();
      std::vector<uint32_t> ref = ref_receipt(a == HONEST ? in.At.data() : in.R.data(), Bref, k, n, pairs);
      gk::PoisonGate pgte = gk::poison_gate(
          out, (size_t)m * n * 4, s, [&](cudaStream_t st) { launch(a, m, k, n, L, Ad, Bd, Rb, Ud, out, st); },
          [&](const void* h, size_t) {
            const uint32_t* r = (const uint32_t*)h;
            for (size_t p = 0; p < pairs.size(); p++)
              if (r[(size_t)pairs[p].first * n + pairs[p].second] != ref[p]) return false;
            return true;
          });
      printf("GATE arm=%s accepted=%d control_rejected=%d left=%zu\n", arm_name(a), pgte.accepted,
             pgte.control_rejected, pgte.left);
      all &= pgte.ok();
    }
    // dump R, U, the honest generic operands and each arm's receipt for ref.py's cross-check
    gk::Dump d(dumpdir);
    d.bytes("R", upload(in.R), in.R.size(), s);
    d.bytes("U", upload(in.U), in.U.size(), s);
    d.bytes("At", upload(in.At), in.At.size(), s);
    d.bytes("Bt", upload(in.Bt), in.Bt.size(), s);
    for (Arm a : arms) {
      auto [Ad, Bd] = inputs_of(a);
      launch(a, m, k, n, L, Ad, Bd, Rb, Ud, out, s);
      CK(cudaStreamSynchronize(s));
      d.bytes(std::string("receipt_") + arm_name(a), out, (size_t)m * n * 4, s);
    }
    return all ? 0 : 3;
  }

  gk::Nvml nv(0);
  std::vector<gk::Arm> gkarms;
  for (Arm a : arms) {
    auto [Ad, Bd] = inputs_of(a);
    gkarms.push_back({arm_name(a), [=](cudaStream_t st, int cnt) {
                        for (int i = 0; i < cnt; i++) launch(a, m, k, n, L, Ad, Bd, Rb, Ud, out, st);
                      }});
  }
  gk::summary(gk::timed_items(gkarms, reps, s, nv), gkarms);
  if (nv.ok()) fprintf(stderr, "uuid=%s\n", nv.uuid().c_str());
  return 0;
}
