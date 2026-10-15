// Cutting P V's noise cost on sm_120a (H = 32 heads, head dimension 128, sequence s). The noise is SHA-512 in counter
// mode (sha_noise.cuh: pm1, u17, u17w), V's fresh for every head, and P's either
//   shared: one E (s x s) a layer for all H heads, P'_h = P_h + E (sha_fill writes E, then E is added from the buffer), or
//   unshared: fresh for every head, generated where P is produced.
// The methods:
//   sepS / sepSd: E, the pass P + E (shared_edit), V's edit pass, then cuBLASLt (or the custom mainloop with duo 32)
//   fusS / fovS / fusSd: E, V's edit pass, and P V's mainloop adding E to each stage of P in shared memory (fgemm.cu's
//         NS < 0, serialized or overlapped; tiles heads-innermost, so the H CTAs reading one tile of E run together)
//   prod*: the producer, S = Q K^T a 128 x 128 tile at a time and the softmax step's quantization
//         P~ = min(119, rint(119 2^(c S - c max_i))) with the row maxima of a first pass (prodmax), written 16 bytes a
//         thread, with the noise added on the way out: none (prodN), shared E (prodS, E generated first) or SHA in the
//         epilogue (prodU, one warp hashing the tile's 32 patches before its mainloop); then V's edit pass and P V on the
//         edited operands by cuBLASLt (prod?lt) or duo 32 (prod?d). The route's cost is its time less prodN's.
// P in [0, 119] (as P~), V, Q and K in [-119, 119]: P' and V' can't saturate and P' V'^T is exact in s32.
//   pvgemm gate DIR S H SEED GRID   every arm at one small shape, inputs and outputs to DIR for pvref.py
//   pvgemm bench REPS S...          interleaved timing, one JSON line a sequence length; PV_ARMS
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a pvgemm.cu -lcuda -lcublasLt -lnvidia-ml
#define MG_LIB
#include "mgemm.cu"
#include "sha_pass.cuh"

constexpr int HD = 128;

// P + E into XA, E shared by the heads (row mod s), one 16-byte chunk a thread, heads innermost so neighbouring warps
// share E's rows
__global__ void __launch_bounds__(256) shared_edit(const uint8_t* __restrict__ P, uint8_t* __restrict__ XA, const uint8_t* __restrict__ E,
                                                   uint32_t s, uint32_t H, size_t chunks) {
  const uint32_t s16 = s / 16;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / s16;
    const uint32_t cc = (uint32_t)(i % s16), h = (uint32_t)(r % H), row = (uint32_t)(r / H);
    const size_t off = ((size_t)h * s + row) * s + cc * 16, eo = (size_t)row * s + cc * 16;
    const uint4 x = *reinterpret_cast<const uint4*>(P + off), e = __ldg(reinterpret_cast<const uint4*>(E + eo));
    *reinterpret_cast<uint4*>(XA + off) =
        make_uint4(add_noise<false>(x.x, e.x), add_noise<false>(x.y, e.y), add_noise<false>(x.z, e.z), add_noise<false>(x.w, e.w));
  }
}

#define GEMMB(NAME, RS, KIND, NZ)                                                                                     \
  __global__ void __launch_bounds__(ML<128, 128, 32, 3, false, NZ, -1>::NT, 1) NAME(const __grid_constant__ Args a) { \
    gemm_body<128, 128, 32, 3, false, RS, KIND, false, NZ, -1>(a);                                                    \
  }
GEMMB(b1_plain, 0, NONE, 1)
GEMMB(b1_duo32, 32, DUO, 1)
GEMMB(b2_plain, 0, NONE, 2)
static_assert(ML<128, 128, 32, 3, false, 2, -1>::SMEM == CSMEM, "one shared-memory size");

struct PArgs {
  const Maps* maps;   // a: Q (H s x 128), b: K (H s x 128)
  int* mx;            // S's row maxima (H s); MODE 0 takes them by atomicMax
  uint8_t* P;         // P~ or P~ + noise (H s x s); MODE 1 writes it
  const uint8_t* E;   // NS < 0: the shared noise (s x s)
  int s, H;
  float c;            // S's scale, in powers of 2
  uint64_t sg[4];
};
using PL = ML<128, 128, 32, 2, false, 0, -1>;   // k = 128 is one stage a tile; heads innermost
constexpr int PSMEM = 1024 + 2 * PL::STAGE + 128 + 2 * 16384;

template <int MODE, int NS> __global__ void __launch_bounds__(PL::NT, 1) prod(const __grid_constant__ PArgs a) {
  using L = PL;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.s, a.s, HD, 8, false, a.H);
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int tiles = a.s / 128, wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3;
  const uint32_t ptile = ml.bars + 128, ntile = ptile + 16384;
  uint32_t kr[L::template NK<NONE>];
  L::template keys<NONE>(kr, lane);
  for (int t = blockIdx.x; t < tiles * tiles * a.H; t += gridDim.x) {
    int bi, tm, tn;
    ml.tile_of(t, bi, tm, tn);
    const int row0 = (bi * tiles + tm) * 128;
    if constexpr (MODE == 1 && NS > 0) {
      if (warp == 1)
        sha_patch<NS>(a.sg, 2 * (NS - 1), (uint32_t)((row0 / 4 + lane) * tiles + tn), [&](int c, const uint32_t(&n)[4]) {
          const uint32_t addr = ntile + (4 * lane + c / 8) * 128 + (c % 8) * 16;
          asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(n[0]), "r"(n[1]), "r"(n[2]), "r"(n[3]));
        });
    }
    St ch[NCH<NONE>];
#pragma unroll
    for (int e = 0; e < NCH<NONE>; e++) ch[e].y = ch[e].z = 0;
    int acc[MI][NI][4];
    ml.template tile<0, NONE>(HD, 0, acc, ch, kr);
    if constexpr (MODE == 0) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int v = acc[mi][0][2 * h];
#pragma unroll
          for (int ni = 0; ni < NI; ni++) v = max(v, max(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]));
          v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
          v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
          if (q == 0) atomicMax(&a.mx[row0 + wm * 32 + 16 * mi + 8 * h + g], v);
        }
    } else {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int r = wm * 32 + 16 * mi + 8 * h + g;
          const float m = (float)a.mx[row0 + r] * a.c;
#pragma unroll
          for (int ni = 0; ni < NI; ni++) {
            const int p0 = min(119, __float2int_rn(119.f * exp2f(fmaf((float)acc[mi][ni][2 * h], a.c, -m))));
            const int p1 = min(119, __float2int_rn(119.f * exp2f(fmaf((float)acc[mi][ni][2 * h + 1], a.c, -m))));
            const uint32_t addr = ptile + r * 128 + (((2 * wn + ni / 2) ^ (r & 7)) << 4) + 8 * (ni & 1) + 2 * q;
            asm volatile("st.shared.u16 [%0], %1;" :: "r"(addr), "h"((unsigned short)(p0 | p1 << 8)));
          }
        }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int c = threadIdx.x + i * L::NCT, r = c >> 3, lc = c & 7;
        uint32_t v0, v1, v2, v3;
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                     : "r"(ptile + r * 128 + ((lc ^ (r & 7)) << 4)));
        if constexpr (NS < 0) {
          const uint4 e = __ldg(reinterpret_cast<const uint4*>(a.E + (size_t)(tm * 128 + r) * a.s + tn * 128 + lc * 16));
          v0 = add_noise<false>(v0, e.x); v1 = add_noise<false>(v1, e.y); v2 = add_noise<false>(v2, e.z); v3 = add_noise<false>(v3, e.w);
        } else if constexpr (NS > 0) {
          uint32_t n0, n1, n2, n3;
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(n0), "=r"(n1), "=r"(n2), "=r"(n3)
                       : "r"(ntile + r * 128 + lc * 16));
          v0 = add_noise<false>(v0, n0); v1 = add_noise<false>(v1, n1); v2 = add_noise<false>(v2, n2); v3 = add_noise<false>(v3, n3);
        }
        *reinterpret_cast<uint4*>(a.P + (size_t)(row0 + r) * a.s + tn * 128 + lc * 16) = make_uint4(v0, v1, v2, v3);
      }
      __syncthreads();
    }
  }
}
typedef void (*PKern)(const PArgs);
static PKern PK_U[3] = {prod<1, 1>, prod<1, 2>, prod<1, 3>};

static const char* ALN[3] = {"pm1", "u17", "u17w"};

struct VProb {
  int s, H, grid, pgrid;
  uint32_t pE, pV;
  uint8_t *P, *Vt, *Q, *K, *XA, *XB, *Pt, *E;
  int *U, *mx;
  uint32_t *R, *G;
  size_t nP, nV, nQ, ub, rb;
  LtG plain, ltx;
  Maps *mPV, *mPX, *mXX, *mQK;
  uint32_t seed, inst = 0;
  float c;
  cudaStream_t st;
};

static void prod_launch(VProb& P, PKern k, uint8_t* out, const Sig& s) {
  PArgs a{P.mQK, P.mx, out, P.E, P.s, P.H, P.c, {s.w[0], s.w[1], s.w[2], s.w[3]}};
  k<<<P.pgrid, PL::NT, PSMEM, P.st>>>(a);
}

static void vsetup(VProb& P, int s, int H, uint32_t seed, int grid_cap) {
  P.s = s; P.H = H; P.seed = seed; P.c = 1.0f / 65536;
  if (s % 128) { fprintf(stderr, "s %d not tiled\n", s); exit(2); }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  P.grid = std::min((s / 128) * H, grid_cap > 0 ? grid_cap : nsm);
  P.pgrid = std::min((s / 128) * (s / 128) * H, grid_cap > 0 ? grid_cap : nsm);
  CK(cudaStreamCreate(&P.st));
  P.nP = (size_t)H * s * s; P.nV = (size_t)H * HD * s; P.nQ = (size_t)H * s * HD;
  P.pE = (uint32_t)((size_t)s * s / 512); P.pV = (uint32_t)(P.nV / 512);
  for (uint8_t** b : {&P.P, &P.XA, &P.Pt}) CK(cudaMalloc(b, P.nP));
  for (uint8_t** b : {&P.Vt, &P.XB}) CK(cudaMalloc(b, P.nV));
  CK(cudaMalloc(&P.Q, P.nQ)); CK(cudaMalloc(&P.K, P.nQ)); CK(cudaMalloc(&P.E, (size_t)s * s));
  CK(cudaMalloc(&P.mx, (size_t)H * s * 4)); CK(cudaMalloc(&P.G, (size_t)P.pV * 4));
  P.ub = (size_t)H * s * HD * 4;
  P.rb = (size_t)(s / 128) * H * CNT * 8 * 4;
  CK(cudaMalloc(&P.U, P.ub)); CK(cudaMalloc(&P.R, P.rb));
  sfill<I8><<<4096, 256, 0, P.st>>>(P.P, P.nP, mix32(seed ^ 0x1111u), 0, 120);
  sfill<I8><<<4096, 256, 0, P.st>>>(P.Vt, P.nV, mix32(seed ^ 0x2222u));
  sfill<I8><<<4096, 256, 0, P.st>>>(P.Q, P.nQ, mix32(seed ^ 0x3333u));
  sfill<I8><<<4096, 256, 0, P.st>>>(P.K, P.nQ, mix32(seed ^ 0x4444u));
  P.plain = ltg(I8, s, HD, s, s, s, HD, P.P, P.Vt, P.U, 0, P.st, H);
  P.ltx = ltg(I8, s, HD, s, s, s, HD, P.XA, P.XB, P.U, 0, P.st, H);
  P.mPV = upload_maps(Maps{tmap(P.P, s, H * s, 128), tmap(P.Vt, s, H * HD, 128), tmap(P.P, s, H * s, 128), tmap(P.Vt, s, H * HD, 128)});
  P.mPX = upload_maps(Maps{tmap(P.P, s, H * s, 128), tmap(P.XB, s, H * HD, 128), tmap(P.P, s, H * s, 128), tmap(P.XB, s, H * HD, 128)});
  P.mXX = upload_maps(Maps{tmap(P.XA, s, H * s, 128), tmap(P.XB, s, H * HD, 128), tmap(P.XA, s, H * s, 128), tmap(P.XB, s, H * HD, 128)});
  P.mQK = upload_maps(Maps{tmap(P.Q, HD, H * s, 128), tmap(P.K, HD, H * s, 128), tmap(P.Q, HD, H * s, 128), tmap(P.K, HD, H * s, 128)});
  for (Kern k : {RCK[0][1], (Kern)b1_plain, (Kern)b1_duo32, (Kern)b2_plain})
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  for (PKern k : {prod<0, 0>, prod<1, 0>, prod<1, -1>, prod<1, 1>, prod<1, 2>, prod<1, 3>})
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, PSMEM));
  CK(cudaMemsetAsync(P.mx, 0x80, (size_t)H * s * 4, P.st));
  prod_launch(P, prod<0, 0>, P.Pt, sigma(seed, 0));
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
}
static void vfree(VProb& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.P, (void*)P.XA, (void*)P.Pt, (void*)P.Vt, (void*)P.XB, (void*)P.Q, (void*)P.K, (void*)P.E, (void*)P.mx,
                  (void*)P.G, (void*)P.U, (void*)P.R, (void*)P.mPV, (void*)P.mPX, (void*)P.mXX, (void*)P.mQK})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(P.st));
}

enum { V_PLAIN, V_CDUO, V_PRODMAX, V_PRODN, V_GENP, V_GENV, V_EDITV, V_SEPS, V_SEPSD, V_FUSS, V_FOVS, V_FUSSD, V_PRODS, V_PRODU,
       V_PRODSLT, V_PRODULT, V_PRODSD, V_PRODUD };
struct VArm { std::string name; int kind, al; };
static std::vector<VArm> vall() {
  std::vector<VArm> v{{"plain", V_PLAIN, 0}, {"cduo", V_CDUO, 0}, {"prodmax", V_PRODMAX, 0}, {"prodN", V_PRODN, 0}};
  const std::pair<const char*, int> per[] = {{"genP", V_GENP}, {"genV", V_GENV}, {"editV", V_EDITV}, {"sepS", V_SEPS},
                                             {"sepSd", V_SEPSD}, {"fusS", V_FUSS}, {"fovS", V_FOVS}, {"fusSd", V_FUSSD},
                                             {"prodS", V_PRODS}, {"prodU", V_PRODU}, {"prodSlt", V_PRODSLT}, {"prodUlt", V_PRODULT},
                                             {"prodSd", V_PRODSD}, {"prodUd", V_PRODUD}};
  for (int al = 0; al < 3; al++)
    for (auto& [n, k] : per) v.push_back({std::string(n) + "_" + ALN[al], k, al});
  return v;
}
static void genP(VProb& P, int al, const Sig& s) {
  const uint32_t g = (P.pE + 255) / 256, KB = P.s / 128;
  if (al == 0) sha_fill<1><<<g, 256, 0, P.st>>>(P.E, P.pE, KB, s);
  else if (al == 1) sha_fill<2><<<g, 256, 0, P.st>>>(P.E, P.pE, KB, s);
  else sha_fill<3><<<g, 256, 0, P.st>>>(P.E, P.pE, KB, s);
}
static void editV(VProb& P, int al, const Sig& s) {
  const uint32_t g = (P.pV + 255) / 256, KB = P.s / 128;
  if (al == 0) sha_edit<1, false><<<g, 256, 0, P.st>>>(nullptr, nullptr, P.Vt, P.XB, 0, P.pV, KB, s);
  else if (al == 1) sha_edit<2, false><<<g, 256, 0, P.st>>>(nullptr, nullptr, P.Vt, P.XB, 0, P.pV, KB, s);
  else sha_edit<3, false><<<g, 256, 0, P.st>>>(nullptr, nullptr, P.Vt, P.XB, 0, P.pV, KB, s);
}
static void pv_gemm(VProb& P, Kern k, const Maps* m, const Sig& s) {
  Args a{m, P.U, P.R, P.s, HD, P.s, 8, P.seed, 0, P.H, 0, 0, {s.w[0], s.w[1], s.w[2], s.w[3]}, P.E};
  k<<<P.grid, CNT, CSMEM, P.st>>>(a);
}
static void sep_shared(VProb& P, int al, const Sig& s) {
  genP(P, al, s);
  const size_t ch = P.nP / 16;
  shared_edit<<<(unsigned)std::min<size_t>((ch + 255) / 256, 1u << 20), 256, 0, P.st>>>(P.P, P.XA, P.E, P.s, P.H, ch);
  editV(P, al, s);
}
static void vrun(VProb& P, const VArm& a) {
  const Sig s = sigma(P.seed, P.inst++);
  const int al = a.al;
  switch (a.kind) {
    case V_PLAIN: ltg_run(P.plain, P.st); break;
    case V_CDUO: pv_gemm(P, RCK[0][1], P.mPV, s); break;
    case V_PRODMAX: prod_launch(P, prod<0, 0>, P.Pt, s); break;
    case V_PRODN: prod_launch(P, prod<1, 0>, P.Pt, s); break;
    case V_GENP: genP(P, al, s); break;
    case V_GENV: {
      const uint32_t g = (P.pV + 255) / 256;
      if (al == 0) sha_gen<1><<<g, 256, 0, P.st>>>(P.G, 0, P.pV, s);
      else if (al == 1) sha_gen<2><<<g, 256, 0, P.st>>>(P.G, 0, P.pV, s);
      else sha_gen<3><<<g, 256, 0, P.st>>>(P.G, 0, P.pV, s);
      break;
    }
    case V_EDITV: editV(P, al, s); break;
    case V_SEPS: sep_shared(P, al, s); ltg_run(P.ltx, P.st); break;
    case V_SEPSD: sep_shared(P, al, s); pv_gemm(P, RCK[0][1], P.mXX, s); break;
    case V_FUSS: genP(P, al, s); editV(P, al, s); pv_gemm(P, b1_plain, P.mPX, s); break;
    case V_FOVS: genP(P, al, s); editV(P, al, s); pv_gemm(P, b2_plain, P.mPX, s); break;
    case V_FUSSD: genP(P, al, s); editV(P, al, s); pv_gemm(P, b1_duo32, P.mPX, s); break;
    case V_PRODS: genP(P, al, s); prod_launch(P, prod<1, -1>, P.XA, s); break;
    case V_PRODU: prod_launch(P, PK_U[al], P.XA, s); break;
    case V_PRODSLT: genP(P, al, s); prod_launch(P, prod<1, -1>, P.XA, s); editV(P, al, s); ltg_run(P.ltx, P.st); break;
    case V_PRODULT: prod_launch(P, PK_U[al], P.XA, s); editV(P, al, s); ltg_run(P.ltx, P.st); break;
    case V_PRODSD: genP(P, al, s); prod_launch(P, prod<1, -1>, P.XA, s); editV(P, al, s); pv_gemm(P, RCK[0][1], P.mXX, s); break;
    case V_PRODUD: prod_launch(P, PK_U[al], P.XA, s); editV(P, al, s); pv_gemm(P, RCK[0][1], P.mXX, s); break;
  }
}
static bool varm_ok(const VProb& P, const VArm& a) {
  if (a.kind == V_PLAIN) return P.plain.algo_id != -2;
  if (a.kind == V_SEPS || a.kind == V_PRODSLT || a.kind == V_PRODULT) return P.ltx.algo_id != -2;
  return true;
}

static int vgate(const std::string& dir, int s, int H, uint32_t seed, int grid) {
  VProb P;
  vsetup(P, s, H, seed, grid);
  dump(dir, "P.bin", P.P, P.nP, true); dump(dir, "Vt.bin", P.Vt, P.nV, true);
  dump(dir, "Q.bin", P.Q, P.nQ, true); dump(dir, "K.bin", P.K, P.nQ, true);
  dump(dir, "mx.bin", P.mx, (size_t)H * s * 4, true);
  std::string arms_json, skipped;
  for (const VArm& a : vall()) {
    if (!varm_ok(P, a)) { skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\""; continue; }
    CK(cudaMemset(P.U, 0xA5, P.ub)); CK(cudaMemset(P.R, 0xA5, P.rb)); CK(cudaMemset(P.XA, 0xA5, P.nP));
    CK(cudaMemset(P.XB, 0xA5, P.nV)); CK(cudaMemset(P.E, 0xA5, (size_t)s * s)); CK(cudaMemset(P.G, 0xA5, (size_t)P.pV * 4));
    if (a.kind == V_PRODN) CK(cudaMemset(P.Pt, 0xA5, P.nP));
    P.inst = 0;
    vrun(P, a);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    switch (a.kind) {
      case V_PRODMAX: dump(dir, "mx_prodmax.bin", P.mx, (size_t)H * s * 4, true); break;
      case V_PRODN: dump(dir, "Pt.bin", P.Pt, P.nP, true); break;
      case V_GENP: dump(dir, ("E_" + a.name + ".bin").c_str(), P.E, (size_t)s * s, true); break;
      case V_GENV: dump(dir, ("G_" + a.name + ".bin").c_str(), P.G, (size_t)P.pV * 4, true); break;
      case V_EDITV: dump(dir, ("XB_" + a.name + ".bin").c_str(), P.XB, P.nV, true); break;
      case V_PRODS: case V_PRODU: dump(dir, ("XA_" + a.name + ".bin").c_str(), P.XA, P.nP, true); break;
      default:
        if (a.kind == V_SEPS) dump(dir, ("XA_" + a.name + ".bin").c_str(), P.XA, P.nP, true);
        dump(dir, ("U_" + a.name + ".bin").c_str(), P.U, P.ub, true);
        if (a.kind == V_CDUO || a.kind == V_SEPSD || a.kind == V_FUSSD || a.kind == V_PRODSD || a.kind == V_PRODUD)
          dump(dir, ("R_" + a.name + ".bin").c_str(), P.R, P.rb, true);
    }
    arms_json += (arms_json.empty() ? "\"" : ", \"") + a.name + "\"";
  }
  const Sig sg = sigma(seed, 0);
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"s\": %d, \"H\": %d, \"d\": %d, \"seed\": %u, \"grid\": %d, \"pgrid\": %d, \"c\": %.9g, \"TM\": 128, \"BN\": 128, "
              "\"WM\": 32, \"NCT\": %d, \"sigma\": [\"%016llx\", \"%016llx\", \"%016llx\", \"%016llx\"], \"arms\": [%s], \"skipped\": [%s]}\n",
          s, H, HD, seed, P.grid, P.pgrid, P.c, CNT, (unsigned long long)sg.w[0], (unsigned long long)sg.w[1],
          (unsigned long long)sg.w[2], (unsigned long long)sg.w[3], arms_json.c_str(), skipped.c_str());
  fclose(fp);
  printf("gate dump %s: s %d H %d grid %d pgrid %d\n", dir.c_str(), s, H, P.grid, P.pgrid);
  vfree(P);
  return 0;
}

static int vbench(int reps, const std::vector<int>& ss, int H) {
  Nv nv;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::vector<VArm> all;
  const char* fa = getenv("PV_ARMS");
  for (const VArm& a : vall()) {
    const std::string want = fa ? std::string(",") + fa + "," : "";
    if (!fa || want.find("," + a.name + ",") != std::string::npos) all.push_back(a);
  }
  for (int s : ss) {
    VProb P;
    vsetup(P, s, H, 0x5EED0000u + s + H, 0);
    std::vector<VArm> sel;
    for (const VArm& a : all) if (varm_ok(P, a)) sel.push_back(a);
    const int NA = (int)sel.size();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_arm = [&](const VArm& a, int calls) {
      CK(cudaEventRecord(e0, P.st));
      for (int c = 0; c < calls; c++) vrun(P, a);
      CK(cudaEventRecord(e1, P.st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms / calls;
    };
    std::vector<int> calls(NA);
    for (int j = 0; j < NA; j++) {
      time_arm(sel[j], 1);
      const double t = time_arm(sel[j], 2);
      calls[j] = std::max(2, std::min(400, (int)(20.0 / std::max(t, 1e-4))));
    }
    CK(cudaGetLastError());
    std::vector<std::vector<double>> ms(NA), mhz(NA);
    for (int r = 0; r < reps; r++)
      for (int j = 0; j < NA; j++) {
        const int a = (j + r) % NA;
        ms[a].push_back(time_arm(sel[a], calls[a]));
        mhz[a].push_back(nv.sm());
      }
    printf("{\"shape\": \"pv_s%d\", \"s\": %d, \"H\": %d, \"sms\": %d, \"gpu\": \"%s\", \"grid\": %d, \"pgrid\": %d, \"lt_algo\": [%d, %d], "
           "\"arms\": {", s, s, H, nsm, prop.name, P.grid, P.pgrid, P.plain.algo_id, P.ltx.algo_id);
    for (int j = 0; j < NA; j++) {
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", sel[j].name.c_str(), median(ms[j]),
             median(mhz[j]), calls[j]);
      for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
      printf("]}");
    }
    printf("}}\n");
    fflush(stdout);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    vfree(P);
  }
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate"))
    return vgate(argv[2], atoi(argv[3]), atoi(argv[4]), (uint32_t)strtoul(argv[5], 0, 0), atoi(argv[6]));
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<int> ss;
    for (int i = 3; i < argc; i++) ss.push_back(atoi(argv[i]));
    return vbench(atoi(argv[2]), ss, 32);
  }
  fprintf(stderr, "usage: pvgemm gate DIR S H SEED GRID | bench REPS S...\n");
  return 2;
}
