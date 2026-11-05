// The honest cost of the edited GEMM on sm_120a (RTX PRO 6000): both dynamic operands edited with fresh noise from
// S = {5, 7, -5, -7} for every instance, A' = A + E_A and B' = B + E_B, and A' B'^T served, with no peel and no mask
// factors (S is not FP4-representable).
// The noise is counter-based: one 16-byte chunk of a stacked row from nz_hash(key, row, k/16, chunk) (fgemm.cu), the keys
// from the seed and the instance. int8 operands are kept in [-120, 120] and E4M3's in [-9, 9], so A' and B' can't
// saturate and every sum is exact (s32, or f32 below 2^24).
//   sep:  the edit as a separate memory pass (edit_form), then the GEMM on A' and B' (cuBLASLt, or fgemm.cu's mainloop)
//   fus:  the edit fused into the mainloop: each stage edited in shared memory after its TMA lands (ML's NZ), no extra pass
//   fov:  fused, with the next stage's edit interleaved with the current stage's atoms (one CTA barrier a stage either way)
// Receipts on the custom mainloop: duo at L = 32 and 64, TB4C at 32, over all of K. Problems batch: A (batch m x k),
// B (batch n x k), U (batch x m x n), the attention products batched over heads.
//   egemm gate DIR M N K BATCH SEED GRID   every arm at one small shape (GRID CTAs), inputs and outputs to DIR for eref.py
//   egemm bench REPS SHAPE...              interleaved timing, one JSON line a shape (SHAPE = name:m:n:k:batch); EG_ARMS
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a egemm.cu -lcuda -lcublasLt -lnvidia-ml
#define MG_LIB
#include "mgemm.cu"

template <bool F8>
__global__ void __launch_bounds__(256) edit_form(const uint8_t* __restrict__ X, uint8_t* __restrict__ Xp, uint32_t chunks,
                                                 uint32_t key) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= chunks) return;
  const uint4 x = reinterpret_cast<const uint4*>(X)[i];
  const uint32_t h = mix32(key ^ mix32(i));   // nz_hash at (row, c16): row k/16 + c16 is the chunk's index
  reinterpret_cast<uint4*>(Xp)[i] = make_uint4(nz_add<F8>(x.x, h, 0), nz_add<F8>(x.y, h, 1), nz_add<F8>(x.z, h, 2), nz_add<F8>(x.w, h, 3));
}
template <int FMT> __global__ void __launch_bounds__(256) efill(uint8_t* X, size_t count, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i) ^ mix32((uint32_t)(i >> 32) + 0x51u));
    X[i] = (uint8_t)enc<FMT>(FMT == I8 ? (int)(h % 241) - 120 : (int)(h % 19) - 9);
  }
}

#define GEMMZ(NAME, RS, KIND, F8, NZ)                                                                               \
  __global__ void __launch_bounds__(ML<128, 128, 32, 3, F8, NZ>::NT, 1) NAME(const Args a) {                       \
    gemm_body<128, 128, 32, 3, false, RS, KIND, F8, NZ>(a);                                                         \
  }
GEMMZ(zi_plain, 0, NONE, false, 1)
GEMMZ(zi_duo32, 32, DUO, false, 1)
GEMMZ(zi_duo64, 64, DUO, false, 1)
GEMMZ(zi_t4c32, 32, TB4C, false, 1)
GEMMZ(zf_plain, 0, NONE, true, 1)
GEMMZ(zf_duo32, 32, DUO, true, 1)
GEMMZ(zf_duo64, 64, DUO, true, 1)
GEMMZ(zf_t4c32, 32, TB4C, true, 1)
GEMMZ(oi_plain, 0, NONE, false, 2)
GEMMZ(oi_duo32, 32, DUO, false, 2)
GEMMZ(oi_duo64, 64, DUO, false, 2)
GEMMZ(oi_t4c32, 32, TB4C, false, 2)
GEMMZ(of_plain, 0, NONE, true, 2)
GEMMZ(of_duo32, 32, DUO, true, 2)
GEMMZ(of_duo64, 64, DUO, true, 2)
GEMMZ(of_t4c32, 32, TB4C, true, 2)
constexpr int NE = 4;   // cg, cduo32, cduo64, ct4c32: RCK's first four
static Kern ZK[2][NE] = {{zi_plain, zi_duo32, zi_duo64, zi_t4c32}, {zf_plain, zf_duo32, zf_duo64, zf_t4c32}};
static Kern OK_[2][NE] = {{oi_plain, oi_duo32, oi_duo64, oi_t4c32}, {of_plain, of_duo32, of_duo64, of_t4c32}};
static_assert(ML<128, 128, 32, 3, true, 2>::SMEM == CSMEM, "one shared-memory size");

struct EProb {
  int m, n, k, batch, grid;
  uint8_t *A[2], *B[2], *XA, *XB;   // codes per format; the edited operands (one buffer, both formats)
  int* U;
  uint32_t* R;
  size_t ub, rb;
  LtG plain[2], sep[2];
  Maps *mab[2], *mx;
  uint32_t seed, inst = 0;
  cudaStream_t st;
};
// instance i's noise keys, from the seed
static void nz_keys(uint32_t seed, uint32_t inst, uint32_t& ka, uint32_t& kb) {
  ka = mix32(seed ^ mix32(2 * inst + 0x0A0A0A0Au));
  kb = mix32(seed ^ mix32(2 * inst + 1 + 0x0A0A0A0Au));
}

static void esetup(EProb& P, int m, int n, int k, int batch, uint32_t seed, int grid_cap) {
  P.m = m; P.n = n; P.k = k; P.batch = batch; P.seed = seed;
  if (m % 128 || n % 128 || k % 128) { fprintf(stderr, "shape %d %d %d not tiled\n", m, n, k); exit(2); }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  P.grid = std::min((m / 128) * (n / 128) * batch, grid_cap > 0 ? grid_cap : nsm);
  CK(cudaStreamCreate(&P.st));
  const size_t na = (size_t)batch * m * k, nb = (size_t)batch * n * k;
  if (na / 16 >= (1ull << 32) || nb / 16 >= (1ull << 32)) { fprintf(stderr, "operand past 2^36 bytes\n"); exit(2); }
  for (int f = 0; f < 2; f++) {
    CK(cudaMalloc(&P.A[f], na)); CK(cudaMalloc(&P.B[f], nb));
    const uint32_t sa = mix32(seed ^ (0x1111u + f)), sb = mix32(seed ^ (0x2222u + f));
    if (f == I8) { efill<I8><<<4096, 256, 0, P.st>>>(P.A[f], na, sa); efill<I8><<<4096, 256, 0, P.st>>>(P.B[f], nb, sb); }
    else { efill<E4><<<4096, 256, 0, P.st>>>(P.A[f], na, sa); efill<E4><<<4096, 256, 0, P.st>>>(P.B[f], nb, sb); }
  }
  CK(cudaMalloc(&P.XA, na)); CK(cudaMalloc(&P.XB, nb));
  P.ub = (size_t)batch * m * n * 4;
  P.rb = (size_t)(m / 128) * (n / 128) * batch * CNT * 8 * 4;
  CK(cudaMalloc(&P.U, P.ub)); CK(cudaMalloc(&P.R, P.rb));
  CK(cudaStreamSynchronize(P.st));
  for (int f = 0; f < 2; f++) {
    P.plain[f] = ltg(f, m, n, k, k, k, n, P.A[f], P.B[f], P.U, 0, P.st, batch);
    P.sep[f] = ltg(f, m, n, k, k, k, n, P.XA, P.XB, P.U, 0, P.st, batch);
    Maps mp{tmap(P.A[f], k, batch * m, 128), tmap(P.B[f], k, batch * n, 128), tmap(P.A[f], k, batch * m, 128), tmap(P.B[f], k, batch * n, 128)};
    P.mab[f] = upload_maps(mp);
  }
  Maps mx{tmap(P.XA, k, batch * m, 128), tmap(P.XB, k, batch * n, 128), tmap(P.XA, k, batch * m, 128), tmap(P.XB, k, batch * n, 128)};
  P.mx = upload_maps(mx);
  for (int f = 0; f < 2; f++)
    for (int j = 0; j < NE; j++) {
      CK(cudaFuncSetAttribute(RCK[f][j], cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
      CK(cudaFuncSetAttribute(ZK[f][j], cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
      CK(cudaFuncSetAttribute(OK_[f][j], cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
    }
}
static void efree(EProb& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.A[0], (void*)P.A[1], (void*)P.B[0], (void*)P.B[1], (void*)P.XA, (void*)P.XB, (void*)P.U, (void*)P.R,
                   (void*)P.mab[0], (void*)P.mab[1], (void*)P.mx})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(P.st));
}

// an arm: plain (cuBLASLt on A, B), sepform (the edit pass alone), sep_lt (the edit pass + cuBLASLt on A', B'); on the
// custom mainloop <rc> (A, B, no edit), sep_<rc> (the edit pass + the mainloop on A', B'), fus_<rc> (the fused edit,
// serialized), fov_<rc> (the fused edit overlapped with the previous stage's atoms)
enum { E_PLAIN, E_SEPFORM, E_SEPLT, E_C, E_SEPC, E_FUS, E_FOV };
struct EArm { std::string name; int fmt, kind, rc; };
static std::vector<EArm> eall() {
  std::vector<EArm> v;
  for (int f = 0; f < 2; f++) {
    const std::string F = std::string("_") + FMTN[f];
    v.push_back({"plain" + F, f, E_PLAIN, 0});
    v.push_back({"sepform" + F, f, E_SEPFORM, 0});
    v.push_back({"sep_lt" + F, f, E_SEPLT, 0});
    for (int j = 0; j < NE; j++) {
      v.push_back({std::string(RCN[j]) + F, f, E_C, j});
      v.push_back({"sep_" + std::string(RCN[j]) + F, f, E_SEPC, j});
      v.push_back({"fus_" + std::string(RCN[j]) + F, f, E_FUS, j});
      v.push_back({"fov_" + std::string(RCN[j]) + F, f, E_FOV, j});
    }
  }
  return v;
}
static bool earm_ok(const EProb& P, const EArm& a) {
  if (a.kind == E_PLAIN) return P.plain[a.fmt].algo_id != -2;
  if (a.kind == E_SEPLT) return P.sep[a.fmt].algo_id != -2;
  return true;
}
static void eform(EProb& P, int f, uint32_t ka, uint32_t kb) {
  const uint32_t ca = (uint32_t)((size_t)P.batch * P.m * P.k / 16), cb = (uint32_t)((size_t)P.batch * P.n * P.k / 16);
  if (f == I8) {
    edit_form<false><<<(ca + 255) / 256, 256, 0, P.st>>>(P.A[f], P.XA, ca, ka);
    edit_form<false><<<(cb + 255) / 256, 256, 0, P.st>>>(P.B[f], P.XB, cb, kb);
  } else {
    edit_form<true><<<(ca + 255) / 256, 256, 0, P.st>>>(P.A[f], P.XA, ca, ka);
    edit_form<true><<<(cb + 255) / 256, 256, 0, P.st>>>(P.B[f], P.XB, cb, kb);
  }
}
static void ecg(EProb& P, Kern kern, const Maps* maps, uint32_t ka, uint32_t kb) {
  Args a{maps, P.U, P.R, P.m, P.n, P.k, 8, P.seed, 0, P.batch, ka, kb};
  kern<<<P.grid, CNT, CSMEM, P.st>>>(a);
}
static void erun(EProb& P, const EArm& a) {
  uint32_t ka, kb;
  nz_keys(P.seed, P.inst++, ka, kb);
  const int f = a.fmt;
  switch (a.kind) {
    case E_PLAIN: ltg_run(P.plain[f], P.st); break;
    case E_SEPFORM: eform(P, f, ka, kb); break;
    case E_SEPLT: eform(P, f, ka, kb); ltg_run(P.sep[f], P.st); break;
    case E_C: ecg(P, RCK[f][a.rc], P.mab[f], 0, 0); break;
    case E_SEPC: eform(P, f, ka, kb); ecg(P, RCK[f][a.rc], P.mx, 0, 0); break;
    case E_FUS: ecg(P, ZK[f][a.rc], P.mab[f], ka, kb); break;
    case E_FOV: ecg(P, OK_[f][a.rc], P.mab[f], ka, kb); break;
  }
}

static int egate(const std::string& dir, int m, int n, int k, int batch, uint32_t seed, int grid) {
  EProb P;
  esetup(P, m, n, k, batch, seed, grid);
  const size_t na = (size_t)batch * m * k, nb = (size_t)batch * n * k;
  for (int f = 0; f < 2; f++) {
    dump(dir, (std::string("A_") + FMTN[f] + ".bin").c_str(), P.A[f], na, true);
    dump(dir, (std::string("B_") + FMTN[f] + ".bin").c_str(), P.B[f], nb, true);
  }
  std::string arms_json, skipped;
  for (const EArm& a : eall()) {
    if (!earm_ok(P, a)) { skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\""; continue; }
    CK(cudaMemset(P.U, 0xA5, P.ub)); CK(cudaMemset(P.R, 0xA5, P.rb));
    CK(cudaMemset(P.XA, 0xA5, na)); CK(cudaMemset(P.XB, 0xA5, nb));
    P.inst = 0;
    erun(P, a);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    if (a.kind == E_SEPFORM) {
      dump(dir, ("XA_" + a.name + ".bin").c_str(), P.XA, na, true);
      dump(dir, ("XB_" + a.name + ".bin").c_str(), P.XB, nb, true);
    } else {
      dump(dir, ("U_" + a.name + ".bin").c_str(), P.U, P.ub, true);
      if ((a.kind == E_C || a.kind == E_SEPC || a.kind == E_FUS || a.kind == E_FOV) && a.rc != 0)
        dump(dir, ("R_" + a.name + ".bin").c_str(), P.R, P.rb, true);
    }
    arms_json += (arms_json.empty() ? "\"" : ", \"") + a.name + "\"";
  }
  uint32_t ka, kb;
  nz_keys(seed, 0, ka, kb);
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"batch\": %d, \"seed\": %u, \"grid\": %d, \"TM\": 128, \"BN\": 128, \"WM\": 32, "
              "\"NCT\": %d, \"nzA\": %u, \"nzB\": %u, \"arms\": [%s], \"skipped\": [%s]}\n",
          m, n, k, batch, seed, P.grid, CNT, ka, kb, arms_json.c_str(), skipped.c_str());
  fclose(fp);
  printf("gate dump %s: m %d n %d k %d batch %d grid %d\n", dir.c_str(), m, n, k, batch, P.grid);
  efree(P);
  return 0;
}

static int ebench(int reps, const std::vector<std::string>& shapes) {
  Nv nv;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::vector<EArm> all;
  const char* fa = getenv("EG_ARMS");
  for (const EArm& a : eall()) {
    const std::string want = fa ? std::string(",") + fa + "," : "";
    if (!fa || want.find("," + a.name + ",") != std::string::npos) all.push_back(a);
  }
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k, batch = 1;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d:%d", name, &m, &n, &k, &batch) < 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    EProb P;
    esetup(P, m, n, k, batch, 0x5EED0000u + m + n + k + batch, 0);
    std::vector<EArm> sel;
    for (const EArm& a : all) if (earm_ok(P, a)) sel.push_back(a);
    const int NS = (int)sel.size();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_arm = [&](const EArm& a, int calls) {
      CK(cudaEventRecord(e0, P.st));
      for (int c = 0; c < calls; c++) erun(P, a);
      CK(cudaEventRecord(e1, P.st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms / calls;
    };
    std::vector<int> calls(NS);
    for (int j = 0; j < NS; j++) {
      time_arm(sel[j], 2);
      const double t = time_arm(sel[j], 3);
      calls[j] = std::max(2, std::min(400, (int)(20.0 / std::max(t, 1e-4))));
    }
    CK(cudaGetLastError());
    std::vector<std::vector<double>> ms(NS), mhz(NS);
    for (int r = 0; r < reps; r++)
      for (int j = 0; j < NS; j++) {
        const int a = (j + r) % NS;
        ms[a].push_back(time_arm(sel[a], calls[a]));
        mhz[a].push_back(nv.sm());
      }
    std::string skipped;
    for (const EArm& a : all) if (!earm_ok(P, a)) skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\"";
    printf("{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"batch\": %d, \"sms\": %d, \"gpu\": \"%s\", \"grid\": %d, "
           "\"skipped\": [%s], \"lt_algo\": [%d, %d], \"arms\": {",
           name, m, n, k, batch, nsm, prop.name, P.grid, skipped.c_str(), P.plain[0].algo_id, P.plain[1].algo_id);
    for (int j = 0; j < NS; j++) {
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", sel[j].name.c_str(), median(ms[j]),
             median(mhz[j]), calls[j]);
      for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
      printf("]}");
    }
    printf("}}\n");
    fflush(stdout);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    efree(P);
  }
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 9 && !strcmp(argv[1], "gate"))
    return egate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), (uint32_t)strtoul(argv[7], 0, 0), atoi(argv[8]));
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> s(argv + 3, argv + argc);
    return ebench(atoi(argv[2]), s);
  }
  fprintf(stderr, "usage: egemm gate DIR M N K BATCH SEED GRID | bench REPS name:m:n:k[:batch]...\n");
  return 2;
}
