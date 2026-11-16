// The edited GEMM with its noise expanded from sigma by SHA-512 in counter mode (the
// generator is sha_noise.cuh), at the minimum bits an entry for two
// alphabets: pm1, S = {+1, -1} (1 bit an entry), and u17, uniform {-8..8} (base-17 digits, about 4.6 bits an entry),
// with u17w the same alphabet from 32-bit words (7 digits a word, no 64-bit division), and u17p and u17wp the noise of
// u17 and u17w with a patch's blocks hashed at once on five lanes (sha_patch5); u17w, u17p and u17wp separate passes only.
// Fresh sigma for every instance. int8 operands are kept in [-119, 119] and E4M3's in [-8, 8], so A' and B' can't
// saturate and every sum is exact. gamma_gen is the generation alone; the edit methods:
//   sform: the generation and the edit as one separate pass (one thread a patch of 4 rows x 128 columns)
//   slt / sduo: sform, then cuBLASLt (no receipts) or the custom mainloop with duo at L = 32 on A', B'
//   fus / fusd: generated inside the mainloop (ML's NS), serialized after each stage's TMA (NZ 1), no receipts / duo 32
//   fov / fovd: the same, overlapped: the next stage's generation between the current stage's atoms 0 and 1 (NZ 2)
// The fused kernels are int8 only. Problems batch as in egemm.cu.
//   sgemm gate DIR M N K BATCH SEED GRID   every arm at one small shape, inputs and outputs to DIR for sref.py
//   sgemm bench REPS SHAPE...              interleaved timing, one JSON line a shape (SHAPE = name:m:n:k:batch); SG_ARMS
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a sgemm.cu -lcuda -lcublasLt -lnvidia-ml
#define MG_LIB
#include "mgemm.cu"

#include "sha_pass.cuh"

#define GEMMS(NAME, RS, KIND, NZ, NS)                                                                                   \
  __global__ void __launch_bounds__(ML<128, 128, 32, 3, false, NZ, NS>::NT, 1) NAME(const __grid_constant__ Args a) {  \
    gemm_body<128, 128, 32, 3, false, RS, KIND, false, NZ, NS>(a);                                                      \
  }
GEMMS(s1_plain, 0, NONE, 1, 1)
GEMMS(s1_duo32, 32, DUO, 1, 1)
GEMMS(o1_plain, 0, NONE, 2, 1)
GEMMS(o1_duo32, 32, DUO, 2, 1)
GEMMS(s2_plain, 0, NONE, 1, 2)
GEMMS(s2_duo32, 32, DUO, 1, 2)
GEMMS(o2_plain, 0, NONE, 2, 2)
GEMMS(o2_duo32, 32, DUO, 2, 2)
// [alphabet][overlapped][duo 32]
static Kern SK[2][2][2] = {{{s1_plain, s1_duo32}, {o1_plain, o1_duo32}}, {{s2_plain, s2_duo32}, {o2_plain, o2_duo32}}};
static_assert(ML<128, 128, 32, 3, false, 2, 2>::SMEM == CSMEM, "one shared-memory size");

struct SProb {
  int m, n, k, batch, grid;
  uint32_t pa, pb, KB;
  uint8_t *A[2], *B[2], *XA, *XB;
  int* U;
  uint32_t *R, *G;
  size_t ub, rb;
  LtG plain[2], sep[2];
  Maps *mab[2], *mx;
  uint32_t seed, inst = 0;
  cudaStream_t st;
};

static void ssetup(SProb& P, int m, int n, int k, int batch, uint32_t seed, int grid_cap) {
  P.m = m; P.n = n; P.k = k; P.batch = batch; P.seed = seed;
  if (m % 128 || n % 128 || k % 128) { fprintf(stderr, "shape %d %d %d not tiled\n", m, n, k); exit(2); }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  P.grid = std::min((m / 128) * (n / 128) * batch, grid_cap > 0 ? grid_cap : nsm);
  CK(cudaStreamCreate(&P.st));
  const size_t na = (size_t)batch * m * k, nb = (size_t)batch * n * k;
  if (na / 512 * 8 >= (1ull << 32) || nb / 512 * 8 >= (1ull << 32)) { fprintf(stderr, "operand past the counter\n"); exit(2); }
  P.KB = k / 128; P.pa = (uint32_t)(na / 512); P.pb = (uint32_t)(nb / 512);
  for (int f = 0; f < 2; f++) {
    CK(cudaMalloc(&P.A[f], na)); CK(cudaMalloc(&P.B[f], nb));
    const uint32_t sa = mix32(seed ^ (0x1111u + f)), sb = mix32(seed ^ (0x2222u + f));
    if (f == I8) { sfill<I8><<<4096, 256, 0, P.st>>>(P.A[f], na, sa); sfill<I8><<<4096, 256, 0, P.st>>>(P.B[f], nb, sb); }
    else { sfill<E4><<<4096, 256, 0, P.st>>>(P.A[f], na, sa); sfill<E4><<<4096, 256, 0, P.st>>>(P.B[f], nb, sb); }
  }
  CK(cudaMalloc(&P.XA, na)); CK(cudaMalloc(&P.XB, nb));
  CK(cudaMalloc(&P.G, (size_t)(P.pa + P.pb) * 4));
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
  for (int f = 0; f < 2; f++) CK(cudaFuncSetAttribute(RCK[f][1], cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  for (auto& a : SK)
    for (auto& b : a)
      for (Kern kern : b) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
}
static void sfree(SProb& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.A[0], (void*)P.A[1], (void*)P.B[0], (void*)P.B[1], (void*)P.XA, (void*)P.XB, (void*)P.U, (void*)P.R,
                   (void*)P.G, (void*)P.mab[0], (void*)P.mab[1], (void*)P.mx})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(P.st));
}

enum { S_PLAIN, S_GEN, S_FORM, S_LT, S_DUO, S_FUS, S_FOV };
struct SArm { std::string name; int kind, al, fmt, duo; };
static std::vector<SArm> sall() {
  std::vector<SArm> v;
  for (int f = 0; f < 2; f++) v.push_back({std::string("plain_") + FMTN[f], S_PLAIN, 0, f, 0});
  for (int al = 0; al < NAL; al++) {
    const std::string A = ALN[al];
    v.push_back({"gen_" + A, S_GEN, al, 0, 0});
    for (int f = 0; f < 2; f++) {
      const std::string F = std::string("_") + FMTN[f];
      v.push_back({"sform_" + A + F, S_FORM, al, f, 0});
      v.push_back({"slt_" + A + F, S_LT, al, f, 0});
      v.push_back({"sduo_" + A + F, S_DUO, al, f, 1});
    }
    if (al >= 2) continue;
    v.push_back({"fus_" + A + "_i8", S_FUS, al, 0, 0});
    v.push_back({"fusd_" + A + "_i8", S_FUS, al, 0, 1});
    v.push_back({"fov_" + A + "_i8", S_FOV, al, 0, 0});
    v.push_back({"fovd_" + A + "_i8", S_FOV, al, 0, 1});
  }
  return v;
}
static bool sarm_ok(const SProb& P, const SArm& a) {
  if (a.kind == S_PLAIN) return P.plain[a.fmt].algo_id != -2;
  if (a.kind == S_LT) return P.sep[a.fmt].algo_id != -2;
  return true;
}
static void sform(SProb& P, int al, int f, const Sig& s) {
  if (f == I8) edit_al<false>(al, P.A[f], P.XA, P.B[f], P.XB, P.pa, P.pb, P.KB, s, P.st);
  else edit_al<true>(al, P.A[f], P.XA, P.B[f], P.XB, P.pa, P.pb, P.KB, s, P.st);
}
static void scg(SProb& P, Kern kern, const Maps* maps, const Sig& s) {
  Args a{maps, P.U, P.R, P.m, P.n, P.k, 8, P.seed, 0, P.batch, 0, 0, {s.w[0], s.w[1], s.w[2], s.w[3]}};
  kern<<<P.grid, CNT, CSMEM, P.st>>>(a);
}
static void srun(SProb& P, const SArm& a) {
  const Sig s = sigma(P.seed, P.inst++);
  const int f = a.fmt;
  switch (a.kind) {
    case S_PLAIN: ltg_run(P.plain[f], P.st); break;
    case S_GEN: gen_al(a.al, P.G, P.pa, P.pb, s, P.st); break;
    case S_FORM: sform(P, a.al, f, s); break;
    case S_LT: sform(P, a.al, f, s); ltg_run(P.sep[f], P.st); break;
    case S_DUO: sform(P, a.al, f, s); scg(P, RCK[f][1], P.mx, s); break;
    case S_FUS: scg(P, SK[a.al][0][a.duo], P.mab[I8], s); break;
    case S_FOV: scg(P, SK[a.al][1][a.duo], P.mab[I8], s); break;
  }
}

static int sgate(const std::string& dir, int m, int n, int k, int batch, uint32_t seed, int grid) {
  SProb P;
  ssetup(P, m, n, k, batch, seed, grid);
  const size_t na = (size_t)batch * m * k, nb = (size_t)batch * n * k;
  for (int f = 0; f < 2; f++) {
    dump(dir, (std::string("A_") + FMTN[f] + ".bin").c_str(), P.A[f], na, true);
    dump(dir, (std::string("B_") + FMTN[f] + ".bin").c_str(), P.B[f], nb, true);
  }
  std::string arms_json, skipped;
  for (const SArm& a : sall()) {
    if (!sarm_ok(P, a)) { skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\""; continue; }
    CK(cudaMemset(P.U, 0xA5, P.ub)); CK(cudaMemset(P.R, 0xA5, P.rb));
    CK(cudaMemset(P.XA, 0xA5, na)); CK(cudaMemset(P.XB, 0xA5, nb)); CK(cudaMemset(P.G, 0xA5, (size_t)(P.pa + P.pb) * 4));
    P.inst = 0;
    srun(P, a);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    if (a.kind == S_GEN) {
      dump(dir, ("G_" + a.name + ".bin").c_str(), P.G, (size_t)(P.pa + P.pb) * 4, true);
    } else if (a.kind == S_FORM) {
      dump(dir, ("XA_" + a.name + ".bin").c_str(), P.XA, na, true);
      dump(dir, ("XB_" + a.name + ".bin").c_str(), P.XB, nb, true);
    } else {
      dump(dir, ("U_" + a.name + ".bin").c_str(), P.U, P.ub, true);
      if (a.duo) dump(dir, ("R_" + a.name + ".bin").c_str(), P.R, P.rb, true);
    }
    arms_json += (arms_json.empty() ? "\"" : ", \"") + a.name + "\"";
  }
  const Sig s = sigma(seed, 0);
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"batch\": %d, \"seed\": %u, \"grid\": %d, \"TM\": 128, \"BN\": 128, \"WM\": 32, "
              "\"NCT\": %d, \"sigma\": [\"%016llx\", \"%016llx\", \"%016llx\", \"%016llx\"], \"arms\": [%s], \"skipped\": [%s]}\n",
          m, n, k, batch, seed, P.grid, CNT, (unsigned long long)s.w[0], (unsigned long long)s.w[1], (unsigned long long)s.w[2],
          (unsigned long long)s.w[3], arms_json.c_str(), skipped.c_str());
  fclose(fp);
  printf("gate dump %s: m %d n %d k %d batch %d grid %d\n", dir.c_str(), m, n, k, batch, P.grid);
  sfree(P);
  return 0;
}

static int sbench(int reps, const std::vector<std::string>& shapes) {
  Nv nv;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::vector<SArm> all;
  const char* fa = getenv("SG_ARMS");
  for (const SArm& a : sall()) {
    const std::string want = fa ? std::string(",") + fa + "," : "";
    if (!fa || want.find("," + a.name + ",") != std::string::npos) all.push_back(a);
  }
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k, batch = 1;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d:%d", name, &m, &n, &k, &batch) < 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    SProb P;
    ssetup(P, m, n, k, batch, 0x5EED0000u + m + n + k + batch, 0);
    std::vector<SArm> sel;
    for (const SArm& a : all) if (sarm_ok(P, a)) sel.push_back(a);
    const int NA = (int)sel.size();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_arm = [&](const SArm& a, int calls) {
      CK(cudaEventRecord(e0, P.st));
      for (int c = 0; c < calls; c++) srun(P, a);
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
    std::string skipped;
    for (const SArm& a : all) if (!sarm_ok(P, a)) skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\"";
    printf("{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"batch\": %d, \"sms\": %d, \"gpu\": \"%s\", \"grid\": %d, "
           "\"patches\": [%u, %u], \"skipped\": [%s], \"lt_algo\": [%d, %d], \"arms\": {",
           name, m, n, k, batch, nsm, prop.name, P.grid, P.pa, P.pb, skipped.c_str(), P.plain[0].algo_id, P.plain[1].algo_id);
    for (int j = 0; j < NA; j++) {
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", sel[j].name.c_str(), median(ms[j]),
             median(mhz[j]), calls[j]);
      for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
      printf("]}");
    }
    printf("}}\n");
    fflush(stdout);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    sfree(P);
  }
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 9 && !strcmp(argv[1], "gate"))
    return sgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), (uint32_t)strtoul(argv[7], 0, 0), atoi(argv[8]));
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> s(argv + 3, argv + argc);
    return sbench(atoi(argv[2]), s);
  }
  fprintf(stderr, "usage: sgemm gate DIR M N K BATCH SEED GRID | bench REPS name:m:n:k[:batch]...\n");
  return 2;
}
