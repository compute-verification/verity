// Where the e2e slice's edited GEMM spends its time: lpgemm.cu at LP_UW 64 (the slice's build), driven three ways.
//   prof bench REPS SHAPE...       lpgemm's interleaved items (LP_ARMS), each arm hot, back to back
//   prof il REPS ARM M N K         the slice's per-call interleaving: ARM then plain cuBLASLt, alternating which goes first
//   prof one ARM M N K             three warm-up calls, then one call of ARM between cudaProfilerStart/Stop (for ncu)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a prof.cu -lcuda -lcublasLt -lnvidia-ml
#define LP_LIB
#define LP_UW 64
#include "../masked_gemm/lpgemm.cu"

static const LArm& find_arm(const char* name) {
  static std::vector<LArm> all = lall();
  for (const LArm& a : all)
    if (a.name == name) return a;
  fprintf(stderr, "no arm %s\n", name);
  exit(2);
}

static int one(const char* arm, int m, int n, int k) {
  LProb P;
  lsetup(P, m, n, k, 1, 0x5EED0000u + m + n + k + 1, 0, false);
  const LArm& a = find_arm(arm);
  for (int i = 0; i < 3; i++) lrun(P, a);
  CK(cudaDeviceSynchronize());
  CK(cudaProfilerStart());
  lrun(P, a);
  CK(cudaDeviceSynchronize());
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d}\n", arm, m, n, k);
  lfree(P);
  return 0;
}

static int interleaved(int reps, const char* arm, int m, int n, int k) {
  LProb P;
  lsetup(P, m, n, k, 1, 0x5EED0000u + m + n + k + 1, 0, false);
  const LArm& a = find_arm(arm);
  const LArm& pl = find_arm("plain");
  cudaEvent_t e[4];
  for (auto& ev : e) CK(cudaEventCreate(&ev));
  Nv nv;
  for (int i = 0; i < 5; i++) { lrun(P, a); lrun(P, pl); }
  std::vector<double> te, tp, mhz;
  for (int r = 0; r < reps; r++) {
    const bool plain_first = r % 2;
    if (plain_first) { CK(cudaEventRecord(e[2], P.st)); lrun(P, pl); CK(cudaEventRecord(e[3], P.st)); }
    CK(cudaEventRecord(e[0], P.st));
    lrun(P, a);
    CK(cudaEventRecord(e[1], P.st));
    if (!plain_first) { CK(cudaEventRecord(e[2], P.st)); lrun(P, pl); CK(cudaEventRecord(e[3], P.st)); }
    CK(cudaStreamSynchronize(P.st));
    float x, y;
    CK(cudaEventElapsedTime(&x, e[0], e[1]));
    CK(cudaEventElapsedTime(&y, e[2], e[3]));
    te.push_back(x); tp.push_back(y); mhz.push_back(nv.sm());
  }
  CK(cudaGetLastError());
  printf("{\"il\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"reps\": %d, \"edited_ms\": %.6f, \"plain_ms\": %.6f, \"ratio\": %.4f, "
         "\"mhz\": %.0f}\n",
         arm, m, n, k, reps, median(te), median(tp), median(te) / median(tp), median(mhz));
  lfree(P);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> s(argv + 3, argv + argc);
    return lbench(atoi(argv[2]), s);
  }
  if (argc >= 7 && !strcmp(argv[1], "il")) return interleaved(atoi(argv[2]), argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return one(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: prof bench REPS SHAPE... | il REPS ARM M N K | one ARM M N K\n");
  return 2;
}
