// cuBLASLt for the PoUW bench harness: plans, the heuristic's candidates, every configuration its capabilities admit
// (ph_lt_enumerate), and calls.  Host code only.  Part of libpouw_harness.so; built alone (-DPH_LT_ONLY, by g++ against
// another cuBLASLt's headers) it is libpouw_lt2.so, a second cuBLASLt the harness loads beside the first (build.sh, LT2_HOME).
//
// The harness's D (m x n, row-major) = A (m x k, row-major) . B^T, B (n x k, row-major): in cuBLASLt's column-major terms
// D^T = op(B) . A^T with op = T, so cuBLASLt's (m, n, k) is our (n, m, k), its A is our B and its B our A ("TN").
#include <cublasLt.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef PH_LT_ONLY
thread_local char ph_err[512];
extern "C" const char* ph_last_error() { return ph_err; }
#else
extern thread_local char ph_err[512];
#endif

namespace {
int fail_lt(cublasStatus_t s, const char* what) {
  snprintf(ph_err, sizeof ph_err, "%s: cublasStatus %d", what, int(s));
  return 2000 + int(s);
}
int fail_arg(const char* what) {
  snprintf(ph_err, sizeof ph_err, "%s", what);
  return 9;
}
#define LT(x)                                                 \
  do {                                                        \
    cublasStatus_t s_ = (x);                                  \
    if (s_ != CUBLAS_STATUS_SUCCESS) return fail_lt(s_, #x); \
  } while (0)

template <class T>
T config_get(const cublasLtMatmulAlgo_t* a, cublasLtMatmulAlgoConfigAttributes_t attr) {
  T v{};
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(a, attr, &v, sizeof v, &w);
  return v;
}

template <class T>
void config_set(cublasLtMatmulAlgo_t* a, cublasLtMatmulAlgoConfigAttributes_t attr, T v) {
  cublasLtMatmulAlgoConfigSetAttribute(a, attr, &v, sizeof v);
}

template <class T>
std::vector<T> cap_list(const cublasLtMatmulAlgo_t* a, cublasLtMatmulAlgoCapAttributes_t attr) {
  size_t need = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(a, attr, nullptr, 0, &need) != CUBLAS_STATUS_SUCCESS || need < sizeof(T)) return {};
  std::vector<T> v(need / sizeof(T));
  size_t w = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(a, attr, v.data(), v.size() * sizeof(T), &w) != CUBLAS_STATUS_SUCCESS) return {};
  v.resize(w / sizeof(T));
  return v;
}

template <class T>
T cap_get(const cublasLtMatmulAlgo_t* a, cublasLtMatmulAlgoCapAttributes_t attr) {
  T v{};
  size_t w;
  cublasLtMatmulAlgoCapGetAttribute(a, attr, &v, sizeof v, &w);
  return v;
}
}  // namespace

extern "C" {

enum { LT_BF16 = 0, LT_FP8 = 1, LT_FP8_ROWWISE = 2, LT_MXFP8 = 3, LT_FP8_BLK128 = 4, LT_NVFP4 = 5, LT_MXFP4 = 6, LT_INT8 = 7,
       LT_KINDS = 8 };

struct ph_lt_plan {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c, d;
  cublasLtMatmulPreference_t pref;
  int kind, m, n, k;
  size_t ws;
  cublasComputeType_t compute;
  cudaDataType_t scale_t, in_t, out_t;
};

struct ph_lt_candidate {
  cublasLtMatmulAlgo_t algo;
  size_t workspace;
  float waves;
  int algo_id, tile, stages, splitk, reduction, swizzle, custom, inner_shape, cluster_shape;
};

unsigned long long ph_lt_version() { return cublasLtGetVersion(); }
int ph_lt_candidate_size() { return int(sizeof(ph_lt_candidate)); }

static cudaDataType_t lt_in(int kind) {
  switch (kind) {
    case LT_BF16: return CUDA_R_16BF;
    case LT_NVFP4: case LT_MXFP4: return CUDA_R_4F_E2M1;
    case LT_INT8: return CUDA_R_8I;
    default: return CUDA_R_8F_E4M3;
  }
}

// Creates the plan for our (m, n, k), rows `ld` elements apart (ld = k but for the known-bad half-k plan); `ws` bytes of
// workspace are allowed.  Scale modes: per kind (see the enum).
int ph_lt_plan_create_ld(int kind, int m, int n, int k, int ld, size_t ws, int swap_blk, ph_lt_plan** out) {
  if (kind < 0 || kind >= LT_KINDS) return fail_arg("ph_lt_plan_create: unknown kind");
  ph_lt_plan* p = new ph_lt_plan();
  p->kind = kind; p->m = m; p->n = n; p->k = k; p->ws = ws;
  LT(cublasLtCreate(&p->h));
  bool i8 = kind == LT_INT8;
  p->compute = i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F;
  p->scale_t = i8 ? CUDA_R_32I : CUDA_R_32F;
  p->in_t = lt_in(kind);
  p->out_t = i8 ? CUDA_R_32I : CUDA_R_16BF;
  LT(cublasLtMatmulDescCreate(&p->op, p->compute, p->scale_t));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  int32_t sa = -1, sb = -1;
  switch (kind) {
    case LT_FP8_ROWWISE: sa = sb = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F; break;
    case LT_MXFP8: case LT_MXFP4: sa = sb = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0; break;
    case LT_NVFP4: sa = sb = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; break;
    case LT_FP8_BLK128:  // cuBLASLt's A is our weight B: 128x128 blocks on it and 1x128 on our A, or the reverse
      sa = swap_blk ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F : CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
      sb = swap_blk ? CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F : CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
      break;
  }
  if (sa >= 0) {
    LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sa, sizeof sa));
    LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sb, sizeof sb));
  }
  LT(cublasLtMatrixLayoutCreate(&p->a, p->in_t, k, n, ld));     // our B, n x k row-major = k x n column-major, op T
  LT(cublasLtMatrixLayoutCreate(&p->b, p->in_t, k, m, ld));     // our A, m x k row-major = k x m column-major
  LT(cublasLtMatrixLayoutCreate(&p->c, p->out_t, n, m, n));
  LT(cublasLtMatrixLayoutCreate(&p->d, p->out_t, n, m, n));
  LT(cublasLtMatmulPreferenceCreate(&p->pref));
  LT(cublasLtMatmulPreferenceSetAttribute(p->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  *out = p;
  return 0;
}

int ph_lt_plan_create(int kind, int m, int n, int k, size_t ws, int swap_blk, ph_lt_plan** out) {
  return ph_lt_plan_create_ld(kind, m, n, k, k, ws, swap_blk, out);
}

int ph_lt_plan_destroy(ph_lt_plan* p) {
  cublasLtMatmulPreferenceDestroy(p->pref);
  cublasLtMatrixLayoutDestroy(p->a);
  cublasLtMatrixLayoutDestroy(p->b);
  cublasLtMatrixLayoutDestroy(p->c);
  cublasLtMatrixLayoutDestroy(p->d);
  cublasLtMatmulDescDestroy(p->op);
  cublasLtDestroy(p->h);
  delete p;
  return 0;
}

static int set_scales(ph_lt_plan* p, const void* a_scale, const void* b_scale) {
  if (p->kind == LT_BF16 || p->kind == LT_INT8) return 0;
  // cuBLASLt's A is our B: its A scale is our B's
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_scale, sizeof b_scale));
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale, sizeof a_scale));
  return 0;
}

// The attributes' own widths: INNER_SHAPE_ID and CLUSTER_SHAPE_ID are uint16_t, SPLITK_NUM int32_t, the rest uint32_t.
static void describe(ph_lt_candidate& c) {
  c.algo_id = config_get<int32_t>(&c.algo, CUBLASLT_ALGO_CONFIG_ID);
  c.tile = int(config_get<uint32_t>(&c.algo, CUBLASLT_ALGO_CONFIG_TILE_ID));
  c.stages = int(config_get<uint32_t>(&c.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID));
  c.splitk = config_get<int32_t>(&c.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
  c.reduction = int(config_get<uint32_t>(&c.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME));
  c.swizzle = int(config_get<uint32_t>(&c.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING));
  c.custom = int(config_get<uint32_t>(&c.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION));
  c.inner_shape = int(config_get<uint16_t>(&c.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID));
  c.cluster_shape = int(config_get<uint16_t>(&c.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID));
}

// Up to `want` heuristic candidates (cuBLASLt's order), their ids and configs; *got is how many.
int ph_lt_candidates(ph_lt_plan* p, const void* a_scale, const void* b_scale, int want, ph_lt_candidate* out, int* got) {
  if (int r = set_scales(p, a_scale, b_scale)) return r;
  cublasLtMatmulHeuristicResult_t res[64];
  if (want > 64) want = 64;
  *got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(p->h, p->op, p->a, p->b, p->c, p->d, p->pref, want, res, got);
  if (s == CUBLAS_STATUS_NOT_SUPPORTED) { *got = 0; return 0; }
  if (s != CUBLAS_STATUS_SUCCESS) return fail_lt(s, "cublasLtMatmulAlgoGetHeuristic");
  for (int i = 0; i < *got; ++i) {
    ph_lt_candidate& c = out[i];
    memset(&c, 0, sizeof c);
    c.algo = res[i].algo;
    c.workspace = res[i].workspaceSize;
    c.waves = res[i].wavesCount;
    describe(c);
  }
  return 0;
}

// Every configuration of every algorithm cuBLASLt has for the plan's types (cublasLtMatmulAlgoGetIds) that
// cublasLtMatmulAlgoCheck accepts within the plan's workspace.  Per algorithm: each tile and stage count its capabilities
// list, with its default inner and cluster shapes and each pair the heuristic returns for that id; then, for each (tile,
// stages, shapes) accepted at split-k 1 without swizzle or custom option, each split-k in `splitks` (with every reduction
// scheme the algorithm supports), CTA swizzle and custom option.  Up to `want` go to `out`; *got is how many were accepted
// (it may exceed `want`) and *checked how many configurations were tried.
int ph_lt_enumerate(ph_lt_plan* p, const void* a_scale, const void* b_scale, const int* splitks, int nsplitks, int want,
                    ph_lt_candidate* out, int* got, int* checked) {
  if (int r = set_scales(p, a_scale, b_scale)) return r;
  *got = *checked = 0;
  int ids[1024], nids = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetIds(p->h, p->compute, p->scale_t, p->in_t, p->in_t, p->out_t, p->out_t, 1024, ids,
                                              &nids);
  if (s == CUBLAS_STATUS_NOT_SUPPORTED) return 0;
  if (s != CUBLAS_STATUS_SUCCESS) return fail_lt(s, "cublasLtMatmulAlgoGetIds");
  struct shape_pair { int id; uint16_t inner, cluster; };
  std::vector<shape_pair> heur;
  {
    cublasLtMatmulHeuristicResult_t res[64];
    int n = 0;
    if (cublasLtMatmulAlgoGetHeuristic(p->h, p->op, p->a, p->b, p->c, p->d, p->pref, 64, res, &n) == CUBLAS_STATUS_SUCCESS)
      for (int i = 0; i < n; ++i)
        heur.push_back({config_get<int32_t>(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID),
                        config_get<uint16_t>(&res[i].algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
                        config_get<uint16_t>(&res[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)});
  }
  auto check = [&](const cublasLtMatmulAlgo_t& a) {
    ++*checked;
    cublasLtMatmulHeuristicResult_t r;
    memset(&r, 0, sizeof r);
    if (cublasLtMatmulAlgoCheck(p->h, p->op, p->a, p->b, p->c, p->d, &a, &r) != CUBLAS_STATUS_SUCCESS) return false;
    if (r.workspaceSize > p->ws) return false;
    if (*got < want) {
      ph_lt_candidate& c = out[*got];
      memset(&c, 0, sizeof c);
      c.algo = a;
      c.workspace = r.workspaceSize;
      c.waves = r.wavesCount;
      describe(c);
    }
    ++*got;
    return true;
  };
  const uint32_t schemes[] = {CUBLASLT_REDUCTION_SCHEME_INPLACE, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                              CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE};
  for (int i = 0; i < nids; ++i) {
    cublasLtMatmulAlgo_t base;
    if (cublasLtMatmulAlgoInit(p->h, p->compute, p->scale_t, p->in_t, p->in_t, p->out_t, p->out_t, ids[i], &base) !=
        CUBLAS_STATUS_SUCCESS)
      continue;
    std::vector<uint32_t> tiles = cap_list<uint32_t>(&base, CUBLASLT_ALGO_CAP_TILE_IDS);
    std::vector<uint32_t> stages = cap_list<uint32_t>(&base, CUBLASLT_ALGO_CAP_STAGES_IDS);
    if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
    if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
    int32_t splitk_ok = cap_get<int32_t>(&base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);
    uint32_t red_mask = cap_get<uint32_t>(&base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
    uint32_t swizzle_ok = cap_get<uint32_t>(&base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
    int32_t custom_max = cap_get<int32_t>(&base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
    std::vector<shape_pair> shapes = {{ids[i], config_get<uint16_t>(&base, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
                                       config_get<uint16_t>(&base, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)}};
    for (const shape_pair& h : heur) {
      bool seen = h.id != ids[i];
      for (const shape_pair& x : shapes) seen = seen || (x.inner == h.inner && x.cluster == h.cluster);
      if (!seen) shapes.push_back(h);
    }
    for (const shape_pair& sh : shapes)
      for (uint32_t tile : tiles)
        for (uint32_t st : stages) {
          cublasLtMatmulAlgo_t a = base;
          config_set<uint16_t>(&a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, sh.inner);
          config_set<uint16_t>(&a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, sh.cluster);
          config_set<uint32_t>(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, tile);
          config_set<uint32_t>(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, st);
          config_set<int32_t>(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1);
          config_set<uint32_t>(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_REDUCTION_SCHEME_NONE);
          if (!check(a)) continue;
          for (int j = 0; j < nsplitks; ++j) {
            int32_t sk = splitks[j];
            if (sk < 1 || (sk > 1 && !splitk_ok)) continue;
            std::vector<uint32_t> reds;
            if (sk == 1) reds.push_back(CUBLASLT_REDUCTION_SCHEME_NONE);
            else
              for (uint32_t r : schemes)
                if (red_mask & r) reds.push_back(r);
            for (uint32_t red : reds)
              for (uint32_t sw = 0; sw <= (swizzle_ok ? 1u : 0u); ++sw)
                for (int32_t cu = 0; cu <= custom_max; ++cu) {
                  if (sk == 1 && sw == 0 && cu == 0) continue;  // the base configuration, checked above
                  cublasLtMatmulAlgo_t b = a;
                  config_set<int32_t>(&b, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, sk);
                  config_set<uint32_t>(&b, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, red);
                  config_set<uint32_t>(&b, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, sw);
                  config_set<uint32_t>(&b, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, uint32_t(cu));
                  check(b);
                }
          }
        }
  }
  return 0;
}

// D = A . B^T (alpha 1, beta 0) with candidate `c`, on `stream`.  Scale pointers are per operand set.
int ph_lt_run(ph_lt_plan* p, const ph_lt_candidate* c, const void* A, const void* B, void* D, const void* a_scale,
              const void* b_scale, void* ws, void* stream) {
  if (int r = set_scales(p, a_scale, b_scale)) return r;
  float one = 1.0f, zero = 0.0f;
  int32_t ione = 1, izero = 0;
  const void* alpha = p->kind == LT_INT8 ? static_cast<const void*>(&ione) : &one;
  const void* beta = p->kind == LT_INT8 ? static_cast<const void*>(&izero) : &zero;
  LT(cublasLtMatmul(p->h, p->op, alpha, B, p->a, A, p->b, beta, D, p->c, D, p->d, &c->algo, ws, p->ws,
                    cudaStream_t(stream)));
  return 0;
}

}  // extern "C"
