// Lane captures on sm_90a: every integer lane form derive_minimal_ssn.py certifies, and the integer bases outside
// Core/Accel's Lane (VIADD, I2IP, PLOP3, the 16x2 and RELU DPX forms, MATCH and VOTE), applied to N input triples on the
// GPU, one kernel a form so its SASS names the opcode ptxas chose. check.py compares each output word with the Python
// semantics the one-instruction-minimal check reads, and a warp form's with its function of the warp's 32 lanes (lanes
// are consecutive indices, every warp full).
//
//   lane_capture OUT.bin [N]     # writes "VLC1", N, the form count, each form's name (NUL-terminated), then a, b, c
//                                # and every form's outputs (two words a form, the second 0 for one-word forms), u32 LE
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "lane_capture: %s: %s\n", #x, cudaGetErrorString(e_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

struct Out2 {
  uint32_t x, y;
};

#define FORMS(F)                                                                                                        \
  F(lop3_96, { asm("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                              \
  F(lop3_e8, { asm("lop3.b32 %0,%1,%2,%3,0xe8;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                              \
  F(lop3_ca, { asm("lop3.b32 %0,%1,%2,%3,0xca;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                              \
  F(lop3_1e, { asm("lop3.b32 %0,%1,%2,%3,0x1e;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                              \
  F(mad_lo, { asm("mad.lo.u32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                                  \
  F(mad_hi, { asm("mad.hi.u32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                                  \
  F(mad_hi_s, { asm("mad.hi.s32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                                \
  F(mad_wide, {                                                                                                         \
    uint64_t w;                                                                                                         \
    asm("mad.wide.u32 %0,%1,%2,%3;" : "=l"(w) : "r"(a), "r"(b), "l"((uint64_t)c << 32 | (a ^ b)));                      \
    o.x = (uint32_t)w;                                                                                                  \
    o.y = (uint32_t)(w >> 32);                                                                                          \
  })                                                                                                                    \
  F(iadd3, { o.x = a + b + c; })                                                                                        \
  F(lea3, { o.x = (a << 3) + b; })                                                                                      \
  F(lea_hi5, { o.x = (uint32_t)((((uint64_t)b << 32 | a) << 5) >> 32) + c; })                                           \
  F(prmt, { asm("prmt.b32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                                      \
  F(shf_l_wrap, { asm("shf.l.wrap.b32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                          \
  F(shf_l_clamp, { asm("shf.l.clamp.b32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                        \
  F(shf_r_wrap, { asm("shf.r.wrap.b32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                          \
  F(shf_r_clamp, { asm("shf.r.clamp.b32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                        \
  F(shl64, {                                                                                                            \
    const uint64_t v = ((uint64_t)b << 32 | a) << (c & 63);                                                             \
    o.x = (uint32_t)v;                                                                                                  \
    o.y = (uint32_t)(v >> 32);                                                                                          \
  })                                                                                                                    \
  F(shr64, {                                                                                                            \
    const uint64_t v = ((uint64_t)b << 32 | a) >> (c & 63);                                                             \
    o.x = (uint32_t)v;                                                                                                  \
    o.y = (uint32_t)(v >> 32);                                                                                          \
  })                                                                                                                    \
  F(sar64, {                                                                                                            \
    const int64_t v = (int64_t)((uint64_t)b << 32 | a) >> (c & 63);                                                     \
    o.x = (uint32_t)v;                                                                                                  \
    o.y = (uint32_t)((uint64_t)v >> 32);                                                                                \
  })                                                                                                                    \
  F(szext_cs, { asm("szext.clamp.s32 %0,%1,%2;" : "=r"(o.x) : "r"(a), "r"(c)); })                                      \
  F(szext_cu, { asm("szext.clamp.u32 %0,%1,%2;" : "=r"(o.x) : "r"(a), "r"(c)); })                                      \
  F(szext_ws, { asm("szext.wrap.s32 %0,%1,%2;" : "=r"(o.x) : "r"(a), "r"(c)); })                                       \
  F(szext_wu, { asm("szext.wrap.u32 %0,%1,%2;" : "=r"(o.x) : "r"(a), "r"(c)); })                                       \
  F(bmsk_c, { asm("bmsk.clamp.b32 %0,%1,%2;" : "=r"(o.x) : "r"(b), "r"(c)); })                                         \
  F(bmsk_w, { asm("bmsk.wrap.b32 %0,%1,%2;" : "=r"(o.x) : "r"(b), "r"(c)); })                                          \
  F(dp4a_ss, { asm("dp4a.s32.s32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                               \
  F(dp4a_uu, { asm("dp4a.u32.u32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                               \
  F(dp4a_su, { asm("dp4a.s32.u32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                               \
  F(dp2a_lo, { asm("dp2a.lo.s32.s32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                            \
  F(dp2a_hi, { asm("dp2a.hi.u32.u32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                            \
  F(vabsdiff4, { o.x = __vabsdiffu4(a, b); })                                                                           \
  F(min_s, { o.x = (uint32_t)min((int)a, (int)b); })                                                                    \
  F(max_u, { o.x = max(a, b); })                                                                                        \
  F(max3_s, { o.x = (uint32_t)__vimax3_s32((int)a, (int)b, (int)c); })                                                  \
  F(addmax_s, { o.x = (uint32_t)__viaddmax_s32((int)a, (int)b, (int)c); })                                              \
  F(abs_s, { o.x = (uint32_t)abs((int)a); })                                                                            \
  F(popc, { o.x = __popc(a); })                                                                                         \
  F(bfind_u, { asm("bfind.u32 %0,%1;" : "=r"(o.x) : "r"(a)); })                                                         \
  F(brev, { o.x = __brev(a); })                                                                                         \
  F(sel, { o.x = (c & 1) ? a : b; })                                                                                    \
  F(viadd_imm, { asm("add.u32 %0,%1,0x9e3779b9;" : "=r"(o.x) : "r"(a)); })                                             \
  F(i2ip_s8, { asm("cvt.pack.sat.s8.s32.b32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                    \
  F(i2ip_u8, { asm("cvt.pack.sat.u8.s32.b32 %0,%1,%2,%3;" : "=r"(o.x) : "r"(a), "r"(b), "r"(c)); })                    \
  F(i2ip_u16, { asm("cvt.pack.sat.u16.s32 %0,%1,%2;" : "=r"(o.x) : "r"(a), "r"(b)); })                                 \
  F(plop3_maj, {                                                                                                        \
    asm("{.reg .pred p,q,r,s,t,u;\n setp.ne.u32 p,%1,0;\n setp.lt.u32 q,%2,%3;\n setp.gt.s32 r,%3,%1;\n"                \
        " and.pred s,p,q;\n and.pred t,q,r;\n and.pred u,p,r;\n or.pred s,s,t;\n or.pred s,s,u;\n selp.u32 %0,1,0,s;}" \
        : "=r"(o.x) : "r"(a), "r"(b), "r"(c));                                                                          \
  })                                                                                                                    \
  F(min_s16x2, { o.x = __vmins2(a, b); })                                                                               \
  F(max_u16x2, { o.x = __vmaxu2(a, b); })                                                                               \
  F(max3_s16x2, { o.x = __vimax3_s16x2(a, b, c); })                                                                     \
  F(addmax_s16x2, { o.x = __viaddmax_s16x2(a, b, c); })                                                                 \
  F(addmax_relu, { o.x = (uint32_t)__viaddmax_s32_relu((int)a, (int)b, (int)c); })                                      \
  F(match_any, { asm("match.any.sync.b32 %0,%1,0xffffffff;" : "=r"(o.x) : "r"(a & 3)); })                              \
  F(match_all, {                                                                                                        \
    asm("{.reg .pred p;\n match.all.sync.b32 %0|p,%1,0xffffffff;}" : "=r"(o.x) : "r"((uint32_t)((c & 15) == 0)));     \
  })                                                                                                                    \
  F(ballot, { o.x = __ballot_sync(0xffffffffu, c & 1); })                                                               \
  F(vote_any, { o.x = __any_sync(0xffffffffu, c & 1); })                                                                \
  F(vote_all, { o.x = __all_sync(0xffffffffu, (c & 7) != 0); })                                                         \
  F(voteu_any, { o.x = __any_sync(0xffffffffu, c & 1) ? a : b; })

#define KERNEL(NAME, BODY)                                                                                   \
  extern "C" __global__ void lc_##NAME(const uint32_t* A, const uint32_t* B, const uint32_t* C, Out2* O, int n) { \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                                                     \
    if (i >= n) return;                                                                                      \
    const uint32_t a = A[i], b = B[i], c = C[i];                                                             \
    Out2 o{0, 0};                                                                                            \
    BODY;                                                                                                    \
    O[i] = o;                                                                                                \
  }
FORMS(KERNEL)
#undef KERNEL

typedef void (*Kern)(const uint32_t*, const uint32_t*, const uint32_t*, Out2*, int);
#define ENTRY(NAME, BODY) {#NAME, lc_##NAME},
static const struct {
  const char* name;
  Kern k;
} TABLE[] = {FORMS(ENTRY)};
#undef ENTRY

static uint64_t sm64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// a word: uniform, or (a quarter each) a small value, an edge (0, 1, 2^31 - 1, 2^31, 2^32 - 1, ±1 of them) or one bit
static uint32_t draw(uint64_t& s) {
  static const uint32_t EDGE[] = {0u, 1u, 2u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu};
  const uint64_t r = sm64(s);
  switch (r & 7) {
    case 0: return (uint32_t)(r >> 32) % 70u;
    case 1: return EDGE[(r >> 32) & 7];
    case 2: return 1u << ((r >> 32) & 31);
    default: return (uint32_t)(r >> 32);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: lane_capture OUT.bin [N]\n");
    return 2;
  }
  const int n = argc > 2 ? atoi(argv[2]) : 1 << 16;
  const int forms = (int)(sizeof TABLE / sizeof TABLE[0]);
  std::vector<uint32_t> a(n), b(n), c(n);
  uint64_t s = 0x5eed1a4e;
  for (int i = 0; i < n; i++) a[i] = draw(s), b[i] = draw(s), c[i] = draw(s);
  uint32_t *dA, *dB, *dC;
  Out2* dO;
  CK(cudaMalloc(&dA, n * 4));
  CK(cudaMalloc(&dB, n * 4));
  CK(cudaMalloc(&dC, n * 4));
  CK(cudaMalloc(&dO, (size_t)n * 8));
  CK(cudaMemcpy(dA, a.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, b.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, c.data(), n * 4, cudaMemcpyHostToDevice));
  FILE* f = fopen(argv[1], "wb");
  if (!f) {
    perror(argv[1]);
    return 1;
  }
  const uint32_t head[3] = {0x31434c56u, (uint32_t)n, (uint32_t)forms};
  fwrite(head, 4, 3, f);
  for (int k = 0; k < forms; k++) fwrite(TABLE[k].name, 1, strlen(TABLE[k].name) + 1, f);
  fwrite(a.data(), 4, n, f);
  fwrite(b.data(), 4, n, f);
  fwrite(c.data(), 4, n, f);
  std::vector<Out2> o(n);
  for (int k = 0; k < forms; k++) {
    CK(cudaMemset(dO, 0xA5, (size_t)n * 8));
    TABLE[k].k<<<(n + 255) / 256, 256>>>(dA, dB, dC, dO, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(o.data(), dO, (size_t)n * 8, cudaMemcpyDeviceToHost));
    fwrite(o.data(), 8, n, f);
  }
  fclose(f);
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("lane_capture: %d forms x %d inputs on %s (sm_%d%d)\n", forms, n, p.name, p.major, p.minor);
  return 0;
}
