// NCP route U on the host's AMX-INT8: C = c0 + (X + beta) . Y, u8 x s8 -> int32 with wrap (mod 2^32), the accumulator
// snapshotted every KS deep (KS = 16: NCP's whole checkpoint stream; 64, 128: k-slice binding; 0: the final A.B only).
//
//   gate:  ncp_amx gate DIR KS                  reads DIR/{shape,x,y,c0}.bin, writes DIR/out_KS.bin (every snapshot, then final)
//   gateform: ncp_amx gateform DIR KS          the same with Y formed from DIR/{b,f1,pinv,negq}.bin (see form_group)
//   bench: ncp_amx bench M K N W KS REPS        W weights of K x N, OpenMP threads; one JSON line (NCP_FORM=1: Y formed)
//
// X is m x K u8 (row-major), Y is given as N columns of K s8 (NCP's `ops.y`), c0 is N int32.  Y is packed once into AMX's
// VNNI layout, a per-epoch cost (F is per epoch and weight).  X is packed per call into 16 x kstep tiles laid end to end, so
// an A tile is one contiguous load (row-major X with a stride of K = 3k bytes, a multiple of 4 KiB, puts all 16 rows of a
// tile in one L1 set).  m a multiple of 32 runs the 2 x 2 form (four C tiles, each A and B tile feeding two products), a
// 32-column half of a 64-column group at a time over every row pair, so the half's Y (32 x K) stays in L2.
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "big_alloc.h"

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

typedef struct {
  uint8_t palette_id, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __attribute__((packed)) tilecfg;

static void cfg_tiles(int kstep) {
  tilecfg c;
  memset(&c, 0, sizeof c);
  c.palette_id = 1;
  for (int t = 0; t < 4; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
  for (int t = 4; t < 6; t++) { c.rows[t] = 16; c.colsb[t] = (uint16_t)kstep; }
  for (int t = 6; t < 8; t++) { c.rows[t] = (uint8_t)(kstep / 4); c.colsb[t] = 64; }
  _tile_loadconfig(&c);
}

// Yp layout: [group g of 64 columns][k-block kb][col block cb (4)][kstep/4 quads][16 cols][4 bytes]
static size_t yp_index(int K, int kstep, int g, int kb, int cb) {
  return ((size_t)g * (K / kstep) + kb) * 4 * (size_t)kstep * 16 + (size_t)cb * kstep * 16;
}

static void pack_y(const int8_t *y, int8_t *yp, int K, int N, int kstep) {
  for (int g = 0; g < N / 64; g++)
    for (int kb = 0; kb < K / kstep; kb++)
      for (int cb = 0; cb < 4; cb++) {
        int8_t *dst = yp + yp_index(K, kstep, g, kb, cb);
        for (int q = 0; q < kstep / 4; q++)
          for (int c = 0; c < 16; c++)
            for (int b = 0; b < 4; b++)
              dst[(q * 16 + c) * 4 + b] = y[(size_t)(g * 64 + cb * 16 + c) * K + kb * kstep + 4 * q + b];
      }
}

// Xp layout: [row block rb of 16][k-block kb][16 rows][kstep bytes]
static void pack_x(const uint8_t *x, uint8_t *xp, int M, int K, int kstep) {
  for (int rb = 0; rb < M / 16; rb++)
    for (int kb = 0; kb < K / kstep; kb++)
      for (int r = 0; r < 16; r++)
        memcpy(xp + (((size_t)rb * (K / kstep) + kb) * 16 + r) * kstep, x + (size_t)(rb * 16 + r) * K + kb * kstep, kstep);
}

static void store4(int32_t *d, int l) {
  _tile_stored(0, d + 0, l * 4);
  _tile_stored(1, d + 16, l * 4);
  _tile_stored(2, d + 32, l * 4);
  _tile_stored(3, d + 48, l * 4);
}

static void store22(int32_t *d, int l) {
  _tile_stored(0, d, l * 4);
  _tile_stored(1, d + 16, l * 4);
  _tile_stored(2, d + 16 * l, l * 4);
  _tile_stored(3, d + 16 * l + 16, l * 4);
}

// 1 x 4: one 16-row block x one 64-column group over the whole K.  Snapshots after depth KS*(s+1) land at out[s] (row
// stride ld) in the gate, in a 4 KiB scratch in the bench (the z a receipt binds is written, not kept).
static void block14(const uint8_t *xp, int K, const int8_t *ypg, const int32_t *c0rows, int kstep, int ks, int32_t *out,
                    size_t snap_stride, int ld, int32_t *scratch) {
  _tile_loadd(0, c0rows + 0, 64 * 4);
  _tile_loadd(1, c0rows + 16, 64 * 4);
  _tile_loadd(2, c0rows + 32, 64 * 4);
  _tile_loadd(3, c0rows + 48, 64 * 4);
  int per = ks ? ks / kstep : 0, s = 0, nkb = K / kstep;
  size_t bstride = 4 * (size_t)kstep * 16, tstride = (size_t)kstep * 16, astride = 16 * (size_t)kstep;
  for (int kb = 0; kb < nkb; kb++) {
    const int8_t *b = ypg + kb * bstride;
    _tile_loadd(4, xp + kb * astride, kstep);
    _tile_loadd(6, b, 64);
    _tile_dpbusd(0, 4, 6);
    _tile_loadd(7, b + tstride, 64);
    _tile_dpbusd(1, 4, 7);
    _tile_loadd(6, b + 2 * tstride, 64);
    _tile_dpbusd(2, 4, 6);
    _tile_loadd(7, b + 3 * tstride, 64);
    _tile_dpbusd(3, 4, 7);
    if (per && (kb + 1) % per == 0 && kb + 1 < nkb) {
      store4(out ? out + s * snap_stride : scratch, out ? ld : 64);
      s++;
    }
  }
  store4(out ? out + s * snap_stride : scratch, out ? ld : 64);
}

// 2 x 2: rows {r0, r0+16} x column blocks {cb, cb+1} of a 64-column group.
static void block22(const uint8_t *xp0, const uint8_t *xp1, int K, const int8_t *ypg, const int32_t *c0rows, int kstep,
                    int ks, int cb, int32_t *out, size_t snap_stride, int ld, int32_t *scratch) {
  _tile_loadd(0, c0rows + cb * 16, 64 * 4);
  _tile_loadd(1, c0rows + cb * 16 + 16, 64 * 4);
  _tile_loadd(2, c0rows + cb * 16, 64 * 4);
  _tile_loadd(3, c0rows + cb * 16 + 16, 64 * 4);
  int per = ks ? ks / kstep : 0, s = 0, nkb = K / kstep;
  size_t bstride = 4 * (size_t)kstep * 16, tstride = (size_t)kstep * 16, astride = 16 * (size_t)kstep;
  for (int kb = 0; kb < nkb; kb++) {
    const int8_t *b = ypg + kb * bstride + cb * tstride;
    _tile_loadd(4, xp0 + kb * astride, kstep);
    _tile_loadd(6, b, 64);
    _tile_loadd(5, xp1 + kb * astride, kstep);
    _tile_loadd(7, b + tstride, 64);
    _tile_dpbusd(0, 4, 6);
    _tile_dpbusd(1, 4, 7);
    _tile_dpbusd(2, 5, 6);
    _tile_dpbusd(3, 5, 7);
    if (per && (kb + 1) % per == 0 && kb + 1 < nkb) {
      store22(out ? out + s * snap_stride + cb * 16 : scratch, out ? ld : 64);
      s++;
    }
  }
  store22(out ? out + s * snap_stride + cb * 16 : scratch, out ? ld : 64);
}

static void c0_rows(const int32_t *c0, int g, int32_t *rows) {
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 64; c++) rows[r * 64 + c] = c0[g * 64 + c];
}

// One 64-column group of one weight against all M rows (xp packed).  out/snap_stride/ld as in block14 (gate), else scratch.
static void group(const uint8_t *xp, int M, int K, const int8_t *ypg, const int32_t *c0, int g, int kstep, int ks,
                  int32_t *out, size_t snap_stride, int ld, int32_t *scratch, int32_t *rows) {
  c0_rows(c0, g, rows);
  size_t xrb = (size_t)16 * K;
  if (M % 32 == 0)
    for (int cb = 0; cb < 4; cb += 2)
      for (int rp = 0; rp < M / 32; rp++)
        block22(xp + 2 * rp * xrb, xp + (2 * rp + 1) * xrb, K, ypg, rows, kstep, ks, cb,
                out ? out + (size_t)rp * 32 * ld + g * 64 : NULL, snap_stride, ld, scratch);
  else
    for (int rb = 0; rb < M / 16; rb++)
      block14(xp + rb * xrb, K, ypg, rows, kstep, ks, out ? out + (size_t)rb * 16 * ld + g * 64 : NULL, snap_stride, ld,
              scratch);
}

// Formed Y: B and F₁ stored once (2 bytes a weight byte) and the group's Y = [B + F₁; B + F₂; B + F₃] formed per call into
// an L2 panel in Yp's group layout.  F₂ = −R·F₁ with R a word-granular signed permutation: F₂'s quad Q (4 entries along k)
// is quad pinv[Q] of F₁ with the lanes in negq[Q] negated (those whose sign is +1), and F₃ = F₁ + F₂.  Bq, Fq layout:
// [group g][col block cb (4)][quad Q (k/4)][16 cols][4 bytes].  The entries stay in s8 (B int7, F₁ in [-31, 32]).
static void pack_q(const int8_t *cols, int8_t *dst, int k, int n) {
  int nq = k / 4;
  for (int g = 0; g < n / 64; g++)
    for (int cb = 0; cb < 4; cb++)
      for (int q = 0; q < nq; q++)
        for (int c = 0; c < 16; c++)
          for (int b = 0; b < 4; b++)
            dst[((((size_t)g * 4 + cb) * nq + q) * 16 + c) * 4 + b] = cols[(size_t)(g * 64 + cb * 16 + c) * k + 4 * q + b];
}

// Y's quad L of column block cb sits in the panel at k-block L / (kstep/4), quad L % (kstep/4) (Yp's group layout); the
// three sections are nq = k/4 quads apart, a whole number of k-blocks.
static void form_group(const int8_t *bq, const int8_t *fq, const int32_t *pinv, const uint64_t *negq, int k, int kstep,
                       int8_t *panel) {
  const int nq = k / 4, qs = kstep / 4, nkb = nq / qs;
  const size_t kbs = 4 * (size_t)kstep * 16, sec = (size_t)nkb * kbs;
  const __m512i zero = _mm512_setzero_si512();
  for (int cb = 0; cb < 4; cb++) {
    const int8_t *b = bq + (size_t)cb * nq * 64, *f = fq + (size_t)cb * nq * 64;
    int8_t *p1 = panel + (size_t)cb * kstep * 16;
    for (int kb = 0; kb < nkb; kb++, p1 += kbs)
      for (int q = 0; q < qs; q++) {
        int L = kb * qs + q;
        _mm512_store_si512(p1 + 64 * q, _mm512_add_epi8(_mm512_load_si512(b + 64 * L), _mm512_load_si512(f + 64 * L)));
      }
    p1 = panel + (size_t)cb * kstep * 16;
    for (int kb = 0; kb < nkb; kb++, p1 += kbs)
      for (int q = 0; q < qs; q++) {
        int L = kb * qs + q;
        __m512i src = _mm512_load_si512(f + 64 * (size_t)pinv[L]);
        __m512i f2 = _mm512_mask_sub_epi8(src, (__mmask64)negq[L], zero, src);
        _mm512_store_si512(p1 + sec + 64 * q, _mm512_add_epi8(_mm512_load_si512(b + 64 * L), f2));
        _mm512_store_si512(p1 + 2 * sec + 64 * q, _mm512_add_epi8(_mm512_load_si512(p1 + 64 * q), f2));
      }
  }
}

static void *rd(const char *dir, const char *name, size_t *len) {
  char p[4096];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END);
  *len = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void *b = malloc(*len);
  if (fread(b, 1, *len, f) != *len) exit(2);
  fclose(f);
  return b;
}

static int kstep_of(int ks) { return ks && ks < 64 ? ks : 64; }

static int gate(const char *dir, int ks) {
  size_t len;
  int32_t *shape = rd(dir, "shape.bin", &len);
  int M = shape[0], K = shape[1], N = shape[2], kstep = kstep_of(ks);
  uint8_t *x = rd(dir, "x.bin", &len);
  int8_t *y = rd(dir, "y.bin", &len);
  int32_t *c0 = rd(dir, "c0.bin", &len);
  if (M % 16 || N % 64 || K % kstep || (ks && ks % kstep) || (ks && K % ks)) { fprintf(stderr, "shape\n"); return 2; }
  int8_t *yp = aligned_alloc(64, (size_t)K * N);
  uint8_t *xp = aligned_alloc(64, (size_t)M * K);
  pack_y(y, yp, K, N, kstep);
  pack_x(x, xp, M, K, kstep);
  int nsnap = ks ? K / ks : 1;
  int32_t *out = calloc((size_t)nsnap * M * N, 4);
  cfg_tiles(kstep);
  int32_t rows[16 * 64] __attribute__((aligned(64)));
  for (int g = 0; g < N / 64; g++)
    group(xp, M, K, yp + yp_index(K, kstep, g, 0, 0), c0, g, kstep, ks, out, (size_t)M * N, N, NULL, rows);
  _tile_release();
  char p[4096];
  snprintf(p, sizeof p, "%s/out_%d.bin", dir, ks);
  FILE *f = fopen(p, "wb");
  fwrite(out, 4, (size_t)nsnap * M * N, f);
  fclose(f);
  printf("{\"gate\": \"wrote\", \"ks\": %d, \"snapshots\": %d, \"form\": \"%s\"}\n", ks, nsnap, M % 32 ? "1x4" : "2x2");
  return 0;
}

// gateform: as gate, with Y formed per group from DIR/{b,f1}.bin (n columns of k s8), pinv.bin (k/4 int32), negq.bin
// (k/4 uint64) in place of y.bin.
static int gateform(const char *dir, int ks) {
  size_t len;
  int32_t *shape = rd(dir, "shape.bin", &len);
  int M = shape[0], K = shape[1], N = shape[2], k = K / 3, kstep = kstep_of(ks);
  uint8_t *x = rd(dir, "x.bin", &len);
  int8_t *b = rd(dir, "b.bin", &len), *f1 = rd(dir, "f1.bin", &len);
  int32_t *pinv = rd(dir, "pinv.bin", &len), *c0 = rd(dir, "c0.bin", &len);
  uint64_t *negq = rd(dir, "negq.bin", &len);
  if (M % 16 || N % 64 || k % kstep || (ks && ks % kstep) || (ks && K % ks)) { fprintf(stderr, "shape\n"); return 2; }
  int8_t *bq = aligned_alloc(64, (size_t)k * N), *fq = aligned_alloc(64, (size_t)k * N);
  int8_t *panel = aligned_alloc(64, (size_t)K * 64);
  uint8_t *xp = aligned_alloc(64, (size_t)M * K);
  pack_q(b, bq, k, N);
  pack_q(f1, fq, k, N);
  pack_x(x, xp, M, K, kstep);
  int nsnap = ks ? K / ks : 1;
  int32_t *out = calloc((size_t)nsnap * M * N, 4);
  cfg_tiles(kstep);
  int32_t rows[16 * 64] __attribute__((aligned(64)));
  for (int g = 0; g < N / 64; g++) {
    form_group(bq + (size_t)g * k * 64, fq + (size_t)g * k * 64, pinv, negq, k, kstep, panel);
    group(xp, M, K, panel, c0, g, kstep, ks, out, (size_t)M * N, N, NULL, rows);
  }
  _tile_release();
  char p[4096];
  snprintf(p, sizeof p, "%s/out_%d.bin", dir, ks);
  FILE *f = fopen(p, "wb");
  fwrite(out, 4, (size_t)nsnap * M * N, f);
  fclose(f);
  printf("{\"gate\": \"wrote\", \"ks\": %d, \"snapshots\": %d, \"form\": \"%s\", \"y\": \"formed\"}\n", ks, nsnap,
         M % 32 ? "1x4" : "2x2");
  return 0;
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

// NCP_REPEAT=R: W/R distinct weights, each used by R consecutive work items (a thread's own L2-resident weight).
static int bench(int M, int K, int N, int W, int ks, int reps) {
  int kstep = kstep_of(ks), R = getenv("NCP_REPEAT") ? atoi(getenv("NCP_REPEAT")) : 1;
  int form = getenv("NCP_FORM") && atoi(getenv("NCP_FORM")), k = K / 3, nq = k / 4;
  size_t wbytes = (size_t)K * N, distinct = (size_t)(W + R - 1) / R, stored = form ? 2 * wbytes / 3 : wbytes;
  int8_t *yp = big_alloc(stored * distinct);  // formed: [B̃ q | F̃₁ q] per weight
  int32_t *pinv = malloc(nq * sizeof *pinv);
  uint64_t *negq = malloc(nq * sizeof *negq);
  for (int q = 0; q < nq; q++) { pinv[q] = q; negq[q] = ~0ull; }
  for (int q = nq - 1; q > 0; q--) {  // a fixed shuffle: F₂'s quads gather from all over F₁'s column block
    int r = (int)(((uint64_t)q * 2654435761u + 12345) % (uint64_t)(q + 1)), t = pinv[q];
    pinv[q] = pinv[r]; pinv[r] = t;
  }
  uint8_t *x = aligned_alloc(64, (size_t)M * K);
  int32_t *c0 = aligned_alloc(64, (size_t)N * 4);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < stored * distinct; i++) yp[i] = (int8_t)(((i * 2654435761u) >> 13) & 0x3F) - 31;
  for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (uint8_t)((i * 40503u) >> 7);
  for (int j = 0; j < N; j++) c0[j] = -128 * j;
  int groups = N / 64, nthreads = omp_get_max_threads();
  double best = 1e30, total = 0;
  uint64_t fold = 0;
  // X is packed once per thread, untimed: per call it is m x 3k bytes against m x 3k x n MACs (about 2% at n = 4096)
  uint8_t **xps = calloc(nthreads, sizeof *xps);
#pragma omp parallel
  {
    uint8_t *xp = aligned_alloc(64, (size_t)M * K);
    pack_x(x, xp, M, K, kstep);
    xps[omp_get_thread_num()] = xp;
  }
  for (int rep = 0; rep < reps + 1; rep++) {
    double t0 = now();
#pragma omp parallel reduction(^ : fold)
    {
      cfg_tiles(kstep);
      int32_t rows[16 * 64] __attribute__((aligned(64)));
      int32_t scratch[32 * 64] __attribute__((aligned(64)));
      const uint8_t *xp = xps[omp_get_thread_num()];
      int8_t *panel = form ? aligned_alloc(64, (size_t)K * 64) : NULL;
#pragma omp single
      nthreads = omp_get_num_threads();
#pragma omp for schedule(static)
      for (int wg = 0; wg < W * groups; wg++) {
        int w = wg / groups, g = wg % groups;
        if (form) {
          const int8_t *wq = yp + (w / R) * stored;
          form_group(wq + (size_t)g * k * 64, wq + stored / 2 + (size_t)g * k * 64, pinv, negq, k, kstep, panel);
          group(xp, M, K, panel, c0, g, kstep, ks, NULL, 0, 0, scratch, rows);
        } else
          group(xp, M, K, yp + (w / R) * wbytes + yp_index(K, kstep, g, 0, 0), c0, g, kstep, ks, NULL, 0, 0, scratch, rows);
        fold ^= (uint64_t)(uint32_t)scratch[w & 1023];
      }
      free(panel);
      _tile_release();
    }
    double dt = now() - t0;
    if (rep > 0) { total += dt; if (dt < best) best = dt; }
  }
  double macs = (double)M * K * N * W, mean = total / reps;
  // NCP's K is 3k: a weight of k x n bytes is K/3 x N, so weight bytes covered = W * K * N / 3 (= macs / 3m).
  printf("{\"kernel\": \"ncp_amx\", \"form\": \"%s\", \"y\": \"%s\", \"stored_bytes_per_weight_byte\": %d, \"m\": %d, \"K\": %d, \"n\": %d, \"weights\": %d, \"ks\": %d, "
         "\"repeat\": %d, \"kstep\": %d, \"thp\": %d, \"threads\": %d, \"reps\": %d, \"best_s\": %.6f, \"mean_s\": %.6f, \"mac_per_s\": %.4e, "
         "\"y_bytes_per_s\": %.4e, \"weight_bytes_per_s\": %.4e, \"weight_bytes_per_s_mean\": %.4e, \"fold\": %llu}\n",
         M % 32 ? "1x4" : "2x2", form ? "formed" : "stored", form ? 2 : 3, M, K, N, W, ks, R, kstep, thp_on(), nthreads, reps, best, mean, macs / best, (double)wbytes * W / best,
         (double)wbytes * W / 3 / best, (double)wbytes * W / 3 / mean, (unsigned long long)fold);
  return 0;
}

int main(int argc, char **argv) {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) { perror("arch_prctl(XTILEDATA)"); return 3; }
  if (argc >= 4 && !strcmp(argv[1], "gate")) return gate(argv[2], atoi(argv[3]));
  if (argc >= 4 && !strcmp(argv[1], "gateform")) return gateform(argv[2], atoi(argv[3]));
  if (argc >= 8 && !strcmp(argv[1], "bench"))
    return bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
  fprintf(stderr, "usage: ncp_amx gate|gateform DIR KS | bench M K N W KS REPS\n");
  return 2;
}
