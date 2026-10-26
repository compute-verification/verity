/* The cheapest one-fork-point star for 8-variable word sets, mod 2 (see fork_steiner.py): over every alternating
 * form Y on 8 variables (2^28 of them), d(0, Y) + sum_x d(Y, Q_x), with d = rank/2.
 * Cases: one row at L = 2 with k = 3 coin blocks; a 2x2 block of cells at L = 2.
 * cc -O2 -o /tmp/fork_median fork_median.c && /tmp/fork_median */
#include <stdint.h>
#include <stdio.h>

#define NV 8
static int pa[28], pb[28];

static int rank_alt(uint32_t m) {
  uint32_t rows[NV] = {0};
  for (int i = 0; i < 28; i++)
    if (m >> i & 1) { rows[pa[i]] ^= 1u << pb[i]; rows[pb[i]] ^= 1u << pa[i]; }
  int r = 0;
  for (int c = 0; c < NV; c++) {
    int p = -1;
    for (int i = r; i < NV; i++) if (rows[i] >> c & 1) { p = i; break; }
    if (p < 0) continue;
    uint32_t t = rows[r]; rows[r] = rows[p]; rows[p] = t;
    for (int i = 0; i < NV; i++) if (i != r && (rows[i] >> c & 1)) rows[i] ^= rows[r];
    r++;
  }
  return r;
}

static uint32_t term(int a, int b) {
  if (a > b) { int t = a; a = b; b = t; }
  for (int i = 0; i < 28; i++) if (pa[i] == a && pb[i] == b) return 1u << i;
  return 0;
}

static void run(const char *name, uint32_t *q, int k, int chains) {
  int best = 1 << 30; uint32_t arg = 0;
  for (uint64_t y = 0; y < (1ull << 28); y++) {
    int c = rank_alt((uint32_t)y);
    if (c / 2 >= best) continue;
    for (int x = 0; x < k && c / 2 < best; x++) c += rank_alt((uint32_t)y ^ q[x]);
    if (c / 2 < best) { best = c / 2; arg = (uint32_t)y; }
  }
  printf("%s: best one-fork star %d products (Y = %#x); chains %d\n", name, best, arg, chains);
}

int main(void) {
  int n = 0;
  for (int a = 0; a < NV; a++) for (int b = a + 1; b < NV; b++) { pa[n] = a; pb[n] = b; n++; }
  /* one row, L = 2: u = 0,1; coin blocks c^(x) = 2+2x, 3+2x */
  uint32_t row[3];
  for (int x = 0; x < 3; x++) row[x] = term(0, 2 + 2 * x) ^ term(1, 3 + 2 * x);
  run("one row, L=2, k=3", row, 3, 6);
  /* 2x2 cells, L = 2: u_ij = 2i+j (0..3), c_lj = 4+2l+j (4..7), Q_il = sum_j u_ij c_lj */
  uint32_t cell[4]; int m = 0;
  for (int i = 0; i < 2; i++) for (int l = 0; l < 2; l++)
    cell[m++] = term(2 * i, 4 + 2 * l) ^ term(2 * i + 1, 4 + 2 * l + 1);
  run("2x2 cells, L=2", cell, 4, 8);
  return 0;
}
