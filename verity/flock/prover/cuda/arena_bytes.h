// The circuit prover's device arena (flock_cuda_prove_circuit): the most bytes its buffers hold at once in any phase, at the
// sizes it allocates them, plus FC_ARENA_MARGIN. Host C, so a test builds it without CUDA: tests/arena_layout.py replays every
// allocation a session makes through DeviceArena's placement, and tests/test_arena_layout.py holds this to every shape.
#pragma once
#include <stddef.h>

// covers the buffers under 64 MiB, which no phase below counts, and the holes the placement leaves
#define FC_ARENA_MARGIN ((size_t)128 << 20)

typedef struct {
    int m, k_log, sub_log;
    int initial_k, recursive_steps;  // Ligerito's ladder: level 0's 2^initial_k lanes, then its recursive levels
    const int* log_inv_rates;        // [recursive_steps + 1]
    const int* recursive_ks;         // [recursive_steps]
    const int* queries;              // [recursive_steps + 1]: level 0's is M1's own count under M1
    int hash_bytes;                  // a Merkle node
    int zk;                          // M1: level 0's hiding
    long long t_pad;                 // M1: level 0's padding words per lane
    long long fold_words;            // the lincheck fold's columns, over every slot type
} FcArenaShape;

static inline size_t fca_max(size_t a, size_t b) { return a > b ? a : b; }
static inline size_t fca_al(size_t b) { return (b + 255) & ~(size_t)255; }

static inline size_t fc_arena_bytes(const FcArenaShape* s) {
    const size_t F = 16;
    const int log_n = s->m - 7, ik = s->initial_k;
    const size_t W = F << log_n, K = F << s->k_log, w = (size_t)1 << ik;
    // level 0's codeword and tree, and under M1 its pads and 4 extra lanes: live from its commitment through Ligerito
    const size_t n = (size_t)1 << (log_n - ik + s->log_inv_rates[0]), lanes = s->zk ? w + 4 : w;
    const size_t l0 = fca_al(n * lanes * F) + fca_al((2 * n - 1) * s->hash_bytes) +
                      (s->zk ? fca_al(lanes * (size_t)s->t_pad * F) + fca_al((F << (log_n - ik)) * 4) : 0);
    // the commitment, beside z, a and b: under M1 the extra lanes' codewords, and a quarter of the witness's lane codewords
    // where they aren't transformed in place (FC_NTT_REGS=0)
    const size_t commit = 3 * W + l0 + (s->zk ? fca_al(n * (w >> (ik < 2 ? ik : 2)) * F) + fca_al(n * 4 * F) : 0);
    // the zerocheck (zerocheck_fused.cuh's packed path past 2^10 output words a block): its eq tables and two fold pairs
    const long long zc_out = 1LL << (s->m - 6);
    const int packed = log_n >= 3 && zc_out / 16 > (1LL << 10);
    const size_t zc_first = F * (packed ? zc_out / 8 : zc_out), zc_next = F * (packed ? zc_out / 4 : zc_out / 2);
    const int lob = log_n > 7 ? log_n - 7 : 0;
    const size_t zerocheck = 3 * W + l0 + fca_al(F << (s->m - 13)) + 2 * fca_al(zc_first) + 2 * fca_al(zc_next) +
                             fca_al(F << lob) + fca_al(F << (log_n - lob));
    // the lincheck, beside z and its transpose: five K-length vectors, then the fold's types or z's initial fold. A K wider
    // than the witness doesn't fit the witness-sized hole a and b leave beside the transpose
    const size_t lincheck = 2 * W + l0 + 5 * fca_al(K) + fca_al(F << s->sub_log) + fca_al(F << (s->m - s->k_log)) +
                            fca_max(fca_al(K), fca_al(F * (size_t)s->fold_words)) + (K > W ? W : 0);
    // Ligerito, beside z and the ring switch's combination: its two ping-pong tables, the induce basis over the largest
    // opened block and its weights and positions over the most queries, the level-OOD eq table, at each level's query
    // phase that level's codeword and tree (level 0's are l0), the next level's, and the opened rows, and under 64-byte nodes
    // (hm96-sha512) the recursive levels' salts, copied from level 0's first fold on and each held until its tree is built
    size_t max_block = n, opening = 0, prev = 0, rows = fca_al((size_t)s->queries[0] * lanes * F), max_q = 0, max_al = 1;
    size_t salts = 0;
    for (int i = 0; i <= s->recursive_steps; i++) max_q = fca_max(max_q, (size_t)s->queries[i]);
    while (max_al < max_q) max_al <<= 1;
    int dim = log_n - ik + 1;
    for (int i = 0; i < s->recursive_steps; i++) {
        const int log_cols = dim - s->recursive_ks[i];
        const size_t bl = (size_t)1 << (log_cols + s->log_inv_rates[i + 1]), cols = (size_t)1 << s->recursive_ks[i];
        const size_t level = fca_al(bl * cols * F) + fca_al((2 * bl - 1) * s->hash_bytes);
        opening = fca_max(opening, prev + level + rows);
        max_block = fca_max(max_block, bl);
        if (s->hash_bytes == 64) salts += fca_al(bl * 192);
        prev = level;
        rows = fca_al((size_t)s->queries[i + 1] * cols * F);
        dim = log_cols + 1;
    }
    opening = fca_max(opening, prev + rows);
    const size_t ligerito = 4 * W + l0 + fca_al(max_block * F) + fca_al(max_al * F) + fca_al(max_q * 8) +
                            fca_al(F << (log_n - ik + 1)) + opening + salts;
    return fca_max(fca_max(commit, zerocheck), fca_max(lincheck, ligerito)) + FC_ARENA_MARGIN;
}
