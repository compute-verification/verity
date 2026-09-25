// verity flock-gpu-link: Flock-CUDA prover for the chunk-block statement (backends/flock/live/src/chunk.rs).
// Included at the end of cuda-ghash/prove_ffi.cu (after backends/flock/cuda_live_patch.py), so it reuses the TU's
// statics: the device arena, the twiddle cache, the challenger, FfiWriter and the ring-switch kernels.
//
// Differences from flock_cuda_prove_blake3:
// - the compressions' inputs come from the host (chained chunks, dummy chunks), one compression per 2^14 sub-block;
// - the block is 2^k_log = 16 sub-blocks, so z_lincheck is transposed for k_log and the lincheck fold is the
//   compression circuit's (2^14 columns) scaled per sub-block, plus the sparse delta rows;
// - link_cb runs after the commitment and before the statement binding: it sends Commit / Link and returns the extra
//   claims' coordinates (m - 6 each, past the skip block);
// - the ring-switch batch opens ab, c and every extra claim.

#include <chrono>

#define CKI(x) do { int e_ = (x); if (e_) return e_; } while (0)

__global__ void chunk_zlin_transpose(const F128* __restrict__ z, int k_log, long long n_stripes, uint8_t* __restrict__ out) {
    const long long wpr = (1LL << k_log) / 128;   // F128 words per block row
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n_stripes * wpr;
         t += (long long)gridDim.x * blockDim.x) {
        long long s = t / wpr, w = t % wpr;
        F128 v[8];
#pragma unroll
        for (int r = 0; r < 8; r++) v[r] = z[(8 * s + r) * wpr + w];
        uint8_t* o = out + (s << k_log) + w * 128;
        for (int b = 0; b < 128; b++) {
            uint8_t byte = 0;
#pragma unroll
            for (int r = 0; r < 8; r++) {
                u64 word = b < 64 ? v[r].lo : v[r].hi;
                byte |= (uint8_t)(((word >> (b & 63)) & 1) << r);
            }
            o[b] = byte;
        }
    }
}

__global__ void chunk_comb_expand(const F128* __restrict__ comb_sub, const F128* __restrict__ ratio, int sub_log,
                                  long long k, F128* __restrict__ comb) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    comb[i] = ghash_mul_karatsuba(ratio[i >> sub_log], comb_sub[i & ((1LL << sub_log) - 1)]);
}

__global__ void chunk_eq_gather(const F128* __restrict__ eq, int sub_log, int subs, F128* __restrict__ out) {
    int j = threadIdx.x;
    if (j < subs) out[j] = eq[(long long)j << sub_log];
}

// comb[c] += coef * eq_inner[i] for every delta pair (i, c); coef = alpha for A, one for B.
__global__ void chunk_delta_scatter(const uint32_t* __restrict__ pairs, int n, const F128* __restrict__ eq,
                                    F128 coef, int use_coef, F128* comb) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t i = pairs[2 * t], c = pairs[2 * t + 1];
    F128 v = use_coef ? ghash_mul_karatsuba(coef, eq[i]) : eq[i];
    atomicXor((unsigned long long*)&comb[c].lo, (unsigned long long)v.lo);
    atomicXor((unsigned long long*)&comb[c].hi, (unsigned long long)v.hi);
}

// Sparse extra claims: a coordinate of exactly 0 or 1 makes its eq factor an indicator, so a claim whose point fixes
// f index bits is supported on len / 2^f packed words. idx(k) deposits k's bits into the free positions over `fixed`.
struct SparseBits { int n_free; int free[48]; unsigned long long fixed; };

__device__ __forceinline__ long long sparse_index(const SparseBits& s, long long k) {
    unsigned long long i = s.fixed;
    for (int t = 0; t < s.n_free; t++) i |= (unsigned long long)((k >> t) & 1) << s.free[t];
    return (long long)i;
}

__global__ void chunk_sparse_gather(const F128* __restrict__ z, SparseBits s, long long n, F128* __restrict__ out) {
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += (long long)gridDim.x * blockDim.x)
        out[k] = z[sparse_index(s, k)];
}

__global__ void chunk_sparse_combine(const F128* __restrict__ e, const F128* __restrict__ weights, SparseBits s, long long n,
                                     F128* __restrict__ combined) {
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += (long long)gridDim.x * blockDim.x) {
        long long i = sparse_index(s, k);
        F128 acc = combined[i];
        F128 v = e[k];
        for (int half = 0; half < 2; half++) {
            u64 bits = half ? v.hi : v.lo;
            while (bits) {
                int bit = __ffsll((long long)bits) - 1;
                F128 w = weights[64 * half + bit];
                acc.lo ^= w.lo;
                acc.hi ^= w.hi;
                bits &= bits - 1;
            }
        }
        combined[i] = acc;
    }
}

__global__ void chunk_combine_accum(const F128* __restrict__ suffix, const F128* __restrict__ weights, long long len,
                                    F128* __restrict__ combined) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    F128 acc = combined[i];
    F128 s = suffix[i];
    for (int half = 0; half < 2; half++) {
        u64 bits = half ? s.hi : s.lo;
        while (bits) {
            int bit = __ffsll((long long)bits) - 1;
            F128 w = weights[64 * half + bit];
            acc.lo ^= w.lo;
            acc.hi ^= w.hi;
            bits &= bits - 1;
        }
    }
    combined[i] = acc;
}

// Copy compression ci's 2^14-bit block (128 words) to block ci / slots, slot ci % slots of the 2^k_log layout.
__global__ void pure_comp_scatter(const F128* __restrict__ src, long long n_comp, int slots, int k_log, F128* __restrict__ dst) {
    const long long total = n_comp * 128;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
        long long ci = t >> 7, w = t & 127;
        long long blk = ci / slots, s = ci % slots;
        dst[(blk << (k_log - 7)) + s * 128 + w] = src[t];
    }
}

// One thread per unit: evaluate the lowered rows (inputs, then rows in order; the constant is 1), then write z and the
// honest a = A z, b = B z of its 2^unit_log bits. Rows past `useful` are zero.
#define PURE_UW 256
__global__ void pure_unit_witness(const uint64_t* __restrict__ inputs, long long n_units, int units, int unit_log, int unit_pos0,
                                  int k_log, const uint32_t* __restrict__ a_off, const uint32_t* __restrict__ a_col,
                                  const uint32_t* __restrict__ b_off, const uint32_t* __restrict__ b_col, int useful, int cpos, int n_in,
                                  F128* __restrict__ z_out, F128* __restrict__ a_out, F128* __restrict__ b_out) {
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_units) return;
    uint32_t zw[PURE_UW];
    for (int i = 0; i < PURE_UW; i++) zw[i] = 0;
    const uint64_t* in = inputs + id * 10;
    for (int i = 0; i < n_in; i++) {
        uint64_t word = in[2 * (i >> 7) + ((i & 127) >= 64)];
        zw[i >> 5] |= (uint32_t)((word >> (i & 63)) & 1) << (i & 31);
    }
    zw[cpos >> 5] |= 1u << (cpos & 31);
    auto bit = [&](uint32_t c) -> uint32_t { return (zw[c >> 5] >> (c & 31)) & 1; };
    for (int i = n_in; i < useful; i++) {
        if (i == cpos) continue;
        uint32_t x = 0, y = 0;
        for (uint32_t k = a_off[i]; k < a_off[i + 1]; k++) x ^= bit(a_col[k]);
        for (uint32_t k = b_off[i]; k < b_off[i + 1]; k++) y ^= bit(b_col[k]);
        zw[i >> 5] |= (x & y) << (i & 31);
    }
    long long blk = id / units, u = id % units;
    long long at = (blk << (k_log - 7)) + ((long long)(unit_pos0 + u) << (unit_log - 7));
    const int words = 1 << (unit_log - 7);
    for (int w = 0; w < words; w++) {
        u64 zl = 0, zh = 0, al = 0, ah = 0, bl = 0, bh = 0;
        for (int j = 0; j < 128; j++) {
            int i = 128 * w + j;
            if (i >= useful) break;
            uint32_t zi = bit(i), ai, bi;
            if (i < n_in || i == cpos) { ai = zi; bi = zi; }
            else {
                ai = 0; bi = 0;
                for (uint32_t k = a_off[i]; k < a_off[i + 1]; k++) ai ^= bit(a_col[k]);
                for (uint32_t k = b_off[i]; k < b_off[i + 1]; k++) bi ^= bit(b_col[k]);
            }
            if (j < 64) { zl |= (u64)zi << j; al |= (u64)ai << j; bl |= (u64)bi << j; }
            else { zh |= (u64)zi << (j - 64); ah |= (u64)ai << (j - 64); bh |= (u64)bi << (j - 64); }
        }
        z_out[at + w] = F128{zl, zh}; a_out[at + w] = F128{al, ah}; b_out[at + w] = F128{bl, bh};
    }
}

__device__ __forceinline__ uint32_t pb_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
#define PB_G(a, b, c, d, mx, my) \
    s[a] = s[a] + s[b] + (mx); s[d] = pb_rotr(s[d] ^ s[a], 16); s[c] = s[c] + s[d]; s[b] = pb_rotr(s[b] ^ s[c], 12); \
    s[a] = s[a] + s[b] + (my); s[d] = pb_rotr(s[d] ^ s[a], 8); s[c] = s[c] + s[d]; s[b] = pb_rotr(s[b] ^ s[c], 7);
__device__ void pb_compress(const uint32_t* cv, const uint32_t* m_in, uint64_t ctr, uint32_t blen, uint32_t flags, uint32_t* out) {
    const uint32_t IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    const int P[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    uint32_t s[16], m[16], t[16];
    for (int i = 0; i < 8; i++) { s[i] = cv[i]; }
    for (int i = 0; i < 4; i++) s[8 + i] = IV[i];
    s[12] = (uint32_t)ctr; s[13] = (uint32_t)(ctr >> 32); s[14] = blen; s[15] = flags;
    for (int i = 0; i < 16; i++) m[i] = m_in[i];
    for (int r = 0; r < 7; r++) {
        PB_G(0, 4, 8, 12, m[0], m[1]) PB_G(1, 5, 9, 13, m[2], m[3]) PB_G(2, 6, 10, 14, m[4], m[5]) PB_G(3, 7, 11, 15, m[6], m[7])
        PB_G(0, 5, 10, 15, m[8], m[9]) PB_G(1, 6, 11, 12, m[10], m[11]) PB_G(2, 7, 8, 13, m[12], m[13]) PB_G(3, 4, 9, 14, m[14], m[15])
        for (int i = 0; i < 16; i++) t[i] = m[P[i]];
        for (int i = 0; i < 16; i++) m[i] = t[i];
    }
    for (int i = 0; i < 8; i++) out[i] = s[i] ^ s[i + 8];
}

// One thread per (block, role): the chunk's 16 compressions (chained from the role key, KEYED_HASH, counter = chunk index,
// block_len 64, CHUNK_START / CHUNK_END) written as compression inputs, and the chunk's output chaining value.
__global__ void pure_chain_inputs(const uint8_t* __restrict__ rx, const uint8_t* __restrict__ rw, int row_bytes, int blocks_real,
                                  long long n_blocks, const uint32_t* __restrict__ keys, int comp_slots, int per_role, int cpb, int nb0,
                                  int nb1, int bpv, int dummy_keyed, uint32_t* cvs, uint32_t* msgs, uint64_t* ctrs, uint32_t* blens,
                                  uint32_t* flagss, uint32_t* cvout) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 2 * n_blocks) return;
    long long b = t >> 1; int role = t & 1;
    bool real = b < blocks_real;
    long long v = b / bpv;
    int slot = role * per_role;
    for (int c = 0; c < cpb; c++) {
        int nb = c == 0 ? nb0 : nb1;
        int chunk = (int)(b % bpv) * cpb + c;
        bool keyed = real || dummy_keyed;
        uint32_t cv[8];
        for (int i = 0; i < 8; i++) cv[i] = keyed ? keys[8 * role + i] : 0;
        uint64_t ctr = real ? (uint64_t)chunk : (dummy_keyed ? (uint64_t)c : 0);
        const uint8_t* row = (role ? rw : rx) + v * row_bytes + chunk * 1024;
        for (int j = 0; j < nb; j++, slot++) {
            long long ci = b * comp_slots + slot;
            uint32_t m[16];
            for (int i = 0; i < 16; i++) m[i] = real ? *(const uint32_t*)(row + 64 * j + 4 * i) : 0;
            uint32_t fl = 16u | (j == 0 ? 1u : 0u) | (j + 1 == nb ? 2u : 0u);
            for (int i = 0; i < 8; i++) cvs[ci * 8 + i] = cv[i];
            for (int i = 0; i < 16; i++) msgs[ci * 16 + i] = m[i];
            ctrs[ci] = ctr; blens[ci] = 64; flagss[ci] = fl;
            uint32_t o[8];
            pb_compress(cv, m, ctr, 64, fl, o);
            for (int i = 0; i < 8; i++) cv[i] = o[i];
        }
        for (int i = 0; i < 8; i++) cvout[((b * 2 + role) * cpb + c) * 8 + i] = cv[i];
    }
}

// One thread per unit: its x and W words (the message words its compressions hash) and its accumulator input (the
// recorded accumulator before it; +0 for the first unit of a VU and for dummy blocks), as 5 F128 (lo, hi).
__global__ void pure_unit_inputs(const uint32_t* __restrict__ msgs, const uint32_t* __restrict__ accs, int units_per_vu,
                                 int blocks_real, long long n_blocks, int units, int comp_slots, int per_role, int bpv, uint64_t* out) {
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_blocks * units) return;
    long long b = id / units; int u = (int)(id % units);
    uint64_t* o = out + id * 10;
    for (int role = 0; role < 2; role++) {
        const uint32_t* m = msgs + (b * comp_slots + role * per_role + u / 2) * 16 + (u % 2) * 8;
        for (int w = 0; w < 2; w++) {
            o[4 * role + 2 * w] = (uint64_t)m[4 * w] | ((uint64_t)m[4 * w + 1] << 32);
            o[4 * role + 2 * w + 1] = (uint64_t)m[4 * w + 2] | ((uint64_t)m[4 * w + 3] << 32);
        }
    }
    uint32_t c_in = 0;
    if (b < blocks_real) {
        long long v = b / bpv; int j = (int)(b % bpv) * units + u;
        if (j > 0) c_in = accs[v * units_per_vu + j - 1];
    }
    o[8] = c_in; o[9] = 0;
}

// The unit region of every block (2^unit_log-bit positions unit_pos0 .. unit_pos0 + units), to or from a cache.
__global__ void pure_unit_region_copy(F128* __restrict__ w, F128* __restrict__ cache, long long n_blocks, int k_log, int unit_log,
                                      int unit_pos0, int units, int to_cache) {
    const long long per = (long long)units << (unit_log - 7);
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n_blocks * per; t += (long long)gridDim.x * blockDim.x) {
        long long b = t / per, k = t % per;
        long long at = (b << (k_log - 7)) + ((long long)unit_pos0 << (unit_log - 7)) + k;
        if (to_cache) cache[t] = w[at]; else w[at] = cache[t];
    }
}

// Unit 31's output words of every real block, read from the witness.
__global__ void pure_unit_outputs(const F128* __restrict__ z, int blocks_real, int k_log, int unit_log, int unit_pos, int col_acc,
                                  int col_y, uint32_t* acc, uint32_t* y16) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks_real) return;
    long long at = ((long long)b << (k_log - 7)) + ((long long)unit_pos << (unit_log - 7));
    acc[b] = (uint32_t)z[at + col_acc].lo;
    y16[b] = col_y >= 0 ? (uint32_t)(z[at + col_y].lo & 0xFFFF) : 0;
}

__global__ void pure_comb_expand(const F128* __restrict__ base_c, const F128* __restrict__ ratio_c, int slots,
                                 const F128* __restrict__ base_u, const F128* __restrict__ ratio_u, int units, int unit_log, int unit_pos0,
                                 long long k, F128* __restrict__ comb) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    long long s = i >> 14, p = i >> unit_log;
    F128 v{0, 0};
    if (s < slots) v = ghash_mul_karatsuba(ratio_c[s], base_c[i & 16383]);
    else if (p >= unit_pos0 && p < unit_pos0 + units) v = ghash_mul_karatsuba(ratio_u[p - unit_pos0], base_u[i & ((1LL << unit_log) - 1)]);
    comb[i] = v;
}

__global__ void pure_eq_gather(const F128* __restrict__ eq, int shift, int off, int n, F128* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) out[j] = eq[(long long)(off + j) << shift];
}

extern "C" {

// Returns the number of extra claims written to x_out (each m - 6 F128, lo then hi), or < 0 to abort.
typedef int (*ChunkLinkCb)(void* ctx, const uint8_t* cap, size_t cap_len, int m, uint64_t* x_out, int max_claims);

typedef struct {
    int n_comp_log;                 // log2 of the compression count (m - 14)
    const uint32_t* cv;             // [n_comp][8]
    const uint32_t* msg;            // [n_comp][16]
    const uint64_t* ctr;            // [n_comp]
    const uint32_t* blen;           // [n_comp]
    const uint32_t* flags;          // [n_comp]
    const uint32_t* delta_a; int n_delta_a;   // (row, col) pairs in block coordinates
    const uint32_t* delta_b; int n_delta_b;
    int sub_log;                    // 14: the compression circuit's columns (the CSC in P is that circuit's)
    ChunkLinkCb link_cb; void* link_ctx;
    int gamma_bits;                 // claim-batch grinding for 2 + n_extra claims (0 = none)
    double* phase_s;                // optional [8]: witness, commit, link, zerocheck, lincheck, ring switch, ligerito, total
    // mode 1 (verity/flock-pure-block/v1): per block `comp_slots` compressions at 2^14 slots 0.., then `units` unit
    // sub-blocks (2^unit_log bits) at 2^unit_log positions unit_pos0.., the rest zero.
    int mode;
    int comp_slots, units, unit_log, unit_pos0;
    const uint32_t* u_a_col_ptr; const uint32_t* u_a_rows; uint32_t u_a_nnz;   // unit circuit CSC (2^unit_log columns)
    const uint32_t* u_b_col_ptr; const uint32_t* u_b_rows; uint32_t u_b_nnz;
    const uint32_t* u_a_off; const uint32_t* u_a_col; const uint32_t* u_b_off; const uint32_t* u_b_col;   // CSR rows
    int u_useful, u_const, u_n_in;
    const uint64_t* u_inputs;       // [blocks * units][5 F128] (lo, hi); NULL with rows_x (built on the device)
    // mode 1, device inputs: the operand rows and recorded accumulators; the compressions and unit inputs are built on
    // the device, and the Commit publics are read back into pub_* before link_cb.
    const uint8_t* rows_x; const uint8_t* rows_w; const uint32_t* accs; int vus; int row_bytes; int units_per_vu;
    const uint32_t* keys;           // [2][8] role keys
    int blocks_real;
    int unit_cache;                 // 0 off, 1 store the unit region after witnessing, 2 load it instead of witnessing
    uint32_t* pub_cv; uint32_t* pub_y16; uint32_t* pub_acc;   // [blocks_real][2][8], [blocks_real], [blocks_real]
    int u_out_acc, u_out_y;         // unit output word columns (c_out, y16; -1 when absent)
    // the block layout (flock_live::pure_block::Layout): compressions per role, chunks per block and their sizes, blocks
    // per VU; dummy_keyed: dummy blocks keep the role keys and chunk counters (fp8, where they are per-slot constants)
    int per_role, chunks_per_block, chunk_nb0, chunk_nb1, blocks_per_vu, dummy_keyed;
} FlockChunkParams;

int flock_cuda_prove_chunk(const FlockCudaProveParams* P, const FlockChunkParams* C, uint8_t** out, size_t* out_len) {
    static std::mutex chunk_mutex;
    std::lock_guard<std::mutex> chunk_lock(chunk_mutex);
    auto t_start = std::chrono::steady_clock::now();
    auto t_ph = t_start;
    auto phase = [&](int i) -> int {
        CK(cudaStreamSynchronize(0));
        auto t = std::chrono::steady_clock::now();
        if (C->phase_s) C->phase_s[i] = std::chrono::duration<double>(t - t_ph).count();
        t_ph = t;
        return 0;
    };
    const int m = P->m, k_log = P->k_log, k_skip = 6, sub_log = C->sub_log;
    const int log_n = m - 7;
    const long long len = 1LL << log_n;
    const long long n_comp = C->mode == 1 ? (1LL << (m - k_log)) * C->comp_slots : 1LL << C->n_comp_log;
    const bool pure = C->mode == 1;
    const int n_blocks_log = m - k_log;
    if (pure ? (C->comp_slots > 64 || (k_log != 20 && k_log != 21))
             : (C->n_comp_log + 14 != m || k_log <= sub_log || k_log - sub_log > 8)) { printf("FFI chunk: bad shape\n"); return 201; }
    const size_t packed_witness_bytes = (size_t)len * sizeof(F128);
    const size_t arena_bytes = packed_witness_bytes / 2 * 29 + 2 * packed_witness_bytes + (64ull << 20);
    CK(device_arena.begin(arena_bytes));
    // ================= witness: host-given compression inputs ================
    F128 *df, *d_a, *d_b; uint8_t* d_zlin;
    CK(ffi_malloc(&df, len * sizeof(F128)));
    CK(ffi_malloc(&d_a, len * sizeof(F128)));
    CK(ffi_malloc(&d_b, len * sizeof(F128)));
    CK(ffi_malloc(&d_zlin, (size_t)len * 16));
    {
        uint32_t *d_cv, *d_m, *d_blen, *d_flags; b3u64* d_ctr;
        CK(ffi_malloc(&d_cv, (size_t)n_comp * 8 * 4)); CK(ffi_malloc(&d_m, (size_t)n_comp * 16 * 4));
        CK(ffi_malloc(&d_blen, (size_t)n_comp * 4)); CK(ffi_malloc(&d_flags, (size_t)n_comp * 4));
        CK(ffi_malloc(&d_ctr, (size_t)n_comp * 8));
        const bool dev_in = pure && C->rows_x;
        const long long n_blocks_p = 1LL << n_blocks_log;
        uint32_t* d_cvout = nullptr; uint8_t *d_rx = nullptr, *d_rw = nullptr; uint32_t *d_accs = nullptr, *d_keys = nullptr;
        if (dev_in) {
            const size_t rb = (size_t)C->vus * C->row_bytes;
            CK(ffi_malloc(&d_rx, rb)); CK(ffi_malloc(&d_rw, rb));
            CK(ffi_malloc(&d_accs, (size_t)C->vus * C->units_per_vu * 4)); CK(ffi_malloc(&d_keys, 64));
            CK(ffi_malloc(&d_cvout, (size_t)n_blocks_p * 2 * C->chunks_per_block * 32));
            CK(cudaMemcpy(d_rx, C->rows_x, rb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_rw, C->rows_w, rb, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_accs, C->accs, (size_t)C->vus * C->units_per_vu * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_keys, C->keys, 64, cudaMemcpyHostToDevice));
            pure_chain_inputs<<<(unsigned)((2 * n_blocks_p + 127) / 128), 128>>>(d_rx, d_rw, C->row_bytes, C->blocks_real, n_blocks_p, d_keys,
                C->comp_slots, C->per_role, C->chunks_per_block, C->chunk_nb0, C->chunk_nb1, C->blocks_per_vu, C->dummy_keyed,
                d_cv, d_m, (uint64_t*)d_ctr, d_blen, d_flags, d_cvout);
            CK(cudaGetLastError());
            if (C->pub_cv) CK(cudaMemcpy(C->pub_cv, d_cvout, (size_t)C->blocks_real * 2 * C->chunks_per_block * 32, cudaMemcpyDeviceToHost));
            ffi_free(d_rx); ffi_free(d_rw); ffi_free(d_keys); ffi_free(d_cvout);
        } else {
            CK(cudaMemcpy(d_cv, C->cv, (size_t)n_comp * 32, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_m, C->msg, (size_t)n_comp * 64, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_ctr, C->ctr, (size_t)n_comp * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_blen, C->blen, (size_t)n_comp * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_flags, C->flags, (size_t)n_comp * 4, cudaMemcpyHostToDevice));
        }
        if (!pure) {
            launch_blake3_witness_blocks(d_cv, d_m, d_ctr, d_blen, d_flags, (int)n_comp, n_comp,
                                         (b3u64*)df, (b3u64*)d_a, (b3u64*)d_b);
            CK(cudaGetLastError());
        } else {
            CK(cudaMemsetAsync(df, 0, len * sizeof(F128))); CK(cudaMemsetAsync(d_a, 0, len * sizeof(F128)));
            CK(cudaMemsetAsync(d_b, 0, len * sizeof(F128)));
            F128 *tz, *ta, *tb;
            CK(ffi_malloc(&tz, n_comp * 128 * sizeof(F128))); CK(ffi_malloc(&ta, n_comp * 128 * sizeof(F128)));
            CK(ffi_malloc(&tb, n_comp * 128 * sizeof(F128)));
            launch_blake3_witness_blocks(d_cv, d_m, d_ctr, d_blen, d_flags, (int)n_comp, n_comp, (b3u64*)tz, (b3u64*)ta, (b3u64*)tb);
            CK(cudaGetLastError());
            pure_comp_scatter<<<8192, 256>>>(tz, n_comp, C->comp_slots, k_log, df);
            pure_comp_scatter<<<8192, 256>>>(ta, n_comp, C->comp_slots, k_log, d_a);
            pure_comp_scatter<<<8192, 256>>>(tb, n_comp, C->comp_slots, k_log, d_b);
            CK(cudaGetLastError());
            ffi_free(tz); ffi_free(ta); ffi_free(tb);
            const long long n_units = (1LL << n_blocks_log) * C->units;
            static F128* g_ucache = nullptr;
            static size_t g_ucache_words = 0;
            const size_t cw = (size_t)n_units << (C->unit_log - 7);
            if (C->unit_cache == 2 && (g_ucache == nullptr || g_ucache_words != cw)) { printf("FFI chunk: no cached unit witness\n"); return 205; }
            if (C->unit_cache == 2) {
                for (int k = 0; k < 3; k++)
                    pure_unit_region_copy<<<8192, 256>>>(k == 0 ? df : k == 1 ? d_a : d_b, g_ucache + k * cw, 1LL << n_blocks_log, k_log,
                                                          C->unit_log, C->unit_pos0, C->units, 0);
                CK(cudaGetLastError());
                if (dev_in) ffi_free(d_accs);
                goto unit_done;
            }
            {
            uint64_t* d_in; uint32_t *d_ao, *d_ac, *d_bo, *d_bc;
            CK(ffi_malloc(&d_in, n_units * 10 * sizeof(uint64_t)));
            if (dev_in) {
                pure_unit_inputs<<<(unsigned)((n_units + 255) / 256), 256>>>(d_m, d_accs, C->units_per_vu, C->blocks_real, 1LL << n_blocks_log,
                                                                             C->units, C->comp_slots, C->per_role, C->blocks_per_vu, d_in);
                CK(cudaGetLastError());
                ffi_free(d_accs);
            } else {
                CK(cudaMemcpy(d_in, C->u_inputs, n_units * 10 * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }
            const size_t na = C->u_a_off[C->u_useful], nb = C->u_b_off[C->u_useful];
            CK(ffi_malloc(&d_ao, (C->u_useful + 1) * 4)); CK(ffi_malloc(&d_bo, (C->u_useful + 1) * 4));
            CK(ffi_malloc(&d_ac, (na ? na : 1) * 4)); CK(ffi_malloc(&d_bc, (nb ? nb : 1) * 4));
            CK(cudaMemcpy(d_ao, C->u_a_off, (C->u_useful + 1) * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_bo, C->u_b_off, (C->u_useful + 1) * 4, cudaMemcpyHostToDevice));
            if (na) CK(cudaMemcpy(d_ac, C->u_a_col, na * 4, cudaMemcpyHostToDevice));
            if (nb) CK(cudaMemcpy(d_bc, C->u_b_col, nb * 4, cudaMemcpyHostToDevice));
            if (C->u_useful > 32 * PURE_UW) { printf("FFI chunk: unit too large\n"); return 204; }
            pure_unit_witness<<<(unsigned)((n_units + 127) / 128), 128>>>(d_in, n_units, C->units, C->unit_log, C->unit_pos0, k_log,
                d_ao, d_ac, d_bo, d_bc, C->u_useful, C->u_const, C->u_n_in, df, d_a, d_b);
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(0));
            ffi_free(d_in); ffi_free(d_ao); ffi_free(d_ac); ffi_free(d_bo); ffi_free(d_bc);
            if (C->unit_cache == 1) {
                if (g_ucache_words != cw) {
                    if (g_ucache) cudaFree(g_ucache);
                    CK(cudaMalloc(&g_ucache, 3 * cw * sizeof(F128)));
                    g_ucache_words = cw;
                }
                for (int k = 0; k < 3; k++)
                    pure_unit_region_copy<<<8192, 256>>>(k == 0 ? df : k == 1 ? d_a : d_b, g_ucache + k * cw, 1LL << n_blocks_log, k_log,
                                                          C->unit_log, C->unit_pos0, C->units, 1);
                CK(cudaGetLastError());
            }
            }
        unit_done:
            if (C->pub_acc) {
                uint32_t *d_pa, *d_py;
                CK(ffi_malloc(&d_pa, (size_t)C->blocks_real * 4)); CK(ffi_malloc(&d_py, (size_t)C->blocks_real * 4));
                pure_unit_outputs<<<(C->blocks_real + 255) / 256, 256>>>(df, C->blocks_real, k_log, C->unit_log, C->unit_pos0 + C->units - 1,
                                                                         C->u_out_acc, C->u_out_y, d_pa, d_py);
                CK(cudaGetLastError());
                CK(cudaMemcpy(C->pub_acc, d_pa, (size_t)C->blocks_real * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(C->pub_y16, d_py, (size_t)C->blocks_real * 4, cudaMemcpyDeviceToHost));
                ffi_free(d_pa); ffi_free(d_py);
            }
        }
        const long long n_stripes = (1LL << (m - k_log)) / 8;
        chunk_zlin_transpose<<<4096, 256>>>(df, k_log, n_stripes, d_zlin);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(0));
        ffi_free(d_cv); ffi_free(d_m); ffi_free(d_blen); ffi_free(d_flags); ffi_free(d_ctr);
    }
    CKI(phase(0));

    // ================= L0 commit ============================================
    F128* d_cw0; uint8_t* d_tree0; long long l0_bl; int l0_ni;
    std::vector<MHash> l0_cap;
    {
        int k_code = (log_n - P->initial_k) + P->log_inv_rates[0];
        int num_ntts = 1 << P->initial_k;
        l0_bl = 1LL << k_code; l0_ni = num_ntts;
        long long cw_len = l0_bl * num_ntts;
        const TwiddleTable* tt;
        F128* d_tw;
        CK(get_cached_twiddles(k_code, tt, d_tw));
        CK(ffi_malloc(&d_cw0, cw_len * sizeof(F128)));
        CK(ffi_malloc(&d_tree0, (size_t)(2 * l0_bl - 1) * 32));
        if (ntt_can_fuse_source(k_code - P->log_inv_rates[0])) {
            launch_ntt(d_cw0, d_tw, *tt, P->log_inv_rates[0], k_code, num_ntts, 256, df, len - 1);
        } else {
            printf("FFI: unfused L0 rate-extend not wired\n"); return 102;
        }
        CK(cudaGetLastError());
        launch_merkle((const uint8_t*)d_cw0, d_tree0, l0_bl, num_ntts * 16);
        CK(cudaStreamSynchronize(0));
        uint32_t c0 = stratified_cap_depth(
            stratified_depths((size_t)P->queries[0], (uint32_t)k_code));
        l0_cap = merkle_cap_layer_device(d_tree0, (size_t)l0_bl, c0);
    }
    CKI(phase(1));

    // ================= link: Commit(root_B) -> points -> y (before any coin) =
    const int max_extra = 16;
    std::vector<uint64_t> x_raw((size_t)max_extra * (m - 6) * 2);
    int n_extra = 0;
    if (C->link_cb) {
        n_extra = C->link_cb(C->link_ctx, (const uint8_t*)l0_cap.data(), l0_cap.size() * 32, m, x_raw.data(), max_extra);
        if (n_extra < 0 || n_extra > max_extra) { printf("FFI chunk: link refused (%d)\n", n_extra); return 202; }
    }
    std::vector<std::vector<F128>> x_extra(n_extra, std::vector<F128>(m - 6));
    for (int e = 0; e < n_extra; e++)
        for (int i = 0; i < m - 6; i++) {
            size_t at = ((size_t)e * (m - 6) + i) * 2;
            x_extra[e][i] = F128{x_raw[at], x_raw[at + 1]};
        }
    CKI(phase(2));

    // ================= challenger + statement binding =======================
    FsChallenger ch(P->domain, P->domain_len);
    if (P->live_cb) ch.enable_live(P->live_cb, P->live_ctx, P->domain, P->domain_len);
    ch.observe_label((const uint8_t*)"flock-r1cs-v0", 13);
    ch.observe_bytes(P->statement_digest, 32);
    ch.observe_bytes((const uint8_t*)l0_cap.data(), l0_cap.size() * 32);

    FfiWriter W;
    W.hashes(l0_cap);

    // ================= zerocheck (unchanged) =================================
    vector<F128> zc_r1ab(64), zc_r1c(64), zc_m1s, zc_mis;
    F128 zc_z, zc_fa, zc_fb, zc_fc;
    vector<F128> zc_r(m), mlv_rhos;
    vector<uint64_t> zc_nonces;
    {
        const long long n_out = 1LL << (m - 6);
        static std::once_flag zerocheck_setup_once;
        static cudaError_t zerocheck_setup_error = cudaSuccess;
        std::call_once(zerocheck_setup_once, [&] {
            zerocheck_setup_error = upload_zerocheck_first_round_tables(P->zc_mcol, P->zc_f8mul, PHI_8_TABLE);
        });
        CK(zerocheck_setup_error);
        F128 *d_eq, *d_r1ab, *d_r1c, *d_ft, *d_am, *d_bm, *d_amn, *d_bmn, *d_p1, *d_pinf, *d_m1d, *d_mid;
        CK(ffi_malloc(&d_eq, (1LL << (m - 13)) * sizeof(F128)));
        CK(ffi_malloc(&d_r1ab, 64 * sizeof(F128))); CK(ffi_malloc(&d_r1c, 64 * sizeof(F128)));
        CK(ffi_malloc(&d_ft, 8 * 256 * sizeof(F128)));
        CK(ffi_malloc(&d_am, n_out * sizeof(F128))); CK(ffi_malloc(&d_bm, n_out * sizeof(F128)));
        CK(ffi_malloc(&d_amn, n_out * sizeof(F128))); CK(ffi_malloc(&d_bmn, n_out * sizeof(F128)));
        CK(ffi_malloc(&d_p1, ZT_MAX_BLOCKS * sizeof(F128))); CK(ffi_malloc(&d_pinf, ZT_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&d_m1d, sizeof(F128))); CK(ffi_malloc(&d_mid, sizeof(F128)));
        const int zt_dfull = m - 7, zt_lobits = zt_dfull > 7 ? zt_dfull - 7 : 0;
        F128 *d_eqlo, *d_eqhi;
        CK(ffi_malloc(&d_eqlo, (1LL << zt_lobits) * sizeof(F128)));
        CK(ffi_malloc(&d_eqhi, (1LL << (zt_dfull - zt_lobits)) * sizeof(F128)));

        ch.observe_label((const uint8_t*)"flock-zerocheck-v0", 18);
        std::vector<ChF128> rs(6);
        if (P->zc_initial_bits > 0)
            zc_nonces.push_back(ch.grind_pow_and_sample_f128_vec((uint32_t)P->zc_initial_bits, rs.data(), 6));
        else
            ch.sample_f128_vec(rs.data(), 6);
        std::vector<ChF128> ro(m - 13); ch.sample_f128_vec(ro.data(), m - 13);
        for (int i = 0; i < 6; i++) zc_r[i] = frch(rs[i]);
        int sm[3] = {0xF7, 0x53, 0xB5};
        for (int i = 0; i < 3; i++) zc_r[6 + i] = PHI_8_TABLE[sm[i]];
        F128 gm[4] = {F128{2, 0}, F128{4, 0}, F128{16, 0}, F128{256, 0}};
        for (int i = 0; i < 4; i++) zc_r[9 + i] = MUL(gm[i], f128_inv_host(ADD(ONE, gm[i])));
        for (int i = 0; i < m - 13; i++) zc_r[13 + i] = frch(ro[i]);

        build_eq_device(d_eq, &zc_r[13], m - 13);
        F128 round1_scale = ONE;
        for (int i = 6; i < 13; i++) round1_scale = MUL(round1_scale, ADD(ONE, zc_r[i]));
        launch_zerocheck_first_round_cpu_structured((const uint8_t*)d_a, (const uint8_t*)d_b,
                                   (const uint8_t*)df, d_eq, 1LL << (m - 13),
                                   round1_scale, d_r1ab, d_r1c);
        CK(cudaGetLastError()); CK(cudaStreamSynchronize(0));
        CK(cudaMemcpy(zc_r1ab.data(), d_r1ab, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(zc_r1c.data(), d_r1c, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
        { std::vector<ChF128> s(64);
          for (int i = 0; i < 64; i++) s[i] = toch(zc_r1ab[i]); ch.observe_f128_slice(s.data(), 64);
          for (int i = 0; i < 64; i++) s[i] = toch(zc_r1c[i]);  ch.observe_f128_slice(s.data(), 64); }
        if (P->zc_skip_bits > 0) {
            ChF128 zc_;
            zc_nonces.push_back(ch.grind_pow_and_sample_f128((uint32_t)P->zc_skip_bits, &zc_));
            zc_z = frch(zc_);
        } else {
            zc_z = frch(ch.sample_f128());
        }
        { vector<F128> wl = lagrange_phi(6, zc_z, 64);
          zc_fc = F128{0, 0};
          for (int i = 0; i < 64; i++) zc_fc = ADD(zc_fc, MUL(wl[i], zc_r1c[i])); }
        { vector<F128> ws = lagrange_phi(6, zc_z, 0);
          vector<F128> ft(8 * 256, F128{0, 0});
          for (int j = 0; j < 8; j++) for (int v = 0; v < 256; v++) { F128 acc{0, 0};
              for (int bb = 0; bb < 8; bb++) if ((v >> bb) & 1) acc = ADD(acc, ws[8 * j + bb]);
              ft[j * 256 + v] = acc; }
          CK(cudaMemcpy(d_ft, ft.data(), 8 * 256 * sizeof(F128), cudaMemcpyHostToDevice)); }
        build_eq_device(d_eqlo, &zc_r[7], zt_lobits);
        build_eq_device(d_eqhi, &zc_r[7 + zt_lobits], zt_dfull - zt_lobits);
        F128 *cA = d_am, *cB = d_bm, *nA = d_amn, *nB = d_bmn;
        long long L = n_out;
        int n_tail = (m - 6) - 1;
        vector<F128> S(n_tail);
        { vector<F128> values(n_tail), prefixes(n_tail); F128 acc = ONE;
          for (int i = 0; i < n_tail; i++) values[i] = ADD(ONE, zc_r[7 + i]);
          for (int i = 0; i < n_tail; i++) { prefixes[i] = acc; acc = MUL(acc, values[i]); }
          F128 inv = f128_inv_host(acc);
          for (int i = n_tail - 1; i >= 0; i--) { S[i] = MUL(prefixes[i], inv); inv = MUL(inv, values[i]); }
          for (int i = 1; i < n_tail; i++) S[i] = MUL(S[i - 1], S[i]); }
        F128 *d_part8, *d_out8;
        CK(ffi_malloc(&d_part8, 8 * ZT_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&d_out8, 8 * sizeof(F128)));
        launch_zerocheck_second_round_fold_with_lookahead((const uint8_t*)d_a, (const uint8_t*)d_b, d_ft,
                                   d_eqlo, d_eqhi, zt_lobits, n_out, d_am, d_bm,
                                   S[0], d_part8, d_out8);
        CK(cudaGetLastError());
        auto observe_message = [&](F128 m1, F128 mi) {
            zc_m1s.push_back(m1); zc_mis.push_back(mi);
            ch.observe_f128(toch(m1)); ch.observe_f128(toch(mi));
            if (P->zc_round_bits > 0) {
                ChF128 rho;
                zc_nonces.push_back(ch.grind_pow_and_sample_f128((uint32_t)P->zc_round_bits, &rho));
                mlv_rhos.push_back(frch(rho));
            } else {
                mlv_rhos.push_back(frch(ch.sample_f128()));
            }
        };
        F128 h[8];
        CK(cudaMemcpy(h, d_out8, 8 * sizeof(F128), cudaMemcpyDeviceToHost));
        observe_message(h[0], h[1]);
        F128 rho0 = mlv_rhos.back();
        observe_message(interp3(h[2], h[3], h[4], rho0), interp3(h[5], h[6], h[7], rho0));
        int k = 2;
        while (k + 1 <= n_tail && L / 16 > ZT_FINISH_OP) {
            launch_zerocheck_tail_lookahead(cA, cB, nA, nB, d_eqlo, d_eqhi, k, zt_lobits, L / 16,
                                mlv_rhos[mlv_rhos.size() - 2], mlv_rhos.back(),
                                S[k - 1], S[k], d_part8, d_out8);
            CK(cudaGetLastError());
            { F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
            L /= 4;
            CK(cudaMemcpy(h, d_out8, 8 * sizeof(F128), cudaMemcpyDeviceToHost));
            observe_message(h[0], h[1]);
            F128 rho = mlv_rhos.back();
            observe_message(interp3(h[2], h[3], h[4], rho), interp3(h[5], h[6], h[7], rho));
            k += 2;
        }
        { long long half = L / 2;
          launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos[mlv_rhos.size() - 2]);
          CK(cudaGetLastError());
          F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t;
          L = half; }
        int i = k - 1;
        for (; i < n_tail; i++) {
            launch_zerocheck_tail_fold_and_message(cA, cB, nA, nB, d_eqlo, d_eqhi, i + 1, zt_lobits,
                                     L / 4, mlv_rhos.back(), S[i], d_p1, d_pinf, d_m1d, d_mid);
            CK(cudaGetLastError());
            { F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
            L /= 2;
            F128 m1, mi;
            CK(cudaMemcpy(&m1, d_m1d, sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&mi, d_mid, sizeof(F128), cudaMemcpyDeviceToHost));
            observe_message(m1, mi);
        }
        { long long half = L / 2; launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos.back());
          CK(cudaGetLastError()); CK(cudaStreamSynchronize(0));
          F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
        CK(cudaMemcpy(&zc_fa, cA, sizeof(F128), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&zc_fb, cB, sizeof(F128), cudaMemcpyDeviceToHost));
        ch.observe_f128(toch(zc_fa)); ch.observe_f128(toch(zc_fb));
        ffi_free(d_eq); ffi_free(d_r1ab); ffi_free(d_r1c); ffi_free(d_ft);
        ffi_free(d_am); ffi_free(d_bm); ffi_free(d_amn); ffi_free(d_bmn);
        ffi_free(d_p1); ffi_free(d_pinf); ffi_free(d_m1d); ffi_free(d_mid);
        ffi_free(d_eqlo); ffi_free(d_eqhi);
        ffi_free(d_part8); ffi_free(d_out8);
    }
    ffi_free(d_a); ffi_free(d_b);
    W.f128s(zc_r1ab); W.f128s(zc_r1c);
    W.u64(zc_m1s.size());
    for (size_t i = 0; i < zc_m1s.size(); i++) { W.f128(zc_m1s[i]); W.f128(zc_mis[i]); }
    W.f128(zc_fa); W.f128(zc_fb); W.f128(zc_fc);
    W.u64(zc_nonces.size());
    for (uint64_t n : zc_nonces) W.u64(n);
    CKI(phase(3));

    const int irl = k_log - k_skip;
    vector<F128> xab_inner(mlv_rhos.begin(), mlv_rhos.begin() + irl);
    vector<F128> xab_outer(mlv_rhos.begin() + irl, mlv_rhos.end());

    // ================= lincheck: structured fold ===========================
    vector<F128> lc_e1s, lc_einfs, lc_zpart(64), lc_rrounds;
    vector<uint64_t> lc_nonces;
    F128 lc_rskip, lc_w;
    F128* d_zvec = nullptr;
    F128* d_zvec_initial = nullptr;
    {
        const int K = 1 << k_log, KS = 1 << sub_log, subs = 1 << (k_log - sub_log);
        const int n_log = m - k_log;
        const long long n_outer = 1LL << n_log, n_stripes = n_outer / 8;
        ch.observe_label((const uint8_t*)"flock-lincheck-v0", 17);
        F128 alpha;
        if (P->lc_alpha_bits > 0) {
            ChF128 a;
            lc_nonces.push_back(ch.grind_pow_and_sample_f128((uint32_t)P->lc_alpha_bits, &a));
            alpha = frch(a);
        } else {
            alpha = frch(ch.sample_f128());
        }
        F128 beta{0, 0}; bool has_pin = P->const_pin_col >= 0;
        if (has_pin) {
            if (P->lc_beta_bits > 0) {
                ChF128 b;
                lc_nonces.push_back(ch.grind_pow_and_sample_f128((uint32_t)P->lc_beta_bits, &b));
                beta = frch(b);
            } else {
                beta = frch(ch.sample_f128());
            }
        }
        static std::once_flag chunk_matrix_once;
        static cudaError_t chunk_matrix_error = cudaSuccess;
        static uint32_t *d_acp = nullptr, *d_ar = nullptr, *d_bcp = nullptr, *d_br = nullptr, *d_da = nullptr, *d_db = nullptr;
        std::call_once(chunk_matrix_once, [&] {
            auto up = [&](uint32_t** d, const uint32_t* h, size_t n) {
                if (chunk_matrix_error == cudaSuccess) chunk_matrix_error = cudaMalloc(d, (n ? n : 1) * sizeof(uint32_t));
                if (chunk_matrix_error == cudaSuccess && n) chunk_matrix_error = cudaMemcpy(*d, h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            };
            up(&d_acp, P->a_col_ptr, KS + 1); up(&d_ar, P->a_rows, P->a_nnz);
            up(&d_bcp, P->b_col_ptr, KS + 1); up(&d_br, P->b_rows, P->b_nnz);
            up(&d_da, C->delta_a, 2 * (size_t)C->n_delta_a); up(&d_db, C->delta_b, 2 * (size_t)C->n_delta_b);
        });
        CK(chunk_matrix_error);

        F128 *d_eq_inner, *d_comb, *d_comb_sub, *d_ratio, *d_eq_outer, *d_nC, *d_nZ, *d_p1, *d_pinf, *d_e1, *d_einf;
        CK(ffi_malloc(&d_eq_inner, K * sizeof(F128))); CK(ffi_malloc(&d_comb, K * sizeof(F128)));
        CK(ffi_malloc(&d_comb_sub, KS * sizeof(F128))); CK(ffi_malloc(&d_ratio, 256 * sizeof(F128)));
        CK(ffi_malloc(&d_zvec, K * sizeof(F128))); CK(ffi_malloc(&d_nC, K * sizeof(F128)));
        CK(ffi_malloc(&d_nZ, K * sizeof(F128)));
        CK(ffi_malloc(&d_eq_outer, n_outer * sizeof(F128)));
        CK(ffi_malloc(&d_p1, LC_MAX_BLOCKS * sizeof(F128))); CK(ffi_malloc(&d_pinf, LC_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&d_e1, sizeof(F128))); CK(ffi_malloc(&d_einf, sizeof(F128)));

        build_quirky_eq_device(d_eq_inner, zc_z, xab_inner, k_skip);
        CK(cudaGetLastError());
        // eq_inner = e_hi (sub-block bits) ⊗ e_lo (in-block bits): fold sub-block 0's slice, scale per sub-block.
        if (pure) {
            const int US = 1 << C->unit_log;
            F128 *d_base_u, *d_ratio_u;
            CK(ffi_malloc(&d_base_u, US * sizeof(F128))); CK(ffi_malloc(&d_ratio_u, 256 * sizeof(F128)));
            static std::once_flag unit_once;
            static cudaError_t unit_err = cudaSuccess;
            static uint32_t *d_uacp = nullptr, *d_uar = nullptr, *d_ubcp = nullptr, *d_ubr = nullptr;
            std::call_once(unit_once, [&] {
                auto up = [&](uint32_t** d, const uint32_t* h, size_t n) {
                    if (unit_err == cudaSuccess) unit_err = cudaMalloc(d, (n ? n : 1) * sizeof(uint32_t));
                    if (unit_err == cudaSuccess && n) unit_err = cudaMemcpy(*d, h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
                };
                up(&d_uacp, C->u_a_col_ptr, US + 1); up(&d_uar, C->u_a_rows, C->u_a_nnz);
                up(&d_ubcp, C->u_b_col_ptr, US + 1); up(&d_ubr, C->u_b_rows, C->u_b_nnz);
            });
            CK(unit_err);
            std::vector<F128> g0(1), gc(C->comp_slots), gu(C->units);
            pure_eq_gather<<<1, 256>>>(d_eq_inner, 0, 0, 1, d_ratio);
            CK(cudaMemcpy(g0.data(), d_ratio, sizeof(F128), cudaMemcpyDeviceToHost));
            if (g0[0].lo == 0 && g0[0].hi == 0) { printf("FFI chunk: eq_inner[0] = 0\n"); return 203; }
            F128 inv = f128_inv_host(g0[0]);
            pure_eq_gather<<<1, 256>>>(d_eq_inner, 14, 0, C->comp_slots, d_ratio);
            pure_eq_gather<<<1, 256>>>(d_eq_inner, C->unit_log, C->unit_pos0, C->units, d_ratio_u);
            CK(cudaMemcpy(gc.data(), d_ratio, C->comp_slots * sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gu.data(), d_ratio_u, C->units * sizeof(F128), cudaMemcpyDeviceToHost));
            for (auto& x : gc) x = MUL(x, inv);
            for (auto& x : gu) x = MUL(x, inv);
            CK(cudaMemcpy(d_ratio, gc.data(), C->comp_slots * sizeof(F128), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_ratio_u, gu.data(), C->units * sizeof(F128), cudaMemcpyHostToDevice));
            launch_linear_check_compressed_column_fold(d_eq_inner, d_acp, d_ar, d_bcp, d_br, alpha, KS, d_comb_sub);
            launch_linear_check_compressed_column_fold(d_eq_inner, d_uacp, d_uar, d_ubcp, d_ubr, alpha, US, d_base_u);
            CK(cudaGetLastError());
            pure_comb_expand<<<(unsigned)((K + 255) / 256), 256>>>(d_comb_sub, d_ratio, C->comp_slots, d_base_u, d_ratio_u,
                C->units, C->unit_log, C->unit_pos0, K, d_comb);
            CK(cudaGetLastError());
            ffi_free(d_base_u); ffi_free(d_ratio_u);
        } else {
            std::vector<F128> g(subs);
            chunk_eq_gather<<<1, 256>>>(d_eq_inner, sub_log, subs, d_ratio);
            CK(cudaGetLastError());
            CK(cudaMemcpy(g.data(), d_ratio, subs * sizeof(F128), cudaMemcpyDeviceToHost));
            if (g[0].lo == 0 && g[0].hi == 0) { printf("FFI chunk: eq_inner[0] = 0\n"); return 203; }
            F128 inv = f128_inv_host(g[0]);
            for (int j = 0; j < subs; j++) g[j] = MUL(g[j], inv);
            CK(cudaMemcpy(d_ratio, g.data(), subs * sizeof(F128), cudaMemcpyHostToDevice));
        }
        if (!pure) {
            launch_linear_check_compressed_column_fold(d_eq_inner, d_acp, d_ar, d_bcp, d_br, alpha, KS, d_comb_sub);
            CK(cudaGetLastError());
            chunk_comb_expand<<<(unsigned)((K + 255) / 256), 256>>>(d_comb_sub, d_ratio, sub_log, K, d_comb);
        }
        if (C->n_delta_a) chunk_delta_scatter<<<(C->n_delta_a + 255) / 256, 256>>>(d_da, C->n_delta_a, d_eq_inner, alpha, 1, d_comb);
        if (C->n_delta_b) chunk_delta_scatter<<<(C->n_delta_b + 255) / 256, 256>>>(d_db, C->n_delta_b, d_eq_inner, alpha, 0, d_comb);
        CK(cudaGetLastError());
        if (has_pin) {
            F128 v; CK(cudaMemcpy(&v, d_comb + P->const_pin_col, sizeof(F128), cudaMemcpyDeviceToHost));
            v = ADD(v, beta);
            CK(cudaMemcpy(d_comb + P->const_pin_col, &v, sizeof(F128), cudaMemcpyHostToDevice));
        }
        build_eq_device(d_eq_outer, xab_outer.data(), n_log);
        CK(cudaGetLastError());
        launch_linear_check_partial_fold(d_zlin, d_eq_outer, n_stripes, K, P->useful_bits, d_zvec);
        CK(cudaGetLastError()); CK(cudaStreamSynchronize(0));
        CK(ffi_malloc(&d_zvec_initial, K * sizeof(F128)));
        CK(cudaMemcpy(d_zvec_initial, d_zvec, K * sizeof(F128), cudaMemcpyDeviceToDevice));

        F128 *cC = d_comb, *cZ = d_zvec, *nC = d_nC, *nZ = d_nZ;
        long long L = K;
        for (int rnd = 0; rnd < irl; rnd++) {
            long long half = L / 2;
            launch_linear_check_message(cC, cZ, half, d_p1, d_pinf, d_e1, d_einf);
            CK(cudaGetLastError());
            F128 e1, einf;
            CK(cudaMemcpy(&e1, d_e1, sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&einf, d_einf, sizeof(F128), cudaMemcpyDeviceToHost));
            lc_e1s.push_back(e1); lc_einfs.push_back(einf);
            ch.observe_f128(toch(e1)); ch.observe_f128(toch(einf));
            F128 r;
            if (P->lc_round_bits > 0) {
                ChF128 rc;
                lc_nonces.push_back(ch.grind_pow_and_sample_f128((uint32_t)P->lc_round_bits, &rc));
                r = frch(rc);
            } else {
                r = frch(ch.sample_f128());
            }
            lc_rrounds.push_back(r);
            launch_linear_check_fold_pair(cC, cZ, nC, nZ, half, r);
            CK(cudaGetLastError()); CK(cudaStreamSynchronize(0));
            F128* t; t = cC; cC = nC; nC = t; t = cZ; cZ = nZ; nZ = t;
            L = half;
        }
        CK(cudaMemcpy(lc_zpart.data(), cZ, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
        { std::vector<ChF128> s(64); for (int i = 0; i < 64; i++) s[i] = toch(lc_zpart[i]);
          ch.observe_f128_slice(s.data(), 64); }
        if (P->lc_skip_bits > 0) {
            ChF128 rs_;
            lc_nonces.push_back(ch.grind_pow_and_sample_f128((uint32_t)P->lc_skip_bits, &rs_));
            lc_rskip = frch(rs_);
        } else {
            lc_rskip = frch(ch.sample_f128());
        }
        { vector<F128> lam = lagrange_weights_host(k_skip, lc_rskip);
          lc_w = F128{0, 0};
          for (int i = 0; i < 64; i++) lc_w = ADD(lc_w, MUL(lam[i], lc_zpart[i])); }
        ffi_free(d_eq_inner); ffi_free(d_comb); ffi_free(d_comb_sub); ffi_free(d_ratio);
        ffi_free(d_zvec); ffi_free(d_nC); ffi_free(d_nZ); ffi_free(d_eq_outer);
        ffi_free(d_p1); ffi_free(d_pinf); ffi_free(d_e1); ffi_free(d_einf);
    }
    ffi_free(d_zlin);
    W.u64(lc_e1s.size());
    for (size_t i = 0; i < lc_e1s.size(); i++) { W.f128(lc_e1s[i]); W.f128(lc_einfs[i]); }
    W.f128s(lc_zpart);
    W.u64(lc_nonces.size());
    for (uint64_t n : lc_nonces) W.u64(n);
    CKI(phase(4));

    vector<F128> r_inner_rest(lc_rrounds.rbegin(), lc_rrounds.rend());

    // ================= ring-switch batch: ab, c, extras ======================
    vector<F128> xfull_ab; xfull_ab.reserve(m - 6);
    xfull_ab.insert(xfull_ab.end(), r_inner_rest.begin(), r_inner_rest.end());
    xfull_ab.insert(xfull_ab.end(), xab_outer.begin(), xab_outer.end());
    vector<F128> xfull_c(zc_r.begin() + 6, zc_r.end());
    const int n_cl = 2 + n_extra;
    vector<vector<F128>> shats(n_cl, vector<F128>(128));
    F128 *d_suffix_ab, *d_suffix_c, *d_shat, *d_shat_partials;
    const int ring_switch_chunks = 64;
    {
        F128 *d_eq_shat_ab;
        const int shat_ab_tail_len = 1 << (irl - 1);
        CK(ffi_malloc(&d_eq_shat_ab, shat_ab_tail_len * sizeof(F128)));
        CK(ffi_malloc(&d_shat, 128 * sizeof(F128)));
        build_eq_device(d_eq_shat_ab, r_inner_rest.data() + 1, irl - 1);
        ring_switch_fold_witness_evaluations<<<128, 128>>>(d_zvec_initial, d_eq_shat_ab, shat_ab_tail_len, d_shat);
        CK(cudaGetLastError());
        CK(cudaMemcpy(shats[0].data(), d_shat, 128 * sizeof(F128), cudaMemcpyDeviceToHost));
        CK(ffi_free(d_eq_shat_ab)); CK(ffi_free(d_zvec_initial));
    }
    CK(ffi_malloc(&d_suffix_ab, len * sizeof(F128)));
    CK(ffi_malloc(&d_suffix_c, len * sizeof(F128)));
    CK(ffi_malloc(&d_shat_partials, 128 * ring_switch_chunks * sizeof(F128)));
    build_eq_device(d_suffix_ab, xfull_ab.data() + 1, (int)xfull_ab.size() - 1);
    auto fold_rows = [&](const F128* suffix, vector<F128>& shat) -> int {
        ring_switch_fold_rows_grouped<<<dim3(ring_switch_chunks, 16), 256>>>(df, suffix, len, ring_switch_chunks, d_shat_partials);
        ring_switch_fold_rows_reduce<<<128, 256>>>(d_shat_partials, ring_switch_chunks, d_shat);
        CK(cudaGetLastError());
        CK(cudaMemcpy(shat.data(), d_shat, 128 * sizeof(F128), cudaMemcpyDeviceToHost));
        return 0;
    };
    build_eq_device(d_suffix_c, xfull_c.data() + 1, (int)xfull_c.size() - 1);
    CKI(fold_rows(d_suffix_c, shats[1]));
    // Extra claims on their support only (see SparseBits): gather the words, fold them against the free-coordinate eq.
    std::vector<SparseBits> sp(n_extra);
    std::vector<std::vector<F128>> free_x(n_extra);
    const F128 F0{0, 0}, F1{1, 0};
    for (int ei = 0; ei < n_extra; ei++) {
        SparseBits& b = sp[ei];
        b.n_free = 0; b.fixed = 0;
        for (int j = 0; j < m - 7; j++) {
            F128 v = x_extra[ei][1 + j];
            if (v.lo == F0.lo && v.hi == F0.hi) continue;
            if (v.lo == F1.lo && v.hi == F1.hi) { b.fixed |= 1ull << j; continue; }
            b.free[b.n_free++] = j;
            free_x[ei].push_back(v);
        }
    }
    auto sparse_n = [&](int ei) { return 1LL << sp[ei].n_free; };
    F128 *d_e, *d_wx;
    long long max_n = 1;
    for (int ei = 0; ei < n_extra; ei++) max_n = std::max(max_n, sparse_n(ei));
    CK(ffi_malloc(&d_e, max_n * sizeof(F128)));
    CK(ffi_malloc(&d_wx, max_n * sizeof(F128)));
    for (int ei = 0; ei < n_extra; ei++) {
        const long long n = sparse_n(ei);
        if (sp[ei].n_free) build_eq_device(d_e, free_x[ei].data(), sp[ei].n_free);
        else { F128 one{1, 0}; CK(cudaMemcpy(d_e, &one, sizeof(F128), cudaMemcpyHostToDevice)); }
        chunk_sparse_gather<<<(unsigned)std::min<long long>((n + 255) / 256, 65535), 256>>>(df, sp[ei], n, d_wx);
        CK(cudaGetLastError());
        ring_switch_fold_rows_grouped<<<dim3(ring_switch_chunks, 16), 256>>>(d_wx, d_e, n, ring_switch_chunks, d_shat_partials);
        ring_switch_fold_rows_reduce<<<128, 256>>>(d_shat_partials, ring_switch_chunks, d_shat);
        CK(cudaGetLastError());
        CK(cudaMemcpy(shats[2 + ei].data(), d_shat, 128 * sizeof(F128), cudaMemcpyDeviceToHost));
    }
    ch.observe_label((const uint8_t*)"flock-pcs-open-batch-v0", 23);
    vector<uint64_t> rs_nonce(n_cl, 0);
    vector<vector<F128>> eq_rd(n_cl);
    vector<F128> claim(n_cl);
    for (int i = 0; i < n_cl; i++) {
        ch.observe_label((const uint8_t*)"flock-ring-switch-v0", 20);
        { std::vector<ChF128> s(128); for (int j = 0; j < 128; j++) s[j] = toch(shats[i][j]);
          ch.observe_f128_slice(s.data(), 128); }
        std::vector<ChF128> rd(7);
        if (P->rs_bits > 0)
            rs_nonce[i] = ch.grind_pow_and_sample_f128_vec((uint32_t)P->rs_bits, rd.data(), 7);
        else
            ch.sample_f128_vec(rd.data(), 7);
        vector<F128> rdf(7); for (int j = 0; j < 7; j++) rdf[j] = frch(rd[j]);
        eq_rd[i] = build_eq_host(rdf.data(), 7);
        vector<F128> shat_u = ta_transpose_host(shats[i]);
        F128 c{0, 0};
        for (int j = 0; j < 128; j++) c = ADD(c, MUL(shat_u[j], eq_rd[i][j]));
        claim[i] = c;
    }
    vector<uint64_t> batching_nonces;
    vector<F128> gam(n_cl);
    {
        std::vector<ChF128> g(n_cl);
        if (C->gamma_bits > 0)
            batching_nonces.push_back(ch.grind_pow_and_sample_f128_vec((uint32_t)C->gamma_bits, g.data(), n_cl));
        else
            ch.sample_f128_vec(g.data(), n_cl);
        for (int i = 0; i < n_cl; i++) gam[i] = frch(g[i]);
    }
    F128 target{0, 0};
    for (int i = 0; i < n_cl; i++) target = ADD(target, MUL(gam[i], claim[i]));
    F128 *d_w, *dcb;
    CK(ffi_malloc(&d_w, 2 * 128 * sizeof(F128)));
    CK(ffi_malloc(&dcb, len * sizeof(F128)));
    {
        vector<F128> w(2 * 128);
        for (int j = 0; j < 128; j++) { w[j] = MUL(gam[0], eq_rd[0][j]); w[128 + j] = MUL(gam[1], eq_rd[1][j]); }
        CK(cudaMemcpy(d_w, w.data(), 256 * sizeof(F128), cudaMemcpyHostToDevice));
        ring_switch_combine_basis<<<(unsigned)((len + 255) / 256), 256>>>(d_suffix_ab, d_suffix_c, d_w, d_w + 128, len, dcb);
        CK(cudaGetLastError());
    }
    CK(ffi_free(d_suffix_ab)); CK(ffi_free(d_suffix_c));
    for (int ei = 0; ei < n_extra; ei++) {
        vector<F128> w(128);
        for (int j = 0; j < 128; j++) w[j] = MUL(gam[2 + ei], eq_rd[2 + ei][j]);
        CK(cudaMemcpy(d_w, w.data(), 128 * sizeof(F128), cudaMemcpyHostToDevice));
        const long long n = sparse_n(ei);
        if (sp[ei].n_free) build_eq_device(d_e, free_x[ei].data(), sp[ei].n_free);
        else { F128 one{1, 0}; CK(cudaMemcpy(d_e, &one, sizeof(F128), cudaMemcpyHostToDevice)); }
        chunk_sparse_combine<<<(unsigned)std::min<long long>((n + 255) / 256, 65535), 256>>>(d_e, d_w, sp[ei], n, dcb);
        CK(cudaGetLastError());
    }
    CK(ffi_free(d_e)); CK(ffi_free(d_wx)); CK(ffi_free(d_w)); CK(ffi_free(d_shat)); CK(ffi_free(d_shat_partials));
    F128 first_u0, first_u2;
    {
        F128 *p0, *p2, *du0, *du2;
        CK(ffi_malloc(&p0, SMC_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&p2, SMC_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&du0, sizeof(F128))); CK(ffi_malloc(&du2, sizeof(F128)));
        launch_sumcheck_message(df, dcb, len / 2, p0, p2, du0, du2);
        CK(cudaGetLastError());
        CK(cudaMemcpy(&first_u0, du0, sizeof(F128), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&first_u2, du2, sizeof(F128), cudaMemcpyDeviceToHost));
        CK(ffi_free(p0)); CK(ffi_free(p2)); CK(ffi_free(du0)); CK(ffi_free(du2));
    }
    W.u64(n_cl);
    for (int i = 0; i < n_cl; i++) { W.f128s(shats[i]); W.u64(rs_nonce[i]); }
    W.u64(batching_nonces.size());
    for (uint64_t n : batching_nonces) W.u64(n);
    CKI(phase(5));

    // ================= ligerito (unchanged) =================================
    LigF256Proof lig;
    {
        LigF256Config LC;
        LC.log_n = log_n;
        LC.initial_k = P->initial_k;
        LC.recursive_steps = P->recursive_steps;
        LC.log_inv_rates = P->log_inv_rates;
        LC.recursive_ks = P->recursive_ks;
        LC.queries = P->queries;
        LC.grinding_bits = P->grinding_bits;
        LC.claim_batch_grinding_bits = P->claim_batch_grinding_bits;
        LC.consistency_batch_grinding_bits = P->consistency_batch_grinding_bits;
        LC.ood_samples = P->ood_samples;
        LigAlloc arena{lig_arena_alloc, lig_arena_release};
        int rc = run_ligerito_f256(LC, ch, target, first_u0, first_u2, df, dcb,
                                   d_cw0, d_tree0, l0_bl, l0_ni, arena, lig);
        if (rc != 0) { printf("FFI: ligerito F256 ladder failed (%d)\n", rc); return rc; }
    }
    ffi_free(d_cw0); ffi_free(d_tree0);
    ffi_free(df); ffi_free(dcb);
    W.u64(lig.recursive_caps.size());
    for (auto& cap : lig.recursive_caps) W.hashes(cap);
    auto write_open = [&](const LigLevelOpen& lo) {
        W.rows(lo.rows_flat, lo.n_rows, lo.row_len);
        W.hashes(lo.path);
    };
    W.u64(2 + lig.recursive_opens.size());
    write_open(lig.initial_open);
    for (auto& lo : lig.recursive_opens) write_open(lo);
    write_open(lig.final_open);
    W.f128s(lig.yr);
    W.u64(lig.transcript.size());
    for (auto& msg : lig.transcript) {
        W.f128(msg.u0.c0); W.f128(msg.u0.c1);
        W.f128(msg.u2.c0); W.f128(msg.u2.c1);
    }
    W.f128s(lig.ood_values);
    W.u64(lig.grinding_nonces.size());
    for (auto n : lig.grinding_nonces) W.u64(n);
    W.u64(lig.claim_batch_nonces.size());
    for (auto n : lig.claim_batch_nonces) W.u64(n);
    W.u64(lig.consistency_batch_nonces.size());
    for (auto n : lig.consistency_batch_nonces) W.u64(n);
    CKI(phase(6));
    if (C->phase_s) C->phase_s[7] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    *out_len = W.buf.size();
    *out = (uint8_t*)malloc(W.buf.size());
    memcpy(*out, W.buf.data(), W.buf.size());
    CK(device_arena.finish());
    return 0;
}

}  // extern "C"
