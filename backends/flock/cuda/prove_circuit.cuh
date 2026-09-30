// verity flock-circuit: Flock-CUDA for verity/flock-circuit (backends/flock/live/src/circuit.rs). Included after
// prove_chunk.cuh at the end of cuda-ghash/prove_ffi.cu by backends/flock/cuda_circuit_patch.py, which also derives
// flock_cuda_prove_circuit from flock_cuda_prove_chunk's body: the same commit, zerocheck, lincheck, ring switch and
// Ligerito, with this file's witness (fc_witness) and lincheck fold (fc_fold) in place of the chunk statement's.
//
// A block holds ranges of slots of a few types (compressions, units, the mask slot), each type's circuit a CSC of
// 2^type_log columns. The witness is built on the device: the compressions from their host-given inputs (natively, fc_sha_tape
// and fc_sha_rows from the plan h_plan, or level by level like a unit), the units by fc_unit_witness (32 units per CTA, bit-sliced: a row is one u32
// over the 32 units; rows in level order, several warps per level; then a 32x32 bit transpose per 32 rows into each
// unit's slot), the mask slot's free cells left zero (the no-op mask hook) or, under M1's masking, given as host slots like
// the hm96 slots. A deep unit's slots come from the host too (u_host), and with host_z the whole witness does (a tampered
// witness). Under M1 (zk_on) level 0's commitment is fc_commit_l0_zk's and the Ligerito lane phase carries level 0's hiding
// (ligerito_zk.cuh).

#define FC_MAX_TYPES 12
#define FC_MAX_RANGES 32
// device memory rep 0 leaves free when it keeps its witness for rep 1: outside the arena, the witness's scratch (at most 2 GiB
// at a time), its inputs, and the first proof's cached type matrices
#define FC_KEEP_MARGIN ((size_t)4 << 30)

typedef struct {
    FlockChunkParams c;                // link, delta, gamma, phase, host witness (c.host_z / host_a / host_b)
    int n_types;
    int type_log[FC_MAX_TYPES];
    // per type, its circuit as CSC over 2^type_log columns; a NULL column pointer is the compression circuit (P's CSC)
    const uint32_t* t_a_col_ptr[FC_MAX_TYPES]; const uint32_t* t_a_rows[FC_MAX_TYPES]; uint32_t t_a_nnz[FC_MAX_TYPES];
    const uint32_t* t_b_col_ptr[FC_MAX_TYPES]; const uint32_t* t_b_rows[FC_MAX_TYPES]; uint32_t t_b_nnz[FC_MAX_TYPES];
    int n_ranges;
    int r_type[FC_MAX_RANGES]; int r_pos0[FC_MAX_RANGES]; int r_count[FC_MAX_RANGES];
    // device witness (c.host_z == NULL): the compression slots (sha512x3), a second circuit evaluated like the units: its range,
    // rows as CSR, level schedule and every slot's input words ([blocks * count][in_words] F128); then the host slots given up
    // front (the hm96 slots: z, a, b words at absolute F128 word offsets)
    int h_range; int h_useful; int h_const; int h_in_words;
    const uint32_t* h_a_off; const uint32_t* h_a_col; const uint32_t* h_b_off; const uint32_t* h_b_col;
    int h_levels; const uint32_t* h_level_off; const uint32_t* h_order;
    const uint64_t* h_inputs;
    long long n_host0; const long long* host0_at; const int* host0_words; const uint64_t* host0_zab;
    // the unit type: its range, its rows as CSR, its level schedule and every unit's input words
    int u_range; int u_useful; int u_const; int u_in_words;
    const uint32_t* u_a_off; const uint32_t* u_a_col; const uint32_t* u_b_off; const uint32_t* u_b_col;
    int u_levels; const uint32_t* u_level_off; const uint32_t* u_order;
    const uint64_t* u_inputs;          // [blocks * count][in_words] F128 (lo, hi)
    int u_warps;                       // warps per CTA (1..8)
    double* witness_s;                 // optional [3]: compressions, units, transpose/lincheck pack (device seconds)
    // a cut template (u_n_late > 0): the unit's rows split in two passes. Pass A: levels [0, u_levels0) (rows that read no
    // late input), then the cut output rows are read back; mid_cb (host) computes the tail and lookups and returns every
    // unit's late input rows (bit-sliced, [groups][u_n_late] u32) and the host slots' witness words; pass B: the late rows,
    // levels [u_levels0, u_levels), then the transpose into the slots.
    int u_levels0; int u_n_late; const uint32_t* u_late_rows; int u_n_cut; const uint32_t* u_cut_rows;
    int (*mid_cb)(void* ctx, const uint32_t* cut, long long groups, uint32_t* late, long long* n_host, const long long** host_at,
                  const int** host_words, const uint64_t** host_zab);
    void* mid_ctx;
    // hm96-sha512 Merkle leaves (the SHA-512 build, cuda_sha512_patch.py): the device expands tree k's salts with ChaCha20 under
    // hm96_key (32 bytes) and the nonce hm96_tree returns for k (trees in build order); the scheme's constants are the salt
    // prefix || leaf prefix (256 bytes) and the key's byte tables (192 x 256 x 8 u64). hm96_tree null: unsalted leaves.
    long long (*hm96_tree)(void* ctx, int tree);
    void* hm96_ctx;
    const uint8_t* hm96_key;
    const uint8_t* hm96_prefixes;
    const uint64_t* hm96_tables;
    long long hm96_level0_leaves;   // the level-0 tree's leaves: salt-stream id 0 is refused for a tree of any other size
    // M1 (lane flock-zk, backends/flock/live/PROTOCOL.md §8), zk_on: Ligerito level 0's hiding. The level-0 codeword has the
    // witness's 2^initial_k lanes and 4 uniform ones (zk_extra: 4 lanes of L words), every lane's message padded by
    // (X_L + c)·μ (zk_pads: t_pad words per lane, lane-major, the witness's lanes first); the lane phase folds rep zk_rep's pair
    // (ligerito_zk.cuh); the proof stream ends in the zk section (fc_write_zk). zk_tamper: the negatives (LigZk::tamper).
    int zk_on; int zk_rep; int zk_t_pad; int zk_tamper;
    const uint64_t* zk_pads; const uint64_t* zk_extra;
    // the units' slots are among the host slots (host0), so the unit kernel does not run (a deep unit's witness is the host's)
    int u_host;
    // rep 1 from rep 0's witness and level-0 commitment (both reps bind one level-0 root): 1 keeps them for the next call with
    // the same reuse_tag, which leaves the device arena open between the two calls; 2 takes them, or proves in full when
    // nothing was kept (rep 0 had no room)
    int rep_reuse;
    unsigned long long reuse_tag;
    // the compression slots' native witness (the SHA-512 build): per row of the sha512x3 circuit, its a and b sources (two u32:
    // 0 zero, 1 one, 2 + 64 w + i bit i of tape word w), and the tape's words per slot; null: the level-by-level pass
    const uint32_t* h_plan;
    int h_tape_words;
} FlockCircuitParams;

// Salts of leaves idx[0..n) of tree `nonce` under `key`, expanded on the device (192 bytes each, in order): a test vector against
// the host's zk_hooks::LeafSalts.
__global__ void fc_hm96_salts(Hm96Key key, uint64_t nonce, const long long* __restrict__ idx, int n, uint32_t* __restrict__ out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) hm96_salt(key, nonce, idx[k], out + 48 * k);
}

extern "C" int flock_cuda_hm96_salts(const uint8_t* key, uint64_t nonce, const long long* idx, int n, uint8_t* out) {
    Hm96Key k{};
    memcpy(k.k, key, 32);
    long long* d_idx = nullptr;
    uint32_t* d_out = nullptr;
    if (cudaMalloc(&d_idx, n * sizeof(long long)) != cudaSuccess || cudaMalloc(&d_out, (size_t)n * 192) != cudaSuccess) return 1;
    cudaMemcpy(d_idx, idx, n * sizeof(long long), cudaMemcpyHostToDevice);
    fc_hm96_salts<<<(n + 127) / 128, 128>>>(k, nonce, d_idx, n, d_out);
    cudaError_t e = cudaMemcpy(out, d_out, (size_t)n * 192, cudaMemcpyDeviceToHost);
    cudaFree(d_idx);
    cudaFree(d_out);
    volatile uint32_t* vk = k.k;
    for (int j = 0; j < 8; j++) vk[j] = 0;
    return e == cudaSuccess ? 0 : 2;
}

__global__ void fc_comp_scatter(const F128* __restrict__ src, long long n_comp, int per_block, const uint32_t* __restrict__ slot,
                                int k_log, F128* __restrict__ dst) {
    const long long total = n_comp * 128;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
        long long ci = t >> 7, w = t & 127;
        long long blk = ci / per_block, j = ci % per_block;
        dst[(blk << (k_log - 7)) + ((long long)slot[j] << 7) + w] = src[t];
    }
}

// lane i holds row i (bit j = column j); after, lane i holds column i (bit j = row j)
__device__ __forceinline__ uint32_t fc_transpose32(uint32_t x, int lane) {
    const uint32_t m[5] = {0x0000FFFFu, 0x00FF00FFu, 0x0F0F0F0Fu, 0x33333333u, 0x55555555u};
#pragma unroll
    for (int k = 0; k < 5; k++) {
        const int j = 16 >> k;
        uint32_t y = __shfl_xor_sync(0xffffffffu, x, j);
        x = (lane & j) ? ((x & ~m[k]) | ((y & ~m[k]) >> j)) : ((x & m[k]) | ((y & m[k]) << j));
    }
    return x;
}

// One CTA per 32 units (units 32 g .. 32 g + 31 of the whole witness, block-major): scratch [3][useful] u32 (z, a, b rows,
// bit l = unit l). Input rows from the host's input words (warp 0, one ballot per bit), the constant row all ones, then the
// schedule's levels (rows of one level independent), then each 32-row block transposed into the units' slots.
__global__ void fc_unit_witness(const uint64_t* __restrict__ inputs, int in_words, long long n_units, int per_block, int pos0,
                                int unit_log, int k_log, const uint32_t* __restrict__ a_off, const uint32_t* __restrict__ a_col,
                                const uint32_t* __restrict__ b_off, const uint32_t* __restrict__ b_col, int useful, int cpos,
                                int n_levels, const uint32_t* __restrict__ level_off, const uint32_t* __restrict__ order,
                                uint32_t* __restrict__ scratch, uint32_t* __restrict__ z_out, uint32_t* __restrict__ a_out,
                                uint32_t* __restrict__ b_out, int phase, int lv0, int lv1, int n_late, const uint32_t* __restrict__ late_rows,
                                const uint32_t* __restrict__ late, int n_cut, const uint32_t* __restrict__ cut_rows, uint32_t* __restrict__ cut,
                                long long g_base) {
    // g_base: this launch's first group (inputs and n_units are the launch's; the slots it writes are absolute)
    const long long g = blockIdx.x;
    uint32_t* Z = scratch + g * 3LL * useful;
    uint32_t* A = Z + useful;
    uint32_t* B = A + useful;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const long long u = 32 * g + lane;
    const bool live = u < n_units;
    // phase 0: inputs, levels [lv0, lv1), cut rows out; phase 1: late rows in, levels [lv0, lv1), transpose; 2: all at once
    if (phase == 1) {
        for (int k = tid; k < n_late; k += blockDim.x) {
            const uint32_t r = late_rows[k], w = late[g * n_late + k];
            Z[r] = w; A[r] = w; B[r] = w;
        }
    }
    if (phase != 1 && warp == 0) {
        for (int w = 0; w < in_words; w++) {
            const uint64_t lo = live ? inputs[2 * (u * in_words + w)] : 0, hi = live ? inputs[2 * (u * in_words + w) + 1] : 0;
            for (int t = 0; t < 128; t++) {
                const uint32_t bit = (uint32_t)(((t < 64) ? (lo >> t) : (hi >> (t - 64))) & 1);
                const uint32_t W = __ballot_sync(0xffffffffu, bit);
                if (lane == 0) { const int r = 128 * w + t; Z[r] = W; A[r] = W; B[r] = W; }
            }
        }
        if (lane == 0) { Z[cpos] = 0xffffffffu; A[cpos] = 0xffffffffu; B[cpos] = 0xffffffffu; }
    }
    __syncthreads();
    for (int L = lv0; L < lv1; L++) {
        const uint32_t end = level_off[L + 1];
        for (uint32_t k = level_off[L] + tid; k < end; k += blockDim.x) {
            const uint32_t r = order[k];
            uint32_t x = 0, y = 0;
            for (uint32_t q = a_off[r]; q < a_off[r + 1]; q++) x ^= Z[a_col[q]];
            for (uint32_t q = b_off[r]; q < b_off[r + 1]; q++) y ^= Z[b_col[q]];
            Z[r] = x & y; A[r] = x; B[r] = y;
        }
        __syncthreads();
    }
    if (phase == 0) {
        for (int k = tid; k < n_cut; k += blockDim.x) cut[g * n_cut + k] = Z[cut_rows[k]];
        return;
    }
    const int nrb = (useful + 31) / 32;
    const long long ul = 32 * (g_base + g) + lane;
    const long long blk = ul / per_block, kk = ul % per_block;
    const long long base = (blk << (k_log - 5)) + ((long long)(pos0 + kk) << (unit_log - 5));
    for (int rb = warp; rb < nrb; rb += nw) {
        const int r = 32 * rb + lane;
        uint32_t zv = r < useful ? Z[r] : 0, av = r < useful ? A[r] : 0, bv = r < useful ? B[r] : 0;
        zv = fc_transpose32(zv, lane); av = fc_transpose32(av, lane); bv = fc_transpose32(bv, lane);
        if (ul < n_units + 32 * g_base) { z_out[base + rb] = zv; a_out[base + rb] = av; b_out[base + rb] = bv; }
    }
}

struct FcRanges {
    int n;
    int log[FC_MAX_RANGES];
    long long start[FC_MAX_RANGES], end[FC_MAX_RANGES];
    const F128* base[FC_MAX_RANGES];
};

// comb[i] = eq[(i with the slot's local bits cleared) | c0] / eq[c0] * base_t[i's local bits] for i in a range of type t, else 0.
__global__ void fc_comb_expand(FcRanges R, const F128* __restrict__ eq, long long c0, F128 inv, long long K, F128* __restrict__ out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < K; i += (long long)gridDim.x * blockDim.x) {
        F128 v{0, 0};
        for (int j = 0; j < R.n; j++) {
            if (i >= R.start[j] && i < R.end[j]) {
                const long long lm = (1LL << R.log[j]) - 1;
                v = ghash_mul_karatsuba(ghash_mul_karatsuba(eq[(i & ~lm) | c0], inv), R.base[j][i & lm]);
                break;
            }
        }
        out[i] = v;
    }
}

// What rep 0 keeps for rep 1 (rep_reuse): its witness (z, a, b; rep 1 transposes z for the lincheck again) and level-0
// codeword, tree and cap, live in the device arena, which stays open until rep 1 releases them.
struct FcKept {
    bool live = false;
    unsigned long long tag = 0;
    int m = 0, k_log = 0;
    F128 *df = nullptr, *d_a = nullptr, *d_b = nullptr;
    F128* d_cw0 = nullptr;
    uint8_t* d_tree0 = nullptr;
    long long l0_bl = 0;
    int l0_ni = 0;
    std::vector<MHash> l0_cap;
};
static FcKept g_fc_kept;
// the most this prover has asked of device_arena, which only grows (a larger request by another prover makes the keep check
// below conservative, never optimistic)
static size_t g_fc_arena = 0;

// a session that stopped between its reps: release what its rep 0 kept
static void fc_kept_drop() {
    if (!g_fc_kept.live) return;
    for (void* p : {(void*)g_fc_kept.df, (void*)g_fc_kept.d_a, (void*)g_fc_kept.d_b, (void*)g_fc_kept.d_cw0, (void*)g_fc_kept.d_tree0})
        ffi_free(p);
    device_arena.finish();
    g_fc_kept = FcKept{};
}

// whether rep 0 can keep its witness: the arena grows by `extra` and the device keeps `margin` free (FC_KEEP_MARGIN)
static bool fc_can_keep(size_t arena, size_t extra, size_t margin) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { cudaGetLastError(); return false; }
    const size_t want = arena + extra;
    const size_t after = want <= g_fc_arena ? free_b : (free_b + g_fc_arena >= want ? free_b + g_fc_arena - want : 0);
    return after >= margin;
}

// rep 1 opens rep 0's level-0 tree: with hm96 leaves it is salt stream 0 (tree 0), and rep 1's own trees count from 1
static void fc_rejoin_level0(const uint8_t* d_tree0) {
#ifdef FLOCK_HB
    if (g_hm96.tree_cb) {
        g_hm96.nonce[d_tree0] = 0;
        g_hm96.trees = 1;
    }
#else
    (void)d_tree0;
#endif
}

extern "C" int flock_cuda_circuit_kept(unsigned long long tag) { return g_fc_kept.live && g_fc_kept.tag == tag; }

#ifdef FLOCK_HB
// The compression slots' witness, native (FlockCircuitParams.h_plan; backends/flock/live/src/sha512_native.rs): each slot's tape,
// its input words, every addition's x ^ c, y ^ c, carries and sum and every Ch and Maj operand pair in compress_gadget's order, then
// its output words, word-major over the batch's slots; then each row's a and b as two tape bits (the plan), z = a & b.
struct FcTape {
    uint64_t* t;
    long long n, s;
    int k;
    __device__ void push(uint64_t w) { t[(long long)k++ * n + s] = w; }
    __device__ uint64_t add(uint64_t x, uint64_t y) {
        const uint64_t v = x + y, c = v ^ x ^ y;
        push(x ^ c); push(y ^ c); push(c); push(v);
        return v;
    }
};

__device__ static inline uint64_t fc_rotr(uint64_t x, int r) { return (x >> r) | (x << (64 - r)); }

__device__ static void fc_sha_compression(FcTape& t, const uint64_t* cv, const uint64_t* msg, uint64_t* out) {
    uint64_t w[80];
    for (int i = 0; i < 16; i++) w[i] = msg[i];
    for (int i = 16; i < 80; i++) {
        const uint64_t s0 = fc_rotr(w[i - 15], 1) ^ fc_rotr(w[i - 15], 8) ^ (w[i - 15] >> 7);
        const uint64_t s1 = fc_rotr(w[i - 2], 19) ^ fc_rotr(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = t.add(t.add(t.add(w[i - 16], s0), w[i - 7]), s1);
    }
    uint64_t a = cv[0], b = cv[1], c = cv[2], d = cv[3], e = cv[4], f = cv[5], g = cv[6], h = cv[7];
    for (int i = 0; i < 80; i++) {
        const uint64_t s1 = fc_rotr(e, 14) ^ fc_rotr(e, 18) ^ fc_rotr(e, 41);
        t.push(e); t.push(f ^ g);
        const uint64_t ch = g ^ (e & (f ^ g));
        const uint64_t t1 = t.add(t.add(t.add(t.add(h, s1), ch), SHA512_K[i]), w[i]);
        const uint64_t s0 = fc_rotr(a, 28) ^ fc_rotr(a, 34) ^ fc_rotr(a, 39);
        t.push(a ^ c); t.push(b ^ c);
        const uint64_t mj = c ^ ((a ^ c) & (b ^ c));
        const uint64_t t2 = t.add(s0, mj);
        const uint64_t ne = t.add(d, t1);
        const uint64_t na = t.add(t1, t2);
        h = g; g = f; f = e; e = ne; d = c; c = b; b = a; a = na;
    }
    const uint64_t st[8] = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < 8; i++) out[i] = t.add(cv[i], st[i]);
}

// one thread per sha512x3 slot: its inputs are in_words F128 (cv_k at u64 word 24k, msg_k at 24k + 8)
__global__ void fc_sha_tape(const uint64_t* __restrict__ inputs, int in_words, long long n, uint64_t* __restrict__ tape) {
    const long long s = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (s >= n) return;
    FcTape t{tape, n, s, 0};
    const uint64_t* in = inputs + 2LL * in_words * s;
    for (int j = 0; j < 2 * in_words; j++) t.push(in[j]);
    uint64_t out[24];
    for (int k = 0; k < 3; k++) fc_sha_compression(t, in + 24 * k, in + 24 * k + 8, out + 8 * k);
    for (int j = 0; j < 24; j++) t.push(out[j]);
}

// one warp per (slot, 32 rows): blockIdx.x the row block, blockIdx.y * warps + warp the batch's slot; the words it writes are
// fc_unit_witness's for that slot (s0: the batch's first slot over every block)
__global__ void fc_sha_rows(const uint64_t* __restrict__ tape, long long n, long long s0, const uint32_t* __restrict__ plan, int useful,
                            int per_block, int pos0, int slot_log, int k_log, uint32_t* __restrict__ z_out, uint32_t* __restrict__ a_out,
                            uint32_t* __restrict__ b_out) {
    const int lane = threadIdx.x & 31;
    const long long s = (long long)blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (s >= n) return;
    const int rb = blockIdx.x, r = 32 * rb + lane;
    auto bit = [&](uint32_t src) -> uint32_t {
        return src < 2 ? src : (uint32_t)(tape[(long long)((src - 2) >> 6) * n + s] >> ((src - 2) & 63)) & 1u;
    };
    const uint32_t a = r < useful ? bit(plan[2 * r]) : 0, b = r < useful ? bit(plan[2 * r + 1]) : 0;
    const uint32_t A = __ballot_sync(0xffffffffu, a), B = __ballot_sync(0xffffffffu, b);
    if (lane == 0) {
        const long long gs = s0 + s, blk = gs / per_block, kk = gs % per_block;
        const long long base = (blk << (k_log - 5)) + ((long long)(pos0 + kk) << (slot_log - 5));
        z_out[base + rb] = A & B; a_out[base + rb] = A; b_out[base + rb] = B;
    }
}
#endif

static int fc_bad_shape(const FlockCudaProveParams* P, const FlockCircuitParams* N) {
    const int k_log = P->k_log;
    if (k_log < 20 || k_log > 26 || N->n_types < 1 || N->n_types > FC_MAX_TYPES || N->n_ranges < 1 || N->n_ranges > FC_MAX_RANGES) return 1;
    for (int j = 0; j < N->n_ranges; j++) {
        const int t = N->r_type[j];
        if (t < 0 || t >= N->n_types || ((long long)(N->r_pos0[j] + N->r_count[j]) << N->type_log[t]) > (1LL << k_log)) return 1;
    }
    return 0;
}

static int fc_witness(const FlockCudaProveParams* P, const FlockCircuitParams* N, int m, int k_log, long long len, F128* df, F128* d_a,
                      F128* d_b, uint8_t* d_zlin) {
    const FlockChunkParams* C = &N->c;
    if (C->host_z) {
        CK(cudaMemcpy(df, C->host_z, len * sizeof(F128), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_a, C->host_a, len * sizeof(F128), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_b, C->host_b, len * sizeof(F128), cudaMemcpyHostToDevice));
    } else {
        auto t0 = std::chrono::steady_clock::now();
        CK(cudaMemsetAsync(df, 0, len * sizeof(F128))); CK(cudaMemsetAsync(d_a, 0, len * sizeof(F128)));
        CK(cudaMemsetAsync(d_b, 0, len * sizeof(F128)));
        {
            static std::once_flag h_once;
            static cudaError_t h_err = cudaSuccess;
            static uint32_t *h_ao = nullptr, *h_ac = nullptr, *h_bo = nullptr, *h_bc = nullptr, *h_lo = nullptr, *h_ord = nullptr;
            std::call_once(h_once, [&] {
                auto up = [&](uint32_t** d, const uint32_t* h, size_t n) {
                    if (h_err == cudaSuccess) h_err = cudaMalloc(d, (n ? n : 1) * sizeof(uint32_t));
                    if (h_err == cudaSuccess && n) h_err = cudaMemcpy(*d, h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
                };
                up(&h_ao, N->h_a_off, N->h_useful + 1); up(&h_bo, N->h_b_off, N->h_useful + 1);
                up(&h_ac, N->h_a_col, N->h_a_off[N->h_useful]); up(&h_bc, N->h_b_col, N->h_b_off[N->h_useful]);
                up(&h_lo, N->h_level_off, N->h_levels + 1); up(&h_ord, N->h_order, N->h_level_off[N->h_levels]);
            });
            CK(h_err);
            const int hr = N->h_range;
            const int hl = N->type_log[N->r_type[hr]];
            const long long hper = N->r_count[hr];
            const long long n_h = (1LL << (m - k_log)) * hper;
            const long long hg = (n_h + 31) / 32;
            uint64_t* d_hin; uint32_t* d_hs = nullptr;
            CK(cudaMalloc(&d_hin, (size_t)n_h * N->h_in_words * 16));
            CK(cudaMemcpy(d_hin, N->h_inputs, (size_t)n_h * N->h_in_words * 16, cudaMemcpyHostToDevice));
            if (N->h_plan) {
#ifdef FLOCK_HB
                static const uint32_t* plan_of = nullptr;
                static uint32_t* d_plan = nullptr;
                if (plan_of != N->h_plan) {
                    if (d_plan) cudaFree(d_plan);
                    CK(cudaMalloc(&d_plan, (size_t)N->h_useful * 8));
                    CK(cudaMemcpy(d_plan, N->h_plan, (size_t)N->h_useful * 8, cudaMemcpyHostToDevice));
                    plan_of = N->h_plan;
                }
                const long long per = (long long)N->h_tape_words * 8;
                const long long batch = std::max(1LL, std::min(n_h, (2LL << 30) / per));
                uint64_t* d_tape;
                CK(cudaMalloc(&d_tape, (size_t)(batch * per)));
                const int nrb = (N->h_useful + 31) / 32;
                for (long long s0 = 0; s0 < n_h; s0 += batch) {
                    const long long nb = std::min(batch, n_h - s0);
                    fc_sha_tape<<<(unsigned)((nb + 127) / 128), 128>>>(d_hin + 2LL * s0 * N->h_in_words, N->h_in_words, nb, d_tape);
                    fc_sha_rows<<<dim3((unsigned)nrb, (unsigned)((nb + 7) / 8)), 256>>>(d_tape, nb, s0, d_plan, N->h_useful, (int)hper,
                        N->r_pos0[hr], hl, k_log, (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b);
                    CK(cudaGetLastError());
                }
                CK(cudaStreamSynchronize(0));
                cudaFree(d_tape);
                d_hs = nullptr;
#else
                printf("FFI circuit: the native compression witness needs the SHA-512 build\n"); return 208;
#endif
            } else {
                const long long per = 3LL * N->h_useful * 4;
                const long long batch = std::max(1LL, std::min(hg, (2LL << 30) / per));
                CK(cudaMalloc(&d_hs, (size_t)(batch * per)));
                for (long long g0 = 0; g0 < hg; g0 += batch) {
                    const long long ng = std::min(batch, hg - g0);
                    CK(cudaMemsetAsync(d_hs, 0, (size_t)(ng * per)));
                    fc_unit_witness<<<(unsigned)ng, 32 * 8>>>(d_hin + 2LL * 32 * g0 * N->h_in_words, N->h_in_words, n_h - 32 * g0, (int)hper,
                        N->r_pos0[hr] + 0, hl, k_log, h_ao, h_ac, h_bo, h_bc, N->h_useful, N->h_const, N->h_levels, h_lo, h_ord, d_hs,
                        (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b, 2, 0, N->h_levels, 0, nullptr, nullptr, 0, nullptr, nullptr, g0);
                    CK(cudaGetLastError());
                }
            }
            CK(cudaStreamSynchronize(0));
            if (d_hs) cudaFree(d_hs);
            cudaFree(d_hin);
            const uint64_t* p = N->host0_zab;
            for (long long i = 0; i < N->n_host0; i++) {
                const size_t w = (size_t)N->host0_words[i];
                CK(cudaMemcpyAsync(df + N->host0_at[i], p, w * 16, cudaMemcpyHostToDevice));
                CK(cudaMemcpyAsync(d_a + N->host0_at[i], p + 2 * w, w * 16, cudaMemcpyHostToDevice));
                CK(cudaMemcpyAsync(d_b + N->host0_at[i], p + 4 * w, w * 16, cudaMemcpyHostToDevice));
                p += 6 * w;
            }
            CK(cudaStreamSynchronize(0));
        }
        auto t1 = std::chrono::steady_clock::now();
        if (!N->u_host) {
            static std::once_flag net_once;
            static cudaError_t net_err = cudaSuccess;
            static uint32_t *d_ao = nullptr, *d_ac = nullptr, *d_bo = nullptr, *d_bc = nullptr, *d_lo = nullptr, *d_ord = nullptr;
            std::call_once(net_once, [&] {
                auto up = [&](uint32_t** d, const uint32_t* h, size_t n) {
                    if (net_err == cudaSuccess) net_err = cudaMalloc(d, (n ? n : 1) * sizeof(uint32_t));
                    if (net_err == cudaSuccess && n) net_err = cudaMemcpy(*d, h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
                };
                const size_t na = N->u_a_off[N->u_useful], nb = N->u_b_off[N->u_useful];
                up(&d_ao, N->u_a_off, N->u_useful + 1); up(&d_bo, N->u_b_off, N->u_useful + 1);
                up(&d_ac, N->u_a_col, na); up(&d_bc, N->u_b_col, nb);
                up(&d_lo, N->u_level_off, N->u_levels + 1); up(&d_ord, N->u_order, N->u_level_off[N->u_levels]);
            });
            CK(net_err);
            const int ur = N->u_range;
            const int ul = N->type_log[N->r_type[ur]];
            const long long per_block = N->r_count[ur];
            const long long n_units = (1LL << (m - k_log)) * per_block;
            const long long groups = (n_units + 31) / 32;
            uint64_t* d_in; uint32_t* d_scratch;
            CK(cudaMalloc(&d_in, (size_t)n_units * N->u_in_words * 16));
            CK(cudaMemcpy(d_in, N->u_inputs, (size_t)n_units * N->u_in_words * 16, cudaMemcpyHostToDevice));
            // groups in batches that keep the scratch under ~2 GB
            const long long per = 3LL * N->u_useful * 4;
            const int warps = N->u_warps < 1 ? 1 : (N->u_warps > 8 ? 8 : N->u_warps);
            if (N->u_n_late == 0) {
                const long long batch = std::max(1LL, std::min(groups, (2LL << 30) / per));
                CK(cudaMalloc(&d_scratch, (size_t)(batch * per)));
                for (long long g0 = 0; g0 < groups; g0 += batch) {
                    const long long ng = std::min(batch, groups - g0);
                    CK(cudaMemsetAsync(d_scratch, 0, (size_t)(ng * per)));
                    fc_unit_witness<<<(unsigned)ng, 32 * warps>>>(d_in + 2LL * 32 * g0 * N->u_in_words, N->u_in_words, n_units - 32 * g0, (int)per_block,
                        N->r_pos0[ur] + 0, ul, k_log, d_ao, d_ac, d_bo, d_bc, N->u_useful, N->u_const, N->u_levels, d_lo, d_ord, d_scratch,
                        (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b, 2, 0, N->u_levels, 0, nullptr, nullptr, 0, nullptr, nullptr, g0);
                    CK(cudaGetLastError());
                }
            } else {
                // both passes over every group at once: the scratch holds the first pass's rows for the second
                uint32_t *d_lr, *d_cr, *d_late, *d_cut;
                CK(cudaMalloc(&d_scratch, (size_t)(groups * per))); CK(cudaMemsetAsync(d_scratch, 0, (size_t)(groups * per)));
                CK(cudaMalloc(&d_lr, N->u_n_late * 4)); CK(cudaMalloc(&d_cr, N->u_n_cut * 4));
                CK(cudaMalloc(&d_late, (size_t)groups * N->u_n_late * 4)); CK(cudaMalloc(&d_cut, (size_t)groups * N->u_n_cut * 4));
                CK(cudaMemcpy(d_lr, N->u_late_rows, N->u_n_late * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_cr, N->u_cut_rows, N->u_n_cut * 4, cudaMemcpyHostToDevice));
                fc_unit_witness<<<(unsigned)groups, 32 * warps>>>(d_in, N->u_in_words, n_units, (int)per_block, N->r_pos0[ur], ul, k_log, d_ao, d_ac,
                    d_bo, d_bc, N->u_useful, N->u_const, N->u_levels, d_lo, d_ord, d_scratch, (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b,
                    0, 0, N->u_levels0, 0, nullptr, nullptr, N->u_n_cut, d_cr, d_cut, 0);
                CK(cudaGetLastError());
                std::vector<uint32_t> cut((size_t)groups * N->u_n_cut), late((size_t)groups * N->u_n_late);
                CK(cudaMemcpy(cut.data(), d_cut, cut.size() * 4, cudaMemcpyDeviceToHost));
                long long n_host = 0; const long long* host_at = nullptr; const int* host_words = nullptr; const uint64_t* host_zab = nullptr;
                if (N->mid_cb(N->mid_ctx, cut.data(), groups, late.data(), &n_host, &host_at, &host_words, &host_zab) != 0) {
                    printf("FFI circuit: the host tail failed\n"); return 206;
                }
                CK(cudaMemcpy(d_late, late.data(), late.size() * 4, cudaMemcpyHostToDevice));
                fc_unit_witness<<<(unsigned)groups, 32 * warps>>>(d_in, N->u_in_words, n_units, (int)per_block, N->r_pos0[ur], ul, k_log, d_ao, d_ac,
                    d_bo, d_bc, N->u_useful, N->u_const, N->u_levels, d_lo, d_ord, d_scratch, (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b,
                    1, N->u_levels0, N->u_levels, N->u_n_late, d_lr, d_late, 0, nullptr, nullptr, 0);
                CK(cudaGetLastError());
                // the host slots (tail stages, lookups): z, a, b words at their word offsets
                const uint64_t* p = host_zab;
                for (long long i = 0; i < n_host; i++) {
                    const size_t w = (size_t)host_words[i];
                    CK(cudaMemcpyAsync(df + host_at[i], p, w * 16, cudaMemcpyHostToDevice));
                    CK(cudaMemcpyAsync(d_a + host_at[i], p + 2 * w, w * 16, cudaMemcpyHostToDevice));
                    CK(cudaMemcpyAsync(d_b + host_at[i], p + 4 * w, w * 16, cudaMemcpyHostToDevice));
                    p += 6 * w;
                }
                cudaFree(d_lr); cudaFree(d_cr); cudaFree(d_late); cudaFree(d_cut);
            }
            CK(cudaStreamSynchronize(0));
            cudaFree(d_scratch); cudaFree(d_in);
        }
        auto t2 = std::chrono::steady_clock::now();
        if (N->witness_s) {
            N->witness_s[0] = std::chrono::duration<double>(t1 - t0).count();
            N->witness_s[1] = std::chrono::duration<double>(t2 - t1).count();
        }
    }
    const long long n_stripes = (1LL << (m - k_log)) / 8;
    chunk_zlin_transpose<<<4096, 256>>>(df, k_log, n_stripes, d_zlin);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(0));
    return 0;
}

static int fc_fold(const FlockCudaProveParams* P, const FlockCircuitParams* N, const F128* d_eq, F128 alpha, int k_log, F128* d_comb) {
    static std::once_flag once;
    static cudaError_t err = cudaSuccess;
    static uint32_t* d_csc[FC_MAX_TYPES][4];
    std::call_once(once, [&] {
        auto up = [&](uint32_t** d, const uint32_t* h, size_t n) {
            if (err == cudaSuccess) err = cudaMalloc(d, (n ? n : 1) * sizeof(uint32_t));
            if (err == cudaSuccess && n) err = cudaMemcpy(*d, h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
        };
        for (int t = 0; t < N->n_types; t++) {
            const size_t cols = (size_t)1 << N->type_log[t];
            const bool comp = N->t_a_col_ptr[t] == nullptr;
            up(&d_csc[t][0], comp ? P->a_col_ptr : N->t_a_col_ptr[t], cols + 1); up(&d_csc[t][1], comp ? P->a_rows : N->t_a_rows[t], comp ? P->a_nnz : N->t_a_nnz[t]);
            up(&d_csc[t][2], comp ? P->b_col_ptr : N->t_b_col_ptr[t], cols + 1); up(&d_csc[t][3], comp ? P->b_rows : N->t_b_rows[t], comp ? P->b_nnz : N->t_b_nnz[t]);
        }
    });
    CK(err);
    int min_log = 64;
    for (int t = 0; t < N->n_types; t++) min_log = std::min(min_log, N->type_log[t]);
    std::vector<F128> head((size_t)1 << std::min(min_log, 10));
    CK(cudaMemcpy(head.data(), d_eq, head.size() * sizeof(F128), cudaMemcpyDeviceToHost));
    long long c0 = -1;
    for (size_t i = 0; i < head.size(); i++) if (head[i].lo || head[i].hi) { c0 = (long long)i; break; }
    if (c0 < 0) { printf("FFI circuit: eq vanishes on the first slot's head\n"); return 203; }
    const F128 inv = f128_inv_host(head[c0]);
    F128* d_base[FC_MAX_TYPES];
    for (int t = 0; t < N->n_types; t++) {
        const int cols = 1 << N->type_log[t];
        CK(ffi_malloc(&d_base[t], (size_t)cols * sizeof(F128)));
        launch_linear_check_compressed_column_fold(d_eq, d_csc[t][0], d_csc[t][1], d_csc[t][2], d_csc[t][3], alpha, cols, d_base[t]);
    }
    CK(cudaGetLastError());
    FcRanges R;
    R.n = N->n_ranges;
    for (int j = 0; j < R.n; j++) {
        const int t = N->r_type[j], l = N->type_log[t];
        R.log[j] = l; R.start[j] = (long long)N->r_pos0[j] << l; R.end[j] = (long long)(N->r_pos0[j] + N->r_count[j]) << l; R.base[j] = d_base[t];
    }
    const long long K = 1LL << k_log;
    fc_comb_expand<<<(unsigned)std::min<long long>((K + 255) / 256, 65535), 256>>>(R, d_eq, c0, inv, K, d_comb);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(0));
    for (int t = 0; t < N->n_types; t++) ffi_free(d_base[t]);
    return 0;
}

// ---- M1 (lane flock-zk): level 0's commitment with its hiding, and the zk section of the proof stream -------------------

// One batch of level-0 lanes before the transform: lane `lane`'s message (src at pos·s_pos + lane·s_lane, zero past l) plus its
// padding, c·μ at [0, t_pad) and μ at [l, l + t_pad) (verity's flock-zk patch, zk::lane_codeword; c = x).
__global__ void fc_zk_lane_input(F128* __restrict__ cw, const F128* __restrict__ src, long long s_pos, long long s_lane, int log_lanes,
                                 long long n, long long l, const F128* __restrict__ pads, int t_pad) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (n << log_lanes)) return;
    const long long pos = i >> log_lanes;
    const int lane = (int)(i & ((1 << log_lanes) - 1));
    F128 v = pos < l ? src[pos * s_pos + lane * s_lane] : F128{0, 0};
    const F128* p = pads + (size_t)lane * t_pad;
    if (pos < t_pad) v = f128_add(v, ghash_mul_karatsuba(F128{2, 0}, p[pos]));
    if (pos >= l && pos < l + t_pad) v = f128_add(v, p[pos - l]);
    cw[i] = v;
}

// Leaves of w + 4 lanes: the witness's w lanes, then the 4 extra ones.
__global__ void fc_zk_interleave(F128* __restrict__ out, const F128* __restrict__ cw_w, const F128* __restrict__ cw_x, int w, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n * (w + 4)) return;
    const long long pos = i / (w + 4);
    const int lane = (int)(i - pos * (w + 4));
    out[i] = lane < w ? cw_w[pos * w + lane] : cw_x[pos * 4 + lane - w];
}

// commit_zk on the device. The padding's upper half is not zero, so the transform runs every layer (M0's commit skips layer 0
// by reading the message replicated); the witness's lanes and the extra ones go as two batches (the NTT's lane tiles are powers
// of two), then are interleaved. The pads and extra lanes stay on the device for the lane phase (the caller frees them).
static int fc_commit_l0_zk(const FlockCudaProveParams* P, const FlockCircuitParams* N, int log_n, const F128* df, F128** d_pads,
                           F128** d_extra, F128** d_cw0, uint8_t** d_tree0, long long* l0_bl, int* l0_ni, std::vector<MHash>& l0_cap) {
    const int k = P->initial_k, w = 1 << k, t_pad = N->zk_t_pad;
    const int k_code = (log_n - k) + P->log_inv_rates[0];
    const long long n = 1LL << k_code, l = 1LL << (log_n - k);
    if (t_pad <= 0 || t_pad > l) { printf("FFI circuit: zk t_pad %d outside (0, %lld]\n", t_pad, l); return 204; }
    CK(ffi_malloc(d_pads, (size_t)(w + 4) * t_pad * sizeof(F128)));
    CK(ffi_malloc(d_extra, (size_t)4 * l * sizeof(F128)));
    CK(cudaMemcpy(*d_pads, N->zk_pads, (size_t)(w + 4) * t_pad * sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(*d_extra, N->zk_extra, (size_t)4 * l * sizeof(F128), cudaMemcpyHostToDevice));
    const TwiddleTable* tt;
    F128* d_tw;
    CK(get_cached_twiddles(k_code, tt, d_tw));
    F128 *cw_w, *cw_x;
    CK(ffi_malloc(&cw_w, (size_t)n * w * sizeof(F128)));
    CK(ffi_malloc(&cw_x, (size_t)n * 4 * sizeof(F128)));
    fc_zk_lane_input<<<(unsigned)((n * w + 255) / 256), 256>>>(cw_w, df, w, 1, k, n, l, *d_pads, t_pad);
    fc_zk_lane_input<<<(unsigned)((n * 4 + 255) / 256), 256>>>(cw_x, *d_extra, 1, l, 2, n, l, *d_pads + (size_t)w * t_pad, t_pad);
    CK(cudaGetLastError());
    launch_ntt(cw_w, d_tw, *tt, 0, k_code, w);
    launch_ntt(cw_x, d_tw, *tt, 0, k_code, 4);
    CK(cudaGetLastError());
    CK(ffi_malloc(d_cw0, (size_t)n * (w + 4) * sizeof(F128)));
    CK(ffi_malloc(d_tree0, (size_t)(2 * n - 1) * FLOCK_HB));
    fc_zk_interleave<<<(unsigned)((n * (w + 4) + 255) / 256), 256>>>(*d_cw0, cw_w, cw_x, w, n);
    CK(cudaGetLastError());
    launch_merkle((const uint8_t*)*d_cw0, *d_tree0, n, (w + 4) * 16);
    CK(cudaStreamSynchronize(0));
    ffi_free(cw_w);
    ffi_free(cw_x);
    *l0_bl = n;
    *l0_ni = w + 4;
    l0_cap = merkle_cap_layer_device(*d_tree0, (size_t)n, stratified_cap_depth(stratified_depths((size_t)P->queries[0], (uint32_t)k_code)));
    return 0;
}

// The proof stream's zk section (after the Ligerito section): e, T', ybar; the lane phase's record (the pre-OOD target, (β, y)
// per OOD sample, the lane challenges, ρ, level 0's queries); lincheck's comb folded to 2^k_skip words. flock_live::gpu reads it.
static void fc_write_zk(FfiWriter& W, const LigZk& zk, const std::vector<F128>& comb) {
    auto f256 = [&](F256Ext v) { W.f128(v.c0); W.f128(v.c1); };
    f256(zk.e[0]); f256(zk.e[1]); f256(zk.t_prime);
    W.u64(zk.ybar.size());
    for (auto& y : zk.ybar) f256(y);
    W.f128(zk.target);
    W.u64(zk.ood.size());
    for (auto& o : zk.ood) { W.f128(o.first); W.f128(o.second); }
    W.u64(zk.challenges.size());
    for (auto& c : zk.challenges) f256(c);
    f256(zk.rho[0]); f256(zk.rho[1]);
    W.u64(zk.queries0.size());
    for (size_t q : zk.queries0) W.u64(q);
    W.f128s(comb);
}
