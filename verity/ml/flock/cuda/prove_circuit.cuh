// verity flock-circuit: Flock-CUDA for verity/flock-circuit (verity/ml/flock/live/src/circuit.rs). Included after
// prove_chunk.cuh at the end of cuda-ghash/prove_ffi.cu by verity/ml/flock/cuda_circuit_patch.py, which also derives
// flock_cuda_prove_circuit from flock_cuda_prove_chunk's body: the same commit, zerocheck, lincheck, ring switch and
// Ligerito, with this file's witness (fc_witness), lincheck fold (fc_fold) and arena size (fc_arena_of, arena_bytes.h) in place
// of the chunk statement's.
//
// A block holds ranges of slots of a few types (compressions, units, the mask slot), each type's circuit a CSC of
// 2^type_log columns. The witness is built on the device: the compressions from their host-given inputs (natively, fc_sha_tape
// and fc_sha_rows from the plan h_plan, or level by level like a unit), the hm96 slots natively beside them (fc_hm96_tape, the
// plan m_plan) or given as host slots, the units by fc_unit_witness (32 units per CTA, bit-sliced: a row is one u32
// over the 32 units; rows in level order, several warps per level; then a 32x32 bit transpose per 32 rows into each
// unit's slot), the mask slot's free cells left zero (the no-op mask hook) or, under M1's masking, given as host slots. A
// deep unit's slots come from the host too (u_host), and with host_z the whole witness does (a tampered
// witness). Under M1 (zk_on) level 0's commitment is fc_commit_l0_zk's and the Ligerito lane phase carries level 0's hiding
// (ligerito_zk.cuh).

#include <cuda_pipeline.h>
#include <unordered_map>

#define FC_MAX_TYPES 12
#define FC_MAX_RANGES 32
// the witness's scratch outside the arena (sha512 tapes, compression and unit rows) goes in batches of at most this many bytes:
// it is held beside the arena while the next session's witness builds
#define FC_WITNESS_BATCH (512LL << 20)
#include "arena_bytes.h"
#include "csc_share.cuh"

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
    // hm96-sha512 Merkle leaves (the SHA-512 build, cuda_sha512_patch.py): hm96_tree returns tree k's salts (trees in build
    // order; 192 bytes per leaf, host memory the host keeps until the prover returns, read from the OS); the scheme's constants
    // are the salt prefix || leaf prefix (256 bytes) and the key's byte tables (192 x 256 x 8 u64). hm96_tree null: unsalted.
    const uint8_t* (*hm96_tree)(void* ctx, int tree, long long leaves);
    void* hm96_ctx;
    const uint8_t* hm96_prefixes;
    const uint64_t* hm96_tables;
    long long hm96_level0_leaves;   // the level-0 tree's leaves: tree 0 (level 0's salts) is refused at any other size
    // M1 (verity/ml/flock/live/PROTOCOL.md §8), zk_on: Ligerito level 0's hiding. The level-0 codeword has the
    // witness's 2^initial_k lanes and 4 uniform ones (zk_extra: 4 lanes of L words), every lane's message padded by
    // (X_L + c)·μ (zk_pads: t_pad words per lane, lane-major, the witness's lanes first); the lane phase folds rep zk_rep's pair
    // (ligerito_zk.cuh); the proof stream ends in the zk section (fc_write_zk). zk_tamper: the negatives (LigZk::tamper).
    int zk_on; int zk_rep; int zk_t_pad; int zk_tamper;
    const uint64_t* zk_pads; const uint64_t* zk_extra;
    // the units' slots are among the host slots (host0), so the unit kernel does not run (a deep unit's witness is the host's)
    int u_host;
    // rep 1 from rep 0's level-0 commitment (both reps bind one level-0 root): 1 keeps it for the next call with the same
    // reuse_tag, which leaves the device arena open between the two calls; 2 takes it and builds the witness again, or proves
    // in full when nothing was kept
    int rep_reuse;
    unsigned long long reuse_tag;
    // the compression slots' native witness (the SHA-512 build): per row of the sha512x3 circuit, its a and b sources (two u32:
    // 0 zero, 1 one, 2 + 64 w + i bit i of tape word w), and the tape's words per slot; null: the level-by-level pass
    const uint32_t* h_plan;
    int h_tape_words;
    // the host slots' layout in host0_zab: 0, each slot's z, a, b words; 1, its a, b words and z = a & b (the host checked every
    // word), which needs host0_zab page-locked and mapped (flock_cuda_host_alloc)
    int host0_ab;
    // the hm96 slots' native witness, beside h_plan's: their range, the hm96 circuit's plan (as h_plan) and useful rows, the
    // tape's words per slot, every hm96 slot's input words ([blocks * count][m_in_words] F128), the salt prefix's midstate (8 u64)
    // and the key's byte tables (hm96 mask: 192 x 256 x 8 u64); m_plan null: the hm96 slots are among the host slots
    int m_range; int m_useful; int m_in_words; int m_tape_words;
    const uint32_t* m_plan; const uint64_t* m_inputs; const uint64_t* m_mid; const uint64_t* m_tables;
    // the key of the slot types' CSC and the unit's CSR on the device (FcTypeUploads: the CSC one copy per GPU, shared by its provers);
    // where it holds a copy, the row arrays (t_a_rows, t_b_rows; u_a_off, u_a_col, u_b_off, u_b_col) may be null. 0: keyed by
    // their host addresses (FcUploads)
    unsigned long long types_gen;
} FlockCircuitParams;

// hm96 leaves of the digests x[0..n) (64 bytes each) under the host's salts (192 bytes each), finished on the device as a tree's
// leaves are: a test vector against flock_merkle::hm96::leaf.
extern "C" int flock_cuda_hm96_leaves(const uint8_t* x, const uint8_t* salts, long long n, const uint8_t* prefixes,
                                      const uint64_t* tables, uint8_t* out) {
    std::vector<uint32_t> frag(2 * HM96_FRAGS);
    hm96_fragments(tables, frag.data());
    uint8_t *d_x = nullptr, *d_s = nullptr, *d_pre = nullptr;
    uint32_t* d_f = nullptr;
    uint64_t* d_mid = nullptr;
    if (cudaMalloc(&d_x, (size_t)n * 64) != cudaSuccess || cudaMalloc(&d_s, (size_t)n * 192) != cudaSuccess ||
        cudaMalloc(&d_pre, 256) != cudaSuccess || cudaMalloc(&d_f, frag.size() * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&d_mid, 16 * sizeof(uint64_t)) != cudaSuccess)
        return 1;
    cudaMemcpy(d_x, x, (size_t)n * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, salts, (size_t)n * 192, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre, prefixes, 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, frag.data(), frag.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    hm96_midstates<<<1, 1>>>(d_pre, d_mid);
    launch_hm96_finish(d_x, (const uint32_t*)d_s, 0, n, 128, d_f, d_mid);
    cudaError_t e = cudaMemcpy(out, d_x, (size_t)n * 64, cudaMemcpyDeviceToHost);
    cudaMemset(d_s, 0, (size_t)n * 192);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_pre); cudaFree(d_f); cudaFree(d_mid);
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

// What rep 0 keeps for rep 1 (rep_reuse): its level-0 codeword, tree and cap, and under M1 level 0's pads and extra lanes (the
// session's, which rep 1's lane phase reads), live in the device arena, which stays open until rep 1 releases them.
struct FcKept {
    bool live = false;
    unsigned long long tag = 0;
    int m = 0, k_log = 0;
    F128* d_cw0 = nullptr;
    uint8_t* d_tree0 = nullptr;
    long long l0_bl = 0;
    int l0_ni = 0;
    std::vector<MHash> l0_cap;
    F128 *d_zk_pads = nullptr, *d_zk_extra = nullptr;
};
static FcKept g_fc_kept;

// A circuit's index arrays on the device, uploaded once per circuit and kept for the process, so one prover proves statements
// of many circuits. Keyed by the host arrays' addresses, which is sound only for arrays the host never frees (gpu_circuit.rs
// `types`'s compression schedule, gpu.rs `statics_for`); the slot types' CSC and the unit's CSR, which it frees, are
// FcTypeUploads'.
struct FcUploads {
    std::mutex mu;
    std::vector<std::pair<std::vector<const void*>, std::vector<uint32_t*>>> held;
    // the device copies of the `n` arrays `h[i]` of `len[i]` words each
    cudaError_t get(const uint32_t* const* h, const size_t* len, size_t n, uint32_t** d) {
        const std::vector<const void*> key(h, h + n);
        std::lock_guard<std::mutex> lock(mu);
        for (const auto& e : held)
            if (e.first == key) { std::copy(e.second.begin(), e.second.end(), d); return cudaSuccess; }
        std::vector<uint32_t*> got(n, nullptr);
        for (size_t i = 0; i < n; i++) {
            cudaError_t err = cudaMalloc(&got[i], (len[i] ? len[i] : 1) * sizeof(uint32_t));
            if (err == cudaSuccess && len[i]) err = cudaMemcpy(got[i], h[i], len[i] * sizeof(uint32_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { for (uint32_t* p : got) if (p) cudaFree(p); return err; }
        }
        held.emplace_back(key, got);
        std::copy(got.begin(), got.end(), d);
        return cudaSuccess;
    }
};

// a session that stopped between its reps: release what its rep 0 kept
static void fc_kept_drop() {
    if (!g_fc_kept.live) return;
    for (void* p : {(void*)g_fc_kept.d_cw0, (void*)g_fc_kept.d_tree0, (void*)g_fc_kept.d_zk_pads, (void*)g_fc_kept.d_zk_extra})
        if (p) ffi_free(p);
    device_arena.finish();
    g_fc_kept = FcKept{};
}

// rep 1 opens rep 0's level-0 tree: with hm96 leaves its salts are level 0's (tree 0), and rep 1's own trees count from 1
static void fc_rejoin_level0(const uint8_t* d_tree0) {
#ifdef FLOCK_HB
    if (g_hm96.tree_cb) {
        g_hm96.salts[d_tree0] = {hm96_tree_salts(0, g_hm96.level0_leaves), g_hm96.level0_leaves};
        g_hm96.trees = 1;
    }
#else
    (void)d_tree0;
#endif
}

extern "C" int flock_cuda_circuit_kept(unsigned long long tag) { return g_fc_kept.live && g_fc_kept.tag == tag; }

#ifdef FLOCK_HB
// The compression slots' witness, native (FlockCircuitParams.h_plan; verity/ml/flock/live/src/sha512_native.rs): each slot's tape,
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

// a compression's words dropped: its output alone
struct FcNoTape {
    __device__ void push(uint64_t) {}
    __device__ uint64_t add(uint64_t x, uint64_t y) { return x + y; }
};

template <class T>
__device__ static void fc_sha_compression(T& t, const uint64_t* cv, const uint64_t* msg, uint64_t* out) {
    uint64_t w[80];
    for (int i = 0; i < 16; i++) w[i] = msg[i];
    for (int i = 16; i < 80; i++) {
        const uint64_t s0 = sx::s0<0>(w[i - 15]);
        const uint64_t s1 = sx::s1<0>(w[i - 2]);
        w[i] = t.add(t.add(t.add(w[i - 16], s0), w[i - 7]), s1);
    }
    uint64_t a = cv[0], b = cv[1], c = cv[2], d = cv[3], e = cv[4], f = cv[5], g = cv[6], h = cv[7];
    for (int i = 0; i < 80; i++) {
        const uint64_t s1 = sx::S1<0>(e);
        t.push(e); t.push(f ^ g);
        const uint64_t ch = sx::ch(e, f, g);
        const uint64_t t1 = t.add(t.add(t.add(t.add(h, s1), ch), sx::K_DEV[i]), w[i]);
        const uint64_t s0 = sx::S0<0>(a);
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

// a compression's tape words: its schedule's 64 x 3 additions, its 80 rounds' 32 words each, its feed-forward's 8 additions
constexpr int FC_SHA_COMP_WORDS = 64 * 3 * 4 + 80 * 32 + 8 * 4;

// fc_sha_tape's words, a thread per compression (three a slot, compression-major so a warp's lanes write adjacent slots):
// compression c's words start at 2 in_words + c FC_SHA_COMP_WORDS on its slot's tape, its outputs at 2 in_words +
// 3 FC_SHA_COMP_WORDS + 8c (tape_words must be 2 in_words + 3 FC_SHA_COMP_WORDS + 24)
__global__ void fc_sha_tape3(const uint64_t* __restrict__ inputs, int in_words, long long n, uint64_t* __restrict__ tape) {
    const long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= 3 * n) return;
    const int c = (int)(g / n);
    const long long s = g - c * n;
    const uint64_t* in = inputs + 2LL * in_words * s;
    FcTape t{tape, n, s, 0};
    if (c == 0)
        for (int j = 0; j < 2 * in_words; j++) t.push(in[j]);
    t.k = 2 * in_words + c * FC_SHA_COMP_WORDS;
    uint64_t out[8];
    fc_sha_compression(t, in + 24 * c, in + 24 * c + 8, out);
    t.k = 2 * in_words + 3 * FC_SHA_COMP_WORDS + 8 * c;
    for (int j = 0; j < 8; j++) t.push(out[j]);
}

struct FcMid { uint64_t w[8]; };

// An hm96 slot's tape (sha512_native's Slot::Hm96), a thread per compression as fc_sha_tape3's: its input words (cv at u64 word 0,
// pad at 8, the salt y at 24, its bytes little-endian), x = the compression of (cv, pad), the salt's compressions from the salt
// prefix's midstate (y's first 16 big-endian words, then its last 8 and SHA-512's padding: the third thread recomputes the
// second's output, untaped), then b || c as 16 little-endian words: x's bytes XOR M(key) y (the key's byte tables), then c's.
__global__ void fc_hm96_tape(const uint64_t* __restrict__ inputs, int in_words, long long n, FcMid mid, const uint64_t* __restrict__ tables,
                             uint64_t* __restrict__ tape) {
    const long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= 3 * n) return;
    const int c = (int)(g / n);
    const long long s = g - c * n;
    const uint64_t* in = inputs + 2LL * in_words * s;
    FcTape t{tape, n, s, 0};
    const int out0 = 2 * in_words + 3 * FC_SHA_COMP_WORDS;
    uint64_t out[8];
    if (c == 0) {
        for (int j = 0; j < 2 * in_words; j++) t.push(in[j]);
        fc_sha_compression(t, in, in + 8, out);
        uint64_t m[8] = {};
        for (int p = 0; p < 192; p++) {
            const uint64_t* e = tables + 8 * (256LL * p + ((in[24 + p / 8] >> (8 * (p % 8))) & 0xff));
            for (int k = 0; k < 8; k++) m[k] ^= e[k];
        }
        t.k = out0;
        for (int j = 0; j < 8; j++) t.push(sx::bswap64(out[j]) ^ m[j]);
        return;
    }
    uint64_t y[24], st[8];
    for (int j = 0; j < 24; j++) y[j] = sx::bswap64(in[24 + j]);
    t.k = 2 * in_words + c * FC_SHA_COMP_WORDS;
    if (c == 1) {
        fc_sha_compression(t, mid.w, y, st);
        return;
    }
    FcNoTape nt;
    fc_sha_compression(nt, mid.w, y, st);
    uint64_t last[16] = {};
    for (int j = 0; j < 8; j++) last[j] = y[16 + j];
    last[8] = 0x80ull << 56;
    last[15] = 8 * (128 + 192);
    fc_sha_compression(t, st, last, out);
    t.k = out0 + 8;
    for (int j = 0; j < 8; j++) t.push(sx::bswap64(out[j]));
}

// one thread per slot (consecutive lanes, consecutive slots: each tape word is read coalesced) and per 8 row blocks of 32 rows:
// it writes words 8g..8g+7 of its slot's z, a and b, the words fc_unit_witness writes (rows from `useful` on are 0, and a
// slot's 2^(slot_log - 5) >= 8 words hold them; s0: the batch's first slot over every block)
__global__ void fc_sha_rows(const uint64_t* __restrict__ tape, long long n, long long s0, const uint32_t* __restrict__ plan, int useful,
                            int per_block, int pos0, int slot_log, int k_log, uint32_t* __restrict__ z_out, uint32_t* __restrict__ a_out,
                            uint32_t* __restrict__ b_out, int cache) {
    const long long s = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int g = blockIdx.x;
    // each source through a one-word cache (cache != 0): consecutive rows read consecutive bits of one tape word (an addition's
    // 64 bits), and every lane follows one plan, so the branches are uniform
    long long wa = -1, wb = -1;
    uint64_t ca = 0, cb = 0;
    auto bit = [&](uint32_t src, long long& w0, uint64_t& c) -> uint32_t {
        if (src < 2) return src;
        const long long w = (long long)((src - 2) >> 6);
        if (!cache) return (uint32_t)(tape[w * n + s] >> ((src - 2) & 63)) & 1u;
        if (w != w0) { c = tape[w * n + s]; w0 = w; }
        return (uint32_t)(c >> ((src - 2) & 63)) & 1u;
    };
    uint32_t A[8], B[8];
    for (int j = 0; j < 8; j++) {
        uint32_t x = 0, y = 0;
        const int r0 = 32 * (8 * g + j);
        for (int i = 0; i < 32 && r0 + i < useful; i++) {
            x |= bit(plan[2 * (r0 + i)], wa, ca) << i;
            y |= bit(plan[2 * (r0 + i) + 1], wb, cb) << i;
        }
        A[j] = x; B[j] = y;
    }
    const long long gs = s0 + s, blk = gs / per_block, kk = gs % per_block;
    const long long at = (blk << (k_log - 5)) + ((long long)(pos0 + kk) << (slot_log - 5)) + 8 * g;
    uint4* zo = reinterpret_cast<uint4*>(z_out + at);
    uint4* ao = reinterpret_cast<uint4*>(a_out + at);
    uint4* bo = reinterpret_cast<uint4*>(b_out + at);
    zo[0] = make_uint4(A[0] & B[0], A[1] & B[1], A[2] & B[2], A[3] & B[3]);
    zo[1] = make_uint4(A[4] & B[4], A[5] & B[5], A[6] & B[6], A[7] & B[7]);
    ao[0] = make_uint4(A[0], A[1], A[2], A[3]); ao[1] = make_uint4(A[4], A[5], A[6], A[7]);
    bo[0] = make_uint4(B[0], B[1], B[2], B[3]); bo[1] = make_uint4(B[4], B[5], B[6], B[7]);
}

// The plan's rows as runs, per 32-row word r < 8 * nrg and operand h (a 0, b 1): a run is consecutive rows of one word reading
// consecutive bits of one tape word, or rows of constant 1 (constant-0 rows are in none), as {mask of its length, first row |
// first bit << 8 | column << 16}. A column is a tape word's place in its group's list gw[goff[g], goff[g + 1]) (the tape words
// group g's 256 rows read, in first-read order), or nwc for the constant 1 (nwc: the most words a group reads). Word r's operand
// h's runs are rq[poff[2r + h], poff[2r + h + 1]), two a uint4, padded with empty runs to an even number of uint4s; block_max:
// the most uint4s 8 consecutive groups' runs take.
struct FcRuns {
    std::vector<uint32_t> poff, goff, gw;
    std::vector<uint4> rq;
    int nwc = 0;
    uint32_t block_max = 0;
};
static void fc_sha_runs(const uint32_t* plan, int useful, int nrg, FcRuns& R) {
    const uint32_t one = 0xffffu;
    R.poff.assign(1, 0); R.goff.assign(1, 0); R.gw.clear(); R.rq.clear(); R.nwc = 0; R.block_max = 0;
    std::vector<uint2> v;
    for (int g = 0; g < nrg; g++) {
        std::unordered_map<uint32_t, uint32_t> col;
        for (int r = 8 * g; r < 8 * g + 8; r++)
            for (int h = 0; h < 2; h++) {
                uint32_t prev = 0, len = 0;
                for (int i = 0; i < 32; i++) {
                    const long long row = 32LL * r + i;
                    const uint32_t src = row < useful ? plan[2 * row + h] : 0;
                    const bool joins = src == 1 ? prev == 1 : src >= 2 && prev >= 2 && src == prev + 1 && ((src - 2) & 63) != 0;
                    if (joins) {
                        len++;
                        v.back().x = len == 32 ? ~0u : (1u << len) - 1;
                    } else if (src >= 1) {
                        uint32_t c = one;
                        if (src >= 2) {
                            const auto e = col.emplace((src - 2) >> 6, (uint32_t)col.size());
                            if (e.second) R.gw.push_back((src - 2) >> 6);
                            c = e.first->second;
                        }
                        len = 1;
                        v.push_back(make_uint2(1u, (uint32_t)i | (src >= 2 ? (src - 2) & 63 : 0) << 8 | c << 16));
                    }
                    prev = src;
                }
                while (v.size() % 4) v.push_back(make_uint2(0, one << 16));
                for (size_t k = 0; k < v.size(); k += 2) R.rq.push_back(make_uint4(v[k].x, v[k].y, v[k + 1].x, v[k + 1].y));
                R.poff.push_back((uint32_t)R.rq.size());
                v.clear();
            }
        R.goff.push_back((uint32_t)R.gw.size());
        R.nwc = std::max(R.nwc, (int)col.size());
    }
    for (uint4& q : R.rq) {
        if ((q.y >> 16) == one) q.y = (q.y & 0xffffu) | (uint32_t)R.nwc << 16;
        if ((q.w >> 16) == one) q.w = (q.w & 0xffffu) | (uint32_t)R.nwc << 16;
    }
    for (int g0 = 0; g0 < nrg; g0 += 8) R.block_max = std::max(R.block_max, R.poff[16 * std::min(g0 + 8, nrg)] - R.poff[16 * g0]);
}

// Words 8g..8g+7 of one slot's a (W[0]) and b (W[1]) from fc_sha_runs's runs (rq: R.rq from uint4 `base` on) and the slot's
// column words of group g, col[c * stride]
__device__ __forceinline__ void fc_rows_words(const uint64_t* col, int stride, const uint32_t* __restrict__ poff, const uint4* rq,
                                              uint32_t base, int g, uint32_t W[2][8]) {
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int q = 2 * (8 * g + j) + h;
            uint32_t x = 0;
            for (uint32_t k = poff[q] - base, e = poff[q + 1] - base; k < e; k += 2) {
                const uint4 u = rq[k], v = rq[k + 1];
                const uint64_t c0 = col[(size_t)(u.y >> 16) * stride], c1 = col[(size_t)(u.w >> 16) * stride];
                const uint64_t c2 = col[(size_t)(v.y >> 16) * stride], c3 = col[(size_t)(v.w >> 16) * stride];
                x |= ((uint32_t)(c0 >> ((u.y >> 8) & 63)) & u.x) << (u.y & 31);
                x |= ((uint32_t)(c1 >> ((u.w >> 8) & 63)) & u.z) << (u.w & 31);
                x |= ((uint32_t)(c2 >> ((v.y >> 8) & 63)) & v.x) << (v.y & 31);
                x |= ((uint32_t)(c3 >> ((v.w >> 8) & 63)) & v.z) << (v.w & 31);
            }
            W[h][j] = x;
        }
}

// fc_sha_rows's words from the plan's runs (fc_sha_runs), a block per 8 groups (a warp a group) and 32 slots (a lane a slot):
// each group's tape words into shared memory as columns of 256 u64 (column nwc: ~0), the 8 groups' runs after `cols` columns,
// fc_rows_words, then z, a and b through a shared tile (a slot's FC_ROWS_TILE u32), so each slot's 64 words of each go out
// whole. Shared memory: cols * 2048 bytes, cols > nwc and holding the tile, then the runs (block_max uint4s).
#define FC_ROWS_TILE (3 * 64 + 4)
__global__ void __launch_bounds__(256) fc_sha_rows_runs(const uint64_t* __restrict__ tape, long long n, long long s0, int nrg, int nwc,
                                                        int cols, const uint32_t* __restrict__ poff, const uint4* __restrict__ rq,
                                                        const uint32_t* __restrict__ goff, const uint32_t* __restrict__ gw, int per_block,
                                                        int pos0, int slot_log, int k_log, uint32_t* __restrict__ z_out,
                                                        uint32_t* __restrict__ a_out, uint32_t* __restrict__ b_out) {
    extern __shared__ uint64_t fc_rows_sm[];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g0 = blockIdx.x * 8, g = g0 + w;
    const long long s = (long long)blockIdx.y * 32 + lane;
    const bool on = g < nrg && s < n;
    uint64_t* col = fc_rows_sm + threadIdx.x;
    const uint32_t base = poff[16 * g0];
    uint4* rs = reinterpret_cast<uint4*>(fc_rows_sm + (size_t)cols * 256);
    for (uint32_t k = threadIdx.x, e = poff[16 * min(g0 + 8, nrg)] - base; k < e; k += 256) rs[k] = rq[base + k];
    if (on) {
        const uint32_t w0 = goff[g], nw = goff[g + 1] - w0;
#pragma unroll 4
        for (uint32_t i = 0; i < nw; i++) __pipeline_memcpy_async(col + (size_t)i * 256, tape + (long long)gw[w0 + i] * n + s, 8);
        __pipeline_commit();
        col[(size_t)nwc * 256] = ~0ull;
        __pipeline_wait_prior(0);
    }
    __syncthreads();
    uint32_t W[2][8] = {};
    if (on) fc_rows_words(col, 256, poff, rs, base, g, W);
    __syncthreads();
    uint32_t* t = reinterpret_cast<uint32_t*>(fc_rows_sm);
    if (on) {
        uint4* r = reinterpret_cast<uint4*>(t + lane * FC_ROWS_TILE + w * 8);
        r[0] = make_uint4(W[0][0] & W[1][0], W[0][1] & W[1][1], W[0][2] & W[1][2], W[0][3] & W[1][3]);
        r[1] = make_uint4(W[0][4] & W[1][4], W[0][5] & W[1][5], W[0][6] & W[1][6], W[0][7] & W[1][7]);
        r[16] = make_uint4(W[0][0], W[0][1], W[0][2], W[0][3]); r[17] = make_uint4(W[0][4], W[0][5], W[0][6], W[0][7]);
        r[32] = make_uint4(W[1][0], W[1][1], W[1][2], W[1][3]); r[33] = make_uint4(W[1][4], W[1][5], W[1][6], W[1][7]);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < 32 * 3 * 16; c += 256) {
        const int p = c & 15, row = c >> 4, arr = row % 3, sl = row / 3;
        const long long ss = (long long)blockIdx.y * 32 + sl;
        if (ss >= n || g0 + (p >> 1) >= nrg) continue;
        const long long gs = s0 + ss, blk = gs / per_block, kk = gs % per_block;
        const long long at = (blk << (k_log - 5)) + ((long long)(pos0 + kk) << (slot_log - 5)) + 8LL * g0 + 4 * p;
        *reinterpret_cast<uint4*>((arr == 0 ? z_out : arr == 1 ? a_out : b_out) + at) =
            *reinterpret_cast<const uint4*>(t + sl * FC_ROWS_TILE + arr * 64 + 4 * p);
    }
}

// A plan's rows on the device, uploaded once a plan: the plan (for fc_sha_rows) and its runs (for fc_sha_rows_runs, whose
// shared memory is smem bytes; 0 when that is past the device's, and the rows go one bit at a time)
struct FcRowsDev {
    const uint32_t* plan_of = nullptr;
    uint32_t *plan = nullptr, *poff = nullptr, *goff = nullptr, *gw = nullptr;
    uint4* rq = nullptr;
    int useful = 0, nrg = 0, nwc = 0, cols = 0;
    size_t smem = 0;
};
static int fc_rows_load(FcRowsDev& D, const uint32_t* plan, int useful) {
    if (D.plan_of == plan) return 0;
    D.plan_of = nullptr;
    for (void* p : {(void*)D.plan, (void*)D.poff, (void*)D.goff, (void*)D.gw, (void*)D.rq}) if (p) cudaFree(p);
    D.plan = D.poff = D.goff = D.gw = nullptr; D.rq = nullptr;
    D.useful = useful; D.nrg = (useful + 255) / 256;
    FcRuns R;
    fc_sha_runs(plan, useful, D.nrg, R);
    auto up = [](void** d, const void* h, size_t bytes) -> int {
        CK(cudaMalloc(d, std::max<size_t>(bytes, 16)));
        if (bytes) CK(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice));
        return 0;
    };
    if (const int e = up((void**)&D.plan, plan, (size_t)useful * 8)) return e;
    if (const int e = up((void**)&D.poff, R.poff.data(), R.poff.size() * 4)) return e;
    if (const int e = up((void**)&D.goff, R.goff.data(), R.goff.size() * 4)) return e;
    if (const int e = up((void**)&D.gw, R.gw.data(), R.gw.size() * 4)) return e;
    if (const int e = up((void**)&D.rq, R.rq.data(), R.rq.size() * sizeof(uint4))) return e;
    D.nwc = R.nwc;
    D.cols = std::max(R.nwc + 1, (32 * FC_ROWS_TILE * 4 + 2047) / 2048);
    D.smem = (size_t)D.cols * 2048 + (size_t)R.block_max * sizeof(uint4);
    int dev = 0, optin = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (D.smem > (size_t)optin) D.smem = 0;
    else if (D.smem > 48 * 1024) CK(cudaFuncSetAttribute(fc_sha_rows_runs, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)D.smem));
    D.plan_of = plan;
    return 0;
}
// a batch's rows (nb slots from s0 in `tape`) into z, a and b: from the runs, or (bits, or the runs past the shared memory)
// from the plan one bit at a time, each tape word through a one-word cache unless cache == 0
static void fc_rows(const FcRowsDev& D, const uint64_t* tape, long long nb, long long s0, int per_block, int pos0, int slot_log,
                    int k_log, uint32_t* z, uint32_t* a, uint32_t* b, bool bits, int cache) {
    if (bits || !D.smem)
        fc_sha_rows<<<dim3((unsigned)D.nrg, (unsigned)((nb + 255) / 256)), 256>>>(tape, nb, s0, D.plan, D.useful, per_block, pos0, slot_log,
                                                                                  k_log, z, a, b, cache);
    else
        fc_sha_rows_runs<<<dim3((unsigned)((D.nrg + 7) / 8), (unsigned)((nb + 31) / 32)), 256, D.smem>>>(
            tape, nb, s0, D.nrg, D.nwc, D.cols, D.poff, D.rq, D.goff, D.gw, per_block, pos0, slot_log, k_log, z, a, b);
}

// The hm96 slots' native witness queued on stream 0 (FlockCircuitParams.m_plan): n slots' tapes from their input words (d_in, on
// the device) in batches through `tape` (tape_bytes), each batch's rows by fc_rows from the hm96 plan at the slots' places
// (per_block, pos0, slot_log, k_log, as the compression slots' are; bits: fc_rows's). The plan's runs and the key's byte tables go
// to the device once and stay for the process.
static int fc_hm96_slots(const uint64_t* d_in, int in_words, long long n, const uint32_t* plan, int useful, int tape_words,
                         const uint64_t* mid, const uint64_t* tables, uint64_t* tape, long long tape_bytes, int per_block, int pos0,
                         int slot_log, int k_log, uint32_t* z, uint32_t* a, uint32_t* b, bool bits) {
    const long long per = (long long)tape_words * 8;
    if (slot_log < 8 || 2 * in_words < 48 || tape_words != 2 * in_words + 3 * FC_SHA_COMP_WORDS + 16 || tape_bytes < per) {
        printf("FFI circuit: the hm96 slots' native witness needs slots of at least 2^8 rows, hm96's tape and room for one\n");
        return 211;
    }
    static FcRowsDev rows;
    static const uint64_t* tables_of = nullptr;
    static uint64_t* d_tables = nullptr;
    if (const int e = fc_rows_load(rows, plan, useful)) return e;
    if (tables_of != tables) {
        const size_t bytes = 192 * 256 * 8 * sizeof(uint64_t);
        tables_of = nullptr;
        if (!d_tables) CK(cudaMalloc(&d_tables, bytes));
        CK(cudaMemcpy(d_tables, tables, bytes, cudaMemcpyHostToDevice));
        tables_of = tables;
    }
    FcMid m;
    for (int k = 0; k < 8; k++) m.w[k] = mid[k];
    const long long batch = std::max(1LL, std::min(n, tape_bytes / per));
    for (long long s0 = 0; s0 < n; s0 += batch) {
        const long long nb = std::min(batch, n - s0);
        fc_hm96_tape<<<(unsigned)((3 * nb + 127) / 128), 128>>>(d_in + 2LL * s0 * in_words, in_words, nb, m, d_tables, tape);
        fc_rows(rows, tape, nb, s0, per_block, pos0, slot_log, k_log, z, a, b, bits, 1);
        CK(cudaGetLastError());
    }
    return 0;
}
#endif

// The hm96 slots' native witness alone (fc_hm96_slots, as fc_witness builds it) for n slots of 2^slot_log rows from their input
// words (host memory, [n][in_words] F128) into z, a and b (host memory, n << (slot_log - 7) F128 words each): a test vector
// against the host's slots.
extern "C" int flock_cuda_hm96_slots(const uint64_t* inputs, int in_words, long long n, const uint32_t* plan, int useful, int tape_words,
                                     const uint64_t* mid, const uint64_t* tables, int slot_log, uint64_t* z, uint64_t* a, uint64_t* b) {
#ifdef FLOCK_HB
    const size_t in_bytes = (size_t)n * in_words * 16, out_bytes = (size_t)n << (slot_log - 3), tape_bytes = (size_t)n * tape_words * 8;
    uint64_t *d_in = nullptr, *d_tape = nullptr;
    uint32_t* d_zab = nullptr;
    int e = 0;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess || cudaMalloc(&d_tape, tape_bytes) != cudaSuccess || cudaMalloc(&d_zab, 3 * out_bytes) != cudaSuccess ||
        cudaMemcpy(d_in, inputs, in_bytes, cudaMemcpyHostToDevice) != cudaSuccess || cudaMemset(d_zab, 0, 3 * out_bytes) != cudaSuccess)
        e = 1;
    if (!e) e = fc_hm96_slots(d_in, in_words, n, plan, useful, tape_words, mid, tables, d_tape, (long long)tape_bytes, (int)n, 0, slot_log,
                              slot_log, d_zab, d_zab + out_bytes / 4, d_zab + 2 * (out_bytes / 4), false);
    uint64_t* out[3] = {z, a, b};
    for (int k = 0; k < 3 && !e; k++)
        if (cudaMemcpy(out[k], (const char*)d_zab + k * out_bytes, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) e = 1;
    if (e == 1) cudaGetLastError();
    cudaFree(d_in); cudaFree(d_tape); cudaFree(d_zab);
    return e;
#else
    (void)inputs; (void)in_words; (void)n; (void)plan; (void)useful; (void)tape_words; (void)mid; (void)tables; (void)slot_log;
    (void)z; (void)a; (void)b;
    return 208;
#endif
}

// Page-locked, mapped host memory (null when the runtime refuses), held for the process: the device reads host slots from it
// directly, beside the compression witness, instead of three pageable copies per slot after it.
extern "C" void* flock_cuda_host_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    return p;
}

// Host memory page-locked until unregistered (gpu_circuit.rs Pinned: hm96 salts read ahead, which launch_merkle512 copies
// without the host waiting); 0 when the runtime refuses
extern "C" int flock_cuda_host_register(void* p, size_t bytes) {
    if (cudaHostRegister(p, bytes, cudaHostRegisterPortable) != cudaSuccess) { cudaGetLastError(); return 0; }
    return 1;
}

extern "C" void flock_cuda_host_unregister(void* p) {
    if (cudaHostUnregister(p) != cudaSuccess) cudaGetLastError();
}

// device memory for a witness's host words copied ahead of its session (gpu_circuit.rs DevBuf), and a copy either way on a
// stream of its own, beside the kernels of the session proving meanwhile
extern "C" void* flock_cuda_dev_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    return p;
}

// The copy goes in pieces of FC_COPY_PIECE_MB (default 8; 0: one piece), each synced before the next: the session proving
// meanwhile waits on the whole device often (Ligerito's transcript syncs, cudaFree), and each such wait then covers one piece
extern "C" int flock_cuda_copy(void* dst, const void* src, size_t bytes) {
    static const size_t piece = [] {
        const char* e = std::getenv("FC_COPY_PIECE_MB");
        const long v = e ? atol(e) : 8;
        return v > 0 ? (size_t)v << 20 : (size_t)0;
    }();
    cudaStream_t s;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) { cudaGetLastError(); return 1; }
    const size_t step = piece ? piece : bytes;
    cudaError_t e = cudaSuccess;
    for (size_t o = 0; o < bytes && e == cudaSuccess; o += step) {
        e = cudaMemcpyAsync((char*)dst + o, (const char*)src + o, std::min(step, bytes - o), cudaMemcpyDefault, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    }
    cudaStreamDestroy(s);
    if (e != cudaSuccess) { cudaGetLastError(); return 1; }
    return 0;
}

// zeroes a DevBuf (gpu_circuit.rs) before the pool takes it back, on a stream of its own and waited for, as flock_cuda_copy
extern "C" int flock_cuda_dev_zero(void* p, size_t bytes) {
    cudaStream_t s;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) { cudaGetLastError(); return 1; }
    cudaError_t e = cudaMemsetAsync(p, 0, bytes, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (e != cudaSuccess) { cudaGetLastError(); return 1; }
    return 0;
}

// A circuit's arrays on the device for the process, keyed by the generation gpu_circuit.rs `types` drew for that circuit's build
// (FlockCircuitParams.types_gen; no two builds share one) and a set: 0, every slot type's CSC (per type A's column pointers and
// rows, then B's); 1, the unit's rows as CSR and its level schedule (fc_witness's six arrays). A set is copied from the host the
// first time its key is asked for, as flock_cuda_copy copies (beside a session proving meanwhile), and its host arrays are never
// read again, so the host frees them once the set is here and passes null. A set not here whose host arrays are null is refused.
// A set asked for with a key (a digest of its arrays) is one device copy shared by every process on this GPU that asks for that
// key (fc_share_get, csc_share.cuh), and a copy of the process's own where it can't be. The arrays the asker marks in `host`
// (bit i, array i) stay where they are instead, page-locked and mapped, and the device reads them there for the process: the host
// must keep them, unmoved, for as long. An array the runtime won't lock is copied.
struct FcTypeUploads {
    std::mutex mu;
    std::vector<std::pair<std::pair<unsigned long long, int>, std::vector<uint32_t*>>> held;
    bool has(unsigned long long gen, int set) {
        std::lock_guard<std::mutex> lock(mu);
        for (const auto& e : held)
            if (e.first == std::make_pair(gen, set)) return true;
        return false;
    }
    cudaError_t get(unsigned long long gen, int set, const uint32_t* const* h, const size_t* len, size_t n, uint32_t** d,
                    unsigned long long host = 0, const uint8_t* key = nullptr) {
        std::lock_guard<std::mutex> lock(mu);
        for (const auto& e : held)
            if (e.first == std::make_pair(gen, set)) {
                if (e.second.size() != n) return cudaErrorInvalidValue;
                std::copy(e.second.begin(), e.second.end(), d);
                return cudaSuccess;
            }
        for (size_t i = 0; i < n; i++)
            if (len[i] && !h[i]) return cudaErrorInvalidValue;
        std::vector<uint32_t*> got(n, nullptr);
        if (key && !host && fc_share_get(key, h, len, n, got.data()) == 0) {
            held.emplace_back(std::make_pair(gen, set), got);
            std::copy(got.begin(), got.end(), d);
            return cudaSuccess;
        }
        std::vector<char> locked(n, 0);
        auto undo = [&] {
            for (size_t i = 0; i < n; i++)
                if (locked[i]) cudaHostUnregister(const_cast<uint32_t*>(h[i]));
                else if (got[i]) cudaFree(got[i]);
        };
        for (size_t i = 0; i < n; i++) {
            if (i < 64 && (host >> i) & 1 && len[i]) {
                void* hp = const_cast<uint32_t*>(h[i]);
                void* dp = nullptr;
                cudaError_t r = cudaHostRegister(hp, len[i] * sizeof(uint32_t), cudaHostRegisterMapped | cudaHostRegisterPortable);
                if (r == cudaSuccess) {
                    if ((r = cudaHostGetDevicePointer(&dp, hp, 0)) == cudaSuccess) {
                        got[i] = (uint32_t*)dp;
                        locked[i] = 1;
                        continue;
                    }
                    cudaHostUnregister(hp);
                }
                cudaGetLastError();
                fprintf(stderr, "FFI circuit: a slot type's rows (%zu words) not page-locked (%s); copied to the device\n", len[i],
                        cudaGetErrorString(r));
            }
            cudaError_t err = cudaMalloc(&got[i], (len[i] ? len[i] : 1) * sizeof(uint32_t));
            if (err == cudaSuccess && len[i] && flock_cuda_copy(got[i], h[i], len[i] * sizeof(uint32_t))) err = cudaErrorUnknown;
            if (err != cudaSuccess) { undo(); return err; }
        }
        held.emplace_back(std::make_pair(gen, set), got);
        std::copy(got.begin(), got.end(), d);
        return cudaSuccess;
    }
};
static FcTypeUploads fc_type_uploads;

// set `set` of generation `gen` onto the device now (FcTypeUploads) from its n host arrays of len[i] words, those marked in
// `host` read in place, shared with this GPU's other processes under `key` (64 bytes; null: this process's own): 0 once it is
// there
extern "C" int flock_cuda_types_upload(unsigned long long gen, int set, const uint32_t* const* h, const size_t* len, size_t n,
                                       unsigned long long host, const uint8_t* key) {
    std::vector<uint32_t*> d(n);
    if (fc_type_uploads.get(gen, set, h, len, n, d.data(), host, key) != cudaSuccess) { cudaGetLastError(); return 1; }
    return 0;
}

extern "C" int flock_cuda_types_held(unsigned long long gen, int set) { return fc_type_uploads.has(gen, set); }

static bool fc_on_device(const void* p) {
    cudaPointerAttributes a;
    if (!p || cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return false; }
    return a.type == cudaMemoryTypeDevice;
}

// the witness's device scratch is zeroed before it's freed, so no value outlives its prove call in device memory (cudaFree
// waits for the memset)
static cudaError_t fc_wipe_free(void* p, size_t bytes) {
    const cudaError_t e = cudaMemsetAsync(p, 0, bytes);
    cudaFree(p);
    return e;
}

// the host slots from mapped host memory: slot i's F128 words at at[i], from zab + off[i] u64 (ab: its a then b words, z = a & b;
// else z, a, b); blocks stride over the slots, a block's threads over one slot's words
__global__ void fc_host_slots(const uint64_t* __restrict__ zab, const long long* __restrict__ at, const long long* __restrict__ off,
                              const int* __restrict__ words, long long n, int ab, uint4* __restrict__ z, uint4* __restrict__ a,
                              uint4* __restrict__ b) {
    for (long long i = blockIdx.x; i < n; i += gridDim.x) {
        const int w = words[i];
        const uint4* src = reinterpret_cast<const uint4*>(zab + off[i]);
        uint4 *zd = z + at[i], *ad = a + at[i], *bd = b + at[i];
        for (int k = threadIdx.x; k < w; k += blockDim.x) {
            if (ab) {
                const uint4 x = src[k], y = src[w + k];
                ad[k] = x; bd[k] = y;
                zd[k] = make_uint4(x.x & y.x, x.y & y.y, x.z & y.z, x.w & y.w);
            } else {
                zd[k] = src[k]; ad[k] = src[w + k]; bd[k] = src[2 * w + k];
            }
        }
    }
}

// z = a & b over the host slots' words (slot i's at at[i]) once their a, b words are in place
__global__ void fc_host_slots_and(const long long* __restrict__ at, const int* __restrict__ words, long long n, uint4* __restrict__ z,
                                  const uint4* __restrict__ a, const uint4* __restrict__ b) {
    for (long long i = blockIdx.x; i < n; i += gridDim.x) {
        const int w = words[i];
        const long long o = at[i];
        for (int k = threadIdx.x; k < w; k += blockDim.x) {
            const uint4 x = a[o + k], y = b[o + k];
            z[o + k] = make_uint4(x.x & y.x, x.y & y.y, x.z & y.z, x.w & y.w);
        }
    }
}

static int fc_bad_shape(const FlockCudaProveParams* P, const FlockCircuitParams* N) {
    const int k_log = P->k_log;
    if (k_log < 20 || k_log > 27 || N->n_types < 1 || N->n_types > FC_MAX_TYPES || N->n_ranges < 1 || N->n_ranges > FC_MAX_RANGES) return 1;
    for (int j = 0; j < N->n_ranges; j++) {
        const int t = N->r_type[j];
        if (t < 0 || t >= N->n_types || ((long long)(N->r_pos0[j] + N->r_count[j]) << N->type_log[t]) > (1LL << k_log)) return 1;
    }
    return 0;
}

static size_t fc_arena_of(const FlockCudaProveParams* P, const FlockCircuitParams* N) {
    long long fold_words = 0;
    for (int t = 0; t < N->n_types; t++) fold_words += 1LL << N->type_log[t];
    const FcArenaShape s = {P->m, P->k_log, N->c.sub_log, P->initial_k, P->recursive_steps, P->log_inv_rates, P->recursive_ks,
                            P->queries, FLOCK_HB, N->zk_on, N->zk_on ? N->zk_t_pad : 0, fold_words};
    return fc_arena_bytes(&s);
}

// chunk_zlin_transpose's former kernel, one byte a thread per store (the same bytes): FC_ZLIN_BYTEWISE=1 times it in the same job
__global__ void fc_zlin_transpose_bytewise(const F128* __restrict__ z, int k_log, long long n_stripes, uint8_t* __restrict__ out) {
    const long long wpr = (1LL << k_log) / 128;
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

static void fc_zlin_transpose(const F128* df, int k_log, long long n_stripes, uint8_t* d_zlin) {
    static const bool bytewise = [] { const char* e = std::getenv("FC_ZLIN_BYTEWISE"); return e && e[0] == '1'; }();
    if (bytewise) fc_zlin_transpose_bytewise<<<4096, 256>>>(df, k_log, n_stripes, d_zlin);
    else chunk_zlin_transpose<<<4096, 256>>>(df, k_log, n_stripes, d_zlin);
}

static int fc_witness(const FlockCudaProveParams* P, const FlockCircuitParams* N, int m, int k_log, long long len, F128* df, F128* d_a,
                      F128* d_b) {
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
            static FcUploads h_up;
            uint32_t* hd[6];
            {
                const uint32_t* h[6] = {N->h_a_off, N->h_b_off, N->h_a_col, N->h_b_col, N->h_level_off, N->h_order};
                const size_t len[6] = {(size_t)N->h_useful + 1, (size_t)N->h_useful + 1, N->h_a_off[N->h_useful],
                                       N->h_b_off[N->h_useful], (size_t)N->h_levels + 1, N->h_level_off[N->h_levels]};
                CK(h_up.get(h, len, 6, hd));
            }
            uint32_t *h_ao = hd[0], *h_bo = hd[1], *h_ac = hd[2], *h_bc = hd[3], *h_lo = hd[4], *h_ord = hd[5];
            const int hr = N->h_range;
            const int hl = N->type_log[N->r_type[hr]];
            const long long hper = N->r_count[hr];
            const long long n_h = (1LL << (m - k_log)) * hper;
            const long long hg = (n_h + 31) / 32;
            static cudaStream_t hs_stream = nullptr;
            static cudaEvent_t hs_zeroed = nullptr, hs_done = nullptr;
            static int n_sm = 0, hs_max_pitch = 0;
            if (N->n_host0 && !hs_stream) {
                CK(cudaStreamCreateWithFlags(&hs_stream, cudaStreamNonBlocking));
                CK(cudaEventCreateWithFlags(&hs_zeroed, cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&hs_done, cudaEventDisableTiming));
                CK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0));
                CK(cudaDeviceGetAttribute(&hs_max_pitch, cudaDevAttrMaxPitch, 0));
            }
            // the input rows, and the host slots below, may be the device's copies already (copied while the witness was built)
            const bool hin_dev = fc_on_device(N->h_inputs);
            uint64_t* d_hin = (uint64_t*)N->h_inputs; uint32_t* d_hs = nullptr; size_t hs_bytes = 0;
            if (!hin_dev) {
                CK(cudaMalloc(&d_hin, (size_t)n_h * N->h_in_words * 16));
                CK(cudaMemcpy(d_hin, N->h_inputs, (size_t)n_h * N->h_in_words * 16, cudaMemcpyHostToDevice));
            }
            // the hm96 slots' input words, copied before any kernel is queued (a pageable copy waits for the stream's work)
            const long long n_m = N->m_plan ? (1LL << (m - k_log)) * N->r_count[N->m_range] : 0;
            uint64_t* d_min = nullptr;
            if (N->m_plan) {
                if (!N->h_plan) { printf("FFI circuit: the hm96 slots' native witness needs the compressions' native witness\n"); return 211; }
                CK(cudaMalloc(&d_min, (size_t)n_m * N->m_in_words * 16));
                CK(cudaMemcpy(d_min, N->m_inputs, (size_t)n_m * N->m_in_words * 16, cudaMemcpyHostToDevice));
            }
            // The host slots, on a stream of their own beside the compressions (other words of z, a, b), once the memsets are done.
            // Mapped ones go to their words by the copy engine (host_copies, issued once the compressions are queued): no SM waits
            // on PCIe reads, which slow the compression kernels beside them several times over, and the compression rows' copy
            // doesn't wait behind them. FC_HS_MAPPED=1 has fc_host_slots read them from mapped memory instead, for a same-job
            // control. Device ones fc_host_slots reads in place; pageable ones take three copies each after the compressions.
            static const bool hs_mapped = [] { const char* e = std::getenv("FC_HS_MAPPED"); return e && e[0] == '1'; }();
            const uint64_t* zab_dev = nullptr;
            bool hs_copy = false;
            if (N->n_host0) {
                if (fc_on_device(N->host0_zab)) {
                    zab_dev = N->host0_zab;
                } else if (cudaHostGetDevicePointer((void**)&zab_dev, (void*)N->host0_zab, 0) != cudaSuccess) {
                    cudaGetLastError();
                    zab_dev = nullptr;
                } else {
                    hs_copy = !hs_mapped;
                }
            }
            if (N->n_host0 && N->host0_ab && !zab_dev) {
                printf("FFI circuit: host slots in the a, b layout need mapped host memory\n"); return 209;
            }
            long long *d_hat = nullptr, *d_hoff = nullptr; int* d_hw = nullptr;
            auto host_slots = [&]() -> int {
                if (!zab_dev) return 0;
                const long long n = N->n_host0;
                std::vector<long long> off((size_t)n);
                for (long long i = 0, o = 0; i < n; i++) { off[i] = o; o += (N->host0_ab ? 4LL : 6LL) * N->host0_words[i]; }
                CK(cudaMalloc(&d_hat, n * sizeof(long long))); CK(cudaMalloc(&d_hoff, n * sizeof(long long)));
                CK(cudaMalloc(&d_hw, n * sizeof(int)));
                CK(cudaMemcpy(d_hat, N->host0_at, n * sizeof(long long), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_hoff, off.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_hw, N->host0_words, n * sizeof(int), cudaMemcpyHostToDevice));
                CK(cudaEventRecord(hs_zeroed, 0));
                CK(cudaStreamWaitEvent(hs_stream, hs_zeroed, 0));
                if (hs_copy) return 0;
                fc_host_slots<<<(unsigned)std::min(n, 2LL * n_sm), 256, 0, hs_stream>>>(zab_dev, d_hat, d_hoff, d_hw, n, N->host0_ab,
                    (uint4*)df, (uint4*)d_a, (uint4*)d_b);
                CK(cudaGetLastError());
                CK(cudaEventRecord(hs_done, hs_stream));
                return 0;
            };
            // A copy a group of slots of one width, each of its z, a, b words' (a, b in the a, b layout, then fc_host_slots_and
            // forms z): a run of slots at one stride on the device, or runs like it each at one stride from the last (a slot
            // range's slots in each block), as a 3D copy, a 2D copy or a copy a slot, as the strides and pitches allow.
            auto host_copies = [&]() -> int {
                if (!hs_copy) return 0;
                const long long n = N->n_host0, per = N->host0_ab ? 4 : 6;
                const long long* at = N->host0_at;
                const int* words = N->host0_words;
                F128* dst[3] = {df, d_a, d_b};
                const uint64_t* src = N->host0_zab;
                // the end of the run from slot a, its stride in *st (its width for a run of one)
                auto run_end = [&](long long a, long long* st) {
                    long long b = a + 1;
                    *st = words[a];
                    if (b < n && words[b] == words[a] && at[b] - at[a] >= words[a]) {
                        *st = at[b] - at[a];
                        while (b < n && words[b] == words[a] && at[b] - at[b - 1] == *st) b++;
                    }
                    return b;
                };
                for (long long i0 = 0; i0 < n;) {
                    const long long w = words[i0];
                    long long stride;
                    const long long i1 = run_end(i0, &stride), len = i1 - i0;
                    long long depth = 1, outer = 0;
                    if (len > 1 && i1 < n && words[i1] == w && (at[i1] - at[i0]) % stride == 0 && at[i1] - at[i0] >= len * stride) {
                        outer = at[i1] - at[i0];
                        for (long long j = i1, st; j < n && words[j] == w && at[j] - at[j - len] == outer && run_end(j, &st) - j == len &&
                                                   st == stride;
                             j += len)
                            depth++;
                    }
                    const bool pitched = stride * 16 <= hs_max_pitch && per * w * 8 <= hs_max_pitch;
                    for (int k = N->host0_ab ? 1 : 0; k < 3; k++) {
                        const uint64_t* s = src + 2 * w * (k - (N->host0_ab ? 1 : 0));
                        if (pitched && depth > 1) {
                            cudaMemcpy3DParms p = {};
                            p.srcPtr = make_cudaPitchedPtr((void*)s, (size_t)(per * w * 8), (size_t)w * 16, (size_t)len);
                            p.dstPtr = make_cudaPitchedPtr((void*)(dst[k] + at[i0]), (size_t)stride * 16, (size_t)w * 16, (size_t)(outer / stride));
                            p.extent = make_cudaExtent((size_t)w * 16, (size_t)len, (size_t)depth);
                            p.kind = cudaMemcpyHostToDevice;
                            CK(cudaMemcpy3DAsync(&p, hs_stream));
                        } else if (pitched && len > 1) {
                            CK(cudaMemcpy2DAsync(dst[k] + at[i0], (size_t)stride * 16, s, (size_t)(per * w * 8), (size_t)w * 16,
                                                 (size_t)len, cudaMemcpyHostToDevice, hs_stream));
                        } else {
                            for (long long i = i0; i < i0 + len * depth; i++)
                                CK(cudaMemcpyAsync(dst[k] + at[i], s + per * w * (i - i0), (size_t)w * 16, cudaMemcpyHostToDevice, hs_stream));
                        }
                    }
                    src += per * w * len * depth;
                    i0 += len * depth;
                }
                if (N->host0_ab) {
                    fc_host_slots_and<<<(unsigned)std::min(n, 2LL * n_sm), 256, 0, hs_stream>>>(d_hat, d_hw, n, (uint4*)df, (const uint4*)d_a,
                        (const uint4*)d_b);
                    CK(cudaGetLastError());
                }
                CK(cudaEventRecord(hs_done, hs_stream));
                return 0;
            };
            if (N->h_plan) {
#ifdef FLOCK_HB
                static FcRowsDev rows;
                if (const int e = fc_rows_load(rows, N->h_plan, N->h_useful)) return e;
                if (hl < 8) { printf("FFI circuit: the native compression witness needs slots of at least 2^8 rows\n"); return 210; }
                // a thread per compression where the tape has sha512x3's layout (FC_SHA_TAPE1=1: a thread per slot)
                static const bool tape1 = [] { const char* e = std::getenv("FC_SHA_TAPE1"); return e && e[0] == '1'; }();
                const bool tape3 = !tape1 && 2 * N->h_in_words >= 3 * 24 && N->h_tape_words == 2 * N->h_in_words + 3 * FC_SHA_COMP_WORDS + 24;
                // the rows from the plan's runs; FC_SHA_ROWS_BITS=1: from its rows one bit at a time, the hm96 slots' too
                // (FC_SHA_ROWS_NOCACHE=1: every row's tape bits read straight from the tape)
                static const bool rows_bits = [] { const char* e = std::getenv("FC_SHA_ROWS_BITS"); return e && e[0] == '1'; }();
                static const int rows_cache = [] { const char* e = std::getenv("FC_SHA_ROWS_NOCACHE"); return !(e && e[0] == '1'); }();
                const long long per = (long long)N->h_tape_words * 8;
                const long long batch = std::max(1LL, std::min(n_h, FC_WITNESS_BATCH / per));
                uint64_t* d_tape;
                CK(cudaMalloc(&d_tape, (size_t)(batch * per)));
                if (const int e = host_slots()) return e;
                for (long long s0 = 0; s0 < n_h; s0 += batch) {
                    const long long nb = std::min(batch, n_h - s0);
                    if (tape3) fc_sha_tape3<<<(unsigned)((3 * nb + 127) / 128), 128>>>(d_hin + 2LL * s0 * N->h_in_words, N->h_in_words, nb, d_tape);
                    else fc_sha_tape<<<(unsigned)((nb + 127) / 128), 128>>>(d_hin + 2LL * s0 * N->h_in_words, N->h_in_words, nb, d_tape);
                    fc_rows(rows, d_tape, nb, s0, (int)hper, N->r_pos0[hr], hl, k_log, (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b, rows_bits,
                            rows_cache);
                    CK(cudaGetLastError());
                }
                if (N->m_plan) {
                    const int mr = N->m_range;
                    if (const int e = fc_hm96_slots(d_min, N->m_in_words, n_m, N->m_plan, N->m_useful, N->m_tape_words, N->m_mid, N->m_tables, d_tape,
                                                    batch * per, N->r_count[mr], N->r_pos0[mr], N->type_log[N->r_type[mr]], k_log, (uint32_t*)df,
                                                    (uint32_t*)d_a, (uint32_t*)d_b, rows_bits))
                        return e;
                }
                if (const int e = host_copies()) return e;
                CK(cudaStreamSynchronize(0));
                CK(fc_wipe_free(d_tape, (size_t)(batch * per)));
                d_hs = nullptr;
#else
                printf("FFI circuit: the native compression witness needs the SHA-512 build\n"); return 208;
#endif
            } else {
                const long long per = 3LL * N->h_useful * 4;
                const long long batch = std::max(1LL, std::min(hg, FC_WITNESS_BATCH / per));
                hs_bytes = (size_t)(batch * per);
                CK(cudaMalloc(&d_hs, hs_bytes));
                if (const int e = host_slots()) return e;
                for (long long g0 = 0; g0 < hg; g0 += batch) {
                    const long long ng = std::min(batch, hg - g0);
                    CK(cudaMemsetAsync(d_hs, 0, (size_t)(ng * per)));
                    fc_unit_witness<<<(unsigned)ng, 32 * 8>>>(d_hin + 2LL * 32 * g0 * N->h_in_words, N->h_in_words, n_h - 32 * g0, (int)hper,
                        N->r_pos0[hr] + 0, hl, k_log, h_ao, h_ac, h_bo, h_bc, N->h_useful, N->h_const, N->h_levels, h_lo, h_ord, d_hs,
                        (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b, 2, 0, N->h_levels, 0, nullptr, nullptr, 0, nullptr, nullptr, g0);
                    CK(cudaGetLastError());
                }
                if (const int e = host_copies()) return e;
            }
            CK(cudaStreamSynchronize(0));
            if (d_hs) CK(fc_wipe_free(d_hs, hs_bytes));
            if (!hin_dev) CK(fc_wipe_free(d_hin, (size_t)n_h * N->h_in_words * 16));
            if (d_min) CK(fc_wipe_free(d_min, (size_t)n_m * N->m_in_words * 16));
            if (zab_dev) {
                CK(cudaEventSynchronize(hs_done));
                cudaFree(d_hat); cudaFree(d_hoff); cudaFree(d_hw);
            } else {
                const uint64_t* p = N->host0_zab;
                for (long long i = 0; i < N->n_host0; i++) {
                    const size_t w = (size_t)N->host0_words[i];
                    CK(cudaMemcpyAsync(df + N->host0_at[i], p, w * 16, cudaMemcpyHostToDevice));
                    CK(cudaMemcpyAsync(d_a + N->host0_at[i], p + 2 * w, w * 16, cudaMemcpyHostToDevice));
                    CK(cudaMemcpyAsync(d_b + N->host0_at[i], p + 4 * w, w * 16, cudaMemcpyHostToDevice));
                    p += 6 * w;
                }
            }
            CK(cudaStreamSynchronize(0));
        }
        auto t1 = std::chrono::steady_clock::now();
        if (!N->u_host) {
            static FcUploads net_up;
            uint32_t* ud[6];
            {
                const uint32_t* h[6] = {N->u_a_off, N->u_b_off, N->u_a_col, N->u_b_col, N->u_level_off, N->u_order};
                const bool rows = N->u_a_off && N->u_b_off;
                const size_t len[6] = {(size_t)N->u_useful + 1, (size_t)N->u_useful + 1, rows ? N->u_a_off[N->u_useful] : 0,
                                       rows ? N->u_b_off[N->u_useful] : 0, (size_t)N->u_levels + 1, N->u_level_off[N->u_levels]};
                CK(N->types_gen ? fc_type_uploads.get(N->types_gen, 1, h, len, 6, ud) : net_up.get(h, len, 6, ud));
            }
            uint32_t *d_ao = ud[0], *d_bo = ud[1], *d_ac = ud[2], *d_bc = ud[3], *d_lo = ud[4], *d_ord = ud[5];
            const int ur = N->u_range;
            const int ul = N->type_log[N->r_type[ur]];
            const long long per_block = N->r_count[ur];
            const long long n_units = (1LL << (m - k_log)) * per_block;
            const long long groups = (n_units + 31) / 32;
            uint64_t* d_in; uint32_t* d_scratch; size_t scratch_bytes;
            CK(cudaMalloc(&d_in, (size_t)n_units * N->u_in_words * 16));
            CK(cudaMemcpy(d_in, N->u_inputs, (size_t)n_units * N->u_in_words * 16, cudaMemcpyHostToDevice));
            // groups in batches that keep the scratch under FC_WITNESS_BATCH
            const long long per = 3LL * N->u_useful * 4;
            const int warps = N->u_warps < 1 ? 1 : (N->u_warps > 8 ? 8 : N->u_warps);
            if (N->u_n_late == 0) {
                const long long batch = std::max(1LL, std::min(groups, FC_WITNESS_BATCH / per));
                scratch_bytes = (size_t)(batch * per);
                CK(cudaMalloc(&d_scratch, scratch_bytes));
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
                scratch_bytes = (size_t)(groups * per);
                CK(cudaMalloc(&d_scratch, scratch_bytes)); CK(cudaMemsetAsync(d_scratch, 0, scratch_bytes));
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
                cudaFree(d_lr); cudaFree(d_cr);
                CK(fc_wipe_free(d_late, (size_t)groups * N->u_n_late * 4)); CK(fc_wipe_free(d_cut, (size_t)groups * N->u_n_cut * 4));
            }
            CK(cudaStreamSynchronize(0));
            CK(fc_wipe_free(d_scratch, scratch_bytes)); CK(fc_wipe_free(d_in, (size_t)n_units * N->u_in_words * 16));
        }
        auto t2 = std::chrono::steady_clock::now();
        if (N->witness_s) {
            N->witness_s[0] = std::chrono::duration<double>(t1 - t0).count();
            N->witness_s[1] = std::chrono::duration<double>(t2 - t1).count();
        }
    }
    CK(cudaStreamSynchronize(0));
    return 0;
}

static int fc_fold(const FlockCudaProveParams* P, const FlockCircuitParams* N, const F128* d_eq, F128 alpha, int k_log, F128* d_comb) {
    static FcUploads up;
    uint32_t* d_csc[FC_MAX_TYPES][4];
    {
        const uint32_t* h[FC_MAX_TYPES * 4];
        size_t len[FC_MAX_TYPES * 4];
        bool any_comp = false;
        for (int t = 0; t < N->n_types; t++) {
            const size_t cols = (size_t)1 << N->type_log[t];
            const bool comp = N->t_a_col_ptr[t] == nullptr;
            any_comp |= comp;
            h[4 * t] = comp ? P->a_col_ptr : N->t_a_col_ptr[t]; len[4 * t] = cols + 1;
            h[4 * t + 1] = comp ? P->a_rows : N->t_a_rows[t]; len[4 * t + 1] = (size_t)(comp ? P->a_nnz : N->t_a_nnz[t]);
            h[4 * t + 2] = comp ? P->b_col_ptr : N->t_b_col_ptr[t]; len[4 * t + 2] = cols + 1;
            h[4 * t + 3] = comp ? P->b_rows : N->t_b_rows[t]; len[4 * t + 3] = (size_t)(comp ? P->b_nnz : N->t_b_nnz[t]);
        }
        // the compression circuit's CSC is P's (gpu.rs statics_for), which no generation covers
        CK(N->types_gen && !any_comp ? fc_type_uploads.get(N->types_gen, 0, h, len, 4 * (size_t)N->n_types, &d_csc[0][0])
                                     : up.get(h, len, 4 * (size_t)N->n_types, &d_csc[0][0]));
    }
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
    for (int t = 0; t < N->n_types; t++) ffi_free(d_base[t]);
    return 0;
}

// ---- M1: level 0's commitment with its hiding, and the zk section of the proof stream ----------------------------------

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

// Leaves of `stride` lanes: `width` lanes' codewords (src, position-major) as lanes [off, off + width) of each leaf.
__global__ void fc_zk_interleave(F128* __restrict__ out, const F128* __restrict__ src, int width, int off, int stride, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    const long long pos = i / width;
    out[pos * stride + off + (i - pos * width)] = src[i];
}

// commit_zk on the device. The padding's upper half is not zero, so the transform runs every layer (M0's commit skips layer 0
// by reading the message replicated); the witness's lanes go in four groups and the extra ones in a batch of their own (the
// NTT's lane tiles are powers of two), each interleaved into the leaves as it's done, so only a quarter of the witness's lane
// codewords is ever held beside the leaves. The pads and extra lanes stay on the device for the lane phase (the caller frees
// them).
static int fc_commit_l0_zk(const FlockCudaProveParams* P, const FlockCircuitParams* N, int log_n, const F128* df, F128** d_pads,
                           F128** d_extra, F128** d_cw0, uint8_t** d_tree0, long long* l0_bl, int* l0_ni, std::vector<MHash>& l0_cap) {
    const int k = P->initial_k, w = 1 << k, t_pad = N->zk_t_pad;
    const int k_code = (log_n - k) + P->log_inv_rates[0];
    const long long n = 1LL << k_code, l = 1LL << (log_n - k);
    if (t_pad <= 0 || t_pad > l) { printf("FFI circuit: zk t_pad %d outside (0, %lld]\n", t_pad, l); return 204; }
    CK(ffi_malloc(d_pads, (size_t)(w + 4) * t_pad * sizeof(F128)));
    CK(ffi_malloc(d_extra, (size_t)4 * l * sizeof(F128)));
    // queued in stream order: the caller's words (page-locked, gpu_circuit.rs Words) outlive the proof's last read, which waits
    // on the stream
    CK(cudaMemcpyAsync(*d_pads, N->zk_pads, (size_t)(w + 4) * t_pad * sizeof(F128), cudaMemcpyHostToDevice, 0));
    CK(cudaMemcpyAsync(*d_extra, N->zk_extra, (size_t)4 * l * sizeof(F128), cudaMemcpyHostToDevice, 0));
    const TwiddleTable* tt;
    F128* d_tw;
    CK(get_cached_twiddles(k_code, tt, d_tw));
    CK(ffi_malloc(d_cw0, (size_t)n * (w + 4) * sizeof(F128)));
    CK(ffi_malloc(d_tree0, (size_t)(2 * n - 1) * FLOCK_HB));
    const int g_log = k < 2 ? k : 2, wg = w >> g_log;
    F128 *cw_w, *cw_x;
    CK(ffi_malloc(&cw_w, (size_t)n * wg * sizeof(F128)));
    CK(ffi_malloc(&cw_x, (size_t)n * 4 * sizeof(F128)));
    for (int g = 0; g < w; g += wg) {
        fc_zk_lane_input<<<(unsigned)((n * wg + 255) / 256), 256>>>(cw_w, df + g, w, 1, k - g_log, n, l, *d_pads + (size_t)g * t_pad, t_pad);
        launch_ntt(cw_w, d_tw, *tt, 0, k_code, wg);
        fc_zk_interleave<<<(unsigned)((n * wg + 255) / 256), 256>>>(*d_cw0, cw_w, wg, g, w + 4, n);
        CK(cudaGetLastError());
    }
    fc_zk_lane_input<<<(unsigned)((n * 4 + 255) / 256), 256>>>(cw_x, *d_extra, 1, l, 2, n, l, *d_pads + (size_t)w * t_pad, t_pad);
    launch_ntt(cw_x, d_tw, *tt, 0, k_code, 4);
    fc_zk_interleave<<<(unsigned)((n * 4 + 255) / 256), 256>>>(*d_cw0, cw_x, 4, w, w + 4, n);
    CK(cudaGetLastError());
    launch_merkle((const uint8_t*)*d_cw0, *d_tree0, n, (w + 4) * 16);
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
