// verity flock-circuit: Flock-CUDA for verity/flock-circuit (backends/flock/live/src/circuit.rs). Included after
// prove_chunk.cuh at the end of cuda-ghash/prove_ffi.cu by backends/flock/cuda_circuit_patch.py, which also derives
// flock_cuda_prove_circuit from flock_cuda_prove_chunk's body: the same commit, zerocheck, lincheck, ring switch and
// Ligerito, with this file's witness (fc_witness) and lincheck fold (fc_fold) in place of the chunk statement's.
//
// A block holds ranges of slots of a few types (compressions, units, the mask slot), each type's circuit a CSC of
// 2^type_log columns. The witness is built on the device: the compressions from their host-given inputs (the BLAKE3 witness
// kernel, then scattered to their slots), the units by fc_unit_witness (32 units per CTA, bit-sliced: a row is one u32
// over the 32 units; rows in level order, several warps per level; then a 32x32 bit transpose per 32 rows into each
// unit's slot), the mask slot's free cells left zero (the no-op mask hook). With host_z the whole witness comes from the
// host instead (a tampered witness).

#define FC_MAX_TYPES 8
#define FC_MAX_RANGES 32

typedef struct {
    FlockChunkParams c;                // link, delta, gamma, phase, host witness (c.host_z / host_a / host_b)
    int n_types;
    int type_log[FC_MAX_TYPES];
    // per type, its circuit as CSC over 2^type_log columns; a NULL column pointer is the compression circuit (P's CSC)
    const uint32_t* t_a_col_ptr[FC_MAX_TYPES]; const uint32_t* t_a_rows[FC_MAX_TYPES]; uint32_t t_a_nnz[FC_MAX_TYPES];
    const uint32_t* t_b_col_ptr[FC_MAX_TYPES]; const uint32_t* t_b_rows[FC_MAX_TYPES]; uint32_t t_b_nnz[FC_MAX_TYPES];
    int n_ranges;
    int r_type[FC_MAX_RANGES]; int r_pos0[FC_MAX_RANGES]; int r_count[FC_MAX_RANGES];
    // device witness (c.host_z == NULL): every compression's inputs, block-major, comp_per_block per block, at the
    // block-local 2^14-slot comp_slot[j]
    long long n_comp; int comp_per_block; const uint32_t* comp_slot;
    const uint32_t* cv; const uint32_t* msg; const uint64_t* ctr; const uint32_t* blen; const uint32_t* flags;
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
    // hm96-sha512 Merkle leaves (the SHA-512 build, cuda_sha512_patch.py): hm96_cb fills tree k's salts (n leaves x 192 bytes,
    // trees in build order); the scheme's constants are the salt prefix || leaf prefix (256 bytes) and the key's byte tables
    // (192 x 256 x 8 u64). hm96_cb null: unsalted leaves.
    int (*hm96_cb)(void* ctx, int tree, long long n, uint8_t* out);
    void* hm96_ctx;
    const uint8_t* hm96_prefixes;
    const uint64_t* hm96_tables;
} FlockCircuitParams;

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
                                const uint32_t* __restrict__ late, int n_cut, const uint32_t* __restrict__ cut_rows, uint32_t* __restrict__ cut) {
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
    const long long ul = 32 * g + lane;
    const long long blk = ul / per_block, kk = ul % per_block;
    const long long base = (blk << (k_log - 5)) + ((long long)(pos0 + kk) << (unit_log - 5));
    for (int rb = warp; rb < nrb; rb += nw) {
        const int r = 32 * rb + lane;
        uint32_t zv = r < useful ? Z[r] : 0, av = r < useful ? A[r] : 0, bv = r < useful ? B[r] : 0;
        zv = fc_transpose32(zv, lane); av = fc_transpose32(av, lane); bv = fc_transpose32(bv, lane);
        if (ul < n_units) { z_out[base + rb] = zv; a_out[base + rb] = av; b_out[base + rb] = bv; }
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
        const long long nc = N->n_comp;
        uint32_t *d_cv, *d_m, *d_blen, *d_flags, *d_slot; b3u64* d_ctr;
        CK(cudaMalloc(&d_cv, (size_t)nc * 32)); CK(cudaMalloc(&d_m, (size_t)nc * 64)); CK(cudaMalloc(&d_blen, (size_t)nc * 4));
        CK(cudaMalloc(&d_flags, (size_t)nc * 4)); CK(cudaMalloc(&d_ctr, (size_t)nc * 8)); CK(cudaMalloc(&d_slot, (size_t)N->comp_per_block * 4));
        CK(cudaMemcpy(d_cv, N->cv, (size_t)nc * 32, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_m, N->msg, (size_t)nc * 64, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ctr, N->ctr, (size_t)nc * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_blen, N->blen, (size_t)nc * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_flags, N->flags, (size_t)nc * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_slot, N->comp_slot, (size_t)N->comp_per_block * 4, cudaMemcpyHostToDevice));
        F128 *tz, *ta, *tb;
        CK(cudaMalloc(&tz, nc * 128 * sizeof(F128))); CK(cudaMalloc(&ta, nc * 128 * sizeof(F128))); CK(cudaMalloc(&tb, nc * 128 * sizeof(F128)));
        launch_blake3_witness_blocks(d_cv, d_m, d_ctr, d_blen, d_flags, (int)nc, nc, (b3u64*)tz, (b3u64*)ta, (b3u64*)tb);
        CK(cudaGetLastError());
        fc_comp_scatter<<<8192, 256>>>(tz, nc, N->comp_per_block, d_slot, k_log, df);
        fc_comp_scatter<<<8192, 256>>>(ta, nc, N->comp_per_block, d_slot, k_log, d_a);
        fc_comp_scatter<<<8192, 256>>>(tb, nc, N->comp_per_block, d_slot, k_log, d_b);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(0));
        cudaFree(tz); cudaFree(ta); cudaFree(tb);
        cudaFree(d_cv); cudaFree(d_m); cudaFree(d_blen); cudaFree(d_flags); cudaFree(d_ctr); cudaFree(d_slot);
        auto t1 = std::chrono::steady_clock::now();

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
                    (uint32_t*)df, (uint32_t*)d_a, (uint32_t*)d_b, 2, 0, N->u_levels, 0, nullptr, nullptr, 0, nullptr, nullptr);
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
                0, 0, N->u_levels0, 0, nullptr, nullptr, N->u_n_cut, d_cr, d_cut);
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
                1, N->u_levels0, N->u_levels, N->u_n_late, d_lr, d_late, 0, nullptr, nullptr);
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
