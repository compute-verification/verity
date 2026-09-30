// The sm_120 FP8 step model on the GPU (the check of fp8_gpucheck.py): fp8.py's groupsum, which is kernels.group_sum_total_batch's
// arithmetic, as integer code, so a capture's words are checked where they are made.
//
//   model_tiles_kernel  one warp per host tile (A u8[n][16][32], Bt u8[n][8][32], C u32[n][16][8], optional UE8M0 SFA u8[n][16],
//                       SFB u8[n][8]): every word under one spec (ngroups equal groups of the 32 products, width, floor).
//   No kernel holds a floating-point instruction, and model_tiles_kernel no tensor-core one (fp8_gpucheck.py's SASS gate): the
//   model is integer code, and ranges are reduced by multiply-high, never by a division the compiler lowers through MUFU.RCP.
//   fused_step_kernel   per warp and tile: the tile's operands generated from a counter hash through a family's code tables (the
//                       numpy port in fp8_gpucheck.py regenerates any tile), EXACTLY ONE mma.sync, and each of the tile's 128 words
//                       checked against up to MAXM specs (the first is the primary).  Out: counts, every word the primary flags,
//                       and the kept tiles: those the hash samples and those holding a flagged word (their D words, per-spec
//                       mismatch counts and, for the first few, operands).
//   fused_chain_kernel  per warp a dependent E4M3 chain: operands generated per step, the accumulator in registers, each step's
//                       word checked against the specs applied to the device's previous word; sampled and flagged steps keep
//                       (prev, D), and one chain per launch keeps every step's word.
//   Each lane generates exactly its own A/B fragment bytes and checks exactly the four words its C/D fragment holds.
//   host_*              the same model and generator compiled for the CPU (tests, and the gate's check of the numpy port).
//
// Decoded operands (dec u32[2][256], built by fp8_gpucheck.decode_table from fp8.operands): bits 0-4 the significand widened to
// the pair's S = max significand bits, bits 8-15 (u + sig - 1) + 64, bit 16 sign, bit 17 NaN, bit 18 inf; a product is
// (ga * gb) << pshift (pshift = 25 - 2S) at 2^(ea + eb), which is fp8.products.
// Fragment layouts and host arrays as mma_fp8.cu.  Build: nvcc -O2 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared.

#include <cuda_runtime.h>
#include <stdint.h>

#define HD __host__ __device__ __forceinline__

constexpr int MAXM = 4;

// The C API's structs (fp8_gpucheck.py's CSpec, CModels, CFam, CHostOut); outside the anonymous namespace, so the extern "C"
// functions taking them keep external linkage.
struct Spec {
    int ngroups, width, floor;
};
struct Models {
    int n;
    Spec s[MAXM];
};
// A generated family: amode (operand structure) 0 iid, 1 pairs (odd k repeats k-1, B negated; ~15% of A replaced from table s),
// 2 cancel16 (A[k+16] = -A[k], B constant along k), 3 isolated (one live k per tile), 4 wide (row/column powers of two, k = 0 the
// largest), 5 spikes (two positions per row from table s); cmode (accumulator) 0 zero, 1 field in [c_lo, c_hi] with random sign
// and mantissa, 2 tiny (fp8._tiny_acc), 3 one of clist, 4 +-2^(ea + eb - r) with r in [18, 30], 5 +-0.
struct Fam {
    uint32_t key;
    int amode, cmode, c_lo, c_hi, ea, eb, min_a, min_b;
    uint32_t clist[8];
};
// The host-side result buffers of a fused launch (sizes follow Out).
struct HostOut {
    unsigned long long* counts;
    uint32_t *nflag, *flags, *nsamp, *meta, *words;
    uint8_t* ops;
    uint32_t flag_cap, samp_cap, ndump;
};

namespace {

constexpr uint32_t NAN_WORD = 0x7FFFFFFFu, INF_WORD = 0x7F800000u, SIGN = 0x80000000u;
constexpr int TAB = 4096;  // entries per code table; a table index is 12 bits of a hash
constexpr int WPB = 8;  // warps per block
constexpr int FLAG_WORDS = 6, META_WORDS = 3 + MAXM, OPS_BYTES = 512 + 256 + 512;
constexpr uint32_t KEPT_SAMPLED = 1u, KEPT_FLAGGED = 2u;

struct Tabs {
    const uint8_t *a, *b, *s;
};

HD uint32_t fmix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}
HD uint32_t mix(uint32_t x) { return fmix32(x * 0x9E3779B1u + 0x7F4A7C15u); }
HD uint32_t tile_key(uint32_t key, uint32_t lo, uint32_t hi) { return mix(mix(key ^ lo) ^ hi); }
HD uint32_t hs(uint32_t tk, uint32_t slot) { return mix(tk ^ slot); }

// hash slots of a tile: A (i, k) i*32+k, B (j, k) 512+j*32+k, C (i, j) 768+i*8+j and 896+i*8+j, rows 1024+i, columns 1040+j,
// the isolated k 1056, the sample draw 1100, the pairs family's residual draw 1200+i*32+k
constexpr uint32_t SLOT_B = 512, SLOT_C1 = 768, SLOT_C2 = 896, SLOT_ROW = 1024, SLOT_COL = 1040, SLOT_K0 = 1056, SLOT_SAMPLE = 1100,
                   SLOT_RES = 1200;

HD int imax(int a, int b) { return a > b ? a : b; }
// x's top bits reduced to [0, n) (multiply-high, no division)
HD uint32_t below(uint32_t x, uint32_t n) { return (uint32_t)(((uint64_t)x * n) >> 32); }

HD uint8_t gen_a(const Fam& f, const Tabs& tb, uint32_t tk, int i, int k) {
    switch (f.amode) {
        case 1: {
            const uint32_t x = hs(tk, SLOT_RES + i * 32 + k);
            if (((x >> 12) & 255u) < 38u) return tb.s[x & (TAB - 1)];
            return tb.a[hs(tk, i * 32 + (k & ~1)) & (TAB - 1)];
        }
        case 2: return k < 16 ? tb.a[hs(tk, i * 32 + k) & (TAB - 1)] : (uint8_t)(tb.a[hs(tk, i * 32 + k - 16) & (TAB - 1)] ^ 0x80);
        case 3: return k == (int)(hs(tk, SLOT_K0) & 31u) ? tb.a[hs(tk, i * 32 + k) & (TAB - 1)] : (uint8_t)0;
        case 4: {
            if (k == 0) return tb.s[f.ea - f.min_a];
            const int lo = imax(f.ea - 15, f.min_a);
            const int e = lo + (int)below(hs(tk, SLOT_ROW + i), (uint32_t)(f.ea - 10 - lo + 1));
            return (uint8_t)(tb.s[e - f.min_a] | ((hs(tk, i * 32 + k) >> 31) << 7));
        }
        case 5: {
            const uint32_t r = hs(tk, SLOT_ROW + i), x = hs(tk, i * 32 + k);
            return (k == (int)(r & 31u) || k == (int)((r >> 5) & 31u)) ? tb.s[x & (TAB - 1)] : tb.a[x & (TAB - 1)];
        }
        default: return tb.a[hs(tk, i * 32 + k) & (TAB - 1)];
    }
}

HD uint8_t gen_b(const Fam& f, const Tabs& tb, uint32_t tk, int j, int k) {
    switch (f.amode) {
        case 1: return (uint8_t)(tb.b[hs(tk, SLOT_B + j * 32 + (k & ~1)) & (TAB - 1)] ^ ((k & 1) << 7));
        case 2: return tb.b[hs(tk, SLOT_B + j * 32) & (TAB - 1)];
        case 3: return k == (int)(hs(tk, SLOT_K0) & 31u) ? tb.b[hs(tk, SLOT_B + j * 32 + k) & (TAB - 1)] : (uint8_t)0;
        case 4: {
            if (k == 0) return tb.s[64 + f.eb - f.min_b];
            const int lo = imax(f.eb - 15, f.min_b);
            const int e = lo + (int)below(hs(tk, SLOT_COL + j), (uint32_t)(f.eb - 10 - lo + 1));
            return tb.s[64 + e - f.min_b];
        }
        default: return tb.b[hs(tk, SLOT_B + j * 32 + k) & (TAB - 1)];
    }
}

HD uint32_t gen_c(const Fam& f, uint32_t tk, int i, int j) {
    const uint32_t h1 = hs(tk, SLOT_C1 + i * 8 + j), h2 = hs(tk, SLOT_C2 + i * 8 + j);
    switch (f.cmode) {
        case 1: return (h2 & SIGN) | ((uint32_t)(f.c_lo + (int)below(h2 << 1, (uint32_t)(f.c_hi - f.c_lo + 1))) << 23) | (h1 & 0x7FFFFFu);
        case 2: {
            const uint32_t e = ((h2 >> 30) & 1u) ? 1u + below(h2 << 2, 27u) : 0u, m = h1 & 0x7FFFFFu;
            return (h1 & SIGN) | (e << 23) | (m ? m : 1u);
        }
        case 3: return f.clist[h1 & 7u];
        case 4: return (h2 & SIGN) | ((uint32_t)(f.ea + f.eb - 18 - (int)below(h1, 13u) + 127) << 23);
        case 5: return h1 & SIGN;
        default: return 0u;
    }
}

HD int bitlen64(uint64_t x) {
#ifdef __CUDA_ARCH__
    return 64 - __clzll((long long)x);
#else
    return x ? 64 - __builtin_clzll(x) : 0;
#endif
}

HD void product(uint32_t da, uint32_t db, int pshift, int scale, bool snan, uint32_t& mag, int& exp, bool& neg, bool& nan, bool& inf) {
    const uint32_t ga = da & 31u, gb = db & 31u;
    const bool na = (da >> 17) & 1u, nb = (db >> 17) & 1u, ia = (da >> 18) & 1u, ib = (db >> 18) & 1u;
    nan = na || nb || (ia && gb == 0 && !nb) || (ib && ga == 0 && !na) || snan;
    inf = (ia || ib) && !nan;
    mag = (nan || inf) ? 0u : (ga * gb) << pshift;
    exp = (int)((da >> 8) & 255u) + (int)((db >> 8) & 255u) - 128 + scale;
    neg = ((da ^ db) >> 16) & 1u;
}

HD uint32_t align(uint32_t m, int e, int top, int rescale) {
    const uint32_t sc = rescale >= 0 ? m << rescale : m >> -rescale;
    const int sh = top - e;
    return sh >= 32 ? 0u : sc >> sh;
}

// One GroupSum step (fp8.groupsum): the accumulator word and 32 decoded (a, b) pairs -> the result word.  Magnitudes stay below
// 2^25 and widths at most 30, so an aligned term fits 32 bits and a group's sum 64.
template <int NG>
HD uint32_t gs_word(int width, int floor, uint32_t acc, const uint32_t* ra, const uint32_t* cb, int pshift, int scale, bool snan) {
    const uint32_t field = (acc >> 23) & 0xFFu, frac = acc & 0x7FFFFFu;
    int state = field == 0xFFu ? (frac ? 1 : 2) : 0;
    bool neg = acc >> 31;
    uint32_t mag = state == 0 ? (field == 0 ? frac : frac | (1u << 23)) : 0u;
    int exp = field == 0 ? -126 : (int)field - 127;
    const int rescale = width - 24;
    constexpr int SZ = 32 / NG;
#pragma unroll
    for (int g = 0; g < NG; g++) {
        bool pos_inf = state == 2 && !neg, neg_inf = state == 2 && neg, any_nan = false;
        int top = floor;
        if (mag != 0 && exp > top) top = exp;
#pragma unroll 4
        for (int q = 0; q < SZ; q++) {
            uint32_t pm;
            int pe;
            bool pn, pnan, pinf;
            product(ra[g * SZ + q], cb[g * SZ + q], pshift, scale, snan, pm, pe, pn, pnan, pinf);
            any_nan |= pnan;
            if (pinf) (pn ? neg_inf : pos_inf) = true;
            if (pm != 0 && pe > top) top = pe;
        }
        const bool is_nan = state == 1 || any_nan || (pos_inf && neg_inf);
        const bool is_inf = !is_nan && (pos_inf || neg_inf);
        int64_t total = 0;
        if (mag != 0) {
            const int64_t al = align(mag, exp, top, rescale);
            total += neg ? -al : al;
        }
#pragma unroll 4
        for (int q = 0; q < SZ; q++) {
            uint32_t pm;
            int pe;
            bool pn, pnan, pinf;
            product(ra[g * SZ + q], cb[g * SZ + q], pshift, scale, snan, pm, pe, pn, pnan, pinf);
            if (pm != 0) {
                const int64_t al = align(pm, pe, top, rescale);
                total += pn ? -al : al;
            }
        }
        const bool s_neg = total < 0;
        uint64_t s_mag = (uint64_t)(s_neg ? -total : total);
        const int bl = bitlen64(s_mag);
        int s_exp = top + bl - width;
        s_mag = bl > width ? s_mag >> (bl - width) : s_mag << (width - bl);
        const int sh = -126 - s_exp;
        if (sh > 0) s_mag = sh >= 64 ? 0 : s_mag >> sh;
        if (s_exp < -126) s_exp = -126;
        s_mag = rescale >= 0 ? s_mag >> rescale : s_mag << -rescale;
        const bool nz = s_mag != 0;
        const bool over = !is_nan && !is_inf && nz && s_exp > 127;
        const int ns = is_nan ? 1 : ((is_inf || over) ? 2 : 0);
        neg = is_inf ? neg_inf : (s_neg && nz);
        mag = ns == 0 ? (uint32_t)s_mag : 0u;
        exp = s_exp;
        state = ns;
    }
    if (state == 1) return NAN_WORD;
    if (state == 2) return neg ? (SIGN | INF_WORD) : INF_WORD;
    if (mag == 0) return 0u;
    const int e = exp < 127 ? exp : 127;
    const uint32_t fo = (uint32_t)(e + 127 - ((mag & (1u << 23)) == 0 ? 1 : 0));
    return ((uint32_t)neg << 31) | (fo << 23) | (mag & 0x7FFFFFu);
}

HD uint32_t model_word(const Spec& s, uint32_t acc, const uint32_t* ra, const uint32_t* cb, int pshift, int scale, bool snan) {
    switch (s.ngroups) {
        case 1: return gs_word<1>(s.width, s.floor, acc, ra, cb, pshift, scale, snan);
        case 2: return gs_word<2>(s.width, s.floor, acc, ra, cb, pshift, scale, snan);
        default: return gs_word<4>(s.width, s.floor, acc, ra, cb, pshift, scale, snan);
    }
}

HD bool finite_dec(uint32_t d) { return (d & (3u << 17)) == 0; }

HD int scale_of(int sa, int sb, bool& snan) {
    snan = sa == 0xFF || sb == 0xFF;
    return snan ? 0 : sa + sb - 254;
}

#define MMA_F8(NAME, MNEMONIC)                                                                                          \
    __device__ __forceinline__ void NAME(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {      \
        asm volatile(MNEMONIC " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                             \
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                                                    \
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), \
                       "f"(c[3]));                                                                                       \
    }

MMA_F8(mma_f8_e4m3, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32")
MMA_F8(mma_f8_e5m2, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32")
MMA_F8(mma_f8_e4m3e5m2, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e5m2.f32")

template <int KIND>
__device__ __forceinline__ void mma_kind(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    if (KIND == 0) mma_f8_e4m3(d, a, b, c);
    else if (KIND == 1) mma_f8_e5m2(d, a, b, c);
    else mma_f8_e4m3e5m2(d, a, b, c);
}

// Device pointers of one launch's results.  counts: [0] tiles (chain steps), [1] elements, [2] gated elements, then per spec m
// [3 + 2m] gated and [4 + 2m] record-only mismatches.  flags: FLAG_WORDS per flagged word (tile lo, tile hi (chain: step),
// element i*8+j, device word, primary word, gated).  kept tiles: META_WORDS (tile lo, tile hi, why kept (KEPT_SAMPLED |
// KEPT_FLAGGED), per-spec mismatches of the tile) and the tile's D words (chain: its previous words, then D); the first ndump kept
// tiles also keep A, Bt, C.
struct Out {
    unsigned long long* counts;
    uint32_t *nflag, *flags, *nsamp, *meta, *words;
    uint8_t* ops;
    uint32_t flag_cap, samp_cap, ndump;
};

// One warp's tile in shared memory: the decoded operands (the model's) and the warp's counters (lane 0 writes them).  The raw
// operands, accumulators and device words stay in each lane's fragment registers: a lane's four words (rows g, g+8, columns 2t,
// 2t+1) are the ones its C/D fragment holds, and it checks exactly those.
struct Warp {
    uint32_t da[512], db[256];
    unsigned mg[MAXM], mr[MAXM], tm[MAXM], gated;
};

__device__ __forceinline__ void load_tables(uint8_t* s_tab, uint32_t* s_dec, Spec* s_ms, const uint8_t* tabs, const uint32_t* dec, const Models& ms) {
    for (int x = threadIdx.x; x < 3 * TAB / 4; x += 32 * WPB) reinterpret_cast<uint32_t*>(s_tab)[x] = reinterpret_cast<const uint32_t*>(tabs)[x];
    for (int x = threadIdx.x; x < 512; x += 32 * WPB) s_dec[x] = dec[x];
    if (threadIdx.x < MAXM) s_ms[threadIdx.x] = ms.s[threadIdx.x];
    __syncthreads();
}

__device__ __forceinline__ int lane_row(int lane, int e) { return (lane >> 2) + 8 * (e >> 1); }
__device__ __forceinline__ int lane_col(int lane, int e) { return 2 * (lane & 3) + (e & 1); }

// The lane's A and B fragment bytes (mma_fp8.cu's layouts: A reg r byte j is row g + 8(r&1), k = 4t + j + 16(r>>1); B reg r byte
// j is column g, k = 4t + j + 16r), generated, and their decoded values stored for the check.
__device__ __forceinline__ void gen_fragments(Warp& w, const Fam& f, const Tabs& tb, const uint32_t* s_dec, uint32_t tk, int lane, uint32_t a[4],
                                              uint32_t b[2]) {
    const int g = lane >> 2, kc = 4 * (lane & 3);
#pragma unroll
    for (int r = 0; r < 4; r++) {
        const int row = g + 8 * (r & 1), k0 = kc + 16 * (r >> 1);
        uint32_t v = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint8_t x = gen_a(f, tb, tk, row, k0 + j);
            v |= (uint32_t)x << (8 * j);
            w.da[row * 32 + k0 + j] = s_dec[x];
        }
        a[r] = v;
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int k0 = kc + 16 * r;
        uint32_t v = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint8_t x = gen_b(f, tb, tk, g, k0 + j);
            v |= (uint32_t)x << (8 * j);
            w.db[g * 32 + k0 + j] = s_dec[256 + x];
        }
        b[r] = v;
    }
}

__device__ __forceinline__ uint32_t pick(const uint32_t v[4], int e) { return e == 0 ? v[0] : e == 1 ? v[1] : e == 2 ? v[2] : v[3]; }

__device__ __forceinline__ void lane_words(uint32_t* dst, const uint32_t v[4], int lane) {
#pragma unroll
    for (int e = 0; e < 4; e++) dst[lane_row(lane, e) * 8 + lane_col(lane, e)] = v[e];
}

// Rows 0-15 and columns 16-23 of the tile: bit set when every operand is finite.
__device__ __forceinline__ unsigned finite_mask(const Warp& w, int lane) {
    bool mine = true;
    if (lane < 16) {
        for (int k = 0; k < 32; k++) mine = mine && finite_dec(w.da[lane * 32 + k]);
    } else if (lane < 24) {
        for (int k = 0; k < 32; k++) mine = mine && finite_dec(w.db[(lane - 16) * 32 + k]);
    }
    return __ballot_sync(0xFFFFFFFFu, mine);
}

// The lane's four words (prev its accumulators, d the device's words) against every spec; the tile's per-spec mismatch counts
// land in w.tm.  A word is gated (fp8._finite_domain) when its accumulator, its operands and the primary's word are finite.
// Returns (warp-uniform) whether the primary flagged any word of the tile.
__device__ __forceinline__ bool check_tile(Warp& w, const Spec* s_ms, int nm, int pshift, int lane, const uint32_t prev[4], const uint32_t d[4],
                                           uint32_t lo, uint32_t hi, const Out& o) {
    bool flagged = false;
    if (lane < MAXM) w.tm[lane] = 0;
    const unsigned fmask = finite_mask(w, lane);
    __syncwarp();
#pragma unroll 1
    for (int e = 0; e < 4; e++) {
        const int i = lane_row(lane, e), j = lane_col(lane, e);
        const uint32_t acc = pick(prev, e), dw = pick(d, e);
        const bool fin = (acc & INF_WORD) != INF_WORD && ((fmask >> i) & 1u) && ((fmask >> (16 + j)) & 1u);
        uint32_t w0 = 0;
        bool gated = false;
#pragma unroll 1
        for (int m = 0; m < nm; m++) {
            const uint32_t mw = model_word(s_ms[m], acc, w.da + i * 32, w.db + j * 32, pshift, 0, false);
            if (m == 0) {
                w0 = mw;
                gated = fin && (mw & INF_WORD) != INF_WORD;
            }
            const bool bad = mw != dw;
            const unsigned bg = __ballot_sync(0xFFFFFFFFu, bad && gated), br = __ballot_sync(0xFFFFFFFFu, bad && !gated);
            if (lane == 0) {
                w.mg[m] += __popc(bg);
                w.mr[m] += __popc(br);
                w.tm[m] += __popc(bg | br);
            }
        }
        const unsigned gb = __ballot_sync(0xFFFFFFFFu, gated);
        if (lane == 0) w.gated += __popc(gb);
        if (w0 != dw) {
            flagged = true;
            const uint32_t slot = atomicAdd(o.nflag, 1u);
            if (slot < o.flag_cap) {
                uint32_t* fl = o.flags + (size_t)slot * FLAG_WORDS;
                fl[0] = lo;
                fl[1] = hi;
                fl[2] = (uint32_t)(i * 8 + j);
                fl[3] = dw;
                fl[4] = w0;
                fl[5] = gated;
            }
        }
    }
    const bool any = __any_sync(0xFFFFFFFFu, flagged);
    __syncwarp();
    return any;
}

// A sampled tile: its words (a step's D, 128; a chain step's previous words then D, 256), metadata, and for the first ndump
// samples the operands each lane holds (A u8[16][32], Bt u8[8][32], C u32[16][8]).
__device__ __forceinline__ void keep_sample(const Warp& w, int lane, uint32_t lo, uint32_t hi, uint32_t why, bool chain, const uint32_t a[4],
                                            const uint32_t b[2], const uint32_t prev[4], const uint32_t d[4], const Out& o) {
    unsigned slot = 0;
    if (lane == 0) slot = atomicAdd(o.nsamp, 1u);
    slot = __shfl_sync(0xFFFFFFFFu, slot, 0);
    if (slot >= o.samp_cap) return;
    const int per = chain ? 256 : 128;
    uint32_t* dst = o.words + (size_t)slot * per;
    if (chain) lane_words(dst, prev, lane);
    lane_words(dst + per - 128, d, lane);
    if (lane == 0) {
        uint32_t* me = o.meta + (size_t)slot * META_WORDS;
        me[0] = lo;
        me[1] = hi;
        me[2] = why;
        for (int m = 0; m < MAXM; m++) me[3 + m] = w.tm[m];
    }
    if (slot < o.ndump) {
        uint8_t* op = o.ops + (size_t)slot * OPS_BYTES;
        const int g = lane >> 2, kc = 4 * (lane & 3);
        for (int r = 0; r < 4; r++) *reinterpret_cast<uint32_t*>(op + (g + 8 * (r & 1)) * 32 + kc + 16 * (r >> 1)) = a[r];
        for (int r = 0; r < 2; r++) *reinterpret_cast<uint32_t*>(op + 512 + g * 32 + kc + 16 * r) = b[r];
        lane_words(reinterpret_cast<uint32_t*>(op + 768), prev, lane);
    }
}

__device__ __forceinline__ void flush_counts(const Out& o, const Warp& w, unsigned tiles, int lane) {
    if (lane == 0) {
        atomicAdd(&o.counts[0], (unsigned long long)tiles);
        atomicAdd(&o.counts[1], 128ull * tiles);
        atomicAdd(&o.counts[2], (unsigned long long)w.gated);
        for (int m = 0; m < MAXM; m++) {
            atomicAdd(&o.counts[3 + 2 * m], (unsigned long long)w.mg[m]);
            atomicAdd(&o.counts[4 + 2 * m], (unsigned long long)w.mr[m]);
        }
    }
}

__device__ __forceinline__ void zero_counts(Warp& w, int lane) {
    if (lane < MAXM) w.mg[lane] = w.mr[lane] = w.tm[lane] = 0;
    if (lane == 0) w.gated = 0;
    __syncwarp();
}

// Tiles t0 + n * W + w for n < per_warp, W the launch's warps and w this warp's index.
template <int KIND>
__global__ void __launch_bounds__(32 * WPB) fused_step_kernel(Fam f, Models ms, int pshift, const uint8_t* __restrict__ tabs,
                                                              const uint32_t* __restrict__ dec, unsigned long long t0, unsigned per_warp,
                                                              unsigned smask, unsigned long long corrupt, Out o) {
    __shared__ __align__(16) uint8_t s_tab[3 * TAB];
    __shared__ uint32_t s_dec[512];
    __shared__ Spec s_ms[MAXM];
    __shared__ Warp s_w[WPB];
    load_tables(s_tab, s_dec, s_ms, tabs, dec, ms);
    const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned W = gridDim.x * WPB, wg = blockIdx.x * WPB + wib;
    const Tabs tb{s_tab, s_tab + TAB, s_tab + 2 * TAB};
    Warp& w = s_w[wib];
    zero_counts(w, lane);
#pragma unroll 1
    for (unsigned n = 0; n < per_warp; n++) {
        const unsigned long long t = t0 + (unsigned long long)n * W + wg;
        const uint32_t lo = (uint32_t)t, hi = (uint32_t)(t >> 32), tk = tile_key(f.key, lo, hi);
        uint32_t a[4], b[2], prev[4], d[4];
        gen_fragments(w, f, tb, s_dec, tk, lane, a, b);
        float cf[4], df[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            prev[e] = gen_c(f, tk, lane_row(lane, e), lane_col(lane, e));
            cf[e] = __uint_as_float(prev[e]);
        }
        __syncwarp();
        mma_kind<KIND>(df, a, b, cf);
#pragma unroll
        for (int e = 0; e < 4; e++) d[e] = __float_as_uint(df[e]);
        if (t == corrupt && lane == 0) d[0] ^= 1u;
        const uint32_t why = (check_tile(w, s_ms, ms.n, pshift, lane, prev, d, lo, hi, o) ? KEPT_FLAGGED : 0u) |
                             ((hs(tk, SLOT_SAMPLE) & smask) == 0 ? KEPT_SAMPLED : 0u);
        if (why) keep_sample(w, lane, lo, hi, why, false, a, b, prev, d, o);
        __syncwarp();
    }
    flush_counts(o, w, per_warp, lane);
}

// One E4M3 chain per warp: chain c0 + w, nsteps dependent steps from C0 (hash tile hi 0xFFFFFFFF).  corrupt = (chain << 32) | step.
// rec: when this warp's chain is rec_chain, C0 then every step's word, u32[nsteps + 1][128].
__global__ void __launch_bounds__(32 * WPB) fused_chain_kernel(Fam f, Models ms, int pshift, const uint8_t* __restrict__ tabs,
                                                               const uint32_t* __restrict__ dec, unsigned c0, unsigned nchains,
                                                               unsigned nsteps, unsigned smask, unsigned long long corrupt, unsigned rec_chain,
                                                               uint32_t* __restrict__ rec, Out o) {
    __shared__ __align__(16) uint8_t s_tab[3 * TAB];
    __shared__ uint32_t s_dec[512];
    __shared__ Spec s_ms[MAXM];
    __shared__ Warp s_w[WPB];
    load_tables(s_tab, s_dec, s_ms, tabs, dec, ms);
    const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned wg = blockIdx.x * WPB + wib;
    if (wg >= nchains) return;
    const unsigned c = c0 + wg;
    const Tabs tb{s_tab, s_tab + TAB, s_tab + 2 * TAB};
    Warp& w = s_w[wib];
    zero_counts(w, lane);
    const uint32_t tk0 = tile_key(f.key, c, 0xFFFFFFFFu);
    uint32_t prev[4];
#pragma unroll
    for (int e = 0; e < 4; e++) prev[e] = gen_c(f, tk0, lane_row(lane, e), lane_col(lane, e));
    const bool recording = c == rec_chain;
    if (recording) lane_words(rec, prev, lane);
#pragma unroll 1
    for (unsigned s = 0; s < nsteps; s++) {
        const uint32_t tk = tile_key(f.key, c, s);
        uint32_t a[4], b[2], d[4];
        gen_fragments(w, f, tb, s_dec, tk, lane, a, b);
        float cf[4], df[4];
#pragma unroll
        for (int e = 0; e < 4; e++) cf[e] = __uint_as_float(prev[e]);
        __syncwarp();
        mma_f8_e4m3(df, a, b, cf);
#pragma unroll
        for (int e = 0; e < 4; e++) d[e] = __float_as_uint(df[e]);
        if ((((unsigned long long)c << 32) | s) == corrupt && lane == 0) d[0] ^= 1u;
        const uint32_t why = (check_tile(w, s_ms, ms.n, pshift, lane, prev, d, c, s, o) ? KEPT_FLAGGED : 0u) |
                             ((hs(tk, SLOT_SAMPLE) & smask) == 0 ? KEPT_SAMPLED : 0u);
        if (why) keep_sample(w, lane, c, s, why, true, a, b, prev, d, o);
        if (recording) lane_words(rec + (size_t)(s + 1) * 128, d, lane);
#pragma unroll
        for (int e = 0; e < 4; e++) prev[e] = d[e];
        __syncwarp();
    }
    flush_counts(o, w, nsteps, lane);
}

// One warp per host tile: the tile decoded into shared memory, then each lane's four words under one spec.
__global__ void __launch_bounds__(32 * WPB) model_tiles_kernel(Spec s, const uint32_t* __restrict__ dec, int pshift, const uint8_t* __restrict__ A,
                                                               const uint8_t* __restrict__ Bt, const uint32_t* __restrict__ C,
                                                               const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB, int scaled,
                                                               uint32_t* __restrict__ out, int ntiles) {
    __shared__ uint32_t s_dec[512];
    __shared__ uint32_t s_da[WPB][512], s_db[WPB][256], s_c[WPB][128];
    __shared__ int s_sa[WPB][16], s_sb[WPB][8];
    for (int x = threadIdx.x; x < 512; x += 32 * WPB) s_dec[x] = dec[x];
    __syncthreads();
    const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int W = gridDim.x * WPB;
#pragma unroll 1
    for (int t = blockIdx.x * WPB + wib; t < ntiles; t += W) {
        const size_t ta = (size_t)t * 512, tb = (size_t)t * 256, tc = (size_t)t * 128;
        for (int x = lane; x < 512; x += 32) s_da[wib][x] = s_dec[A[ta + x]];
        for (int x = lane; x < 256; x += 32) s_db[wib][x] = s_dec[256 + Bt[tb + x]];
        for (int x = lane; x < 128; x += 32) s_c[wib][x] = C[tc + x];
        if (lane < 16) s_sa[wib][lane] = scaled ? SFA[(size_t)t * 16 + lane] : 127;
        if (lane < 8) s_sb[wib][lane] = scaled ? SFB[(size_t)t * 8 + lane] : 127;
        __syncwarp();
#pragma unroll 1
        for (int e = 0; e < 4; e++) {
            const int i = lane_row(lane, e), j = lane_col(lane, e);
            bool snan = false;
            const int sc = scaled ? scale_of(s_sa[wib][i], s_sb[wib][j], snan) : 0;
            out[tc + i * 8 + j] = model_word(s, s_c[wib][i * 8 + j], s_da[wib] + i * 32, s_db[wib] + j * 32, pshift, sc, snan);
        }
        __syncwarp();
    }
}

struct Dev {
    void* p = nullptr;
    ~Dev() { cudaFree(p); }
};

static int upload(Dev& d, const void* host, size_t bytes) {
    cudaError_t err = cudaMalloc(&d.p, bytes ? bytes : 1);
    if (err == cudaSuccess && bytes) err = host ? cudaMemcpy(d.p, host, bytes, cudaMemcpyHostToDevice) : cudaMemset(d.p, 0, bytes);
    return (int)err;
}

static int download(void* host, const Dev& d, size_t bytes) {
    return bytes ? (int)cudaMemcpy(host, d.p, bytes, cudaMemcpyDeviceToHost) : 0;
}

static bool specs_ok(const Models& ms) {
    if (ms.n < 1 || ms.n > MAXM) return false;
    for (int m = 0; m < ms.n; m++) {
        const Spec& s = ms.s[m];
        if ((s.ngroups != 1 && s.ngroups != 2 && s.ngroups != 4) || s.width < 8 || s.width > 30) return false;
    }
    return true;
}

struct DevOut {
    Dev counts, nflag, flags, nsamp, meta, words, ops;
    Out o;
    int alloc(const HostOut& h, size_t per) {
        int rc;
        if ((rc = upload(counts, nullptr, sizeof(unsigned long long) * (3 + 2 * MAXM))) || (rc = upload(nflag, nullptr, 4)) ||
            (rc = upload(flags, nullptr, (size_t)h.flag_cap * FLAG_WORDS * 4)) || (rc = upload(nsamp, nullptr, 4)) ||
            (rc = upload(meta, nullptr, (size_t)h.samp_cap * META_WORDS * 4)) || (rc = upload(words, nullptr, (size_t)h.samp_cap * per * 4)) ||
            (rc = upload(ops, nullptr, (size_t)h.ndump * OPS_BYTES)))
            return rc;
        o = Out{(unsigned long long*)counts.p, (uint32_t*)nflag.p, (uint32_t*)flags.p, (uint32_t*)nsamp.p, (uint32_t*)meta.p, (uint32_t*)words.p,
                (uint8_t*)ops.p, h.flag_cap, h.samp_cap, h.ndump};
        return 0;
    }
    int fetch(const HostOut& h, size_t per) {
        int rc;
        if ((rc = download(h.counts, counts, sizeof(unsigned long long) * (3 + 2 * MAXM))) || (rc = download(h.nflag, nflag, 4)) ||
            (rc = download(h.nsamp, nsamp, 4)))
            return rc;
        const uint32_t nf = *h.nflag < h.flag_cap ? *h.nflag : h.flag_cap, ns = *h.nsamp < h.samp_cap ? *h.nsamp : h.samp_cap;
        const uint32_t nd = ns < h.ndump ? ns : h.ndump;
        if ((rc = download(h.flags, flags, (size_t)nf * FLAG_WORDS * 4)) || (rc = download(h.meta, meta, (size_t)ns * META_WORDS * 4)) ||
            (rc = download(h.words, words, (size_t)ns * per * 4)) || (rc = download(h.ops, ops, (size_t)nd * OPS_BYTES)))
            return rc;
        return 0;
    }
};

struct Timer {
    cudaEvent_t a, b;
    Timer() {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
    }
    ~Timer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    float stop() {
        float ms = 0;
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

}  // namespace

extern "C" {

// The model's words of n host tiles under one spec (scaled: apply SFA/SFB as UE8M0 scales).
int gc_model_tiles(const Spec* s, const uint32_t* dec, int pshift, const uint8_t* A, const uint8_t* Bt, const uint32_t* C, const uint8_t* SFA,
                   const uint8_t* SFB, int scaled, uint32_t* out, int n) {
    Models ms{1, {*s}};
    if (!specs_ok(ms) || n < 1) return (int)cudaErrorInvalidValue;
    Dev dD, dA, dB, dC, dSA, dSB, dO;
    int rc;
    if ((rc = upload(dD, dec, 512 * 4)) || (rc = upload(dA, A, (size_t)n * 512)) || (rc = upload(dB, Bt, (size_t)n * 256)) ||
        (rc = upload(dC, C, (size_t)n * 512)) || (rc = upload(dO, nullptr, (size_t)n * 512)))
        return rc;
    if (scaled && ((rc = upload(dSA, SFA, (size_t)n * 16)) || (rc = upload(dSB, SFB, (size_t)n * 8)))) return rc;
    const int blocks = n < 188 * 4 * WPB ? (n + WPB - 1) / WPB : 188 * 4;
    model_tiles_kernel<<<blocks, 32 * WPB>>>(*s, (const uint32_t*)dD.p, pshift, (const uint8_t*)dA.p, (const uint8_t*)dB.p, (const uint32_t*)dC.p,
                                             (const uint8_t*)dSA.p, (const uint8_t*)dSB.p, scaled, (uint32_t*)dO.p, n);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;
    return download(out, dO, (size_t)n * 512);
}

int gc_host_model_tiles(const Spec* s, const uint32_t* dec, int pshift, const uint8_t* A, const uint8_t* Bt, const uint32_t* C, const uint8_t* SFA,
                        const uint8_t* SFB, int scaled, uint32_t* out, int n) {
    Models ms{1, {*s}};
    if (!specs_ok(ms)) return (int)cudaErrorInvalidValue;
    uint32_t ra[32], cb[32];
    for (long long t = 0; t < n; t++)
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 32; k++) {
                    ra[k] = dec[A[t * 512 + i * 32 + k]];
                    cb[k] = dec[256 + Bt[t * 256 + j * 32 + k]];
                }
                bool snan = false;
                const int sc = scaled ? scale_of(SFA[t * 16 + i], SFB[t * 8 + j], snan) : 0;
                out[t * 128 + i * 8 + j] = model_word(*s, C[t * 128 + i * 8 + j], ra, cb, pshift, sc, snan);
            }
    return 0;
}

// The generator on the CPU: tiles (lo[x], hi[x]) of family f -> A u8[n][16][32], Bt u8[n][8][32], C u32[n][16][8].
int gc_host_gen(const Fam* f, const uint8_t* tabs, const uint32_t* lo, const uint32_t* hi, int n, uint8_t* A, uint8_t* Bt, uint32_t* C) {
    const Tabs tb{tabs, tabs + TAB, tabs + 2 * TAB};
    for (int x = 0; x < n; x++) {
        const uint32_t tk = tile_key(f->key, lo[x], hi[x]);
        for (int i = 0; i < 16; i++)
            for (int k = 0; k < 32; k++) A[(size_t)x * 512 + i * 32 + k] = gen_a(*f, tb, tk, i, k);
        for (int j = 0; j < 8; j++)
            for (int k = 0; k < 32; k++) Bt[(size_t)x * 256 + j * 32 + k] = gen_b(*f, tb, tk, j, k);
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 8; j++) C[(size_t)x * 128 + i * 8 + j] = gen_c(*f, tk, i, j);
    }
    return 0;
}

// kind: 0 e4m3, 1 e5m2, 2 e4m3 x e5m2.  One launch of blocks x WPB warps, per_warp tiles each, from tile t0.
int gc_fused_step(int kind, const Fam* f, const Models* ms, int pshift, const uint8_t* tabs, const uint32_t* dec, unsigned long long t0, int blocks,
                  unsigned per_warp, unsigned smask, unsigned long long corrupt, const HostOut* h, float* ms_elapsed) {
    if (!specs_ok(*ms) || kind < 0 || kind > 2 || blocks < 1) return (int)cudaErrorInvalidValue;
    Dev dT, dD;
    DevOut out;
    int rc;
    if ((rc = upload(dT, tabs, 3 * TAB)) || (rc = upload(dD, dec, 512 * 4)) || (rc = out.alloc(*h, 128))) return rc;
    auto tb = (const uint8_t*)dT.p;
    auto dc = (const uint32_t*)dD.p;
    Timer tm;
    switch (kind) {
        case 0: fused_step_kernel<0><<<blocks, 32 * WPB>>>(*f, *ms, pshift, tb, dc, t0, per_warp, smask, corrupt, out.o); break;
        case 1: fused_step_kernel<1><<<blocks, 32 * WPB>>>(*f, *ms, pshift, tb, dc, t0, per_warp, smask, corrupt, out.o); break;
        default: fused_step_kernel<2><<<blocks, 32 * WPB>>>(*f, *ms, pshift, tb, dc, t0, per_warp, smask, corrupt, out.o); break;
    }
    cudaError_t err = cudaGetLastError();
    *ms_elapsed = tm.stop();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;
    return out.fetch(*h, 128);
}

// nchains E4M3 chains from chain c0, nsteps each; rec (u32[nsteps + 1][128]) receives chain rec_chain's words when it is in range.
int gc_fused_chain(const Fam* f, const Models* ms, int pshift, const uint8_t* tabs, const uint32_t* dec, unsigned c0, unsigned nchains, unsigned nsteps,
                   unsigned smask, unsigned long long corrupt, unsigned rec_chain, uint32_t* rec, const HostOut* h, float* ms_elapsed) {
    if (!specs_ok(*ms) || nchains < 1) return (int)cudaErrorInvalidValue;
    Dev dT, dD, dR;
    DevOut out;
    int rc;
    if ((rc = upload(dT, tabs, 3 * TAB)) || (rc = upload(dD, dec, 512 * 4)) || (rc = upload(dR, nullptr, ((size_t)nsteps + 1) * 512)) ||
        (rc = out.alloc(*h, 256)))
        return rc;
    const unsigned blocks = (nchains + WPB - 1) / WPB;
    Timer tm;
    fused_chain_kernel<<<blocks, 32 * WPB>>>(*f, *ms, pshift, (const uint8_t*)dT.p, (const uint32_t*)dD.p, c0, nchains, nsteps, smask, corrupt,
                                            rec_chain, (uint32_t*)dR.p, out.o);
    cudaError_t err = cudaGetLastError();
    *ms_elapsed = tm.stop();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;
    if (rec && rec_chain - c0 < nchains && (rc = download(rec, dR, ((size_t)nsteps + 1) * 512))) return rc;
    return out.fetch(*h, 256);
}

int device_info(int* major, int* minor, int* sms, char* name, int name_len, unsigned char* uuid) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *major = prop.major;
    *minor = prop.minor;
    *sms = prop.multiProcessorCount;
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    name[i] = 0;
    for (int j = 0; j < 16; j++) uuid[j] = (unsigned char)prop.uuid.bytes[j];
    return 0;
}

}  // extern "C"
