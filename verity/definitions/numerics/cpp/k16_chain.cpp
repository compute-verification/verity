// `dense_rows._chain` in integers: the ascending k16 chain of `DotBf16_v2{K, DOT}` for every (row of X, row of W) pair of a
// coordinate span.  A bf16 x bf16 product is MxMw * 2^(pn - 14) (8-bit significands, pn the sum of the nominal exponents less
// 254); a group's terms align as floor(|t| * 2^(width - 1 - me)), an integer shift; their sum with the running accumulator is
// an exact integer below 2^32; the step's word keeps its top 24 bits.  A coordinate is flagged (`bad`) where `_chain` hands it
// back: a non-finite W word, or a step result at or past 2^128 or below 2^-126; its word is then 0.  No floating point.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

inline int bitlen(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }

// floor(m * 2^(c - d)) for d >= 0, m << c exact; 0 for m == 0 at any d
inline uint64_t align(uint64_t m, int c, int d) { return (m << c) >> std::min((unsigned)d, 63u); }

struct Row {
    const uint8_t* m;   // significand with its implicit bit (0 for a zero word)
    const int16_t* e;   // nominal exponent: max(field, 1), less 254 on the X side
    const uint8_t* s;   // sign
};

template <int G>
uint32_t chain(const Row& x, const Row& w, int K, int width, int floor_exp, bool flag, bool* bad) {
    bool neg = false;
    uint64_t A = 0;
    int ea = 0;
    for (int g = 0; g < K; g += G) {
        uint32_t prod[G];
        int pn[G];
        int pe = -30000;
        for (int p = 0; p < G; ++p) {
            prod[p] = (uint32_t)x.m[g + p] * w.m[g + p];
            pn[p] = x.e[g + p] + w.e[g + p];
            pe = std::max(pe, prod[p] ? pn[p] : -30000);
        }
        int me = std::max(pe, floor_exp);
        if (A) me = std::max(me, std::max(ea + bitlen(A) - 1, -126));
        int64_t tot = 0;
        for (int p = 0; p < G; ++p) {
            const int64_t t = (int64_t)align(prod[p], width - 15, me - pn[p]);
            const int64_t sg = -(int64_t)(x.s[g + p] ^ w.s[g + p]);
            tot += (t ^ sg) - sg;
        }
        if (A) {
            const int64_t t = (int64_t)align(A, width - 1, me - ea);
            tot += neg ? -t : t;
        }
        neg = tot < 0;
        const uint64_t T = (uint64_t)(neg ? -tot : tot);
        if (!T) {
            A = 0;
            continue;
        }
        const int L = bitlen(T), ye = me - width + L;
        flag |= ye >= 128 || ye < -126;
        A = L > 24 ? T >> (L - 24) : T;
        ea = me - width + 1 + std::max(L - 24, 0);
    }
    *bad = flag;
    if (!A || flag) return 0;
    const int L = bitlen(A);
    return (uint32_t(neg) << 31) | (uint32_t(ea + L - 1 + 127) << 23) | (uint32_t(A << (24 - L)) & 0x7FFFFFu);
}

void decode(const uint16_t* words, int n, int bias, uint8_t* m, int16_t* e, uint8_t* s, bool* nonfinite) {
    bool nf = false;
    for (int k = 0; k < n; ++k) {
        const int f = (words[k] >> 7) & 0xFF;
        m[k] = (uint8_t)((words[k] & 0x7F) | (f ? 0x80 : 0));
        e[k] = (int16_t)(std::max(f, 1) - bias);
        s[k] = words[k] >> 15;
        nf |= f == 0xFF;
    }
    if (nonfinite) *nonfinite = nf;
}

}  // namespace

extern "C" {

// X: M x K bf16 words; W: N x K bf16 words (row j = coordinate j); out / bad: M rows of `ld` entries, coordinate j at [j].
// G products per group (8 or 16), `width`-bit adder, exponent floor `floor_exp` (Ampere 8, 25, -132; Hopper 16, 26, -133).
// Returns 0, or -1 for a G it does not take.
int k16_chain(const uint16_t* X, int M, const uint16_t* W, long N, int K, int G, int width, int floor_exp, uint32_t* out,
              uint8_t* bad, long ld) {
    if (G != 8 && G != 16) return -1;
    std::vector<uint8_t> xm((size_t)M * K), xs((size_t)M * K), wm(K), ws(K);
    std::vector<int16_t> xe((size_t)M * K), we(K);
    for (int r = 0; r < M; ++r)
        decode(X + (size_t)r * K, K, 254, &xm[(size_t)r * K], &xe[(size_t)r * K], &xs[(size_t)r * K], nullptr);
    const Row w{wm.data(), we.data(), ws.data()};
    for (long j = 0; j < N; ++j) {
        bool wnf;
        decode(W + (size_t)j * K, K, 0, wm.data(), we.data(), ws.data(), &wnf);
        for (int r = 0; r < M; ++r) {
            const Row x{&xm[(size_t)r * K], &xe[(size_t)r * K], &xs[(size_t)r * K]};
            bool b;
            out[(size_t)r * ld + j] = G == 8 ? chain<8>(x, w, K, width, floor_exp, wnf, &b) : chain<16>(x, w, K, width, floor_exp, wnf, &b);
            bad[(size_t)r * ld + j] = b;
        }
    }
    return 0;
}

}  // extern "C"
