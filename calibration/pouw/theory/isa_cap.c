/* Two radix-M packings that put more incidences into one sm_120 instruction than ISA-CAP (<= 32/7), checked
   over every R bit and U entry (U in -64..-1, 1..64), with each instruction evaluated from its packed registers and its
   destination decoded as a mixed-radix tuple of fields:

   IDP.4A.U8.S8  a = (r1 + M r2_0) | M r2_1 << 8 | M r2_2 << 16 | M r2_3 << 24,  b = U[j, p_0..p_3] as s8 bytes:
                 one 32-bit word holding r1·u_0 (1 entry) and Σ_k r2_k·u_k (4 entries), 5 incidences;
   IMAD.WIDE     signed s32 × s32 + s64, a = r1 + 2^29 r2, b = 8 Σ_j M^j u_j over 4 columns: four 1-entry fields
                 r1·u_j in the low word and four r2·u_j in the high word, 8 incidences.

   At M = 129 (a 1-entry field takes 129 values) both hold everywhere; M = 128 is the mutation and must fail.
   usage: isa_cap       prints  IDP4A cases fails  and  IMADW cases fails
   build: gcc -O2 -fopenmp -DRADIX=M -o isa_cap isa_cap.c */
#include <stdint.h>
#include <stdio.h>

static inline uint32_t idp4a_u8s8(uint32_t a, uint32_t b, uint32_t c) {
    int64_t d = c;
    for (int k = 0; k < 4; k++) d += (int64_t)((a >> (8 * k)) & 0xFF) * (int8_t)((b >> (8 * k)) & 0xFF);
    return (uint32_t)d;
}

static inline uint64_t imad_wide_s32(uint32_t a, uint32_t b, uint64_t c) {
    return (uint64_t)((int64_t)(int32_t)a * (int64_t)(int32_t)b) + c;
}

static inline int uval(uint64_t idx, int k) {
    int i = (int)((idx >> (7 * k)) & 127);
    return i < 64 ? i - 64 : i - 63;
}

int main(void) {
    const int64_t M = RADIX;
    const uint64_t n = 1ull << 28;

    uint64_t fails = 0;
    for (int rm = 0; rm < 32; rm++) {
        int r1 = rm & 1, r2[4];
        for (int k = 0; k < 4; k++) r2[k] = (rm >> (1 + k)) & 1;
        uint32_t a = (uint32_t)(r1 + M * r2[0]);
        for (int k = 1; k < 4; k++) a |= (uint32_t)(M * r2[k]) << (8 * k);
        const uint32_t c = (uint32_t)(64 + M * 256);
#pragma omp parallel for reduction(+ : fails) schedule(static)
        for (uint64_t idx = 0; idx < n; idx++) {
            int u[4];
            uint32_t b = 0;
            int64_t hi_want = 0;
            for (int k = 0; k < 4; k++) {
                u[k] = uval(idx, k);
                b |= (uint32_t)(uint8_t)(int8_t)u[k] << (8 * k);
                hi_want += r2[k] * u[k];
            }
            uint32_t d = idp4a_u8s8(a, b, c);
            fails += (int64_t)(d % M) - 64 != r1 * u[0] || (int64_t)(d / M) - 256 != hi_want;
        }
    }
    printf("IDP4A %llu %llu\n", (unsigned long long)(32 * n), (unsigned long long)fails);

    fails = 0;
    int64_t s64 = 0, top = 1;
    for (int j = 0; j < 4; j++) s64 += 64 * top, top *= M;
    for (int rm = 0; rm < 4; rm++) {
        int r1 = rm & 1, r2 = rm >> 1;
        const uint32_t a = (uint32_t)r1 | (uint32_t)r2 << 29;
        const uint64_t c = (uint64_t)(8 * s64) + ((uint64_t)s64 << 32);
#pragma omp parallel for reduction(+ : fails) schedule(static)
        for (uint64_t idx = 0; idx < n; idx++) {
            int u[4];
            int64_t b = 0, pw = 1;
            for (int j = 0; j < 4; j++) u[j] = uval(idx, j), b += 8 * pw * u[j], pw *= M;
            if (b < INT32_MIN || b > INT32_MAX) { fails++; continue; }
            uint64_t d = imad_wide_s32(a, (uint32_t)(int32_t)b, c);
            uint64_t lo = d & 0xFFFFFFFFull, hi = d >> 32;
            int bad = lo % 8 != 0;
            lo /= 8;
            for (int j = 0; j < 4; j++) {
                bad |= (int64_t)(lo % M) - 64 != r1 * u[j] || (int64_t)(hi % M) - 64 != r2 * u[j];
                lo /= M, hi /= M;
            }
            fails += bad || lo != 0 || hi != 0;
        }
    }
    printf("IMADW %llu %llu\n", (unsigned long long)(4 * n), (unsigned long long)fails);
    return 0;
}
