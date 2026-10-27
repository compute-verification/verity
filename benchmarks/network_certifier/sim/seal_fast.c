/* The sealed pair link's AEAD (`verity.ml.crypto.aead`) in C over OpenSSL's SHA-512: the fast path `mockdc.py seal`
 * measures beside the Python. Its units are the Python's, byte for byte: it prints the SHA-512 of its first M units,
 * which `mockdc.fast_path` compares with the Python's before it reports a rate.
 *
 *     seal_fast FRAME_BYTES FRAMES ENC_HEX MAC_HEX
 *
 * Frame k (k < M) is FRAME_BYTES bytes of k, sealed at nonce (window 0, bucket k, slot 0). The timed loops seal and open
 * those M frames FRAMES times in all: a measurement only, since no unit leaves the process. OpenSSL's SHA512_CTX and
 * its three calls are declared here, as <openssl/sha.h> has them since 0.9.8, so that only libcrypto is needed. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    unsigned long long h[8], Nl, Nh;
    union { unsigned long long d[16]; unsigned char p[128]; } u;
    unsigned int num, md_len;
} SHA512_CTX;
int SHA512_Init(SHA512_CTX *c);
int SHA512_Update(SHA512_CTX *c, const void *data, size_t len);
int SHA512_Final(unsigned char *md, SHA512_CTX *c);

enum { TAG = 64, NONCE = 16, M = 64 };
static unsigned char enc[64];
static SHA512_CTX inner, outer; /* HMAC-SHA512 under K_mac, after its ipad block and its opad block */

static void hmac_key(const unsigned char mac[64]) {
    unsigned char pad[128];
    for (int i = 0; i < 128; i++) pad[i] = (i < 64 ? mac[i] : 0) ^ 0x36;
    SHA512_Init(&inner);
    SHA512_Update(&inner, pad, 128);
    for (int i = 0; i < 128; i++) pad[i] = (i < 64 ? mac[i] : 0) ^ 0x5c;
    SHA512_Init(&outer);
    SHA512_Update(&outer, pad, 128);
}

static void tag(const unsigned char *n, const unsigned char *c, size_t len, unsigned char out[64]) {
    unsigned char d[64];
    SHA512_CTX x = inner;
    SHA512_Update(&x, n, NONCE);
    SHA512_Update(&x, c, len);
    SHA512_Final(d, &x);
    x = outer;
    SHA512_Update(&x, d, 64);
    SHA512_Final(out, &x);
}

/* out = in ^ (SHA-512(K_enc || n || u64be 0) || SHA-512(K_enc || n || u64be 1) || ...), one compression a block */
static void xor_stream(const unsigned char *n, const unsigned char *in, unsigned char *out, size_t len) {
    unsigned char block[88], ks[64];
    memcpy(block, enc, 64);
    memcpy(block + 64, n, NONCE);
    for (uint64_t c = 0; c * 64 < len; c++) {
        for (int i = 0; i < 8; i++) block[80 + i] = (unsigned char)(c >> (56 - 8 * i));
        SHA512_CTX x;
        SHA512_Init(&x);
        SHA512_Update(&x, block, sizeof block);
        SHA512_Final(ks, &x);
        size_t m = len - c * 64 < 64 ? len - c * 64 : 64;
        for (size_t i = 0; i < m; i++) out[c * 64 + i] = in[c * 64 + i] ^ ks[i];
    }
}

static void seal(const unsigned char *n, const unsigned char *frame, size_t len, unsigned char *unit) {
    xor_stream(n, frame, unit, len);
    tag(n, unit, len, unit + len);
}

static int unseal(const unsigned char *n, const unsigned char *unit, size_t len, unsigned char *frame) {
    unsigned char t[64], d = 0;
    tag(n, unit, len, t);
    for (int i = 0; i < 64; i++) d |= t[i] ^ unit[len + i];
    if (d) return 0;
    xor_stream(n, unit, frame, len);
    return 1;
}

static void nonce(uint32_t bucket, unsigned char n[NONCE]) {
    memset(n, 0, NONCE);
    for (int i = 0; i < 4; i++) n[8 + i] = (unsigned char)(bucket >> (24 - 8 * i));
}

static int unhex(const char *s, unsigned char *out, size_t n) {
    if (strlen(s) != 2 * n) return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(s + 2 * i, "%2x", &v) != 1) return 0;
        out[i] = (unsigned char)v;
    }
    return 1;
}

static double now(clockid_t id) {
    struct timespec t;
    clock_gettime(id, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    unsigned char mac[64], n[NONCE], digest[64];
    if (argc != 5 || !unhex(argv[3], enc, 64) || !unhex(argv[4], mac, 64)) {
        fprintf(stderr, "usage: seal_fast FRAME_BYTES FRAMES ENC_HEX MAC_HEX\n");
        return 2;
    }
    size_t B = strtoul(argv[1], NULL, 10), N = strtoul(argv[2], NULL, 10), U = B + TAG;
    hmac_key(mac);
    unsigned char *frames = malloc(M * B), *units = malloc(M * U), *out = malloc(B);
    for (int k = 0; k < M; k++) memset(frames + k * B, k, B);
    for (int k = 0; k < M; k++) {
        nonce(k, n);
        seal(n, frames + k * B, B, units + k * U);
    }
    SHA512_CTX x;
    SHA512_Init(&x);
    SHA512_Update(&x, units, M * U);
    SHA512_Final(digest, &x);
    size_t wrong = 0;
    for (int k = 0; k < M; k++) {
        nonce(k, n);
        wrong += !unseal(n, units + k * U, B, out) || memcmp(out, frames + k * B, B);
    }
    double c0 = now(CLOCK_PROCESS_CPUTIME_ID), w0 = now(CLOCK_MONOTONIC);
    for (size_t k = 0; k < N; k++) {
        nonce(k % M, n);
        seal(n, frames + (k % M) * B, B, units + (k % M) * U);
    }
    double c1 = now(CLOCK_PROCESS_CPUTIME_ID), w1 = now(CLOCK_MONOTONIC);
    size_t refused = 0;
    unsigned checksum = 0;
    for (size_t k = 0; k < N; k++) {
        nonce(k % M, n);
        refused += !unseal(n, units + (k % M) * U, B, out);
        checksum += out[k % B];
    }
    double c2 = now(CLOCK_PROCESS_CPUTIME_ID), w2 = now(CLOCK_MONOTONIC);
    printf("{\"frame_bytes\": %zu, \"frames\": %zu, \"digest\": \"", B, N);
    for (int i = 0; i < 64; i++) printf("%02x", digest[i]);
    printf("\", \"wrong\": %zu, \"refused\": %zu, \"checksum\": %u, \"seal_cpu_s\": %.6f, \"seal_wall_s\": %.6f, "
           "\"open_cpu_s\": %.6f, \"open_wall_s\": %.6f}\n", wrong, refused, checksum, c1 - c0, w1 - w0, c2 - c1, w2 - w1);
    return wrong || refused;
}
