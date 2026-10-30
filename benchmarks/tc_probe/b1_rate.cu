// Sustained `mma.sync` rate of the 1-bit instructions against the 8-bit one, per SM clock:
//
//   b1_and   mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc   32,768 1-bit MACs a warp instruction
//   b1_xor   mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc   32,768
//   s8       mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32             4,096
//
// Register-resident: one block on every SM, WARPS warps a block, CHAINS independent accumulators a warp, operands loaded
// once.  Thread 0 of each block stamps clock64, %globaltimer and %smid after a barrier at both ends, so a block's rate is its
// own SM's clocks.  Each (instruction, WARPS) runs a short calibration and then at least SUSTAIN_S seconds; one tile is first
// checked against a host reference.  One JSON line per measurement on stdout.
//
//   nvcc -O3 -gencode arch=compute_90a,code=sm_90a -o b1_rate b1_rate.cu && ./b1_rate [SUSTAIN_S]

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));           \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

constexpr int CHAINS = 8;
enum Op { B1_AND = 0, B1_XOR = 1, S8 = 2 };

template <int OP>
__device__ __forceinline__ void mma(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    if (OP == B1_AND) {
        asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if (OP == B1_XOR) {
        asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

struct Stamp {
    long long c0, c1;
    unsigned long long t0, t1;
    int smid, pad;
};

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// in: 32 lanes x 6 operand words (a0..a3, b0, b1).  dbg: lane x 4 words of chain 0 of block 0's first warp.
template <int OP>
__global__ void rate_kernel(const uint32_t* __restrict__ in, int* __restrict__ out, int* __restrict__ dbg,
                            Stamp* __restrict__ st, long long iters) {
    const int lane = threadIdx.x & 31;
    uint32_t a[4], b[2];
    for (int i = 0; i < 4; ++i) a[i] = in[lane * 6 + i];
    b[0] = in[lane * 6 + 4];
    b[1] = in[lane * 6 + 5];
    int acc[CHAINS][4];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[c][j] = 0;
    __syncthreads();
    long long c0 = clock64();
    unsigned long long t0 = globaltimer();
#pragma unroll 1
    for (long long it = 0; it < iters; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) mma<OP>(acc[c], a, b);
    }
    __syncthreads();
    long long c1 = clock64();
    unsigned long long t1 = globaltimer();
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        st[blockIdx.x] = Stamp{c0, c1, t0, t1, smid, 0};
    }
    if (blockIdx.x == 0 && threadIdx.x < 32)
        for (int j = 0; j < 4; ++j) dbg[lane * 4 + j] = acc[0][j];
    int s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c)
#pragma unroll
        for (int j = 0; j < 4; ++j) s += acc[c][j] * (c + 1);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// D of one tile from the lanes' operand words.  Fragments (PTX ISA, m16n8k256 b1 and m16n8k32 s8): lane L, g = L>>2,
// t = L&3; a0/a1 hold rows g/g+8 at word t of the first half of K, a2/a3 the same in the second half; b0/b1 hold column g
// at word t of each half; d0..d3 are (g,2t), (g,2t+1), (g+8,2t), (g+8,2t+1).  A and B map an element to the same bit
// or byte of a word, so the reference needs no element order inside a word.
static void reference(int op, const std::vector<uint32_t>& in, int d[32][4]) {
    uint32_t A[16][8], B[8][8];
    for (int L = 0; L < 32; ++L) {
        int g = L >> 2, t = L & 3;
        A[g][t] = in[L * 6 + 0];
        A[g + 8][t] = in[L * 6 + 1];
        A[g][4 + t] = in[L * 6 + 2];
        A[g + 8][4 + t] = in[L * 6 + 3];
        B[g][t] = in[L * 6 + 4];
        B[g][4 + t] = in[L * 6 + 5];
    }
    auto dot = [&](int r, int c) {
        int s = 0;
        for (int w = 0; w < 8; ++w) {
            uint32_t x = A[r][w], y = B[c][w];
            if (op == B1_AND) s += __builtin_popcount(x & y);
            else if (op == B1_XOR) s += __builtin_popcount(x ^ y);
            else
                for (int k = 0; k < 4; ++k) s += int(int8_t(x >> (8 * k))) * int(int8_t(y >> (8 * k)));
        }
        return s;
    };
    for (int L = 0; L < 32; ++L) {
        int g = L >> 2, t = L & 3;
        d[L][0] = dot(g, 2 * t);
        d[L][1] = dot(g, 2 * t + 1);
        d[L][2] = dot(g + 8, 2 * t);
        d[L][3] = dot(g + 8, 2 * t + 1);
    }
}

typedef void (*Kernel)(const uint32_t*, int*, int*, Stamp*, long long);

static const char* NAMES[] = {"b1_and", "b1_xor", "s8"};
static const char* PTX[] = {"mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc",
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc",
                            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"};
static const long long MACS[] = {16 * 8 * 256, 16 * 8 * 256, 16 * 8 * 32};
static const Kernel KERNELS[] = {rate_kernel<B1_AND>, rate_kernel<B1_XOR>, rate_kernel<S8>};

struct Result {
    double wall_s, macs_per_clk_sm, clk_mhz, min_sm, max_sm;
    int sms;
};

static Result run(int op, int sms, int warps, long long iters, const uint32_t* in, int* out, int* dbg, Stamp* st) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    KERNELS[op]<<<sms, 32 * warps>>>(in, out, dbg, st, iters);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    std::vector<Stamp> h(sms);
    CK(cudaMemcpy(h.data(), st, sms * sizeof(Stamp), cudaMemcpyDeviceToHost));
    // Per SM: its blocks' MACs over its own span of clocks (one block an SM is the expected placement).
    std::map<int, std::pair<long long, long long>> span;
    std::map<int, int> blocks;
    double cyc = 0, ns = 0;
    for (auto& s : h) {
        auto it = span.find(s.smid);
        if (it == span.end()) span[s.smid] = {s.c0, s.c1};
        else it->second = {std::min(it->second.first, s.c0), std::max(it->second.second, s.c1)};
        blocks[s.smid]++;
        cyc += double(s.c1 - s.c0);
        ns += double(s.t1 - s.t0);
    }
    const double per_block = double(iters) * CHAINS * warps * MACS[op];
    double tot_cyc = 0, mn = 1e30, mx = 0;
    for (auto& kv : span) {
        double c = double(kv.second.second - kv.second.first);
        double r = per_block * blocks[kv.first] / c;
        tot_cyc += c;
        mn = std::min(mn, r);
        mx = std::max(mx, r);
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return Result{ms / 1e3, per_block * sms / tot_cyc, cyc / ns * 1e3, mn, mx, int(span.size())};
}

int main(int argc, char** argv) {
    const double sustain_s = argc > 1 ? atof(argv[1]) : 2.5;
    int dev = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    sms = p.multiProcessorCount;
    printf("{\"device\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"chains\": %d}\n", p.name, p.major, p.minor, sms, CHAINS);

    std::vector<uint32_t> hin(32 * 6);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (auto& w : hin) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        w = uint32_t(x >> 16);
    }
    uint32_t* in;
    int *out, *dbg;
    Stamp* st;
    CK(cudaMalloc(&in, hin.size() * 4));
    CK(cudaMalloc(&out, size_t(sms) * 1024 * 4));
    CK(cudaMalloc(&dbg, 32 * 4 * 4));
    CK(cudaMalloc(&st, sms * sizeof(Stamp)));
    CK(cudaMemcpy(in, hin.data(), hin.size() * 4, cudaMemcpyHostToDevice));

    const int warp_counts[] = {4, 8, 16, 32};
    for (int op = 0; op < 3; ++op) {
        KERNELS[op]<<<1, 32>>>(in, out, dbg, st, 1);
        CK(cudaDeviceSynchronize());
        int hd[32][4], ref[32][4];
        CK(cudaMemcpy(hd, dbg, sizeof(hd), cudaMemcpyDeviceToHost));
        reference(op, hin, ref);
        int bad = 0;
        for (int L = 0; L < 32; ++L)
            for (int j = 0; j < 4; ++j) bad += hd[L][j] != ref[L][j];
        printf("{\"op\": \"%s\", \"ptx\": \"%s\", \"check\": \"%s\", \"mismatches\": %d, \"d0\": %d, \"ref0\": %d}\n",
               NAMES[op], PTX[op], bad ? "mismatch" : "ok", bad, hd[0][0], ref[0][0]);
        fflush(stdout);
        for (int warps : warp_counts) {
            const long long cal = 4096;
            Result r0 = run(op, sms, warps, cal, in, out, dbg, st);
            long long iters = (long long)(cal * sustain_s / std::max(r0.wall_s, 1e-6)) + 1;
            Result r = run(op, sms, warps, iters, in, out, dbg, st);
            printf("{\"op\": \"%s\", \"warps_per_sm\": %d, \"iters\": %lld, \"wall_s\": %.3f, \"sms_seen\": %d, "
                   "\"macs_per_instr\": %lld, \"macs_per_clk_sm\": %.1f, \"min_sm\": %.1f, \"max_sm\": %.1f, "
                   "\"sm_clk_mhz\": %.0f, \"instr_per_clk_sm\": %.4f}\n",
                   NAMES[op], warps, iters, r.wall_s, r.sms, MACS[op], r.macs_per_clk_sm, r.min_sm, r.max_sm, r.clk_mhz,
                   r.macs_per_clk_sm / MACS[op]);
            fflush(stdout);
        }
    }
    return 0;
}
