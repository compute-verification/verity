// verity flock-gpu-link: additive_ntt_shared_memory_tile's K layers (cuda-ghash/ntt_f128.cuh) with each thread's butterflies
// in registers. Included by ntt_f128.cuh (verity/ml/flock/cuda_chunk_patch.py) after the tile kernel, whose tiles, twiddles
// and butterflies it keeps: the tile is 2^K positions by lb lanes and layer L + j's butterfly (u, u + 2^(K-1-j)) takes
// twiddle 2^j - 1 + (u >> (K - j)). The layers go in rounds of at most three: a thread holds one lane's 2^R positions
// that round's R layers pair among themselves and applies them in order, so a position's values are the tile kernel's.
// The first round reads the source, the last writes data, and the tile passes through shared memory once between rounds
// (the tile kernel: once a layer). FC_NTT_REGS=0 launches the tile kernel.
#pragma once

#include <cstdlib>

// layers [J0, J0 + R) of the tile at gbase: group g's position i is (hi << (K - J0)) + (i << (K - J0 - R)) + lo, with
// hi = g >> (K - J0 - R) and lo = g's low K - J0 - R bits
template <int K, int J0, int R, bool FIRST, bool LAST>
__device__ __forceinline__ void ntt_tile_round(F128* data, const F128* src, long long smask, F128* sm, const F128* twid,
                                               long long gbase, long long stride, int log_lb) {
    constexpr int LOW = K - J0 - R;
    const int lbm = (1 << log_lb) - 1;
    const int items = (1 << (K - R)) << log_lb;
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int lin = it & lbm, g = it >> log_lb;
        const int hi = g >> LOW, lo = g & ((1 << LOW) - 1);
        const int p0 = (hi << (K - J0)) + lo;
        F128 x[1 << R];
#pragma unroll
        for (int i = 0; i < (1 << R); i++) {
            const int p = p0 + (i << LOW);
            x[i] = FIRST ? src[(gbase + p * stride + lin) & smask] : sm[(p << log_lb) + lin];
        }
#pragma unroll
        for (int jj = 0; jj < R; jj++) {
            const int h = 1 << (R - 1 - jj);
            const F128* tw = twid + (1 << (J0 + jj)) - 1 + (hi << jj);
#pragma unroll
            for (int i = 0; i < (1 << R); i++)
                if (!(i & h)) apply_ntt_butterfly(x, i, i + h, tw[i >> (R - jj)]);
        }
#pragma unroll
        for (int i = 0; i < (1 << R); i++) {
            const int p = p0 + (i << LOW);
            if (LAST) data[gbase + p * stride + lin] = x[i];
            else sm[(p << log_lb) + lin] = x[i];
        }
    }
}

template <int K>
__global__ void __launch_bounds__(256) ntt_tile_regs(F128* data, const F128* d_tw, int L, int log_d, int num_ntts, int log_lb,
                                                     const F128* src, long long smask) {
    extern __shared__ F128 sm[];
    constexpr int TILE = 1 << K, NTW = TILE - 1;
    const int lb = 1 << log_lb;
    const int log_seg = log_d - L - K;
    const long long block_size = 1LL << (log_d - L);
    int log_nlt = 0;
    while ((lb << log_nlt) < num_ntts) log_nlt++;
    const long long pos_tile = (long long)blockIdx.x >> log_nlt;
    const long long lane_tile = (long long)blockIdx.x & ((1 << log_nlt) - 1);
    const long long r = pos_tile & ((1LL << log_seg) - 1);
    const long long block = pos_tile >> log_seg;
    const long long gbase = block * block_size * (long long)num_ntts + r * (long long)num_ntts + lane_tile * (long long)lb;
    const long long stride = (long long)num_ntts << log_seg;
    F128* twid = sm + ((long long)TILE << log_lb);
    for (int t = threadIdx.x; t < NTW; t += blockDim.x) {
        const int j = 31 - __clz(t + 1), sub = (t + 1) - (1 << j);
        twid[t] = evaluate_ntt_twiddle(ntt_basis_for_layer(d_tw, L + j), L + j, (block << j) + sub);
    }
    __syncthreads();
    if constexpr (K <= 3) {
        ntt_tile_round<K, 0, K, true, true>(data, src, smask, sm, twid, gbase, stride, log_lb);
    } else if constexpr (K <= 6) {
        ntt_tile_round<K, 0, 3, true, false>(data, src, smask, sm, twid, gbase, stride, log_lb);
        __syncthreads();
        ntt_tile_round<K, 3, K - 3, false, true>(data, src, smask, sm, twid, gbase, stride, log_lb);
    } else {
        ntt_tile_round<K, 0, 3, true, false>(data, src, smask, sm, twid, gbase, stride, log_lb);
        __syncthreads();
        ntt_tile_round<K, 3, (K - 3) / 2 + (K - 3) % 2, false, false>(data, src, smask, sm, twid, gbase, stride, log_lb);
        __syncthreads();
        ntt_tile_round<K, 3 + (K - 3) / 2 + (K - 3) % 2, (K - 3) / 2, false, true>(data, src, smask, sm, twid, gbase, stride,
                                                                                   log_lb);
    }
}

inline bool ntt_tile_regs_on() {
    static const bool on = [] { const char* e = std::getenv("FC_NTT_REGS"); return !(e && e[0] == '0'); }();
    return on;
}

// at most `cap` layers a pass (the tile kernel's 7 under FC_NTT_REGS=0); FC_NTT_KMAX lowers it
inline int ntt_tile_kmax(int cap) {
    static const int k = [] { const char* e = std::getenv("FC_NTT_KMAX"); return e ? std::atoi(e) : 0; }();
    const int c = ntt_tile_regs_on() || cap < 7 ? cap : 7;
    return k >= 1 && k < c ? k : c;
}

template <int K>
inline void launch_ntt_tile_regs(F128* d_data, const F128* d_tw, int layer, int log_d, int num_ntts, int log_lb, size_t smem,
                                 const F128* src, long long smask) {
    cudaFuncSetAttribute(ntt_tile_regs<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    const int lb = 1 << log_lb;
    const long long tiles = (1LL << (log_d - K)) * (long long)(num_ntts / lb);
    const int per_round = ((1 << K) >> (K < 3 ? K : 3)) << log_lb;
    const int tpb = per_round < 256 ? per_round : 256;
    ntt_tile_regs<K><<<(unsigned)tiles, tpb, smem>>>(d_data, d_tw, layer, log_d, num_ntts, log_lb, src, smask);
}
