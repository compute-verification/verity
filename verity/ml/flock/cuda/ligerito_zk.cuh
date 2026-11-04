// verity C-Flock ZK: Ligerito level 0's hiding in the F256 ladder (cuda-ghash/ligerito_f256.cuh), as
// verity's flock-zk patch adds it to the Rust ladder (flock_core::pcs::ligerito::zk; verity/ml/flock/live/PROTOCOL.md
// §8). Included by ligerito_f256.cuh before run_ligerito_f256, which calls these at five points when its `zk` argument
// is set (installed by verity/ml/flock/cuda_circuit_patch.py): the open label after level 0's cap, each OOD pair and
// lane challenge recorded, the lane phase's hiding step after the last lane fold and before the code switch, and level
// 0's queries recorded.
#pragma once

// The prover's level-0 hiding for one opening: the witness's lanes, every lane's padding (lanes + 4 lanes of t_pad words,
// lane-major) and the 4 extra lanes (L words each), on the device; the repetition whose pair this opening folds; what the lane
// phase sent (e, T', ybar) and saw (the pre-OOD target, (β, y) per OOD sample, the lane challenges, ρ, level 0's queries).
struct LigZk {
    int rep = 0, t_pad = 0, lanes = 0;
    // a dishonest prover (the negatives, as Level0Zk::tamper): 1 folds another first extra lane than the committed one, 2 sends
    // another T', 3 another ybar, 4 another e_0 (with the honest T')
    int tamper = 0;
    const F128* d_pads = nullptr;
    const F128* d_extra = nullptr;
    F256Ext e[2]{}, t_prime{}, rho[2]{};
    std::vector<F256Ext> ybar, challenges;
    F128 target{0, 0};
    std::vector<std::pair<F128, F128>> ood;
    std::vector<size_t> queries0;
};

static const uint8_t LF_ZK_OPEN[] = "verity-flock-zk-l0-open-v0";
static const uint8_t LF_ZK_CLOSE[] = "verity-flock-zk-l0-close-v0";

// e_i = Σ_x b[x]·R_i[x]: block partials (e_0 in p0, e_1 in p1), then lf_msg_combine.
__global__ void lf_zk_sums_partial(const F256Ext* __restrict__ B, const F128* __restrict__ R0, const F128* __restrict__ R1,
                                   long long n, F256Ext* p0, F256Ext* p1) {
    __shared__ F256Ext s0[LF_TPB];
    __shared__ F256Ext s1[LF_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F256Ext e0 = f256x_zero(), e1 = f256x_zero();
    for (long long x = t; x < n; x += stride) {
        e0 = f256x_add(e0, f256x_mul_base(B[x], R0[x]));
        e1 = f256x_add(e1, f256x_mul_base(B[x], R1[x]));
    }
    int i = threadIdx.x;
    s0[i] = e0; s1[i] = e1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) { s0[i] = f256x_add(s0[i], s0[i + s]); s1[i] = f256x_add(s1[i], s1[i + s]); }
        __syncthreads();
    }
    if (i == 0) { p0[blockIdx.x] = s0[0]; p1[blockIdx.x] = s1[0]; }
}

// f[x] += ρ_0·R_0[x] + ρ_1·R_1[x] in place, and block partials of T' = Σ_x f[x]·b[x] (p0; p2 stays zero for lf_msg_combine).
__global__ void lf_zk_blind_partial(F256Ext* __restrict__ F, const F256Ext* __restrict__ B, const F128* __restrict__ R0,
                                    const F128* __restrict__ R1, F256Ext rho0, F256Ext rho1, long long n, int flip0,
                                    F256Ext* p0, F256Ext* p2) {
    __shared__ F256Ext s0[LF_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F256Ext acc = f256x_zero();
    for (long long x = t; x < n; x += stride) {
        F128 r0 = R0[x];
        if (flip0 && x == 0) r0.lo ^= 1;
        F256Ext f = f256x_add(F[x], f256x_add(f256x_mul_base(rho0, r0), f256x_mul_base(rho1, R1[x])));
        F[x] = f;
        acc = f256x_add(acc, f256x_mul(f, B[x]));
    }
    int i = threadIdx.x;
    s0[i] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) s0[i] = f256x_add(s0[i], s0[i + s]);
        __syncthreads();
    }
    if (i == 0) { p0[blockIdx.x] = s0[0]; p2[blockIdx.x] = f256x_zero(); }
}

// ybar[i] = ρ_0·pad_{lanes+2rep}[i] + ρ_1·pad_{lanes+2rep+1}[i] + Σ_l eq(r)[l]·pad_l[i], eq in flock's low-to-high order
// (challenge j weighs bit j of the lane).
__global__ void lf_zk_ybar(const F128* __restrict__ pads, int lanes, int t_pad, int rep, const F256Ext* __restrict__ r, int k,
                           F256Ext rho0, F256Ext rho1, F256Ext* __restrict__ ybar) {
    extern __shared__ F256Ext w[];
    const F256Ext one = f256x_from_base(F128{1, 0});
    for (int l = threadIdx.x; l < lanes; l += blockDim.x) {
        F256Ext acc = one;
        for (int j = 0; j < k; j++) acc = f256x_mul(acc, ((l >> j) & 1) ? r[j] : f256x_add(one, r[j]));
        w[l] = acc;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t_pad) return;
    const F128* p0 = pads + (size_t)(lanes + 2 * rep) * t_pad;
    F256Ext acc = f256x_add(f256x_mul_base(rho0, p0[i]), f256x_mul_base(rho1, p0[t_pad + i]));
    for (int l = 0; l < lanes; l++) acc = f256x_add(acc, f256x_mul_base(w[l], pads[(size_t)l * t_pad + i]));
    ybar[i] = acc;
}

inline void lf_zk_open(FsChallenger& ch, LigZk& zk, F128 target) {
    ch.observe_label(LF_ZK_OPEN, sizeof(LF_ZK_OPEN) - 1);
    zk.target = target;
}

inline bool lf_zk_rho_independent(const F256Ext rho[2]) {
    F128 d = f128_add_hd(f128_mul_hd(rho[0].c0, rho[1].c1), f128_mul_hd(rho[0].c1, rho[1].c0));
    return d.lo != 0 || d.hi != 0;
}

inline void lf_zk_observe256(FsChallenger& ch, F256Ext v) {
    ch.observe_f256(ChF256{lf_toch(v.c0), lf_toch(v.c1)});
}

// Level 0's hiding step, after the last lane fold (f and b of n elements, the message length L) and before the code switch:
// e for this rep's pair (sent before ρ), ρ (two squeezes), the blind f += ρ_0 R_0 + ρ_1 R_1 with T' = Σ f·b, the close
// label, T' and ybar. Returns 240 when ρ_0, ρ_1 are dependent over F128 (the prover stops: degenerate coins).
inline int lf_zk_lane_phase(FsChallenger& ch, LigZk& zk, F256Ext* d_f, const F256Ext* d_b, long long n, F256Ext* d_p0, F256Ext* d_p2,
                            F256Ext* d_out2, const LigAlloc& A) {
    long long blocks_ll = std::min<long long>(std::max<long long>(1, (n + LF_TPB - 1) / LF_TPB), LF_MAX_BLOCKS);
    int blocks = (int)blocks_ll;
    const F128* R0 = zk.d_extra + (size_t)(2 * zk.rep) * n;
    const F128* R1 = zk.d_extra + (size_t)(2 * zk.rep + 1) * n;
    F256Ext h[2];
    lf_zk_sums_partial<<<blocks, LF_TPB>>>(d_b, R0, R1, n, d_p0, d_p2);
    lf_msg_combine<<<1, LF_TPB>>>(d_p0, d_p2, blocks, d_out2);
    LFCK(cudaGetLastError());
    LFCK(cudaMemcpy(h, d_out2, sizeof(h), cudaMemcpyDeviceToHost));
    zk.e[0] = h[0];
    zk.e[1] = h[1];
    if (zk.tamper == 4) zk.e[0].c0.lo ^= 1;
    lf_zk_observe256(ch, zk.e[0]);
    lf_zk_observe256(ch, zk.e[1]);
    zk.rho[0] = lf_frch256(ch.sample_f256());
    zk.rho[1] = lf_frch256(ch.sample_f256());
    if (!lf_zk_rho_independent(zk.rho)) {
        printf("PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128\n");
        return 240;
    }
    lf_zk_blind_partial<<<blocks, LF_TPB>>>(d_f, d_b, R0, R1, zk.rho[0], zk.rho[1], n, zk.tamper == 1, d_p0, d_p2);
    lf_msg_combine<<<1, LF_TPB>>>(d_p0, d_p2, blocks, d_out2);
    LFCK(cudaGetLastError());
    LFCK(cudaMemcpy(h, d_out2, sizeof(h), cudaMemcpyDeviceToHost));
    zk.t_prime = h[0];
    if (zk.tamper == 2) zk.t_prime.c0.lo ^= 1;
    ch.observe_label(LF_ZK_CLOSE, sizeof(LF_ZK_CLOSE) - 1);
    lf_zk_observe256(ch, zk.t_prime);
    const int k = (int)zk.challenges.size();
    F256Ext *d_r, *d_ybar;
    LFCK(A.alloc((void**)&d_r, std::max(k, 1) * sizeof(F256Ext)));
    LFCK(A.alloc((void**)&d_ybar, std::max(zk.t_pad, 1) * sizeof(F256Ext)));
    LFCK(cudaMemcpy(d_r, zk.challenges.data(), k * sizeof(F256Ext), cudaMemcpyHostToDevice));
    lf_zk_ybar<<<(unsigned)((zk.t_pad + LF_TPB - 1) / LF_TPB), LF_TPB, zk.lanes * sizeof(F256Ext)>>>(
        zk.d_pads, zk.lanes, zk.t_pad, zk.rep, d_r, k, zk.rho[0], zk.rho[1], d_ybar);
    LFCK(cudaGetLastError());
    zk.ybar.resize(zk.t_pad);
    LFCK(cudaMemcpy(zk.ybar.data(), d_ybar, zk.t_pad * sizeof(F256Ext), cudaMemcpyDeviceToHost));
    LFCK(A.release(d_r));
    LFCK(A.release(d_ybar));
    if (zk.tamper == 3) zk.ybar[0].c0.lo ^= 1;
    std::vector<ChF128> coords(2 * (size_t)zk.t_pad);
    for (int i = 0; i < zk.t_pad; i++) { coords[2 * i] = lf_toch(zk.ybar[i].c0); coords[2 * i + 1] = lf_toch(zk.ybar[i].c1); }
    ch.observe_f128_slice(coords.data(), coords.size());
    return 0;
}
