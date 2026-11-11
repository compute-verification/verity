// Verity: host-declared configuration of the FA3 M1 tap (see verity_tap_fa3.h).  Plain C++ so flash.h (host and
// device translation units) can carry it in Flash_fwd_params; the launch template copies it into the mainloop arguments.
#pragma once
#include <cstdint>

struct Verity_tap {
    void *acc_ptr;          // unused by the M1 tap (kept for parameter-layout parity with the FA2 tap)
    long acc_cap_words;
    float *diag_ptr;        // flat f32/u32 stream buffer, layout Verity_mat_layout(H, M, NB, BN, D)
    int mode;               // 0 off, 3 M1
    int H, M, N, NB;        // caps: H = b * h_k slabs, M = packed query rows (max_seqlen_q * qheads_per_kvhead), N = NB * BN, NB key blocks
    int tag;                // per-launch tag
    int BN, D;              // declared kernel tile (kBlockN, kHeadDim): the layout is arithmetic in it; must equal the selected kernel's
};
extern Verity_tap g_verity_tap;          // host-side config, defined in the torch-lib TU
extern int64_t g_verity_tap_skipped;     // launches on which the tap was disarmed (declaration != selected kernel / caps / unsupported)
