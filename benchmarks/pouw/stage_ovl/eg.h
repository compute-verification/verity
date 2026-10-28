// ovl.cu's edited arms: EGEMM's edited GEMM with windowed duo receipts and the receipts' commitment, built in eg.cu (its
// own translation unit, so commit_gpu's constants for the receipts' 2 KiB rows sit beside ovl.cu's for X's tile leaves).
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace eg {
struct Lin;   // one chunk's linear stage: its X, W', P_A, P_B, receipts and their committer
void init();
void set_carve(int p);   // the receipts' committer kernels' preferred shared-memory carveout (ovl.cu's CARVE)
uint16_t* make_x(int rows, int k, uint32_t seed);   // the bf16 activations a quantizer reads (egemm.cu's xfill)
void fill_w(int8_t* W, size_t count, uint32_t seed);   // weights in [-63, 63], so W' = W + U stays int8
// m x k operand X (written by qwe, the committed one), x its bf16 source, Y (m x n s32); S the receipts' window in
// 32-step segments (k / 32 a multiple of it); the receipts' tree under the CGP1 file rparams in domain slot dom with
// `slots` salt slots
Lin* lin_new(int m, int n, int k, int S, int8_t* X, const uint16_t* x, int* Y, const std::string& rparams, int dom, int slots);
int lin_nw(const Lin* L);
size_t lin_leaves(const Lin* L);
size_t lin_rbytes(const Lin* L);
void lin_kernel(Lin* L, int kern);   // 0 w_d (lag 3, the IMAD tie every fourth MMA), 1 w_d4q (e_l4q's lag 4, LOP3 tie every fourth)
int lin_kernel_of(const Lin* L);
void lin_flags(Lin* L, bool fuse, bool pf);
// a call: X, P_A from x (dither) and W', P_B from W, under key; the GEMM under seed (the receipts' keys, the draw's)
void lin_qwe(Lin* L, const int8_t* W, uint32_t key, cudaStream_t st);
void lin_gemm(Lin* L, uint32_t seed, cudaStream_t st);
// plain serving's activation quantizer (quant<false>: round, no dither, no P_A) from x into a scratch operand of L's own
void lin_quant(Lin* L, cudaStream_t st);
void lin_rsalts(Lin* L, cudaStream_t st);              // every slot's salts, F and the pads (ahead)
void lin_rcommit(Lin* L, int slot, cudaStream_t st);   // the receipts' leaves and tree
void lin_rpoison(Lin* L);
void lin_root(const Lin* L, uint8_t* out64);
// the GEMM alone with kernel kern, the median of reps calls (ms)
double lin_time(Lin* L, int kern, int reps, cudaStream_t st);
// one call of both kernels on 0xA5-poisoned Y and R: Y against cuBLASLt's X W^T, the two kernels' receipts and Y against
// each other; with dir, X, W', R, the receipts' levels (after lin_rcommit at slot) and meta to dir for eref.py
std::string lin_gate(Lin* L, const int8_t* W, uint32_t key, uint32_t seed, int slot, const std::string& dir, bool& ok);
}  // namespace eg
