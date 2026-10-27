// The scalar pipes alone, the facets of the TS model's polytope that hold no MMA (TENSOR's M3 INT32-pipe
// assignment of LOP3, IMAD and LEA.HI; M4d's lanes a SM). One source for every family: build with
// ../opcode_probe/probe.py build --arch sm_90a | sm_100a | sm_120a; gated and poisoned as every opcode probe (probe.cuh).
//   s_<lane>            1024 threads (8 warps a SMSP), 64 ops of one lane an iteration (duo: 32 words, 3 ops each; a
//                       memory lane, lde, atome, redg, atoms or lds128: 32 accesses, LDS.128's 8)
//   p_<x>_<y>           the same with x and y alternating (independent chains): a pair rate above both alone rates
//                       means x and y issue to different pipes; at the alone rate, they share one
//   s_<lane>_w<W>       W warps a SMSP: the lane role's warps in sm90_wgmma.cu's wgU_* (W = 1), alone
// The beside-the-tensor-pipe rows are sm90_wgmma.cu's and sm100_tcgen05.cu's; dc/analyze.py reads all three.
#include "../../opcode_probe/probe.cuh"
#include "lanes.cuh"

constexpr int R = 64;

template <int X>
DC_DEV void s_alone(const OpArgs& a) {
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) { lane_ops<X, X == L_DUO ? 3 * R / 2 : (X == L_REDUCE ? R / 8 : R)>(l); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}
// A memory lane alone (lanes.cuh's mem_ops): R accesses a trip (LDS.128's 8: 32 would spill at 1024 threads).
template <int M, int R = 32>
DC_DEV void s_mem(const OpArgs& a) {
  constexpr int SX = 32 * MEM_SX;
  __shared__ __align__(16) unsigned char sx[SX];
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  unsigned char* gp = mem_global(a.g);
  const uint32_t sb = mem_shared<SX>(sx);
  OP_BEGIN
  OP_LOOP(it) { mem_ops<M, R, SX>(l, gp, sb, it); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}
template <int X, int Y>
DC_DEV void s_pair(const OpArgs& a) {
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) { lane_pair<X, Y, R>(l); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}

OP_KERNEL(s_mix, 1024) { s_alone<L_MIX>(a); }
OP_KERNEL(s_iadd, 1024) { s_alone<L_IADD>(a); }
OP_KERNEL(s_lop3, 1024) { s_alone<L_LOP3>(a); }
OP_KERNEL(s_imad, 1024) { s_alone<L_IMAD>(a); }
OP_KERNEL(s_ffma, 1024) { s_alone<L_FFMA>(a); }
OP_KERNEL(s_shf, 1024) { s_alone<L_SHF>(a); }
OP_KERNEL(s_lea, 1024) { s_alone<L_LEA>(a); }
OP_KERNEL(s_shfl, 1024) { s_alone<L_SHFL>(a); }
OP_KERNEL(s_redux, 1024) { s_alone<L_REDUX>(a); }
OP_KERNEL(s_reduce, 1024) { s_alone<L_REDUCE>(a); }
OP_KERNEL(s_duo, 1024) { s_alone<L_DUO>(a); }
OP_KERNEL(s_imadw, 1024) { s_alone<L_IMADW>(a); }
OP_KERNEL(s_lde, 1024) { s_mem<M_LDE>(a); }
OP_KERNEL(s_atome, 1024) { s_mem<M_ATOME>(a); }
OP_KERNEL(s_redg, 1024) { s_mem<M_REDG>(a); }
OP_KERNEL(s_atoms, 1024) { s_mem<M_ATOMS>(a); }
OP_KERNEL(s_lds128, 1024) { s_mem<M_LDS128, 8>(a); }
OP_KERNEL(p_imad_lop3, 1024) { s_pair<L_IMAD, L_LOP3>(a); }
OP_KERNEL(p_imad_iadd, 1024) { s_pair<L_IMAD, L_IADD>(a); }
OP_KERNEL(p_lop3_iadd, 1024) { s_pair<L_LOP3, L_IADD>(a); }
OP_KERNEL(p_ffma_imad, 1024) { s_pair<L_FFMA, L_IMAD>(a); }
OP_KERNEL(p_ffma_lop3, 1024) { s_pair<L_FFMA, L_LOP3>(a); }
OP_KERNEL(p_ffma_iadd, 1024) { s_pair<L_FFMA, L_IADD>(a); }
OP_KERNEL(p_shf_lop3, 1024) { s_pair<L_SHF, L_LOP3>(a); }
OP_KERNEL(p_shf_imad, 1024) { s_pair<L_SHF, L_IMAD>(a); }
OP_KERNEL(p_shf_ffma, 1024) { s_pair<L_SHF, L_FFMA>(a); }
OP_KERNEL(p_lea_lop3, 1024) { s_pair<L_LEA, L_LOP3>(a); }
OP_KERNEL(p_lea_imad, 1024) { s_pair<L_LEA, L_IMAD>(a); }
OP_KERNEL(p_lea_iadd, 1024) { s_pair<L_LEA, L_IADD>(a); }
OP_KERNEL(p_lea_ffma, 1024) { s_pair<L_LEA, L_FFMA>(a); }
OP_KERNEL(s_li_w1, 128) { s_alone<L_LI>(a); }
OP_KERNEL(s_lif_w1, 128) { s_alone<L_LIF>(a); }
OP_KERNEL(s_lop3_w1, 128) { s_alone<L_LOP3>(a); }
OP_KERNEL(s_imad_w1, 128) { s_alone<L_IMAD>(a); }
OP_KERNEL(s_mix_w1, 128) { s_alone<L_MIX>(a); }
OP_KERNEL(s_li_w2, 256) { s_alone<L_LI>(a); }

#define P(NAME, PAT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 1024, .expect = PAT, .iters = 4096}
static const OpProbe PROBES[] = {
    P(s_mix, "I*ADD*=32,FFMA=32"),
    P(s_iadd, "I*ADD*=64"),
    P(s_lop3, "LOP3.LUT=64"),
    P(s_imad, "IMAD=64"),
    P(s_ffma, "FFMA=64"),
    P(s_shf, "SHF.L.W.U32.HI=64"),
    P(s_lea, "LEA.HI=64"),
    P(s_shfl, "SHFL.BFLY=64"),
    P(s_redux, "REDUX*=64"),
    P(s_reduce, "SHFL.BFLY=40,LOP3.LUT=40"),
    P(s_duo, "LOP3.LUT=32,IMAD=32,LEA.HI=32"),
    P(s_imadw, "IMAD.WIDE.U32=64"),
    P(s_lde, "LD.E*=32"),
    P(s_atome, "ATOM.E*=32"),
    P(s_redg, "REDG*=32"),
    P(s_atoms, "ATOMS*=32"),
    P(s_lds128, "LDS.128=8"),
    P(p_imad_lop3, "IMAD=32,LOP3.LUT=32"),
    P(p_imad_iadd, "IMAD=32,I*ADD*=32"),
    P(p_lop3_iadd, "LOP3.LUT=32,I*ADD*=32"),
    P(p_ffma_imad, "FFMA=32,IMAD=32"),
    P(p_ffma_lop3, "FFMA=32,LOP3.LUT=32"),
    P(p_ffma_iadd, "FFMA=32,I*ADD*=32"),
    P(p_shf_lop3, "SHF.L.W.U32.HI=32,LOP3.LUT=32"),
    P(p_shf_imad, "SHF.L.W.U32.HI=32,IMAD=32"),
    P(p_shf_ffma, "SHF.L.W.U32.HI=32,FFMA=32"),
    P(p_lea_lop3, "LEA.HI=32,LOP3.LUT=32"),
    P(p_lea_imad, "LEA.HI=32,IMAD=32"),
    P(p_lea_iadd, "LEA.HI=32,I*ADD*=32"),
    P(p_lea_ffma, "LEA.HI=32,FFMA=32"),
#define PW(NAME, TPB, PAT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .iters = 4096}
    PW(s_li_w1, 128, "LOP3.LUT=32,IMAD=32"),
    PW(s_lif_w1, 128, "LOP3.LUT=28,IMAD=28,FFMA=8"),
    PW(s_lop3_w1, 128, "LOP3.LUT=64"),
    PW(s_imad_w1, 128, "IMAD=64"),
    PW(s_mix_w1, 128, "I*ADD*=32,FFMA=32"),
    PW(s_li_w2, 256, "LOP3.LUT=32,IMAD=32"),
#undef PW
};
#undef P

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
