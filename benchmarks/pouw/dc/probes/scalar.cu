// EPS1 DC: the scalar pipes alone, the facets of the TS model's polytope that hold no MMA (TENSOR's M3 INT32-pipe
// assignment of LOP3, IMAD and LEA.HI; M4d's lanes a SM). One source for every family: build with
// ../opcode_probe/probe.py build --arch sm_90a | sm_100a | sm_120a; gated and poisoned as every opcode probe (probe.cuh).
//   s_<lane>            1024 threads (8 warps a SMSP), 64 ops of one lane an iteration (duo: 32 words, 3 ops each)
//   p_<x>_<y>           the same with x and y alternating (independent chains): a pair rate above both alone rates
//                       means x and y issue to different pipes; at the alone rate, they share one
// The beside-the-tensor-pipe rows are sm90_wgmma.cu's and sm100_tcgen05.cu's; dc/analyze.py reads all three.
#include "../../opcode_probe/probe.cuh"
#include "lanes.cuh"

constexpr int R = 64;

template <int X>
DC_DEV void s_alone(const OpArgs& a) {
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) { lane_ops<X, X == L_DUO ? 3 * R / 2 : R>(l); }
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
OP_KERNEL(s_duo, 1024) { s_alone<L_DUO>(a); }
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
    P(s_duo, "LOP3.LUT=32,IMAD=32,LEA.HI=32"),
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
};
#undef P

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
