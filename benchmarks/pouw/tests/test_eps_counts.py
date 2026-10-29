"""sass_census's eps_counts: ε_skip's per-instruction counts from `nvdisasm -c -gi` SASS.

On a synthetic apv.cu and a listing in nvdisasm's inline-chain format: an instruction's chain is the run of `//## File`
lines above it and one with none inherits the last; a converged warp never takes `BRA.DIV`, so the shuffle fallbacks
after the EXIT don't count; the tail is apv_l7qs's atail call less apv_cc's fold, a warp-tile's 8 tiles, thread 0's TMA
issue at 1/16; vfold's loop runs s / 1024 rounds; and each architecture is read against sm_120a's by the difference.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SC = Path(__file__).resolve().parents[1] / "sass_census"
spec = importlib.util.spec_from_file_location("eps_counts", SC / "eps_counts.py")
E = importlib.util.module_from_spec(spec)
sys.modules["eps_counts"] = E
spec.loader.exec_module(E)

SRC = """\
DEV void issue() {
  tma_load();
}
template <int OUT> DEV void atail() {
  imma();
  issue();
}
DEV void body() {
  if constexpr (OUT < 2)
    ml.template atail<OUT>(acc, ch, dg, g, q);
  else cabs_all<L>(ch, acc);
  if constexpr (OUT == 1) {
  } else if constexpr (OUT != 2) {
    store();
  }
}
DEV void vfold() {
  for (int i = threadIdx.x; i < nC; i += VF_NT) {
    sha();
  }
  for (int c0 = 0; c0 < nC; c0 += 8) {
    x();
    if (threadIdx.x < 64) {
      scan();
    }
  }
}
"""

F = '"/t/x/apv.cu"'
SASS = f"""\
.text._Z8apv_l7qs6ApArgs:
\t//## File {F}, line 8
        /*0000*/                   S2R R0, SR_TID.X ;
\t//## File {F}, line 5 inlined at {F}, line 10
\t//## File {F}, line 10 inlined at {F}, line 8
\t//## File {F}, line 8
        /*0010*/                   IMMA.16832.S8.S8 R4, R8.ROW, R12.COL, R4 ;
        /*0020*/                   IMAD R5, R6, 0x100, R5 ;
\t//## File {F}, line 2 inlined at {F}, line 6
\t//## File {F}, line 6 inlined at {F}, line 10
\t//## File {F}, line 10 inlined at {F}, line 8
        /*0030*/              @P0  UTMALDG.2D [UR8], [UR4] ;
\t//## File {F}, line 14 inlined at {F}, line 8
        /*0040*/                   STG.E [R2.64], R4 ;
        /*0050*/                   BRA.DIV UR4, `(.L_x_1) ;
        /*0060*/                   EXIT ;
.L_x_1:
\t//## File {F}, line 5 inlined at {F}, line 10
        /*0070*/                   WARPSYNC.ALL ;
        /*0080*/                   BRA `(.L_x_1) ;
.text._Z6apv_cc6ApArgs:
\t//## File {F}, line 11 inlined at {F}, line 8
        /*0000*/                   LOP3.LUT R1, R2, R3, R4, 0x96, !PT ;
\t//## File {F}, line 14 inlined at {F}, line 8
        /*0010*/                   STG.E [R2.64], R4 ;
        /*0020*/                   EXIT ;
.text._Z5other:
        /*0000*/                   EXIT ;
"""


def test_regions_find_the_counted_lines():
    R = E.regions(SRC)
    assert (R["issue"], R["atail"], R["tail_call"], R["cc_fold"], R["epilogue"]) == ((1, 3), (4, 7), (10, 10), (11, 11),
                                                                                       (13, 15))
    assert R["y2s"] == (0, -1)
    assert (R["vf_hash"], R["vf_loop"], R["vf_tid64"]) == ((18, 20), (21, 26), (23, 25))


def test_parse_reads_each_instructions_whole_chain():
    fns = E.parse(SASS, "|".join(E.KERNELS.values()))
    assert set(fns) == {"_Z8apv_l7qs6ApArgs", "_Z6apv_cc6ApArgs"}   # `keep` drops the rest
    l7 = fns["_Z8apv_l7qs6ApArgs"]
    assert [i.op for i in l7] == ["S2R", "IMMA.16832.S8.S8", "IMAD", "UTMALDG.2D", "STG.E", "BRA.DIV", "EXIT",
                                  "WARPSYNC.ALL", "BRA"]
    assert l7[1].frames == (("apv.cu", 5), ("apv.cu", 10), ("apv.cu", 8))
    assert l7[2].frames == l7[1].frames   # no `//##` above it: the IMMA's chain
    assert l7[3].frames == (("apv.cu", 2), ("apv.cu", 6), ("apv.cu", 10), ("apv.cu", 8))


def test_a_converged_warp_never_takes_bra_div():
    l7 = E.parse(SASS)["_Z8apv_l7qs6ApArgs"]
    assert [i.live for i in l7] == [True] * 7 + [False, False]


def test_the_tail_is_the_atail_call_less_apv_ccs_fold():
    R, ks = E.regions(SRC), E.kernels(E.parse(SASS))
    t = E.tail(ks["apv_l7qs"], ks["apv_cc"], R)
    assert dict(t) == pytest.approx({"IMMA": 1 / 8, "IMAD": 1 / 8, "UTMALDG": 1 / 128, "LOP3": -1 / 8})
    assert E.collapse(t) == pytest.approx({"IMMA": 1 / 8, "IMAD": 1 / 8, "ALU": 1 / 128 - 1 / 8})
    # BRA.DIV and EXIT have no `//##` lines of their own, so they're the STG's line; the untaken path isn't counted
    assert dict(E.epilogue(ks["apv_l7qs"], R)) == {"STG": 1 / 8, "BRA": 1 / 8, "EXIT": 1 / 8}


def test_vfolds_weights():
    R = E.regions(SRC)
    ins = lambda n: E.Instr(0, "IADD3", (("apv.cu", n),))  # noqa: E731
    assert E.vf_weight(ins(22), R, 8192) == 8                 # the loop body, s / 1024 rounds
    assert E.vf_weight(ins(24), R, 8192) == 8 * 2 / 8         # threadIdx.x < 64: 2 of the 8 warps
    assert E.vf_weight(ins(19), R, 2048) == 1 / 8             # 16 digests: one warp's threads hash
    assert E.vf_weight(ins(19), R, 32768) == 1.0              # 256: every warp's
    assert E.vf_weight(ins(17), R, 2048) == 1.0


def test_each_architecture_is_read_against_sm120a():
    res = {"t": {"sm_120a": {"tail": {"classes": {"IMMA": 4, "IMAD": 4.5, "ALU": 12.0}, "total": 20.5},
                             "serve": {"total": 4.5}},
                 "sm_90a": {"tail": {"classes": {"IMMA": 4, "IMAD": 7.5, "ALU": 16.0}, "total": 27.5},
                            "serve": {"total": 6.0}}}}
    d = E.against(res)["t"]["sm_90a"]
    assert d["tail.IMAD"]["delta"] == 3.0 and d["tail.IMAD"]["estimate"] == E.RECORDED["tail"]["IMAD"] + 3.0
    assert d["serve"]["estimate"] == pytest.approx(E.RECORDED["serve"] + 1.5)
    assert E.against(res)["t"]["sm_120a"]["tail.total"]["delta"] == 0.0
