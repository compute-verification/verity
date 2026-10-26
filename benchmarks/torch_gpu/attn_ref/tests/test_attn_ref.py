"""attn_ref on the CPU: the SASS gate's verdicts, and fixture.py's words against the Definition's reference evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixture as F  # noqa: E402
import sass_gate  # noqa: E402

PTXAS = """ptxas info    : Compiling entry function 'attn_ref_qk' for 'sm_90a'
ptxas info    : Function properties for attn_ref_qk
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers
ptxas info    : Compiling entry function 'attn_ref_pv' for 'sm_90a'
ptxas info    : Function properties for attn_ref_pv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers
"""
SASS = """
        Function : attn_ref_qk
        /*0090*/                   HFMA2.MMA R4, -RZ, RZ, 0, 0 ;
        /*0100*/                   HMMA.16816.F32.BF16 R4, R8, R12, R4 ;
        /*0200*/                   F2FP.BF16.F32.PACK_AB R3, RZ, R4 ;
        Function : attn_ref_pv
        /*0100*/                   HMMA.16816.F32.BF16 R4, R8, R12, R4 ;
        /*0200*/                   F2FP.BF16.F32.PACK_AB R3, RZ, R4 ;
"""


def test_gate_passes_a_clean_listing():
    funcs, problems = sass_gate.gate(PTXAS, SASS)
    assert problems == []
    assert funcs["attn_ref_qk"]["HMMA"] == 1 and funcs["attn_ref_pv"]["F2FP"] == 1


@pytest.mark.parametrize("ptxas, sass, why", [
    (PTXAS, SASS.replace("F2FP.BF16.F32.PACK_AB R3, RZ, R4 ;\n        Function : attn_ref_pv",
                         "FFMA R3, R4, R5, R6 ;\n        Function : attn_ref_pv"), "floating-point instructions other than"),
    (PTXAS, SASS + "        /*0300*/  FADD R1, R2, R3 ;\n", "floating-point instructions other than"),
    (PTXAS.replace("0 bytes spill stores", "8 bytes spill stores", 1), SASS, "local memory"),
    (PTXAS, SASS.replace("HMMA.16816.F32.BF16", "HMMA.1688.F32.TF32", 1), "tensor-core forms other than"),
    (PTXAS.replace("'sm_90a'", "'sm_120a'", 1), SASS, "compiled for sm_120a"),
    (PTXAS + "ptxas warning : Registers are spilled\n", SASS, "Registers are spilled"),
    (PTXAS, SASS.split("        Function : attn_ref_pv")[0], "attn_ref_pv: missing"),
])
def test_gate_refuses(ptxas, sass, why):
    _, problems = sass_gate.gate(ptxas, sass)
    assert any(why in p for p in problems), problems


@pytest.mark.parametrize("kind", ["specials", "tiny"])
def test_pool_words_are_the_definitions(kind):
    """A sequence's S and O at every length agree with the reference evaluator on the sample fixture.py holds it to, and O at
    a length off the 16-grid is the coordinate over both operands zero-padded."""
    i = F.KINDS.index(kind)
    seq = F.expected(i)
    assert seq["ref"]["qk"][1] == 0 and seq["ref"]["pv"][1] == 0 and seq["ref"]["qk"][0] == F.REF_PER_SEQ
    rng = np.random.default_rng(0)
    for n in (1, 15, 17, 31):
        h, d = rng.integers(0, F.H, 8), rng.integers(0, F.D, 8)
        x = np.zeros((8, F.pad16(n)), np.uint16)
        w = np.zeros((8, F.pad16(n)), np.uint16)
        x[:, :n], w[:, :n] = seq["P"][h, :n], seq["V"][:n, d].T
        assert (F.coordinate(x, w, kernel="reference") == seq["O"][n - 1, h, d]).all()


@pytest.mark.slow
def test_every_kind_agrees_with_the_reference():
    """Every kind's reference sample (about 1 s a sequence): what the two quick kinds above don't reach (cancellations,
    overflow, uniform words)."""
    for i in range(len(F.KINDS)):
        ref = F.expected(i)["ref"]
        assert ref["qk"][1] == 0 and ref["pv"][1] == 0, (F.KINDS[i], ref)


def test_odd_case():
    doc, ref = F.odd()
    assert ref["qk"][1] == 0 and ref["pv"][1] == 0
    lens = doc["lens"]
    assert (doc["O"][lens == 0] == 0).all(), "an empty dot is +0, cast to 0x0000"
    for b, n in enumerate(lens):
        assert (doc["S"][b, :, n:] == F.FILL).all()
