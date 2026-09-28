"""Lane PAD-IMPL: the torch-frontend authoring of `Serve@3` (`tests/program/serve3_authored.py`)
against the registered definitions (`verity_vllm/program/registry/serve3.py`) — `check_identity` at export, and
digest equality / graph-and-reference correspondence per program, exactly as `test_b1_authored.py` checks
`Serve@1/@2`.  Needs torch; skips without it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch = pytest.importorskip("torch")

from verity.ml.circuits import Program, bind, program_digest  # noqa: E402
from verity_vllm.program.frontend import export_report  # noqa: E402
from verity_vllm.program.frontend.correspond import correspond  # noqa: E402
from tests.program import serve3_authored as A  # noqa: E402
from verity_vllm.program.registry import b1, serve3  # noqa: E402

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
        "tie_word_embeddings": True, "profile": {}}
C3 = {**TINY, "EOS_ID": 7}
AMPERE = {"GEMM": b1.GEMM_AMPERE, "ATTENTION": b1.ATTENTION_FA2, "NORM": b1.norm_bundle(TINY)}

CASES = [
    ("PadGuard", A.PadGuard, serve3.PadGuard, {"EOS": 7}),
    ("PadDecodeStep{T=6}", A.PadDecodeStep, serve3.PadDecodeStep, {"C": C3, "T": 6, **AMPERE}),
    ("Serve_v3{LP=3,D_MAX=2}", A.ServeV3, serve3.ServeV3, {"C": C3, "LP": 3, "D_MAX": 2, **AMPERE}),
]


@pytest.mark.parametrize("label,front,reg,statics", CASES, ids=[c[0] for c in CASES])
def test_authored_corresponds_to_registered(label, front, reg, statics):
    """The authored program passes `check_identity` against the registered definition (run inside export)
    and corresponds gate-for-gate and reference-for-reference; where the digest differs it is only the
    zero-row broadcast encoding (frontend zero-stride `Strided` vs registry repeated-ref runs)."""
    rep = export_report(front, static=statics, dead_gates="ignore", id_limit=10 ** 6)
    P_reg = Program(bind(reg, **statics))
    r = correspond(rep.program, P_reg)
    assert r.corresponds, (label, [m.to_json() for m in r.mismatches][:3], [u.to_json() for u in r.unverified][:3])
    assert rep.program.gates == P_reg.gates
    if r.digest_equal:
        assert rep.digest == program_digest(P_reg), label
    else:
        assert r.same_references >= 1, label
