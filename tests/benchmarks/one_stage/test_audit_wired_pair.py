"""A wired periods window of two templates through the served audit (`records_audit.py audit`, `a5.audit`, the CPU
`flock-circuit`, the Lean verifier), as `test_audit_two_members` runs its unwired one: the writer ``Gemm_v2{K=16,N=16}``
and the reader ``Gemm_v2{K=16,N=8}``, one unit each, whose ``x`` is the writer's ``o0`` row by one `IF.Wire`. It catches
what no cheaper test does: the served route's records of a wired window read end to end by the audit's own steps."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.commitments import randomness as RND
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.ops import gemm as G
from verity.commitments.periods import interface as IF
from verity.protocols.proofs.spot.verifier.one_stage import docket as R

from test_audit_two_members import RUN, _audit, binaries, fixed  # noqa: F401

WRITER = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
READER = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
WID, RID = descriptor_id(WRITER), descriptor_id(READER)

pytestmark = pytest.mark.slow


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _window(d):
    rng = np.random.default_rng(0)
    x, w, w8 = _bf16(rng.standard_normal((1, 16))), _bf16(rng.standard_normal((1, 16, 16)) / 4), \
        _bf16(rng.standard_normal((1, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(WRITER, x, w, kernel="reference"), dtype=np.uint16).reshape(1, 16)
    y8 = np.asarray(evaluate_batch(READER, y, w8, kernel="reference"), dtype=np.uint16).reshape(1, 8)
    circuit = IF.Circuit("wired-pair", ((WRITER, 1), (READER, 1)),
                         wires=(IF.Wire(RID, "x", WID, "o0", at=0, radix=(1,), base=0, step=(1,)),))
    win = IF.Window(circuit, RUN, RND.parse_design("stratified:1"), 4)
    win.record(WRITER, [x, w], [y])
    win.record(READER, [None, w8], [y8])
    assert win.mismatches() == {}
    win.write(win.close(salts_of=fixed(win)), d, {"label": "wired-pair"}, registrar=R)
    return d


def test_a_wired_pair_reaches_the_member_checks_before_the_draw_and_its_leak_count_counts_the_wired_port(tmp_path, binaries):  # noqa: F811
    rc, row, _out, said = _audit(tmp_path, _window(tmp_path / "window"), binaries, more={"FLOCK_CHALLENGER": "0"},
                                 design="stratified-rate:1/2")
    assert row, f"no summary: the audit stopped before it wrote one\n{said}"
    assert "TypeError" not in said, said
    assert row["leak"].get("wired") == 16 * 16 and row["leak"]["bound"] != "refused", row["leak"]
    a5 = next((json.loads(x[3:]) for x in said.splitlines() if x.startswith("A5 {")), {})
    assert rc != 0 and row["checks_before_draw"] == {"m0": False, "m1": False, "registration": True, "source": True}, a5
