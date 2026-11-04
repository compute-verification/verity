"""v2's wired reads on the wired stand-in's case (`a5_wired`): ``sampled-proofs-one-stage-zk verdicts --check`` on the
case a5 made of the stand-in's two members (a writer Gemm whose ``o0`` is its reader Gemm's ``x``) and on
`a5_wired.cases`' changes of it, read from the fetched fixtures ``a5-wired-v2`` (the case, its blobs named relative to it)
and ``a5-wired-circuits`` (the members' circuits, by SHA-512). As made, every session is accepted naming the wiring;
with ``wires`` null or absent, every session is accepted naming none; with the wire moved or the reader's rows altered
(the commitment's ``<reader>/x`` root theirs), the reader is refused by its wired reads; with the commitment's
``<writer>/o0`` root another, the writer is refused by the statement's root check and the reader by its wired reads.

The wired reads' derivation has its difftest (`test_wires_difftest`); these alone run them on proved sessions, the bytes a
full one-stage ZK audit made, where a reader row's opening against its writer's committed root decides."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.ml.windows import interface as IF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_wired as AW  # noqa: E402

CASE = HERE / "fixtures" / "a5-wired-v2"
LAKE = shutil.which("lake") is not None


@pytest.fixture(scope="module")
def v2_exe() -> Path:
    return PF.v2_executable()


def kept() -> dict:
    c = json.loads((CASE / "case.json").read_text())
    return {**c, "blobs": [str((CASE / b).resolve()) for b in c["blobs"]]}


def null(case: dict) -> dict:
    pr = case["params"]["proofs"]
    return {**case, "params": {**case["params"], "proofs": {**pr, "wires": None}}}


@pytest.mark.slow
@pytest.mark.skipif(not LAKE, reason="no Lean toolchain")
@pytest.mark.parametrize("name", ["wired", "unwired", "null", "moved", "altered", "writer"])
def test_v2_holds_each_session_to_its_wired_reads(tmp_path, v2_exe, name):
    """Each session's verdict is as `a5_wired.cases` says for ``name``, in the statements' order (`a5_wired.held`):
    accepted naming the case's wiring, or refused for the reason it names ("does not open" for the wired reads)."""
    case = kept()
    mine = IF.members(AW.circuit(), AW.UNITS)
    reader = next(m for m in mine if m.template == AW.RID)
    writer = next(m.key for m in mine if m.template == AW.WID)
    members = [st["member"] for st in AW.evidence_of(case)["proofs"]["statements"] for _ in st["sessions"]]
    assert sorted(members) == sorted(m.key for m in mine) and AW.wiring_digest(case) is not None
    cases = AW.cases(case, writer, reader, tmp_path / "altered")
    c, want = (null(case), cases["unwired"][1]) if name == "null" else cases[name]
    r = AW.verdicts(str(v2_exe), c, tmp_path / name)
    assert AW.held(r, members, want, AW.wiring_digest(c)), r
