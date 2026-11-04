"""v2's wired reads on the wired stand-in's case (`a5_wired`): ``sampled-proofs-one-stage-zk verdicts --check`` on the
case a5 made of the stand-in's two members (a writer ``Bf16AddRow_v1{N=64}`` whose ``o0`` is its reader
``ArgmaxChunkBest_v1{C=64}``'s ``x``) and on `a5_wired.cases`' changes of it, read from the fetched fixtures
``a5-wired-v2`` (the case, its blobs named relative to it) and ``a5-wired-circuits`` (the members' circuits, each
xz-compressed as ``sha512/<hex>.xz`` and held to its name once decompressed, `circuits`).
As made, every session is accepted naming the wiring; with ``wires`` null or absent, every session is accepted naming
none; with the wire moved or the reader's rows altered (the commitment's ``<reader>/x`` root theirs), the reader is
refused by its wired reads; with the commitment's ``<writer>/o0`` root another, the writer is refused by the statement's
root check (`Registered.statementHeld`) and the reader by its wired reads.

The fixtures are one run of the stand-in (``STANDIN=wired`` in ``a5_tp.sh``): its case, with each member's public file
and the one session of its drawn unit (``stratified:1``), and both members' circuits. One writer and one reader member
are the least a wired read needs. A member's circuit is mostly its rows' hashing, whatever its Definition, so even the
stand-in's small Definitions make circuits of tens of MB; compressed, they are a fixture of a few.

The wired reads' derivation has its difftest (`test_wires_difftest`); these alone run them on proved sessions, the bytes a
full one-stage ZK audit made, where a reader row's opening against its writer's committed root decides."""

from __future__ import annotations

import hashlib
import json
import lzma
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


@pytest.fixture(scope="module")
def circuits(tmp_path_factory) -> Path:
    """The case's circuits (its blobs ``…/sha512/<hex>.xz``), decompressed once into ``sha512/<hex>``, each SHA-512 its name."""
    d = tmp_path_factory.mktemp("circuits") / "sha512"
    d.mkdir()
    for b in json.loads((CASE / "case.json").read_text())["blobs"]:
        if b.endswith(".xz"):
            out = d / Path(b).name.removesuffix(".xz")
            out.write_bytes(lzma.decompress((CASE / b).read_bytes()))
            assert hashlib.sha512(out.read_bytes()).hexdigest() == out.name, f"{b}: not the circuit its name pins"
    return d


def kept(circuits: Path) -> dict:
    c = json.loads((CASE / "case.json").read_text())
    return {**c, "blobs": [str(circuits / Path(b).name.removesuffix(".xz") if b.endswith(".xz") else (CASE / b).resolve())
                           for b in c["blobs"]]}


def null(case: dict) -> dict:
    pr = case["params"]["proofs"]
    return {**case, "params": {**case["params"], "proofs": {**pr, "wires": None}}}


@pytest.mark.slow
@pytest.mark.skipif(not LAKE, reason="no Lean toolchain")
@pytest.mark.parametrize("name", ["wired", "unwired", "null", "moved", "altered", "writer"])
def test_v2_holds_each_session_to_its_wired_reads(tmp_path, v2_exe, circuits, name):
    """Each session's verdict is as `a5_wired.cases` says for ``name``, in the statements' order (`a5_wired.held`):
    accepted naming the case's wiring, or refused for the reason it names ("does not open" for the wired reads)."""
    case = kept(circuits)
    mine = IF.members(AW.circuit(), AW.UNITS)
    reader = next(m for m in mine if m.template == AW.RID)
    writer = next(m.key for m in mine if m.template == AW.WID)
    members = [st["member"] for st in AW.evidence_of(case)["proofs"]["statements"] for _ in st["sessions"]]
    assert sorted(members) == sorted(m.key for m in mine) and AW.wiring_digest(case) is not None
    cases = AW.cases(case, writer, reader, tmp_path / "altered")
    c, want = (null(case), cases["unwired"][1]) if name == "null" else cases[name]
    r = AW.verdicts(str(v2_exe), c, tmp_path / name)
    assert AW.held(r, members, want, AW.wiring_digest(c)), r
