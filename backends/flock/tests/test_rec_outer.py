"""`verity_flock.rec_outer`, V*'s outer statements: a level of inner openings staged as one statement of `RecOpen` instances
whose outputs the verifier registers. Staged here at the smallest shape (8 lanes, one sibling: a synthetic tree of four
leaves) with three honest openings and one whose sibling is another node's. Composing the unit's circuit takes about two
minutes, so the module is slow; it catches what `test_rec_open` can't: the statement's ports, program, partition and
registered output rows as the Lean verifier's `statement` reads them (when it is built), and an opening that does not climb
to its cap node found before anything is proved."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_rec_open import _tree
from verity_flock import rec_outer as O
from verity_flock import rec_vstar as V

BIN = Path(__file__).resolve().parents[1] / "verifier" / "lean" / ".lake" / "build" / "bin" / "flock-verify"

pytestmark = pytest.mark.slow


def _queries():
    rows, salts, layers = _tree(2, 8)
    q = lambda rep, j, p, sib: V.Query(rep, 0, j, p, 2, 1, layers[1][p >> 1], rows[p], salts[p], [sib])
    return [q(0, 0, 1, layers[0][0]), q(0, 1, 2, layers[0][3]), q(1, 0, 3, layers[0][2]), q(1, 1, 0, layers[0][3])]


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    (level,) = O.levels(_queries())
    d = tmp_path_factory.mktemp("rec-outer")
    return d, O.stage(level, d)


def test_a_level_is_one_template_over_both_reps(staged):
    d, s = staged
    assert s["template"] == "RecOpen_v1{LANES=8,H=1}" and s["instances"] == 4 and s["compressions"] == 8
    assert s["opened"] == 3 and s["not_opened"] == [(1, 1)]
    own = json.loads((d / "registered.json").read_text())
    assert own["ports"]["out"]["reads"] == [0, 1, 2, 3] and own["ports"]["out"]["root"] == s["registered_root"]
    meta = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert list(meta["program_digests"]) == [s["template"]] and meta["program_sha512"] == s["program_sha512"]
    assert [p[0] for p in meta["ports"]] == ["row", "aux", "out"]
    with pytest.raises(ValueError, match="one template"):
        O.levels([*_queries(), V.Query(0, 0, 2, 0, 3, 1, b"", [0] * 8, b"", [b"", b""])])


@pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")
def test_the_lean_verifier_reads_the_statement_under_the_verifiers_own_files(staged):
    d, _ = staged
    p = subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                        "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
                        "--registered", str(d / "registered.json")], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["partition_check"]["units"] == "derived"
    assert line["registered"] == {"out": {"value": "rec-L0", "root": json.loads((d / "registered.json").read_text())["ports"]["out"]["root"]}}
