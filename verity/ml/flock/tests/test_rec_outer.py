"""`verity_flock.rec_outer`, V*'s outer statements: a level of inner openings staged as one statement of `RecOpen` instances
whose nodes the verifier registers and whose running sums the prover registers as one chain. Staged here at the smallest
shape (8 lanes, one sibling: a synthetic tree of four leaves) with three honest openings and one whose sibling is another
node's. Composing the unit's circuit takes about two minutes, so that test is slow; it catches what `test_rec_open` can't:
the statement's ports, program, partition, public input and registered rows (the outputs, the running sums and, with
`link_rows`, the opened rows) as the Lean verifier's `statement` reads them (when it is built), and an opening that does not
climb to its cap node found before anything is proved."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from test_rec_open import _tree
from verity_flock import rec_open as R
from verity_flock import rec_outer as O
from verity_flock import rec_vstar as V

BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"
PROGRAM = O.session_program(b"a test session")


def _queries():
    rows, salts, layers = _tree(2, 8)
    q = lambda rep, j, p, sib: V.Query(rep, 0, j, p, 2, 1, layers[1][p >> 1], rows[p], salts[p], [sib])
    return [q(0, 0, 1, layers[0][0]), q(0, 1, 2, layers[0][3]), q(1, 0, 3, layers[0][2]), q(1, 1, 0, layers[0][3])]


def _coefs(qs) -> dict:
    g = random.Random(3)
    kappa = [[[g.getrandbits(128) for _ in range(8)] for _ in range(2)]]
    ea = [g.getrandbits(128) for _ in range(2)]
    return {(q.rep, q.level, q.j): O.coefficients(kappa, ea, q.level, q.j) for q in qs}


def test_the_chain_runs_rep_major_and_each_row_adds_its_openings_terms():
    qs = _queries()
    coefs = _coefs(qs)
    ch = O.Chain.build(PROGRAM, qs, coefs)
    assert O.chain_order(qs) == [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)] and ch.reg.leaves == 5
    assert ch.rep_ends(0) == (0, 2) and ch.rep_ends(1) == (2, 4) and ch.reg.rows[0] == (0,) * R.ACC_WORDS
    by = {(q.rep, q.level, q.j): q for q in qs}
    for k, p in ch.at.items():
        assert list(ch.reg.rows[p + 1]) == R.acc_ref(ch.reg.rows[p], coefs[k], R.words(by[k].row))
    assert ch.reg.entry()["value"] == O.ACC_VALUE and ch.reg.program == PROGRAM


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    qs = _queries()
    (level,) = O.levels(qs)
    coefs = _coefs(qs)
    d = tmp_path_factory.mktemp("rec-outer")
    return d, O.stage(level, d, O.Chain.build(PROGRAM, qs, coefs), coefs, link_rows=True)


@pytest.mark.slow
def test_a_level_is_one_template_over_both_reps(staged):
    d, s = staged
    assert s["template"] == "RecOpen_v2{LANES=8,H=1}" and s["instances"] == 4 and s["compressions"] == 8
    assert s["opened"] == 3 and s["not_opened"] == [(1, 1)] and s["summed"] == 4
    own = json.loads((d / "registered.json").read_text())["ports"]
    assert own["out"]["reads"] == [0, 1, 2, 3] and own["out"]["root"] == s["registered_root"]
    assert own["row"]["reads"] == [0, 1, 2, 3] and own["row"]["root"] == s["rows_root"] != s["registered_root"]
    assert own["acc"]["reads"] == [0, 1, 2, 3] and own["acc_out"]["reads"] == [1, 2, 3, 4]
    assert own["acc"]["root"] == own["acc_out"]["root"] == s["acc_root"] and own["acc"]["leaves"] == 5
    head = json.loads((d / "pub.bin").read_bytes().split(b"\n", 1)[0])
    assert {k: v["value"] for k, v in head["registered"].items()} == {"row": "rec-rows-L0", "out": "rec-L0", "acc": O.ACC_VALUE,
                                                                     "acc_out": O.ACC_VALUE}
    meta = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert list(meta["program_digests"]) == [s["template"]] and meta["program_sha512"] == s["program_sha512"]
    assert [p[0] for p in meta["ports"]] == ["row", "aux", "acc", "out", "acc_out"]
    assert [(p["port"], p["words"]) for p in meta["public_inputs"]] == [("coef", R.coef_words(8))]
    with pytest.raises(ValueError, match="one template"):
        O.levels([*_queries(), V.Query(0, 0, 2, 0, 3, 1, b"", [0] * 8, b"", [b"", b""])])


@pytest.mark.slow
@pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")
def test_the_lean_verifier_reads_the_statement_under_the_verifiers_own_files(staged):
    d, _ = staged
    p = subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                        "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
                        "--registered", str(d / "registered.json"), "--public-inputs", str(d / "public-inputs.bin")],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["partition_check"]["units"] == "derived"
    own = json.loads((d / "registered.json").read_text())["ports"]
    assert line["registered"] == {k: {"value": v["value"], "root": v["root"]} for k, v in own.items()}
