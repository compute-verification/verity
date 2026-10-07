"""`verity_flock.rec_residuals`, V*'s algebra statement: the M0 unit's leaf-summed products against `gf2k.residuals`, and one
rep of a real session (rec-algebra's fixture) holding with the whole comb as witness and failing at a changed coin."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from verity_catalog.definitions.boolean import gf2k as G
from verity_flock import rec_algebra as A
from verity_flock import rec_residuals as RR

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "verifier" / "lean" / ".lake" / "build" / "bin" / "flock-verify"
SET = HERE / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"
INNER = json.loads((HERE / "fixtures" / "inner_claims.json").read_text())

#: three residuals over 130 + 3 elements: three products (an element, a transposed block, a sum) and an added term; no product;
#: one product
SMALL = {"nw": 130, "nv": 3, "ops": [[["w", 0]], [["wt", 1, 3], ["v", 0]], [["v", 1]], [["w", 129], ["w", 0]]],
         "res": [{"mul": [[0, 1], [2, 3], [1, 3]], "add": [["v", 2]]}, {"mul": [], "add": [["w", 129]]},
                 {"mul": [[2, 0]], "add": []}]}


@pytest.mark.slow
def test_the_unit_sums_each_residuals_products_leaf_by_leaf():
    """The unit (through its M0 layout) equals the reference on random ports: one Karatsuba (2,187 ANDs) a product, and one
    copy row a leaf for the residual of several products."""
    g = random.Random(7)
    low = RR.lowering(SMALL, "rec-residuals/small")
    ands = sum(1 for k in low.circuit.kind if k == "and")
    assert ands == 4 * RR.LEAVES + RR.LEAVES
    rows = []
    for _ in range(3):
        w = [g.getrandbits(128) for _ in range(SMALL["nw"])]
        v = [g.getrandbits(128) for _ in range(SMALL["nv"])]
        rows.append((RR.words(w, RR.w_words(SMALL)) + RR.words(v, RR.v_words(SMALL)), RR.residual_words(SMALL, w, v)))
    got, _cut, ok = low.evaluate(np.array([r for r, _ in rows], np.uint64))
    assert int(ok) == 0b111
    assert [got[i].astype(np.uint16).tolist() for i in range(3)] == [want for _, want in rows]
    assert any(rows[0][1][:8]) and not any(rows[0][1][24:])


def _inner(tmp: Path) -> Path:
    assert (SET / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures backends/flock/tests/fixtures/hidden-outputs`")
    (tmp / "session").mkdir(parents=True)
    record = json.loads((SET / "records-cpu" / "honest.json").read_text())["record"]
    (tmp / "session" / "session.json").write_text(json.dumps(record))
    for r in range(2):
        shutil.copy(SET / "records-cpu" / f"honest.rep{r}.proof", tmp / "session" / f"circuit.rep{r}.bin")
    return tmp


def test_a_real_rep_holds_with_the_comb_as_witness_and_fails_at_a_changed_coin(tmp_path):
    """Both reps of the fixture's honest session: every residual zero with the whole comb in w and none in v (the stub); the
    verifier's v at a changed coin of rep 0's round 1 (the zerocheck's point; round 0's six coins are pinned out, §9.2) makes
    the zerocheck's residual and Ligerito's final check nonzero."""
    inner = _inner(tmp_path / "honest")
    sh = A.Shape(INNER["m"], INNER["k_log"], 2 + len(INNER["reps"][0]["extras"]))
    S, _ = A.structure(sh)
    for r in range(2):
        w, v = RR.rep_inputs(inner, INNER, r, sh)
        assert len(w) == S["nw"] and len(v) == S["nv"] and v[-64:] == [0] * 64
        assert G.residuals(S, w, v) == [0] * len(S["res"])
    forged = _inner(tmp_path / "forged")
    rec = json.loads((forged / "session" / "session.json").read_text())
    c = rec["streams"][0]["rounds"][1]["coins"]
    c[0] = G.hex128(G.of_hex(c[0]) ^ 1)
    (forged / "session" / "session.json").write_text(json.dumps(rec))
    w, _ = RR.rep_inputs(inner, INNER, 0, sh)
    _, v = RR.rep_inputs(forged, INNER, 0, sh)
    out = G.residuals(S, w, v)
    assert [i for i, x in enumerate(out) if x] == [0, len(out) - 2, len(out) - 1]


def _holding(g: random.Random) -> tuple[list[int], list[int]]:
    """Ports on which SMALL's residuals are zero: the added terms set to the residuals they cancel."""
    w = [g.getrandbits(128) for _ in range(SMALL["nw"])]
    v = [g.getrandbits(128) for _ in range(SMALL["nv"])]
    w[129], v[1] = 0, 0
    v[2] ^= G.residuals(SMALL, w, v)[0]
    assert G.residuals(SMALL, w, v) == [0, 0, 0]
    return w, v


@pytest.mark.slow
def test_the_statement_reads_v_as_public_words_and_registers_its_outputs_as_zero(tmp_path):
    """Two instances, one holding and one whose first residual is not zero (v's added constant changed): the files the
    verifier reads (the circuit's public input port v, its own recomputation of v, the zero output rows' registration) and,
    when Lean is built, its `statement` under them."""
    g = random.Random(11)
    (w0, v0), (w1, v1) = _holding(g), _holding(g)
    v1[2] ^= 1
    s = RR.write_statement(RR.statement(SMALL, "rec-residuals/small", 2), [w0, w1], [v0, v1], tmp_path)
    assert s["holds"] == [True, False] and s["nonzero_residuals"] == [[], [0]] and s["unit_agrees"] == [True, True]
    meta = json.loads((tmp_path / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert [p[0] for p in meta["ports"]] == ["w", "out"]
    [pin] = meta["public_inputs"]
    assert pin["port"] == "v" and pin["words"] == RR.v_words(SMALL) and pin["derivation"] == RR.DERIVATION
    head, body = (tmp_path / "public-inputs.bin").read_bytes().split(b"\n", 1)
    assert json.loads(head)["instances"] == 2 and json.loads(head)["ports"] == [pin]
    assert len(body) == 2 * 2 * RR.v_words(SMALL) and (tmp_path / "pub.bin").read_bytes().endswith(body)
    own = json.loads((tmp_path / "registered.json").read_text())["ports"]
    assert list(own) == ["out"] and own["out"]["reads"] == [0, 1] and own["out"]["root"] == s["registered_root"]
    pub_head = json.loads((tmp_path / "pub.bin").read_bytes().split(b"\n", 1)[0])
    assert pub_head["registered"]["out"]["value"] == "rec-residuals-zero" and pub_head["public_inputs"] == [pin]
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    p = subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(tmp_path / "circuit.txt"),
                        "--public", str(tmp_path / "pub.bin"), "--public-inputs", str(tmp_path / "public-inputs.bin"),
                        "--partition", str(tmp_path / "partition.json"), "--program", str(tmp_path / "program.json"),
                        "--registered", str(tmp_path / "registered.json")], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["registered"] == {"out": {"value": "rec-residuals-zero", "root": s["registered_root"]}}
