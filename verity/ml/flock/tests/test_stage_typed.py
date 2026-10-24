"""Staging a typed statement (`typed_statement.stage`): the instance files name the typed statement and pin its digest and
class, and their rows (the output rows too), salts and commit strings are the flat statement's byte for byte, since they
depend on the rows and salts, never on the circuit's layout, and the public file holds nothing but roots, hiding leaves and
structure. The prover's copy of the rows sits beside the statement."""
from __future__ import annotations

import json

import pytest

from verity_flock import circuit as FC
from verity_flock import typed_statement as TS

pytest.importorskip("verity_numerical")
pytest.importorskip("verity.ml.definitions.registry.prims")
from verity_numerical.bench import templates as TM  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from test_circuit import assert_public_only  # noqa: E402

KEY = bytes(range(32))


def _staged(sid_or_sub, tmp_path, n=4):
    sub = TM.parse_subcircuit_id(sid_or_sub) if isinstance(sid_or_sub, str) else sid_or_sub
    s = InputSet.open(generate(sub, n, 20260926, tmp_path / "set")["path"])
    low = FC.lowering_for_set(s)
    flat = FC.stage(low, s, 0, n, tmp_path / "flat", salt_key=KEY)
    typed = TS.stage(low, s, 0, n, tmp_path / "typed", salt_key=KEY)
    return low, flat, typed


def _split(path):
    data = path.read_bytes()
    head, _, body = data.partition(b"\n")
    return json.loads(head), body


@pytest.mark.parametrize("sid,rows", [("rope-head/d64/neox-bf16", {"unit.txt"}), ("gemm-coordinate/k64/sm80-mma-bf16", None)])
@pytest.mark.slow
def test_typed_staging(sid, rows, tmp_path):
    low, flat, typed = _staged(sid, tmp_path)
    text = (typed.parent / "circuit.txt").read_text()
    h0, b0 = _split(flat)
    h1, b1 = _split(typed)
    assert h1["statement"] == TS.STATEMENT and h1["circuit_sha512"] == FC.digest(text)
    assert h1["units"]["classes"] == [FC.class_digest(text)] * h1["instances"]
    assert b1 == b0, "the rows (the output rows too), salts and commit strings are the flat statement's"
    for k in ("set", "subcircuit", "range", "instances", "frame_v3", "rows"):
        assert h1[k] == h0[k]
    staged = {p.name for p in (typed.parent / "rows").iterdir()}
    if rows is not None:
        assert staged == rows
    else:
        assert {"root.txt", "delta.txt"} <= staged and len(staged) >= 3
        assert (typed.parent / "circuit-forged.txt").exists()
    pub = typed.parent / f"pub-{h1['instances']}.bin"
    hp, bp = _split(pub)
    assert hp["rows"] is False and hp["statement"] == TS.STATEMENT and b1.endswith(bp)
    assert_public_only(text, typed.read_bytes(), pub.read_bytes())
