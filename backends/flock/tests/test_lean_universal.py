"""The universal unit's Lean interpreter (`verifier/lean/soundness/FlockSoundness/Discharge/PrivateCircuit/Interp.lean`, which
`pad_correct` reads) against `verity.ml.boolean.universal.UniversalUnit_v1` run by `verity.evaluation`, through vectors the
interpreter generates (`fixtures/universal_lean.json`, `lean/universal.sh`). This is a test, not a proof: `interp` is tested
equal to the Definition whose standalone program has the digest `DIGESTS` pins, so a change to `UniversalUnit_v1` fails
here until the vectors and the pin are renewed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from verity import evaluation as E
from verity.evaluation.reference import standalone_call
from verity.ir import codec
from verity.ir.defs import bind
from verity.ml.boolean import universal as U

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "universal_lean.json").read_text())
ARGS = ("op", "src", "out", "x")
#: (G, N_IN, N_OUT) -> the program digest of `UniversalUnit_v1{G, N_IN, N_OUT}`'s standalone call, the Definition checked
DIGESTS = {
    (16, 16, 8): "0038131f58f2fa2d9d0422ffa9a069ee9e9d9569cf32f3bacdcd4122ac1ae9f3",
    (20, 13, 5): "0e5dfd3b15cd3cafc2b5353273d21a1de7c39c68ffa92fc3fe59769c3c140e94",
}


def _bits(s: str) -> list[int]:
    return [int(c) for c in s]


def _agree(vectors: list[dict]) -> None:
    """Every vector's outputs are the Definition's on its arguments, the Definition pinned by digest: the bit-sliced
    evaluator on all of them, the plain-integer reference on the first two of each class."""
    by_cls: dict[tuple, list[dict]] = {}
    for v in vectors:
        by_cls.setdefault(tuple(v["cls"]), []).append(v)
    for (G, N_IN, N_OUT), vs in by_cls.items():
        unit = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
        assert codec.program_digest(codec.encode_program(standalone_call(unit)[0])) == DIGESTS[G, N_IN, N_OUT]
        cols = [np.array([_bits(v[a]) for v in vs], dtype=np.uint8) for a in ARGS]
        got = np.asarray(E.evaluate_batch(unit, *cols)).reshape(len(vs), N_OUT)
        assert got.tolist() == [_bits(v["y"]) for v in vs]
        for v in vs[:2]:
            assert list(E.evaluate(unit, *(_bits(v[a]) for a in ARGS))) == _bits(v["y"])


def test_lean_interp_is_the_definition_on_random_programs():
    """`interp` on random programs at (16, 16, 8) and (20, 13, 5), with every K-bit index (forward reads, and past W at
    (20, 13, 5)), gives `UniversalUnit_v1`'s outputs."""
    assert {tuple(v["cls"]) for v in FIXTURE["random"]} == {(16, 16, 8), (20, 13, 5)} and len(FIXTURE["random"]) == 96
    _agree(FIXTURE["random"])


def test_lean_pad_is_universal_pad():
    """`padUnit`'s program for a random in-range netlist is `universal.pad`'s, bit for bit; the Definition runs it to the
    outputs `interp` gives, which are `interpret`'s on the netlist, in Lean and in Python."""
    assert len(FIXTURE["pad"]) == 64
    for v in FIXTURE["pad"]:
        G, N_IN, _ = v["cls"]
        n = v["netlist"]
        net = U.Netlist(n["n_in"], tuple(map(tuple, n["gates"])), tuple(n["outputs"]))
        x = _bits(v["x"])
        assert [list(a) for a in U.arguments(U.pad(net, G, N_IN), x)] == [_bits(v[a]) for a in ARGS]
        assert U.interpret(net, x[:net.n_in]) == _bits(n["interpret"]) == _bits(v["y"])
    assert len({len(v["netlist"]["gates"]) for v in FIXTURE["pad"]}) > 4
    _agree(FIXTURE["pad"])


def test_lean_dummy_outputs_zeros():
    """`dummyUnit`'s program, run by the Definition on random data, outputs zeros, as `interp` does."""
    assert all(set(v["y"]) == {"0"} for v in FIXTURE["dummy"]) and len({v["x"] for v in FIXTURE["dummy"]}) > 1
    _agree(FIXTURE["dummy"])


def test_the_lean_vectors_were_generated_from_this_tree():
    """The fixture records the sha256 of the generator, the interpreter and the toolchain pins. A change to any of them
    fails here until the vectors are regenerated."""
    stale = [p for p, h in FIXTURE["inputs"].items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not stale, f"changed since the vectors were generated, rerun backends/flock/tests/lean/universal.sh: {stale}"
