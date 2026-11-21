"""Lean's lift to bits and lowering (`flock-lower-check`, `Flock.Lift.closure` and `Flock.Lower.lower`) beside the Python twins (`codec.lift_shape`,
`codec.odd_fields`) on Definitions whose parameters are collections of mixed widths: an array of a tuple of an 8-bit and a 16-bit
value, a record of 1, 8 and 32 bits. A Definition that reaches a word primitive is lifted and lowered by both; one that reads
only a bit and leaves the collection unread is left unlifted by both and refused by Lean's lowering by name (`Lower.WIDE`)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import stand_in  # noqa: F401  (the test primitives' Boolean bodies)
from verity.circuits import codec
from verity.circuits.boolean import NOT
from verity.circuits.codec import descriptor_id, lift_shape, odd_fields
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, RecordType, Tuple, Value
from verity.protocols.proof.spot.verifier.one_stage import proofs as PF

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

V1, V8, V16, V32 = Value(1), Value(8), Value(16), Value(32)
PAIRS = Array(2, Tuple(V8, V16))
RECORD = RecordType(("p", V1), ("q", V8), ("w", V32))


def _word(name: str, w: int) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, (("a", Value(w)),), Value(w), lambda a: a, register=False)


OP8, OP32 = _word("MixOp8", 8), _word("MixOp32", 32)


def _template(name: str, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


#: name: (the template, whether it reaches a word primitive)
TEMPLATES = {
    "pairs": _template("MixPairs", (("a", PAIRS),), V8, lambda B, S, a: B.call(OP8, a[0][0])),
    "record": _template("MixRecord", (("r", RECORD),), Tuple(V8, V32), lambda B, S, r: tuple_of(B.call(OP8, r.q), B.call(OP32, r.w))),
    "both": _template("MixBoth", (("a", PAIRS), ("r", RECORD)), Tuple(V8, V32),
                      lambda B, S, a, r: tuple_of(B.call(OP8, a[1][0]), B.call(OP32, r.w))),
    "unread": _template("MixUnread", (("a", PAIRS), ("b", V1)), V1, lambda B, S, a, b: B.call(NOT, b)),
}
LIFTED = ("pairs", "record", "both")


def _description(template) -> dict:
    params, ret = template.signature()
    root = CompositeDefinition(f"MixRoot{template.name}", 1, (), lambda S: (tuple(params), ret), lambda B, S, *xs: B.call(template, *xs),
                               register=False)
    return json.loads(b"".join(codec.descriptor_chunks(Program(root))))


def _lean(template, tmp_path: Path) -> subprocess.CompletedProcess:
    params, ret = template.signature()
    ins = [sum(t.leaf_widths()) for _, t in params]
    outs = [sum((f[1] if isinstance(f, tuple) else f).leaf_widths()) for f in (getattr(ret, "fields", None) or (ret,))]
    (tmp_path / "c.json").write_text(json.dumps(_description(template)))
    (tmp_path / "in.txt").write_text("0\n" * sum(ins))
    return subprocess.run([str(PF.lower_executable()), str(tmp_path / "c.json"), descriptor_id(template), ",".join(map(str, ins)),
                           ",".join(map(str, outs)), str(tmp_path / "in.txt"), str(tmp_path / "o.tmpl"), str(tmp_path / "o.txt")],
                          capture_output=True, text=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", LIFTED)
def test_a_mixed_width_definition_that_reaches_a_word_primitive_is_lifted_and_lowered_by_both(name, tmp_path):
    t = TEMPLATES[name]
    assert lift_shape(_description(t), descriptor_id(t)) == ([descriptor_id(t)], [])
    p = _lean(t, tmp_path)
    assert p.returncode == 0 and "LOWERED" in p.stdout and "REFUSED" not in p.stdout, p.stdout + p.stderr


@pytest.mark.slow
def test_a_mixed_width_definition_reaching_no_word_primitive_is_left_unlifted_by_both_and_refused_by_leans_lowering(tmp_path):
    t = TEMPLATES["unread"]
    key = descriptor_id(t)
    assert lift_shape(_description(t), key) == ([], [key])
    p = _lean(t, tmp_path)
    assert p.returncode == 1 and f"REFUSED {key}: {key}: a parameter or return leaf wider than one bit, not lifted to bits" in p.stdout


def test_the_fields_the_door_cannot_take_whole_are_the_records_41_bits_and_the_8_bit_returns():
    odd = {name: odd_fields(_description(t), descriptor_id(t)) for name, t in TEMPLATES.items()}
    assert odd == {"pairs": [("ret", "0", 8, False)],
                   "record": [("param", "r", 41, False), ("ret", "0", 8, True)],
                   "both": [("param", "r", 41, False), ("ret", "0", 8, True)],
                   "unread": [("param", "b", 1, False), ("ret", "0", 1, False)]}
