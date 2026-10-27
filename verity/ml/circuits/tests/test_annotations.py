"""`verity/annotations/v0` (`verity.ml.circuits.annotations`): keys resolve against a descriptor's structure, stem keys in every
specialization, guards report staleness, groups close and their members resolve, and attaching annotations moves no program
digest."""
from __future__ import annotations

import pytest

from verity.ml.circuits import annotations as A
from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, I32Add

F32 = I32 = Value(32)


def _c(name, params, ret, body, statics=(), sig=None):
    return CompositeDefinition(name, 1, statics, sig or (lambda S: (params, ret)), body, register=False)


#: fmaxf over N words then |.|, so specializations differ in node count
MaxAbs = _c("AMaxAbs", None, None,
            lambda B, S, x: B.call(F32Fabs, _fold(B, [x[i] for i in range(S.N)])),
            statics=("N",), sig=lambda S: ((("x", Array(S.N, F32)),), F32))


def _fold(B, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = B.call(F32Fmaxf, acc, x)
    return acc


Pair = bind(_c("APair", (("x", Array(2, I32)),), Tuple(I32, I32),
               lambda B, S, x: (lambda s: tuple_of(s, B.call(I32Add, s, x[0])))(B.call(I32Add, x[0], x[1]))))
M2, M3 = bind(MaxAbs, N=2), bind(MaxAbs, N=3)
Top = bind(_c("ATop", (("a", Array(2, F32)), ("b", Array(3, F32)), ("c", Array(2, I32))), Tuple(F32, F32, Tuple(I32, I32)),
              lambda B, S, a, b, c: tuple_of(B.call(M2, a), B.call(M3, b), B.call(Pair, c))))


@pytest.fixture(scope="module")
def desc():
    return codec.encode_program(Program(Top))


def ann(*entries):
    return {"format": A.FORMAT, "entries": list(entries)}


def test_keys_resolve_against_the_descriptor(desc):
    v = A.validate(ann(
        {"key": "def:AMaxAbs_v1", "name": "max of magnitudes"},
        {"key": "port:AMaxAbs_v1/in/x", "name": "words", "format": "FP32"},
        {"key": "port:AMaxAbs_v1/out/0:1", "name": "result"},
        {"key": "node:AMaxAbs_v1/0", "callee": "F32Fmaxf_v1", "name": "first max"},
        {"key": "wire:APair_v1/1/0", "name": "the sum, read again"},
        {"key": "wire:APair_v1/ret"},
        {"key": "port:F32Fmaxf_v1/in/0", "name": "a"},
        {"key": "group:pair-sums", "members": ["node:APair_v1/0", "node:APair_v1/1"], "name": "sums"},
        {"key": "def:Absent_v1", "name": "a Definition this program lacks"},
        {"key": "call:3", "program": codec.program_digest(desc), "name": "the max of two"},
    ), desc, codec.program_digest(desc))
    assert v.ok, v.errors
    assert v.absent == ["def:Absent_v1"] and not v.stale


@pytest.mark.parametrize("entry,code", [
    ({"key": "node:AMaxAbs_v1/2"}, "unresolved"),                  # N=2 has nodes 0..1: a stem key must hold in every specialization
    ({"key": "node:F32Fmaxf_v1/0"}, "unresolved"),                  # a primitive has no body
    ({"key": "wire:APair_v1/1/2"}, "unresolved"),
    ({"key": "port:APair_v1/in/y"}, "unresolved"),
    ({"key": "port:APair_v1/out/0:3"}, "unresolved"),
    ({"key": "call:0", "program": "0" * 128}, None),                # another program's Call: not checked here
    ({"key": "call:x"}, "program"),
    ({"key": "gate:12"}, "key"),
    ({"key": "def:APair_v1", "colour": "red"}, "entry"),
    ({"key": "group:g", "members": []}, "members"),
    ({"key": "group:g", "members": ["group:missing"]}, "members"),
    ({"key": "group:g", "members": ["node:APair_v1/9"]}, "members"),      # a member must resolve like any key
    ({"key": "def:APair_v1", "aliases": "pair"}, "text"),
    (["not", "an", "entry"], "entry"),
])
def test_refusals(desc, entry, code):
    v = A.validate(ann(entry), desc, codec.program_digest(desc))
    assert [e["code"] for e in v.errors][:1] == ([code] if code else [])


def test_a_field_is_set_once(desc):
    v = A.validate(ann({"key": "def:APair_v1", "name": "a"}, {"key": "def:APair_v1", "name": "b"}), desc)
    assert [e["code"] for e in v.errors] == ["duplicate"]


def test_guards_report_staleness(desc):
    fid = codec.descriptor_id(M3)
    good = A.definition_digest(desc, fid)
    v = A.validate(ann({"key": f"def:{fid}", "digest": good}, {"key": "node:AMaxAbs_v1/1", "callee": "F32Fmaxf_v1"}), desc)
    assert v.ok and v.stale == [{"key": "node:AMaxAbs_v1/1", "definition": codec.descriptor_id(M2), "callee": "F32Fabs_v1",
                                 "written_for": "F32Fmaxf_v1"}]
    v = A.validate(ann({"key": f"def:{fid}", "digest": "0" * 128}), desc)
    assert v.ok and v.stale[0]["digest"] == good


def test_the_definition_digest_does_not_depend_on_the_program(desc):
    alone = codec.encode_program(Program(bind(_c("AOnly", (("b", Array(3, F32)),), F32, lambda B, S, b: B.call(M3, b)))))
    fid = codec.descriptor_id(M3)
    assert A.definition_digest(alone, fid) == A.definition_digest(desc, fid)


def test_attaching_moves_no_digest(desc):
    a = ann({"key": "def:APair_v1", "name": "pair"}, {"key": "port:APair_v1/in/x", "name": "operands"})
    with_a = A.attach(desc, a)
    assert with_a["annotations"][A.FORMAT] == a["entries"]
    assert codec.program_digest(with_a) == codec.program_digest(desc)
    assert codec.program_digest(with_a) == codec.program_digest(desc)


def test_merge_layers_later_fields_win():
    m = A.merge(ann({"key": "def:APair_v1", "name": "pair", "description": "d"}), ann({"key": "def:APair_v1", "name": "sum pair"}))
    assert m["entries"] == [{"key": "def:APair_v1", "name": "sum pair", "description": "d"}]
