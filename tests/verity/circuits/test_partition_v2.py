"""`verity/partition/v2` (`verity.circuits.partition_v2`): a partition object's `blocks` as advice rows under the advice root
of `verity.commitments.advice`, the statement, the global check and its refusals, a sampled block's rows and its range, the
public form, and the pinned vector `partition_v2_vectors.json`. Its `--write` rewrites it; `REWRITE` names the command.

The program is v1's pinned program (`partition_vectors.json`) on 16-bit values, its six root Calls the units of a
template query."""
from __future__ import annotations

import functools
import hashlib
import json
import sys
from pathlib import Path

import pytest

import test_partition_object as T1

import stand_in  # noqa: F401  (the test primitives' Boolean bodies, also when run as a script)
from verity.circuits import codec, partition_object as PO, partition_v2 as V
from verity.circuits.defs import REGISTRY
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.commitments import advice as A
from verity.commitments.frame_v3.vectors import expand
from verity.circuits.ops.fp32 import Bf16Add
from verity.circuits.ops.prims import const
from verity.circuits.ops.scalar import Bf16GtStrict, SelectBf16
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as door

PATH = Path(__file__).parent / "partition_v2_vectors.json"
REWRITE = ("the vector differs from the tree's; a change meant to move it rewrites it with "
           "`uv run python tools/check/pins.py --write tests/verity/circuits/partition_v2_vectors.json`")
BF16 = Value(16)


def _max(B, a, b):
    return B.call(SelectBf16, B.call(Bf16GtStrict, a, b), a, b)


def _fan(B, S, x):
    a = B.call(Bf16Add, x[0], x[1])
    B.call(Bf16Add, x[0], x[2])                                        # dead: read by nothing, returned by nothing
    return tuple_of(B.call(Bf16Add, a, x[2]), B.call(Bf16Add, a, x[3]))


Max3 = T1._c("HMax3", (("x", Array(3, BF16)),), BF16, lambda B, S, x: _max(B, _max(B, x[0], x[1]), x[2]))
CALLS = (
    Max3,
    T1._c("HPair", (("x", Array(2, BF16)),), Tuple(BF16, BF16),
          lambda B, S, x: (lambda s: tuple_of(s, B.call(Bf16Add, s, x[0])))(B.call(Bf16Add, x[0], x[1]))),
    T1._c("HTwice", (("x", Array(2, BF16)),), BF16,
          lambda B, S, x: B.call(Bf16Add, B.call(Bf16Add, x[0], x[1]), B.call(Bf16Add, x[0], x[1]))),
    T1._c("HRows", (("x", Array(2, Array(3, BF16))),), Array(2, BF16), lambda B, S, x: B.batch(Max3, x, axes=(0,))),
    T1._c("HAddOne", (("x", Array(1, BF16)),), BF16,
          lambda B, S, x: B.call(Bf16Add, x[0], B.call(Bf16Add, B.call(const(16, 0x3F80)), B.call(const(16, 0x4000))))),
    T1._c("HFan", (("x", Array(4, BF16)),), Tuple(BF16, BF16), _fan),
)
Top = T1._c("HTop", tuple((n, fn.signature()[0][0][1]) for n, fn in zip("abcdef", CALLS)), Tuple(*[fn.signature()[1] for fn in CALLS]),
            lambda B, S, *a: tuple_of(*(B.call(fn, v) for fn, v in zip(CALLS, a))))
QUERY = PO.template_instances_query(CALLS)
BLOCKS = [(0, 2), (2, 3), (3, 6)]
PREDS = V.predicates(128, max_gates=64, max_in_bits=256)
Inc = T1._c("HInc", (("x", Array(2, BF16)),), BF16, lambda B, S, x: B.call(Bf16Add, x[0], x[1]))
WIDE = 20


def _program() -> Program:
    return Program(Top)


def _descriptor(P: Program) -> str:
    return codec.canonical_json({k: v for k, v in codec.encode_program(P).items() if k != "annotations"}).decode()


def _salts(label: str, n: int) -> list[bytes]:
    return [expand(f"verity/partition/v2/vectors/{label}/salt/{i}", A.INST.salt_bytes) for i in range(n)]


@functools.cache
def top() -> tuple[Program, bytes]:
    P = _program()
    return P, b"".join(codec.descriptor_chunks(P))


@functools.cache
def wide() -> tuple[Program, bytes, dict]:
    """A batch of `WIDE` units of one template: block 15's boundaries are in two rows."""
    P = Program(T1._c("HWide", (("x", Array(WIDE, Array(2, BF16))),), Array(WIDE, BF16),
                      lambda B, S, x: B.batch(Inc, x, axes=(0,))))
    return P, b"".join(codec.descriptor_chunks(P)), PO.template_instance_query(Inc)


def _registered(P, q, levels, preds=PREDS, label="vtop", starts=(0,)):
    adv = V.advice(P, q, levels, starts)
    rs = V.rows(adv, preds["grid_bits"])
    reg = A.commit(rs, V.ROW_BITS, V.BINDING, _salts(label, len(rs)))
    return adv, rs, reg, V.statement(q, preds, adv, reg.root)


@functools.cache
def registered():
    P, _c = top()
    return _registered(P, QUERY, [BLOCKS])


def test_the_grid():
    assert [V.grid(x) for x in (0, 1, 15, 16, 17, 31, 32, 33, 2049)] == [0, 1, 15, 16, 18, 32, 32, 36, 2304]
    assert V.grid(11, 2) == 12 and V.grid(2304) == 2304
    for x in range(1, 5000):
        y = V.grid(x)
        assert x <= y < x * (1 + 2 ** (1 - V.GRID_BITS)) and V.grid(y) == y
    with pytest.raises(ValueError):
        V.grid(-1)


def test_the_size_classes():
    assert V.SIZE_CLASSES[0] == 16 and V.SIZE_CLASSES[-1] == V.SIZE_CLASS_MAX
    assert [V.size_class(b) for b in (1, 16, 17, 80, V.SIZE_CLASS_MAX, V.SIZE_CLASS_MAX + 1)] == [16, 16, 32, 128, V.SIZE_CLASS_MAX, None]


def test_the_rows_round_trip():
    P, _c = top()
    adv, rs, _reg, st = registered()
    got, sections, design = V.decode(rs)
    assert got == adv and adv.population == 6 and adv.phases == (((0, 2, 3, 6),),)
    assert [(s.phase, s.level, s.first, s.rows, s.blocks, s.bounds) for s in sections] == [(0, 0, 3, 1, 3, (0, 2, 3, 6))]
    assert design == (4, 0)
    assert len(rs) == V.grid(len(rs)) == st["advice"]["rows"] and all(len(r) == V.ROW_BYTES for r in rs)
    assert V.validate(st).ok and st["blocks"] == [[3]] and st["program"] == codec.program_digest(P)


def test_the_global_check_holds():
    _P, c = top()
    _adv, rs, _reg, st = registered()
    v = V.check(st, rs, c)
    assert v.ok, v.codes
    assert v.detail == {"population": 6, "blocks": [[3]], "rows": len(rs)}


def _edit(row: int, at: int, value: int, n: int = 8):
    def f(rs):
        rs = list(rs)
        rs[row] = rs[row][:at] + int(value).to_bytes(n, "little") + rs[row][at + n:]
        return rs
    return f


M = len(V.MAGIC)
#: (label, an edit of the rows, of the statement, of C, the code)
REFUSALS = [
    ("short", lambda rs: rs[:-1], None, None, "rows-count"),
    ("magic", _edit(0, 0, 0, 1), None, None, "rows-malformed"),
    ("padding", _edit(1, 120, 1), None, None, "rows-malformed"),
    ("kind", _edit(2, 0, 2), None, None, "rows-malformed"),
    ("program", _edit(0, M, 0), None, None, "header-program"),
    ("object", _edit(1, 0, 0), None, None, "header-object"),
    ("population", _edit(0, M + 64, 7), None, None, "header-population"),
    ("design-first", _edit(0, M + 80, 5), None, None, "header-layout"),
    ("design-rows", _edit(0, M + 88, 1), None, None, "design-rows"),
    ("start", _edit(3, 0, 1), None, None, "blocks-gap"),
    ("short-cover", _edit(3, 24, 5), None, None, "blocks-gap"),
    ("long-cover", _edit(3, 24, 7), None, None, "blocks-range"),
    ("backwards", _edit(3, 16, 1), None, None, "blocks-form"),
    ("filler", None, None, None, "filler"),
    ("bucket", "bucket", None, None, "blocks-bucket"),
    ("coarser crossed", "refine", None, None, "blocks-refine"),
    ("phase span", "span", None, None, "phase-span"),
    ("other C", None, None, "wide", "program"),
    ("too wide", None, {"predicates": {"X": 64}}, None, "block-too-wide"),
    ("too many gates", None, {"predicates": {"max_gates": 8}}, None, "block-too-large"),
    ("too many bits in", None, {"predicates": {"max_in_bits": 64}}, None, "block-too-large"),
]


def _refused(edit, sedit, cedit):
    P, c = top()
    if edit is None and sedit is None and cedit is None:
        adv, rs, _reg, st = _registered(P, QUERY, [BLOCKS], V.predicates(128, grid_bits=1))
        rs = _edit(3, 32, 5)(rs)
    elif edit == "refine":
        _adv, rs, _reg, st = _registered(P, QUERY, [[(0, 2), (2, 6)], [(0, 3), (3, 6)]], label="vrefine")
        edit = None
    elif edit == "bucket":
        adv = V.advice(P, QUERY, [BLOCKS])
        rs = V.rows(adv, 1)
        st = {**V.statement(QUERY, PREDS, adv, A.commit(rs, V.ROW_BITS, V.BINDING, _salts("vbucket", len(rs))).root),
              "blocks": [[4]]}
        edit = None
    elif edit == "span":
        adv = V.advice(P, QUERY, [BLOCKS, BLOCKS])
        adv = V.Advice(adv.program, adv.object, 6, (((0, 2), (0, 3)), ((2, 3, 6), (3, 6))))
        rs = V.rows(adv)
        st = V.statement(QUERY, PREDS, adv, A.commit(rs, V.ROW_BITS, V.BINDING, _salts("vspan", len(rs))).root)
        edit = None
    else:
        _adv, rs, _reg, st = registered()
    st = json.loads(json.dumps(st))
    for k, v in (sedit or {}).items():
        st[k] = {**st[k], **v} if isinstance(v, dict) else v
    return V.check(st, edit(rs) if edit else rs, wide()[1] if cedit else c)


@pytest.mark.parametrize("label,edit,sedit,cedit,code", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_each_global_refusal(label, edit, sedit, cedit, code):
    v = _refused(edit, sedit, cedit)
    assert not v.ok and code in v.codes, (label, v.codes)


def test_the_count_is_on_the_grid():
    st = json.loads(json.dumps(registered()[3]))
    st["blocks"] = [[17]]
    assert "blocks" in V.validate(st).codes


@pytest.mark.parametrize("edit,code", [
    (lambda st: st.pop("query"), "keys"),
    (lambda st: st.update(query=PO.query()), "query"),
    (lambda st: st["predicates"].update(X=V.SIZE_CLASS_MAX + 1), "predicates"),
    (lambda st: st["predicates"].update(width=16), "predicate-keys"),
    (lambda st: st.update(blocks=[]), "blocks"),
    (lambda st: st.update(blocks=[3]), "blocks"),
    (lambda st: st.update(blocks=[[1], [1, 1]]), "blocks"),
    (lambda st: st["advice"].update(root="00"), "advice"),
])
def test_the_statement_form_is_checked(edit, code):
    st = json.loads(json.dumps(registered()[3]))
    edit(st)
    assert code in V.validate(st).codes


def _opened(rs, reg, st, j, l, i):
    rows = {}
    for r in V.openings(st, j, l, i):
        o = reg.open(r)
        assert A.verify(bytes.fromhex(st["advice"]["root"]), st["advice"]["rows"], V.ROW_BITS, V.BINDING, o)
        rows[r] = o.row
    return rows


def test_a_sampled_block_opens_its_rows_alone():
    P, c, q = wide()
    blocks = [(u, u + 1) for u in range(WIDE)]
    _adv, rs, reg, st = _registered(P, q, [blocks], V.predicates(16), "vwide")
    assert V.check(st, rs, c).ok and st["blocks"] == [[V.grid(WIDE)]] == [[20]]
    first = V.section_row(st["blocks"], 0, 0)
    assert [V.openings(st, 0, 0, i) for i in (0, 14, 15, 16, 19)] == [[first], [first], [first, first + 1], [first + 1], [first + 1]]
    for i in (0, 15, 19):
        b = V.check_block(c, st, 0, 0, i, _opened(rs, reg, st, 0, 0, i))
        assert b.units == ((i, i + 1),)
    with pytest.raises(V.Malformed):
        V.check_block(c, st, 0, 0, 15, {first: rs[first]})


def test_the_opened_form_is_the_partition_objects_blocks():
    P, c = top()
    _adv, rs, reg, st = registered()
    assert V.blocks(rs) == [BLOCKS]
    obj = PO.build(P, QUERY, V.blocks(rs))
    assert PO.verify(obj, P).ok and PO.levels(obj) == [BLOCKS] and PO.blocks(obj) == BLOCKS
    for i in range(len(BLOCKS)):
        mine = V.check_block(c, st, 0, 0, i, _opened(rs, reg, st, 0, 0, i))
        theirs = door.restrict(c, obj, door.block(c, obj, i))
        assert mine.units == (BLOCKS[i],) and mine.descriptor == theirs


def test_a_filler_block_is_empty():
    P, c = top()
    blocks = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    _adv, rs, reg, st = _registered(P, QUERY, [blocks], V.predicates(128, grid_bits=1), "vfill")
    assert st["blocks"] == [[V.grid(6, 1)]] == [[8]] and V.check(st, rs, c).ok
    assert V.decode(rs)[1][0].bounds == (0, 1, 2, 3, 4, 5, 6, 6, 6)
    assert [V.check_block(c, st, 0, 0, i, _opened(rs, reg, st, 0, 0, i)) for i in (6, 7)] == [None, None]
    assert V.blocks(rs) == [blocks]


def test_two_phases_lie_side_by_side():
    P, c = top()
    _adv, rs, reg, st = _registered(P, QUERY, [BLOCKS], label="vphases", starts=(0, 2))
    assert V.check(st, rs, c).ok and st["blocks"] == [[1], [2]] and V.blocks(rs) == [BLOCKS]
    assert V.check_block(c, st, 1, 0, 1, _opened(rs, reg, st, 1, 0, 1)).units == ((3, 6),)
    adv = V.advice(P, QUERY, [BLOCKS], (0, 2))
    adv = V.Advice(adv.program, adv.object, 6, (((0, 2),), ((3, 6),)))
    rs = V.rows(adv)
    st = V.statement(QUERY, PREDS, adv, A.commit(rs, V.ROW_BITS, V.BINDING, _salts("vphases", len(rs))).root)
    assert [0, [2, 3]] in V.check(st, rs, c).detail["blocks-gap"]["gates"]
    with pytest.raises(ValueError):
        V.advice(P, QUERY, [BLOCKS], (0, 1))


def test_levels_refine_coarsest_first():
    P, c = top()
    levels = [[(0, 3), (3, 6)], BLOCKS]
    _adv, rs, reg, st = _registered(P, QUERY, levels, V.predicates(128, max_in_bits=256), "vlevels", starts=(0, 3))
    assert V.check(st, rs, c).ok and st["blocks"] == [[1, 2], [1, 1]] and V.blocks(rs) == levels
    assert [(s.phase, s.level) for s in V.decode(rs)[1]] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert V.check_block(c, st, 0, 0, 0, _opened(rs, reg, st, 0, 0, 0)).units == ((0, 3),)
    mine = V.check_block(c, st, 0, 1, 1, _opened(rs, reg, st, 0, 1, 1))
    obj = PO.build(P, QUERY, V.blocks(rs))
    assert mine.units == ((2, 3),) and PO.levels(obj) == levels and mine.descriptor == door.restrict(c, obj, [(2, 3)])
    with pytest.raises(door.Refused, match="blocks-levels"):
        door.block(c, obj, 1)
    _adv, rs, _reg, st = _registered(P, QUERY, [BLOCKS, levels[0]], label="vlevels")
    assert V.check(st, rs, c).detail["blocks-refine"]["gates"] == [[1, [0, 3]]]


def build() -> dict:
    P, c = top()
    adv, rs, reg, st = registered()
    return {
        "about": "verity/partition/v2 on its descriptor, v1's pinned program (partition_vectors.json) on BF16 values, its six "
                 "root Calls the units of a template query in three blocks: the advice rows, their hm96-sha512/row/v2 "
                 "frame-v3-sha512 registration under partition_v2.BINDING with salt i = "
                 "expand('verity/partition/v2/vectors/vtop/salt/<i>', 72) (frame_v3.vectors.expand), the statement and its "
                 "digest, each block's [index, rows opened, range], and the global refusals: [label, codes].  Written by "
                 "tests/verity/circuits/test_partition_v2.py --write.",
        "descriptor": _descriptor(P),
        "program_digest": codec.program_digest(P),
        "rows": [r.hex() for r in rs],
        "leaves": [x.hex() for x in reg.leaves],
        "root": reg.root.hex(),
        "statement": st,
        "statement_digest": V.digest(st),
        "blocks": [[i, V.openings(st, 0, 0, i), list(V.blocks(rs)[0][i])] for i in range(st["blocks"][0][0])],
        "refusals": [[label, list(_refused(e, s, k).codes)] for label, e, s, k, _code in REFUSALS],
    }


def test_the_pinned_vector():
    want = json.loads(PATH.read_text())
    got = json.loads(json.dumps(build()))
    for k in want:
        assert got[k] == want[k], f"{k}: {REWRITE}"
    assert set(got) == set(want), REWRITE
    rs = [bytes.fromhex(r) for r in want["rows"]]
    P = codec.decode_program(json.loads(want["descriptor"]), REGISTRY)           # the verifier's side: from the bytes alone
    assert codec.program_digest(P) == want["program_digest"] == codec.program_digest(_program())
    assert V.check(want["statement"], rs, b"".join(codec.descriptor_chunks(P))).ok
    assert A.commit(rs, V.ROW_BITS, V.BINDING, _salts("vtop", len(rs))).root.hex() == want["root"] == want["statement"]["advice"]["root"]
    assert hashlib.sha512(V.FORMAT.encode() + b"\0" + codec.canonical_json(want["statement"])).hexdigest() == want["statement_digest"]


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
