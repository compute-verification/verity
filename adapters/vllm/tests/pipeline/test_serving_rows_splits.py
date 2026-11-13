"""A sparse-MLA split row cut into its heads' per-split key-block chains (`pipeline.serving_rows`' ``chain`` layout, `chain_rows`,
the cut `sparse_mla_split.split_chain`): a ``SparseMlaSplit_v1`` row in NS > 1 splits holds, per head, each split's
``SparseMlaHeadBlock_v1`` chain (its first visited block FIRST at its key count, every later one the full block), each split's
end in column blocks (``SparseMlaSplitEndCols_v1``, 64 columns each at DV = 512) and a ``SparseMlaCombine_v1`` of each column
block over the splits (at decode's T = 272 in five splits, per head, four 64-key first blocks, one 16-key one, forty column ends
and eight combines); at NS = 1, prefill's, it is a ``SparseMla_v1`` row's chain.  A window of them at a small static (T = 40 keys
in blocks of 16, 16 and 8, in two splits, two heads, DV = 32: one column block):

* reads each block's operands off the row and its state off the twin, and each column end's partial columns and lse off the
  twin's split end, whose combine is the row's served words, so each unit's Definition on its rows, parsed as Flock's stager
  parses them, is its committed output;
* wires every later block's state, every column end's state and every combine slot to the output row of the unit that wrote
  it, commits it under that row's salt, so the verifier's check on the public bodies holds, and refuses a row that is not its
  writer's, a partial in another combine slot, a row whose served words are not its splits' combine, and a partition that
  leaves out a writer's template.

A row's key blocks are served as scores and values units instead where the partition names those templates (``UNITS``, the cut
`sparse_mla_split.split_chain` at ``cols``): per key block a ``SparseMlaHeadScoresCols_v1``, a ``SparseMlaHeadValues_v1`` per
column block and a ``SparseMlaHeadKeysWire_v1`` per column block of its key rows, which every head's scores and values units
read, each split's end on each column block (``SparseMlaSplitEnd_v1``) and the combines of those.  The same checks hold of that
window at DV = 64 and DR = 32 in column blocks of 32 (``HALF``): what each unit reads and returns, the wires, the Boolean
bodies, and the commitment.  Every scores and values unit reads its keys from one committed row a column block, its writer's,
so a unit whose key rows are not its writer's is refused: when the window is committed, at registration where they are a
tree of its own, and when the unit is sampled where its reference names another row of its writer's table.

`test_a_window_of_one_row_commits_and_its_wires_hold` takes minutes, all of it the window's population program lowering the
five templates: it is the one test of the committed window and its public bodies, which the others reach only through the rows.
`benchmarks/one_stage/a5_sparse_mla.py` audits the same row's window (`window`, `Served`) over ``BITS``, each key block its
Boolean body (``SparseMlaHeadBlock_v2``): the same units, rows and wires, staged flat, the form `verify --zk` takes; and with
``--units`` ``HALF``'s over ``UNITS_BITS``.
"""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.commitments.windows import interface as IF
from verity.commitments.windows import rows as CS
from verity.commitments.windows import wiring as WI
from verity_vllm.pipeline import chain_rows as CH
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline.serving_templates import values_widths
from verity.definitions import boolean as BB
from verity.definitions.numerics import sparse_mla_split_twins as TW
from verity.definitions.registry import boolean_sparse_mla as BM
from verity.definitions.registry import boolean_sparse_mla_split as BX
from verity.definitions.registry import hopper
from verity.definitions.registry import sparse_mla as M
from verity.definitions.registry import sparse_mla_split as W
from verity_vllm.query.manifest import format as MF
from verity_vllm.query.program_view import spec_statics

from verity.protocols.spot.verifier.one_stage import draw as D, registration as R
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.types import Tuple, field_types
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
BOOL = {"DOT": BX.SPARSE_MLA_GLM53["DOT"], "INV": BX.SPARSE_MLA_GLM53["INV"]}
#: DV = 32 is the least at which a combine's partial row (4 DV bytes) takes a compression before its last block, which Flock's
#: compose needs some row of a circuit to have (the pin's sha512x3 slot)
TOY = {"T": 40, "NS": 2, "NH": 2, "DR": 16, "DV": 32, "BN": 16, "SCALE": 0.0625}
#: TOY at DV = 128: two column blocks of 64, and at DR = 64, so its key rows are three of 64
WIDE = {**TOY, "DV": 128, "DR": 64}
GLM = {"T": 272, "NS": 5, "NH": 8, "DR": 64, "DV": 512, "BN": 64, "SCALE": 0.0625}
#: step 16's sparse-MLA row (`glm53_rows.sparse_mla_split`), 78 of them at TP 8
SERVED = ('SparseMlaSplit_v1{T=272,NS=5,NH=8,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},INV={"fn":"Fa3InvSum_v1"},'
          'SCALE={"f64":"0x1.0000000000000p-4"}}')


def row(S: dict = TOY, **over) -> str:
    return descriptor_id(bind(W.SparseMlaSplit, **{**S, **over}, **FA3))


def block(nvis: int | None = None, S: dict = TOY, boolean: bool = False) -> str:
    """The FIRST block template at ``nvis`` keys, or with None the full block after it."""
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadBlockV2, **at, **st, DOT=BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadBlock, **at, **st, DOT=FA3["DOT"]))


def end(S: dict = TOY) -> str:
    return descriptor_id(bind(M.SparseMlaHeadEnd, DV=S["DV"], INV=FA3["INV"]))


def cols(c: int, S: dict = TOY, boolean: bool = False) -> str:
    G = W.combine_cols(S["DV"])
    return descriptor_id(bind(BX.SparseMlaSplitEndColsV2, DV=S["DV"], G=G, C=c, INV=BOOL["INV"], SCALE=S["SCALE"]) if boolean
                         else bind(W.SparseMlaSplitEndCols, DV=S["DV"], G=G, C=c, INV=FA3["INV"], SCALE=S["SCALE"]))


def comb(n: int, S: dict = TOY, boolean: bool = False, G: int | None = None) -> str:
    G = G or W.combine_cols(S["DV"])
    return descriptor_id(bind(BX.SparseMlaCombineV2, NSPLIT=n, DV=G, INV=BOOL["INV"]) if boolean
                         else bind(W.SparseMlaCombine, NSPLIT=n, DV=G, INV=FA3["INV"]))


def scores(nvis: int | None = None, S: dict = TOY, boolean: bool = False, G: int | None = None) -> str:
    """The FIRST block's scores template at ``nvis`` keys, or with None the full block's after it, its key rows in column
    blocks of ``G`` columns (half the row's DV by default)."""
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadScoresColsV2, **at, **st, G=G or S["DV"] // 2, DOT=BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadScoresCols, **at, **st, G=G or S["DV"] // 2, DOT=FA3["DOT"]))


def keys(nvis: int | None = None, S: dict = TOY, boolean: bool = False, G: int | None = None, C: int = 0) -> str:
    """The writer of column block ``C`` of ``G`` columns (half the row's DV by default) of a key block's key rows at ``nvis``
    keys, or with None at BN."""
    at = {"NVIS": nvis or S["BN"], "W": G or S["DV"] // 2, "C": C}
    return descriptor_id(bind(BM.SparseMlaHeadKeysWireV2, **at) if boolean else bind(M.SparseMlaHeadKeysWire, **at))


def values(nvis: int | None = None, S: dict = TOY, boolean: bool = False, G: int | None = None, C: int = 0) -> str:
    """The values template of `scores`' block on column block ``C`` of ``G`` columns (half the row's by default)."""
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadValuesV2, **at, BN=S["BN"], G=G or S["DV"] // 2, C=C, DOT=BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadValues, **at, BN=S["BN"], G=G or S["DV"] // 2, C=C, DOT=FA3["DOT"]))


def split_end(S: dict = TOY, G: int | None = None) -> str:
    """A split's end on a column block of ``G`` columns (half the row's by default)."""
    return descriptor_id(bind(W.SparseMlaSplitEnd, DV=G or S["DV"] // 2, INV=FA3["INV"], SCALE=S["SCALE"]))


F16, B16, F8, E0, C2 = block(16), block(), block(8), cols(0), comb(2)
TMPLS = [F16, B16, F8, E0, C2]
PER_CHUNK = {F16: 3, B16: 4, F8: 4, E0: 3, C2: 4}
#: ``TMPLS`` with each key block its Boolean body, what a partition names for a window proved under ``--zk``
BITS = [block(16, boolean=True), block(boolean=True), block(8, boolean=True), E0, C2]
BITS_PER_CHUNK = dict(zip(BITS, PER_CHUNK.values()))
#: TOY's places in a head's chain: split 0's blocks 1 and 0 (visit order) and its column end, split 1's block 2 and its column
#: end, then the combine; per block place, its split, its block in the split's visit order and its keys
PLACES = {F16: [0], B16: [1], E0: [2, 4], F8: [3], C2: [5]}
BLOCKS = {0: (0, 0, (16, 32)), 1: (0, 1, (0, 16)), 3: (1, 0, (32, 40))}
#: TOY at DV = 64, where a values unit's O on half the columns is a row of 128 bytes: Flock's compose needs some row of a
#: circuit to take a compression before its last block, and a combine of 16 columns has none; and at DR = 32, so its key rows
#: are three column blocks of 32
HALF = {**TOY, "DV": 64, "DR": 32}
def chain_units(S: dict, boolean: bool = False, G: int | None = None, combine: str | None = None) -> list[str]:
    """A TOY-shaped row's templates with each key block its scores unit and its values units on ``G`` columns (half the row's
    DV by default): the scores of split 0's 16-key first block, of the full block after it and of split 1's 8-key first block,
    each one's values on column block 0, a split's end on a column block, the two-split combine of one (``combine``), the
    writers of column block 0 of the 16-key and 8-key blocks' key rows, then the values on each later column block and the
    writers of each later column block."""
    G = G or S["DV"] // 2
    return [*(scores(n, S, boolean, G) for n in (16, None, 8)), *(values(n, S, boolean, G) for n in (16, None, 8)),
            split_end(S, G), combine or comb(2, S, G=G), *(keys(n, S, boolean, G) for n in (16, 8)),
            *(values(n, S, boolean, G, c) for c in range(1, S["DV"] // G) for n in (16, None, 8)),
            *(keys(n, S, boolean, G, c) for c in range(1, (S["DV"] + S["DR"]) // G) for n in (16, 8))]


#: ``HALF``'s row with each key block its scores unit and its values units on 32 columns: the scores of split 0's 16-key first
#: block, of the full block after it and of split 1's 8-key first block, each one's values on column block 0, a split's end on
#: a column block, the two-split combine of one, the writers of column block 0 of the 16-key and 8-key blocks' key rows, of 32
#: columns, then the values on column block 1 and the writers of column blocks 1 and 2
UNITS = chain_units(HALF)
UNITS_CHUNK = dict(zip(UNITS, (3, 4, 4, 3, 4, 4, 5, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4)))
#: ``UNITS`` with each scores, values and key rows' unit its Boolean body
UNITS_BITS = chain_units(HALF, True)
UNITS_BITS_CHUNK = dict(zip(UNITS_BITS, UNITS_CHUNK.values()))
#: ``UNITS``' places in a head's chain (`chain_rows.places`), past the six of its whole key blocks: the scores units, then per
#: scores unit its values units on column blocks 0 and 1, each split's ends on them, and the combines; a key rows' writer's in
#: its row (`chain_rows.key_places`): split 0's column blocks 0, 1 and 2, each its blocks in visit order, then split 1's
UNITS_PLACES = dict(zip(UNITS, ([6], [7], [8], [9], [11], [13], [15, 16, 17, 18], [19, 20], [0, 1], [6], [10], [12], [14],
                                [2, 3], [7], [4, 5], [8])))


class Served:
    """``n`` rows of ``spec`` (row p's q_nope, q_rope and kv are arguments 3p, 3p + 1 and 3p + 2) and the store of their served
    outputs, the twin's whole row (`sparse_mla_split_twins.head_split_rows`, the word ``SparseMlaHeadSplit_v1`` per head), with
    each split's block states (`split_chain_rows`)."""

    def __init__(self, spec: str = row(), n: int = 1, seed: int = 3):
        S = spec_statics(spec)
        T, NS, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NS", "NH", "DR", "DV", "BN"))
        rng = np.random.default_rng(seed)

        def bf16(shape, s=1.0):
            return ((rng.standard_normal(shape).astype(np.float32) * np.float32(s)).view(np.uint32) >> 16).astype(np.uint16)

        self.ops, self.rows, ys, self.chains = {}, [], [], []
        for p in range(n):
            qn, qr, kv = bf16((NH, DV)), bf16((NH, DR)), bf16((T, DV + DR), 0.5)
            y, nf = TW.head_split_rows(qn, qr, kv, T=T, NS=NS, DR_=DR, DV=DV, BN=BN, scale=float(S["SCALE"]))
            assert not nf.any()
            self.chains.append(TW.split_chain_rows(qn, qr, kv, T=T, NS=NS, DR_=DR, DV=DV, BN=BN, scale=float(S["SCALE"]))[0])
            ys.append(np.asarray(y, dtype=np.uint16).reshape(-1))
            self.ops.update({3 * p: qn.reshape(-1), 3 * p + 1: qr.reshape(-1), 3 * p + 2: kv.reshape(-1)})
            r = {"i": 5 + p, "spec": spec, "args": [[["x", 3 * p + g, 0, self.ops[3 * p + g].size]] for g in range(3)]}
            d = {"statics": S, "op_path": "model.layers.3.self_attn.attn", "members": [("out", NH * DV, "bf16")], "row": p, "step": 16}
            self.rows.append(("r0", None, r, d))
        self.y = np.concatenate(ys)

    def slices(self, _pi, _store, _weights, _rid, _arrive, g):
        """`serving_rows._slices`: an argument group's elements."""
        return [self.ops[g[0][1]]]

    def words(self, rid, step, op, member, lo, hi):
        return self.y[lo:hi], None


def members(s: Served, tmpls=TMPLS, per_chunk=PER_CHUNK) -> list[CS.Member]:
    return CH.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, per_chunk), tmpls, [s.rows] * len(tmpls))


def window(s: Served, tmpls=TMPLS, per_chunk=PER_CHUNK, design: dict | None = None) -> CS.Binding:
    """The window of ``s``'s rows over ``tmpls`` (`members`) under ``design`` (one unit of each template by default)."""
    ms = members(s, tmpls, per_chunk)
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Binding(*CS.population_program(defs, several=True), design or D.parse_design("stratified:1"), "ab" * 32, tuple(ms))


def read(s: Served, ms, tmpls=TMPLS) -> list[tuple[dict, list]]:
    SRW._slices = s.slices
    return SRW.read_all(ms, [s.rows] * len(tmpls), s, None)


def held_values(w: CS.Binding, got: list[tuple[dict, list]]) -> list[dict]:
    """``w``'s members' values ``got`` (`read`'s) as serving commits them: each port a member holds in its writer's tree given
    as its writer's (`chain_rows.held_rows`)."""
    return CH.held_rows(w, [g for g, _e in got])


@pytest.fixture(autouse=True)
def _slices(monkeypatch):
    """`serving_rows._slices` back after each test, which `read` replaces."""
    monkeypatch.setattr(SRW, "_slices", SRW._slices)


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "split-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def committed(ms, got) -> dict:
    """Every port's words of ``ms`` as read (``got``), each wired row held to its writer's output row (`wiring.salts`)."""
    ws = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, ws, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    return ws


def is_keys(t: str) -> bool:
    return CH.kind(SRW.template(t)) == "keys"


def units(ms, tmpls, places: dict[str, list[int]] = PLACES, NH: int = TOY["NH"], rows: int = 1) -> dict[tuple, int]:
    """The global unit of each (row, head, place) in its head's chain: a template's units row by row, then head, then its
    places; a key rows' writer's (`key_units`) are not a head's."""
    out: dict[tuple, int] = {}
    for t in tmpls:
        if is_keys(t):
            continue
        u = min(m.base for m in ms if m.template == t)
        for p in range(rows):
            for h in range(NH):
                for j in places[t]:
                    out[(p, h, j)], u = u, u + 1
    return out


def key_units(ms, tmpls, places: dict[str, list[int]], rows: int = 1) -> dict[tuple, int]:
    """The global unit of each (row, place) of a key rows' writer in its row (`chain_rows.key_places`): a template's units row
    by row, then its places."""
    out: dict[tuple, int] = {}
    for t in filter(is_keys, tmpls):
        u = min(m.base for m in ms if m.template == t)
        for p in range(rows):
            for j in places[t]:
                out[(p, j)], u = u, u + 1
    return out


def words(x) -> list[int]:
    return np.asarray(x, dtype=np.uint32).reshape(-1).view("<u2").tolist()


def test_a_split_row_holds_each_split_s_chain_its_column_ends_and_the_combines():
    """The units `sparse_mla_split.split_chain` cuts the row into, each a ``chain`` template, in a head's places as
    `chain_rows.plan` orders them; a whole block's EX2 reads go in lookup slots, the rest inline, and a key block's Boolean
    body (``BITS``, each `boolean_version`'s), reading no table, is flat.  At NS = 1 (prefill's) the row is a ``SparseMla_v1``
    row's chain."""
    assert SRW.unit_templates(row())[:6] == ((row(), 1), (F16, 2), (F8, 2), (B16, 2), (E0, 4), (C2, 2))
    assert [(p.key, p.split, p.block, p.keys, p.col, p.reads) for p in CH.plan(TOY)] == [
        (("block", 16, True), 0, 0, (16, 32), 0, ()),
        (("block", 16, False), 0, 1, (0, 16), 0, tuple((x, None, o, 0) for x, o in (("m_prev", "o0"), ("l_prev", "o1"), ("o_prev", "o2")))),
        (("cols", 0), 0, 1, (0, 0), 0, tuple((x, None, o, 1) for x, o in (("m", "o0"), ("l", "o1"), ("o", "o2")))),
        (("block", 8, True), 1, 0, (32, 40), 0, ()),
        (("cols", 0), 1, 0, (0, 0), 0, tuple((x, None, o, 3) for x, o in (("m", "o0"), ("l", "o1"), ("o", "o2")))),
        (("combine", 2), 0, 0, (0, 0), 0, (("lse", 0, "o1", 2), ("lse", 1, "o1", 4), ("o", 0, "o0", 2), ("o", 1, "o0", 4)))]
    assert {t: CH.positions(SRW.template(t), TOY) for t in TMPLS} == PLACES
    assert row(GLM) == SERVED
    assert SRW.unit_templates(SERVED)[:12] == ((SERVED, 1), (block(64, GLM), 32), (block(16, GLM), 8),
                                               *((cols(c, GLM), 40) for c in range(8)), (comb(5, GLM), 64))
    assert SRW.unit_templates(row(NS=1))[:4] == ((row(NS=1), 1), (F8, 2), (B16, 4), (end(), 2))
    assert SRW.unit_templates(row(GLM, T=130, NS=1))[:4] == ((row(GLM, T=130, NS=1), 1), (block(2, GLM), 8),
                                                             (block(None, GLM), 16), (end(GLM), 8))
    assert [(SRW.template(t).layout, SRW.template(t).word, CR.reads(SRW.template(t)), CH.kind(SRW.template(t))) for t in TMPLS] == [
        ("chain", F16, "slots", "first"), ("chain", B16, "slots", "block"), ("chain", F8, "inline", "first"),
        ("chain", E0, "inline", "cols"), ("chain", C2, "inline", "combine")]
    assert [SRW.template(t).word for t in (cols(0, boolean=True), comb(2, boolean=True))] == [E0, C2]
    assert [(SRW.template(t).word, CR.reads(SRW.template(t))) for t in BITS] == [(t, "inline") for t in TMPLS]
    assert [descriptor_id(BB.boolean_version(SRW.template(t).fn)) for t in TMPLS[:3]] == BITS[:3]
    DV = TOY["DV"]
    assert CR.unit_rows(SRW.template(E0)) == ([("in0", V2, 32, ["m"]), ("in1", V2, 4 * 32, ["l"]), ("in2", V2, 32 * DV, ["o"])],
                                              [(V2, (0,)), (V2, (1,))])
    assert CR.unit_rows(SRW.template(C2)) == ([(f"in0.{s}", V2, 32, [f"lse[{s}]"]) for s in range(2)]
                                              + [(f"in1.{s}", V2, 32 * DV, [f"o[{s}]"]) for s in range(2)], [(V2, (0,))])
    for t, key in ((comb(3), "('combine', 3)"), (end(), "('end',)"), (block(4), "('block', 4, True)")):
        with pytest.raises(ValueError, match=re.escape(f"a row of T=40 keys in NS=2 splits holds no unit {key} of it")):
            CH.positions(SRW.template(t), spec_statics(row()))


def test_the_key_blocks_boolean_bodies_are_the_same_units_rows_and_wires():
    """A window over ``BITS`` is the window over ``TMPLS`` but for its key blocks' template ids: each member's rows, outputs,
    units and wires, and every word serving reads for it."""
    s = Served(n=2)
    ms, bits = members(s), members(s, BITS, BITS_PER_CHUNK)
    of = dict(zip(TMPLS, BITS))
    assert [(of[m.template], m.rows, m.outs, m.n, m.base, m.wired) for m in ms] == [
        (m.template, m.rows, m.outs, m.n, m.base, m.wired) for m in bits]
    for (g, e), (gb, eb) in zip(read(s, ms), read(s, bits, BITS), strict=True):
        assert e == eb and all(np.array_equal(g[x][q], gb[x][q]) for x in ("rows", "outs") for q in g[x] | gb[x])


def test_each_read_row_is_wired_to_the_unit_that_wrote_it():
    ms = members(Served(n=2))
    assert [(m.template, m.n) for m in ms] == [(F16, 3), (F16, 1), (B16, 4), (F8, 4), (E0, 3), (E0, 3), (E0, 2), (C2, 4)]
    unit = units(ms, TMPLS, rows=2)
    at = {u: key for key, u in unit.items()}
    writer = {1: 0, 2: 1, 4: 3}
    for m in ms:
        keys = [at[m.base + i] for i in range(m.n)]
        if m.template in (F16, F8):
            assert m.wired == ()
        elif m.template == C2:
            assert m.wired == tuple((f"in{g}.{s}", o, tuple(unit[(p, h, (2, 4)[s])] for p, h, _j in keys))
                                    for g, o in ((0, "o1"), (1, "o0")) for s in range(2))
        else:
            prev = tuple(unit[(p, h, writer[j])] for p, h, j in keys)
            names = ("m_prev", "l_prev", "o_prev") if m.template == B16 else ("m", "l", "o")
            g = {nm: i for i, (nm, _ty) in enumerate(SRW.template(m.template).fn.signature()[0])}
            assert m.wired == tuple((f"in{g[x]}", o, prev) for x, o in zip(names, ("o0", "o1", "o2")))
    WI.validate(ms)


def test_the_units_read_the_twin_s_states_and_split_ends_whose_combine_is_the_served_row():
    """Every wired row is its writer's output row, each unit's ports are its block's keys, the twin's states and split ends,
    and each unit's Definition on its rows (as Flock's stager parses them) computes its committed outputs."""
    s = Served(n=2)
    ms = members(s)
    got = read(s, ms)
    ws = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, ws, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    unit, S = units(ms, TMPLS, rows=2), TOY
    at = {u: key for key, u in unit.items()}
    ends = [[TW.split_end_rows(*c[-1], scale=S["SCALE"]) for c in chains] for chains in s.chains]

    def state(p, h, sp, k):
        return [words(x[h]) for x in s.chains[p][sp][k]]

    for m, (g, _entries) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
        for i in range(m.n):
            p, h, j = at[m.base + i]
            if j in BLOCKS:
                sp, k, (lo, hi) = BLOCKS[j]
                qn, qr = s.ops[3 * p].reshape(S["NH"], -1)[h], s.ops[3 * p + 1].reshape(S["NH"], -1)[h]
                kv = s.ops[3 * p + 2].reshape(S["T"], -1)[lo:hi]
                assert [g["rows"][x][i].tolist() for x in ("in0", "in1", "in2")] == [qn.tolist(), qr.tolist(), kv.reshape(-1).tolist()]
                if k:
                    assert [g["rows"][x][i].tolist() for x in ("in3", "in4", "in5")] == state(p, h, sp, k - 1)
                assert [g["outs"][x][i].tolist() for x in ("o0", "o1", "o2")] == state(p, h, sp, k)
            elif j in (2, 4):
                sp = (j - 2) // 2
                assert [g["rows"][x][i].tolist() for x in ("in0", "in1", "in2")] == state(p, h, sp, len(s.chains[p][sp]) - 1)
                part, lse = ends[p][sp]
                assert [g["outs"]["o0"][i].tolist(), g["outs"]["o1"][i].tolist()] == [words(part[h]), words(lse[h])]
            else:
                assert [g["rows"][f"in0.{sp}"][i].tolist() for sp in range(2)] == [words(ends[p][sp][1][h]) for sp in range(2)]
                assert [g["rows"][f"in1.{sp}"][i].tolist() for sp in range(2)] == [words(ends[p][sp][0][h]) for sp in range(2)]
                assert g["outs"]["o0"][i].tolist() == s.y.reshape(2, S["NH"], -1)[p, h].tolist()
    for t, o in ((F16, "o2"), (E0, "o0"), (E0, "o1"), (C2, "o0")):
        g = next(g for m, (g, _e) in zip(ms, got) if m.template == t)
        with pytest.raises(AssertionError, match="outputs are not the IR reference's"):     # an output its unit does not compute
            stager(t).reference_check(g["rows"], {**g["outs"], o: g["outs"][o] ^ np.uint16(1)})


def test_a_combine_of_two_column_blocks_reads_each_column_end_of_its_own():
    """At DV = 128 each split ends in two column blocks of 64, and the combine of column block c reads column end c of every
    split; the combines' outputs, column block by column block, are the served row."""
    s = Served(row(WIDE), n=1)
    tmpls = [block(16, WIDE), block(None, WIDE), block(8, WIDE), cols(0, WIDE), cols(1, WIDE), comb(2, WIDE)]
    assert SRW.unit_templates(row(WIDE))[1:7] == tuple(zip([tmpls[0], tmpls[2], tmpls[1], *tmpls[3:]], (2, 2, 2, 4, 4, 4)))
    ms = members(s, tmpls, dict.fromkeys(tmpls, 64))
    places = dict(zip(tmpls, ([0], [1], [4], [2, 5], [3, 6], [7, 8])))
    assert {t: CH.positions(SRW.template(t), WIDE) for t in tmpls} == places
    unit = units(ms, tmpls, places)
    (c2,) = [m for m in ms if m.template == tmpls[-1]]
    assert c2.wired == tuple((f"in{g}.{sp}", o, tuple(unit[(0, h, (2 + c, 5 + c)[sp])] for h in range(2) for c in range(2)))
                             for g, o in ((0, "o1"), (1, "o0")) for sp in range(2))
    got = read(s, ms, tmpls)
    ws = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, ws, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    assert ws[c2.port("o0")].reshape(-1).tolist() == s.y.tolist()
    ends = [TW.split_end_rows(*c[-1], scale=WIDE["SCALE"]) for c in s.chains[0]]
    for c in range(2):
        (m,) = [m for m in ms if m.template == tmpls[3 + c]]
        want = [words(ends[sp][0][h, 64 * c:64 * (c + 1)]) for h in range(2) for sp in range(2)]
        assert ws[m.port("o0")].tolist() == want


def test_a_split_row_at_ns_1_reads_and_wires_as_a_chain():
    """Prefill's split count: the row's units are a ``SparseMla_v1`` row's, its end the served head."""
    s = Served(row(NS=1), n=1)
    tmpls = [F8, B16, end()]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 8))
    got = read(s, ms, tmpls)
    ws = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, ws, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    assert ws[ms[-1].port("o0")].reshape(-1).tolist() == s.y.tolist()


def test_a_row_s_statics_as_the_record_spells_them_read_as_a_binding_s():
    """A TP window's rows carry their statics as the record spells them (`query.manifest.format.spec_statics`: every value a
    string, SCALE ``{"f64":"<hex>"}``), and read as the binding's."""
    s = Served(n=2)
    want = read(s, members(s))
    s.rows = [(rid, pi, r, {**d, "statics": MF.spec_statics(r["spec"])}) for rid, pi, r, d in s.rows]
    assert s.rows[0][3]["statics"]["SCALE"] == '{"f64":"0x1.0000000000000p-4"}'
    got = read(s, members(s))
    assert [{x: {q: a.tolist() for q, a in g[x].items()} for x in ("rows", "outs")} for g, _e in got] == \
        [{x: {q: a.tolist() for q, a in g[x].items()} for x in ("rows", "outs")} for g, _e in want]


def test_a_row_whose_served_words_are_not_its_splits_combine_is_refused():
    s = Served()
    s.y[3] ^= np.uint16(1)
    with pytest.raises(ValueError, match="the twin's end of its 3 blocks is not the row's served out words"):
        read(s, members(s))


def test_a_partition_without_a_writer_s_template_is_refused():
    s = Served()
    for left, why in ((F16, "block 0 (visit order) is the output of a block template"),
                      (F8, "block 0 (visit order) of split 1 is the output of a block template"),
                      (E0, "split 0's end column block 0 is the output of a column end template")):
        with pytest.raises(ValueError, match=re.escape(f"{why} the partition does not name")):
            members(s, [t for t in TMPLS if t != left])
    assert all(m.wired == () for m in members(s, [F16, F8]))


def test_a_row_at_the_served_statics_reads_and_wires_its_424_units():
    """One row of step 16's statics, its 424 units (32 64-key first blocks, 8 16-key ones, 320 column ends, 64 combines), each
    read row wired to its writer and every wired row its writer's output, the combines' outputs the served row; the window
    itself (its population program at these statics lowers the 64-key block) is the circuit-check bindings'."""
    s = Served(SERVED, n=1, seed=16)
    tmpls = [block(64, GLM), block(16, GLM), *(cols(c, GLM) for c in range(8)), comb(5, GLM)]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 512))
    assert [(m.template, m.n) for m in ms] == list(zip(tmpls, (32, 8, *(40,) * 8, 64)))
    got = read(s, ms, tmpls)
    ws = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, ws, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    assert ws[ms[-1].port("o0")].reshape(-1).tolist() == s.y.tolist()


@pytest.mark.slow
def test_a_window_of_one_row_commits_and_its_wires_hold():
    s = Served()
    w = window(s)
    got = read(s, w.members)
    for m, (g, _e) in zip(w.members, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    j = next(j for j, m in enumerate(w.members) if m.template == C2)
    m = w.members[j]
    (p, o, ws), *rest = m.wired

    def rewired(wired):
        return [replace(x, wired=wired) if i == j else x for i, x in enumerate(w.members)]

    assert list(WI.check(rewired(((p, o, ws[1:] + ws[:1]), *rest)), publics)) == [m.key]       # another unit's lse
    slots = {q: x for q, _o, x in m.wired}
    swapped = tuple((q, oo, slots["in1.1"] if q == "in1.0" else slots["in1.0"] if q == "in1.1" else x) for q, oo, x in m.wired)
    assert list(WI.check(rewired(swapped), publics)) == [m.key]                     # split 1's partial in slot 0
    names = [q for q, _ in m.all_rows]
    body = bytearray(publics[m.key])
    body[names.index("in1.0") * WI.BC_BYTES] ^= 1
    assert list(WI.check(w.members, {**publics, m.key: bytes(body)})) == [m.key]
    bad = [dict(g, rows={**g["rows"], "in1.0": g["rows"]["in1.0"] ^ np.uint16(1)}) if i == j else g for i, (g, _e) in enumerate(got)]
    with pytest.raises(ValueError, match=re.escape(f"{m.key}: a in1.0 row is not the o0 row")):
        CS.commit(w, bad, salt_key=bytes(32))


def test_a_split_row_also_holds_each_key_block_as_its_scores_and_values_units():
    """Past its whole key blocks' units a row holds, at each width of `serving_templates.values_widths`, the units
    `sparse_mla_split.split_chain` cuts it into at ``cols``: each key block's scores unit and its values units, its key rows'
    writers (once for all the heads), each split's end on a column block and the combines of those, a Definition two widths
    hold listed once; in a head's places past `plan`'s as `chain_rows.places` orders them, a key rows' writer in its row's as
    `chain_rows.key_places` does.  A width divides DR too, so a key row is whole column blocks.  A whole block's scores unit
    reads EX2 in lookup slots, the rest inline; the Boolean bodies are flat."""
    assert [values_widths(dv, dr) for dv, dr in ((512, 64), (512, 16), (64, 32), (32, 16), (16, 16))] == [
        [64, 32, 16], [16], [32, 16], [16], []]
    def block_of(nvis, n, nk, **at):
        return ((scores(nvis, **at), n), *((values(nvis, C=c, **at), n) for c in range(2)),
                *((keys(nvis, C=c, **at), nk) for c in range(3)))

    assert SRW.unit_templates(row())[6:] == (*block_of(16, 2, 2), *block_of(8, 2, 1), *block_of(None, 2, 2)[:3],
                                             (split_end(), 8), (comb(2, G=16), 4))
    assert SRW.unit_templates(row(HALF))[6:23] == tuple(zip([UNITS[j] for j in (0, 3, 10, 8, 13, 15, 2, 5, 12, 9, 14, 16, 1, 4,
                                                                                11, 6, 7)],
                                                            (2,) * 9 + (1,) * 3 + (2,) * 3 + (8, 4)))
    served = dict(SRW.unit_templates(SERVED)[12:])
    assert len(served) == sum(2 + 2 * 512 // g + 2 * 576 // g + 1 for g in (64, 32, 16)) + 2 and comb(5, GLM) not in served
    assert [served[t] for t in (scores(64, GLM, G=64), scores(16, GLM, G=64), values(64, GLM, G=64, C=7),
                                values(16, GLM, G=64), keys(64, GLM, G=64, C=8), keys(16, GLM, G=64), keys(64, GLM, G=16, C=35),
                                split_end(GLM, 64), split_end(GLM, 32), comb(5, GLM, G=32))] == [32, 8, 32, 8, 4, 1, 4, 320, 640, 128]
    assert dict(SRW.unit_templates(row(NS=1))[4:]) == {
        **dict(block_of(8, 2, 1)), **dict(block_of(None, 4, 2)), descriptor_id(bind(M.SparseMlaHeadEnd, DV=16, INV=FA3["INV"])): 4}
    assert CH.places(HALF)[:6] == CH.plan(HALF) and len(CH.places(HALF)) == 6 + (3 + 6 + 4 + 2) + (3 + 12 + 8 + 4)
    blocks = [[(16, (16, 32)), (16, (0, 16))], [(8, (32, 40))]]
    assert [(p.key, p.split, p.block, p.keys, p.col) for p in CH.key_places(HALF)[:9]] == [
        (("keys", n, 32), s, k, lohi, c) for s in range(2) for c in range(3) for k, (n, lohi) in enumerate(blocks[s])]
    assert len(CH.key_places(HALF)) == 9 + 18
    pb, later, state = ("pb", None, "o2"), (("rescale", None, "o3", 7),), (("m", None, "o0"), ("l", None, "o1"))
    assert [(p.key, p.split, p.block, p.keys, p.col, p.reads) for p in CH.places(HALF)[6:21]] == [
        (("scores", 16, True, 32), 0, 0, (16, 32), 0, ()),
        (("scores", 16, False, 32), 0, 1, (0, 16), 0, (("m_prev", None, "o0", 6), ("l_prev", None, "o1", 6))),
        (("scores", 8, True, 32), 1, 0, (32, 40), 0, ()),
        *((("values", 16, True, 32), 0, 0, (16, 32), c, ((*pb, 6),)) for c in range(2)),
        *((("values", 16, False, 32), 0, 1, (0, 16), c, ((*pb, 7), *later, ("o_prev", None, "o0", 9 + c))) for c in range(2)),
        *((("values", 8, True, 32), 1, 0, (32, 40), c, ((*pb, 8),)) for c in range(2)),
        *((("split", 32), 0, 1, (0, 0), c, (*((*x, 7) for x in state), ("o", None, "o0", 11 + c))) for c in range(2)),
        *((("split", 32), 1, 0, (0, 0), c, (*((*x, 8) for x in state), ("o", None, "o0", 13 + c))) for c in range(2)),
        *((("combine", 2, 32), 0, 0, (0, 0), c, tuple((x, sp, o, 15 + 2 * sp + c) for x, o in (("lse", "o1"), ("o", "o0"))
                                                         for sp in range(2))) for c in range(2))]
    assert {t: CH.positions(SRW.template(t), HALF) for t in UNITS} == UNITS_PLACES
    assert [(SRW.template(t).layout, SRW.template(t).word, CR.reads(SRW.template(t)), CH.kind(SRW.template(t))) for t in UNITS] == [
        ("chain", t, r, k) for t, r, k in zip(UNITS, ("slots", "slots", *("inline",) * 15),
                                              ("scores",) * 3 + ("values",) * 3 + ("split", "combine", "keys", "keys")
                                              + ("values",) * 3 + ("keys",) * 4)]
    assert [CH.units(SRW.template(t), HALF) for t in (UNITS[0], UNITS[8], UNITS[9], UNITS[16])] == [2, 2, 1, 1]
    assert [(SRW.template(t).word, CR.reads(SRW.template(t))) for t in UNITS_BITS] == [(t, "inline") for t in UNITS]
    assert [descriptor_id(BB.boolean_version(SRW.template(t).fn)) for t in UNITS[:6] + UNITS[8:]] == UNITS_BITS[:6] + UNITS_BITS[8:]
    assert CR.unit_rows(SRW.template(UNITS[4])) == (
        [("in0", V2, 16 * 16, ["pb"]), ("in1", V2, 16 * 16 * 32, ["kv"]), ("in2", V2, 32, ["rescale"]), ("in3", V2, 32 * 32, ["o_prev"])],
        [(V2, (0,))])
    assert [p for p, *_ in CR.unit_rows(SRW.template(UNITS[1]))[0]] == [f"in{g}" for g in range(7)]
    assert CR.unit_rows(SRW.template(UNITS[8])) == ([("in0", V2, 16 * 16 * 32, ["kv"])], [(V2, (0,))])
    for t, key in ((values(16, HALF, G=8), "('values', 16, True, 8) of column block 0"),
                   (values(16, HALF, C=2), "('values', 16, True, 32) of column block 2"),
                   (scores(4, HALF), "('scores', 4, True, 32)"), (keys(4, HALF), "('keys', 4, 32) of column block 0"),
                   (keys(16, HALF, C=3), "('keys', 16, 32) of column block 3"), (comb(3, HALF, G=32), "('combine', 3, 32)"),
                   (split_end(HALF, 8), "('split', 8)")):
        with pytest.raises(ValueError, match=re.escape(f"a row of T=40 keys in NS=2 splits holds no unit {key} of it")):
            CH.positions(SRW.template(t), spec_statics(row(HALF)))


def test_each_scores_and_values_unit_is_wired_to_the_units_that_wrote_what_it_reads():
    """A later scores unit reads the scores unit before it; a values unit its own block's scores unit and, past the first,
    the values unit of its column block before it; a split's end the last scores unit and the last values unit of its column
    block; a combine each split's end on its column block.  Each scores unit's key rows are wired to its key block's writers,
    one a column block, and each values unit's to its column block's; those wired rows have no tree of their own, each its
    writer member's output rows (`rows.Member.wired_trees`), so a member is cut where a wire's writer member changes."""
    ms = members(Served(row(HALF), n=2), UNITS, UNITS_CHUNK)
    assert sum(m.n for m in ms) == 2 * (2 * 15 + 9)
    unit, kunit = units(ms, UNITS, UNITS_PLACES, rows=2), key_units(ms, UNITS, UNITS_PLACES, rows=2)
    at = {u: key for key, u in unit.items()}
    pl, kp = CH.places(HALF), CH.key_places(HALF)
    writer = {(p, kp[j].split, kp[j].block, kp[j].col): u for (p, j), u in kunit.items() if kp[j].key[2] == 32}
    bases = WI.bases(ms)
    for m in ms:
        kind = CH.kind(SRW.template(m.template))
        if kind == "keys":
            assert (m.wired, m.wired_trees, m.tables) == ((), True, ())
            continue
        keys = [at[m.base + i] for i in range(m.n)]
        g = {nm: i for i, (nm, _ty) in enumerate(SRW.template(m.template).fn.signature()[0])}
        reads = pl[keys[0][2]].reads
        cols = {"scores": [(f"kv{c}", lambda _j, c=c: c) for c in range(3)], "values": [("kv", lambda j: pl[j].col)]}.get(kind, [])
        assert m.wired == tuple((f"in{g[x]}" + ("" if sp is None else f".{sp}"), o,
                                 tuple(unit[(p, h, pl[j].reads[i][3])] for p, h, j in keys)) for i, (x, sp, o, _to) in enumerate(reads)) + \
            tuple((f"in{g[x]}", "o0", tuple(writer[(p, pl[j].split, pl[j].block, c(j))] for p, _h, j in keys)) for x, c in cols)
        assert bool(m.wired) == (m.template not in (UNITS[0], UNITS[2])) or kind == "scores"
        assert m.wired_trees == (kind not in ("scores", "values"))
        if not m.wired_trees:
            mine = {p: WI.writers(ms, ws, bases) for p, _o, ws in m.wired}
            assert all(len(x) == 1 for x in mine.values())
            assert m.tables == tuple((p, mine[p][0][0].n if p in mine else m.n) for p, _w in m.rows)
    later = [m.key for m in ms if m.template == UNITS[4]]
    assert later == [f"{UNITS[4]}/c{j}" for j in range(len(later))] and len(later) > 2 * 2 // UNITS_CHUNK[UNITS[4]]
    WI.validate(ms)
    s = Served(row(HALF))
    for left, why in ((UNITS[0], "block 0 (visit order)'s scores are the output of a scores template"),
                      (UNITS[5], "block 0 (visit order) of split 1's O on column block 0 is the output of a values template"),
                      (UNITS[6], "split 0's end column block 0 is the output of a column end template"),
                      (UNITS[8], "block 0 (visit order)'s key rows' column block 0 of 32 columns are the output of a keys "
                                 "template"),
                      (UNITS[9], "block 0 (visit order) of split 1's key rows' column block 0 of 32 columns are the output of "
                                 "a keys template")):
        with pytest.raises(ValueError, match=re.escape(f"{why} the partition does not name")):
            members(s, [t for t in UNITS if t != left], UNITS_CHUNK)


def test_the_scores_and_values_units_read_the_twin_s_and_their_combines_are_the_served_row():
    """Every wired row is its writer's output row; a scores unit's ports are its block's keys, in column blocks, and the twin's
    running max, lanes, probabilities and rescaling factor, a values unit's its column block of its block's keys and of the
    twin's O, a key rows' writer's its column block of its block's keys, which it returns; each unit's Definition on its rows
    (as Flock's stager parses them) computes its committed outputs, and the combines' outputs, column block by column block,
    are the served row."""
    s = Served(row(HALF), n=2)
    ms = members(s, UNITS, UNITS_CHUNK)
    got = read(s, ms, UNITS)
    ws = committed(ms, got)
    S, pl, kp = HALF, CH.places(HALF), CH.key_places(HALF)
    unit = units(ms, UNITS, UNITS_PLACES, rows=2)
    at = {u: key for key, u in unit.items()}
    kat = {u: key for key, u in key_units(ms, UNITS, UNITS_PLACES, rows=2).items()}
    tw = [TW.split_chain_rows(s.ops[3 * p].reshape(S["NH"], -1), s.ops[3 * p + 1].reshape(S["NH"], -1),
                              s.ops[3 * p + 2].reshape(S["T"], -1), T=S["T"], NS=S["NS"], DR_=S["DR"], DV=S["DV"], BN=S["BN"],
                              scale=S["SCALE"], units=True)[0] for p in range(2)]
    for m, (g, _entries) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
        for i in range(m.n):
            if m.base + i in kat:
                p, j = kat[m.base + i]
                (lo, hi), c = kp[j].keys, kp[j].col
                kv = s.ops[3 * p + 2].reshape(S["T"], -1)[lo:hi, 32 * c:32 * (c + 1)].reshape(-1).tolist()
                assert g["rows"]["in0"][i].tolist() == g["outs"]["o0"][i].tolist() == kv
                continue
            p, h, j = at[m.base + i]
            sp, k, (lo, hi), c = pl[j].split, pl[j].block, pl[j].keys, pl[j].col
            kv = s.ops[3 * p + 2].reshape(S["T"], -1)[lo:hi]
            if m.template in UNITS[:3]:
                mm, ll, _o, pb, rescale = tw[p][sp][k]
                qn, qr = s.ops[3 * p].reshape(S["NH"], -1)[h], s.ops[3 * p + 1].reshape(S["NH"], -1)[h]
                assert [g["rows"][x][i].tolist() for x in ("in0", "in1")] == [qn.tolist(), qr.tolist()]
                assert [g["rows"][f"in{2 + b}"][i].tolist() for b in range(3)] == [kv[:, 32 * b:32 * (b + 1)].reshape(-1).tolist()
                                                                                  for b in range(3)]
                assert [g["outs"][x][i].tolist() for x in ("o0", "o1")] == [words(mm[h]), words(ll[h])]
                assert g["outs"]["o2"][i].tolist() == np.asarray(pb[h], dtype=np.uint16).tolist()
                assert (rescale is None) == ("o3" not in g["outs"]) and (rescale is None or g["outs"]["o3"][i].tolist() == words(rescale[h]))
            elif CH.kind(SRW.template(m.template)) == "values":
                assert g["rows"]["in1"][i].tolist() == kv[:, 32 * c:32 * (c + 1)].reshape(-1).tolist()
                assert g["outs"]["o0"][i].tolist() == words(tw[p][sp][k][2][h, 32 * c:32 * (c + 1)])
    assert np.concatenate([ws[m.port("o0")].reshape(-1) for m in ms if m.template == UNITS[7]]).tolist() == s.y.tolist()
    for t, o in ((UNITS[0], "o2"), (UNITS[1], "o3"), (UNITS[4], "o0"), (UNITS[6], "o1"), (UNITS[7], "o0")):
        g = next(g for m, (g, _e) in zip(ms, got) if m.template == t)
        with pytest.raises(AssertionError, match="outputs are not the IR reference's"):     # an output its unit does not compute
            stager(t).reference_check(g["rows"], {**g["outs"], o: g["outs"][o] ^ np.uint16(1)})


def test_the_scores_and_values_units_boolean_bodies_are_the_same_units_rows_and_wires():
    s = Served(row(HALF), n=2)
    ms, bits = members(s, UNITS, UNITS_CHUNK), members(s, UNITS_BITS, UNITS_BITS_CHUNK)
    of = dict(zip(UNITS, UNITS_BITS))
    assert [(of[m.template], m.rows, m.outs, m.n, m.base, m.wired, m.wired_trees, m.tables) for m in ms] == [
        (m.template, m.rows, m.outs, m.n, m.base, m.wired, m.wired_trees, m.tables) for m in bits]
    for (g, e), (gb, eb) in zip(read(s, ms, UNITS), read(s, bits, UNITS_BITS), strict=True):
        assert e == eb and all(np.array_equal(g[x][q], gb[x][q]) for x in ("rows", "outs") for q in g[x] | gb[x])


def test_a_combine_of_64_columns_reads_each_split_s_end_from_the_unit_the_partition_names():
    """At DV = 128 a combine of 64 columns is one template whether its splits end in column blocks of whole key blocks or on
    column blocks of values units: it reads each split's end from the one the partition names; a partition that names
    neither is refused, and so is one that names the row's key blocks both ways, or as values units of two widths."""
    s = Served(row(WIDE), n=1)
    ends, c2 = [cols(0, WIDE), cols(1, WIDE)], comb(2, WIDE)
    whole = [block(16, WIDE), block(None, WIDE), block(8, WIDE), *ends, c2]
    tmpls = chain_units(WIDE, combine=c2)
    assert CH.positions(SRW.template(c2), WIDE) == [7, 8] and CH.plan(WIDE)[7].reads[0] == ("lse", 0, "o1", 2)
    assert [CH.places(WIDE)[j].reads for j in (7, 8)] == [
        tuple((x, sp, o, ((2, 5)[sp] + c, 18 + 2 * sp + c)) for x, o in (("lse", "o1"), ("o", "o0")) for sp in range(2))
        for c in range(2)]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 64))
    places = dict(zip(tmpls, ([9], [10], [11], [12], [14], [16], [18, 19, 20, 21], [7, 8], [0, 1], [6], [13], [15], [17],
                              [2, 3], [7], [4, 5], [8])))
    assert {t: CH.positions(SRW.template(t), WIDE) for t in tmpls} == places
    unit = units(ms, tmpls, places)
    (m,) = [m for m in ms if m.template == c2]
    assert m.wired == tuple((f"in{g}.{sp}", o, tuple(unit[(0, h, 18 + 2 * sp + c)] for h in range(2) for c in range(2)))
                            for g, o in ((0, "o1"), (1, "o0")) for sp in range(2))
    assert committed(ms, read(s, ms, tmpls))[m.port("o0")].reshape(-1).tolist() == s.y.tolist()
    chunk = dict.fromkeys([*whole, *tmpls, values(16, WIDE, G=32)], 64)
    for both, ways in (([*whole[:5], *tmpls], f"whole ({whole[0]}) and as units of 64 columns ({tmpls[0]})"),
                       ([*tmpls[3:], *ends], f"whole ({ends[0]}) and as units of 64 columns ({tmpls[3]})"),
                       ([*tmpls[:4], values(16, WIDE, G=32)],
                        f"as units of 32 columns ({values(16, WIDE, G=32)}) and as units of 64 columns ({tmpls[0]})")):
        with pytest.raises(ValueError, match=re.escape(f"row 5: the partition names its key blocks {ways}; a row's key blocks "
                                                       f"are served one way")):
            members(s, both, chunk)
    with pytest.raises(ValueError, match=re.escape("split 0's end column block 0 is the output of a column end template the "
                                                   "partition does not name")):
        members(s, [*tmpls[:6], c2, *tmpls[8:]], dict.fromkeys(tmpls, 64))


def test_a_row_named_at_every_width_of_its_units_is_refused_by_its_two_narrowest():
    """A row of DV = 512 and DR = 64 has units of three widths (`serving_templates.values_widths`, each dividing DR too); a
    partition that names its key blocks at all three is refused as one that names two is, and the refusal names the two
    narrowest."""
    S = {**TOY, "DV": 512, "DR": 64}
    widths = values_widths(512, 64)
    assert widths == [64, 32, 16]
    tmpls = [scores(16, S, G=64), *(values(16, S, G=g) for g in widths)]
    with pytest.raises(ValueError, match=re.escape(
            f"row 5: the partition names its key blocks as units of 16 columns ({values(16, S, G=16)}) and as units of 32 "
            f"columns ({values(16, S, G=32)}); a row's key blocks are served one way")):
        members(Served(row(S), n=1), tmpls, dict.fromkeys(tmpls, 64))


def test_a_row_at_ns_1_in_scores_and_values_units_ends_on_each_column_block():
    """Prefill's split count: the chain's end on each column block reads the last scores unit's lanes and the last values
    unit of its column block's O, and the ends' outputs, column block by column block, are the served row."""
    s = Served(row(HALF, NS=1), n=1)
    e = descriptor_id(bind(M.SparseMlaHeadEnd, DV=32, INV=FA3["INV"]))
    tmpls = [scores(8, HALF), scores(None, HALF), values(8, HALF), values(None, HALF), e, keys(8, HALF), keys(None, HALF),
             values(8, HALF, C=1), values(None, HALF, C=1), *(keys(n, HALF, C=c) for c in (1, 2) for n in (8, None))]
    assert [CH.kind(SRW.template(t)) for t in tmpls] == ["scores", "scores", "values", "values", "end", "keys", "keys", "values",
                                                         "values", "keys", "keys", "keys", "keys"]
    assert {t: CH.positions(SRW.template(t), {**HALF, "NS": 1}) for t in tmpls} == dict(zip(tmpls, (
        [4], [5, 6], [7], [9, 11], [13, 14], [0], [1, 2], [8], [10, 12], [3], [4, 5], [6], [7, 8])))
    assert [CH.places({**HALF, "NS": 1})[j].reads for j in (13, 14)] == [(("l", None, "o1", 6), ("o", None, "o0", 11 + c)) for c in range(2)]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 16))
    got = read(s, ms, tmpls)
    for m, (g, _e) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    (end,) = [m for m in ms if m.template == e]
    assert committed(ms, got)[end.port("o0")].reshape(-1).tolist() == s.y.tolist()


def test_a_row_at_the_served_statics_reads_and_wires_its_789_units_in_columns_of_64():
    """One row of step 16's statics with each key block its scores unit, eight values units and nine key rows' writers of 64
    columns: 40 scores units, 320 values units, 45 writers, 320 split ends and the 64 combines its whole key blocks' chain
    holds too, each read row wired to its writer and every wired row its writer's output, the combines' outputs the served
    row."""
    s = Served(SERVED, n=1, seed=16)
    tmpls = served_units()
    ms = members(s, tmpls, dict.fromkeys(tmpls, 512))
    assert [(m.template, m.n) for m in ms] == list(zip(tmpls, (32, 8, *(32,) * 8, *(8,) * 8, 320, 64, *(4,) * 9, *(1,) * 9)))
    (c,) = [m for m in ms if m.template == comb(5, GLM)]
    assert committed(ms, read(s, ms, tmpls))[c.port("o0")].reshape(-1).tolist() == s.y.tolist()


def served_units(boolean: bool = False) -> list[str]:
    """What a partition names to serve step 16's rows in scores and values units of 64 columns: each key block's scores unit,
    values units (one template a column block) and key rows' writers (one a column block), their Boolean bodies with
    ``boolean``, a split's end on a column block and the combine served beside whole key blocks too."""
    return [scores(64, GLM, boolean, 64), scores(16, GLM, boolean, 64), *(values(n, GLM, boolean, 64, c) for n in (64, 16)
                                                                          for c in range(8)),
            split_end(GLM, 64), comb(5, GLM), *(keys(n, GLM, boolean, 64, c) for n in (64, 16) for c in range(9))]


SERVED_UNITS = served_units(True)


def test_every_served_scores_and_values_unit_composes_and_verify_zk_takes_it():
    """At GLM-5.3's statics each unit of a row served in scores, values and key rows' units of 64 columns is a proof block
    Flock's composer takes (`interface.check_proof_block`: each has a row past 111 bytes, and its rows fit 2^26 bits at
    most), and `verify --zk` takes it (`chunk_rows.zk_refusal`): a scores, values or key rows' unit's Boolean body reads no
    table, and a split's end and the combine read theirs inline.  Every returned field, each one a wire may read, is whole
    16-bit words."""
    floors = []
    for t in SERVED_UNITS:
        tp = SRW.template(t)
        assert CR.zk_refusal(tp) is None and CR.statement(tp)["reads"] == "inline", t
        assert CR.slot_tables(tp.fn) == (() if tp.id != tp.word else CR.slot_tables(tp.fn)), t
        ret = tp.fn.signature()[1]
        assert all(sum(f.leaf_widths()) % 16 == 0 for f in (field_types(ret) if isinstance(ret, Tuple) else (ret,))), t
        floors.append(IF.check_proof_block(tp.fn))
    assert floors == [26, 25, *(23,) * 8, *(22,) * 8, 22, 22, *(24,) * 9, *(22,) * 9]


def test_a_window_of_scores_and_values_units_commits_and_its_wires_hold():
    """The window commits with every port a scores or values unit holds in its writer's tree given as its writer's
    (`chain_rows.held_rows`): such a port has no root of its own, its writer's output port does, and every wire holds on the
    public bodies; a wire to another unit's row is refused by the wiring check, and a held row that is not its writer's
    output row before anything is committed."""
    s = Served(row(HALF))
    w = window(s, UNITS, UNITS_CHUNK)
    got = read(s, w.members, UNITS)
    c = CS.commit(w, held_values(w, got), salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    held = [(m, p) for m in w.members if not m.wired_trees for p, _o, _ws in m.wired]
    assert len({m.template for m, _p in held}) == 9 and all(m.port(p) not in c.roots and w.held(m, p) in c.roots for m, p in held)
    j = next(j for j, m in enumerate(w.members) if m.template == UNITS[4])
    m = w.members[j]
    (p, o, ws), *rest = m.wired
    assert (p, o) == ("in0", "o2")
    rewired = [replace(x, wired=((p, o, ws[1:] + ws[:1]), *rest)) if i == j else x for i, x in enumerate(w.members)]
    assert list(WI.check(rewired, publics)) == [m.key]                              # another block's probabilities
    bad = [dict(g, rows={**g["rows"], "in3": g["rows"]["in3"] ^ np.uint16(1)}) if i == j else g for i, (g, _e) in enumerate(got)]
    with pytest.raises(ValueError, match=re.escape(f"{m.key}: its 'in3' rows are not the 'o0' rows of the ")):
        CH.held_rows(w, bad)


def other_keys(s: Served, w: CS.Binding) -> tuple[list[dict], int]:
    """``s``'s rows read for ``w`` (over ``UNITS``) with head 0's values unit of split 1's first block on the keys' other 32
    columns, as its column block of the keys, and its O, its split's end and the combine recomputed from them, so that every
    unit's outputs are its Definition's on its own rows and the combines' row is not the served one; and that unit's
    member."""
    got = [g for g, _e in read(s, w.members, UNITS)]
    unit = units(w.members, UNITS, UNITS_PLACES)

    def held(t: str, at: int) -> tuple[int, int]:
        u = unit[(0, 0, UNITS_PLACES[t][at])]
        k = next(k for k, m in enumerate(w.members) if m.base <= u < m.base + m.n)
        return k, u - w.members[k].base

    def put(k: int, i: int, **ports) -> None:
        for q, a in ports.items():
            x = "rows" if q.startswith("in") else "outs"
            got[k] = {**got[k], x: {**got[k][x], q: got[k][x][q].copy()}}
            got[k][x][q][i] = np.asarray(a).reshape(-1)

    def f32(a) -> np.ndarray:
        return np.ascontiguousarray(a).view("<u4")

    (sc, i), (va, j), (en, e), (co, c) = held(UNITS[2], 0), held(UNITS[5], 0), held(UNITS[6], 2), held(UNITS[7], 0)
    assert np.array_equal(got[va]["rows"]["in1"][j], got[sc]["rows"]["in2"][i])
    other = got[sc]["rows"]["in3"][i]
    o = evaluate_batch(SRW.template(UNITS[5]).fn, got[va]["rows"]["in0"][j:j + 1], other.reshape(1, -1)).astype(np.uint32)
    part, _lse = TW.split_end_rows(f32(got[en]["rows"]["in0"][e]), f32(got[en]["rows"]["in1"][e]).reshape(1, 4), o,
                                   scale=HALF["SCALE"])
    y = TW.combine_rows(np.stack([f32(got[co]["rows"][f"in0.{sp}"][c]) for sp in range(2)], axis=1),
                        np.stack([f32(got[co]["rows"]["in1.0"][c]).reshape(1, 32), part], axis=1))
    put(va, j, in1=other, o0=words(o))
    put(en, e, in2=words(o), o0=words(part))
    put(co, c, **{"in1.1": words(part), "o0": y})
    for m, g in zip(w.members, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    assert not np.array_equal(got[va]["rows"]["in1"][j], got[sc]["rows"]["in2"][i])
    served = np.concatenate([g["outs"]["o0"].reshape(-1) for m, g in zip(w.members, got) if m.template == UNITS[7]])
    assert served.tolist() != s.y.tolist()
    return got, va


def test_a_values_unit_on_other_keys_than_its_block_s_scores_unit_is_refused():
    """Every head's scores and values units of a key block read its keys from one committed row a column block, its writer's
    (`chain_rows.wire`): their key ports have no tree of their own.  A values unit on other keys than its block's scores unit
    (`other_keys`) is refused before anything is committed, since its key rows are not its writer's output rows; a prover that
    commits them in a tree of their own is refused at registration, and one whose reference names another row of its writer's
    table when the unit is sampled (the two tests after this one)."""
    s = Served(row(HALF))
    w = window(s, UNITS, UNITS_CHUNK)
    got, va = other_keys(s, w)
    m = w.members[va]
    writer = w.held(m, "in1").rsplit("/", 1)[0]
    with pytest.raises(ValueError, match=re.escape(f"{m.key}: its 'in1' rows are not the 'o0' rows of the {writer} units that "
                                                   "write them")):
        CH.held_rows(w, got)


def test_readers_whose_key_rows_are_trees_of_their_own_are_refused_at_registration():
    """The copies a prover could commit instead of its writers' rows: the window with every scores and values unit's key rows
    a tree of its own and wired to nothing, one values unit on other keys than its block's scores unit (`other_keys`).  It
    commits, and its registration is refused on the ports the verifier derives, which hold no tree of a unit's key rows
    (``R3-ports``); the window's own registers on them."""
    s, design = Served(row(HALF)), D.parse_design("stratified:1")
    w = window(s, UNITS, UNITS_CHUNK, design)
    at, keyed = WI.bases(w.members), {m.key for m in w.members if is_keys(m.template)}
    copies = replace(w, members=tuple(replace(m, wired=tuple(x for x in m.wired if WI.writers(w.members, x[2], at)[0][0].key
                                                              not in keyed), wired_trees=True, tables=()) for m in w.members))
    assert sum(len(m.wired) for m in copies.members) < sum(len(m.wired) for m in w.members)

    def codes(win: CS.Binding, data: list[dict]) -> tuple[str, ...]:
        c = CS.commit(win, data, salt_key=bytes(32))
        rec = CS.registration(c, {"by": "test"}, None, registrar=R)
        return tuple(R.check(rec, w.program, w.query, design, CS.SCHEME, CS.expected_ports(w),
                             leaf_layer=CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(win.members)})).codes)

    assert "R3-ports" in codes(copies, other_keys(s, copies)[0])
    assert "R3-ports" not in codes(w, held_values(w, read(s, w.members, UNITS)))


def test_a_unit_in_the_sample_whose_key_reference_names_another_row_of_its_writer_s_table_is_refused():
    """A scores or values unit's key rows are its reference into its writer member's table (its public body, which the leaf
    layer binds).  Head 0's values unit of split 0's first block, on column block 0, with its reference to the next row of
    the table, another key block's, or with its port wired to the writer of column block 1 of its own key block, a member
    of another template: the wiring check on the sample refuses it when the sample holds that unit, and not when it holds
    only the member's other units."""
    s = Served(row(HALF))
    w = window(s, UNITS, UNITS_CHUNK)
    c = CS.commit(w, held_values(w, read(s, w.members, UNITS)), salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    kp = CH.key_places(HALF)
    writer = {(kp[j].split, kp[j].block, kp[j].col): u for (_p, j), u in key_units(w.members, UNITS, UNITS_PLACES).items()}
    u = units(w.members, UNITS, UNITS_PLACES)[(0, 0, UNITS_PLACES[UNITS[3]][0])]
    m = next(m for m in w.members if m.base <= u < m.base + m.n)
    wm = next(x for x in w.members if x.port("o0") == w.held(m, "in1"))
    i, names = u - m.base, [q for q, _ in m.all_rows]
    assert wm.base + w.held_leaves(m, "in1")[i] == writer[(0, 0, 0)] and not wm.base <= writer[(0, 0, 1)] < wm.base + wm.n

    def ref(r: int) -> dict[str, bytes]:
        at = WI.BC_BYTES * sum(dict(m.tables).get(q, m.n) for q in names) + 4 * (i * len(names) + names.index("in1"))
        body = bytearray(publics[m.key])
        body[at:at + 4] = int(r).to_bytes(4, "little")
        return {**publics, m.key: bytes(body)}

    on_sample = WI.check_drawn
    assert on_sample(w.members, ref(writer[(0, 0, 0)] - wm.base), [u]) == {}
    w1 = next(x for x in w.members if x.base <= writer[(0, 0, 1)] < x.base + x.n)
    rewired = [replace(x, wired=tuple((q, o, tuple(writer[(0, 0, 1)] if k == i else v for k, v in enumerate(ws)) if q == "in1"
                                       else ws) for q, o, ws in x.wired)) if x is m else x for x in w.members]
    for ms, bodies, who in ((w.members, ref((writer[(0, 0, 0)] - wm.base + 1) % wm.n), wm), (rewired, publics, w1)):
        got = on_sample(ms, bodies, [u])
        assert list(got) == [m.key] and got[m.key].endswith(f" in1 strings are not the o0 strings of the {who.key} units that wrote them")
        assert on_sample(ms, bodies, [x for x in range(m.base, m.base + m.n) if x != u]) == {}


def test_a_key_rows_writer_whose_return_is_not_its_input_is_refused_by_its_definition():
    """A key rows' writer returns its input: an output row that is not its input row is not its Definition's output (the
    Boolean body's own refusal is the Definitions' test)."""
    s = Served(row(HALF))
    ms = members(s, UNITS, UNITS_CHUNK)
    got = [(m, g) for m, (g, _e) in zip(ms, read(s, ms, UNITS)) if is_keys(m.template)]
    assert {m.template for m, _g in got} == set(filter(is_keys, UNITS)) and len(got) == 6
    for m, g in got:
        assert np.array_equal(g["rows"]["in0"], g["outs"]["o0"])
        stager(m.template).reference_check(g["rows"], g["outs"])
        with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
            stager(m.template).reference_check(g["rows"], {**g["outs"], "o0": g["outs"]["o0"] ^ np.uint16(1)})
