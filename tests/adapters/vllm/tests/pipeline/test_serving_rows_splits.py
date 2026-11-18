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
from verity.commitments.periods import interface as IF
from verity.commitments.periods import rows as CS
from verity.commitments.periods import wiring as WI
from verity_vllm.pipeline import chain_rows as CH
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline.serving_templates import values_widths
from verity.circuits.ops.boolean import versions as BB
from verity_vllm.definitions.numerics import sparse_mla_split_twins as TW
from verity_vllm.definitions.registry import boolean_sparse_mla as BM
from verity_vllm.definitions.registry import boolean_sparse_mla_split as BX
from verity_vllm.definitions.registry import hopper
from verity_vllm.definitions.registry import sparse_mla as M
from verity_vllm.definitions.registry import sparse_mla_split as W
from verity_vllm.query.program_view import spec_statics

from verity.protocols.proofs.spot.verifier.one_stage import docket as R
from verity.commitments import randomness as RND
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
GLM = {"T": 272, "NS": 5, "NH": 8, "DR": 64, "DV": 512, "BN": 64, "SCALE": 0.0625}
#: step 16's sparse-MLA row (`glm53_rows.sparse_mla_split`), 78 of them at TP 8
SERVED = ('SparseMlaSplit_v1{T=272,NS=5,NH=8,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},INV={"fn":"Fa3InvSum_v1"},'
          'SCALE={"f64":"0x1.0000000000000p-4"}}')


#: `chain_rows.pieces`' refusal of a port reading several templates' trees
NAME = "a wired port reads one template's tree"

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
#: TOY's places in a head's chain: split 0's blocks 1 and 0 (visit order) and its column end, split 1's block 2 and its column
#: end, then the combine; per block place, its split, its block in the split's visit order and its keys
PLACES = {F16: [0], B16: [1], E0: [2, 4], F8: [3], C2: [5]}
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
    return CS.Binding(*IF.listed_program(defs, ()), design or RND.parse_design("stratified:1"), "ab" * 32, tuple(ms))


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
    """Every port's words of ``ms`` as read (``got``), each wired row held to its writer's output row (`chain_rows.check_held`)."""
    ws = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    CH.check_held(ms, [g for g, _e in got])
    return ws


def is_keys(t: str) -> bool:
    return CH.kind(SRW.template(t)) == "keys"


def units(ms, tmpls, places: dict[str, list[int]] = PLACES, NH: int = int(TOY["NH"]), rows: int = 1) -> dict[tuple, int]:
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
    s, design = Served(row(HALF)), RND.parse_design("stratified:1")
    w = window(s, UNITS, UNITS_CHUNK, design)
    at, keyed = WI.bases(w.members), {m.key for m in w.members if is_keys(m.template)}
    copies = replace(w, members=tuple(replace(m, wired=tuple(x for x in m.wired if WI.writers(w.members, x[2], at)[0][0].key
                                                              not in keyed), wired_trees=True, tables=()) for m in w.members))
    assert sum(len(m.wired) for m in copies.members) < sum(len(m.wired) for m in w.members)

    def codes(win: CS.Binding, data: list[dict]) -> tuple[str, ...]:
        c = CS.commit(win, data)
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
    c = CS.commit(w, held_values(w, read(s, w.members, UNITS)))
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


def test_a_split_end_reading_two_block_templates_is_refused_by_name():
    """A split's end reads the running state of blocks of two templates (a full split's and the last split's): one port
    reading two templates' trees, which `chain_rows.pieces` refuses by name."""
    with pytest.raises(ValueError, match=NAME):
        members(Served(n=2))
