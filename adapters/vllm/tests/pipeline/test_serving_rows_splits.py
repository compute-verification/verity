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
`sparse_mla_split.split_chain` at ``cols``): per key block a ``SparseMlaHeadScores_v1`` and a ``SparseMlaHeadValues_v1`` per
column block, each split's end on each column block (``SparseMlaSplitEnd_v1``) and the combines of those.  The same checks hold
of that window at DV = 64 in column blocks of 32 (``HALF``): what each unit reads and returns, the wires, the Boolean bodies,
and the commitment.

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

from verity.protocols.spot.verifier.one_stage import draw as D
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
#: TOY at DV = 128: two column blocks of 64
WIDE = {**TOY, "DV": 128}
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


def scores(nvis: int | None = None, S: dict = TOY, boolean: bool = False) -> str:
    """The FIRST block's scores template at ``nvis`` keys, or with None the full block's after it."""
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadScoresV2, **at, **st, DOT=BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadScores, **at, **st, DOT=FA3["DOT"]))


def values(nvis: int | None = None, S: dict = TOY, boolean: bool = False, G: int | None = None) -> str:
    """The values template of `scores`' block on ``G`` columns (half the row's by default)."""
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadValuesV2, **at, BN=S["BN"], G=G or S["DV"] // 2, DOT=BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadValues, **at, BN=S["BN"], G=G or S["DV"] // 2, DOT=FA3["DOT"]))


def split_end(S: dict = TOY, G: int | None = None) -> str:
    """A split's end on a column block of ``G`` columns (half the row's by default)."""
    return descriptor_id(bind(W.SparseMlaSplitEnd, DV=G or S["DV"] // 2, INV=FA3["INV"], SCALE=S["SCALE"]))


F16, B16, F8, E0, C2 = block(16), block(), block(8), cols(0), comb(2)
TMPLS = [F16, B16, F8, E0, C2]
CAPS = {F16: 3, B16: 4, F8: 4, E0: 3, C2: 4}
#: ``TMPLS`` with each key block its Boolean body, what a partition names for a window proved under ``--zk``
BITS = [block(16, boolean=True), block(boolean=True), block(8, boolean=True), E0, C2]
BITS_PER_CHUNK = dict(zip(BITS, CAPS.values()))
#: TOY's places in a head's chain: split 0's blocks 1 and 0 (visit order) and its column end, split 1's block 2 and its column
#: end, then the combine; per block place, its split, its block in the split's visit order and its keys
PLACES = {F16: [0], B16: [1], E0: [2, 4], F8: [3], C2: [5]}
BLOCKS = {0: (0, 0, (16, 32)), 1: (0, 1, (0, 16)), 3: (1, 0, (32, 40))}
#: TOY at DV = 64, where a values unit's O on half the columns is a row of 128 bytes: Flock's compose needs some row of a
#: circuit to take a compression before its last block, and a combine of 16 columns has none
HALF = {**TOY, "DV": 64}
#: ``HALF``'s row with each key block its scores unit and its values units on 32 columns: the scores of split 0's 16-key first
#: block, of the full block after it and of split 1's 8-key first block, each one's values, a split's end on a column block
#: and the two-split combine of one
UNITS = [scores(16, HALF), scores(None, HALF), scores(8, HALF), values(16, HALF), values(None, HALF), values(8, HALF),
         split_end(HALF), comb(2, HALF, G=32)]
UNITS_CHUNK = dict(zip(UNITS, (3, 4, 4, 3, 4, 4, 5, 4)))
#: ``UNITS`` with each scores and values unit its Boolean body
UNITS_BITS = [scores(16, HALF, True), scores(None, HALF, True), scores(8, HALF, True), values(16, HALF, True),
              values(None, HALF, True), values(8, HALF, True), *UNITS[6:]]
UNITS_BITS_CHUNK = dict(zip(UNITS_BITS, UNITS_CHUNK.values()))
#: ``UNITS``' places in a head's chain (`chain_rows.places`), past the six of its whole key blocks: the scores units, then per
#: scores unit its values units on column blocks 0 and 1, each split's ends on them, and the combines
UNITS_PLACES = dict(zip(UNITS, ([6], [7], [8], [9, 10], [11, 12], [13, 14], [15, 16, 17, 18], [19, 20])))


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


def members(s: Served, tmpls=TMPLS, per_chunk=CAPS) -> list[CS.Member]:
    return CH.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, per_chunk), tmpls, [s.rows] * len(tmpls))


def window(s: Served, tmpls=TMPLS, per_chunk=CAPS, design: dict | None = None) -> CS.Binding:
    """The window of ``s``'s rows over ``tmpls`` (`members`) under ``design`` (one unit of each template by default)."""
    ms = members(s, tmpls, per_chunk)
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Binding(*CS.population_program(defs, several=True), design or D.parse_design("stratified:1"), "ab" * 32, tuple(ms))


def read(s: Served, ms, tmpls=TMPLS) -> list[tuple[dict, list]]:
    SRW._slices = s.slices
    return SRW.read_all(ms, [s.rows] * len(tmpls), s, None)


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


def units(ms, tmpls, places: dict[str, list[int]] = PLACES, NH: int = TOY["NH"], rows: int = 1) -> dict[tuple, int]:
    """The global unit of each (row, head, place) in its head's chain: a template's units row by row, then head, then its
    places."""
    out: dict[tuple, int] = {}
    for t in tmpls:
        u = min(m.base for m in ms if m.template == t)
        for p in range(rows):
            for h in range(NH):
                for j in places[t]:
                    out[(p, h, j)], u = u, u + 1
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
    `sparse_mla_split.split_chain` cuts it into at ``cols``: each key block's scores unit and its values units, each split's end
    on a column block and the combines of those, a Definition two widths hold listed once; in a head's places past `plan`'s
    as `chain_rows.places` orders them.  A whole block's scores unit reads EX2 in lookup slots, the rest inline; the Boolean
    bodies are flat."""
    assert [values_widths(dv) for dv in (512, 32, 16)] == [[256, 128, 64, 32, 16], [16], []]
    assert SRW.unit_templates(row())[6:] == (
        (scores(16), 2), (values(16), 4), (scores(8), 2), (values(8), 4), (scores(), 2), (values(), 4), (split_end(), 8),
        (comb(2, G=16), 4))
    assert SRW.unit_templates(row(HALF))[6:14] == tuple(zip([UNITS[j] for j in (0, 3, 2, 5, 1, 4, 6, 7)], (2, 4, 2, 4, 2, 4, 8, 4)))
    served = dict(SRW.unit_templates(SERVED)[12:])
    assert len(served) == 2 + 5 * 3 + 4 and comb(5, GLM) not in served
    assert [served[t] for t in (scores(64, GLM), scores(16, GLM), values(64, GLM, G=64), values(16, GLM, G=64),
                                split_end(GLM, 64), split_end(GLM, 32), comb(5, GLM, G=32))] == [32, 8, 256, 64, 320, 640, 128]
    assert dict(SRW.unit_templates(row(NS=1))[4:]) == {scores(8): 2, values(8): 4, scores(): 4, values(): 8,
                                                       descriptor_id(bind(M.SparseMlaHeadEnd, DV=16, INV=FA3["INV"])): 4}
    assert CH.places(HALF)[:6] == CH.plan(HALF) and len(CH.places(HALF)) == 6 + 3 + (6 + 4 + 2) + (12 + 8 + 4)
    pb, later, state = ("pb", None, "o2"), (("rescale", None, "o3", 7),), (("m", None, "o0"), ("l", None, "o1"))
    assert [(p.key, p.split, p.block, p.keys, p.col, p.reads) for p in CH.places(HALF)[6:21]] == [
        (("scores", 16, True), 0, 0, (16, 32), 0, ()),
        (("scores", 16, False), 0, 1, (0, 16), 0, (("m_prev", None, "o0", 6), ("l_prev", None, "o1", 6))),
        (("scores", 8, True), 1, 0, (32, 40), 0, ()),
        *((("values", 16, True, 32), 0, 0, (16, 32), c, ((*pb, 6),)) for c in range(2)),
        *((("values", 16, False, 32), 0, 1, (0, 16), c, ((*pb, 7), *later, ("o_prev", None, "o0", 9 + c))) for c in range(2)),
        *((("values", 8, True, 32), 1, 0, (32, 40), c, ((*pb, 8),)) for c in range(2)),
        *((("split", 32), 0, 1, (0, 0), c, (*((*x, 7) for x in state), ("o", None, "o0", 11 + c))) for c in range(2)),
        *((("split", 32), 1, 0, (0, 0), c, (*((*x, 8) for x in state), ("o", None, "o0", 13 + c))) for c in range(2)),
        *((("combine", 2, 32), 0, 0, (0, 0), c, tuple((x, sp, o, 15 + 2 * sp + c) for x, o in (("lse", "o1"), ("o", "o0"))
                                                         for sp in range(2))) for c in range(2))]
    assert {t: CH.positions(SRW.template(t), HALF) for t in UNITS} == UNITS_PLACES
    assert [(SRW.template(t).layout, SRW.template(t).word, CR.reads(SRW.template(t)), CH.kind(SRW.template(t))) for t in UNITS] == [
        ("chain", t, r, k) for t, r, k in zip(UNITS, ("slots", "slots", *("inline",) * 6),
                                              ("scores",) * 3 + ("values",) * 3 + ("split", "combine"))]
    assert [(SRW.template(t).word, CR.reads(SRW.template(t))) for t in UNITS_BITS] == [(t, "inline") for t in UNITS]
    assert [descriptor_id(BB.boolean_version(SRW.template(t).fn)) for t in UNITS[:6]] == UNITS_BITS[:6]
    assert CR.unit_rows(SRW.template(UNITS[4])) == (
        [("in0", V2, 16 * 16, ["pb"]), ("in1", V2, 16 * 16 * 32, ["kv"]), ("in2", V2, 32, ["rescale"]), ("in3", V2, 32 * 32, ["o_prev"])],
        [(V2, (0,))])
    for t, key in ((values(16, HALF, G=8), "('values', 16, True, 8)"), (scores(4, HALF), "('scores', 4, True)"),
                   (comb(3, HALF, G=32), "('combine', 3, 32)"), (split_end(HALF, 8), "('split', 8)")):
        with pytest.raises(ValueError, match=re.escape(f"a row of T=40 keys in NS=2 splits holds no unit {key} of it")):
            CH.positions(SRW.template(t), spec_statics(row(HALF)))


def test_each_scores_and_values_unit_is_wired_to_the_units_that_wrote_what_it_reads():
    """A later scores unit reads the scores unit before it; a values unit its own block's scores unit and, past the first,
    the values unit of its column block before it; a split's end the last scores unit and the last values unit of its column
    block; a combine each split's end on its column block."""
    ms = members(Served(row(HALF), n=2), UNITS, UNITS_CHUNK)
    assert sum(m.n for m in ms) == 2 * 2 * 15
    unit = units(ms, UNITS, UNITS_PLACES, rows=2)
    at = {u: key for key, u in unit.items()}
    pl = CH.places(HALF)
    for m in ms:
        keys = [at[m.base + i] for i in range(m.n)]
        g = {nm: i for i, (nm, _ty) in enumerate(SRW.template(m.template).fn.signature()[0])}
        reads = pl[keys[0][2]].reads
        assert m.wired == tuple((f"in{g[x]}" + ("" if sp is None else f".{sp}"), o,
                                 tuple(unit[(p, h, pl[j].reads[i][3])] for p, h, j in keys)) for i, (x, sp, o, _to) in enumerate(reads))
        assert bool(m.wired) == (m.template not in (UNITS[0], UNITS[2]))
    WI.validate(ms)
    s = Served(row(HALF))
    for left, why in ((UNITS[0], "block 0 (visit order)'s scores are the output of a scores template"),
                      (UNITS[5], "block 0 (visit order) of split 1's O on column block 0 is the output of a values template"),
                      (UNITS[6], "split 0's end column block 0 is the output of a column end template")):
        with pytest.raises(ValueError, match=re.escape(f"{why} the partition does not name")):
            members(s, [t for t in UNITS if t != left], UNITS_CHUNK)


def test_the_scores_and_values_units_read_the_twin_s_and_their_combines_are_the_served_row():
    """Every wired row is its writer's output row; a scores unit's ports are its block's keys and the twin's running max, lanes,
    probabilities and rescaling factor, a values unit's its column block of its block's keys and of the twin's O; each unit's
    Definition on its rows (as Flock's stager parses them) computes its committed outputs, and the combines' outputs, column
    block by column block, are the served row."""
    s = Served(row(HALF), n=2)
    ms = members(s, UNITS, UNITS_CHUNK)
    got = read(s, ms, UNITS)
    ws = committed(ms, got)
    S, pl = HALF, CH.places(HALF)
    unit = units(ms, UNITS, UNITS_PLACES, rows=2)
    at = {u: key for key, u in unit.items()}
    tw = [TW.split_chain_rows(s.ops[3 * p].reshape(S["NH"], -1), s.ops[3 * p + 1].reshape(S["NH"], -1),
                              s.ops[3 * p + 2].reshape(S["T"], -1), T=S["T"], NS=S["NS"], DR_=S["DR"], DV=S["DV"], BN=S["BN"],
                              scale=S["SCALE"], units=True)[0] for p in range(2)]
    for m, (g, _entries) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
        for i in range(m.n):
            p, h, j = at[m.base + i]
            sp, k, (lo, hi), c = pl[j].split, pl[j].block, pl[j].keys, pl[j].col
            kv = s.ops[3 * p + 2].reshape(S["T"], -1)[lo:hi]
            if m.template in UNITS[:3]:
                mm, ll, _o, pb, rescale = tw[p][sp][k]
                qn, qr = s.ops[3 * p].reshape(S["NH"], -1)[h], s.ops[3 * p + 1].reshape(S["NH"], -1)[h]
                assert [g["rows"][x][i].tolist() for x in ("in0", "in1", "in2")] == [qn.tolist(), qr.tolist(), kv.reshape(-1).tolist()]
                assert [g["outs"][x][i].tolist() for x in ("o0", "o1")] == [words(mm[h]), words(ll[h])]
                assert g["outs"]["o2"][i].tolist() == np.asarray(pb[h], dtype=np.uint16).tolist()
                assert (rescale is None) == ("o3" not in g["outs"]) and (rescale is None or g["outs"]["o3"][i].tolist() == words(rescale[h]))
            elif m.template in UNITS[3:6]:
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
    assert [(of[m.template], m.rows, m.outs, m.n, m.base, m.wired) for m in ms] == [
        (m.template, m.rows, m.outs, m.n, m.base, m.wired) for m in bits]
    for (g, e), (gb, eb) in zip(read(s, ms, UNITS), read(s, bits, UNITS_BITS), strict=True):
        assert e == eb and all(np.array_equal(g[x][q], gb[x][q]) for x in ("rows", "outs") for q in g[x] | gb[x])


def test_a_combine_of_64_columns_reads_each_split_s_end_from_the_unit_the_partition_names():
    """At DV = 128 a combine of 64 columns is one template whether its splits end in column blocks of whole key blocks or on
    column blocks of values units: it reads each split's end from the one the partition names; a partition that names
    neither is refused, and so is one that names the row's key blocks both ways, or as values units of two widths."""
    s = Served(row(WIDE), n=1)
    ends, c2 = [cols(0, WIDE), cols(1, WIDE)], comb(2, WIDE)
    whole = [block(16, WIDE), block(None, WIDE), block(8, WIDE), *ends, c2]
    tmpls = [scores(16, WIDE), scores(None, WIDE), scores(8, WIDE), values(16, WIDE), values(None, WIDE), values(8, WIDE),
             split_end(WIDE), c2]
    assert CH.positions(SRW.template(c2), WIDE) == [7, 8] and CH.plan(WIDE)[7].reads[0] == ("lse", 0, "o1", 2)
    assert [CH.places(WIDE)[j].reads for j in (7, 8)] == [
        tuple((x, sp, o, ((2, 5)[sp] + c, 18 + 2 * sp + c)) for x, o in (("lse", "o1"), ("o", "o0")) for sp in range(2))
        for c in range(2)]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 64))
    places = dict(zip(tmpls, ([9], [10], [11], [12, 13], [14, 15], [16, 17], [18, 19, 20, 21], [7, 8])))
    assert {t: CH.positions(SRW.template(t), WIDE) for t in tmpls} == places
    unit = units(ms, tmpls, places)
    (m,) = [m for m in ms if m.template == c2]
    assert m.wired == tuple((f"in{g}.{sp}", o, tuple(unit[(0, h, 18 + 2 * sp + c)] for h in range(2) for c in range(2)))
                            for g, o in ((0, "o1"), (1, "o0")) for sp in range(2))
    assert committed(ms, read(s, ms, tmpls))[m.port("o0")].reshape(-1).tolist() == s.y.tolist()
    chunk = dict.fromkeys([*whole, *tmpls, values(16, WIDE, G=32)], 64)
    for both, ways in (([*whole[:5], *tmpls], f"whole ({whole[0]}) and as scores and values units ({tmpls[0]})"),
                       ([*tmpls[3:], *ends], f"whole ({ends[0]}) and as values units of 64 columns ({tmpls[3]})"),
                       ([*tmpls[:4], values(16, WIDE, G=32)],
                        f"as values units of 32 columns ({values(16, WIDE, G=32)}) and as values units of 64 columns ({tmpls[3]})")):
        with pytest.raises(ValueError, match=re.escape(f"row 5: the partition names its key blocks {ways}; a row's key blocks "
                                                       f"are served one way")):
            members(s, both, chunk)
    with pytest.raises(ValueError, match=re.escape("split 0's end column block 0 is the output of a column end template the "
                                                   "partition does not name")):
        members(s, [*tmpls[:6], c2], dict.fromkeys(tmpls, 64))


def test_a_row_named_at_every_width_of_its_values_units_is_refused_by_its_two_narrowest():
    """A row of DV = 512 has values units of five widths (`serving_templates.values_widths`); a partition that names its key
    blocks at all five is refused as one that names two is, and the refusal names the two narrowest."""
    S = {**TOY, "DV": 512}
    widths = values_widths(512)
    assert widths == [256, 128, 64, 32, 16]
    tmpls = [scores(16, S), *(values(16, S, G=g) for g in widths)]
    with pytest.raises(ValueError, match=re.escape(
            f"row 5: the partition names its key blocks as values units of 16 columns ({values(16, S, G=16)}) and as values "
            f"units of 32 columns ({values(16, S, G=32)}); a row's key blocks are served one way")):
        members(Served(row(S), n=1), tmpls, dict.fromkeys(tmpls, 64))


def test_a_row_at_ns_1_in_scores_and_values_units_ends_on_each_column_block():
    """Prefill's split count: the chain's end on each column block reads the last scores unit's lanes and the last values
    unit of its column block's O, and the ends' outputs, column block by column block, are the served row."""
    s = Served(row(HALF, NS=1), n=1)
    e = descriptor_id(bind(M.SparseMlaHeadEnd, DV=32, INV=FA3["INV"]))
    tmpls = [scores(8, HALF), scores(None, HALF), values(8, HALF), values(None, HALF), e]
    assert [CH.kind(SRW.template(t)) for t in tmpls] == ["scores", "scores", "values", "values", "end"]
    assert {t: CH.positions(SRW.template(t), {**HALF, "NS": 1}) for t in tmpls} == dict(zip(tmpls, (
        [4], [5, 6], [7, 8], [9, 10, 11, 12], [13, 14])))
    assert [CH.places({**HALF, "NS": 1})[j].reads for j in (13, 14)] == [(("l", None, "o1", 6), ("o", None, "o0", 11 + c)) for c in range(2)]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 16))
    got = read(s, ms, tmpls)
    for m, (g, _e) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    assert committed(ms, got)[ms[-1].port("o0")].reshape(-1).tolist() == s.y.tolist()


def test_a_row_at_the_served_statics_reads_and_wires_its_744_units_in_columns_of_64():
    """One row of step 16's statics with each key block its scores unit and eight values units of 64 columns: 40 scores units,
    320 values units, 320 split ends and the 64 combines its whole key blocks' chain holds too, each read row wired to its
    writer and every wired row its writer's output, the combines' outputs the served row."""
    s = Served(SERVED, n=1, seed=16)
    tmpls = [scores(64, GLM), scores(16, GLM), values(64, GLM, G=64), values(16, GLM, G=64), split_end(GLM, 64), comb(5, GLM)]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 512))
    assert [(m.template, m.n) for m in ms] == list(zip(tmpls, (32, 8, 256, 64, 320, 64)))
    assert committed(ms, read(s, ms, tmpls))[ms[-1].port("o0")].reshape(-1).tolist() == s.y.tolist()


#: what a partition names to serve step 16's rows in scores and values units of 64 columns: each key block's scores unit and
#: values units their Boolean bodies, a split's end on a column block and the combine served beside whole key blocks too
SERVED_UNITS = [scores(64, GLM, True), scores(16, GLM, True), values(64, GLM, True, 64), values(16, GLM, True, 64),
                split_end(GLM, 64), comb(5, GLM)]


def test_every_served_scores_and_values_unit_composes_and_verify_zk_takes_it():
    """At GLM-5.3's statics each unit of a row served in scores and values units of 64 columns is a proof block Flock's
    composer takes (`interface.check_proof_block`: each has a row past 111 bytes, and its rows fit 2^26 bits at most), and
    `verify --zk` takes it (`chunk_rows.zk_refusal`): a scores or values unit's Boolean body reads no table, and a split's end
    and the combine read theirs inline.  Every returned field, each one a wire may read, is whole 16-bit words."""
    floors = []
    for t in SERVED_UNITS:
        tp = SRW.template(t)
        assert CR.zk_refusal(tp) is None and CR.statement(tp)["reads"] == "inline", t
        assert CR.slot_tables(tp.fn) == (() if tp.id != tp.word else CR.slot_tables(tp.fn)), t
        ret = tp.fn.signature()[1]
        assert all(sum(f.leaf_widths()) % 16 == 0 for f in (field_types(ret) if isinstance(ret, Tuple) else (ret,))), t
        floors.append(IF.check_proof_block(tp.fn))
    assert floors == [26, 24, 23, 22, 22, 22]


def test_a_window_of_scores_and_values_units_commits_and_its_wires_hold():
    s = Served(row(HALF))
    w = window(s, UNITS, UNITS_CHUNK)
    got = read(s, w.members, UNITS)
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    j = next(j for j, m in enumerate(w.members) if m.template == UNITS[4])
    m = w.members[j]
    (p, o, ws), *rest = m.wired
    assert (p, o) == ("in0", "o2")
    rewired = [replace(x, wired=((p, o, ws[1:] + ws[:1]), *rest)) if i == j else x for i, x in enumerate(w.members)]
    assert list(WI.check(rewired, publics)) == [m.key]                              # another block's probabilities
    bad = [dict(g, rows={**g["rows"], "in3": g["rows"]["in3"] ^ np.uint16(1)}) if i == j else g for i, (g, _e) in enumerate(got)]
    with pytest.raises(ValueError, match=re.escape(f"{m.key}: a in3 row is not the o0 row")):
        CS.commit(w, bad, salt_key=bytes(32))


def test_a_values_unit_on_other_keys_than_its_block_s_scores_unit_passes_every_check():
    """A known gap, stated so that it is not a silence.  A key block's scores unit and each of its values units commit their own
    row of the block's keys; no unit writes those rows and nothing registered holds them, so nothing holds two copies equal.
    Here head 0's values unit of split 1's first block commits, as its column block of the keys, the keys' other 32 columns, and
    its O, its split's end and the combine are recomputed from them.  The window commits, the wiring check passes on the public
    bodies and every unit's outputs are its Definition's on its own rows, and the combines' row is not the served one.  Once a
    unit writes the key rows and every key port is wired to it, this window is refused, and this test asserts the refusal."""
    s = Served(row(HALF))
    w = window(s, UNITS, UNITS_CHUNK)
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
    keys = got[sc]["rows"]["in2"][i].reshape(8, -1)
    assert np.array_equal(got[va]["rows"]["in1"][j].reshape(8, 32), keys[:, :32])
    other = keys[:, 32:64]
    o = evaluate_batch(SRW.template(UNITS[5]).fn, got[va]["rows"]["in0"][j:j + 1], other.reshape(1, -1)).astype(np.uint32)
    part, _lse = TW.split_end_rows(f32(got[en]["rows"]["in0"][e]), f32(got[en]["rows"]["in1"][e]).reshape(1, 4), o,
                                   scale=HALF["SCALE"])
    y = TW.combine_rows(np.stack([f32(got[co]["rows"][f"in0.{sp}"][c]) for sp in range(2)], axis=1),
                        np.stack([f32(got[co]["rows"]["in1.0"][c]).reshape(1, 32), part], axis=1))
    put(va, j, in1=other, o0=words(o))
    put(en, e, in2=words(o), o0=words(part))
    put(co, c, **{"in1.1": words(part), "o0": y})
    sealed = CS.commit(w, got, salt_key=bytes(32))
    assert WI.check(w.members, {m.key: sealed.public_body(k) for k, m in enumerate(w.members)}) == {}
    for m, g in zip(w.members, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    assert not np.array_equal(got[va]["rows"]["in1"][j].reshape(8, 32), got[sc]["rows"]["in2"][i].reshape(8, -1)[:, :32])
    served = np.concatenate([g["outs"]["o0"].reshape(-1) for m, g in zip(w.members, got) if m.template == UNITS[7]])
    assert served.tolist() != s.y.tolist()
