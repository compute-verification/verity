"""A sparse-MLA row cut into FA3's split units (`pipeline.serving_rows`' ``split`` layout, `split_rows`): a ``SparseMlaSplit_v1``
row in NS > 1 splits holds a ``SparseMlaSplitPart_v1`` template at each split's key count and one ``SparseMlaCombine_v1`` over its
splits (at step 16's statics, per head, four 64-key parts, one 16-key part and a five-split combine), and a window of the three
at a small static (T = 40 keys in splits of 16, 16 and 8, two heads):

* reads each part's operands off the row and its partial and lse off the twin, whose combine is the row's served words, so each
  unit's Definition on its rows, parsed as Flock's stager parses them, is its committed output;
* wires every combine slot to the output row of the part unit of its row, head and split, commits it under that row's salt, so
  the verifier's check on the public bodies holds, and refuses a partial that is not its writer's, a partial in another combine
  slot, a row whose served words are not its splits' combine, and a partition that leaves out a part's template.

`test_a_window_of_one_row_commits_and_its_wires_hold` takes about two minutes, all of it the window's population program
lowering the three templates: it is the one test of the committed window and its public bodies, which the others reach only
through the rows.  `benchmarks/one_stage/a5_sparse_mla.py` audits the same window (`window`, `Served`).
"""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.ml.windows import rows as CS
from verity.ml.windows import wiring as WI
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import split_rows as SP
from verity.ml.definitions.numerics import sparse_mla_split_twins as TW
from verity.ml.definitions.registry import boolean_sparse_mla_split as BX
from verity.ml.definitions.registry import hopper
from verity.ml.definitions.registry import sparse_mla_split as W
from verity_vllm.query.manifest import format as MF
from verity_vllm.query.program_view import spec_statics

from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
BOOL = {"DOT": BX.SPARSE_MLA_GLM53["DOT"], "INV": BX.SPARSE_MLA_GLM53["INV"]}
#: DV = 32 is the least at which a combine's partial row (4 DV bytes) takes a compression before its last block, which Flock's
#: compose needs some row of a circuit to have (the pin's sha512x3 slot)
TOY = {"T": 40, "NS": 3, "NH": 2, "DR": 16, "DV": 32, "BN": 16, "SCALE": 0.0625}
GLM = {"T": 272, "NS": 5, "NH": 8, "DR": 64, "DV": 512, "BN": 64, "SCALE": 0.0625}
#: step 16's sparse-MLA row (`glm53_rows.sparse_mla_split`), 78 of them at TP 8
SERVED = ('SparseMlaSplit_v1{T=272,NS=5,NH=8,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},INV={"fn":"Fa3InvSum_v1"},'
          'SCALE={"f64":"0x1.0000000000000p-4"}}')


def row(S: dict = TOY, **over) -> str:
    return descriptor_id(bind(W.SparseMlaSplit, **{**S, **over}, **FA3))


def part(k: int, S: dict = TOY, boolean: bool = True) -> str:
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    return descriptor_id(bind(BX.SparseMlaSplitPartV2, T=k, **st, **BOOL) if boolean else bind(W.SparseMlaSplitPart, T=k, **st, **FA3))


def comb(n: int, S: dict = TOY, boolean: bool = True) -> str:
    return descriptor_id(bind(BX.SparseMlaCombineV2, NSPLIT=n, DV=S["DV"], INV=BOOL["INV"]) if boolean
                         else bind(W.SparseMlaCombine, NSPLIT=n, DV=S["DV"], INV=FA3["INV"]))


#: the row's own unit templates (`SRW.unit_templates`), as step 16's proofs of record are: their EX2 reads the pinned table, which
#: a whole-block part reads in lookup slots, where a ``_v2`` body's EX2 is gates (`mufu.table_word`) and lowers flat either way
P16, P8, C3 = part(16, boolean=False), part(8, boolean=False), comb(3, boolean=False)
TMPLS = [P16, P8, C3]
CAPS = {P16: 3, P8: 4, C3: 4}


class Served:
    """``n`` rows of ``spec`` (row p's q_nope, q_rope and kv are arguments 3p, 3p + 1 and 3p + 2) and the store of their served
    outputs, the twin's whole row (`sparse_mla_split_twins.head_split_rows`, the word ``SparseMlaHeadSplit_v1`` per head)."""

    def __init__(self, spec: str = row(), n: int = 1, seed: int = 3):
        S = spec_statics(spec)
        T, NS, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NS", "NH", "DR", "DV", "BN"))
        rng = np.random.default_rng(seed)

        def bf16(shape, s=1.0):
            return ((rng.standard_normal(shape).astype(np.float32) * np.float32(s)).view(np.uint32) >> 16).astype(np.uint16)

        self.ops, self.rows, ys = {}, [], []
        for p in range(n):
            qn, qr, kv = bf16((NH, DV)), bf16((NH, DR)), bf16((T, DV + DR), 0.5)
            y, nf = TW.head_split_rows(qn, qr, kv, T=T, NS=NS, DR_=DR, DV=DV, BN=BN, scale=float(S["SCALE"]))
            assert not nf.any()
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


def members(s: Served, tmpls=TMPLS, caps=CAPS) -> list[CS.Member]:
    return SP.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, caps), tmpls, [s.rows] * len(tmpls))


def window(s: Served, tmpls=TMPLS, caps=CAPS, law: dict | None = None) -> CS.Window:
    """The window of ``s``'s rows over ``tmpls`` (`members`) under ``law`` (one unit of each template by default)."""
    ms = members(s, tmpls, caps)
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Window(*CS.population_program(defs, several=True), law or D.parse_law("stratified:1"), "ab" * 32, tuple(ms))


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


def units(ms, tmpls, S: dict = TOY, rows: int = 1) -> dict[tuple, int]:
    """The global unit of each (row, head, split) part and (row, head) combine: a template's units row by row, then head, then
    its splits."""
    keys = [min(S["T"], b1 * S["BN"]) - b0 * S["BN"] for b0, b1 in W.split_blocks(S["T"], S["NS"], S["BN"])]
    out: dict[tuple, int] = {}
    for t in tmpls:
        u = min(m.base for m in ms if m.template == t)
        k = spec_statics(SRW.template(t).word).get("T")
        for p in range(rows):
            for h in range(S["NH"]):
                for s in ([None] if k is None else [s for s, x in enumerate(keys) if x == int(k)]):
                    out[(p, h) if s is None else (p, h, s)], u = u, u + 1
    return out


def test_a_split_row_holds_a_part_at_each_split_s_keys_and_a_combine():
    assert SRW.unit_templates(row()) == ((row(), 1), (part(16, boolean=False), 4), (part(8, boolean=False), 2),
                                         (comb(3, boolean=False), 2))
    assert row(GLM) == SERVED
    assert SRW.unit_templates(SERVED) == ((SERVED, 1), (part(64, GLM, False), 32), (part(16, GLM, False), 8), (comb(5, GLM, False), 8))
    assert SRW.unit_templates(row(NS=1)) == ((row(NS=1), 1),)
    assert [(SRW.template(t).layout, SRW.template(t).word, CR.reads(SRW.template(t))) for t in TMPLS] == [
        ("split", P16, "slots"), ("split", P8, "inline"), ("split", C3, "inline")]
    assert [CR.reads(SRW.template(t)) for t in (part(64, GLM, False), part(16, GLM, False), comb(5, GLM, False))] == [
        "slots", "inline", "inline"]
    assert [SRW.template(t).word for t in (part(16), comb(3))] == [P16, C3]
    DV, DR = TOY["DV"], TOY["DR"]
    assert CR.unit_rows(SRW.template(P16)) == ([("in0", V2, 16 * DV, ["q_nope"]), ("in1", V2, 16 * DR, ["q_rope"]),
                                                ("in2", V2, 16 * 16 * (DV + DR), ["kv"])], [(V2, (0,)), (V2, (1,))])
    assert CR.unit_rows(SRW.template(C3)) == ([(f"in0.{s}", V2, 32, [f"lse[{s}]"]) for s in range(3)]
                                              + [(f"in1.{s}", V2, 32 * DV, [f"o[{s}]"]) for s in range(3)], [(V2, (0,))])
    with pytest.raises(ValueError, match="has 3 non-empty ones"):
        SP.positions(SRW.template(comb(2)), spec_statics(row()))


def test_each_combine_slot_is_wired_to_the_part_unit_of_its_row_head_and_split():
    ms = members(Served(n=2))
    assert [(m.template, m.n) for m in ms] == [(P16, 3), (P16, 3), (P16, 2), (P8, 4), (C3, 4)]
    unit = units(ms, TMPLS, rows=2)
    at = {u: key for key, u in unit.items()}
    for m in ms:
        if m.template != C3:
            assert m.wired == ()
            continue
        assert m.wired == tuple((f"in{g}.{s}", o, tuple(unit[(*at[m.base + i], s)] for i in range(m.n)))
                                for g, o in ((0, "o1"), (1, "o0")) for s in range(3))
    WI.validate(ms)


def test_the_parts_read_the_twin_s_partials_and_lse_whose_combine_is_the_served_row():
    s = Served(n=2)
    ms = members(s)
    got = read(s, ms)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    ys = {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs}
    WI.salts(ms, words, ys)                     # every wired row is its writer's output row
    unit, S = units(ms, TMPLS, rows=2), TOY
    at = {u: key for key, u in unit.items()}
    keys = np.cumsum([0, 16, 16, 8])
    for m, (g, _entries) in zip(ms, got):
        for i in range(m.n):
            key = at[m.base + i]
            p, h = key[:2]
            qn, qr = s.ops[3 * p].reshape(S["NH"], -1)[h], s.ops[3 * p + 1].reshape(S["NH"], -1)[h]
            if len(key) == 2:
                assert g["outs"]["o0"][i].tolist() == s.y.reshape(2, S["NH"], -1)[p, h].tolist()
                continue
            kv = s.ops[3 * p + 2].reshape(S["T"], -1)[keys[key[2]]:keys[key[2] + 1]]
            o, lse, nf = TW.part_rows(qn[None], qr[None], kv, T=len(kv), DR_=S["DR"], DV=S["DV"], BN=S["BN"], scale=S["SCALE"])
            assert not nf.any()
            assert [g["rows"][x][i].tolist() for x in ("in0", "in1", "in2")] == [qn.tolist(), qr.tolist(), kv.reshape(-1).tolist()]
            assert (g["outs"]["o0"][i].tolist(), g["outs"]["o1"][i].tolist()) == (o[0].view("<u2").tolist(), lse.view("<u2").tolist())


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
    with pytest.raises(ValueError, match="the twin's combine of its 3 splits is not the row's served out words"):
        read(s, members(s))


def test_a_partition_without_a_part_s_template_is_refused():
    s = Served()
    for tmpls, keys in (([P16, C3], [8]), ([P8, C3], [16])):
        with pytest.raises(ValueError, match=re.escape(f"splits of {keys} keys are the outputs of part templates the partition does not name")):
            members(s, tmpls)
    assert all(m.wired == () for m in members(s, [P16, P8]))


def test_a_row_at_the_served_statics_reads_and_wires_its_48_units():
    """One row of step 16's statics, its 48 units (32 64-key parts, 8 16-key parts, 8 five-split combines), each combine slot
    wired to its part and every wired row its writer's output; the window itself (its population program at these statics
    lowers the 64-key part) is a5's."""
    s = Served(SERVED, n=1, seed=16)
    tmpls = [part(64, GLM, False), part(16, GLM, False), comb(5, GLM, False)]
    ms = members(s, tmpls, {tmpls[0]: 256, tmpls[1]: 256, tmpls[2]: 256})
    assert [(m.template, m.n) for m in ms] == [(tmpls[0], 32), (tmpls[1], 8), (tmpls[2], 8)]
    unit = units(ms, tmpls, GLM)
    assert ms[2].wired == tuple((f"in{g}.{sp}", o, tuple(unit[(0, h, sp)] for h in range(8))) for g, o in ((0, "o1"), (1, "o0"))
                                for sp in range(5))
    got = read(s, ms, tmpls)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    assert words[ms[2].port("o0")].reshape(-1).tolist() == s.y.tolist()


@pytest.mark.slow
def test_a_window_of_one_row_commits_and_its_wires_hold():
    s = Served()
    w = window(s)
    got = read(s, w.members)
    for m, (g, _e) in zip(w.members, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    k = next(k for k, m in enumerate(w.members) if m.template == P16)
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):     # a partial its part does not compute
        stager(P16).reference_check(got[k][0]["rows"], {**got[k][0]["outs"], "o0": got[k][0]["outs"]["o0"] ^ np.uint16(1)})
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    j = next(j for j, m in enumerate(w.members) if m.wired)
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
