"""A sparse-MLA row cut into its heads' key-block chains (`pipeline.serving_rows`' ``chain`` layout, `chain_rows`): a
``SparseMla_v1`` row holds a ``SparseMlaHeadBlock_v1`` FIRST template at its first visited block's key count, the full-block
template for every later block and one ``SparseMlaHeadEnd_v1`` a head (at GLM-5.3's statics and T = 130, per head, a 2-key first
block, two 64-key blocks and the end), and a window of the three at a small static (T = 40 keys in blocks of 8, 16 and 16, two
heads):

* reads each block's operands off the row and its state off the twin, whose end is the row's served words, so each unit's
  Definition on its rows, parsed as Flock's stager parses them, is its committed output;
* wires every later block's state and every end's lanes and O to the output rows of the unit of the block before it, commits it
  under that row's salt, so the verifier's check on the public bodies holds, and refuses a state that is not its writer's, a
  row whose served words are not its chain's end, and a partition that leaves out a block's template.

`test_a_window_of_one_row_commits_and_its_wires_hold` takes minutes, all of it the window's population program lowering the
three templates: it is the one test of the committed window and its public bodies, which the others reach only through the rows.
"""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.ml.windows import rows as CS
from verity.ml.windows import wiring as WI
from verity_vllm.pipeline import chain_rows as CH
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions.numerics import sparse_mla_split_twins as TW
from verity.ml.definitions.registry import boolean_sparse_mla as BM
from verity.ml.definitions.registry import hopper
from verity.ml.definitions.registry import sparse_mla as M
from verity_vllm.query.manifest import format as MF
from verity_vllm.query.program_view import spec_statics

from verity.core.protocols.spot.one_stage import draw as D
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
BOOL = {"DOT": BM.SPARSE_MLA_GLM53["DOT"], "INV": BM.SPARSE_MLA_GLM53["INV"]}
TOY = {"T": 40, "NH": 2, "DR": 16, "DV": 32, "BN": 16, "SCALE": 0.0625}
GLM = {"T": 130, "NH": 8, "DR": 64, "DV": 512, "BN": 64, "SCALE": 0.0625}
SERVED = ('SparseMla_v1{T=130,NH=8,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},INV={"fn":"Fa3InvSum_v1"},'
          'SCALE={"f64":"0x1.0000000000000p-4"}}')


def row(S: dict = TOY, **over) -> str:
    return descriptor_id(bind(M.SparseMla, **{**S, **over}, **FA3))


def block(nvis: int | None = None, S: dict = TOY, boolean: bool = False) -> str:
    """The FIRST block template at ``nvis`` keys, or with None the full block after it."""
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadBlockV2, **at, **st, DOT=BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadBlock, **at, **st, DOT=FA3["DOT"]))


def end(S: dict = TOY, boolean: bool = False) -> str:
    return descriptor_id(bind(BM.SparseMlaHeadEndV2, DV=S["DV"], INV=BOOL["INV"]) if boolean
                         else bind(M.SparseMlaHeadEnd, DV=S["DV"], INV=FA3["INV"]))


F8, B16, E = block(8), block(), end()
TMPLS = [F8, B16, E]
CAPS = {F8: 2, B16: 3, E: 4}


class Served:
    """``n`` rows of ``spec`` (row p's q_nope, q_rope and kv are arguments 3p, 3p + 1 and 3p + 2) and the store of their served
    outputs, the twin's end of its chain (`sparse_mla_split_twins.chain_rows`, `head_end_rows`)."""

    def __init__(self, spec: str = row(), n: int = 1, seed: int = 3):
        S = spec_statics(spec)
        T, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NH", "DR", "DV", "BN"))
        rng = np.random.default_rng(seed)

        def bf16(shape, s=1.0):
            return ((rng.standard_normal(shape).astype(np.float32) * np.float32(s)).view(np.uint32) >> 16).astype(np.uint16)

        self.ops, self.rows, ys, self.states = {}, [], [], []
        for p in range(n):
            qn, qr, kv = bf16((NH, DV)), bf16((NH, DR)), bf16((T, DV + DR), 0.5)
            states, nf = TW.chain_rows(qn, qr, kv, T=T, DR_=DR, DV=DV, BN=BN, scale=float(S["SCALE"]))
            assert not nf.any()
            self.states.append(states)
            ys.append(np.asarray(TW.head_end_rows(states[-1][1], states[-1][2]), dtype=np.uint16).reshape(-1))
            self.ops.update({3 * p: qn.reshape(-1), 3 * p + 1: qr.reshape(-1), 3 * p + 2: kv.reshape(-1)})
            r = {"i": 5 + p, "spec": spec, "args": [[["x", 3 * p + g, 0, self.ops[3 * p + g].size]] for g in range(3)]}
            d = {"statics": S, "op_path": "model.layers.3.self_attn.attn", "members": [("out", NH * DV, "bf16")], "row": p, "step": 4}
            self.rows.append(("r0", None, r, d))
        self.y = np.concatenate(ys)

    def slices(self, _pi, _store, _weights, _rid, _arrive, g):
        """`serving_rows._slices`: an argument group's elements."""
        return [self.ops[g[0][1]]]

    def words(self, rid, step, op, member, lo, hi):
        return self.y[lo:hi], None


def members(s: Served, tmpls=TMPLS, caps=CAPS) -> list[CS.Member]:
    return CH.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, caps), tmpls, [s.rows] * len(tmpls))


def window(s: Served, tmpls=TMPLS, caps=CAPS, law: dict | None = None) -> CS.Binding:
    """The window of ``s``'s rows over ``tmpls`` (`members`) under ``law`` (one unit of each template by default)."""
    ms = members(s, tmpls, caps)
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Binding(*CS.population_program(defs, several=True), law or D.parse_design("stratified:1"), "ab" * 32, tuple(ms))


def read(s: Served, ms, tmpls=TMPLS) -> list[tuple[dict, list]]:
    SRW._slices = s.slices
    return SRW.read_all(ms, [s.rows] * len(tmpls), s, None)


@pytest.fixture(autouse=True)
def _slices(monkeypatch):
    """`serving_rows._slices` back after each test, which `read` replaces."""
    monkeypatch.setattr(SRW, "_slices", SRW._slices)


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "chain-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def units(ms, tmpls, S: dict = TOY, rows: int = 1) -> dict[tuple, int]:
    """The global unit of each (row, head, place) in its head's chain, place j the block j in visit order, nb the end: a
    template's units row by row, then head, then its places."""
    nb = -(-S["T"] // S["BN"])
    out: dict[tuple, int] = {}
    for t in tmpls:
        u = min(m.base for m in ms if m.template == t)
        k = CH.kind(SRW.template(t))
        for p in range(rows):
            for h in range(S["NH"]):
                for j in ([0] if k == "first" else range(1, nb) if k == "block" else [nb]):
                    out[(p, h, j)], u = u, u + 1
    return out


def test_a_chain_row_holds_its_first_block_its_later_blocks_and_its_end():
    """The units `sparse_mla.head_chain` cuts the row into, each a ``chain`` template; a whole block's EX2 reads go in lookup
    slots, a partial first block's and the end's inline."""
    assert SRW.unit_templates(row()) == ((row(), 1), (F8, 2), (B16, 4), (E, 2))
    assert row(GLM) == SERVED
    assert SRW.unit_templates(SERVED) == ((SERVED, 1), (block(2, GLM), 8), (block(None, GLM), 16), (end(GLM), 8))
    assert SRW.unit_templates(row(T=16)) == ((row(T=16), 1), (block(16), 2), (E, 2))
    assert [(SRW.template(t).layout, SRW.template(t).word, CR.reads(SRW.template(t))) for t in TMPLS] == [
        ("chain", F8, "inline"), ("chain", B16, "slots"), ("chain", E, "inline")]
    assert [CR.reads(SRW.template(t)) for t in (block(2, GLM), block(64, GLM), block(None, GLM), end(GLM))] == [
        "inline", "slots", "slots", "inline"]
    assert [SRW.template(t).word for t in (block(8, boolean=True), block(boolean=True), end(boolean=True))] == TMPLS
    assert [CH.kind(SRW.template(t)) for t in TMPLS] == ["first", "block", "end"]
    DV, DR = TOY["DV"], TOY["DR"]
    assert CR.unit_rows(SRW.template(B16)) == ([("in0", V2, 16 * DV, ["q_nope"]), ("in1", V2, 16 * DR, ["q_rope"]),
                                                ("in2", V2, 16 * 16 * (DV + DR), ["kv"]), ("in3", V2, 32, ["m_prev"]),
                                                ("in4", V2, 4 * 32, ["l_prev"]), ("in5", V2, 32 * DV, ["o_prev"])],
                                               [(V2, (0,)), (V2, (1,)), (V2, (2,))])
    assert CR.unit_rows(SRW.template(E)) == ([("in0", V2, 4 * 32, ["l"]), ("in1", V2, 32 * DV, ["o"])], [(V2, (0,))])
    with pytest.raises(ValueError, match="visits a first block of 8, not NVIS=16"):
        CH.positions(SRW.template(block(16)), spec_statics(row()))


def test_each_later_block_and_end_is_wired_to_the_block_before_it():
    ms = members(Served(n=2))
    assert [(m.template, m.n) for m in ms] == [(F8, 2), (F8, 2), (B16, 3), (B16, 3), (B16, 2), (E, 4)]
    unit = units(ms, TMPLS, rows=2)
    at = {u: key for key, u in unit.items()}
    for m in ms:
        if m.template == F8:
            assert m.wired == ()
            continue
        prev = tuple(unit[(*at[m.base + i][:2], at[m.base + i][2] - 1)] for i in range(m.n))
        if m.template == B16:
            assert m.wired == (("in3", "o0", prev), ("in4", "o1", prev), ("in5", "o2", prev))
        else:
            assert m.wired == (("in0", "o1", prev), ("in1", "o2", prev))
    WI.validate(ms)


def test_the_units_read_the_twin_s_states_whose_end_is_the_served_row():
    """Every wired row is its writer's output row, each unit's ports are its block's keys and the twin's states, and each
    unit's Definition on its rows (as Flock's stager parses them) computes its committed outputs."""
    s = Served(n=2)
    ms = members(s)
    got = read(s, ms)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    unit, S = units(ms, TMPLS, rows=2), TOY
    at = {u: key for key, u in unit.items()}
    vs = M.visits(S["T"], S["BN"])

    def state(p, h, j):
        return [np.asarray(x[h], dtype=np.uint32).reshape(-1).view("<u2").tolist() for x in s.states[p][j]]

    for m, (g, _entries) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
        for i in range(m.n):
            p, h, j = at[m.base + i]
            if j == len(vs):
                assert [g["rows"][x][i].tolist() for x in ("in0", "in1")] == state(p, h, j - 1)[1:]
                assert g["outs"]["o0"][i].tolist() == s.y.reshape(2, S["NH"], -1)[p, h].tolist()
                continue
            qn, qr = s.ops[3 * p].reshape(S["NH"], -1)[h], s.ops[3 * p + 1].reshape(S["NH"], -1)[h]
            kv = s.ops[3 * p + 2].reshape(S["T"], -1)[vs[j][0]:vs[j][1]]
            assert [g["rows"][x][i].tolist() for x in ("in0", "in1", "in2")] == [qn.tolist(), qr.tolist(), kv.reshape(-1).tolist()]
            if j:
                assert [g["rows"][x][i].tolist() for x in ("in3", "in4", "in5")] == state(p, h, j - 1)
            assert [g["outs"][x][i].tolist() for x in ("o0", "o1", "o2")] == state(p, h, j)
    for t, o in ((F8, "o0"), (B16, "o2"), (E, "o0")):
        g = next(g for m, (g, _e) in zip(ms, got) if m.template == t)
        with pytest.raises(AssertionError, match="outputs are not the IR reference's"):     # an output its unit does not compute
            stager(t).reference_check(g["rows"], {**g["outs"], o: g["outs"][o] ^ np.uint16(1)})


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


def test_a_row_whose_served_words_are_not_its_chain_s_end_is_refused():
    s = Served()
    s.y[3] ^= np.uint16(1)
    with pytest.raises(ValueError, match="the twin's end of its 3 blocks is not the row's served out words"):
        read(s, members(s))


def test_a_partition_without_a_block_s_template_is_refused():
    s = Served()
    for tmpls, j in (([F8, E], 2), ([B16, E], 0)):
        with pytest.raises(ValueError, match=re.escape(f"block {j} (visit order) is the output of a block template the partition "
                                                       "does not name")):
            members(s, tmpls)
    assert all(m.wired == () for m in members(s, [F8]))


def test_a_row_at_the_served_statics_reads_and_wires_its_32_units():
    """One row of GLM-5.3's statics at T = 130, its 32 units (8 2-key first blocks, 16 64-key blocks, 8 ends), each block's
    state and each end wired to the block before it and every wired row its writer's output; the window itself (its
    population program at these statics lowers the 64-key block) is the circuit-check bindings'."""
    s = Served(SERVED, n=1, seed=16)
    tmpls = [block(2, GLM), block(None, GLM), end(GLM)]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 256))
    assert [(m.template, m.n) for m in ms] == [(tmpls[0], 8), (tmpls[1], 16), (tmpls[2], 8)]
    unit = units(ms, tmpls, GLM)
    assert ms[2].wired == tuple((f"in{g}", o, tuple(unit[(0, h, 2)] for h in range(8))) for g, o in ((0, "o1"), (1, "o2")))
    got = read(s, ms, tmpls)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    assert words[ms[2].port("o0")].reshape(-1).tolist() == s.y.tolist()


@pytest.mark.slow
def test_a_window_of_one_row_commits_and_its_wires_hold():
    s = Served()
    w = window(s)
    got = read(s, w.members)
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    j = next(j for j, m in enumerate(w.members) if m.template == B16 and m.n > 1)
    m = w.members[j]
    (p, o, ws), *rest = m.wired

    def rewired(wired):
        return [replace(x, wired=wired) if i == j else x for i, x in enumerate(w.members)]

    assert list(WI.check(rewired(((p, o, ws[1:] + ws[:1]), *rest)), publics)) == [m.key]       # another unit's running max
    swapped = tuple((q, "o1" if oo == "o2" else "o2" if oo == "o1" else oo, x) for q, oo, x in m.wired)
    assert list(WI.check(rewired(swapped), publics)) == [m.key]                     # the lanes read as O and O as the lanes
    names = [q for q, _ in m.all_rows]
    body = bytearray(publics[m.key])
    body[names.index("in5") * WI.BC_BYTES] ^= 1
    assert list(WI.check(w.members, {**publics, m.key: bytes(body)})) == [m.key]
    bad = [dict(g, rows={**g["rows"], "in5": g["rows"]["in5"] ^ np.uint16(1)}) if i == j else g for i, (g, _e) in enumerate(got)]
    with pytest.raises(ValueError, match=re.escape(f"{m.key}: a in5 row is not the o2 row")):
        CS.commit(w, bad, salt_key=bytes(32))
