"""A greedy select row cut into its stages (`pipeline.serving_rows`' ``stage`` layout, `stage_rows`): a ``TokenSelect_v1{V}`` row
holds the units `greedy_stages.plan` lists (at GLM-5.3's V = 154,880, 10 of 3 Definitions; at V = 200, 4 of 3), and a window of
a V = 200 row's every stage template:

* reads each unit's logits lanes off the row (the combine's, the last chunk's) and the combine's pairs off the twin, whose token
  is the row's served one, so each unit's Definition on its rows, parsed as Flock's stager parses them, is its committed output;
* wires every combine pair to the output row of the chunk that wrote it, and refuses a row whose served token is not the twin's
  and a partition that leaves out a chunk's template.

At the served statics each unit is a proof block Flock's composer takes (`interface.check_proof_block`).  A chunk lowers flat to
a circuit linear in its lanes (`test_a_chunk_lowers_linearly_in_its_width`), so the served one's AND rows hold a small share of
the GPU prover's 2^32 nonzeros a net.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from verity_flock import ir_lower as IL
from verity_flock.class_statement import UnitStager
from verity_flock.gf2 import Circuit
from verity.commitments.windows import interface as IF
from verity.commitments.windows import wiring as WI
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import stage_rows as ST
from verity_vllm.query.program_view import spec_statics

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.definitions.registry import b1
from verity.definitions.registry import greedy_stages as G

V = 200
SERVED_V = 154880
ROW = descriptor_id(bind(b1.TokenSelect, V=V))
TMPLS = [t for t, _n in SRW.unit_templates(ROW)[1:]]
PER_CHUNK = dict.fromkeys(TMPLS, 7)
STEP = 4


class Served:
    """``n`` rows of ``spec`` (row p's logits are argument p, as `slices` reads them, every other row's rounded to integers, so
    its maximum ties) and the store of their served tokens, `TokenSelect_v1{V}`'s as the circuit evaluator computes them."""

    def __init__(self, spec: str = ROW, n: int = 1, seed: int = 7):
        S = spec_statics(spec)
        v = int(S["V"])
        rng = np.random.default_rng(seed)
        self.ops, self.rows, toks = {}, [], []
        for p in range(n):
            x = (rng.standard_normal(v) * 3).astype(np.float32)
            x = np.round(x).astype(np.float32) if p % 2 else x
            self.ops[p] = (x.view(np.uint32) >> 16).astype(np.uint16)
            toks.append(int(evaluate(bind(b1.TokenSelect, V=v), [int(w) for w in self.ops[p]])[0]))
            r = {"i": 9 + p, "spec": spec, "args": [[["x", p, 0, v]]]}
            d = {"statics": S, "op_path": "runner.sampler", "members": [("sampled_token_ids", 1, "i64")], "row": p, "step": STEP}
            self.rows.append(("r0", None, r, d))
        self.tok = np.asarray(toks, dtype=np.int64)

    def slices(self, _pi, _store, _weights, _rid, _arrive, g):
        """`serving_rows._slices`: an argument group's elements."""
        return [self.ops[g[0][1]]]

    def words(self, rid, step, op, member, lo, hi):
        return self.tok[lo:hi], None


def members(s: Served, tmpls=TMPLS, per_chunk=PER_CHUNK):
    return ST.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, per_chunk), tmpls, [s.rows] * len(tmpls))


def read(s: Served, ms, tmpls=TMPLS):
    SRW._slices = s.slices
    return SRW.read_all(ms, [s.rows] * len(tmpls), s, None)


@pytest.fixture(autouse=True)
def _slices(monkeypatch):
    """`serving_rows._slices` back after each test, which `read` replaces."""
    monkeypatch.setattr(SRW, "_slices", SRW._slices)


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "stage-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def places(ms, tmpls, rows: int = 1, v: int = V) -> dict[int, tuple[int, tuple]]:
    """Each global unit's (row, plan key): a template's units row by row, then in the plan's order."""
    plan, out = G.plan(v), {}
    for t in tmpls:
        u = min(m.base for m in ms if m.template == t)
        for p in range(rows):
            for k in ST.positions(SRW.template(t), {"V": v}):
                out[u], u = (p, plan[k].key), u + 1
    return out


def test_a_greedy_row_holds_its_stages():
    """The units `greedy_stages.cut` cuts the row into, each a ``stage`` template of 16-bit ports, and at GLM-5.3's vocabulary
    10 of 3; the combine reads the last chunk's 8 lanes as one row and each other chunk's best and index as rows of their own."""
    cut = [(descriptor_id(d), n) for d, n in G.cut(bind(b1.TokenSelect, V=V))]
    assert SRW.unit_templates(ROW) == ((ROW, 1), *cut) and len(cut) == 3 and sum(n for _t, n in cut) == len(G.plan(V)) == 4
    assert {(SRW.template(t).layout, SRW.template(t).word == t, SRW.template(t).packed, CR.reads(SRW.template(t)))
            for t in TMPLS} == {("stage", True, None, "slots")}
    served = SRW.unit_templates(descriptor_id(bind(b1.TokenSelect, V=SERVED_V)))[1:]
    assert len(served) == 3 and sum(n for _t, n in served) == 10
    combine = descriptor_id(G.plan(V)[-1].defn)
    pairs = [(f"in{1 + 2 * c + j}", V2, 16, [f"{'bi'[j]}{c}"]) for c in range(3) for j in (0, 1)]
    assert CR.unit_rows(SRW.template(combine)) == ([("in0", V2, 128, ["logits"]), *pairs], [(V2, (0,)), (V2, (1,))])


def test_every_served_unit_s_proof_block_composes():
    """At GLM-5.3's vocabulary every unit of the cut is a proof block Flock's composer takes (`interface.check_proof_block`):
    a chunk's logits are a row past 111 bytes, and so are the combine's, the last chunk's 7,424 lanes, beside its chunks'
    2-byte pairs, which alone it would refuse; each unit's rows fit 2^25 bits."""
    got = {descriptor_id(d): IF.check_proof_block(d) for d, _n in G.cut(bind(b1.TokenSelect, V=SERVED_V))}
    assert got == {"GreedyChunk_v1{N=16384,FIRST=true}": 25, "GreedyChunk_v1{N=16384,FIRST=false}": 25,
                   "GreedyCombine_v1{V=154880,N=16384}": 25}


def test_verify_zk_takes_every_served_unit():
    """Each served unit reads in slots but no table a slot holds, so `verify --zk` takes it (`chunk_rows.zk_refusal`)."""
    for d, _n in G.cut(bind(b1.TokenSelect, V=SERVED_V)):
        tp = SRW.template(descriptor_id(d))
        assert (CR.zk_refusal(tp), CR.slot_tables(tp.fn)) == (None, ()), tp.id


def test_each_combine_parameter_is_wired_to_the_chunk_that_wrote_it():
    s = Served(n=2)
    ms = members(s)
    assert sum(m.n for m in ms) == 2 * 4
    WI.validate(ms)
    at = places(ms, TMPLS, rows=2)
    unit = {key: u for u, key in at.items()}
    plan = {u.key: u for u in G.plan(V)}
    for m in ms:
        names = [x for x, _ty in SRW.template(m.template).fn.signature()[0]]
        for i in range(m.n):
            p, key = at[m.base + i]
            want = {(f"in{names.index(x)}", f"o{j}", unit[(p, w)]) for x, w, j in plan[key].wires}
            assert {(q, o, ws[i]) for q, o, ws in m.wired} == want, (m.key, key)


def test_the_units_read_the_twin_s_words_whose_token_is_the_served_row():
    """Every wired row is its writer's output row, each unit reads its logits lanes, the combine returns the served token's two
    words, and each unit's Definition on its rows (as Flock's stager parses them) computes its committed outputs."""
    s = Served(n=2)
    ms = members(s)
    got = read(s, ms)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    at, plan = places(ms, TMPLS, rows=2), {u.key: u for u in G.plan(V)}
    for m, (g, entries) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
        lo = min(x.base for x in ms if x.template == m.template)
        per = sum(x.n for x in ms if x.template == m.template) // 2
        rows = [[lo + p * per, lo + (p + 1) * per, "r0", STEP, None, "runner.sampler", "sampled_token_ids", p, 9 + p] for p in range(2)]
        assert entries == [e for e in rows if e[0] < m.base + m.n and e[1] > m.base]
        for i in range(m.n):
            p, key = at[m.base + i]
            assert g["rows"]["in0"][i].tolist() == s.ops[p][plan[key].lanes[0]:plan[key].lanes[1]].tolist()
            if key[0] == "combine":
                assert [int(g["outs"][o][i][0]) for o in ("o0", "o1")] == [int(s.tok[p]) & 0xFFFF, int(s.tok[p]) >> 16]
    t = TMPLS[0]
    g = next(g for m, (g, _e) in zip(ms, got) if m.template == t)
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):     # an output its unit does not compute
        stager(t).reference_check(g["rows"], {**g["outs"], "o1": g["outs"]["o1"] ^ np.uint16(1)})


def test_a_row_whose_served_token_is_not_the_twin_s_is_refused():
    s = Served()
    s.tok[0] = (int(s.tok[0]) + 1) % V
    with pytest.raises(ValueError, match=re.escape(f"the twin's token {(int(s.tok[0]) - 1) % V} is not the row's served "
                                                   "sampled_token_ids")):
        read(s, members(s))


def test_a_partition_without_a_chunk_s_template_is_refused():
    s = Served()
    later = descriptor_id(G.plan(V)[1].defn)
    with pytest.raises(ValueError, match=re.escape("row 9's b1 is the output of its stage ('chunk', 1), whose template the "
                                                   "partition does not name")):
        members(s, [t for t in TMPLS if t != later])
    assert all(m.wired == () for m in members(s, [descriptor_id(G.plan(V)[0].defn)]))


def _flat(N: int) -> tuple[int, int, int]:
    """``GreedyChunk_v1{N, FIRST=false}``'s flat lowering (`ir_lower`): its ANDs, their operands' bits (the AND rows' nonzeros)
    and the widest operand."""
    Cc = Circuit()
    IL.lower(Cc, bind(G.GreedyChunk, N=N, FIRST=False), [Cc.input(f"x{j}", 16) for j in range(N)])
    ops = [len(f.s) for k, d in zip(Cc.kind, Cc.data) if k == "and" for f in d]
    return len(ops) // 2, sum(ops), max(ops)


def test_a_chunk_lowers_linearly_in_its_width():
    """Each lane's ANDs read the running (best, index) as forms of a few bits at every width (each select keeps it as the
    branch under its AND), so lanes 130 .. 249 each add the same ANDs and nonzeros and no operand widens: a served chunk of
    16,384 lanes holds about 19M nonzeros in its AND rows, under 1/200 of the GPU prover's 2^32 a net."""
    (a1, z1, w1), (a2, z2, w2), (a3, z3, w3) = _flat(130), _flat(190), _flat(250)
    assert w1 == w2 == w3
    assert a3 - a2 == a2 - a1 > 0 and z3 - z2 == z2 - z1 > 0
    assert z3 + (G.CHUNK - 250) * (z3 - z2) // 60 < 2**32 // 200


@pytest.mark.slow
def test_a_row_at_the_served_statics_reads_and_wires_its_10_units():
    """One row of GLM-5.3's vocabulary: its 10 units wired and read, each Definition computing its committed outputs, and the
    combine the served token's two words.  It catches what V = 200 can't reach: chunks of 16,384 lanes, the combine's last chunk
    of 7,424, and a token whose high word is not 0; its seconds are the evaluator's on the row and its units."""
    spec = descriptor_id(bind(b1.TokenSelect, V=SERVED_V))
    s = Served(spec, n=1, seed=16)
    s.ops[0][150000] = 0x7F00
    s.tok[0] = 150000
    tmpls = [t for t, _n in SRW.unit_templates(spec)[1:]]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 16))
    WI.validate(ms)
    assert sum(m.n for m in ms) == 10
    got = read(s, ms, tmpls)
    for m, (g, _e) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    (m, (g, _e)), = [(m, x) for m, x in zip(ms, got) if m.template == descriptor_id(G.plan(SERVED_V)[-1].defn)]
    assert [int(g["outs"][o][0][0]) for o in ("o0", "o1")] == [150000 & 0xFFFF, 2]
