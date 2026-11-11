"""A stochastic top-p select row cut into its stages (`pipeline.serving_rows`' ``stage`` layout, `stage_rows`): a
``GumbelTopPTokenSelect_v1{V}`` or ``GumbelTopPTokenSelectSharedGreedy_v1{V}`` row holds the units `topp_stages.plan` lists (at
GLM-5.3's V = 154,880, 3,924 of 85 Definitions; at V = 16, 392 of 59), and a window of a V = 16 row's every stage template:

* reads each unit's operands off the row (its logits lanes, the Program's constants, the sampler's seed and splits) and its
  carried words off the twin, whose token is the row's served one, so each unit's Definition on its rows, parsed as Flock's
  stager parses them, is its committed output;
* wires every carried parameter to the output row of the unit that wrote it, and refuses a row whose served token is not the
  twin's and a partition that leaves out a stage's template.

`test_a_row_reads_the_program_s_constants_and_the_sampler_s_words_at_its_step` holds `stage_rows._operand` to the replay's
reads of those operands; the others replace `serving_rows._slices`, which reads the rest.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.commitments.windows import wiring as WI
from verity_vllm.check.replay.index import SAMPLER_OP
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import stage_rows as ST
from verity_vllm.query.program_view import spec_statics

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.definitions.registry import sampling as SA
from verity.definitions.registry import sampling_rows as SR
from verity.definitions.registry import topp_keep_builder as K
from verity.definitions.registry import topp_stages as T

V = 16
SERVED_V = 154880
ROW = descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=V))
TMPLS = [t for t, _n in SRW.unit_templates(ROW)[1:]]
CAPS = dict.fromkeys(TMPLS, 7)
STEP = 4


def f32(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


#: per row: temperature, top_p and the split count serving launched
SETTINGS = ((0.7, 0.9, 16), (0.0, 0.8, 1), (1.3, 0.5, 32), (1.0, 0.95, 8))


class Served:
    """``n`` rows of ``spec`` (row p's logits, top_p, seed, pos, temp and splits are arguments 6p .. 6p + 5, as `slices` reads
    them) and the store of their served tokens (`sampling_rows.gumbel_topp_token_select_row`'s)."""

    def __init__(self, spec: str = ROW, n: int = 1, seed: int = 7):
        S = spec_statics(spec)
        rng = np.random.default_rng(seed)
        self.ops, self.rows, toks = {}, [], []
        for p in range(n):
            x = (rng.standard_normal(int(S["V"])) * 3).astype(np.float32)
            x = np.round(x).astype(np.float32) if p % 2 else x
            temp, top_p, splits = SETTINGS[p % len(SETTINGS)]
            vals = [(x.view(np.uint32) >> 16).astype(np.uint16), np.array([f32(top_p)], np.uint32),
                    np.array([rng.integers(-2**63, 2**63 - 1)], np.int64), np.array([rng.integers(0, 4096)], np.uint64),
                    np.array([f32(temp)], np.uint32), np.array([splits], np.int64)]
            self.ops.update({6 * p + g: v for g, v in enumerate(vals)})
            toks.append(SR.gumbel_topp_token_select_row(vals[0], f32(top_p), int(vals[2][0]) % 2**64, int(vals[3][0]), f32(temp),
                                                        splits)["token"])
            r = {"i": 9 + p, "spec": spec, "args": [[["x", 6 * p + g, 0, vals[g].size]] for g in range(6)]}
            d = {"statics": S, "op_path": "sampler.select", "members": [("token", 1, "i64")], "row": p, "step": STEP}
            self.rows.append(("r0", None, r, d))
        self.tok = np.asarray(toks, dtype=np.int64)

    def slices(self, _pi, _store, _weights, _rid, _arrive, g):
        """`serving_rows._slices`: an argument group's elements."""
        return [self.ops[g[0][1]]]

    def words(self, rid, step, op, member, lo, hi):
        return self.tok[lo:hi], None


def members(s: Served, tmpls=TMPLS, caps=CAPS):
    return ST.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, caps), tmpls, [s.rows] * len(tmpls))


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
    plan, out = T.plan(v), {}
    for t in tmpls:
        u = min(m.base for m in ms if m.template == t)
        for p in range(rows):
            for k in ST.positions(SRW.template(t), {"V": v}):
                out[u], u = (p, plan[k].key), u + 1
    return out


def test_a_select_row_holds_its_stages():
    """The units `topp_stages.cut` cuts the row into, each a ``stage`` template whose EX2 reads go in lookup slots; both Call
    forms hold the same ones, and at GLM-5.3's vocabulary 3,924 of 85."""
    plain = descriptor_id(bind(SA.GumbelTopPTokenSelect, V=V))
    cut = [(descriptor_id(d), n) for d, n in T.cut(bind(SA.GumbelTopPTokenSelect, V=V))]
    assert SRW.unit_templates(ROW) == ((ROW, 1), *cut) and SRW.unit_templates(plain) == ((plain, 1), *cut)
    assert len(cut) == 59 and sum(n for _t, n in cut) == len(T.plan(V)) == 392
    assert {(SRW.template(t).layout, SRW.template(t).word == t, CR.reads(SRW.template(t))) for t in TMPLS} == {("stage", True, "slots")}
    served = SRW.unit_templates(descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=SERVED_V)))[1:]
    assert len(served) == 85 and sum(n for _t, n in served) == 3924
    last = descriptor_id(T.plan(V)[-1].defn)
    N = T.plan(V)[-1].lanes[1] - T.plan(V)[-1].lanes[0]
    sw = [(f"in{5 + i}", V2, 32 * T.scw(S), [f"sw{S}"]) for i, S in enumerate(K.SPLIT_COUNTS)]
    assert CR.unit_rows(SRW.template(last)) == ([("in0", V2, 16 * N, ["logits"]), ("in1", V2, 32, ["temp"]), ("in2", V2, 64, ["seed"]),
                                                 ("in3", V2, 64, ["pos"]), ("in4", V2, 32, ["splits"]), *sw,
                                                 (f"in{5 + len(sw)}", V2, 32 * T.CARRY, ["cw"])], [(V2, (0,))])


def test_each_carried_parameter_is_wired_to_the_unit_that_wrote_it():
    s = Served(n=2)
    ms = members(s)
    assert sum(m.n for m in ms) == 2 * 392 and max(m.n for m in ms) == 7
    WI.validate(ms)
    at = places(ms, TMPLS, rows=2)
    unit = {key: u for u, key in at.items()}
    plan = {u.key: u for u in T.plan(V)}
    for m in ms:
        names = [x for x, _ty in SRW.template(m.template).fn.signature()[0]]
        for i in range(m.n):
            p, key = at[m.base + i]
            want = {(f"in{names.index(x)}", f"o{j}", unit[(p, w)]) for x, w, j in plan[key].wires}
            assert {(q, o, ws[i]) for q, o, ws in m.wired} == want, (m.key, key)


def test_the_units_read_the_twin_s_words_whose_token_is_the_served_row():
    """Every wired row is its writer's output row, each unit reads its logits lanes and the row's operands, the select's last
    chunk returns the served token, and each unit's Definition on its rows (as Flock's stager parses them) computes its
    committed outputs."""
    s = Served(n=2)
    ms = members(s)
    got = read(s, ms)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    at, plan = places(ms, TMPLS, rows=2), {u.key: u for u in T.plan(V)}
    ops = [{x: int(s.ops[6 * p + g].view(f"<u{s.ops[6 * p + g].dtype.itemsize}")[0]) for g, x in enumerate(ST.OPERANDS) if g}
           for p in range(2)]
    for m, (g, entries) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
        lo = min(x.base for x in ms if x.template == m.template)
        per = sum(x.n for x in ms if x.template == m.template) // 2
        rows = [[lo + p * per, lo + (p + 1) * per, "r0", STEP, None, "sampler.select", "token", p, 9 + p] for p in range(2)]
        assert entries == [e for e in rows if e[0] < m.base + m.n and e[1] > m.base]
        names = [x for x, _ty in SRW.template(m.template).fn.signature()[0]]
        for i in range(m.n):
            p, key = at[m.base + i]
            u = plan[key]
            if "logits" in names:
                assert g["rows"][f"in{names.index('logits')}"][i].tolist() == s.ops[6 * p][u.lanes[0]:u.lanes[1]].tolist()
            for x in set(names) & set(ops[p]):
                assert int(g["rows"][f"in{names.index(x)}"][i].view(f"<u{ST.BITS[x] // 8}")[0]) == ops[p][x], (key, x)
            if key == T.plan(V)[-1].key:
                assert int(g["outs"]["o0"][i].view("<u4")[0]) == int(s.tok[p])
    t = TMPLS[0]
    g = next(g for m, (g, _e) in zip(ms, got) if m.template == t)
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):     # an output its unit does not compute
        stager(t).reference_check(g["rows"], {**g["outs"], "o0": g["outs"]["o0"] ^ np.uint16(1)})


def test_a_row_whose_served_token_is_not_the_twin_s_is_refused():
    s = Served()
    s.tok[0] = (int(s.tok[0]) + 1) % V
    with pytest.raises(ValueError, match=re.escape(f"the twin's token {(int(s.tok[0]) - 1) % V} is not the row's served token")):
        read(s, members(s))


def test_a_partition_without_a_stage_s_template_is_refused():
    s = Served()
    head = descriptor_id(next(u.defn for u in T.plan(V) if u.key[0] == "probs" and u.defn.bindings["HEAD"]))
    with pytest.raises(ValueError, match=re.escape("row 9's pr is the output of its stage ('probs', 1, 0), whose template the "
                                                   "partition does not name")):
        members(s, [t for t in TMPLS if t != head])
    first = descriptor_id(next(u.defn for u in T.plan(V) if u.key[0] == "head" and u.defn.bindings["FIRST"]))
    assert all(m.wired == () for m in members(s, [first]))


def test_a_row_reads_the_program_s_constants_and_the_sampler_s_words_at_its_step():
    """A Program constant is its one word; the sampler's seed and splits are the request's committed words at the row's step
    (``splits``' slice names that step); any other row's reads refuse them, as before."""
    class Pi:
        pnames = ["weights", "seed", "splits"]
        by_i = {3: {"spec": "Const32[0x3f666666]_v1"}, 4: {"spec": "Const64[0x2a]_v1"}}

        def member_at(self, a, lo, hi):
            return None

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            assert (rid, op, lo, hi) == ("r0", SAMPLER_OP, 0, 1)
            return (np.array([step * 100 + {"seed": 1, "splits": 2}[member]], np.int64), None)

    got = ST._operand(Pi(), Store(), None, "r0", 10, [["n", 3, 0, 1], ["n", 4, 0, 1], ["p", 1, 0, 1], ["p", 2, 5, 6]], 5, SRW._slices)
    assert [(a.dtype.name, a.tolist()) for a in got] == [("uint32", [0x3F666666]), ("uint64", [42]), ("int64", [1501]),
                                                         ("int64", [1502])]
    with pytest.raises(LookupError, match=re.escape("splits slice [4, 5) is not the sampler's one word at the row's step 5")):
        ST._operand(Pi(), Store(), None, "r0", 10, [["p", 2, 4, 5]], 5, SRW._slices)
    with pytest.raises(LookupError, match="is not inside one committed member"):
        ST._operand(Pi(), Store(), None, "r0", 10, [["n", 7, 0, 1]], 5, SRW._slices)
    with pytest.raises(LookupError, match="the serving rows read committed members and registered weights only"):
        SRW._slices(Pi(), Store(), None, "r0", 10, [["p", 1, 0, 1]])


@pytest.mark.slow
def test_a_row_at_the_served_statics_reads_and_wires_its_3924_units():
    """One row of GLM-5.3's vocabulary: its 3,924 units wired and read, every wired row its writer's output and the select's
    last chunk the served token.  It catches what V = 16 can't reach: a stats block cut in parts, slices of thousands of lanes
    whose select chunks cross them, and the twin and wiring at those shapes; its minutes are the twin's six pipelines over
    154,880 lanes."""
    spec = descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=SERVED_V))
    s = Served(spec, n=1, seed=16)
    tmpls = [t for t, _n in SRW.unit_templates(spec)[1:]]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 4096))
    WI.validate(ms)
    assert sum(m.n for m in ms) == 3924
    got = read(s, ms, tmpls)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    last = descriptor_id(T.plan(SERVED_V)[-1].defn)
    (m,) = [m for m in ms if m.template == last]
    assert int(words[m.port("o0")][0].view("<u4")[0]) == int(s.tok[0])
