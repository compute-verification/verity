"""A stochastic top-p select row cut into its stages (`pipeline.serving_rows`' ``stage`` layout, `stage_rows`): a
``GumbelTopPTokenSelect_v1{V}`` or ``GumbelTopPTokenSelectSharedGreedy_v1{V}`` row holds the units `topp_stages.plan` lists (at
GLM-5.3's V = 154,880, 8,159 of 85 Definitions; at V = 16, 392 of 59), and a window of a V = 16 row's every stage template:

* reads each unit's operands off the row (its logits lanes, the Program's constants, the split count among them, and the
  sampler's seed) and its
  carried words off the twin, whose token is the row's served one, so each unit's Definition on its rows, parsed as Flock's
  stager parses them, is its committed output;
* wires every carried parameter to the output row of the unit that wrote it, and refuses a row whose served token is not the
  twin's and a partition that leaves out a stage's template.

`test_a_row_reads_the_program_s_constants_and_the_sampler_s_words_at_its_step` holds `stage_rows._operand` to the replay's
reads of those operands; the others replace `serving_rows._slices`, which reads the rest.

A stage proved under ``--zk`` is its Boolean version (`boolean_topp_stages`): a partition naming it holds the same units on the
same rows, and `test_each_stage_on_bits_is_staged_in_the_form_verify_zk_takes` composes one of each family as serving's stager
does and holds its statement's form to `one_stage.proofs.ks_form_why`.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.commitments.windows import wiring as WI
from verity.definitions import boolean as BB
from verity.definitions.registry import boolean_topp_stages as BT
from verity.definitions.registry import sampling as SA
from verity.definitions.registry import sampling_rows as SR
from verity.definitions.registry import topp_keep_builder as K
from verity.definitions.registry import topp_stages as T
from verity.protocols.spot.verifier.one_stage import proofs as PF
from verity_flock.class_statement import UnitStager

from verity_vllm.check.replay.index import SAMPLER_OP
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import stage_rows as ST
from verity_vllm.query.program_view import spec_base, spec_statics

V = 16
SERVED_V = 154880
ROW = descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=V))
TMPLS = [t for t, _n in SRW.unit_templates(ROW)[1:]]
PER_CHUNK = dict.fromkeys(TMPLS, 7)
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


def boolean(t: str) -> str:
    """Stage template ``t``'s Boolean version, the template a ``--zk`` partition names: `boolean_topp_stages`' family of its
    name at its statics (what `boolean_version` resolves it to, without loading the catalog)."""
    fn = SRW.template(t).fn
    return descriptor_id(bind(next(d for d in BT.BOOLEAN_STAGE_DEFINITIONS if d.name == fn.name), **fn.bindings))


def composed(t: str, d: Path) -> UnitStager:
    """Template ``t``'s statement as serving's stager composes it (its rows `chunk_rows.unit_rows`, its table reads laid out
    as `chunk_rows.statement` says, bound as ``compose_circuit.py --template`` binds it), its circuit ``d/circuit.txt``."""
    tp = SRW.template(t)
    in_rows, out_rows = CR.unit_rows(tp)
    return UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=d / "circuit.txt",
                      bind={"partition": "0" * 128, "program": "0" * 128}, **CR.statement(tp))


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
    forms hold the same ones, and at GLM-5.3's vocabulary 8,159 of 85."""
    plain = descriptor_id(bind(SA.GumbelTopPTokenSelect, V=V))
    cut = [(descriptor_id(d), n) for d, n in T.cut(bind(SA.GumbelTopPTokenSelect, V=V))]
    assert SRW.unit_templates(ROW) == ((ROW, 1), *cut) and SRW.unit_templates(plain) == ((plain, 1), *cut)
    assert len(cut) == 59 and sum(n for _t, n in cut) == len(T.plan(V)) == 392
    assert {(SRW.template(t).layout, SRW.template(t).word == t, CR.reads(SRW.template(t))) for t in TMPLS} == {("stage", True, "slots")}
    served = SRW.unit_templates(descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=SERVED_V)))[1:]
    assert len(served) == 85 and sum(n for _t, n in served) == 8159
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


@pytest.mark.slow
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
    ops = [{x: int(s.ops[6 * p + g].view(f"<u{s.ops[6 * p + g].dtype.itemsize}")[0])
            for g, x in enumerate(ST.OPERANDS[spec_base(ROW)]) if g} for p in range(2)]
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


def test_a_partition_of_the_stages_on_bits_holds_the_same_units_on_the_same_rows():
    """Each stage's Boolean version is a ``stage`` template of the same word on the same rows, laid out flat (it has no table
    to read); a partition naming them wires and reads the row's units as one naming the words does."""
    bits = [boolean(t) for t in TMPLS]
    assert len(set(bits)) == len(TMPLS) and not set(bits) & set(TMPLS)
    for b, t in zip(bits, TMPLS):
        tb = SRW.template(b)
        assert (tb.layout, tb.word, CR.reads(tb), CR.statement(tb)) == ("stage", t, "inline", {"reads": "inline"}), b
        assert CR.unit_rows(tb) == CR.unit_rows(SRW.template(t)) and tb.fn is not SRW.template(t).fn, b
    s = Served()
    ms, mb = members(s), members(s, bits, dict.fromkeys(bits, 7))
    WI.validate(mb)
    of = dict(zip(bits, TMPLS))
    assert [replace(m, template=of[m.template], name=x.name) for m, x in zip(mb, ms, strict=True)] == ms
    for (g, e), (gb, eb) in zip(read(s, ms), read(s, mb, bits), strict=True):
        assert e == eb and all(np.array_equal(g[part_][q], gb[part_][q]) for part_ in ("rows", "outs") for q in g[part_])


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
    """A Program constant is its one word, the split count S among them; the sampler's seed is the request's committed word at
    the row's step; a ``splits`` Program input is no sampler input, so the serving rows' reads refuse it, as any other."""
    class Pi:
        pnames = ["weights", "seed", "splits"]
        by_i = {3: {"spec": "Const32[0x3f666666]_v1"}, 4: {"spec": "Const64[0x2a]_v1"}, 5: {"spec": "Const32[0x8]_v1"}}

        def member_at(self, a, lo, hi):
            return None

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            assert (rid, op, lo, hi) == ("r0", SAMPLER_OP, 0, 1)
            return (np.array([step * 100 + {"seed": 1}[member]], np.int64), None)

    got = ST._operand(Pi(), Store(), None, "r0", 10, [["n", 3, 0, 1], ["n", 4, 0, 1], ["p", 1, 0, 1], ["n", 5, 0, 1]], 5, SRW._slices)
    assert [(a.dtype.name, a.tolist()) for a in got] == [("uint32", [0x3F666666]), ("uint64", [42]), ("int64", [1501]),
                                                         ("uint32", [8])]
    with pytest.raises(LookupError, match=re.escape("seed slice [4, 6) is not the sampler's one word at the row's step 5")):
        ST._operand(Pi(), Store(), None, "r0", 10, [["p", 1, 4, 6]], 5, SRW._slices)
    with pytest.raises(LookupError, match="the serving rows read committed members and registered weights only"):
        ST._operand(Pi(), Store(), None, "r0", 10, [["p", 2, 5, 6]], 5, SRW._slices)
    with pytest.raises(LookupError, match="is not inside one committed member"):
        ST._operand(Pi(), Store(), None, "r0", 10, [["n", 7, 0, 1]], 5, SRW._slices)
    with pytest.raises(LookupError, match="the serving rows read committed members and registered weights only"):
        SRW._slices(Pi(), Store(), None, "r0", 10, [["p", 1, 0, 1]])


@pytest.mark.slow
def test_a_row_at_the_served_statics_reads_and_wires_its_8159_units():
    """One row of GLM-5.3's vocabulary: its 8,159 units wired and read, every wired row its writer's output and the select's
    last chunk the served token.  It catches what V = 16 can't reach: a stats block cut in parts, slices of thousands of lanes
    whose select chunks cross them, and the twin and wiring at those shapes; its minutes are the twin's six pipelines over
    154,880 lanes."""
    spec = descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=SERVED_V))
    s = Served(spec, n=1, seed=16)
    tmpls = [t for t, _n in SRW.unit_templates(spec)[1:]]
    ms = members(s, tmpls, dict.fromkeys(tmpls, 4096))
    WI.validate(ms)
    assert sum(m.n for m in ms) == 8159
    got = read(s, ms, tmpls)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    last = descriptor_id(T.plan(SERVED_V)[-1].defn)
    (m,) = [m for m in ms if m.template == last]
    assert int(words[m.port("o0")][0].view("<u4")[0]) == int(s.tok[0])


#: one unit Definition of each stage family whose word reads a table, (V, family, statics), at the least statics run here
#: that its rows compose at: a stats block's rows are past one SHA-512 block, as a staged row must be, only from 56 lanes
ZK_UNITS = ((64, "TopPStatsHead", {"N": 64}), (64, "TopPStatsExp", {"N": 64}), (16, "TopPProbs", {"S": 2, "HEAD": False}),
            (16, "TopPStep", {"S": 2, "COMBINE": 0}), (16, "GumbelTopPSelect", {"LAST": True}))


@pytest.mark.slow
def test_each_stage_on_bits_is_staged_in_the_form_verify_zk_takes(tmp_path):
    """`verify --zk` takes a statement whose unit returns its own outputs and refuses one whose last stage returns them
    (`one_stage.proofs.ks_form_why`, the mirror of Lean's `KsScope`).  A stage's word template is staged with its table reads
    in lookup slots, the refused form.  Its Boolean version has no table to read: one unit Definition of each of the five
    families, in a partition naming every stage on bits, is staged flat, the form it takes, and its circuit on the unit's rows
    returns the words serving commits for it (the twin's).  Minutes, all of them composing the six circuits: the form is the
    composed circuit's (its META ``out_net``), and the words are its gates', which nothing short of composing it reads."""
    refused = f"refused: {PF.SCOPE_WHY}: a statement form the soundness proof does not cover"
    seen = []
    for v in sorted({v for v, _name, _statics in ZK_UNITS}):
        spec = descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=v))
        want = {descriptor_id(u.defn): u.defn.name for u in T.plan(v) for x, name, statics in ZK_UNITS
                if x == v and u.defn.name == name and all(u.defn.bindings[k] == y for k, y in statics.items())}
        bits = [boolean(t) for t, _n in SRW.unit_templates(spec)[1:]]
        assert bits == [descriptor_id(BB.boolean_version(SRW.template(t).fn)) for t, _n in SRW.unit_templates(spec)[1:]]
        s = Served(spec, n=1, seed=v)
        ms = members(s, bits, dict.fromkeys(bits, 2))
        for m, (g, _e) in zip(ms, read(s, ms, bits)):
            name = want.pop(SRW.template(m.template).word, None)
            if name is None:
                continue
            d = tmp_path / name
            st = composed(m.template, d)
            assert (st.meta["lookups"], st.meta.get("out_net", "unit")) == ({}, "unit"), m.template
            got = st.stage(g["rows"], d, bytes(32), indices=range(m.base, m.base + m.n))
            assert g["outs"] and all(np.array_equal(got["outs"][q], a) for q, a in g["outs"].items()), m.template
            assert PF.ks_form_why(d / "circuit.txt", d / "pub.bin") is None, m.template
            seen.append(name)
        assert want == {}
    assert sorted(seen) == sorted(d.name for d in T.STAGE_DEFINITIONS)
    (probs,) = [t for t in TMPLS if SRW.template(t).fn.name == "TopPProbs" and spec_statics(t)["S"] == 2 and not spec_statics(t)["HEAD"]]
    word = composed(probs, tmp_path / "word")
    assert set(word.meta["lookups"]) == {"ex2", "rcp"} and word.meta["out_net"].startswith("stage")
    assert PF.ks_form_why(tmp_path / "word" / "circuit.txt", tmp_path / "word" / "pub.bin") == refused
