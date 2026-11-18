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
from pathlib import Path

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock.class_statement import UnitStager

from verity_vllm.check.replay.index import SAMPLER_OP
from verity_vllm.definitions.registry import boolean_topp_stages as BT
from verity_vllm.definitions.registry import sampling as SA
from verity_vllm.definitions.registry import sampling_rows as SR
from verity_vllm.definitions.registry import topp_keep_builder as K
from verity_vllm.definitions.registry import topp_stages as T
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import stage_rows as ST
from verity_vllm.query.program_view import spec_statics

V = 16
SERVED_V = 154880
ROW = descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=V))
TMPLS = [t for t, _n in SRW.unit_templates(ROW)[1:]]
PER_CHUNK = dict.fromkeys(TMPLS, 7)
STEP = 4


#: `chain_rows.pieces`' refusal of a port reading several templates' trees
NAME = "a wired port reads one template's tree"

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


def test_a_row_whose_served_token_is_not_the_twin_s_is_refused():
    s = Served()
    s.tok[0] = (int(s.tok[0]) + 1) % V
    with pytest.raises(ValueError, match=re.escape(f"the twin's token {(int(s.tok[0]) - 1) % V} is not the row's served token")):
        read(s, SRW.members(TMPLS, [s.rows] * len(TMPLS), None, PER_CHUNK))


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


def test_a_stage_reading_its_first_and_its_later_stages_is_refused_by_name():
    """A stage's carried parameter is a FIRST stage's output at one step and a later stage's at the next: one port reading
    two templates' trees, which `chain_rows.pieces` refuses by name."""
    with pytest.raises(ValueError, match=NAME):
        members(Served(n=2))
