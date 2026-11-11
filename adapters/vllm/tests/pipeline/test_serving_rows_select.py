"""A greedy select row split into its group units and combine (`pipeline.serving_rows`' ``stage`` layout, `select_rows`): a
``TokenSelect_v1{V}`` row holds the units a served window's consumer splits it into (`tp.wires`; `serving_templates.select_split`),
and a window of both templates:

* reads each group unit's logits off the row, in rows of the split's width, and its groups' bests and indices off the twin,
  whose token is the row's served one (`numerics.rows.token_select_first_max`, the row's own twin), so each unit's Definition on
  its rows, parsed as Flock's stager parses them, is its committed output;
* wires the combine's row s of bests and of indices to group unit s's two output rows, and refuses a row whose served token is
  not the twin's and a partition that names the combine without its group units.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.commitments.windows import interface as IF
from verity.commitments.windows import wiring as WI
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import select_rows as SE
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as TM
from verity_vllm.pipeline import stage_rows as ST
from verity_vllm.query.program_view import spec_statics

from verity.definitions.numerics.rows import token_select_first_max

V, W, C, G = 32768, 4, 512, TM.SELECT_GROUP
SERVED_V = 154880
ROW = f"TokenSelect_v1{{V={V}}}"
GROUPS, COMBINE = f"TokenSelectGroups_v1{{C={C},G={G}}}", f"TokenSelectCombine_v2{{V={V},C={G}}}"
TMPLS = [GROUPS, COMBINE]
STEP = 4
NAN, NEG_INF = 0x7FC0, 0xFF80


def logits(rng: np.random.Generator, v: int, p: int) -> np.ndarray:
    """Row p's logits: ties everywhere on an odd row, a NaN first logit on row 2, and a group of NaNs and -inf alone."""
    x = (rng.standard_normal(v) * 3).astype(np.float32)
    x = (np.round(x).astype(np.float32) if p % 2 else x).view(np.uint32) >> 16
    x[G:2 * G] = [NAN, NEG_INF] * (G // 2)
    if p == 2:
        x[0] = NAN
    return x.astype(np.uint16)


class Served:
    """``n`` rows of ``spec`` (row p's logits are argument p, as `slices` reads it) and the store of their served tokens."""

    def __init__(self, spec: str = ROW, n: int = 1, seed: int = 7):
        v = int(spec_statics(spec)["V"])
        rng = np.random.default_rng(seed)
        self.x = [logits(rng, v, p) for p in range(n)]
        self.rows = [("r0", None, {"i": 9 + p, "spec": spec, "args": [[["x", p, 0, v]]]},
                      {"statics": {"V": v}, "op_path": "sampler.select", "members": [("token", 1, "i64")], "row": p, "step": STEP})
                     for p in range(n)]
        self.tok = np.asarray([token_select_first_max(x) for x in self.x], dtype=np.int64)

    def slices(self, _pi, _store, _weights, _rid, _arrive, g):
        """`serving_rows._slices`: an argument group's elements."""
        return [self.x[g[0][1]]]

    def words(self, rid, step, op, member, lo, hi):
        return self.tok[lo:hi], None


def members(s: Served, tmpls=TMPLS, cap: int = 100):
    return ST.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, dict.fromkeys(tmpls, cap)), tmpls, [s.rows] * len(tmpls))


def read(s: Served, ms, tmpls=TMPLS):
    SRW._slices = s.slices
    return SRW.read_all(ms, [s.rows] * len(tmpls), s, None)


@pytest.fixture(autouse=True)
def _slices(monkeypatch):
    """`serving_rows._slices` back after each test, which `read` replaces."""
    monkeypatch.setattr(SRW, "_slices", SRW._slices)


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "select-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def test_a_greedy_select_row_holds_its_group_units_and_combine():
    """The split a served window's consumer gives the row: the least row width with one, whole groups and rows within a
    proof block's rows; none for a vocabulary no width divides so, nor for one whose group unit has no row past a SHA-512
    block, and a template of the two families at other statics is the row it was (a combine of `TokenSelect_v4`'s chunks,
    which an interior piece reads whole).  At GLM-5.3's vocabulary the split is of eight-word rows, 121 group units of 1,280
    logits and the combine, and each unit's rows as the window wires them alone fill Flock's largest block
    (`interface.check_proof_block`), where its Definition's own three or four rows leave 2^22 and 2^26."""
    assert TM.select_split(V) == (W, C) and SRW.unit_templates(ROW) == ((ROW, 1), (GROUPS, V // C), (COMBINE, 1))
    for v in (16, 64, 100, 4096):
        assert TM.select_split(v) is None and SRW.unit_templates(f"TokenSelect_v1{{V={v}}}") == ((f"TokenSelect_v1{{V={v}}}", 1),)
    assert [SRW.template(t).layout for t in TMPLS] == ["stage", "stage"]
    assert [SRW.template(t).layout for t in ("TokenSelectCombine_v2{V=64,C=16}", f"TokenSelectCombine_v2{{V={SERVED_V},C=64}}",
                                             "TokenSelectGroups_v1{C=64,G=16}", "TokenSelectGroups_v1{C=1280,G=32}")] == ["row"] * 4
    m = SRW.unit_member(GROUPS, 1)
    assert m.rows == tuple((f"in0.{j}", W) for j in range(C // W)) and m.outs == (("o0", C // G), ("o1", 2 * C // G))
    assert TM.select_split(SERVED_V) == (8, 1280) == (8, TM.select_width(SERVED_V, 8, G))
    assert all(TM.select_width(SERVED_V, w, G) is None for w in (1, 2, 4))
    row, g, c = SRW.unit_templates(f"TokenSelect_v1{{V={SERVED_V}}}")
    assert (g, c) == (("TokenSelectGroups_v1{C=1280,G=16}", 121), (f"TokenSelectCombine_v2{{V={SERVED_V},C=16}}", 1))
    m = SRW.unit_member(g[0], 121)
    assert m.rows == tuple((f"in0.{j}", 8) for j in range(160)) and m.outs == (("o0", 80), ("o1", 160))
    m = SRW.unit_member(c[0], 1)
    assert m.rows == (("in0", 1), *((f"in1.{s}", 80) for s in range(121)), *((f"in2.{s}", 160) for s in range(121)))
    for t, uncut in ((g[0], 22), (c[0], 26)):
        tp = SRW.template(t)
        assert (IF.check_proof_block(tp.fn), IF.check_proof_block(tp.fn, SE.cuts(tp))) == (uncut, IF.K_MAX)


def test_the_units_read_the_row_s_logits_and_the_twin_s_groups_whose_token_is_the_served_row():
    """Each group unit reads its C logits in rows of the split's width and the combine logits[0] and every group unit's bests
    and indices; each unit's Definition on its rows (as Flock's stager parses them) computes its committed outputs, and the
    combine's is the served token: ties to the first, a NaN first logit the token, a group of NaNs and -inf never taken."""
    s = Served(n=3)
    ms = members(s)
    assert [(m.template, m.n) for m in ms] == [(GROUPS, 100), (GROUPS, 92), (COMBINE, 3)]
    got = read(s, ms)
    per = V // C
    for m, (g, entries) in zip(ms, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
        lo = min(x.base for x in ms if x.template == m.template)
        n = per if m.template == GROUPS else 1
        rows = [[lo + p * n, lo + (p + 1) * n, "r0", STEP, None, "sampler.select", "token", p, 9 + p] for p in range(3)]
        assert entries == [e for e in rows if e[0] < m.base + m.n and e[1] > m.base]
        for i in range(m.n):
            p, u = divmod(m.base + i - lo, n)
            if m.template == GROUPS:
                assert np.concatenate([g["rows"][f"in0.{j}"][i] for j in range(C // W)]).tolist() == s.x[p][u * C:(u + 1) * C].tolist()
            else:
                assert int(g["rows"]["in0"][i][0]) == int(s.x[p][0]) and int(g["outs"]["o0"][i].view("<u4")[0]) == int(s.tok[p])
    assert s.tok[2] == 0 and len(set(s.tok.tolist())) == 3
    t = GROUPS
    g = next(g for m, (g, _e) in zip(ms, got) if m.template == t)
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):     # an output its unit does not compute
        stager(t).reference_check(g["rows"], {**g["outs"], "o0": g["outs"]["o0"] ^ np.uint16(1)})


def test_the_combine_s_rows_are_wired_to_its_group_units_outputs():
    """Row s of the combine's bests and of its indices is group unit s's output row, of the same token's row, and every wired
    row holds its writer's words."""
    s = Served(n=3)
    ms = members(s)
    WI.validate(ms)
    per, base = V // C, min(m.base for m in ms if m.template == GROUPS)
    (c,) = [m for m in ms if m.template == COMBINE]
    assert all(m.wired == () for m in ms if m.template == GROUPS)
    assert set(c.wired) == {(f"in{g}.{u}", o, tuple(base + p * per + u for p in range(3)))
                            for g, o in ((1, "o0"), (2, "o1")) for u in range(per)}
    got = read(s, ms)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})


def test_a_row_whose_served_token_is_not_the_twin_s_is_refused():
    s = Served()
    s.tok[0] = (int(s.tok[0]) + 1) % V
    with pytest.raises(ValueError, match=re.escape(f"the twin's token {(int(s.tok[0]) - 1) % V} is not the row's served token")):
        read(s, members(s))


def test_a_partition_without_the_group_units_template_is_refused():
    s = Served()
    with pytest.raises(ValueError, match=re.escape("row 9's best and idx are the outputs of its group units, whose template the "
                                                   "partition does not name")):
        members(s, [COMBINE])
    assert all(m.wired == () for m in members(s, [GROUPS]))


def test_a_row_at_the_served_vocabulary_reads_and_wires_its_122_units():
    """One row of GLM-5.3's vocabulary: 121 group units of 160 eight-word rows and the combine of 121 rows of bests and of
    indices, every wired row its writer's output and the combine's token the served one."""
    spec = f"TokenSelect_v1{{V={SERVED_V}}}"
    s = Served(spec, n=1, seed=16)
    tmpls = [t for t, _n in SRW.unit_templates(spec)[1:]]
    ms = members(s, tmpls, 4096)
    WI.validate(ms)
    assert [m.n for m in ms] == [121, 1]
    got = read(s, ms, tmpls)
    words = {m.port(q): a for m, (g, _e) in zip(ms, got) for part_ in ("rows", "outs") for q, a in g[part_].items()}
    WI.salts(ms, words, {m.port(q): bytes(m.n * WI.SALT_BYTES) for m in ms for q, _w in m.outs})
    assert int(words[ms[1].port("o0")][0].view("<u4")[0]) == int(s.tok[0])
    x = np.concatenate([words[ms[0].port(f"in0.{j}")] for j in range(160)], axis=1)
    assert x.reshape(-1).tolist() == s.x[0].tolist()
