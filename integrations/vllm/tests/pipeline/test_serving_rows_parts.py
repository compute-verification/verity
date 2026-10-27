"""The fused all-reduce + residual add + RMSNorm rows split into their parts (`pipeline.serving_rows`' parts, ``PARTS``): an
``AllReduceRmsNormMnnvl_v1`` row holds one ``AllReduceResidualMnnvl_v1`` at its width and W / w of it at each halving w (as
``COLS``), and one ``RmsNormMnnvl_v1``, each at the row's statics it has.  Each part reads the row's operands and committed outputs
its entry names, with the types the row gives them; together the parts cover the row's two outputs once each (the residual
part's blocks its r in order, the norm part its normed row, each under one index entry naming that output) and compute them: each
part's Definition on its units' committed rows is their outputs.  A row is read by one member per part, never by two members of
one part, nor by the row and a part."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch

from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST
from verity.ml.definitions.registry import boolean_sigma as SG
from verity.ml.definitions.registry import deepgemm as DG
from verity.ml.definitions.registry import tp_allreduce as TA

WORLD, N = 4, 256
STATICS = {"WORLD": WORLD, "N": N, "SHOT": 1, "CLUSTER": 1, "EPS": 1e-5, "FMA": False}
OP = "model.layers.6.post_attention_layernorm"
MEMBERS = [("normed", N, "bf16"), ("residual", N, "bf16")]


def fused(boolean: bool = False):
    return bind(SG.AllReduceRmsNormMnnvl if boolean else TA.AllReduceRmsNormMnnvl, **STATICS)


def resid(w: int, boolean: bool = False):
    return bind(SG.AllReduceResidualMnnvl if boolean else TA.AllReduceResidualMnnvl, WORLD=WORLD, N=w, SHOT=1)


def norm(boolean: bool = False):
    return bind(SG.RmsNormMnnvl if boolean else TA.RmsNormMnnvl, N=N, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False)


def ev(fn, *args: np.ndarray) -> list[np.ndarray]:
    out = evaluate_batch(fn, *(np.asarray(a).astype(np.uint64) for a in args))
    return [np.asarray(a).reshape(len(args[0]), -1).astype(np.uint16) for a in (out if isinstance(out, (tuple, list)) else [out])]


def bf16s(rng, n: int, scale: float) -> np.ndarray:
    v = ((rng.standard_normal(n) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    v[:3] = [0x8000, 0x0001, 0x0000]
    return v


@pytest.fixture
def row(monkeypatch):
    """A synthetic row (its partials and residual committed members, its weight a registered weight) and its outputs by the fused
    Definition, behind a ProgramIndex stand-in that holds it as row 9."""
    def make(short: int = 0):
        rng = np.random.default_rng(1)
        vals = {"parts": bf16s(rng, WORLD * N, 0.5), "res": bf16s(rng, N, 2.0)}
        w = bf16s(rng, N, 0.1)
        (both,) = ev(fused(), vals["parts"][None], vals["res"][None], w[None])
        normed, r = both[0, :N], both[0, N:]
        vals |= {"normed": normed, "residual": r}
        spec = descriptor_id(fused())
        rr = {"i": 9, "spec": spec, "args": [[["n", 0, 0, WORLD * N - short]], [["n", 1, 0, N]], [["p", 0, 0, N]]]}
        d = {"statics": dict(STATICS), "op_path": OP, "members": MEMBERS, "row": 0, "step": 16}

        class Index:
            pnames = ["weights"]
            by_i = {9: rr}

            def __init__(self, *_a, **_k):
                pass

            def describe(self, _r):
                return d

            def member_at(self, a, lo, hi):
                nm = ("parts", "res")[a]
                return {"members": [(nm, vals[nm].size, "bf16")], "step": 16, "op_path": f"m.{nm}", "row": 0}, nm, lo, hi, None

            def weight_slice(self, lo, hi):
                return "w", lo, hi, None

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return vals[member][lo:hi], None

        monkeypatch.setattr(SRW, "ProgramIndex", Index)
        monkeypatch.setattr(SRW, "index_of", lambda inst, _reuse, **kw: Index(inst, **kw), raising=False)
        return {"spec": spec, "store": Store(), "weights": lambda f, lo, hi: w[lo:hi], "w": w, **vals}
    return make


def test_a_row_holds_each_part_at_its_statics_and_the_residual_at_every_halving():
    spec = descriptor_id(fused())
    assert SRW.unit_templates(spec) == ((spec, 1), (descriptor_id(resid(N)), 1),
                                        *((descriptor_id(resid(N // k)), k) for k in (2, 4, 8, 16, 32, 64, 128, 256)),
                                        (descriptor_id(norm()), 1))
    t = SRW.template(descriptor_id(resid(N // 2, boolean=True)))
    assert (t.layout, t.word, t.packed) == ("cols", descriptor_id(resid(N // 2)), None)
    t = SRW.template(descriptor_id(norm(boolean=True)))
    assert (t.layout, t.word, t.packed) == ("part", descriptor_id(norm()), None)
    assert SRW.part(descriptor_id(norm()), spec) == SRW.PARTS["AllReduceRmsNormMnnvl_v1"][1]
    assert SRW.part(descriptor_id(resid(N)), descriptor_id(resid(N))) is None


@pytest.mark.parametrize("family", list(SRW.PARTS))
def test_each_part_reads_what_the_row_gives_it_and_the_parts_return_every_output_once(family):
    whole = {"AllReduceRmsNormMnnvl_v1": fused, "DeepGemmFp8Linear_v1": lambda: bind(DG.DeepGemmFp8Linear, K=256, N=200)}[family]()
    params, ret = whole.signature()
    outs = ret.fields
    assert sorted(j for p in SRW.PARTS[family] for j in p[2]) == list(range(len(outs)))
    for fam, reads, returns in SRW.PARTS[family]:
        pp, pr = SRW._bind(descriptor_id(whole), fam, **ST.PART_STATICS.get(family, {})).signature()
        assert [ty for _n, ty in pp] == [(params[g][1] if src == "arg" else outs[g]) for src, g in reads], fam
        assert list(getattr(pr, "fields", None) or (pr,)) == [outs[j] for j in returns], fam


@pytest.mark.parametrize("w", [N, N // 2, N // 8])
def test_the_parts_cover_the_row_s_outputs_and_compute_them(row, w):
    x = row()
    tmpls = [descriptor_id(resid(w, boolean=True)), descriptor_id(norm(boolean=True))]
    rows, _defs, counts = SRW.member_rows({"r0": {}}, tmpls)
    nh = N // w
    assert counts == [nh, 1] and [[(rid, r["i"]) for rid, _pi, r, _d in rs] for rs in rows] == [[("r0", 9)]] * 2
    mr, mn = SRW.members(tmpls, rows, counts)
    assert (mr.n, mr.base, mr.rows, mr.outs) == (nh, 0, (("in0", WORLD * w), ("in1", w)), (("out.residual", w),))
    assert (mn.n, mn.base, mn.rows, mn.outs) == (1, nh, (("in0", N), ("in1", N)), (("out.normed", N),))
    dr, er = SRW.read(mr, rows[0], x["store"], x["weights"])
    dn, en = SRW.read(mn, rows[1], x["store"], x["weights"])
    assert er == [[0, nh, "r0", 16, 6, OP, "residual", 0, 9]] and en == [[nh, nh + 1, "r0", 16, 6, OP, "normed", 0, 9]]
    P = x["parts"].reshape(WORLD, N)
    for j in range(nh):
        assert dr["rows"]["in0"][j].tolist() == P[:, j * w:(j + 1) * w].reshape(-1).tolist()
        assert dr["rows"]["in1"][j].tolist() == x["res"][j * w:(j + 1) * w].tolist()
    assert dr["outs"]["out.residual"].reshape(-1).tolist() == x["residual"].tolist()
    assert dn["rows"]["in0"].reshape(-1).tolist() == x["residual"].tolist()
    assert dn["rows"]["in1"].reshape(-1).tolist() == x["w"].tolist()
    assert dn["outs"]["out.normed"].reshape(-1).tolist() == x["normed"].tolist()
    assert ev(resid(w), dr["rows"]["in0"], dr["rows"]["in1"])[0].tolist() == dr["outs"]["out.residual"].tolist()
    assert ev(norm(), dn["rows"]["in0"], dn["rows"]["in1"])[0].tolist() == dn["outs"]["out.normed"].tolist()


@pytest.mark.parametrize("tmpls", [lambda: [resid(N // 2, True), resid(N // 4, True)], lambda: [fused(True), norm(True)]],
                         ids=["two-of-one-part", "the-row-and-a-part"])
def test_a_row_read_by_two_members_that_are_not_its_parts_is_refused(row, tmpls):
    row()
    with pytest.raises(ValueError, match=r"row 9 .* one member reads a row, or one member per part of it \(PARTS\)"):
        SRW.member_rows({"r0": {}}, [descriptor_id(t) for t in tmpls()])


def test_a_row_whose_partials_are_not_its_statics_is_refused(row):
    x = row(short=2)
    tmpls = [descriptor_id(resid(N // 2, boolean=True))]
    rows, _defs, counts = SRW.member_rows({"r0": {}}, tmpls)
    (m,) = SRW.members(tmpls, rows, counts)
    with pytest.raises(ValueError, match=rf"row 9 \({OP}\): inputs and outputs of \[{WORLD * N - 2}, {N}, {N}\] words are not 2"):
        SRW.read(m, rows[0], x["store"], x["weights"])
