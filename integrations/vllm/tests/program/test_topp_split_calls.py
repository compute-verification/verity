"""`GumbelTopPTokenSelect_v3{V,S}` (`registry.topp_split_calls`): v2 as the split kernels' Calls.  The masked row and the token are
v2's on every row (ties, NaN, +-inf, uniform rows, top_p 0 and 1, a `splits` other than S), as the IR's evaluator computes both; every
piece is cut by `Q_word` with no violation and no redundant gate, far under the gate limit at the largest vocabulary; the rule binds
v3 only above `V2_MAX_V` and calls its pieces at the Program's root."""
from __future__ import annotations

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.types import Array
from verity_vllm.program.registry import sampling as SM
from verity_vllm.program.registry import topp_split_calls as TC
from verity_vllm.program.registry import topp_words as TW
from verity_vllm.program.registry.prims import F32
from verity_vllm.query import word as W

from tests.program.test_topp_words import _rows


def _masked(V, S):
    """(v2's masked row, v3's): TemperatureScale then `TopPMask_v2{V,S}`, and the chain up to its select's operand."""
    params, _ret = bind(TW.GumbelTopPTokenSelectV2, V=V, S=S).signature()
    sig = lambda st: (tuple(params), Array(V, F32))

    def v2(B, st, logits, top_p, seed, pos, temp, splits):
        return B.call(bind(TW.TopPMaskV2, V=V, S=S), B.call(bind(SM.TemperatureScale, V=V), logits, temp), top_p, splits)

    def v3(B, st, logits, top_p, seed, pos, temp, splits):
        call = lambda d, *a: a[0] if d.id.startswith("GumbelTopPSplitSelect") else B.call(d, *a)
        return TC.chain(call, V, S, logits, top_p, seed, pos, temp, splits)

    return [bind(CompositeDefinition(f"Masked{k}V{V}S{S}", 1, (), sig, body, register=False)) for k, body in (("V2", v2), ("V3", v3))]


def _args(x, p, i, S):
    xb = (x.astype(np.float32).view(np.uint32) >> 16).tolist()        # the row's bf16 words
    return xb, int(np.float32(p).view(np.uint32)), 1234 + i, 5 + i, 0x3F4CCCCD, S


@pytest.mark.parametrize("V,S,n", [(64, 4, 8), (100, 8, 8), pytest.param(1024, 32, 3, marks=pytest.mark.slow)])
def test_the_masked_row_is_v2s(V, S, n):
    """spt 4 with full slices (64/4, 1024/32), spt 1 with a short last slice (100/8)."""
    f2, f3 = _masked(V, S)
    for x, p in _rows(V, n, V + S):
        a = _args(x, p, 0, S)
        assert evaluate(f3, *a) == evaluate(f2, *a), (V, S, float(p))


@pytest.mark.parametrize("V,S", [(64, 4), (100, 8)])
def test_the_token_is_v2s(V, S):
    """Including top_p = 1 (the Gumbel-only row the Build still emits the select for) and a `splits` that is not S (no lane kept)."""
    f2, f3 = bind(TW.GumbelTopPTokenSelectV2, V=V, S=S), bind(TC.GumbelTopPTokenSelectV3, V=V, S=S)
    rng = np.random.default_rng(V)
    for i, p in enumerate((0.95, 0.5, 1.0, 0.0)):
        x = (rng.standard_normal(V) * 3).astype(np.float32)
        for spl in (S, S + 1):
            a = _args(x, p, i, spl)
            assert evaluate(f3, *a) == evaluate(f2, *a), (V, S, p, spl)


def _pieces(V, S):
    return [TC.KINDS["topp_split_temperature"](V=V), TC.KINDS["topp_split_select"](V=V), TC.KINDS["topp_row_stats"](S=S),
            TC.KINDS["topp_mask_setup"](S=S), *(TC.KINDS["topp_combine"](S=S, RND=r) for r in range(TC.R)),
            *(TC.KINDS[k](V=V, S=S, SLICE=i) for k in ("topp_split_stats", "topp_split_probs", "topp_split_step", "topp_split_mask")
              for i in {0, S - 1})]


@pytest.mark.parametrize("V,S", [(100, 8), pytest.param(256000, 32, marks=pytest.mark.slow)])
def test_q_word_cuts_every_piece(V, S):
    """Gemma-2's vocabulary is the largest cached: its largest piece (the select, 8 gates a lane) is a tenth of the limit."""
    for fn in _pieces(V, S):
        r = W.unit_rule(fn)
        assert not r["violations"] and r["redundant_gates"] == 0 and r["max_out_bits"] <= 32, (fn.id, r["violations"])
        assert fn.gates * 10 < W.MAX_GATES, (fn.id, fn.gates)


def test_v3_is_bound_above_v2s_gate_limit_only():
    assert not TC.uses_v3(27995, 32) and TC.uses_v3(27996, 32) and TC.uses_v3(128256, 32)
    assert not TC.uses_v3(128256, 16), "a split count v2's limit was not measured at keeps v2"


def test_the_rule_calls_v3s_pieces_at_the_root(monkeypatch):
    """At V = 64 with the threshold lowered to 16: no v2 select; per sampled step one chain, its last Call the sampling event
    `GumbelTopPSplitSelect_v1{V}` and its first `TemperatureScale_v1{V}` on the step's logits row, the row's Const32[S] read by the
    mask setup alone."""
    from tests.program.test_derive_stochastic import _base, _derive_constant, _tiny_config

    monkeypatch.setitem(TC.V2_MAX_V, 32, 16)
    C, LP, STEPS, S = _tiny_config(), 3, 1, 32
    rep = _derive_constant(C, LP, STEPS, S, temperature=0.8, top_p=0.95)
    assert rep.ok, rep.unsupported
    nodes = rep.flat_definition.body.nodes
    bases = [_base(n) for n in nodes]
    assert "GumbelTopPTokenSelect_v2" not in bases and "GumbelTopPTokenSelect_v3" not in bases
    sel = [k for k, b in enumerate(bases) if b == "GumbelTopPSplitSelect_v1"]
    assert len(sel) == STEPS + 1
    per_chain = 1 + S + 1 + S + TC.R * (S + 1) + 1 + S + 1
    for k in sel:
        first = k - per_chain + 1
        assert bases[first] == "TemperatureScale_v1" and nodes[k].fn.id == f"GumbelTopPSplitSelect_v1{{V={C['V']}}}"
        assert nodes[first - 1].fn.id == f"Const32[0x{S:08x}]_v1"
        readers = [bases[j] for j, n in enumerate(nodes) for a in n.args for r in a.refs if r[0] == "n" and r[1] == first - 1]
        assert readers == ["TopPMaskSetup_v1"], readers
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "GumbelTopPTokenSelect[vllm]_v3"]
    assert apps and all(a["emitted"].startswith(f"GumbelTopPTokenSelect_v3{{V={C['V']},S={S}}} as the Calls of its pieces") for a in apps)


def test_the_rule_keeps_v2_at_or_below_the_threshold():
    from tests.program.test_derive_stochastic import _base, _derive_constant, _tiny_config

    rep = _derive_constant(_tiny_config(), 3, 1, 32, temperature=0.8, top_p=0.95)
    assert rep.ok and "GumbelTopPSplitSelect_v1" not in {_base(n) for n in rep.flat_definition.body.nodes}
