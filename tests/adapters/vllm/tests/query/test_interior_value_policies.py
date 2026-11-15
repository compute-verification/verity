"""The four policies that require values interior to a Call (`norm_scales`, `router_softmax`, `scale_products`, `vocab_range`):
what each one's `kernel_rows` and `slots` refuse, and the header each returns when the policy is off."""
from __future__ import annotations

import types

import pytest

from verity_vllm.query import norm_scales as NS
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import scale_products as SP
from verity_vllm.query import vocab_range as VR

#: the module the policies cut a Call's graph with, as one of them names it: a stand-in put in it is the one all three read
W = SP.W
SITE = "model.layers.1.mlp"
SHARED = "ScaledMmFp8BlockSharedScale_v1"
ROUTER = {"E": 8, "TOPK": 2, "VPT": 8}

#: the policy, then two Calls (family, statics) that one site may not run in one step, then what the refusal says of them
TWO_AT_ONE_SITE = [
    (NS, ("RMSNormTriton_v1", {}), ("RMSNormFusedCuda_v2", {}),
     f"norm scales: module impl of {SITE} step 3 runs two norm kernels (RMSNormTriton_v1, RMSNormFusedCuda_v2)"),
    (RS, ("MoeRouterTopKOrdered_v1", ROUTER), ("MoeRouterTopKOrderedNorm_v1", ROUTER),
     f"router softmax: module impl of {SITE} step 3 runs two routers (MoeRouterTopKOrdered_v1, MoeRouterTopKOrderedNorm_v1)"),
    (RS, ("MoeRouterTopKOrdered_v1", ROUTER), ("MoeRouterTopKOrdered_v1", dict(ROUTER, TOPK=4)),
     f"router softmax: module impl of {SITE} step 3 runs two routers (MoeRouterTopKOrdered_v1, MoeRouterTopKOrdered_v1)"),
    (SP, (SHARED, {"K": 256, "N": 256, "G": 128}), (SHARED, {"K": 256, "N": 320, "G": 128}),
     (f"scale products: module impl of {SITE} step 3 runs two block geometries ({{'spec': '{SHARED}', 'n_blocks': 2, 'k_blocks': 2, "
      f"'rows': 1}}, {SHARED}{{K=256,N=320,G=128}})")),
    (VR, ("EmbeddingShard_v1", {"VS": 8, "START": 0}), ("EmbeddingShard_v1", {"VS": 8, "START": 8}),
     f"vocab range: module impl of {SITE} step 3 runs two shards ({{'VS': 8, 'START': 0, 'rows': 1}}, {{'VS': 8, 'START': 8}})"),
]


def _population(calls, step_of):
    made = [types.SimpleNamespace(id=i, family=family, statics=statics, name=f"{SITE}/node_{i}",
                                  definition=f"{family}{{{','.join(f'{k}={v}' for k, v in statics.items())}}}")
            for i, (family, statics) in enumerate(calls)]
    return types.SimpleNamespace(P=types.SimpleNamespace(calls=made), step_of=step_of, program_path=lambda i: SITE)


def _impl(path):
    return f"impl of {path}"


@pytest.mark.parametrize("policy,first,second,said", TWO_AT_ONE_SITE, ids=[f"{t[0].FAMILY}: {t[2][0]}" for t in TWO_AT_ONE_SITE])
def test_two_calls_that_disagree_at_one_site_and_step_are_refused_and_at_two_steps_are_two_identities(policy, first, second, said):
    added = []

    def add(*a, **kw):
        added.append(a)

    with pytest.raises(ValueError) as e:
        policy.kernel_rows(_population([first, second], lambda i: 3), add, _impl, {})
    assert str(e.value) == said and added == []
    policy.kernel_rows(_population([first, second], lambda i: 3 + i), add, _impl, {})
    assert [(a[0], a[1]) for a in added] == [(policy.FAMILY, 3), (policy.FAMILY, 4)]


@pytest.mark.parametrize("policy", [NS, RS, VR], ids=lambda p: p.FAMILY)
def test_the_header_of_a_policy_that_is_off_is_the_query_header_itself(policy):
    q = {"query_text": "the query", "required_families": ["a family"]}
    assert policy.header(q, None) is q and q == {"query_text": "the query", "required_families": ["a family"]}


#: `slots` (below its cache) and its arguments, and how its refusals name the Call
SLOTS = [
    (RS, (8, 2, 8, True), "router softmax: the partition of E=8 TOPK=2 renormalize=True"),
    (SP, (256, 256, 128), "scale products: the partition of K=256 N=256 G=128"),
    (VR, (8, 32, 0), "vocab range: the partition of EmbeddingShard_v1{VS=8,H=32,START=0}"),
]


@pytest.mark.parametrize("policy,args,named", SLOTS, ids=[t[0].FAMILY for t in SLOTS])
def test_slots_refuses_a_partition_whose_cut_fails_or_that_recomputes_a_gate(policy, args, named, monkeypatch):
    """The cut's own verdict counts a gate computed twice only across two units; `slots` refuses one inside a unit too."""
    slots = policy.slots.__wrapped__
    whole = slots(*args)
    assert whole == policy.slots(*args) and len(whole) > 0
    units, graph = W.units, W.Graph

    def a_cut_that_fails(*a, **kw):
        return dict(units(*a, **kw), cut=types.SimpleNamespace(ok=False, codes=("a code of the cut",)))

    def a_graph_that_recomputes(fn):
        g = graph(fn)
        g.recomputed = [(0, 0), (1, 1)]
        return g

    with monkeypatch.context() as m:
        m.setattr(W, "units", a_cut_that_fails)
        with pytest.raises(ValueError) as e:
            slots(*args)
        assert str(e.value) == f"{named} fails ('a code of the cut',) with 0 recomputed gates"
    with monkeypatch.context() as m:
        m.setattr(W, "Graph", a_graph_that_recomputes)
        with pytest.raises(ValueError) as e:
            slots(*args)
        assert str(e.value) == f"{named} fails () with 2 recomputed gates"
    assert slots(*args) == whole


OTHER_LAYOUTS = [
    ("one value fewer", (("local", "I32Add_v1"), ("ge", "I32Le_v1")), "['I32Add_v1', 'I32Le_v1']"),
    ("another primitive", (("local", "I32Add_v1"), ("ge", "I32Le_v1"), ("le", "I32Lt_v1")), "['I32Add_v1', 'I32Le_v1', 'I32Lt_v1']"),
    ("another order", (("ge", "I32Le_v1"), ("local", "I32Add_v1"), ("le", "I32Le_v1")), "['I32Le_v1', 'I32Add_v1', 'I32Le_v1']"),
]


@pytest.mark.parametrize("layout,acquired", [t[1:] for t in OTHER_LAYOUTS], ids=[t[0] for t in OTHER_LAYOUTS])
def test_slots_refuses_a_shard_that_commits_other_values_than_the_layout(layout, acquired, monkeypatch):
    monkeypatch.setattr(VR, "LAYOUT", layout)
    with pytest.raises(ValueError) as e:
        VR.slots.__wrapped__(8, 32, 0)
    assert str(e.value) == ("vocab range: EmbeddingShard_v1{VS=8,H=32,START=0} commits ['I32Add_v1', 'I32Le_v1', 'I32Le_v1'], "
                            f"the source acquires {acquired}")


def test_slots_refuses_a_gemm_that_commits_another_count_of_products_than_its_blocks(monkeypatch):
    assert SP.words({"K": 256, "N": 256, "G": 128}) == 4
    monkeypatch.setattr(SP, "words", lambda st: 5)
    with pytest.raises(ValueError) as e:
        SP.slots.__wrapped__(256, 256, 128)
    assert str(e.value) == "scale products: K=256 N=256 G=128 commits 4 interior values (['F32Mul_v1']), the source acquires 5 F32Mul_v1"


def test_slots_refuses_a_router_whose_committed_values_are_not_the_layouts_and_names_the_first_word_that_differs(monkeypatch):
    probs, select = "MoeRouterProbs_v1", "SelectF32_v1"
    with monkeypatch.context() as m:
        m.setattr(RS, "_CLASSES", tuple((name, (probs, "F32Exp_v1") if name == "ex" else cls) for name, cls in RS._CLASSES))
        with pytest.raises(ValueError) as e:
            RS.slots.__wrapped__(8, 2, 8, False)
        assert str(e.value) == ("router softmax: E=8 TOPK=2 renormalize=False commits 18 interior values, the tap writes 18; "
                                f"first difference at word 1: [('{probs}', '{select}')] vs [('{probs}', 'F32Exp_v1')]")
    layout = RS.layout

    def one_probability_fewer(E, TOPK, renormalize):
        return tuple((name, n - 1 if name == "p" else n) for name, n in layout(E, TOPK, renormalize))

    monkeypatch.setattr(RS, "layout", one_probability_fewer)
    with pytest.raises(ValueError) as e:
        RS.slots.__wrapped__(8, 2, 8, False)
    assert str(e.value) == ("router softmax: E=8 TOPK=2 renormalize=False commits 18 interior values, the tap writes 17; "
                            f"first difference at word 17: [('{probs}', '{select}')] vs []")
