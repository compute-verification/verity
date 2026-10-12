"""Query / sampler conformance (D82 §4, lane h4b): `verity_vllm.check.replay.vu_query` over the synthetic toy Program, whose VU
population is small enough to enumerate by brute force.

Configurations: (a) the default budget (every instance fits: instance granularity), (b) `max_arg_leaves=100` (the
configuration `test_tier_a_descends_into_child_units` uses: Embedding -> one gather gate, Gemm -> one
`GemmCoordinate`), (c) `max_arg_leaves=8` (two-level descent RoPE -> RoPEHead -> RopeOut, specializations without
any admissible unit, the `scan` gate-level fallback), plus depth cut-offs.  The brute-force reference below is a
literal, deterministic transcription of `Replayer._descend` at the gate level (`FunctionCall.args` / `.returns`);
`vu_query` decides the same tests from the reference structure.

Fixtures are copied from `test_replay_synthetic` (a toy log whose sampled tokens are the Program's own, snapshots
derived from a reference evaluation)."""
from __future__ import annotations

import math
from collections import Counter

import pytest

from verity_vllm.check import fold_compare as cmp
from verity_vllm.check.replay import replay, tier_a_pool
from verity_vllm.check.replay import vu_query as vq
from verity_vllm.observe.fold.fold import fold
from tests.observe import test_fold_synthetic as tfs
from tests.check.test_compare_synthetic import ToyWeights, _evaluate_program, _write_snapshots
from verity.ml.circuits.layout import FunctionCall

REQ = tfs.REQ
C = tfs.C

CONFIGS = [(4_000_000, 3), (100, 3), (100, 0), (8, 3), (8, 0), (20, 3), (40, 3), (1, 3)]


def _computed_tokens(r, weights) -> list[int]:
    t = _evaluate_program(r, weights)
    return [int(t[g]) for g in r.program.circuit.root_outputs()]


@pytest.fixture(scope="module")
def consistent_toy():
    """(FoldResult, ToyWeights, transcript, tokens) of a toy log whose sampled tokens are the Program's own."""
    if not tfs._real_memory_available():
        tfs.install_stub_memory()
    original = list(tfs.SAMPLED)
    try:
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        weights = ToyWeights(r.params[-1][1].leaves)
        t0 = _computed_tokens(r, weights)[0]
        tfs.SAMPLED[:] = [t0, original[1]]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t1 = _computed_tokens(r, weights)[1]
        tfs.SAMPLED[:] = [t0, t1]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t = _evaluate_program(r, weights)
        toks = [int(t[g]) for g in r.program.circuit.root_outputs()]
        assert toks == [t0, t1] and r.sampled == {REQ: [(0, t0), (1, t1)]}
        yield r, weights, t, toks
    finally:
        tfs.SAMPLED[:] = original


@pytest.fixture(scope="module")
def snapshots(consistent_toy, tmp_path_factory):
    r, weights, t, _ = consistent_toy
    d = tmp_path_factory.mktemp("snaps")
    _write_snapshots(r, t, d)
    return cmp.SnapshotStore(d)


# ---------------------------------------------------------------------------------------------------------
# the independent reference: `_descend` transcribed, deterministic and exhaustive, at the gate level
# ---------------------------------------------------------------------------------------------------------


def brute_force(rep: replay.Replayer, k: int, budget: int, depth: int) -> list[vq.VU]:
    """Every child unit of root instance `k` that `Replayer._descend` could return under SOME shuffle / member
    draw, in canonical order (body order, members ascending, depth-first); the instance itself when it fits."""
    if rep.arg_leaves(k) <= budget:
        return [vq.VU(k)]
    root = rep.call(k)
    rix = set(root.returns)
    rlo, rhi = root.lo, root.hi
    out: list[vq.VU] = []

    def rec(cur: FunctionCall, path: list, members: list, depth: int) -> None:
        if cur.is_primitive:
            return
        sc = cur.scope()
        for j, node in enumerate(sc.spec.body.nodes):
            nm = 1 if node.form in ("primitive", "call") else node.n
            for m in range(nm):
                child = FunctionCall(rep.circuit, sc, j, m)
                if not all(g in rix for g in child.returns):
                    continue
                leaves = sum(t.leaves for t in child.arg_types())
                if leaves <= budget:
                    if all(not (rlo <= g < rhi) for g in child.args):
                        out.append(vq.VU(k, tuple(path + [(j, nm)]), m, tuple(members)))
                    continue
                if depth > 0:
                    rec(child, path + [(j, nm)], members + [m], depth - 1)

    rec(root, [], [], depth)
    return out


def brute_force_all(r, budget: int, depth: int) -> list[vq.VU]:
    rep = replay.Replayer(r, None, None, chain=False, snapshots=False)
    return [v for k in range(len(r.body.nodes)) for v in brute_force(rep, k, budget, depth)]


# ---------------------------------------------------------------------------------------------------------
# reachability: every intended VU exactly once; index arithmetic reproduces the enumeration
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget,depth", CONFIGS)
def test_enumeration_equals_brute_force_exactly_once(consistent_toy, budget, depth):
    r, _, _, _ = consistent_toy
    mine = list(vq.enumerate_vus(r, max_arg_leaves=budget, depth=depth))
    ref = brute_force_all(r, budget, depth)
    assert mine == ref, (len(mine), len(ref), sorted(set(mine) ^ set(ref))[:5])
    assert len(set(mine)) == len(mine)                                        # no duplicates
    n = vq.vu_count(r, max_arg_leaves=budget, depth=depth)
    assert n == len(mine)
    assert [vq.vu_by_index(r, i, max_arg_leaves=budget, depth=depth) for i in range(n)] == mine
    ix = vq.index(r, max_arg_leaves=budget, depth=depth)
    assert [ix.index_of(v) for v in mine] == list(range(n))                   # the inverse
    assert all(v in ix for v in mine)
    with pytest.raises(IndexError):
        ix.by_index(n)


def test_population_sizes_against_hand_count(consistent_toy):
    """Counts computed from the toy's dimensions alone (no walk of the circuit)."""
    r, _, _, _ = consistent_toy
    by_spec = Counter(m["spec"].split("{")[0] for m in r.instances)
    n_emb, n_gemm_qkv = by_spec["Embedding_v1"], sum(1 for m in r.instances if m["spec"] == f"Gemm_v1{{K={C['H']},N={C['QKV']}}}")
    n_gemm_other = by_spec["Gemm_v1"] - n_gemm_qkv
    total = len(r.instances)
    # (a) default budget: every instance is one unit
    assert vq.vu_count(r, max_arg_leaves=4_000_000) == total
    # (b) budget 100: Embedding (1 + V*H = 257 leaves) -> H gathers; Gemm (K + K*N leaves) -> N coordinates
    expect = (total - n_emb - by_spec["Gemm_v1"]) + n_emb * C["H"] + n_gemm_qkv * C["QKV"] + n_gemm_other * C["H"]
    assert vq.vu_count(r, max_arg_leaves=100) == expect == 391
    s = vq.index(r, max_arg_leaves=100).summary()
    assert s["count"] == expect and s["oversize_instances"] == n_emb + by_spec["Gemm_v1"] and s["instances_without_units"] == 0
    assert s["specs"][f"Embedding_v1{{V={C['V']},H={C['H']}}}"] == {"instances": n_emb, "vus": n_emb * C["H"], "granularity": ["child:GatherBf16x16_v1"]}
    assert s["specs"][f"Gemm_v1{{K={C['H']},N={C['QKV']}}}"]["granularity"] == [f"child:GemmCoordinate_v1{{K={C['H']}}}"]
    # (c) budget 8: RoPE -> RoPEHead (16 leaves, too big) -> RopeOut / RopeOutAdd members (4 leaves): 2 nodes x D/2 per head;
    #     RMSNormFusedCuda -> the residual-add batch (N units); SiluMul -> N units; Embedding / Gemm / Attention / norms: none
    D, NH, KVH = C["D"], C["NH"], C["KVH"]
    rope = sum(NH if m["spec"].startswith("RoPE_v1{NHEADS=2") else KVH for m in r.instances if m["spec"].startswith("RoPE")) * 2 * (D // 2)
    expect8 = rope + by_spec["RMSNormFusedCuda_v2"] * C["H"] + by_spec["SiluMul_v1"] * C["I"]
    assert vq.vu_count(r, max_arg_leaves=8) == expect8 == 216
    s8 = vq.index(r, max_arg_leaves=8).summary()
    none = {k for k, v in s8["specs"].items() if v["granularity"] == ["none"]}
    assert {k.split("{")[0] for k in none} == {"Embedding_v1", "RMSNormTriton_v1", "Gemm_v1", "Attention_v3", "TokenSelect_v1"}
    assert s8["gate_level_checks"] > 0                                           # the TokenSelect scan took the gate-level path
    # depth cut-off: depth=0 keeps level-1 units only (RoPE's level-2 units disappear)
    assert vq.vu_count(r, max_arg_leaves=8, depth=0) == expect8 - rope == 144
    # an impossible budget: an empty population, and the sampler refuses
    assert vq.vu_count(r, max_arg_leaves=1) == 0
    with pytest.raises(ValueError):
        vq.production_sample(r, seed=0, n=1, max_arg_leaves=1)


def test_per_instance_counts_match_independent_walk(consistent_toy):
    r, _, _, _ = consistent_toy
    rep = replay.Replayer(r, None, None, chain=False, snapshots=False)
    for budget, depth in CONFIGS:
        ix = vq.index(r, max_arg_leaves=budget, depth=depth)
        walked = [len(brute_force(rep, k, budget, depth)) for k in range(len(r.body.nodes))]
        assert ix.per_instance.tolist() == walked, (budget, depth)
        assert ix.count == sum(walked)
        for k in range(len(r.body.nodes)):
            assert list(ix.of_instance(k)) == brute_force(rep, k, budget, depth)


# ---------------------------------------------------------------------------------------------------------
# child descent: operands / returns map to the right slices; bit-exact; members tile the parent's returns
# ---------------------------------------------------------------------------------------------------------


def test_child_units_are_bit_exact_and_tile_the_parent_returns(consistent_toy, snapshots):
    r, weights, t, _ = consistent_toy
    rep = replay.Replayer(r, snapshots, weights, chain=False, snapshots=True)
    ix = vq.index(r, max_arg_leaves=100)
    n_child = 0
    for k in range(len(r.body.nodes)):
        parent = ix.call(k)
        p_args, p_rets = set(parent.args), list(parent.returns)
        units = list(ix.of_instance(k))
        if ix.fits[k]:
            assert units == [vq.VU(k)]
            assert vq.operands_of(r, units[0]) == tuple(parent.args) and vq.returns_of(r, units[0]) == tuple(parent.returns)
            e = vq.evaluate_vu(r, snapshots, weights, units[0], rep=rep)
            assert e["status"] == "pass" and e["granularity"] == "instance" and e["vu"] == units[0].to_json(), e
            continue
        covered: Counter = Counter()
        for vu in units:
            ops, rets = vq.operands_of(r, vu), vq.returns_of(r, vu)
            assert set(ops) <= p_args and set(rets) <= set(p_rets), vu
            assert not any(parent.lo <= g < parent.hi for g in ops)
            covered.update(rets)
            # the unit's expected words, taken from the parent's snapshot by return position, equal the transcript
            e = vq.evaluate_vu(r, snapshots, weights, vu, rep=rep)
            assert e["status"] == "pass" and e["n_returns"] == len(rets) and e["n_args"] == len(ops), e
            assert e["granularity"].startswith("child:") and e["child"]["gates"] == [ix.vu_call(vu).lo, ix.vu_call(vu).hi]
            # the parent's snapshot at the unit's return positions is the transcript at the unit's return gates
            rix = rep.ret_index(k)
            assert [int(rep.output_words(k)[rix[g]]) for g in rets] == [t[g] for g in rets]
            n_child += 1
        # every return gate of the parent produced by exactly one member: no gaps, no overlaps
        assert covered == Counter(p_rets), (k, r.instances[k]["spec"])
        assert set(covered.values()) == {1}
    assert n_child == 368


def test_two_level_units_are_bit_exact_and_disjoint(consistent_toy, snapshots):
    """Budget 8: RoPE units two levels down (`members != ()`), residual-add / SiluMul members one level down."""
    r, weights, _, _ = consistent_toy
    rep = replay.Replayer(r, snapshots, weights, chain=False, snapshots=True)
    ix = vq.index(r, max_arg_leaves=8)
    deep = [vu for vu in ix if len(vu.path) == 2]
    assert deep and all(len(vu.members) == 1 for vu in deep)
    assert {ix.granularity(vu) for vu in deep} == {"child:RopeOut_v1", "child:RopeOutAdd_v1"}
    assert {ix.labels(vu)[0].split("#")[0] for vu in deep} == {"RoPEHead_v1{D=8}"}
    for k in range(len(r.body.nodes)):
        units = list(ix.of_instance(k))
        if not units:
            continue
        parent = ix.call(k)
        covered: Counter = Counter()
        for vu in units:
            e = vq.evaluate_vu(r, snapshots, weights, vu, rep=rep)
            assert e["status"] == "pass", e
            covered.update(vq.returns_of(r, vu))
        assert set(covered) <= set(parent.returns) and set(covered.values()) == {1}
        if r.instances[k]["spec"].startswith(("RoPE", "SiluMul")):
            assert covered == Counter(parent.returns)                        # these tile the parent completely
        else:
            assert r.instances[k]["spec"].startswith("RMSNormFusedCuda") and len(covered) == C["H"]   # the residual half only


def test_evaluate_vu_detects_a_corrupted_parent_snapshot(consistent_toy, tmp_path):
    r, weights, t, _ = consistent_toy
    bad = next(m for m in r.instances if m["spec"].startswith("Gemm"))["node"]
    _write_snapshots(r, t, tmp_path, corrupt_node=bad)
    store = cmp.SnapshotStore(tmp_path)
    rep = replay.Replayer(r, store, weights, chain=False, snapshots=True)
    ix = vq.index(r, max_arg_leaves=100)
    res = [vq.evaluate_vu(r, store, weights, vu, rep=rep)["status"] for vu in ix.of_instance(bad)]
    assert res and set(res) == {"fail"}                                       # every coordinate's expected word was flipped
    other = next(k for k in range(len(r.body.nodes)) if not ix.fits[k] and k != bad)
    assert {vq.evaluate_vu(r, store, weights, vu, rep=rep)["status"] for vu in ix.of_instance(other)} == {"pass"}


# ---------------------------------------------------------------------------------------------------------
# production sampler: uniform over VUs, deterministic per seed
# ---------------------------------------------------------------------------------------------------------


def test_production_sample_is_uniform_over_vus_and_deterministic(consistent_toy):
    r, _, _, _ = consistent_toy
    budget = 100
    ix = vq.index(r, max_arg_leaves=budget)
    pop = list(ix)
    N = len(pop)
    assert N == 391
    seeds, per_seed = 40, 2000
    counts: Counter = Counter()
    for s in range(seeds):
        a = vq.production_sample(r, seed=s, n=per_seed, max_arg_leaves=budget)
        assert a == vq.production_sample(r, seed=s, n=per_seed, max_arg_leaves=budget)      # deterministic per seed
        assert len(a) == per_seed and all(v in ix for v in a)
        counts.update(a)
    assert vq.production_sample(r, seed=1, n=50, max_arg_leaves=budget) != vq.production_sample(r, seed=2, n=50, max_arg_leaves=budget)
    total = seeds * per_seed
    expect = total / N
    chi2 = sum((counts.get(v, 0) - expect) ** 2 / expect for v in pop)
    df = N - 1
    # chi-square with df=390: mean 390, sd sqrt(780) ~ 28; 5 sd above the mean (p ~ 1e-6) is the failure line
    assert chi2 < df + 5 * math.sqrt(2 * df), (chi2, df)
    assert all(counts.get(v, 0) > 0 for v in pop)
    # the measure is uniform over UNITS: an oversize Gemm instance (N coordinates) carries N times the mass of a
    # one-unit instance, and the instance-level share of the sample is (#one-unit instances) / N
    per_instance = Counter(v.node for v in counts.elements())
    heavy = next(k for k in range(len(r.body.nodes)) if r.instances[k]["spec"] == f"Gemm_v1{{K={C['H']},N={C['QKV']}}}")
    light = next(k for k in range(len(r.body.nodes)) if ix.fits[k])
    ratio = per_instance[heavy] / per_instance[light]
    assert C["QKV"] * 0.6 < ratio < C["QKV"] * 1.5, ratio
    share = sum(1 for v in counts.elements() if v.is_instance) / total
    assert abs(share - sum(ix.fits) / N) < 0.02


# ---------------------------------------------------------------------------------------------------------
# tier A (the CI sampler) draws from this population and covers every stratum
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [4_000_000, 100])
def test_tier_a_evaluations_are_population_members_and_cover_strata(consistent_toy, snapshots, budget):
    r, weights, _, _ = consistent_toy
    a = replay.tier_a(r, snapshots, weights, k=2, seed=0, max_arg_leaves=budget)
    assert a["status"] == "pass", a
    ix = vq.index(r, max_arg_leaves=budget)
    by_region = {}
    for vu in ix:
        if not vu.is_instance:
            call = ix.vu_call(vu)
            key = (vu.node, call.lo, call.hi)
            assert key not in by_region
            by_region[key] = vu
    seen = []
    for e in a["instances"]:
        assert e["status"] == "pass"
        if e["granularity"] == "instance":
            vu = vq.VU(e["node"])
            assert vu in ix and ix.fits[e["node"]]
        else:
            lo, hi = e["child"]["gates"]
            vu = by_region.get((e["node"], lo, hi))
            assert vu is not None, e
            assert ix.granularity(vu) == e["granularity"] and ix.labels(vu) == e["child"]["path"]
            assert not ix.fits[e["node"]]
        seen.append(vu)
    assert len(seen) == len(a["instances"]) and all(v in ix for v in seen)
    # every (specialization, regime) stratum present at a snapshot step was sampled and passed
    present = {(m["spec"], tier_a_pool._regime(m)) for m in r.instances if m["step"] in snapshots.steps}
    sampled = {(e["spec"], e["regime"]) for e in a["instances"] if e["status"] == "pass"}
    assert sampled == present and a["strata"] == len(present)
    # and the sampled units of one specialization are distinct population members when k > 1 units exist
    if budget == 100:
        kids = [v for v in seen if not v.is_instance]
        assert kids and len(set(kids)) == len(kids)


# ---------------------------------------------------------------------------------------------------------
# the unit record
# ---------------------------------------------------------------------------------------------------------


def test_vu_record_roundtrip_and_validation(consistent_toy):
    r, _, _, _ = consistent_toy
    ix = vq.index(r, max_arg_leaves=8)
    for vu in list(ix)[:20] + [vq.VU(3)]:
        assert vq.VU.from_json(vu.to_json()) == vu
    with pytest.raises(ValueError):
        vq.VU(0, ((0, 16), (1, 4)), 2)                        # two levels need one upper member
    with pytest.raises(ValueError):
        vq.VU(0, (), 1)                                        # instance granularity has member 0
    with pytest.raises(ValueError):
        ix.vu_call(vq.VU(0, ((0, 5),), 1))                      # wrong member count for the node
    with pytest.raises(ValueError):
        ix.index_of(vq.VU(0, ((0, 16),), 1))                    # an Embedding gather does not fit budget 8: not a member
    assert vq.VU(0) not in ix                                   # instance 0 is oversize under budget 8
    assert vq.VU(0) in vq.index(r, max_arg_leaves=4_000_000)
