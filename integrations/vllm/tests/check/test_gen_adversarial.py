"""Lane ov-adversarial (Agent 10, swarm 2026-09-17): minimal failing tests for the CHECKER HOLES found by mutating the
fold's OUTPUT through the patterns — the lever an untrusted adapter author holds — and running the sealed checkers
(`provenance`, `replay.tier_a`, `replay.tier_chain`, `vu_query`, `census`, `gates.quarantine_specs`) over the trusted
record (the synthetic `consistent_toy`: log + snapshots derived from the GENUINE Program + toy weights).

Every hole test is `xfail(strict=True)`: it fails today (that is the hole) and XPASSes — the suite reports it — when the
integrator closes it.  The premises (what the harness relies on) and the deterministic / probabilistic rejections in
the report table are asserted positively so the report stays true.  Harness: `tests/check/adversarial.py`;
report: `out/gen/lanes/ov-adversarial/REPORT.md`.
"""
from __future__ import annotations

import pytest

from verity_vllm.check import census, gates, operand_provenance as provenance, replay
from verity_vllm.query import vu_query as vq
from tests.check import adversarial as G
from verity_vllm.observe.fold.resolver import Compute, Transparent, spec_module

K_ALL = 8   # >= every toy stratum: tier a evaluates EVERY instance (no sampling luck in the assertions)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    rec = G.ToyRecord(tmp_path_factory.mktemp("adv_snaps"))
    try:
        rec.bake()
        yield rec
    finally:
        rec.restore()


@pytest.fixture(scope="module")
def toy_eq(tmp_path_factory):
    """The toy with a REPEATED prompt token: two different Embedding instances with byte-identical outputs."""
    rec = G.ToyRecord(tmp_path_factory.mktemp("adv_snaps_eq"), prompt=[5, 5])
    try:
        yield rec
    finally:
        rec.restore()


def _mutated(rec, mid: str, **kw):
    mut = rec.catalogue()[mid]
    prof, state = G.mutate_profile(rec.profile, mut.target, mut.rewrite, weight_field=mut.weight_field)
    r = rec.fold(prof)
    assert G.hits(state) > 0 or mut.weight_field is not None, f"{mid}: the mutation never fired"
    return r, mut


def _checkers(rec, r, *, k=K_ALL, seed=0, max_arg_leaves=4_000_000):
    return {"provenance": provenance.check(r, None),
            "tier_a": replay.tier_a(r, rec.store, rec.weights, k=k, seed=seed, max_arg_leaves=max_arg_leaves),
            "tier_chain": replay.tier_chain(r, rec.store, rec.weights, n=3, seed=seed, max_arg_leaves=max_arg_leaves)}


# ---------------------------------------------------------------------------------------------------------
# premises
# ---------------------------------------------------------------------------------------------------------


def test_premise_genuine_toy_passes_every_checker(toy):
    c = _checkers(toy, toy.genuine)
    assert c["provenance"]["status"] == "pass" and c["tier_a"]["status"] == "pass" and c["tier_chain"]["status"] == "pass"


def test_premise_mutating_pattern_is_an_ordinary_pattern(toy):
    """The hostile profile is a `Profile` whose patterns return ordinary `Group`s: nothing in the fold distinguishes it."""
    r, _ = _mutated(toy, "M01")
    assert r.errors == [] and len(r.instances) == len(toy.genuine.instances)
    assert r.digest != toy.genuine.digest                     # the Program changed ...
    assert provenance.check(r, None)["status"] == "pass"      # ... and the wiring check reproduces every read of it


# ---------------------------------------------------------------------------------------------------------
# HOLE-1: provenance verifies the fold against itself over the PATTERN-DECLARED operand views; a pattern that declares
# the wrong view (wrong producer / wrong slice) is invisible to it.  Only the tier-a sample can react, and only when
# the bytes differ AND the mutated instance is drawn.
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="HOLE-1: pattern-declared operand views are not cross-checked against the event's tensor arguments")
def test_hole1_wrong_producer_is_invisible_to_provenance(toy):
    r, _ = _mutated(toy, "M01")                               # o_proj reads the input_layernorm output instead of the attention output
    assert provenance.check(r, None)["status"] == "fail"


def test_hole1_wrong_producer_is_caught_only_when_sampled(toy):
    """Different bytes: tier a fails when it draws a mutated instance (k >= stratum) and passes for a seed that misses it."""
    r, _ = _mutated(toy, "M01")
    full = replay.tier_a(r, toy.store, toy.weights, k=K_ALL, seed=0)
    assert full["status"] == "fail" and any(e["status"] == "fail" and "o_proj" in e["name"] for e in full["instances"])
    S = G.strata(r)["Gemm_v1{K=16,N=16}"]                     # the o_proj instances share their stratum with down_proj / lm_head
    p_miss = G.miss_probability(S["prefill"], 2, 1) * G.miss_probability(S["decode"], 1, 1)
    assert 0.3 < p_miss < 0.5                                 # 3/5 * 2/3 = 0.4 for k=1
    missed = [s for s in range(40) if replay.tier_a(r, toy.store, toy.weights, k=1, seed=s)["status"] == "pass"]
    assert missed, "some k=1 seed must miss the mutated instances (p = 0.4 each)"


@pytest.mark.xfail(strict=True, reason="HOLE-1 (equal bytes): no byte-based check can react and the identity check does not see the declared view")
def test_hole1_equal_bytes_wrong_producer_accepted_by_every_checker(toy_eq):
    r, _ = _mutated(toy_eq, "M11")                            # row 1's norm reads row 0's embedding (same token => same bytes)
    c = _checkers(toy_eq, r)
    assert r.digest != toy_eq.genuine.digest                  # the Program IS different (a different producer is wired)
    assert any(c[k]["status"] != "pass" for k in ("provenance", "tier_a", "tier_chain"))


# ---------------------------------------------------------------------------------------------------------
# HOLE-2: partial VU tiling is informational.  An oversize instance whose admissible child units cover half of its
# returns passes tier a while the other half is WRONG; `vu_population.partial_tiling_specs` names it and nothing fails.
# ---------------------------------------------------------------------------------------------------------


def test_partial_tiling_is_reported_and_never_sampled(toy):
    r, _ = _mutated(toy, "M09")
    a = replay.tier_a(r, toy.store, toy.weights, k=K_ALL, seed=0, max_arg_leaves=100)
    vp = a["vu_population"]
    assert vp["partial_tiling_specs"] == ["AdvGemmHalfTiled_v1{K=16,N=16,BAD=1}"] and vp["zero_mass_specs"] == []
    t = vp["tiling"]["AdvGemmHalfTiled_v1{K=16,N=16,BAD=1}"]
    assert t["covered_returns"] * 2 == t["returns"]
    # the exhaustive population (and so any sampler over it, `production_sample` included) never touches the un-tiled half
    nodes = {m["node"] for m in r.instances if m["spec"].startswith("AdvGemmHalfTiled")}
    covered = set()
    for vu in vq.enumerate_vus(r, max_arg_leaves=100):
        if vu.node in nodes:
            covered.update(vq.returns_of(r, vu))
    all_returns = set()
    for k in nodes:
        all_returns.update(vq.returns_of(r, vq.VU(node=k)))
    assert covered and len(covered) * 2 == len(all_returns)
    drawn = {g for vu in vq.production_sample(r, seed=1, n=2000, max_arg_leaves=100) if vu.node in nodes for g in vq.returns_of(r, vu)}
    assert drawn <= covered


@pytest.mark.xfail(strict=True, reason="HOLE-2: partial tiling (un-tiled returns are never evaluated) does not fail tier a / G4")
def test_hole2_partial_tiling_with_wrong_remainder_fails(toy):
    r, _ = _mutated(toy, "M09")
    a = replay.tier_a(r, toy.store, toy.weights, k=K_ALL, seed=0, max_arg_leaves=100)
    assert a["status"] == "fail"


def test_zero_mass_instance_is_rejected(toy):
    """The neighbouring case IS caught: no admissible child unit => uncovered specialization => tier a fails (D85)."""
    r, _ = _mutated(toy, "M08")
    a = replay.tier_a(r, toy.store, toy.weights, k=K_ALL, seed=0, max_arg_leaves=100)
    assert a["status"] == "fail" and a["uncovered_specs"] and "no child unit fits" in a["uncovered_specs"][0]["reason"]
    assert a["vu_population"]["zero_mass_specs"] == ["AdvGemmZeroMass_v1{K=16,N=16}"]


# ---------------------------------------------------------------------------------------------------------
# HOLE-3: "new semantics" are recognised by the SUBSTRING `quarantine` in the body's module.  A Definition authored in
# any other lane-writable module (a new profile module, a bench module) is not a quarantine spec: G4 demands no
# differential tests, quarantine_lint never sees it, and tier a evaluates its body against snapshots it was baked from.
# ---------------------------------------------------------------------------------------------------------


def test_baked_constant_body_passes_tier_a(toy):
    r, _ = _mutated(toy, "M06")
    specs = {m["spec"] for m in r.instances if m["spec"].startswith("AdvBakedOutput")}
    assert specs, "the decode SiluMul became a per-instance AdvBakedOutput{N,TAG} specialization"
    a = replay.tier_a(r, toy.store, toy.weights, k=K_ALL, seed=0)
    assert a["status"] == "pass" and provenance.check(r, None)["status"] == "pass"


@pytest.mark.xfail(strict=True, reason="HOLE-3: a Definition outside verity_vllm.program.registry that is not under *.quarantine.* is not a quarantine spec for G4")
def test_hole3_unregistered_definition_outside_quarantine_is_a_quarantine_spec(toy):
    r, _ = _mutated(toy, "M06")
    mods = {m["spec"]: spec_module(r.body.nodes[m["node"]].fn) for m in r.instances}
    baked = {s: m for s, m in mods.items() if s.startswith("AdvBakedOutput")}
    assert baked and all(not m.startswith("verity_vllm.program.registry") for m in baked.values())
    assert gates.quarantine_specs({"spec_modules": mods, "by_spec": {s: {} for s in mods}}) != []


# ---------------------------------------------------------------------------------------------------------
# HOLE-4: the census accepts a COMPUTE kernel claimed by a Transparent.  A swallowed Compute (silu_and_mul resolved as
# a `move` of the gate half that keeps `kernels=("act_and_mul_kernel",)`) leaves the residue at zero.
# ---------------------------------------------------------------------------------------------------------


def test_swallowed_compute_keeps_residue_zero_and_is_caught_only_downstream(toy):
    r, _ = _mutated(toy, "M03")
    assert not any(m["spec"].startswith("SiluMul") for m in r.instances)          # the spec vanished from the Program
    assert provenance.check(r, None)["status"] == "pass"
    a = replay.tier_a(r, toy.store, toy.weights, k=K_ALL, seed=0)                # the consumer (down_proj) disagrees ...
    assert a["status"] == "fail" and any("down_proj" in e["name"] for e in a["instances"] if e["status"] == "fail")
    assert any(replay.tier_a(r, toy.store, toy.weights, k=1, seed=s)["status"] == "pass" for s in range(40))   # ... when drawn


@pytest.mark.xfail(strict=True, reason="HOLE-4: census.declared_kernels_of_step / reconcile do not distinguish Transparent from Compute claims of a compute kernel")
def test_hole4_transparent_claiming_a_compute_kernel_is_flagged():
    class Root:
        id = 7
    t = Transparent("move", (7,), None, None, 1, "layers.0.mlp.act_fn", ("act_and_mul_kernel",), "act.silu_and_mul")
    pats, events, parts_n = census.declared_kernels_of_step([t], [Root()])
    assert "act_and_mul_kernel" in census.COMPUTE_KERNELS
    assert "act_and_mul_kernel" not in pats, "a compute kernel must not be declared by a Transparent"


# ---------------------------------------------------------------------------------------------------------
# deterministic rejections (the report's premises)
# ---------------------------------------------------------------------------------------------------------


def test_swallowed_without_rewiring_is_an_unresolved_read(toy):
    mut = toy.catalogue()["M03b"]
    prof, _ = G.mutate_profile(toy.profile, mut.target, mut.rewrite)
    with pytest.raises(Exception, match="unresolved read"):
        toy.fold(prof)


def test_extra_operand_on_an_existing_spec_is_a_fold_error(toy):
    mut = toy.catalogue()["M04"]
    prof, _ = G.mutate_profile(toy.profile, mut.target, mut.rewrite)
    with pytest.raises(Exception, match="2 operands for 1 parameters"):
        toy.fold(prof)


def test_weight_root_swap_is_caught_only_by_sampled_readers(toy):
    """Same-shaped weight Roots swapped in the adapter's weight-name map: provenance and (on M1) root policy pass; the
    instances that READ the swapped roots disagree with their snapshots when drawn (the toy has one layer: all of them)."""
    r, _ = _mutated(toy, "M05")
    assert provenance.check(r, None)["status"] == "pass"
    a = replay.tier_a(r, toy.store, toy.weights, k=K_ALL, seed=0)
    assert a["status"] == "fail" and {e["spec"].split("{")[0] for e in a["instances"] if e["status"] == "fail"} == {"RMSNormTriton_v1", "RMSNormFusedCuda_v2"}
