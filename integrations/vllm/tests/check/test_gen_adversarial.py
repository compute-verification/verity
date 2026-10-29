"""Minimal failing tests for the holes found by mutating the fold's OUTPUT
through the patterns — the lever an untrusted adapter author holds — over the trusted record (the synthetic `consistent_toy`:
log + snapshots derived from the GENUINE Program + toy weights).

Every hole test is `xfail(strict=True)`: it fails today (that is the hole) and XPASSes — the suite reports it — when the
integrator closes it.  The premises (what the harness relies on) and the deterministic rejections are asserted positively.
Harness: `tests/check/adversarial.py`.
"""
from __future__ import annotations

import pytest

from verity_vllm.properties import census
from verity_vllm.check import gates
from tests.check import adversarial as G
from verity_vllm.observe.fold.resolver import Transparent, spec_module


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    rec = G.ToyRecord(tmp_path_factory.mktemp("adv_snaps"))
    try:
        rec.bake()
        yield rec
    finally:
        rec.restore()


def _mutated(rec, mid: str, **kw):
    mut = rec.catalogue()[mid]
    prof, state = G.mutate_profile(rec.profile, mut.target, mut.rewrite, weight_field=mut.weight_field)
    r = rec.fold(prof)
    assert G.hits(state) > 0 or mut.weight_field is not None, f"{mid}: the mutation never fired"
    return r, mut


# ---------------------------------------------------------------------------------------------------------
# premises
# ---------------------------------------------------------------------------------------------------------


def test_premise_mutating_pattern_is_an_ordinary_pattern(toy):
    """The hostile profile is a `Profile` whose patterns return ordinary `Group`s: nothing in the fold distinguishes it."""
    r, _ = _mutated(toy, "M01")
    assert r.errors == [] and len(r.instances) == len(toy.genuine.instances)
    assert r.digest != toy.genuine.digest                     # the Program changed


# ---------------------------------------------------------------------------------------------------------
# HOLE-3: "new semantics" are recognised by the SUBSTRING `quarantine` in the body's module.  A Definition authored in
# any other author-writable module (a new profile module, a bench module) is not a quarantine spec: G4 demands no
# differential tests and quarantine_lint never sees it.
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="HOLE-3: a Definition outside verity.ml.definitions.registry that is not under *.quarantine.* is not a quarantine spec for G4")
def test_hole3_unregistered_definition_outside_quarantine_is_a_quarantine_spec(toy):
    r, _ = _mutated(toy, "M06")
    mods = {m["spec"]: spec_module(r.body.nodes[m["node"]].fn) for m in r.instances}
    baked = {s: m for s, m in mods.items() if s.startswith("AdvBakedOutput")}
    assert baked and all(not m.startswith("verity.ml.definitions.registry") for m in baked.values())
    assert gates.quarantine_specs({"spec_modules": mods, "by_spec": {s: {} for s in mods}}) != []


# ---------------------------------------------------------------------------------------------------------
# HOLE-4: the census accepts a COMPUTE kernel claimed by a Transparent.  A swallowed Compute (silu_and_mul resolved as
# a `move` of the gate half that keeps `kernels=("act_and_mul_kernel",)`) leaves the residue at zero.
# ---------------------------------------------------------------------------------------------------------


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
