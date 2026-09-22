"""REC-15 / lane COMPACT-P1: ``e2e.population.make_plan`` builds P1 as ``UnitRuns`` -- the same partition as the dense
plan (``partition_digest`` equal, verdicts equal), streamed validation agrees with the retained one, and the B0 unit
counts per kind equal the Table-3a census."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src"), str(POC / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e import population as popmod  # noqa: E402

CENSUS = ROOT / "docs" / "data" / "table3a" / "B0-serve-lp32-s15.json"
RECORD = ROOT / "data" / "rec" / "p1-b0-lp32s15-validated.json"
# the TINY P1/cu=whole partition, MEASURED 2026-09-10 (data/rec/compact-probe-p1-tiny.json validated the same plan)
TINY_P1_WHOLE_PARTITION = "3646488c15e665a5ae16135763e62840f96ed29e01cf6e6df5c6f75619fa18dc"


@pytest.fixture(scope="module")
def tiny():
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    program, adapted, circuit, pdigest = popmod.build(popmod.TINY, popmod.TINY_LP, popmod.TINY_STEPS)
    return program, adapted, circuit, pdigest


@pytest.mark.parametrize("spec", ["P1/cu=whole", "P1/cu=operator"])
def test_make_plan_p1_is_compact_and_the_same_partition_as_dense(tiny, spec):
    from veritor.plans.runs import UnitRuns
    from veritor.plans.summary import partition_digest, plan_digest
    from veritor.plans.validate import validate_plan

    _, adapted, circuit, pdigest = tiny
    compact, meta_c = popmod.make_plan(spec, adapted, circuit)
    dense, meta_d = popmod.make_plan(spec, adapted, circuit, compact=False)
    assert isinstance(compact.verification_units, UnitRuns) and compact.compact
    assert not dense.compact
    assert len(compact.verification_units) == len(dense.verification_units) == 53113
    assert partition_digest(compact) == partition_digest(dense)
    assert plan_digest(compact)[0] == "plan-rule/v1" and plan_digest(dense)[0] == "plan-json/v1"
    if spec == "P1/cu=whole":
        assert partition_digest(compact) == TINY_P1_WHOLE_PARTITION
    vd, vc = validate_plan(circuit, dense), validate_plan(circuit, compact)
    assert vd.status == vc.status == "valid"
    assert (vd.vu_edges, vd.cu_edges) == (vc.vu_edges, vc.cu_edges)
    assert [b.to_json() for b in vd.vu_boundaries] == [b.to_json() for b in vc.vu_boundaries]
    assert [b.to_json() for b in vd.cu_boundaries] == [b.to_json() for b in vc.cu_boundaries]
    vs = validate_plan(circuit, compact, retain=False)
    assert (vs.status, vs.vu_edges, vs.cu_edges, vs.escape_steps) == (vc.status, vc.vu_edges, vc.cu_edges, vc.escape_steps)
    assert vs.vu_stats.units == 53113 and vs.vu_stats.gates == sum(b.boundary_size() for b in vc.vu_boundaries)
    # population / RU membership over the streamed verdict
    pop_c, pop_s = popmod.population(adapted.program, compact, vc, pdigest), popmod.population(adapted.program, compact, vs, pdigest)
    assert pop_c.partition_digest == pop_s.partition_digest == partition_digest(dense)
    assert popmod.ru_membership(vs, compact) == popmod.ru_membership(vc)
    with pytest.raises(ValueError):
        popmod.ru_membership(vs)
    art = popmod.validation_artifact(adapted.program, compact, vs, pdigest, meta_c, seconds=None, maxrss_mb=None, verdict_source="test")
    assert art["partition_digest"] == partition_digest(dense) and art["validator"] == "validate_plan(retain=False)"


def test_make_plan_p3_stays_dense(tiny):
    """The P3 pins (plan-json/v1 descriptor digests recorded in earlier records) are untouched by REC-15."""

    from veritor.plans.summary import plan_digest

    _, adapted, circuit, _ = tiny
    plan, _ = popmod.make_plan("P3/cu=block", adapted, circuit)
    assert not plan.compact and plan_digest(plan)[0] == "plan-json/v1"


def _b0_slice(nl: int, v: int, lp: int, steps: int):
    from verity_vllm.registry import b1

    config = dict(b1.B0_CONFIG)
    config["NL"], config["V"] = nl, v
    return popmod.build(config, lp, steps)


@pytest.mark.slow
def test_b0_slice_dense_digest_equals_compact_digest():
    """The B0 sub-program NL=1, V=1024, LP=2, STEPS=1 (5,489,429 gates, 4,726,149 P1 units): dense plan + retained
    validation vs compact plan + streamed validation -- same partition digest, same verdict fields.  The dense side keeps
    one Boundary + Obligation per unit (~3.5 KB each: ~16 GB): a pod job, not a laptop one (the laptop rule is 2 GB; with
    B0's H=576 every NL=1 slice has > 1 M weight-Input units, so no B0 slice fits the dense path on the laptop).
    Smaller shape for a machine with less RAM: ``COMPACT_P1_SLICE=nl,v,lp,steps``."""

    from veritor.plans.summary import partition_digest
    from veritor.plans.validate import validate_plan

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    shape = tuple(int(x) for x in os.environ.get("COMPACT_P1_SLICE", "1,1024,2,1").split(","))
    _, adapted, circuit, _ = _b0_slice(*shape)
    compact, _ = popmod.make_plan("P1/cu=whole", adapted, circuit)
    dense, _ = popmod.make_plan("P1/cu=whole", adapted, circuit, compact=False)
    assert len(compact.verification_units) == len(dense.verification_units)
    assert partition_digest(dense) == partition_digest(compact)
    vs = validate_plan(circuit, compact, retain=False)
    vd = validate_plan(circuit, dense)
    assert vd.status == vs.status == "valid"
    assert (vd.vu_edges, vd.cu_edges, vd.escape_steps) == (vs.vu_edges, vs.cu_edges, vs.escape_steps)
    assert vs.vu_stats.gates == sum(b.boundary_size() for b in vd.vu_boundaries)
    assert vs.cu_stats.gates == sum(b.boundary_size() for b in vd.cu_boundaries)


def test_sharded_parts_merge_to_the_single_process_verdict(tiny):
    """The pod driver's pieces: work-balanced cuts tile the VU index, each range's part (``_worker``) merged by
    ``merge_parts`` gives the single-process streamed verdict (status, edges, escape steps, boundary totals); a missing
    range is reported as exactly that gap, never as ``valid``."""

    from compact_p1_validate import _SHARED, _worker, coverage_gaps, merge_parts, weighted_cuts
    from veritor.plans.validate import validate_plan

    _, adapted, circuit, _ = tiny
    plan, _ = popmod.make_plan("P1/cu=whole", adapted, circuit)
    n = len(plan.verification_units)
    bounds = weighted_cuts(plan, circuit, 5)
    assert bounds[0] == 0 and bounds[-1] == n and bounds == sorted(bounds)
    _SHARED["circuit"], _SHARED["plan"] = circuit, plan
    parts = [_worker((i, bounds[i], bounds[i + 1])) for i in range(5)]
    whole = validate_plan(circuit, plan, retain=False)
    merged = merge_parts(parts, bounds, workers=1, chunks=5, n=n, seconds=0.0)
    assert merged["status"] == whole.status == "valid"
    assert (merged["vu_edges"], merged["cu_edges"], merged["escape_steps"], merged["cu_escape_steps"]) == (
        whole.vu_edges, whole.cu_edges, whole.escape_steps, whole.cu_escape_steps)
    assert merged["vu_stats"] == whole.vu_stats.to_json() and merged["cu_stats"] == whole.cu_stats.to_json()
    partial = merge_parts(parts[:2] + parts[3:], None, workers=1, chunks=4, n=n, seconds=0.0)
    assert partial["status"] == "rejected" and partial["gaps"] == [[bounds[2], bounds[3]]]
    assert coverage_gaps([(0, 3), (3, 7)], 10) == [[7, 10]] and coverage_gaps([(0, 5), (4, 10)], 10) == [[5, 4]]


def test_b0_kind_counts_equal_the_census():
    """Whole B0 (LP=32, STEPS=15) compact P1/cu=whole plan: 31,157,512 non-input units, per (fn, form) kind equal to the
    Table-3a census; 135,039,328 Input units (the horizon-independent weight/prompt coordinates).  Plan only, ~15 s, < 2 GB."""

    from compact_p1_validate import census_compare, kind_counts

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    from verity_vllm.registry import b1

    program, adapted, circuit, pdigest = popmod.build(dict(b1.B0_CONFIG), 32, 15)
    assert pdigest == "d4f4b1f1b4d592d697eef5b6c9c803108bdf02d0ef24c945eace6e705832e1b8"
    plan, _ = popmod.make_plan("P1/cu=whole", adapted, circuit)
    assert plan.compact and len(plan.verification_units) == 166_196_840 and len(plan.commitment_units) == 1
    counts = kind_counts(plan, circuit)
    cmp = census_compare(counts, CENSUS)
    assert cmp["agree"], cmp["mismatches"]
    assert cmp["plan_non_input_units"] == 31_157_512 == cmp["census_N_V"]
    assert cmp["plan_input_units"] == 135_039_328 == program.input_gates
    if RECORD.exists():  # the validated record pins the same counts and digest as the plan built here
        rec = json.loads(RECORD.read_text())
        assert rec["plan"]["vus"] == 166_196_840 and rec["plan"]["non_input_units"] == 31_157_512
        if os.environ.get("COMPACT_P1_CHECK_DIGEST"):  # 66 s on the laptop: opt in
            from veritor.plans.summary import partition_digest

            assert partition_digest(plan) == rec["plan"]["partition_digest"]
