"""Commitment accounting (GUIDANCE §18.5): the three counts and the fail-closed assertion, on synthetic layouts and on
every local case (RA's structural-plan fixtures, the identity fixture, W3c's authenticated rows, TA1's B0/B1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.ir import accounting as A

REPO = Path(__file__).resolve().parents[2]


def _unit(i, ins, outs, internal=()):
    return A.Unit(f"u{i}", [A.ValueRun(n, 1, 16) for n in ins], [A.ValueRun(n, 1, 16) for n in outs], [A.ValueRun(n, 1, 16) for n in internal])


def _leaf(name, *vals, nbytes=None):
    vs = [A.ValueRun(v, 1, 16) for v in vals]
    return A.Leaf(name, "ru", vs, nbytes if nbytes is not None else 2 * len(vs))


def test_counts_and_padding():
    units = [_unit(0, ["x"], ["a"], internal=["acc"]), _unit(1, ["a"], ["y"])]
    protocol = [A.ValueRun("x", 1, 16), A.ValueRun("y", 1, 16)]
    leaves = [_leaf("l0", "x"), _leaf("l1", "a", nbytes=4), _leaf("l2", "y")]
    acc = A.account("t", "p", units, protocol, leaves)
    assert (acc.required_values, acc.required_bytes) == (3, 6)
    assert (acc.committed_leaves, acc.committed_values, acc.committed_bytes) == (3, 3, 8)
    assert (acc.extra_bytes, acc.padding_bytes, acc.conservative_capture_bytes) == (2, 2, 0)
    assert acc.ok and A.assert_accounted(acc) is acc


def test_internal_accumulator_committed_fails_closed():
    units = [_unit(0, ["x"], ["a"], internal=["acc"])]
    leaves = [_leaf("l0", "x"), _leaf("l1", "a"), _leaf("l2", "acc")]
    acc = A.account("t", "p", units, [A.ValueRun("x", 1, 16)], leaves)
    assert acc.internal_committed and acc.internal_committed[0]["value"] == "acc"
    with pytest.raises(A.AccountingError, match="acc"):
        A.assert_accounted(acc)


def test_undeclared_extra_fails_and_declared_extra_passes():
    units = [_unit(0, ["x"], ["a"])]
    leaves = [_leaf("l0", "x"), _leaf("l1", "a"), _leaf("l2", "stray")]
    acc = A.account("t", "p", units, [], leaves)
    assert [o["value"] for o in acc.offending] == ["stray"]
    with pytest.raises(A.AccountingError, match="stray"):
        A.assert_accounted(acc)
    acc2 = A.account("t", "p", units, [], leaves, declared_extra={"stray": "conservative capture"})
    assert acc2.ok and acc2.conservative_capture_bytes == 2 and acc2.declared_extra_bytes == 2
    # a leaf committing more of a run than required: the surplus is extra and must be declared
    units3 = [A.Unit("u", [A.ValueRun("row", 4, 16)], [A.ValueRun("o", 1, 16)])]
    leaves3 = [A.Leaf("row", "ru", [A.ValueRun("row", 6, 16)], 12), _leaf("o", "o")]
    acc3 = A.account("t", "p", units3, [], leaves3)
    assert acc3.offending and acc3.conservative_capture_values == 2 and acc3.conservative_capture_bytes == 4
    assert A.account("t", "p", units3, [], leaves3, declared_extra={"row": "row leaf"}).ok


def test_structural_plan_fixtures_are_accounted():
    sp = REPO / "fixtures/verity-ir/structural-plans/v0"
    n = 0
    for p in sorted(sp.glob("*.json")):
        doc = json.loads(p.read_text())
        for i in range(len(doc["plans"])):
            acc = A.structural_plan_fixture(doc, i)
            if acc is None:
                continue
            n += 1
            A.assert_accounted(acc)
            assert acc.extra_bytes == 0, (p.name, acc.plan)
    assert n >= 20


def test_identity_fixture_fine_plan_accounts_and_coarse_plans_fail():
    doc = json.loads((REPO / "fixtures/verity-ir/identity/identity-v0.json").read_text())
    for layout in ("scalar", "packed"):
        fine = A.identity_fixture(doc, "plan/fine", layout)
        A.assert_accounted(fine)
        assert (fine.required_values, fine.committed_bytes, fine.extra_bytes) == (12, 27, 0)
        dot = A.identity_fixture(doc, "plan/dot", layout)
        assert not dot.ok
        assert {d["value"] for d in dot.internal_committed} == {"Dot[0]/Square[0]/t", "F[0]/a", "G[0]/b"}
        with pytest.raises(A.AccountingError, match="Dot\\[0\\]/Square\\[0\\]/t"):
            A.assert_accounted(dot)


def test_w3c_fixtures():
    w3 = REPO / "fixtures/B1-b3eba618b1b5-20260907T1731Z"
    if not w3.is_dir():
        pytest.skip("W3c fixtures not present")
    dirs = sorted(x for x in w3.iterdir() if x.is_dir() and (x / "fixture.json").exists())
    assert len(dirs) == 30
    for d in dirs:
        acc = A.assert_accounted(A.w3c_fixture(d))
        f = json.loads((d / "fixture.json").read_text())
        assert acc.required_values == 2 * f["K"] + 1
        assert acc.conservative_capture_bytes == acc.extra_bytes == acc.committed_bytes - acc.required_bytes


def test_ta1_captures():
    ta1 = REPO / "fixtures/results/TA1-typed-b1-20260907"
    if not (ta1 / "B0/description.json").exists():
        pytest.skip("TA1 results not present")
    for cfg, leaves, act_bytes, fp8 in (("B0", 14134, 25137408, 480), ("B1", 271350, 1832337408, 448)):
        summ = json.loads((ta1 / cfg / "capture_summary.json").read_text())
        assert (summ["leaves"], summ["bytes_captured"]) == (leaves, act_bytes)
        per_op = A.assert_accounted(A.ta1_results(ta1 / cfg, "per-op-row"))
        ru = per_op.domains["ru"]
        assert ru["leaves"] == leaves
        n_sampled = sum(len(json.loads(p.read_text())["output_records"]) for p in (ta1 / cfg / "ru").glob("*/boundary.json"))
        assert ru["bytes"] == act_bytes + 4 * n_sampled  # capture_summary's bytes exclude the int32 sampler leaves
        assert per_op.padding_bytes == 0 and per_op.extra_bytes == fp8  # only the fp8 scale scalars
        named = A.assert_accounted(A.ta1_results(ta1 / cfg, "named-calls"))
        assert named.committed_bytes == per_op.committed_bytes
        assert named.extra_bytes > per_op.extra_bytes and named.required_bytes < per_op.required_bytes
        assert named.required_bytes + named.conservative_capture_bytes == named.committed_bytes
