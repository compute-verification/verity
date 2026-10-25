"""`admit_release.py`'s own arithmetic: the release-order add times, the step cost fit, and which traces it reads."""
from __future__ import annotations

import json

import pytest


def test_release_order_adds_wait_for_the_boundary_and_every_earlier_release(mod):
    ar = mod("admit_release")
    rel = ar.Release(origin=100.0, tau=0.1, bucket=[3, 1, 1], ordinal=[0, 1, 0], ready=[100.05, 100.32, 100.1], source="")
    order, hold = ar.counterfactual(rel, aligned=False)
    assert order == [2, 1, 0] and hold == pytest.approx([100.2, 100.32, 100.4])
    _, aligned = ar.counterfactual(rel, aligned=True)
    assert aligned == pytest.approx([100.2, 100.4, 100.4])


def test_fit_drops_a_negative_per_token_cost(mod):
    ar = mod("admit_release")
    rows = [(1, 1), (2, 1), (1, 2), (3, 3), (4, 1), (2, 4)]
    X = [[1.0, float(t), float(r)] for t, r in rows]
    y = [1 + 0.5 * r - 0.2 * t for t, r in rows]
    assert ar.lstsq(X, y) == pytest.approx([1.0, -0.2, 0.5])
    c = ar.fit(X, y)
    assert c[1] == 0.0 and c[2] > 0


def test_a_warden_counting_egress_from_the_admission_is_refused(tmp_path, mod):
    ar = mod("admit_release")
    (tmp_path / "meta.json").write_text(json.dumps({"warden": {
        "params": {"bucket_ms": 100, "delay": 1200}, "origin_unix": 0.0,
        "egress_base": "admission s: egress cell i at s + D + i/r"}}))
    (tmp_path / "requests.jsonl").write_text("")
    with pytest.raises(ValueError, match="not the release"):
        ar.release_of(tmp_path, [], 0.0)
