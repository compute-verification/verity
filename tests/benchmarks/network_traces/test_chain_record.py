"""chain_record.leak: K on a network attester run's files counts the scan's widths and terms, and it refuses a sample and a
network attester the scan's class doesn't serve."""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest


HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "network_traces")


def _load(name: str):
    sys.path.insert(0, str(HERE))
    sys.path.append(str(HERE.parents[1]))  # the root, for benchmarks.network_traces
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load("chain_record")


def test_leak_counts_the_scans_terms_on_the_network_attester_run(tmp_path):
    import math
    from dataclasses import asdict
    from verity.protocols.accounting.network.verifier import network_attester as W
    run = tmp_path / "c1"
    (run / "out").mkdir(parents=True)
    (run / "out/deployment.json").write_text(json.dumps({"params": asdict(W.W63)}))
    scan = {"draw": {"honest": 65}, "T": 32}
    (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": {"accepted": True, "windows": 75}}))
    for request, why in ((None, "asks for no number of requests"), ({"client_port": 8012, "windows": 75}, "relay session"),
                         ({"requests": True}, "asks for no number"), ({"requests": 0}, "asks for no number")):
        if request is not None:
            (run / "request.json").write_text(json.dumps(request))
        with pytest.raises(SystemExit, match=why):
            cr.leak(run, scan)
    (run / "request.json").write_text(json.dumps({"device": "warden", "requests": 100}))
    for verdict in ({"accepted": False, "windows": 75}, {"windows": 75, "requests": 100}):
        (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": verdict}))
        with pytest.raises(SystemExit, match="isn't accepted"):
            cr.leak(run, scan)
    (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": {"accepted": True, "windows": 75}}))
    doc = cr.leak(run, scan)
    assert doc["run"]["requests"] == 100 and doc["run"]["requests_per_link_window"] == 100 / 75
    assert doc["widths"] == {"NS": 149_648, "NE": 1_136, "entries": 8, "NV": 4_832, "egress_row": 65_536}
    assert doc["terms"]["b"] == 154_480 and doc["terms"]["fmax"] == {"0.01": 5, "1e-06": 14}
    assert doc["terms"]["schedules_per_request"] == doc["terms"]["cells_per_request"] == 7
    assert (doc["terms"]["jitter_bound"], doc["terms"]["jitter_per_block"]) == (2, 32)
    assert doc["links"] == {"link": 1, "datacenter": 12_500, "world": 46_870_000}
    assert doc["weights"]["label"] == "GLM-5.3 in FP4" and "GLM-5.3 in FP4" in cr.leak_markdown(doc)
    assert any("leak_bounds.at_link_rate" in c and "not the scope's K" in c for c in doc["conditions"])
    json.dumps(doc)
    rows = {(r["period"], r["scope"], r["prog"]): r for r in doc["rows"]}
    week = {p: rows["week", "link", p] for p in ("registered", "anchors")}
    lw, rate, cap = 604_800 / 3.2, 100 / 75, 2 * 7
    assert week["registered"]["per_period"] == pytest.approx(lw * (8 * 1_136 + 32 + rate * cap))
    assert week["anchors"]["per_link_window"] == pytest.approx(rate * (1_121 + cap) + 32)
    assert week["anchors"]["pooled"]["0.01"] - week["anchors"]["per_period"] == pytest.approx(
        5 * (154_480 + math.log2(lw * 65 + 1)))
    assert Fraction(week["anchors"]["eps"]) == Fraction(189_000, 2 ** 20)
    for scope, n in doc["links"].items():
        assert rows["week", scope, "registered"]["per_period"] == pytest.approx(n * week["registered"]["per_period"])
        assert Fraction(rows["week", scope, "registered"]["eps"]) == n * Fraction(189_000, 2 ** 20)
    with pytest.raises(SystemExit, match="sampled 16 of 65"):
        cr.leak(run, {"draw": {"honest": 16}})
    (run / "out/deployment.json").write_text(json.dumps({"params": {**asdict(W.W63), "lanes": 8}}))
    with pytest.raises(ValueError, match="lanes = 4"):
        cr.leak(run)


