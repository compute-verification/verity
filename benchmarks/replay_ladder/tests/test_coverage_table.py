"""The coverage table's checks, rendering and ladder merge, on a synthetic sidecar."""
from __future__ import annotations

import copy

import pytest

import coverage_table as C


def cell(status: str = "representable", *ev: str) -> dict:
    return {"status": status, "evidence": list(ev or ("tree:x:family_facts.py",))}


def row(model: str, arch: list[str], params: float, **kw) -> dict:
    return {"model": model, "arch": arch, "params_b": params, "precision": ["BF16"], "gpus": "1x H200",
            "cells": {c: cell() for c in C.COLUMNS}, **kw}


COV = {"pin": "d9105ea80", "date": "2026-11-06", "rows": [
    row("Big", ["MoE", "DSA"], 753, active_b=42),
    row("Mid", ["MoE"], 30.5, active_b=3.3, ladder="mid"),
    row("Small", ["Dense"], 8.2),
    row("Tiny", ["Dense"], 0.6),
]}


def test_a_glyph_needs_evidence_and_a_known_status():
    cov = copy.deepcopy(COV)
    cov["rows"][0]["cells"]["proved"] = {"status": "measured", "evidence": []}
    cov["rows"][1]["cells"]["replay"] = {"status": "done", "evidence": ["r1"]}
    cov["rows"][2]["arch"] = ["SSM"]
    with pytest.raises(SystemExit) as e:
        C.check(cov)
    msg = str(e.value)
    assert "Big.proved: no run or art id" in msg and "Mid.replay: status" in msg and "Small: architecture" in msg
    C.check(COV)


def test_shares_and_sizes():
    assert C.pct(1.0) == "100%"
    assert C.pct(1 - 345 / 4219968) == "99.99%"
    assert C.pct(1 - 1e-7) != "100%"
    assert [C.size(b) for b in (1600, 753, 70.6, 14.7, 8.0, 1.24, 0.6)] == \
        ["1.6T", "753B", "70.6B", "14.7B", "8B", "1.24B", "0.6B"]


def test_rows_are_grouped_by_architecture_then_sorted_by_size():
    md = C.markdown(COV).splitlines()
    assert [ln.split(" | ")[0][2:] for ln in md[2:6]] == ["Tiny", "Small", "Mid", "Big"]
    assert "| Big | MoE, DSA | 753B (42B) | BF16 | 1x H200 | ○ | ○ | ○ |" in md


def test_the_ladder_fills_its_rows_replay_cell():
    summary = {"models": [{"model": "mid", "serve_run_id": "r0", "alone": {"requests": 2, "requests_tokens_differing": 1},
                           "replays": [
                               {"run_id": "r0", "bit_exact": True, "rows_exact_share": 1.0, "requests": 4, "tokens_served": 40,
                                "requests_tokens_differing": 0, "compile": "load"},
                               {"run_id": "r0", "bit_exact": False, "rows_exact_share": 0.9, "requests": 4,
                                "tokens_served": 40, "requests_tokens_differing": 1, "compile": "compile"},
                               {"index": 3, "error": "no replay.json"}]}]}
    cov = C.merge(copy.deepcopy(COV), [summary])
    c = cov["rows"][1]["cells"]["replay"]
    assert (c["status"], c["evidence"], c["replays_exact"], c["fresh_compile"]) == ("measured", ["r0"], "1 of 2", [False, True])
    assert "| Mid | MoE | 30.5B (3.3B) | BF16 | 1x H200 | ○ | ● 90% | ○ |" in C.markdown(cov)
    assert cov["rows"][2]["cells"]["replay"]["status"] == "representable"
