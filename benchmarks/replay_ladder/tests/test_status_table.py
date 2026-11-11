"""The per-model status table's checks, rendering and ladder merge, on a synthetic sidecar."""
from __future__ import annotations

import copy

import pytest

import status_table as C


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
    assert md[0].endswith("| Circuit | Replay | Replay, unpinned | Proved |")
    assert "| Big | MoE, DSA | 753B (42B) | BF16 | 1x H200 | ○ | ○ | ○ | ○ |" in md


def ladder_model(pin: str, run: str, shares: list[float]) -> dict:
    return {"model": "mid", "pin": pin, "serve_run_id": run, "alone": {"requests": 2, "requests_tokens_differing": 1},
            "replays": [{"run_id": run, "bit_exact": s == 1, "rows_exact_share": s, "requests": 4, "tokens_served": 40,
                         "requests_tokens_differing": int(s < 1), "compile": "compile" if i else "load"}
                        for i, s in enumerate(shares)] + [{"index": len(shares) + 1, "error": "no replay.json"}]}


def test_the_ladder_fills_each_replay_column_from_its_arm():
    summary = {"models": [ladder_model("pinned", "r1", [1.0, 1.0]), ladder_model("unpinned", "r0", [1.0, 0.9])]}
    cov = C.merge(copy.deepcopy(COV), [summary])
    pinned, unpinned = cov["rows"][1]["cells"]["replay"], cov["rows"][1]["cells"]["replay_unpinned"]
    assert (pinned["status"], pinned["evidence"], pinned["replays_exact"]) == ("measured", ["r1"], "2 of 2")
    assert (unpinned["evidence"], unpinned["replays_exact"], unpinned["fresh_compile"]) == (["r0"], "1 of 2", [False, True])
    assert "| Mid | MoE | 30.5B (3.3B) | BF16 | 1x H200 | ○ | ● 100% | ● 90% | ○ |" in C.markdown(cov)
    assert cov["rows"][2]["cells"]["replay"]["status"] == "representable"


def test_the_footnote_names_the_models_that_need_the_pin():
    big = ladder_model("unpinned", "r2", [1.0, 1.0]) | {"model": "big", "tilings": {"runs": 2, "reductions": 1, "split": 1}}
    mid = ladder_model("unpinned", "r0", [1.0, 1.0]) | {"tilings": {"runs": 2, "reductions": 1, "split": 0}}
    small = ladder_model("unpinned", "r3", [1.0, 0.9]) | {"model": "small"}
    tiny = ladder_model("unpinned", "r4", [1.0]) | {"model": "tiny"}
    cov = copy.deepcopy(COV)
    for row, name in zip(cov["rows"], ("big", "mid", "small", "tiny")):
        row["ladder"] = name
    cov = C.merge(cov, [{"models": [big, mid, small, tiny]}])
    assert C.pin_verdicts(cov) == (["Big", "Small"], ["Mid", "Tiny"])
    assert "differed from serving or kept different tilings: Big, Small; every replay matched serving: Mid, Tiny." \
        in C.markdown(cov)


def test_an_arm_the_ladder_did_not_run_keeps_its_sidecar_cell():
    cov = C.merge(copy.deepcopy(COV), [{"models": [ladder_model("unpinned", "r0", [1.0]), ladder_model("rblock", "r5", [1.0])]}])
    assert cov["rows"][1]["cells"]["replay"]["status"] == "representable"
    assert cov["rows"][1]["cells"]["replay_unpinned"]["status"] == "measured"
