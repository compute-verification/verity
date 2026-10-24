"""served.py's ratios: which arm is which role, each ratio paired by interleaved rep (median, min, max over the pairs), a failed gate
marking only the ratios that read its arm, and two runs of one model kept apart in the tables."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import served as S

KEYS = ("time_to_first_token_s", "inter_token_latency_p50_s", "time_per_output_token_s")


def doc(*, fp8: bool = False, commit_passed: bool = True, eager_commit: bool = False) -> dict:
    """One point, B32: bi-eager stock, compiled stock and a rows-4k Commit, three reps whose metrics are all equal per rep."""
    stock = "stock-fp8" if fp8 else "stock"
    commit = ("commit-rows-4k-eager", None) if eager_commit else ("served-commit-rows-4k", "compiled")
    per_rep = {stock: (10.0, 10.0, 10.0), f"{stock}@compiled": (4.0, 5.0, 4.0), commit[0]: (5.0, 5.5, 6.0)}

    def reps(xs: tuple[float, ...]) -> list[dict]:
        return [{"metrics": {k: x for k in KEYS}} for x in xs]

    return {"baseline": stock,
            "identity": {"gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                         "preset": {"model": "unsloth/Llama-3.1-8B" if fp8 else "Qwen/Qwen3-8B",
                                    "precision": "fp8" if fp8 else "bfloat16", "gpu_id": "rtx-pro-6000"},
                         "arms": [{"id": stock}, {"id": f"{stock}@compiled", "execution": "compiled"},
                                  {"id": commit[0], "execution": commit[1]}, {"id": "stock-prof"}]},
            "gates": [{"arm": stock, "passed": True}, {"arm": f"{stock}@compiled", "passed": True},
                      {"arm": commit[0], "kind": "tokens", "passed": commit_passed}],
            "points": [{"point": {"batch": 32}, "arms": {a: {"reps": reps(xs)} for a, xs in per_rep.items()}}]}


def test_roles_follow_the_arms_id_and_execution():
    assert [S.role(a) for a in doc(fp8=True)["identity"]["arms"]] == ["stock-eager", "stock-compiled", "commit-compiled", None]
    assert S.role({"id": "commit-rows-4k-eager"}) == "commit-eager"
    assert S.model_label(doc(fp8=True)["identity"]["preset"]) == "Llama-3.1-8B FP8"


def test_each_ratio_pairs_the_arms_reps_and_takes_their_median_min_and_max():
    ms = {(m.ratio, m.metric): m for m in S.measures(doc(), run="r1")}
    assert {r for r, _ in ms} == {"served", "compiled-bi", "f"}
    served = ms[("served", "itl_p50")]
    assert (served.median, served.lo, served.hi, served.pairs) == (pytest.approx(1.25), pytest.approx(1.1), 1.5, 3)
    assert (served.arm_ms, served.over_ms, served.gpu, served.batch) == (5500.0, 4000.0, "RTX PRO 6000", 32)
    f = ms[("f", "tpot")]
    assert (f.median, f.lo, f.hi) == (pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.5))


def test_a_failed_gate_marks_only_the_ratios_that_read_its_arm():
    ms = S.measures(doc(commit_passed=False), run="r1")
    assert {m.ratio for m in ms if not m.gates} == {"served", "compiled-bi"}
    assert "(gates failed)" in S.report(ms, {}) and "Gates failed in: r1." in S.report(ms, {})


def test_two_runs_of_one_model_keep_their_own_rows():
    ms = S.measures(doc(), run="r-compiled") + S.measures(doc(eager_commit=True), run="r-eager")
    md = S.report(ms, {"r-compiled": {32: (63.0, 4.0, 67.0)}})
    f_rows = [line for line in md.splitlines() if line.startswith("| Qwen3-8B | r-") and "| decode |" in line]
    assert [row.split(" | ")[1] for row in f_rows] == ["r-compiled", "r-eager"]
    assert "| Qwen3-8B | 32 | 1.250×" in md and "| r-compiled | 32 | 63 | 4 | 67 |" in md
    assert {m.ratio for m in ms if m.run == "r-eager"} == {"eager-served", "eager", "f"}
