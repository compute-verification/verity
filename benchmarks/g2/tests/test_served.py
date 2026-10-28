"""served.py's ratios: which arm is which role (by its module and its execution, else the spec's), at most one arm per role, each
ratio paired by interleaved rep (median, min, max over the pairs), a failed gate marking only the ratios that read its arm, the
harness's Commit arms read as G2's are, a run's compiled Commit run time taken with another run's tail (``--tail``), two runs
of one model kept apart in the tables, and which ratios count by curve.py's rules (a compiled arm's only from a run pinned to
the Build of record, its `pin_check.txt`)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import served as S

KEYS = ("time_to_first_token_s", "inter_token_latency_p50_s", "time_per_output_token_s")
#: (id, module, execution, per-rep metric) of a Commit arm: G2's served-arm arms and the harness's
ROWS_4K = ("served-commit-rows-4k", "served_commit.served_arm:ROWS_4K", "compiled", (5.0, 5.5, 6.0))
EAGER_ROWS_4K = ("commit-rows-4k-eager", "served_commit.served_arm:EAGER_ROWS_4K", None, (5.0, 5.5, 6.0))
COMMIT = ("commit", "verity_bench.vllm.commit_arm:COMMIT", None, (20.0, 30.0, 25.0))
COMMIT_COMPILED = ("commit@compiled", "verity_bench.vllm.commit_arm:COMMIT_COMPILED", "compiled", (5.0, 5.5, 6.0))


def doc(*, fp8: bool = False, commit_passed: bool = True, commits: tuple[tuple, ...] = (ROWS_4K,), eager: bool = True) -> dict:
    """One point, B32: bi-eager stock, compiled stock and `commits`, three reps whose metrics are all equal per rep, in an eager
    spec (the harness writes its eagerness into the preset) unless `eager` is False."""
    stock = "stock-fp8" if fp8 else "stock"
    per_rep = {stock: (10.0, 10.0, 10.0), f"{stock}@compiled": (4.0, 5.0, 4.0)} | {c[0]: c[3] for c in commits}

    def reps(xs: tuple[float, ...]) -> list[dict]:
        return [{"metrics": {k: x for k in KEYS}} for x in xs]

    return {"baseline": stock,
            "identity": {"gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                         "preset": {"model": "unsloth/Llama-3.1-8B" if fp8 else "Qwen/Qwen3-8B",
                                    "precision": "fp8" if fp8 else "bfloat16", "gpu_id": "rtx-pro-6000",
                                    "engine_settings": {"enforce_eager": eager, "cuda_graphs": not eager}},
                         "arms": [{"id": stock, "module_spec": None, "execution": None},
                                  {"id": f"{stock}@compiled", "module_spec": None, "execution": "compiled"},
                                  *({"id": i, "module_spec": m, "execution": e} for i, m, e, _ in commits),
                                  {"id": "stock-prof", "module_spec": "served_commit.prof_arm:STOCK_PROF", "execution": None}]},
            "gates": [{"arm": stock, "passed": True}, {"arm": f"{stock}@compiled", "passed": True},
                      *({"arm": c[0], "kind": "tokens", "passed": commit_passed} for c in commits)],
            "points": [{"point": {"batch": 32}, "arms": {a: {"reps": reps(xs)} for a, xs in per_rep.items()}}]}


def test_roles_follow_the_arms_module_and_execution_else_the_specs():
    ident = doc(fp8=True, commits=(ROWS_4K, EAGER_ROWS_4K))["identity"]
    assert [S.role(a, S.spec_execution(ident)) for a in ident["arms"]] == [
        "stock-eager", "stock-compiled", "commit-compiled", "commit-eager", None]
    assert S.roles(ident) == {"stock-eager": "stock-fp8", "stock-compiled": "stock-fp8@compiled",
                              "commit-compiled": "served-commit-rows-4k", "commit-eager": "commit-rows-4k-eager"}
    compiled = doc(commits=(EAGER_ROWS_4K,), eager=False)["identity"]
    assert S.spec_execution(compiled) == "compiled"
    with pytest.raises(ValueError, match="stock and stock@compiled are both stock-compiled"):
        S.roles(compiled)
    compiled["arms"].pop(1)
    assert S.roles(compiled) == {"stock-compiled": "stock", "commit-compiled": "commit-rows-4k-eager"}
    assert S.role({"id": "served-commit-rows-4k", "module_spec": "served_commit.prof_arm:SERVED_COMMIT_ROWS_4K"}, "eager") is None
    assert S.model_label(ident["preset"]) == "Llama-3.1-8B FP8"


def test_a_run_with_two_arms_in_one_role_is_refused():
    with pytest.raises(ValueError, match="served-commit-rows-4k and commit@compiled are both commit-compiled"):
        S.measures(doc(commits=(ROWS_4K, COMMIT_COMPILED)), run="r1")


def test_each_ratio_pairs_the_arms_reps_and_takes_their_median_min_and_max():
    ms = {(m.ratio, m.metric): m for m in S.measures(doc(), run="r1")}
    assert {r for r, _ in ms} == {"served", "compiled-bi", "f"}
    served = ms[("served", "itl_p50")]
    assert (served.median, served.lo, served.hi, served.pairs) == (pytest.approx(1.25), pytest.approx(1.1), 1.5, 3)
    assert (served.arm_ms, served.over_ms, served.gpu, served.batch) == (5500.0, 4000.0, "RTX PRO 6000", 32)
    f = ms[("f", "tpot")]
    assert (f.median, f.lo, f.hi) == (pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.5))


def test_the_harness_commit_arms_are_read_as_the_served_arms_are():
    """llama31-8b-fp8-1gpu-commit-served's arms: `commit` is eager by its spec, `commit@compiled` compiled by its own."""
    d = doc(fp8=True, commits=(COMMIT, COMMIT_COMPILED))
    ms = {(m.ratio, m.metric): m for m in S.measures(d, run="r-harness")}
    assert {r for r, _ in ms} == set(S.RATIOS)
    assert (ms[("served", "tpot")].arm, ms[("served", "tpot")].over, ms[("served", "tpot")].median) == (
        "commit@compiled", "stock-fp8@compiled", pytest.approx(1.25))
    assert (ms[("eager", "tpot")].arm, ms[("eager", "tpot")].over, ms[("eager", "tpot")].median) == (
        "commit", "stock-fp8", pytest.approx(2.5))
    md = S.report(list(ms.values()), {}, {"r-harness": S.run_facts(d)})
    assert "| Llama-3.1-8B FP8 | 32 | 1.250×" in md and "| 4000.000 | commit@compiled | r-harness |" in md
    assert "| r-harness | Llama-3.1-8B FP8 | commit, commit@compiled | 1 |" in md


def test_the_cuda_graph_steps_are_the_compiled_commits():
    d = doc(fp8=True, commits=(COMMIT, COMMIT_COMPILED))
    d["points"][0]["arms"]["commit"]["extras"] = {"replayed_steps": {"mean": 0.0}, "steps": {"mean": 67.0}}
    d["points"][0]["arms"]["commit@compiled"]["extras"] = {k: {"mean": v} for k, v in
                                                          (("replayed_steps", 63.0), ("eager_steps", 4.0), ("steps", 67.0))}
    assert S.graph_steps(d) == {32: (63.0, 4.0, 67.0)}
    assert S.graph_steps(doc(commits=(EAGER_ROWS_4K,))) == {}


def test_a_failed_gate_marks_only_the_ratios_that_read_its_arm():
    ms = S.measures(doc(commit_passed=False), run="r1")
    assert {m.ratio for m in ms if not m.gates} == {"served", "compiled-bi"}
    assert "(gates failed)" in S.report(ms, {}) and "Gates failed in: r1." in S.report(ms, {})


def test_a_run_without_timed_points_still_reports_why_and_its_hosts_load():
    d = doc(fp8=True, commit_passed=False)
    d["gates"][-1].update(point="b32-p512-mixed", control="stock-fp8@compiled", divergent_requests=17,
                          first_divergence={"request": "b32-p512-mixed-10", "position": 2})
    for rep in d["points"][0]["arms"]["stock-fp8"]["reps"]:
        rep["host"] = {"loadavg_1m": 62.8}
    f = S.run_facts(d)
    assert f["load"] == (62.8, 62.8) and f["model"] == "Llama-3.1-8B FP8" and f["commit"] == ["served-commit-rows-4k"]
    assert f["failed"] == [("served-commit-rows-4k tokens b32-p512-mixed against stock-fp8@compiled, divergent requests 17 "
                            "(first b32-p512-mixed-10 at token 2)")]
    d["points"] = []
    md = S.report([], {}, {"r-gated": S.run_facts(d)})
    assert "| r-gated | Llama-3.1-8B FP8 | served-commit-rows-4k | 0 |  | served-commit-rows-4k tokens" in md


def _timed(d: dict, arm: str, walls: tuple[float, ...], tails: tuple[float, ...]) -> dict:
    for rep, wall, tail in zip(d["points"][0]["arms"][arm]["reps"], walls, tails):
        rep.update(wall_s=wall, generated_tokens=100, host_timing={"device_tail_s": tail})
    return d


def test_a_runs_compiled_commit_takes_a_fixed_runs_tail_in_its_run_time_alone():
    d = _timed(doc(), "served-commit-rows-4k", (500.0, 550.0, 600.0), (70.0, 75.0, 110.0))
    fixed = _timed(doc(), "served-commit-rows-4k", (9.0, 9.0, 9.0), (30.0, 20.0, 25.0))
    got = {(m.ratio, m.metric): m for m in S.measures(S.retail(d, fixed), run="r1 (tail r2)")}
    served = got[("served", "tpot")]
    assert (served.median, served.lo, served.hi) == (pytest.approx(455 / 400), pytest.approx(1.0), pytest.approx(515 / 400))
    before = {(m.ratio, m.metric): m for m in S.measures(d, run="r1")}
    assert {k for k in before if got[k] != replace(before[k], run="r1 (tail r2)")} == {("served", "tpot"), ("compiled-bi", "tpot")}
    assert d["points"][0]["arms"]["served-commit-rows-4k"]["reps"][0]["metrics"]["time_per_output_token_s"] == 5.0


def test_a_tail_for_a_run_that_isnt_rendered_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["served.py", "r1", "--tail", "r9=r2"])
    with pytest.raises(SystemExit):
        S.main()
    assert "--tail names runs that aren't rendered: r9" in capsys.readouterr().err


def test_a_ratio_with_a_compiled_arm_counts_only_from_a_run_pinned_to_the_build_of_record():
    pinned = {(m.ratio, m.metric): m for m in S.measures(doc(), run="r1", pin="pinned")}
    assert {k: m.counts for k, m in pinned.items() if m.metric == "tpot"} == {
        ("served", "tpot"): "", ("compiled-bi", "tpot"): "", ("f", "tpot"): ""}
    for pin in ("unpinned", "pin-failed"):
        ms = S.measures(doc(), run="r1", pin=pin)
        assert {m.counts for m in ms} == {f"{pin}: not the Build of record's kernels"}
        assert {m.pin for m in ms} == {pin}
    eager = {m.ratio: m for m in S.measures(doc(commits=(EAGER_ROWS_4K,)), run="r2") if m.metric == "tpot"}
    assert {r: (m.pin, m.counts) for r, m in eager.items()} == {
        "eager": ("eager", ""), "eager-served": ("unpinned", "unpinned: not the Build of record's kernels"),
        "f": ("unpinned", "unpinned: not the Build of record's kernels")}
    h200 = doc()
    h200["identity"]["preset"]["gpu_id"] = "h200"
    assert {m.counts for m in S.measures(h200, run="r3", pin="pinned") if m.ratio == "served"} == {
        "H200's served preset on RTX PRO 6000"}
    failed = S.measures(doc(commit_passed=False), run="r4", pin="pinned")
    assert {m.ratio: m.counts for m in failed if m.metric == "tpot"} == {"served": "gates failed", "compiled-bi": "gates failed",
                                                                        "f": ""}


def test_the_harness_commit_counts_as_rows_sha512_whatever_its_name():
    ms = S.measures(doc(fp8=True, commits=(COMMIT, COMMIT_COMPILED)), run="r-harness")
    assert {m.ratio: m.counts for m in ms if m.metric == "tpot"}["eager"] == ""


def test_the_tables_say_which_ratios_count_and_each_runs_pin():
    ms = S.measures(doc(), run="r-pinned", pin="pinned") + S.measures(doc(), run="r-free")
    md = S.report(ms, {}, {"r-pinned": S.run_facts(doc()) | {"pin": "pinned"}, "r-free": S.run_facts(doc()) | {"pin": "unpinned"}})
    served = {line.split(" | ")[7]: line.rsplit(" | ", 1)[1] for line in md.splitlines() if line.startswith("| Qwen3-8B | 32 |")}
    assert served == {"r-pinned": "yes |", "r-free": "unpinned: not the Build of record's kernels |"}
    every = [line for line in md.splitlines() if line.startswith("| Qwen3-8B | r-") and "| decode |" in line]
    assert ["(not counted)" in line for line in every] == [True, False]
    assert "| r-pinned | Qwen3-8B | served-commit-rows-4k | 1 |  | none | pinned |" in md


def test_a_runs_pin_is_its_pin_check(tmp_path):
    out = tmp_path / "r1" / "out"
    out.mkdir(parents=True)
    (out / "result.json").write_text('{"points": []}')
    assert S.load("r1", tmp_path, tmp_path / "scratch") == ({"points": []}, "unpinned")
    (out / "pin_check.txt").write_text("[served-pin] /d: winners equal to artifact 533721fc9d37833a's\n")
    assert S.load("r1", tmp_path, tmp_path / "scratch")[1] == "pinned"
    (out / "pin_check.txt").write_text("[served-pin] /d: a3/x.best_config: winner differs from artifact 533721fc9d37833a's\n")
    assert S.load("r1", tmp_path, tmp_path / "scratch")[1] == "pin-failed"


def test_two_runs_of_one_model_keep_their_own_rows():
    ms = S.measures(doc(), run="r-compiled") + S.measures(doc(commits=(EAGER_ROWS_4K,)), run="r-eager")
    md = S.report(ms, {"r-compiled": {32: (63.0, 4.0, 67.0)}})
    f_rows = [line for line in md.splitlines() if line.startswith("| Qwen3-8B | r-") and "| decode |" in line]
    assert [row.split(" | ")[1] for row in f_rows] == ["r-compiled", "r-eager"]
    assert "| Qwen3-8B | 32 | 1.250×" in md and "| r-compiled | 32 | 63 | 4 | 67 |" in md
    assert {m.ratio for m in ms if m.run == "r-eager"} == {"eager-served", "eager", "f"}
