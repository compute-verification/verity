"""`tools/verity/check/train_times.py` over tiny synthetic trains: the phases around the steps, the critical path walked back
through a group and a job's `after`, the long pole, and the per-step stats (a reused suite counting 0)."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_train_times", ROOT / "tools" / "verity" / "check" / "train_times.py")
TT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TT)

T0 = datetime(2026, 10, 19, 1, 0, 0, tzinfo=timezone.utc)
CI = """
[[job]]
name = "pytest"
group = 1
[[job]]
name = "flock-circuit-build"
group = 2
[[job]]
name = "circuit-check"
group = 2
[[job]]
name = "flock-rust-tests"
group = 2
[[job]]
name = "lean-build"
group = 3
[[job]]
name = "lean-audit"
group = 4
[[job]]
name = "lean-suites"
group = 6
after = "lean-build"
"""


def at(s: float) -> str:
    return (T0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")


def train(tmp: Path, name: str, steps: dict[str, tuple[int, int]], end: int, suites: list[dict] = ()) -> Path:
    """A train's directory: each step (start, seconds) after T0, a slot taken at 300 s after joining at 120 s, its commit at -600 s,
    registered at 0 s and started at 120 s."""
    d = tmp / name
    d.mkdir()
    (d / "steps.jsonl").write_text("".join(json.dumps({"step": n, "status": "passed", "seconds": float(s), "ended_utc": at(b + s)}) + "\n"
                                           for n, (b, s) in steps.items()))
    (d / "slot.json").write_text(json.dumps({"state": "held", "since": at(120), "took": at(300)}))
    (d / "suites.json").write_text(json.dumps(suites))
    (d / "train.json").write_text(json.dumps({"run": name, "commit": "c" * 40, "machine": "vy-nebius-1", "verdict": "passed",
                                              "committed_utc": at(-600), "registered_utc": at(0), "start_utc": at(120),
                                              "end_utc": at(end)}))
    return d


def jobs(tmp: Path) -> dict:
    (tmp / "ci.toml").write_text(CI)
    return TT.jobs(tmp / "ci.toml")


PRE = {"preflight-lock": (400, 1), "preflight-lints": (401, 9)}


def test_the_circuit_group_is_the_critical_path_when_it_ends_last(tmp_path):
    d = train(tmp_path, "r20261019-010000-aaaa", {**PRE, "pytest": (410, 100), "flock-circuit-build": (410, 5),
                                                  "circuit-check": (415, 300), "flock-rust-tests": (715, 20),
                                                  "lean-build": (410, 50), "lean-suites": (460, 200), "lean-audit": (410, 120)},
              end=800, suites=[{"suite": "verity/ml/flock", "status": "passed", "seconds": 90.0},
                               {"suite": "tools/research", "status": "cached", "seconds": 0.1}])
    c = TT.costs(TT.load_dir(d), jobs(tmp_path))
    assert c["long_pole"] == "circuit-check"
    assert [n for n, _ in c["path"]] == ["queued", "launch", "slot wait", "setup", "preflight-lock", "preflight-lints",
                                         "flock-circuit-build", "circuit-check", "flock-rust-tests", "finish", "other"]
    path = dict(c["path"])
    assert (path["queued"], path["launch"], path["slot wait"], path["setup"], path["finish"]) == (600, 120, 180, 100, 65)
    assert path["other"] == 0 and sum(path.values()) == c["total_s"] == 1400
    assert c["check_s"] == 335
    assert c["times"]["pytest: verity/ml/flock"] == 90 and "pytest: tools/research" not in c["times"]


def test_lean_suites_chain_back_through_its_after_job(tmp_path):
    d = train(tmp_path, "r20261019-020000-bbbb", {**PRE, "pytest": (410, 100), "flock-circuit-build": (410, 5),
                                                  "circuit-check": (415, 100), "lean-build": (410, 50),
                                                  "lean-suites": (461, 400), "lean-audit": (410, 120)}, end=900)
    c = TT.costs(TT.load_dir(d), jobs(tmp_path))
    assert [n for n, _ in c["path"]][6:8] == ["lean-build", "lean-suites"]
    assert c["long_pole"] == "lean-suites"
    assert dict(c["path"])["other"] == 1                # lean-suites started a second after lean-build ended


def test_report_takes_median_p90_and_max_and_counts_a_reused_suite_as_zero(tmp_path):
    js = jobs(tmp_path)
    steps = {**PRE, "pytest": (410, 100), "circuit-check": (410, 300)}
    a = train(tmp_path, "r20261019-010000-aaaa", steps, end=800,
              suites=[{"suite": "verity/ml/flock", "status": "passed", "seconds": 60.0}])
    b = train(tmp_path, "r20261019-020000-bbbb", {**steps, "circuit-check": (410, 500)}, end=1000)
    rep = TT.report([TT.load_dir(a), TT.load_dir(b)], js)
    assert rep["steps"]["circuit-check"] == {"n": 2, "median": 400, "p90": 500, "max": 500}
    assert rep["steps"]["pytest: verity/ml/flock"] == {"n": 2, "median": 30, "p90": 60, "max": 60}
    assert rep["steps"]["slot wait"]["median"] == 180
    assert list(rep["path"])[:3] == ["queued", "circuit-check", "slot wait"]
    assert rep["path"]["circuit-check"]["long_pole_in"] == 2
    text = TT.render(rep)
    assert "| r20261019-020000-bbbb | cccccccccccc | vy-nebius-1 | passed | 10.0 | 2.0 | 3.0 | 1.7 |" in text
    assert "| pytest: verity/ml/flock | 2 | 0.5 | 1.0 | 1.0 |" in text


def test_the_command_reads_train_directories(tmp_path, capsys):
    d = train(tmp_path, "r20261019-010000-aaaa", {**PRE, "pytest": (410, 100)}, end=600)
    assert TT.main(["--from", str(d), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["trains"][0]["long_pole"] == "pytest" and rep["steps"]["pytest"]["max"] == 100
