"""`tools/verity/check/live.py`: the live record a check writes as its steps go (on an injected clock), published as a new
object per write and read back as the newest, and `verity ci`'s rendering of running, passed, failed, stale and missing
records, where nothing recorded never reads as passed."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"repo_{name}", ROOT / "tools" / "verity" / "check" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


LIVE = _load("live")
T0 = 1_793_000_000.0                    # 2026-10-26T07:33:20Z


class Clock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(LIVE, "CLOCK", c)
    return c


def test_the_record_follows_each_step_and_each_write_is_a_new_key(tmp_path, clock):
    put: dict[str, dict] = {}
    live = LIVE.Live(tmp_path, run_id="r20261026-093320-abcd", put=lambda k, d: put.__setitem__(k, json.loads(d)),
                     commit="c" * 40, what="check", machine="vy-node")
    live.plan(["preflight-lock", "pytest", "circuit-check", "lean-audit"])
    live.ended({"step": "preflight-lock", "status": "passed", "seconds": 2.0})
    live.started("pytest")
    live.started("circuit-check")
    (tmp_path / "pytest.log").write_text("cached  verity                       passed at abc\npassed  research                     3 passed in 9 s\n"
                                         "failed  verity-check                 1 passed, 1 failed in 4 s\nsome other line\n")
    (tmp_path / "circuit-check.json.progress.jsonl").write_text(
        json.dumps({"target": "a", "failures": 0}) + "\n" + json.dumps({"target": "b", "failures": 2}) + "\n")
    clock.t += 125
    first = live.write()
    assert first["state"] == "running" and first["seconds"] == 125 and first["at_utc"] == "2026-10-26T07:35:25Z"
    by = {s["step"]: s for s in first["steps"]}
    assert [s["step"] for s in first["steps"]] == ["preflight-lock", "pytest", "circuit-check", "lean-audit"]
    assert by["preflight-lock"] == {"step": "preflight-lock", "state": "passed", "seconds": 2.0, "started_utc": "2026-10-26T07:33:18Z"}
    assert by["pytest"]["state"] == "running" and by["pytest"]["seconds"] == 125 and by["pytest"]["started_utc"] == "2026-10-26T07:33:20Z"
    assert by["pytest"]["progress"] == {"suites_passed": 1, "suites_failed": 1, "suites_cached": 1, "failed": ["verity-check"]}
    assert by["circuit-check"]["progress"] == {"targets_done": 2, "targets_failed": 1, "failed": ["b"]}
    assert by["lean-audit"] == {"step": "lean-audit", "state": "pending"}
    clock.t += 60
    live.ended({"step": "pytest", "status": "failed", "seconds": 185.0, "detail": "1 suite failed"})
    final = live.close("failed")
    assert sorted(put) == ["live/r20261026-093320-abcd/000000.json", "live/r20261026-093320-abcd/000001.json"]
    assert put["live/r20261026-093320-abcd/000001.json"] == final
    assert json.loads((tmp_path / LIVE.LIVE_FILE).read_text()) == final
    by = {s["step"]: s for s in final["steps"]}
    assert final["state"] == "failed" and by["pytest"]["state"] == "failed" and by["pytest"]["seconds"] == 185.0
    assert by["circuit-check"]["seconds"] == 185 and final["commit"] == "c" * 40 and final["machine"] == "vy-node"


def test_the_record_names_the_machine_research_placed_the_run_on(tmp_path, clock):
    (tmp_path / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-nebius-cpu-1"}}}))
    assert LIVE.Live(tmp_path, run_id="r1").doc()["machine"] == "vy-nebius-cpu-1"
    (tmp_path / "job.json").write_text("{}")
    assert LIVE.Live(tmp_path, run_id="r1").doc()["machine"] == LIVE.platform.node()


def test_a_publish_that_fails_leaves_the_run_dir_record_and_says_so_once(tmp_path, clock, capsys):
    def put(key, data):
        raise OSError("denied")

    live = LIVE.Live(tmp_path, run_id="r1", put=put)
    live.plan(["pytest"])
    live.write()
    live.write()
    assert json.loads((tmp_path / LIVE.LIVE_FILE).read_text())["seq"] == 1 and live.failures == 2
    assert capsys.readouterr().out.count("publishing check-live.json failed (OSError: denied)") == 1


def test_steps_report_their_starts_and_ends_to_the_live_record(tmp_path, monkeypatch):
    check = _load("check")
    monkeypatch.setattr(check.LIVE, "CLOCK", Clock())
    live = check.LIVE.Live(tmp_path, run_id="r1")
    steps = check.Steps(tmp_path, keep_going=True, live=live)
    live.plan(["pytest", "lean-audit"])
    rec = check._step("pytest", [sys.executable, "-c", "print('passed  research  3 passed in 9 s')"], tmp_path, tmp_path, False, steps)
    assert rec["status"] == "passed"
    by = {s["step"]: s for s in live.doc()["steps"]}
    assert by["pytest"]["state"] == "passed" and by["pytest"]["progress"]["suites_passed"] == 1
    assert by["lean-audit"]["state"] == "pending"


def test_the_newest_record_of_each_run_is_read_back_from_the_remote(tmp_path, clock):
    from research.store.remote import FsRemote

    remote = FsRemote(tmp_path / "bucket")

    def put(key: str, data: bytes) -> None:
        (p := tmp_path / "staged").write_bytes(data)
        remote.put(key, p)

    for rid in ("r20261025-000000-aaaa", "r20261026-000000-bbbb"):
        (tmp_path / rid).mkdir()
        live = LIVE.Live(tmp_path / rid, run_id=rid, put=put)
        live.plan(["pytest"])
        live.started("pytest")
        live.write()
        clock.t += 10
        live.write()
    docs = LIVE.live_docs(remote, "r20261026")
    assert list(docs) == ["r20261026-000000-bbbb"] and docs["r20261026-000000-bbbb"]["seq"] == 1
    assert LIVE.live_doc(remote, "r20261025-000000-aaaa")["seq"] == 1 and LIVE.live_doc(remote, "r20261027-000000-cccc") is None


NOW = LIVE.parse_utc("2026-10-28T23:30:00Z")
RUNNING = {"schema": LIVE.SCHEMA, "run_id": "r20261028-220000-abcd", "commit": "d40fe21d26aa" + "0" * 28, "what": "check",
           "machine": "vy-nebius-cpu-3", "state": "running", "started_utc": "2026-10-28T22:00:00Z", "at_utc": "2026-10-28T23:28:00Z",
           "seconds": 5280.0, "seq": 30,
           "steps": [{"step": "preflight-lock", "state": "passed", "seconds": 1.0, "started_utc": "2026-10-28T22:00:00Z"},
                     {"step": "pytest", "state": "running", "started_utc": "2026-10-28T22:01:00Z", "seconds": 5220.0,
                      "progress": {"suites_passed": 40, "suites_failed": 1, "suites_cached": 30, "failed": ["verity-vllm"]}},
                     {"step": "circuit-check", "state": "running", "started_utc": "2026-10-28T23:00:00Z", "seconds": 1680.0,
                      "progress": {"targets_done": 812, "targets_failed": 0, "failed": []}},
                     {"step": "lean-audit", "state": "failed", "seconds": 300.0, "started_utc": "2026-10-28T22:01:00Z",
                      "detail": "a guarantee's record is stale"},
                     {"step": "lean-suites", "state": "pending"}]}
BASIS = {"pytest": [3000.0, 2400.0, 2700.0], "circuit-check": [600.0]}


def lines(text: str) -> dict[str, str]:
    return {ln.split()[0]: ln for ln in text.splitlines() if ln.startswith("  ") and ln.split()}


def test_inspect_shows_a_running_checks_steps_etas_and_reds(clock):
    text = LIVE.render_inspect(RUNNING["run_id"], RUNNING, BASIS, NOW)
    assert text.splitlines()[0] == "check r20261028-220000-abcd  running  check of d40fe21d26aa on vy-nebius-cpu-3"
    row = lines(text)
    assert "passed" in row["preflight-lock"]
    assert "running" in row["pytest"] and "1h29m" in row["pytest"] and "over by 44m00s (median 45m00s of 3)" in row["pytest"]
    assert "suites 40 passed, 1 failed, 30 reused" in row["pytest"]
    assert "~0s" not in row["circuit-check"] and "30m00s" in row["circuit-check"] and "over by 20m00s (median 10m00s of 1)" in row["circuit-check"]
    assert "targets 812 done, 0 failed" in row["circuit-check"]
    assert "failed" in row["lean-audit"] and "pending" in row["lean-suites"]
    assert text.splitlines()[-3:] == ["reds so far:", "  pytest: 1 suite failed: verity-vllm",
                                      "  lean-audit failed: a guarantee's record is stale"]


def test_a_step_without_a_basis_has_an_unknown_eta():
    row = lines(LIVE.render_inspect(RUNNING["run_id"], RUNNING, {}, NOW))
    assert "unknown (no recorded basis)" in row["pytest"] and "unknown (no recorded basis)" in row["circuit-check"]
    assert "eta" not in row["lean-audit"] and "unknown" not in row["lean-audit"]


def test_an_ended_check_shows_its_unstarted_steps_as_not_run():
    ended = {**RUNNING, "state": "failed", "steps": [{**s, "state": "stopped"} if s["state"] == "running" else s for s in RUNNING["steps"]]}
    text = LIVE.render_inspect(ended["run_id"], ended, BASIS, NOW)
    assert text.splitlines()[0].split()[2] == "failed"
    row = lines(text)
    assert "not run" in row["lean-suites"] and "stopped" in row["pytest"] and "~" not in row["pytest"]


def test_a_passed_check_shows_its_cache_use():
    passed = {**RUNNING, "state": "passed", "steps": [
        {"step": "pytest", "state": "passed", "seconds": 2000.0, "cache": {"stored": "k"}},
        {"step": "circuit-check", "state": "passed", "seconds": 90.0, "cache": {"hits": 8600, "stored": 13}},
        {"step": "lean-agreement", "state": "passed", "seconds": 0.0, "cache": {"hit": "k2"}}]}
    text = LIVE.render_inspect(passed["run_id"], passed, BASIS, NOW)
    row = lines(text)
    assert "passed (8600 reused)" in row["circuit-check"] and "passed (cached)" in row["lean-agreement"]
    assert text.splitlines()[-1] == "reds so far: none recorded"


def test_a_stale_running_record_is_unknown_or_what_the_store_says_never_running():
    later = NOW + 3600
    text = LIVE.render_inspect(RUNNING["run_id"], RUNNING, BASIS, later)
    assert text.splitlines()[0].split()[2] == "unknown" and "running?" in lines(text)["pytest"]
    assert "unknown (its record is stale)" in lines(text)["pytest"] and "over by" not in text
    assert LIVE.render_inspect(RUNNING["run_id"], RUNNING, BASIS, later, "failed").splitlines()[0].split()[2] == "failed"


def test_a_run_with_no_live_record_is_unknown():
    text = LIVE.render_inspect("r20261028-000000-ffff", None, BASIS, NOW)
    assert text.startswith("check r20261028-000000-ffff: unknown: no live record") and "passed" not in text


def test_the_basis_takes_passes_without_a_cache_hit_newest_first():
    records = [[{"step": "pytest", "state": "passed", "seconds": 10.0}, {"step": "circuit-check", "state": "failed", "seconds": 5.0}],
               [{"step": "pytest", "status": "passed", "seconds": 20.0}, {"step": "lean-agreement", "status": "passed", "seconds": 0.0,
                                                                           "cache": {"hit": "k"}}],
               [{"step": "pytest", "status": "passed", "seconds": 30.0}, {"step": "circuit-check", "status": "passed", "seconds": 7.0,
                                                                           "cache": {"hits": 3}}]]
    assert LIVE.basis(records, 2) == {"pytest": [10.0, 20.0], "circuit-check": [7.0]}


def test_ls_lists_live_and_catalogued_runs_newest_first_and_judges_stale_ones():
    stale = {**RUNNING, "run_id": "r20261028-200000-0000", "at_utc": "2026-10-28T21:00:00Z"}
    done = {**RUNNING, "run_id": "r20261028-210000-1111", "state": "passed", "at_utc": "2026-10-28T22:00:00Z"}
    checks = [{"run_id": "r20261027-100000-2222", "state": "done", "commit": "e" * 40, "machine": "vy-nebius-2",
               "started_utc": "2026-10-27T10:00:00Z", "seconds": 4000.0, "result": None}]
    rows = LIVE.ls_rows({RUNNING["run_id"]: RUNNING, stale["run_id"]: stale, done["run_id"]: done}, checks, NOW,
                        {"r20261028-200000-0000": "cancelled"}.get)
    assert [(r["run_id"], r["state"]) for r in rows] == [
        ("r20261028-220000-abcd", "running"), ("r20261028-210000-1111", "passed"),
        ("r20261028-200000-0000", "cancelled (no final live record)"), ("r20261027-100000-2222", "passed")]
    text = LIVE.render_ls(rows, NOW)
    assert text.splitlines()[0].split() == ["run", "state", "commit", "machine", "started", "elapsed", "what"]
    assert "1h30m" in text.splitlines()[1] and "1h06m" in text.splitlines()[4]
