"""`research lean-times`: each Lean task's wall time from the store's attempts, records runs (their work, wait and coldness from
their run record) and checks' Lean steps, summarized per kind and package."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from research import lean_times as L
from research.store.local import LocalStore

T = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
ARGV = ["python3", "tools/verity/lean/lean_changed.py", "--records", "Security", "--update"]
CHECK_ARGV = ["python3", "tools/verity/check/slot.py", "--", "uv", "run", "python", "tools/verity/check/check.py", "--keep-going"]
STDOUT = """lean-records: taking the slot path: check.py on /c/lean-records/build-1
AUDIT /c/lean-records/build-1/Security: PASS  24170 declarations in 646 modules; axioms propext; 1003 guarantees; {'build': 46.3}
AUDIT /c/lean-records/build-1/Security/Proofs: PASS  160227 declarations in 3191 modules; axioms propext; 0 guarantees; {}
lean-records: Security: check.py exit 0 in 776 s on /c/lean-records/build-1 (69 files changed there), without the kernel replay
AUDIT /c/lean-records/build-1/Security/Properties: FAIL  1387 declarations in 54 modules; axioms propext; 302 guarantees; {}
AUDIT /c/lean-records/build-1/Security/Proofs: PASS  160227 declarations in 3191 modules; axioms propext; 0 guarantees; {}
lean-records: Security/Properties: check.py --dev-cache exit 1 in 136 s on /c/lean-records/build-1 (1 files changed there)
"""
BUILD = ("✔ [1/9] Built Security.A (456ms)\n✔ [2/9] Built Security.A:c.o (12ms)\n⚠ [3/9] Replayed Security.B\n"
         "✔ [4/9] Built ArkLib.C (6.3s)\n")
BUILD_PROOFS = "✔ [1/4] Built Security.Proofs.D (1m 3s)\n✔ [2/4] Built ArkLib.C (2.0s)\n"
FILES = {"stdout.log": STDOUT, "slot.json": json.dumps({"state": "held", "since": "2026-10-28T10:00:00Z", "took": "2026-10-28T10:01:00Z"}),
         "lean-audit/audit-Security/build.log": BUILD, "lean-audit/audit-Proofs/build-roots.log": BUILD_PROOFS,
         "lean-audit/audit-Proofs/report.json": "Built Not.A.Build.Log (1ms)"}


def _doc(argv=ARGV, wall=2400.0, end="2026-10-28T11:00:00Z", **ex) -> dict:
    return {"argv": argv, "execution": {"wall_s": wall, "end_utc": end, **ex}}


def _meta(path="slot", status="passed", pkg="Security") -> dict:
    return {"kind": L.FAST_KIND, "validation": {"status": status}, "workload_fingerprint": {"package": pkg, "path": path}}


def _read(files: dict[str, str]):
    return lambda p: files[p].encode()


def test_a_records_run_is_its_wall_with_its_work_wait_and_coldness_beside_it():
    t = L.records_task("r20261028-100000-aaaa", _doc(machine="vy-nebius-cpu-2"), _meta(), list(FILES), _read(FILES))
    assert t["kind"] == "records" and t["packages"] == ["Security"] and t["node"] == "vy-nebius-cpu-2"
    assert t["ended"] == "2026-10-28T11:00:00Z" and t["outcome"] == "passed" and t["path"] == "slot"
    assert t["minutes"] == 40.0, "the attempt's wall"
    assert t["work_min"] == (776 + 136) / 60, "each package's check.py, the --dev-cache one included"
    assert t["wait_min"] == (2400 - 912) / 60 and t["check_slot_wait_min"] == 1.0
    assert t["built"] == 3, "the distinct modules Lake compiled in the build logs: not c.o, a replay, or another file's lines"
    assert t["modules"] == 646 + 3191 + 54, "each audited package's modules once"


def test_a_run_that_audited_nothing_works_its_wall_less_its_check_slot_wait():
    files = {"slot.json": FILES["slot.json"], "stdout.log": "lean-records: reused the record of a passing audit\n"}
    t = L.records_task("r20261028-100000-aaaa", _doc(wall=300.0, remote={"machine": "n2"}), _meta(path="reused"), list(files),
                       _read(files))
    assert (t["minutes"], t["work_min"], t["wait_min"]) == (5.0, 4.0, 1.0)
    assert (t["built"], t["modules"], t["path"], t["node"]) == (0, None, "reused", "n2")


def test_a_records_run_without_its_result_is_unfinished_under_its_argv_package():
    def unread(p):
        raise AssertionError(f"an unfinished run's record isn't read ({p})")

    t = L.records_task("r20261028-100000-aaaa", _doc(), None, ["stdout.log"], unread)
    assert t["packages"] == ["Security"] and t["minutes"] is None and t["outcome"] == "unfinished"
    assert L.records_task("r20261028-100000-aaaa", _doc(argv=["bash"]), {"kind": "other"}, [], unread)["packages"] == [L.NONE]


def _step(step, status="passed", seconds=60.0, ended="2026-10-28T11:00:00Z", **k) -> dict:
    return {"step": step, "status": status, "seconds": seconds, "ended_utc": ended, **k}


def test_a_checks_lean_steps_are_tasks_lean_audit_under_each_package_it_audited_and_lean_suites_one_per_suite():
    meta = {"kind": "check/v1", "steps": [
        _step("pytest", seconds=900.0), _step("lean-audit", seconds=1800.0), _step("lean-suites", "failed", 1200.0),
        _step("lean-build", "skipped"), _step("lean-agreement", seconds=900.0), _step("lean-changed", "stopped", None),
        _step("lean-audit:verity/core/service/flock", seconds=120.0, cache={"hit": True}), "not a step"],
        "reuse": {"steps": {"lean-audit": {"reran": ["Security", "Security/Proofs"], "reused": ["X"]},
                            "lean-suites": {"reran": [{"suite": "verity/ml/flock", "seconds": 1100.0, "forced_by": []},
                                                      {"suite": "benchmarks/pouw", "seconds": None}]}}}}
    got = L.check_tasks("r20261028-100000-cccc", _doc(CHECK_ARGV, host="vy-nebius-cpu-1"), meta)
    assert [(t["kind"], t["packages"], t["minutes"], t["outcome"]) for t in got] == [
        ("lean-audit", ["Security", "Security/Proofs"], 30.0, "passed"),
        ("lean-suites", ["verity/ml/flock"], 1100 / 60, "failed"),
        ("lean-suites", ["benchmarks/pouw"], None, "unfinished"),
        ("lean-changed", [L.NONE], None, "unfinished"),
        ("lean-audit", ["verity/core/service/flock"], 2.0, "passed")]
    assert {t["node"] for t in got} == {"vy-nebius-cpu-1"} and [t["cached"] for t in got] == [False] * 4 + [True]


def test_a_step_that_ran_no_package_or_suite_is_one_task_under_none():
    meta = {"steps": [_step("lean-audit", seconds=6.0), _step("lean-suites", seconds=12.0)], "reuse": {"steps": {}}}
    assert [(t["kind"], t["packages"], t["minutes"]) for t in L.check_tasks("r20261028-100000-cccc", _doc(CHECK_ARGV), meta)] == [
        ("lean-audit", [L.NONE], 0.1), ("lean-suites", [L.NONE], 0.2)]


def test_the_window_holds_the_tasks_that_ended_in_it_both_ends_included():
    start, end = datetime(2026, 10, 28, 10, tzinfo=UTC), T
    tasks = [{"ended": e} for e in ("2026-10-28T10:00:00Z", "2026-10-28T12:00:00Z", "2026-10-28T09:59:59Z", "2026-10-28T12:00:01Z",
                                    None, "garbled")]
    assert [t["ended"] for t in L.within(tasks, start, end)] == ["2026-10-28T10:00:00Z", "2026-10-28T12:00:00Z"]


def _rec(run, minutes, pkg="Security", work=None, wait=1.0) -> dict:
    t = {"kind": "records", "packages": [pkg], "run": run, "node": "n", "minutes": minutes, "outcome": "unfinished"}
    if minutes is None:
        return t
    return {**t, "outcome": "passed", "work_min": minutes - wait if work is None else work, "wait_min": wait,
            "check_slot_wait_min": 0.0, "path": "slot", "built": 3, "modules": 9}


def test_percentiles_are_nearest_rank():
    assert L.pct([3.0], 0.95) == 3.0
    assert L.pct([float(v) for v in range(1, 21)], 0.5) == 11.0 and L.pct([float(v) for v in range(1, 21)], 0.95) == 19.0
    assert L.pct([5.0, 1.0, 3.0, 2.0], 0.5) == 3.0


def test_the_summary_is_per_kind_and_package_with_all_first_and_records_wait_and_work():
    tasks = [_rec(f"r{i}", float(m)) for i, m in enumerate((10, 20, 30, 40))] + [_rec("r9", None), _rec("r8", 16.0, "Other", work=2)]
    tasks.append({"kind": "lean-audit", "packages": ["Security", "Other"], "run": "c", "node": "n", "minutes": 50.0, "outcome": "passed"})
    s = L.summary(tasks)
    assert list(s) == ["records", "lean-audit"] and list(s["records"]) == [L.ALL, "Security", "Other"]
    assert s["records"][L.ALL] == {"n": 5, "unfinished": 1, "over": 4, "p50_min": 20.0, "p95_min": 40.0, "max_min": 40.0,
                                   "wait_p50_min": 1.0, "wait_p95_min": 1.0, "wait_max_min": 1.0, "work_p50_min": 19.0,
                                   "work_p95_min": 39.0, "work_max_min": 39.0, "work_over": 3}
    assert s["records"]["Other"]["work_over"] == 0 and s["records"]["Security"]["unfinished"] == 1
    assert s["lean-audit"] == {k: {"n": 1, "unfinished": 0, "over": 1, "p50_min": 50.0, "p95_min": 50.0, "max_min": 50.0}
                               for k in (L.ALL, "Other", "Security")}
    assert L.summary(tasks, over=45.0)["lean-audit"][L.ALL]["over"] == 1 and L.summary(tasks, over=45.0)["records"][L.ALL]["over"] == 0
    assert L.summary([_rec("r9", None)])["records"][L.ALL] == {"n": 0, "unfinished": 1, "over": 0, "work_over": 0}


def test_the_table_names_each_task_over_the_limit_the_longest_first():
    tasks = [_rec("r20261028-100000-aaaa", 20.0, work=5.0), _rec("r20261028-100000-bbbb", 90.0, "verity/core/protocols/remote_bandwidth/lean"),
             _rec("r20261028-100000-dddd", 3.0), _rec("r20261028-100000-eeee", None),
             {"kind": "lean-suites", "packages": ["verity/ml/flock"], "run": "r20261028-100000-cccc", "node": "n", "minutes": 33.0,
              "outcome": "failed"}]
    out = L.render(L.summary(tasks), tasks, T.replace(hour=0), T)
    assert out.startswith("Lean tasks that ended 2026-10-28 00:00Z to 2026-10-28 12:00Z")
    assert "verity/core/protocols/remote_bandwidth/lean " in out, "the longest package path whole"
    assert "records runs over 15 min: 2, 1 of them over 15 min of Lean work" in out
    slow = out.split("records runs over")[1]
    assert slow.index("r20261028-100000-bbbb") < slow.index("r20261028-100000-aaaa") and "r20261028-100000-dddd" not in slow
    assert "check steps' tasks over 15 min: 1" in out and "r20261028-100000-cccc  lean-suites" in out


@pytest.fixture
def remote(tmp_path):
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    return tmp_path, cfg


def _publish(node: LocalStore, tmp, rid: str, argv: list[str], *, meta: dict | None, files: dict[str, str] | None = None,
             end: str, wall: float = 2400.0, kind: str = L.FAST_KIND) -> None:
    """An attempt another machine published, as `research run` does: its result (``meta``) and its run record (``files``)."""
    outputs = {}
    if meta is not None:
        outputs["result"] = node.put_artifact(kind, meta)
    if files is not None:
        tree = tmp / "records" / rid
        for rel, text in files.items():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text(text)
        outputs["run_record"] = node.put_artifact("run-record/v1", {}, tree=tree)
    node.put_attempt({
        "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "python3", "version": None}, "derivation": "drv:" + "0" * 64,
        "reuse_key": None, "argv": argv, "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
        "source": {"commit": "0" * 40, "dirty": False}, "closure": None,
        "execution": {"wall_s": wall, "end_utc": end, "machine": "vy-nebius-cpu-2"}, "cost": None,
        "status": {"state": "done" if meta else "failed", "rc": 0 if meta else 143}, "outputs": outputs, "telemetry": None})
    for art in outputs.values():
        node.push(art)
    node.push_attempt(rid)


def test_lean_times_reads_the_remotes_attempts_of_the_window(remote, capsys):
    tmp, cfg = remote
    node = LocalStore(tmp / "node", config_path=cfg)
    _publish(node, tmp, "r20261028-100000-aaaa", ARGV, meta=_meta(), files=FILES, end="2026-10-28T10:40:00Z")
    _publish(node, tmp, "r20261028-100500-bbbb", ARGV, meta=None, files={"stdout.log": ""}, end="2026-10-28T10:30:00Z")
    _publish(node, tmp, "r20261028-101000-cccc", CHECK_ARGV, kind="check/v1", end="2026-10-28T11:30:00Z",
             meta={"kind": "check/v1", "steps": [_step("lean-audit", seconds=1500.0, ended="2026-10-28T11:20:00Z")],
                   "reuse": {"steps": {"lean-audit": {"reran": ["Security"]}}}})
    _publish(node, tmp, "r20261027-080000-dddd", ARGV, meta=_meta(), files=FILES, end="2026-10-27T08:40:00Z")
    _publish(node, tmp, "r20261028-101500-eeee", ["python3", "-m", "pytest"], meta=None, end="2026-10-28T10:20:00Z")
    _publish(node, tmp, "r20261028-115500-ffff", ARGV, meta=_meta(), files=FILES, end="2026-10-28T12:30:00Z")

    reader = LocalStore(tmp / "reader", config_path=cfg)
    assert L.main(["--until", "2026-10-28T12:00:00Z", "--hours", "6", "--json"], store=reader) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["window"] == {"start": "2026-10-28T06:00:00Z", "end": "2026-10-28T12:00:00Z"} and out["over_min"] == 15.0
    assert [(t["kind"], t["run"], t["outcome"]) for t in out["tasks"]] == [
        ("lean-audit", "r20261028-101000-cccc", "passed"),
        ("records", "r20261028-100500-bbbb", "unfinished"), ("records", "r20261028-100000-aaaa", "passed")]
    rec = out["tasks"][2]
    assert (rec["minutes"], rec["built"], rec["modules"], rec["node"]) == (40.0, 3, 3891, "vy-nebius-cpu-2")
    recs = out["summary"]["records"]
    assert recs["Security"] == recs[L.ALL] and (recs[L.ALL]["n"], recs[L.ALL]["unfinished"]) == (1, 1)
    assert out["summary"]["lean-audit"]["Security"]["p95_min"] == 25.0
    assert not [p for p in (tmp / "reader" / "objects").rglob("*") if p.is_file()], "the run records' blobs are read, not kept"


def test_the_window_ends_now_by_default_and_the_text_table_is_the_default(remote, capsys, monkeypatch):
    tmp, cfg = remote
    node = LocalStore(tmp / "node", config_path=cfg)
    _publish(node, tmp, "r20261028-100000-aaaa", ARGV, meta=_meta(), files=FILES, end="2026-10-28T10:40:00Z")
    monkeypatch.setattr(L, "clock", lambda: T)
    assert L.main([], store=LocalStore(tmp / "reader", config_path=cfg)) == 0
    out = capsys.readouterr().out
    assert "ended 2026-10-27 12:00Z to 2026-10-28 12:00Z" in out and "records runs over 15 min: 1" in out
    assert "r20261028-100000-aaaa" in out.split("records runs over")[1]


def test_without_a_remote_it_says_so_and_answers_nothing(tmp_path, capsys):
    assert L.main(["--until", "2026-10-28T12:00:00Z"], store=LocalStore(tmp_path / "s")) == 1
    assert "no remote" in capsys.readouterr().err
