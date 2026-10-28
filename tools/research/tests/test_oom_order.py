"""pods/nebius/oom_order.py: each live run's oom_score_adj from its memory use against its research.mem reservation, so the OOM
killer takes a run over its reservation, or without one, before a run within it (INF-135: T1R's replay r20261019-110314-3fd7 stayed
inside its 200 GiB and was killed as the node's largest process beside a run at ~150 GB on 64 GB and one that reserved nothing)."""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research.pods.nebius import oom_order as O

T0 = datetime(2026, 10, 19, 11, 30, tzinfo=timezone.utc)
GB = 10**9
TOTAL = 1000 * GB


class Node:
    """A fake node under tmp_path: /proc (stat, status, oom_score_adj, environ, meminfo), a runs root and a ramlock dir."""

    def __init__(self, root: Path):
        self.proc, self.runs, self.ramlock = root / "proc", root / "runs", root / "ramlock"
        for d in (self.proc, self.runs, self.ramlock):
            d.mkdir(parents=True)
        (self.proc / "meminfo").write_text(f"MemTotal:       {TOTAL // 1024} kB\nMemFree:        1 kB\n")

    def run(self, rid: str, *, state: str = "running", pgid: int | None = None, gb: float | None = None, runner: int | None = None) -> str:
        d = self.runs / rid
        d.mkdir()
        (d / "status.json").write_text(json.dumps({"state": state, "pgid": pgid}))
        if gb is not None:
            (self.ramlock / f"research-{rid}.json").write_text(json.dumps({"owner": "research", "gb": gb, "pid": runner, "run_id": rid}))
        return str(d.resolve())

    def process(self, pid: int, *, pgid: int, anon_gb: float, shmem_gb: float = 0.0, adj: int = 0, run_dir: str | None = None,
                pss_gb: float | None = None, scope: str | None = None, cmdline: bytes = b"python\0") -> None:
        d = self.proc / str(pid)
        d.mkdir()
        (d / "cgroup").write_text(f"0::{scope}\n" if scope else "0::/user.slice/user-1001.slice/user@1001.service/app.slice\n")
        (d / "cmdline").write_bytes(cmdline)
        if pss_gb is not None:
            (d / "smaps_rollup").write_text(f"Rss:\t{int(anon_gb * GB) // 1024} kB\nPss:\t{int(pss_gb * GB) // 1024} kB\n"
                                            f"Pss_Anon:\t{int(pss_gb * GB) // 1024} kB\nPss_File:\t0 kB\nPss_Shmem:\t0 kB\n")
        (d / "stat").write_text(f"{pid} (py thon) S 1 {pgid} {pgid} 0 -1 4194560 0 0 0 0 0 0 0 0 20 0 1 0 100 0 0\n")
        (d / "status").write_text(f"Name:\tpython\nVmRSS:\t{int((anon_gb + shmem_gb + 9) * GB) // 1024} kB\n"
                                  f"RssAnon:\t{int(anon_gb * GB) // 1024} kB\nRssFile:\t{9 * GB // 1024} kB\n"
                                  f"RssShmem:\t{int(shmem_gb * GB) // 1024} kB\n")
        (d / "oom_score_adj").write_text(f"{adj}\n")
        (d / "environ").write_bytes(b"PATH=/usr/bin\0" + (f"RESEARCH_RUN_DIR={run_dir}\0".encode() if run_dir else b""))

    def adj(self, pid: int) -> int:
        return int((self.proc / str(pid) / "oom_score_adj").read_text())

    def main(self, *extra: str) -> int:
        return O.main(["--proc", str(self.proc), "--runs", str(self.runs), "--ramlock", str(self.ramlock),
                       "--log", str(self.proc.parent / "oom-order.jsonl"), *extra])

    def log(self) -> list[dict]:
        p = self.proc.parent / "oom-order.jsonl"
        return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


@pytest.fixture
def node(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "clock", lambda: T0)
    return Node(tmp_path)


def test_a_run_over_its_reservation_or_without_one_is_picked_before_one_within_it(node, capsys):
    t1r = node.run("r20261019-110314-3fd7", pgid=100, gb=215, runner=99)               # 200 GiB
    node.process(99, pgid=99, anon_gb=0.1, adj=7)                                      # its runner: no RESEARCH_RUN_DIR, its own group
    node.process(100, pgid=100, anon_gb=150, run_dir=t1r)
    node.process(101, pgid=100, anon_gb=50, shmem_gb=10, adj=7)                        # in the workload's group, environment cleared
    node.process(102, pgid=102, anon_gb=0.5, run_dir=t1r)                              # a worker that left the group
    node.process(103, pgid=200, anon_gb=0.5, run_dir=t1r)                              # T1R's, though in the neighbour's group
    over = node.run("r20261019-090000-aaaa", pgid=200, gb=64)
    node.process(200, pgid=200, anon_gb=150, run_dir=over)
    none = node.run("r20261019-100000-bbbb", pgid=300)
    node.process(300, pgid=300, anon_gb=20, run_dir=none)
    node.process(301, pgid=300, anon_gb=1, adj=O.DISABLED, run_dir=none)               # OOM-disabled by root on purpose
    node.process(900, pgid=900, anon_gb=300, adj=7)                                    # no run's: keeps its adj
    ended = node.run("r20261018-000000-cccc", state="done", pgid=400)
    node.process(400, pgid=400, anon_gb=5, adj=543, run_dir=ended)
    node.run("r20261019-100000-9999", pgid=600)                                        # a stale pgid: no member of its own in it
    node.process(600, pgid=600, anon_gb=4, adj=7)

    assert node.main() == 0
    assert [node.adj(p) for p in (100, 101, 102, 103)] == [0, 0, 0, 0], "within its reservation (211 of 215 GB)"
    assert node.adj(200) == 500 + round(500 * 86 / 1000) == 543, "86 GB over 64 GB"
    assert node.adj(300) == 500 + round(500 * 21 / 1000) == 510, "no reservation: all of its 21 GB is over"
    assert [node.adj(p) for p in (301, 900, 99, 400, 600)] == [-1000, 7, 7, 543, 7]
    recs = {r["run"]: r for r in node.log()}
    assert recs == {
        "r20261019-110314-3fd7": {"at_utc": "2026-10-19T11:30:00Z", "run": "r20261019-110314-3fd7", "use_gb": 211.0, "reserved_gb": 215.0,
                                  "adj": 0, "set": [[101, 7]], "dry_run": False},
        "r20261019-090000-aaaa": {"at_utc": "2026-10-19T11:30:00Z", "run": "r20261019-090000-aaaa", "use_gb": 150.0, "reserved_gb": 64.0,
                                  "adj": 543, "set": [[200, 0]], "dry_run": False},
        "r20261019-100000-bbbb": {"at_utc": "2026-10-19T11:30:00Z", "run": "r20261019-100000-bbbb", "use_gb": 21.0, "reserved_gb": 0.0,
                                  "adj": 510, "set": [[300, 0]], "dry_run": False}}
    assert capsys.readouterr().out.count("\n") == 3

    assert node.main() == 0 and len(node.log()) == 3, "a pass that changes nothing writes nothing"


def test_a_run_back_within_its_reservation_goes_back_to_0_and_one_further_over_goes_higher(node):
    rd = node.run("r20261019-120000-dddd", pgid=10, gb=100)
    node.process(10, pgid=10, anon_gb=80, adj=600, run_dir=rd)
    big = node.run("r20261019-120000-eeee", pgid=20, gb=10)
    node.process(20, pgid=20, anon_gb=2000, run_dir=big)
    assert node.main() == 0
    assert node.adj(10) == 0 and node.adj(20) == 1000, "at most 1000"
    assert node.log()[0]["set"] == [[10, 600]]


def test_a_check_slot_s_reservation_counts_for_its_run_and_the_largest_of_a_run_s_reservations_wins(node):
    """A run that reserves at launch and then takes a check slot holds two files (research.mem's reserved_bytes counts its largest);
    a slot's file names its run and the check's pid."""
    rd = node.run("r20261019-130000-ffff", pgid=30, gb=16)
    node.process(30, pgid=30, anon_gb=90, adj=600, run_dir=rd)
    (node.ramlock / "check-a.json").write_text(json.dumps({"owner": "check", "gb": 96, "pid": 30, "run_id": "r20261019-130000-ffff"}))
    (node.ramlock / "bad.json").write_text("{")
    other = node.run("r20261019-130000-0000", pgid=40)
    node.process(40, pgid=40, anon_gb=50, adj=600, run_dir=other)
    (node.ramlock / "hand-job.json").write_text(json.dumps({"owner": "x", "gb": 60, "pid": 40, "note": "by hand"}))
    assert node.main() == 0
    assert node.adj(30) == 0 and node.adj(40) == 0


def test_forked_workers_copy_on_write_pages_count_once_so_a_share_within_its_reservation_stays_at_0(node):
    """24 Oct, n2 03:44Z: a PRF-131 share (345 GB) whose ~278 GB parent forked 8 workers summed to 2,505 GB of RssAnon and was set
    to 1000, the node's first pick; split among the processes sharing them (Pss), its pages come to 300 GB."""
    share = node.run("r20261024-011205-2882", pgid=70, gb=345)
    node.process(70, pgid=70, anon_gb=278, pss_gb=278 / 9 + 10, run_dir=share)
    for i in range(1, 9):
        node.process(70 + i, pgid=70, anon_gb=278, pss_gb=278 / 9 + 0.5, run_dir=share)
    over = node.run("r20261024-011205-3333", pgid=90, gb=10)
    node.process(90, pgid=90, anon_gb=100, pss_gb=50, run_dir=over)
    node.process(91, pgid=90, anon_gb=100, run_dir=over)                              # no smaps_rollup: its Rss counts
    assert node.main() == 0
    assert all(node.adj(70 + i) == 0 for i in range(9)), "Pss 292 GB, within 345"
    assert node.adj(90) == node.adj(91) == 500 + round(500 * 140 / 1000), "still over: Pss 50 + Rss 100 against 10"
    assert {r["run"]: r["use_gb"] for r in node.log()} == {"r20261024-011205-3333": 150.0}


def test_workers_that_left_a_runs_group_and_environment_count_by_its_session_scope(node):
    """24 Oct, cpu-3: suites.py's verity-circuit-check run e48b (no reservation) held 383 GB in its session scope, read here at
    0.1 GB, so the kernel took small runs and a reserved 80 GB verify first. Its runner stays out; a scope two runs share counts
    for neither."""
    s = "/user.slice/user-1001.slice/session-14429.scope"
    suite = node.run("r20261024-041008-e48b", pgid=10)
    node.process(9, pgid=9, anon_gb=0.1, scope=s, cmdline=b"python3\0-m\0research\0run\0--id\0x\0")    # its runner
    node.process(10, pgid=10, anon_gb=0.1, run_dir=suite, scope=s)
    node.process(11, pgid=11, anon_gb=383, scope=s)                                                   # xdist worker, scrubbed
    verify = node.run("r20261024-024255-adf9", pgid=20, gb=130)
    node.process(20, pgid=20, anon_gb=80, run_dir=verify, scope="/user.slice/user-1001.slice/session-1.scope")
    a, b = node.run("r-a", pgid=30), node.run("r-b", pgid=40)
    shared = "/user.slice/user-1001.slice/session-2.scope"
    node.process(30, pgid=30, anon_gb=1, run_dir=a, scope=shared)
    node.process(40, pgid=40, anon_gb=1, run_dir=b, scope=shared)
    node.process(41, pgid=41, anon_gb=50, scope=shared)
    assert node.main() == 0
    assert node.adj(10) == node.adj(11) == 500 + round(500 * 383.1 / 1000)
    assert node.adj(9) == 0 and node.adj(20) == 0 and node.adj(41) == 0


def test_a_train_checks_priority_reservation_ranks_it_below_research_runs(node):
    """ci 24 Oct 05:47Z: a train check's slot reservation carries "priority": true (#3486), so a research job within its own
    reservation is still picked before the train, and a train over its reservation ranks with a research job within one."""
    train = node.run("r-train", pgid=10)
    (node.ramlock / "check-b.json").write_text(json.dumps({"owner": "check", "gb": 300, "pid": 10, "run_id": "r-train",
                                                            "priority": True}))
    node.process(10, pgid=10, anon_gb=250, run_dir=train)
    lane = node.run("r-lane", pgid=20, gb=100)
    node.process(20, pgid=20, anon_gb=90, run_dir=lane)
    over = node.run("r-train2", pgid=30)
    (node.ramlock / "check-a.json").write_text(json.dumps({"gb": 300, "pid": 30, "run_id": "r-train2", "priority": True}))
    node.process(30, pgid=30, anon_gb=400, run_dir=over)
    assert node.main() == 0
    assert node.adj(10) == -500 and node.adj(20) == 0 and node.adj(30) == round(500 * 100 / 1000)


def test_a_dry_run_says_what_it_would_set_and_writes_nothing(node, capsys):
    rd = node.run("r20261019-140000-1111", pgid=50)
    node.process(50, pgid=50, anon_gb=10, run_dir=rd)
    assert node.main("--dry-run") == 0
    assert node.adj(50) == 0 and node.log() == []
    assert json.loads(capsys.readouterr().out)["set"] == [[50, 0]]


def test_the_script_is_stdlib_only_and_deploy_toml_runs_it_as_root_on_the_shared_nodes():
    src = Path(O.__file__)
    mods = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else (n.module or "").split(".")[0]
            for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert mods - {"__future__"} <= set(sys.stdlib_module_names), "deploy.toml installs this file alone"
    import tomllib
    m = tomllib.loads((src.parent / "deploy.toml").read_text())
    rows = {e["path"]: e for e in m["file"]}
    nodes = {"vy-nebius-1", "vy-nebius-2", "vy-nebius-cpu-1", "vy-nebius-cpu-2", "vy-nebius-cpu-3", "vy-nebius-cpu-4",
             "vy-nebius-cpu-5", "vy-nebius-cpu-6"}
    for path in ("/usr/local/lib/vy-monitoring/oom_order.py", "/etc/systemd/system/vy-oom-order.service",
                 "/etc/systemd/system/vy-oom-order.timer"):
        assert set(rows[path]["nodes"]) == nodes and nodes <= set(m["nodes"]), path
    assert rows["/etc/systemd/system/vy-oom-order.timer"]["run"] == ["systemd:vy-oom-order.timer"]
    unit = (src.parent / "vy-oom-order.service").read_text()
    assert "ExecStart=/usr/bin/python3 /usr/local/lib/vy-monitoring/oom_order.py\n" in unit and "User=" not in unit
