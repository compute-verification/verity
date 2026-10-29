"""`tools/agent-guard/oom_watch.py` records each OOM kill on an agent VM with the worktree its victim ran in,
before the kernel's ring buffer loses it."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)     # no __pycache__ in tools/agent-guard, which boot.sh copies file by file
    spec = importlib.util.spec_from_file_location("oom_watch_under_test", ROOT / "tools/agent-guard/oom_watch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "LOG", tmp_path / "agent-oom.jsonl")
    monkeypatch.setattr(m, "RESEARCH", tmp_path / "no-research")          # nothing is announced
    monkeypatch.setattr(m, "MIN_KB", 0)
    return m


def test_a_kill_is_recorded_once_with_the_worktree_its_victim_ran_in(monkeypatch, tmp_path):
    m = load(monkeypatch, tmp_path)
    wt = tmp_path / "worktrees" / "ir-typing"
    (wt / ".git").mkdir(parents=True)
    (wt / "src").mkdir()
    victim = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], cwd=wt / "src", stdin=subprocess.PIPE)
    try:
        w = m.Watch()
        m.sample(w.seen, 1_800_000_000.0)
        assert w.seen[victim.pid]["cwd"] == str(wt / "src")
    finally:
        victim.kill()
        victim.wait()
    pid = victim.pid
    kmsg = [(7, 5_000_000, f"oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,"
                           f"task_memcg=/system.slice/pod-x/cursor-agent/workload,task=python3,pid={pid},uid=1000"),
            (8, 5_000_100, f"Out of memory: Killed process {pid} (python3) total-vm:5000000kB, anon-rss:4404019kB, file-rss:0kB, "
                           f"shmem-rss:0kB, UID:1000 pgtables:9000kB oom_score_adj:0"),
            (9, 5_000_200, "Out of memory: Killed process 99999999 (pytest) total-vm:2000000kB, anon-rss:1363148kB, file-rss:0kB")]
    assert m.record_kills(w, iter(kmsg), m.last_seq()) == 9
    recs = [json.loads(ln) for ln in (tmp_path / "agent-oom.jsonl").read_text().splitlines()]
    assert [(r["seq"], r["pid"], r["comm"], r["anon_rss_mb"]) for r in recs] == [(8, pid, "python3", 4300), (9, 99999999, "pytest", 1331)]
    assert recs[0]["worktree"] == str(wt) and recs[0]["memcg"] == "/system.slice/pod-x/cursor-agent/workload"
    assert recs[0]["cgroup_limit"] is False and recs[1]["cwd"] is None and recs[1]["worktree"] is None
    assert m.record_kills(m.Watch(), iter(kmsg), m.last_seq()) == 9, "a restart records only kills after the last one"
    assert len((tmp_path / "agent-oom.jsonl").read_text().splitlines()) == 2


def test_a_second_recorder_on_the_vm_steps_aside(monkeypatch, tmp_path):
    m = load(monkeypatch, tmp_path)
    monkeypatch.setattr(m, "LOCK", tmp_path / "lock")
    import fcntl
    import os
    fd = os.open(tmp_path / "lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert m.loop() == 0
    finally:
        os.close(fd)
