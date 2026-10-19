#!/usr/bin/env python3
"""An agent VM's OOM recorder (INF-55 f): every kernel OOM kill here, with the victim's worktree, kept and announced.

    oom_watch.py loop     boot.sh starts it detached on every boot (one per VM: it holds $OOM_WATCH_LOCK)
    oom_watch.py scan     print the OOM kills still in the kernel's ring buffer as JSON, record nothing

The kernel's ring buffer turns over in minutes on a busy VM and agent VMs keep no journal, so a kill (nine workers sharing a 15 GB
coordinator VM lost four python processes at 00:39Z 18 Oct) left no trace by the time anyone looked. This reads /dev/kmsg as it is
written and samples every process over $OOM_WATCH_MIN_MB (default 300) every few seconds for its cwd, which a dead process no
longer has, so each kill is recorded with the worktree it ran in: the worker. A kill is a line in $OOM_WATCH_LOG (default
~/.research/agent-oom.jsonl; pid, comm, anon RSS, memory cgroup, cwd, worktree, the VM and its agent) and, best effort, a post to
#agent-alerts through `research msg send` (the record stays either way). A restart reports only kills after the last one recorded.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

LOG = Path(os.environ.get("OOM_WATCH_LOG", str(Path.home() / ".research" / "agent-oom.jsonl")))
LOCK = Path(os.environ.get("OOM_WATCH_LOCK", "/tmp/agent-oom-watch.lock"))
MIN_KB = int(os.environ.get("OOM_WATCH_MIN_MB", "300")) * 1024
SAMPLE_S = 3.0
KEEP_S = 600.0
KILLED = re.compile(r"Killed process (\d+) \(([^)]*)\).*?anon-rss:(\d+)kB")
OOM_KILL = re.compile(r"oom-kill:.*?task_memcg=([^,]*),task=([^,]*),pid=(\d+)")
RESEARCH = Path(os.environ.get("OOM_WATCH_RESEARCH", "/workspace/.venv/bin/research"))


def worktree(cwd: str) -> str | None:
    p = Path(cwd)
    for d in (p, *p.parents):
        if (d / ".git").exists():
            return str(d)
    return None


def sample(seen: dict[int, dict], now: float) -> None:
    """Every process over MIN_KB resident: its comm and cwd, kept KEEP_S after it was last seen."""
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            m = re.search(r"^VmRSS:\s+(\d+) kB", status, re.M)
            if not m or int(m.group(1)) < MIN_KB:
                continue
            cwd = os.readlink(f"/proc/{pid}/cwd")
            comm = re.search(r"^Name:\s+(.*)$", status, re.M).group(1)
        except (OSError, AttributeError):
            continue
        seen[pid] = {"comm": comm, "cwd": cwd, "rss_kb": int(m.group(1)), "seen": now}
    for pid in [p for p, v in seen.items() if now - v["seen"] > KEEP_S]:
        del seen[pid]


def records(fd: int):
    """(seq, usec since boot, message) of each /dev/kmsg record ready to read."""
    while True:
        try:
            raw = os.read(fd, 8192)
        except BlockingIOError:
            return
        except OSError as e:
            if e.errno == errno.EPIPE:                    # overwritten before it was read: go on with the next
                continue
            raise
        head, _, msg = raw.decode(errors="replace").partition(";")
        f = head.split(",")
        if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
            yield int(f[1]), int(f[2]), msg.split("\n", 1)[0]


def last_seq() -> int:
    try:
        lines = LOG.read_text().splitlines()
        return max(json.loads(ln).get("seq", -1) for ln in lines) if lines else -1
    except (OSError, ValueError):
        return -1


class Watch:
    def __init__(self) -> None:
        self.seen: dict[int, dict] = {}
        self.memcg: dict[int, str] = {}
        boot = time.time() - float(Path("/proc/uptime").read_text().split()[0])
        self.boot = boot

    def judge(self, seq: int, usec: int, msg: str) -> dict | None:
        if m := OOM_KILL.search(msg):
            self.memcg[int(m.group(3))] = m.group(1)
            return None
        m = KILLED.search(msg)
        if not m:
            return None
        pid = int(m.group(1))
        s = self.seen.get(pid, {})
        return {"seq": seq, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.boot + usec / 1e6)),
                "vm": socket.gethostname(), "agent": os.environ.get("RESEARCH_AGENT_URL"), "pid": pid, "comm": m.group(2),
                "anon_rss_mb": int(m.group(3)) // 1024, "memcg": self.memcg.pop(pid, None),
                "cgroup_limit": msg.startswith("Memory cgroup out of memory"), "cwd": s.get("cwd"),
                "worktree": worktree(s["cwd"]) if s.get("cwd") else None}


def announce(rec: dict) -> None:
    if not RESEARCH.exists():
        return
    where = rec["worktree"] or rec["cwd"] or "an unknown cwd (not sampled before it died)"
    text = (f"OOM kill on agent VM {rec['vm']} at {rec['t']}: pid {rec['pid']} ({rec['comm']}, {rec['anon_rss_mb']} MB) in {where}"
            + (f"; the VM's agent: {rec['agent']}" if rec.get("agent") else "")
            + (" (its cgroup's limit)" if rec["cgroup_limit"] else " (the VM ran out of memory)")
            + f". Recorded in {LOG}. Heavy work belongs on a CPU node: `research run --on NODE --mem-gb G`.")
    try:
        subprocess.run([str(RESEARCH), "msg", "send", "--channel", "agent-alerts", "--as", "infra", "--kind", "alert", "--text", text],
                       cwd="/workspace", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass


def scan() -> list[dict]:
    w = Watch()
    fd = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK)
    try:
        return [r for seq, usec, msg in records(fd) if (r := w.judge(seq, usec, msg))]
    finally:
        os.close(fd)


def loop() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(LOCK, os.O_WRONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("oom_watch: another one runs on this VM", file=sys.stderr)
        return 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    w, after = Watch(), last_seq()
    fd = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK)
    while True:
        sample(w.seen, time.time())
        after = record_kills(w, records(fd), after)
        select.select([fd], [], [], SAMPLE_S)


def record_kills(w: Watch, recs, after: int) -> int:
    """Record and announce each kill among `recs` past seq `after`; the last seq recorded."""
    for seq, usec, msg in recs:
        rec = w.judge(seq, usec, msg)
        if rec and seq > after:
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            after = seq
            announce(rec)
    return after


def main(argv: list[str]) -> int:
    if argv[:1] == ["scan"]:
        print(json.dumps(scan()))
        return 0
    if argv[:1] == ["loop"]:
        return loop()
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
