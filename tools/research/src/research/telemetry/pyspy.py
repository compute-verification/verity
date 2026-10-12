"""py-spy dumps of a run's Python processes: every thread's stack, for a run that has stopped making progress.

    processes  status.json `pid` (none when its start time is not `starttime_ticks`: the pid was reused) and its descendants by
               /proc ppid, the Python ones (comm or exe name starting `python`), at most MAX_PROCS
    dump       `py-spy dump --pid P --nonblocking` for each (read, never paused), each within DUMP_TIMEOUT_S, under a
               `== pid P (comm)` header with py-spy's error when it failed

Under kernel.yama.ptrace_scope 1 py-spy reads only its own descendants unless it runs as root or has CAP_SYS_PTRACE: the
harness's py-spy is the workload's sibling, so without either each dump is py-spy's permission error.

Self-contained (stdlib, no package imports): `tele run`'s stall watch imports it, and `research inspect --py-spy` sends this
file to a live run's machine as `python3 - RUN_DIR`, which prints the dump (exit 3 when py-spy is not on PATH)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_PROCS = 16
DUMP_TIMEOUT_S = 30.0
MISSING = "py-spy is not on PATH"


def _stat(proc: Path, pid: int) -> tuple[str, list[str]] | None:
    """(comm, the /proc/<pid>/stat fields after it: [state, ppid, pgrp, ...]; starttime is index 19)."""
    try:
        raw = (proc / str(pid) / "stat").read_text()
    except OSError:
        return None
    if "(" not in raw or ")" not in raw:
        return None
    return raw[raw.index("(") + 1:raw.rindex(")")], raw.rsplit(")", 1)[1].split()


def _is_python(proc: Path, pid: int, comm: str) -> bool:
    if comm.startswith("python"):
        return True
    try:
        return os.path.basename(os.readlink(proc / str(pid) / "exe")).startswith("python")
    except OSError:
        return False


def python_tree(root_pid: int, starttime: int | None = None, proc: Path | None = None) -> list[tuple[int, str]]:
    """(pid, comm) of the Python processes among `root_pid` and its descendants, root first, at most MAX_PROCS."""
    proc = Path(proc or os.environ.get("RESEARCH_PROC_ROOT", "/proc"))
    root = _stat(proc, root_pid)
    if root is None or (starttime is not None and root[1][19:20] != [str(starttime)]):
        return []
    comms, children = {root_pid: root[0]}, {}
    try:
        entries = sorted(int(e) for e in os.listdir(proc) if e.isdigit())
    except OSError:
        entries = []
    for p in entries:
        if (s := _stat(proc, p)) is not None and p != root_pid:
            comms[p] = s[0]
            children.setdefault(int(s[1][1]), []).append(p)
    tree, todo = [], [root_pid]
    while todo:
        p = todo.pop(0)
        tree.append(p)
        todo += [c for c in children.get(p, []) if c not in tree]
    return [(p, comms[p]) for p in tree if _is_python(proc, p, comms[p])][:MAX_PROCS]


def dump(procs: list[tuple[int, str]], *, run: Callable[..., Any] = subprocess.run,
         which: Callable[[str], str | None] = shutil.which) -> list[dict[str, Any]] | None:
    """[{pid, comm, rc, text}] per process (rc None: py-spy timed out or did not start), or None when py-spy is not on PATH."""
    exe = which("py-spy")
    if exe is None:
        return None
    out = []
    for pid, comm in procs:
        try:
            r = run([exe, "dump", "--pid", str(pid), "--nonblocking"], capture_output=True, timeout=DUMP_TIMEOUT_S)
            rc, text = r.returncode, r.stdout.decode(errors="replace")
            if rc:
                text += f"[py-spy rc {rc}] {r.stderr.decode(errors='replace').strip()}\n"
        except subprocess.TimeoutExpired:
            rc, text = None, f"[py-spy took over {DUMP_TIMEOUT_S:g} s]\n"
        except OSError as e:
            rc, text = None, f"[py-spy did not start: {e}]\n"
        out.append({"pid": pid, "comm": comm, "rc": rc, "text": text})
    return out


def render(results: list[dict[str, Any]]) -> str:
    if not results:
        return "no Python process in the workload's tree\n"
    return "".join(f"== pid {r['pid']} ({r['comm']})\n{r['text']}" + ("" if r["text"].endswith("\n") else "\n") for r in results)


def _status(run_dir: Path) -> dict[str, Any]:
    try:
        st = json.loads((Path(run_dir) / "status.json").read_text())
    except (OSError, ValueError):
        return {}
    return st if isinstance(st, dict) else {}


def of_run(run_dir: Path, proc: Path | None = None, **kw: Any) -> list[dict[str, Any]] | None:
    """`dump` of the live workload that `run_dir`'s status.json names ([] when it names none, or a reused pid)."""
    st = _status(run_dir)
    pid = st.get("pid")
    procs = python_tree(pid, st.get("starttime_ticks"), proc) if isinstance(pid, int) else []
    return dump(procs, **kw)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python3 pyspy.py RUN_DIR", file=sys.stderr)
        return 2
    if (state := _status(Path(argv[0])).get("state")) != "running":
        print(f"its status.json says {state or 'nothing'}: no live workload to dump", file=sys.stderr)
        return 1
    results = of_run(Path(argv[0]))
    if results is None:
        print(MISSING, file=sys.stderr)
        return 3
    sys.stdout.write(render(results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
