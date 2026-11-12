"""`scan_zk.sh`'s `SC_LEAN_ONE`: every Lean verifier process the scan starts runs on one core, the lowest the job has,
and leaves a line in $OUT/lean.jsonl with its command, the session archive it verifies (under the scan's working
directory), that core, its exit and what it used; the verifier's own exit is the wrapper's. Without the setting the verifier is the build's binary and nothing is written. The block runs as
the script has it, on a stand-in for the verifier that says which cores it may run on."""
from __future__ import annotations

import json
import os
import sys

from test_out_request_values import block, run

BLOCK = ('if [ -n "${SC_LEAN_ONE:-}" ]; then', "fi")
VERIFIER = '#!/usr/bin/env bash\ngrep Cpus_allowed_list /proc/self/status | tr -dc "0-9,\\n-" >> "$SEEN"\n[ "$1" = verify ]\n'


def test_sc_lean_one_runs_each_verifier_process_on_one_core_and_keeps_a_line(tmp_path):
    w, out, lean, seen = tmp_path / "W", tmp_path / "OUT", tmp_path / "verifier", tmp_path / "seen"
    w.mkdir()
    out.mkdir()
    lean.write_text(VERIFIER)
    lean.chmod(0o755)
    calls = '"$LEAN" verify --zk --archive "$W/sczk/honest/session-first/archive"; echo "rc $?"\n' \
        '"$LEAN" archive-put a; echo "rc $?"\n'
    got = run(block("scan_zk.sh", *BLOCK) + calls, W=w, OUT=out, PY=sys.executable, LEAN=lean, SEEN=seen, SC_LEAN_ONE=1)
    assert (got.returncode, got.stdout) == (0, "rc 0\nrc 1\n"), got.stderr
    core = min(os.sched_getaffinity(0))
    assert seen.read_text() == f"{core}\n{core}\n"
    rows = [json.loads(x) for x in (out / "lean.jsonl").read_text().splitlines()]
    assert [(r["command"], r["archive"], r["core"], r["rc"]) for r in rows] == [
        (["verify"], "sczk/honest/session-first/archive", core, 0), (["archive-put"], None, core, 1)]
    assert all(set(r) == {"command", "archive", "core", "rc", "wall_s", "cpu_s", "peak_gb"} for r in rows)


def test_without_sc_lean_one_the_verifier_is_the_builds(tmp_path):
    w, out, lean = tmp_path / "W", tmp_path / "OUT", tmp_path / "verifier"
    w.mkdir()
    out.mkdir()
    got = run(block("scan_zk.sh", *BLOCK) + 'echo "$LEAN"\n', W=w, OUT=out, PY=sys.executable, LEAN=lean)
    assert (got.returncode, got.stdout) == (0, f"{lean}\n"), got.stderr
    assert not (out / "lean.jsonl").exists() and not list(w.iterdir())
