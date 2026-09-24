"""[r19-int-21, F-r19-int-19] row_pod.sh refuses to START a Commit over a Match record whose account files are not beside it.

The Commit links its executed prefix to the Match account (R17-27 a/b): match/capture/tokens.json + match/control/tokens.json beside
global_match.json.  Those live INSIDE the capture directory that the disk-freeing move to /vault (F-r19-int-14) takes away whole: #39's
Commit (l40s4) started 8 min after its capture was moved, ran 47 min and failed BY NAME on the missing file.  The precheck names the
missing file(s) as PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE (infra, exit 14) and the Commit is not run; a complete record, or a row with no
Match record at all (B=1 R12-era rows), falls through unchanged.  The real block is extracted from row_pod.sh.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
ROW_POD = TREE / "verity_vllm" / "ops" / "row_pod.sh"


def _block() -> str:
    src = ROW_POD.read_text()
    m = re.search(r"# --- row_pod\.commit-record-precheck begin.*?\n(.*?)\n  # --- row_pod\.commit-record-precheck end", src, re.DOTALL)
    assert m, "commit-record-precheck block not found in row_pod.sh"
    return m.group(1)


def _row(tmp_path: Path, *, match: bool = True, missing: tuple[str, ...] = ()) -> Path:
    d = tmp_path / "row"
    d.mkdir(parents=True)
    if match:
        (d / "match" / "capture").mkdir(parents=True)
        (d / "match" / "control").mkdir(parents=True)
        (d / "match" / "instances.jsonl").write_text("")
        for f in ("match/capture/tokens.json", "match/control/tokens.json", "global_match.json"):
            if f not in missing:
                (d / f).write_text("{}")
    return d


def _run(d: Path) -> tuple[int, str]:
    script = f"""#!/bin/bash
set -uo pipefail
D={d}
log() {{ echo "LOG $*"; }}
tl() {{ echo "$*" >> "$D/timeline.txt"; }}
verdict() {{ echo "$1 $2 $3" | tee -a "$D/stages.txt"; }}
{_block()}
echo FELL-THROUGH
exit 0
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    return r.returncode, r.stdout


def test_complete_record_and_no_record_fall_through(tmp_path):
    rc, out = _run(_row(tmp_path / "a"))
    assert rc == 0 and "FELL-THROUGH" in out and "PRECHECK_FAIL" not in out
    rc, out = _run(_row(tmp_path / "b", match=False))
    assert rc == 0 and "FELL-THROUGH" in out


def test_a_moved_capture_refuses_the_commit_by_name(tmp_path):
    d = _row(tmp_path, missing=("match/capture/tokens.json",))
    rc, out = _run(d)
    assert rc == 14 and "FELL-THROUGH" not in out
    assert "PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE: the Match record beside this attempt lacks match/capture/tokens.json" in out
    stages = (d / "stages.txt").read_text()
    assert stages.startswith("commit FAIL PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE") and "match/capture/tokens.json" in stages and "NOT run" in stages
    tl = (d / "timeline.txt").read_text()
    assert "stage.commit end rc=3 fail_class=PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE missing=match/capture/tokens.json" in tl
    assert "row end ok=false stopped_at=commit" in tl


def test_every_missing_file_is_named(tmp_path):
    d = _row(tmp_path, missing=("match/capture/tokens.json", "match/control/tokens.json", "global_match.json"))
    rc, out = _run(d)
    assert rc == 14
    assert "lacks match/capture/tokens.json match/control/tokens.json global_match.json" in out
