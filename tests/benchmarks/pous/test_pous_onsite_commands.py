"""`onsite_piece_commands` (onsite_pieces.sh), sourced and called as served_verdicts.sh does: each onsite piece's command
is the tree's or none, and a command of the caller's own in WIPE_CMD, POUS_CMD or BANDWIDTH_CMD is refused by name before
anything runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CALL = """set -uo pipefail
. "$ONSITE_PIECES"
onsite_piece_commands
printf '%s\\n' "${WIPE_CMD-unset}" "${POUS_CMD-unset}" "${BANDWIDTH_CMD-unset}"
"""


def _call(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    (tmp_path / ".ssh").mkdir(exist_ok=True)
    (tmp_path / ".ssh" / "vy-auditor2-pous").touch()
    return subprocess.run(["bash", "-c", CALL], capture_output=True, text=True, check=False,
                          env={"PATH": os.defpath, "HOME": str(tmp_path), "TREE": str(ROOT),
                               "ONSITE_PIECES": str(ROOT / "benchmarks" / "pous" / "onsite_pieces.sh"), **env})


def _own(name: str) -> str:
    got = subprocess.run(["bash", "-c", f'. "$1"; printf %s "$ONSITE_{name}"', "_",
                          str(ROOT / "benchmarks" / "pous" / "onsite_pieces.sh")], capture_output=True, text=True,
                         check=True)
    return got.stdout


def test_unset_each_is_the_trees_and_set_to_the_trees_or_empty_each_is_kept(tmp_path) -> None:
    trees = [_own(n) for n in ("WIPE_CMD", "POUS_CMD", "BANDWIDTH_CMD")]
    r = _call(tmp_path)
    assert (r.returncode, r.stdout.splitlines()) == (0, trees), r.stderr
    r = _call(tmp_path, WIPE_CMD="", POUS_CMD=trees[1], BANDWIDTH_CMD="")
    assert (r.returncode, r.stdout.splitlines()) == (0, ["", trees[1], ""]), r.stderr


@pytest.mark.parametrize("name", ["WIPE_CMD", "POUS_CMD", "BANDWIDTH_CMD"])
def test_a_command_of_the_callers_own_is_refused_by_name(tmp_path, name) -> None:
    r = _call(tmp_path, **{name: f"touch {tmp_path}/ran"})
    assert r.returncode == 3 and r.stdout == ""
    assert r.stderr.startswith(f"PREFLIGHT: {name} names a command of its own: an onsite piece runs only the tree's")
    assert not (tmp_path / "ran").exists()
