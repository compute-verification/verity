"""`onsite_piece_commands` (onsite_pieces.sh), sourced and called as served_verdicts.sh does: each onsite piece's command
is the tree's or none, and a command of the caller's own in WIPE_CMD, POUS_CMD or BANDWIDTH_CMD is refused by name before
anything runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
NAMES = ("WIPE_CMD", "POUS_CMD", "BANDWIDTH_CMD")
CALL = """set -uo pipefail
. "$ONSITE_PIECES"
[ "${1:-}" = trees ] && { printf '%s\\n' "$ONSITE_WIPE_CMD" "$ONSITE_POUS_CMD" "$ONSITE_BANDWIDTH_CMD"; exit; }
onsite_piece_commands
printf '%s\\n' "${WIPE_CMD-unset}" "${POUS_CMD-unset}" "${BANDWIDTH_CMD-unset}"
"""


def _call(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    (tmp_path / ".ssh").mkdir(exist_ok=True)
    (tmp_path / ".ssh" / "vy-auditor2-pous").touch()
    return subprocess.run(["bash", "-c", CALL, "_", *args], capture_output=True, text=True, check=False,
                          env={"PATH": os.defpath, "HOME": str(tmp_path), "TREE": str(ROOT),
                               "ONSITE_PIECES": str(ROOT / "benchmarks" / "pous" / "onsite_pieces.sh"), **env})


def test_unset_each_is_the_trees_and_set_to_the_trees_or_empty_each_is_kept(tmp_path) -> None:
    trees = _call(tmp_path, "trees").stdout.splitlines()
    assert len(trees) == 3 and all(trees) and _call(tmp_path).stdout.splitlines() == trees
    r = _call(tmp_path, WIPE_CMD="", POUS_CMD=trees[1], BANDWIDTH_CMD="")
    assert (r.returncode, r.stdout.splitlines()) == (0, ["", trees[1], ""]), r.stderr


@pytest.mark.parametrize("name", NAMES)
def test_a_command_of_the_callers_own_is_refused_by_name(tmp_path, name) -> None:
    r = _call(tmp_path, **{name: f"touch {tmp_path}/ran"})
    assert r.returncode == 3 and r.stdout == "" and not (tmp_path / "ran").exists()
    assert r.stderr.startswith(f"PREFLIGHT: {name} names a command of its own: an onsite piece runs only the tree's")


def test_the_wipes_run_keys_form_is_the_trees_and_no_other_piece_has_one(tmp_path) -> None:
    wipe = _call(tmp_path, "trees").stdout.splitlines()[0]
    r = _call(tmp_path, WIPE_CMD=f"{wipe} --run-keys")
    assert (r.returncode, r.stdout.splitlines()[0]) == (0, f"{wipe} --run-keys"), r.stderr
    for name, value in (("WIPE_CMD", f"{wipe} --run-keys --x"), ("POUS_CMD", f"{wipe} --run-keys")):
        assert _call(tmp_path, **{name: value}).returncode == 3, name
