"""`tools/verity/lean/setup.sh` against a fake elan, lean and lake: it installs the package's pinned toolchain and makes it elan's
default only on a machine that has none, so `lean` works outside a package on a fresh VM without overriding a choice."""

from __future__ import annotations

import subprocess
from pathlib import Path

SETUP = Path(__file__).resolve().parents[1] / "setup.sh"
PINNED = "leanprover/lean4:v9.9.9"

FAKE = {
    "elan": """#!/bin/sh
echo "elan $*" >> "$STATE/log"
[ "$1" = default ] && echo "$2" > "$STATE/default"
exit 0
""",
    # elan's proxy: a toolchain from `lean-toolchain` in a package, else elan's default, else this error
    "lean": """#!/bin/sh
if [ -f lean-toolchain ] || [ -f "$STATE/default" ]; then echo "Lean (version 9.9.9)"; exit 0; fi
echo "error: no default toolchain configured" >&2
exit 1
""",
    "lake": """#!/bin/sh
echo "lake $*" >> "$STATE/log"
exit 0
""",
}


def run_setup(tmp: Path, default: str | None) -> tuple[subprocess.CompletedProcess, list[str], Path]:
    bin_, state, pkg = tmp / "bin", tmp / "state", tmp / "pkg"
    for d in (bin_, state, pkg):
        d.mkdir()
    for name, body in FAKE.items():
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    (pkg / "lean-toolchain").write_text(PINNED + "\n")
    if default is not None:
        (state / "default").write_text(default + "\n")
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp), "STATE": str(state), "LEAN_CACHE": "off",
           "GIT_CEILING_DIRECTORIES": str(tmp.parent)}
    r = subprocess.run(["bash", str(SETUP), str(pkg)], env=env, capture_output=True, text=True)
    return r, (state / "log").read_text().splitlines(), state / "default"


def test_a_machine_without_a_default_gets_the_pinned_toolchain(tmp_path):
    r, log, default = run_setup(tmp_path, None)
    assert r.returncode == 0, r.stderr
    assert log[:2] == [f"elan toolchain install {PINNED}", f"elan default {PINNED}"], log
    assert default.read_text().strip() == PINNED
    assert "lake exe cache get" in log


def test_a_machine_that_chose_its_own_default_keeps_it(tmp_path):
    r, log, default = run_setup(tmp_path, "stable")
    assert r.returncode == 0, r.stderr
    assert not any(line.startswith("elan default") for line in log), log
    assert default.read_text().strip() == "stable"
