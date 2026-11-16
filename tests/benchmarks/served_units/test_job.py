"""job.sh's `stage_one` hands stage.py a unit as one argument: a census name's statics are commas inside braces, which an
unquoted name in its ``bash -c`` string would brace-expand into one argument a static (stage.py's usage refusal)."""
from __future__ import annotations

import os
import subprocess

from conftest import HERE

#: `stage_one` alone, with job.sh's helpers stubbed: the stager records its arguments and writes a stage.json
SCRIPT = r"""
set -u
eval "$(sed -n '/^stage_one() {/,/^}/p' "$JOB")"
tree_key() { echo key; }
vlimit() { :; }
tm() { local o=$1; shift; "$@"; local rc=$?; : > "$o"; return $rc; }
TIME=tm SU=$T/su N=1 STAGE_GB=1 W=$T HZ_CACHE=$T/cache
mkdir -p "$T/d" "$T/o"
stage_one "$SPEC" "$T/d" "$T/o"
"""
STAGER = '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS"\nmkdir -p "$4" && : > "$4/stage.json"\n'


def test_stage_one_passes_a_census_name_whole(tmp_path):
    spec = "census:DsaMaskPositions_v2{N=40960,AT=0,K=2048}"
    py = tmp_path / "bin" / "py"
    py.parent.mkdir()
    py.write_text(STAGER)
    py.chmod(0o755)
    args = tmp_path / "args.txt"
    env = {**os.environ, "JOB": str(HERE / "job.sh"), "T": str(tmp_path), "SPEC": spec, "PY": str(py), "ARGS": str(args)}
    r = subprocess.run(["bash", "-c", SCRIPT], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = args.read_text().splitlines()
    assert got[:3] == [str(tmp_path / "su" / "stage.py"), spec, "1"] and len(got) == 4
    assert (tmp_path / "d" / "stage.json").is_symlink()
