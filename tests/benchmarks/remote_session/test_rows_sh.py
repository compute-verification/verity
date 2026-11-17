"""`rows.sh` builds the leaf pass's 64-row kernels (-DLEAF_PASS64, to session.cu's build and the base's) only for a run
with 64 among ROWS and --leaf-ctas in ROWS_ARGS, and otherwise asks `sass_gate.py` to refuse them (--no-leaf64): run on
a CPU with BUILD_ONLY, its nvcc, cuobjdump and checkers stubs that record what they were given."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SESSION = "verity/protocols/resource_accounting/memory/poub/prover/remote_session/session.cu"
STUB = """#!/usr/bin/env bash
echo "{name} $*" >> "$STUB_LOG"
{body}
exit 0
"""
OUTPUT = 'for ((i = 1; i <= $#; i++)); do [ "${!i}" = -o ] && { j=$((i + 1)); : > "${!j}"; }; done'


def _run(tmp_path: Path, rows: str, rows_args: str) -> list[str]:
    tree, bin_, log = tmp_path / "tree", tmp_path / "cuda" / "bin", tmp_path / "log.txt"
    (tree / "benchmarks" / "remote_session").mkdir(parents=True)
    (tree / "benchmarks" / "remote_session" / "rows.sh").write_text((ROOT / "benchmarks/remote_session/rows.sh").read_text())
    (tree / SESSION).parent.mkdir(parents=True)
    (tree / SESSION).write_text("// the base\n")
    for script in ("same_sass.py", "sign_key.py", "sass_gate.py"):
        (tree / "benchmarks" / "remote_session" / script).write_text(
            f"import os, sys\nopen(os.environ['STUB_LOG'], 'a').write('{script} ' + ' '.join(sys.argv[1:]) + '\\n')\n")
    bin_.mkdir(parents=True)
    for name, body in {"nvcc": OUTPUT, "cuobjdump": "", "nvdisasm": ""}.items():
        (bin_ / name).write_text(STUB.format(name=name, body=body))
        (bin_ / name).chmod(0o755)
    git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp_path), "STUB_LOG": str(log), "OUT": str(tmp_path / "out"),
           "CUDA_HOME": str(bin_.parent), "ARCH": "90a", "BUILD_ONLY": "1", "BASE_REV": "HEAD", "ROWS": rows,
           "ROWS_ARGS": rows_args}
    done = subprocess.run(["bash", "benchmarks/remote_session/rows.sh"], cwd=tree, env=env, capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return log.read_text().splitlines()


@pytest.mark.parametrize("rows, rows_args, leaf64", [("64 16", "--leaf-ctas 132", True), ("64", "--leaf-ctas=-1", True),
                                                     ("16", "--leaf-ctas 132", False), ("64 16", "", False),
                                                     ("64 16", "--whole-store", False)])
def test_the_leaf_pass64_define_is_a_64_row_leaf_runs_alone(tmp_path, rows, rows_args, leaf64):
    log = _run(tmp_path, rows, rows_args)
    builds = [line for line in log if line.startswith("nvcc ") and line.endswith("session.cu")]
    assert len(builds) == 2, log
    assert all(("-DLEAF_PASS64" in line.split()) == leaf64 for line in builds), builds
    [gate] = [line for line in log if line.startswith("sass_gate.py ")]
    assert ("--no-leaf64" in gate.split()) != leaf64, gate
    assert not any("-DLEAF_PASS64" in line for line in log if "kv.cu" in line)
