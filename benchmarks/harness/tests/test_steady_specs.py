"""The setup a steady-state spec names for this tree's Flock prover, under the launcher on a made-up tree, with stand-ins
for the prover's build, the toolkit's script and uv. No GPU, no node, no store and no clock."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SETUP = "benchmarks/harness/setup/flock_prover.sh"
LAUNCHER = "benchmarks/harness/steady_point.sh"


#: a stand-in for verity/protocols/flock/pod/cuda-13.3.sh, in its shape: it names a toolkit it finds, else unpacks one,
#: which resets the sourcing shell's arguments, and exits where it can't
TOOLKIT = """for c in "$FLOCK_WORK/cuda-root"; do
  [ -z "${CUDA_TK:-}" ] && [ -x "$c/bin/nvcc" ] && export CUDA_TK=$c
done
if [ -z "${CUDA_TK:-}" ]; then
  set -- unpacking resets the arguments
  [ -e "$FLOCK_WORK/no-toolkit" ] && { echo "CUDA 13.3 unpack failed"; exit 1; }
  mkdir -p "$FLOCK_WORK/cuda-root/bin" && printf '#!/bin/sh\\n' > "$FLOCK_WORK/cuda-root/bin/nvcc" \\
    && chmod +x "$FLOCK_WORK/cuda-root/bin/nvcc"
  export CUDA_TK=$FLOCK_WORK/cuda-root
fi
"""
#: a stand-in for 70-class-sweep.sh BUILD_ONLY=1: it keeps a binary under FLOCK_WORK and prints it after a traced line
#: of the same form, as the build script's `set -x` does, and says what it was run with
BUILD = """for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
echo "+ echo BINARY=/a/traced/line"
[ -e "$FLOCK_WORK/no-build" ] && { echo "error: the build failed"; exit 1; }
bin=$FLOCK_WORK/flock-circuit/bin-key-g$GPU/flock-circuit
mkdir -p "$(dirname "$bin")" "$OUT" && printf '#!/bin/sh\\n' > "$bin" && chmod +x "$bin"
printf '%s\\n' "BUILD_ONLY=$BUILD_ONLY GPU=$GPU" "$RESEARCH_RUN_DIR" "$OUT" "$(command -v nvcc)" "$PWD" > "$FLOCK_WORK/build-saw.txt"
echo "BINARY=$bin"
"""
#: a stand-in for uv: `uv venv ... DIR` makes DIR/bin/python3, which answers the setup's import check and, as the
#: launcher's last command, prints the arguments and the environment it was started with
UV = """#!/bin/sh
echo "uv $*" >> "$FLOCK_WORK/uv-ran.txt"
[ -e "$FLOCK_WORK/no-uv" ] && exit 1
[ "$1" = venv ] || exit 0
for last; do :; done
mkdir -p "$last/bin"
cat > "$last/bin/python3" <<'PYTHON'
#!/bin/sh
if [ "$1" = -c ]; then echo "python 3.12.0 numpy 2.0 blake3 1.0"; exit 0; fi
printf '%s\\n' "$*" "$VERITY_BENCH_STEADY_BIN" "$VERITY_BENCH_STEADY_PY" "$VERITY_BENCH_STEADY_SCRATCH" "$PATH" \\
  "$LD_LIBRARY_PATH" "$PYTHONPATH" "$FLOCK_WORK"
PYTHON
chmod +x "$last/bin/python3"
"""
ARGS = ["--spec", "S", "--point", "P", "--identity", "I"]


class Launcher:
    """The launcher and the setup, as the tree has them, in a made-up tree beside the stand-ins, with a work directory
    and a home of the test's own."""

    def __init__(self, tmp_path: Path):
        tmp_path = tmp_path.resolve()
        self.root, self.work, self.run_dir, home = (tmp_path / name for name in ("tree", "work", "run", "home"))
        texts = {LAUNCHER: (ROOT / LAUNCHER).read_text(), SETUP: (ROOT / SETUP).read_text(),
                 "verity/protocols/flock/pod/cuda-13.3.sh": TOOLKIT, "verity/protocols/flock/pod/70-class-sweep.sh": BUILD,
                 "home/.local/bin/uv": UV}
        for path, text in texts.items():
            path = tmp_path / path if path.startswith("home/") else self.root / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (home / ".local/bin/uv").chmod(0o755)
        self.run_dir.mkdir()
        self.env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "FLOCK_WORK": str(self.work),
                    "RESEARCH_RUN_DIR": str(self.run_dir), "VERITY_BENCH_SETUP": SETUP}

    def __call__(self) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", LAUNCHER, *ARGS], cwd=self.root, env=self.env, capture_output=True, text=True)


def test_the_setup_builds_the_prover_and_leaves_the_environment_a_point_runs_in(tmp_path):
    launcher = Launcher(tmp_path)
    root, work = launcher.root, launcher.work
    ran = launcher()
    assert ran.returncode == 0, ran.stderr
    args, binary, python, scratch, path, libraries, pythonpath, flock_work = ran.stdout.splitlines()
    assert args == " ".join(["-m", "verity_bench.steady_node", *ARGS]), "the point's arguments outlive the toolkit's unpacking"
    assert binary == f"{work}/flock-circuit/bin-key-g1/flock-circuit" and os.access(binary, os.X_OK)
    assert python == f"{work}/flock-circuit/py/bin/python3" and flock_work == str(work)
    assert scratch == f"{work}/scratch" and Path(scratch).is_dir(), "the scratch is on disk beside the builds"
    toolkit = f"{work}/cuda-root"
    assert path.split(":")[0] == f"{toolkit}/bin", "the toolkit the prover is built with is first on PATH"
    assert libraries == f"{toolkit}/lib64", "the toolkit's own libraries, and no compat library over the node's driver"
    assert pythonpath.split(":") == [
        f"{root}/benchmarks/harness", f"{root}/verity/configuration/config", str(root), f"{root}/tools/research/src",
        str(root), f"{root}/verity/protocols/flock/prover/python", f"{root}/adapters/vllm",
        f"{root}/tools/verity/circuit_check/src", f"{root}/benchmarks/numerical/python"]
    setup = launcher.run_dir / "out" / "setup"
    assert (work / "build-saw.txt").read_text().splitlines() == [
        "BUILD_ONLY=1 GPU=1", f"{setup}/build", f"{setup}/build/sweep", f"{toolkit}/bin/nvcc", str(root)]
    assert {"build.txt", "cuda.log", "venv.log", "python.txt"} <= {path.name for path in setup.iterdir()}
    assert (setup / "python.txt").read_text() == "python 3.12.0 numpy 2.0 blake3 1.0\n"
    assert len((work / "uv-ran.txt").read_text().splitlines()) == 2

    again = launcher()
    assert again.returncode == 0 and again.stdout == ran.stdout
    assert len((work / "uv-ran.txt").read_text().splitlines()) == 2, "a later point finds the Python the first made"


def test_the_setups_default_work_directory_is_the_harnesss_own_under_the_nodes_jobs():
    setup = (ROOT / SETUP).read_text()
    assert "export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/verity-bench-steady}\n" in setup
    assert "VERITY_BENCH_STEADY_SCRATCH=$FLOCK_WORK/scratch" in setup and "/dev/shm" not in setup


def test_the_setup_removes_only_the_scratch_a_killed_point_left_long_ago(tmp_path):
    launcher = Launcher(tmp_path)
    scratch = launcher.work / "scratch"
    for name in ("verity-bench-steady-old", "verity-bench-steady-live", "another-lanes"):
        (scratch / name / "z").mkdir(parents=True)
    for name in ("verity-bench-steady-old", "another-lanes"):
        os.utime(scratch / name, (0, 0))
    assert launcher().returncode == 0
    assert sorted(path.name for path in scratch.iterdir()) == ["another-lanes", "verity-bench-steady-live"]


#: a step of the setup that fails: the file that makes the stand-in fail, and what the launcher then says
FAILS = [
    ("no-toolkit", "setup: the node has no CUDA 13.3 toolkit and none could be unpacked (out/setup/cuda.log)"),
    ("no-build", "setup: the prover's build failed (out/setup/build.txt)"),
    ("no-uv", "setup: no Python 3.12 could be made under"),
]


@pytest.mark.parametrize("flag, says", FAILS, ids=[flag for flag, _ in FAILS])
def test_a_step_of_the_setup_that_fails_ends_the_run_with_4_and_starts_no_point(tmp_path, flag, says):
    launcher = Launcher(tmp_path)
    launcher.work.mkdir()
    (launcher.work / flag).write_text("")
    ran = launcher()
    assert ran.returncode == 4 and says in ran.stderr and ran.stdout == ""
    log = {"no-toolkit": "cuda.log", "no-build": "build.txt", "no-uv": "venv.log"}[flag]
    assert (launcher.run_dir / "out" / "setup" / log).is_file()
