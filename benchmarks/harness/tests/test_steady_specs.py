"""The steady-state specs of `benchmarks/harness/specs` and the setup they name: what each spec's points are, that a row's pin and
its stored statement are the spec's alone (another pin is planned and launched by the same code), and the setup under
the launcher on a made-up tree, with stand-ins for the prover's build, the toolkit's script and uv. No GPU, no node, no
store and no clock."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from test_steady_point import tree

from verity_bench import jobs
from verity_bench import prove_records as R
from verity_bench import steady as S
from verity_bench import steady_node as N
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP
from verity_bench.vllm.presets import SPEC_ROOT

ROOT = Path(__file__).resolve().parents[3]
STEADY = sorted(path.stem for path in SPEC_ROOT.glob("*.toml") if SP.is_steady(path))
HEAD = "glm53-flock-steady-head-block-h200"
SETUP = "benchmarks/harness/setup/flock_prover.sh"
LAUNCHER = "benchmarks/harness/steady_point.sh"


def test_the_specs_directory_holds_steady_state_specs():
    assert HEAD in STEADY


@pytest.mark.parametrize("name", STEADY)
def test_a_steady_spec_names_a_workload_with_a_record_and_a_setup_that_names_the_prover_and_its_python(name):
    spec = SP.load_spec(name)
    assert spec.id == name and spec.workload in R.WORKLOADS
    assert spec.admission.max_minutes is not None, "`run --direct` refuses a spec without a bound"
    setup = (ROOT / spec.setup).read_text()
    assert spec.setup.startswith("benchmarks/harness/setup/")
    assert all(f"{exported}=" in setup for exported in ("VERITY_BENCH_STEADY_BIN", "VERITY_BENCH_STEADY_PY"))
    if any(row.stored for row in spec.row):
        assert "VERITY_BENCH_STEADY_SCRATCH=" in setup and "/dev/shm" not in setup, (
            "a statement from the store is fetched onto the node's disk: the setup names a scratch there")


def test_the_head_blocks_spec_times_both_forms_at_one_two_and_four_provers_over_whole_bursts():
    spec = SP.load_spec(HEAD)
    assert (spec.workload, spec.card, spec.cores, spec.setup) == ("glm53", "h200", 10, SETUP)
    false, true = spec.row
    assert false.type.startswith("census:SparseMlaHeadBlock_v2{") and "NVIS=64" in false.type
    assert true.type == false.type.replace("FIRST=false", "FIRST=true") != false.type
    assert false.pin != true.pin and (false.n, false.warm, false.period) == (true.n, true.warm, true.period)
    assert [(point.row is false, point.provers) for point in spec.points] == [
        (True, 1), (True, 2), (True, 4), (False, 1), (False, 2), (False, 4)]
    for point in spec.points:
        burst, gaps = point.row.period * point.provers, SP.window_gaps(point.sessions, point.provers)
        assert point.sessions >= spec.sessions and gaps % (3 * burst) == 0, "the window and each third are whole bursts"
        filings = [float(i * i) for i in range(gaps + 1)]
        assert S.burst_parts(filings, burst) == S.thirds(filings), "`level` cuts such a window at its thirds"
        assert len(set(S.thirds(filings))) == 3
    assert len({point.id for point in spec.points}) == 6


def test_the_head_blocks_statements_are_two_directories_of_one_artifact_in_the_store():
    false, true = SP.load_spec(HEAD).row
    (art, one), (same, other) = false.stored, true.stored
    assert art == same and one != other
    assert not any(row.staged.startswith("/") for row in (false, true)), "no point reads a directory another lane staged"


def real(tmp_path: Path, **changes: str) -> tuple[Path, str]:
    """The head block's spec and its setup, as the tree has them, in a made-up tree (`test_steady_point.tree`);
    `changes`: text of the spec replaced."""

    text = (SPEC_ROOT / f"{HEAD}.toml").read_text()
    for old, new in changes.items():
        assert old in text, old
        text = text.replace(old, new)
    path = f"benchmarks/harness/specs/{HEAD}.toml"
    root, _ = tree(tmp_path, files={path: text, SETUP: (ROOT / SETUP).read_text(), "benchmarks/harness/specs/steady.toml": None})
    return root, path


def launches(root: Path, path: str, node: str = "vy-dc-h200") -> tuple[list, list[list[str]]]:
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None))
    commands: list[list[str]] = []
    assert P.dispatch(plan, root / path, root=root, direct=node,
                      execute=lambda command, cwd: commands.append(list(command)) or 0) == 0
    return list(plan.points), commands


def test_each_point_of_the_head_blocks_spec_is_launched_with_the_stores_key_within_the_specs_bound(tmp_path):
    root, path = real(tmp_path)
    points, commands = launches(root, path)
    assert len(commands) == len(points) == 6
    for item, command in zip(points, commands, strict=True):
        flags, job = command[:command.index("--")], command[command.index("--") + 1:]
        assert "--store-read" in flags and flags[flags.index("--timeout") + 1] == "90m"
        assert [flags[i + 1] for i, token in enumerate(flags) if token == "--env"] == [
            "RUST_MIN_STACK=67108864", f"VERITY_BENCH_SETUP={SETUP}"]
        assert job[-8:] == ["bash", LAUNCHER, "--spec", path, "--point", item.point.id, "--identity", item.identity]


@pytest.mark.parametrize("node", ["vy-spot-glm53", "vy-dc-h200-3"])
def test_the_head_blocks_spec_runs_direct_on_the_h200_nodes_whose_slots_give_ten_cores_of_one_numa_node(tmp_path, node):
    from research.pods.nebius.slot_only import through_slot

    assert node in jobs.DIRECT_NODES
    points, commands = launches(*real(tmp_path), node=node)
    assert len(commands) == len(points) == 6
    for command in commands:
        flags, workload = command[:command.index("--")], command[command.index("--") + 1:]
        assert flags[:4] == ["research", "run", "--on", node] and "--queue" not in flags and "--gpus" not in flags
        assert through_slot(workload) and workload[:2] == ["env", f"CHECK_SLOTS={jobs.GPU_SLOTS}"]
        assert "gpu-lease" not in command, "the node side takes the card, on the NUMA node of the slot's cores"


def test_a_node_outside_the_direct_nodes_is_refused_at_dispatch_and_no_point_starts_there(tmp_path):
    """A node joins `jobs.DIRECT_NODES` with its slots file: one without is refused here, where the tuple and the file
    to add are named, and not by the slot at the start of each point's run."""

    root, path = real(tmp_path)
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None))
    launched: list = []
    for node in ("vy-dc-h200-without-slots", "vy-nebius-cpu-1"):
        assert node not in jobs.DIRECT_NODES
        with pytest.raises(ValueError, match=f"--direct {node}: not a GPU node with gpu slots.*{jobs.GPU_SLOTS} joins"):
            P.dispatch(plan, root / path, root=root, direct=node, execute=lambda command, cwd: launched.append(command) or 0)
    assert launched == []


def test_another_pin_and_another_stored_statement_are_a_change_to_the_spec_alone(tmp_path):
    """The pins and the artifact are fields of the spec: with others in their place the same code plans and launches the
    rows, each point with another identity, and nothing launched names the ones the spec held before."""

    spec = SP.load_spec(HEAD)
    (art, _), pins = spec.row[0].stored, [row.pin for row in spec.row]
    before, _ = launches(*real(tmp_path / "before"))
    changes = {art: "art:" + "7" * 64, pins[0]: "12" * 64, pins[1]: "34" * 64}
    after, commands = launches(*real(tmp_path / "after", **changes))
    assert [item.point.row.pin for item in after] == ["12" * 64] * 3 + ["34" * 64] * 3
    assert {item.point.row.stored[0] for item in after} == {"art:" + "7" * 64}
    assert not {item.identity for item in before} & {item.identity for item in after}
    assert [(a.point.id, a.point.sessions) for a in after] == [(b.point.id, b.point.sessions) for b in before]
    said = " ".join(token for command in commands for token in command)
    assert not any(old in said for old in (art, *pins))
    harness = "".join(path.read_text() for path in sorted((ROOT / "benchmarks/harness/verity_bench").glob("*.py")))
    assert not any(old[:16] in harness for old in (art.removeprefix("art:"), *pins)), "no pin or artifact is a constant in code"


def test_the_lease_tool_a_node_without_gpu_lease_is_told_to_install_is_the_trees():
    """The refusal of a node that can't run `gpu-lease` names a script of the tree: it is there, and it is the tool the
    kit's lease calls."""
    script = ROOT / N.LEASE_SOURCE
    assert script.is_file(), f"{N.LEASE_SOURCE} is gone: name the lease tool's script where it is now"
    assert "gpu-lease N [--wait]" in script.read_text(), "the script is the lease tool: its usage names it `gpu-lease`"


def test_the_setup_is_part_of_every_points_identity(tmp_path):
    root, path = real(tmp_path / "as-is")
    other, _ = real(tmp_path / "edited")
    (other / SETUP).write_text((other / SETUP).read_text() + "export FLOCK_SOMETHING=1\n")
    as_is, _ = launches(root, path)
    edited, _ = launches(other, path)
    assert not {item.identity for item in as_is} & {item.identity for item in edited}


# ---- the setup under the launcher ------------------------------------------------------------------------------------------

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
