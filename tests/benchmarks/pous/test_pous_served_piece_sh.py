"""`served_piece.sh`, PoUS's piece of a served window (served_verdicts.sh's POUS_CMD), on a CPU: `lake`, `gpu-lease`
and the piece's Python are stand-ins on PATH, the library is one named already (P2_LIB) and HOME holds no elan. The
library is loaded once by the piece's Python with no card to see, and one it doesn't load, or a node that refuses
`setarch -R` and so builds none, ends the piece before anything else. The two Lean verifiers that decide,
`pous-run-protocol` and `poub-p2-protocol`, are built in a scratch copy of ``verity/`` beside the run's key, the card
and the sessions, and go to `served_piece.py` as ``--case-exe`` and ``--bw-case-exe``; ones named already (POUS_EXE,
BW_CASE_EXE) are passed as they are and nothing is built; one that doesn't build is no answer, and ends the piece with
its own error wherever it was. The piece's Python then runs twice: its sessions under the lease (the node's
`gpu-lease`, else the tree's own; none of the piece's own only inside the caller's), then, once that process ended
and with it the lease, and both builds have ended, its answers with no card; the timeline of both is ``stamps.txt``."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmarks" / "pous" / "served_piece.sh"
READY, RUN_ATTESTER = SCRIPT.parent / "lean_ready.sh", SCRIPT.parent / "run_attester.sh"
CHECK, AUDIT = "verity/protocols/resource_accounting/memory/pous/verifier/check", "verity/protocols/resource_accounting/memory/poub/verifier/lean"
RUN_EXE, BW_EXE = "pous-run-protocol", "poub-p2-protocol"
#: the tree's own lease script, which the piece takes where the node has no `gpu-lease`
TREE_LEASE = "tools/research/src/research/pods/sh/gpu_lease.sh"

LAKE = """#!/usr/bin/env bash
echo "lake $PWD $* $([ -f lakefile.toml ] || [ -f lakefile.lean ] && echo package)" >> "$FAKE_LOG"
[ "$2" != "${FAKE_LAKE_HOLD:-}" ] || ! grep -qs "the Lean builds start" "$FAKE_STAMPS" || read -r _ < "$FAKE_FIFO"
[ "$2" = "${FAKE_LAKE_FAILS:-}" ] && { [ -z "${FAKE_FAIL_IN_SESSIONS:-}" ] \\
                                         || ! grep -qs "the Lean builds start" "$FAKE_STAMPS" || read -r _ < "$FAKE_FIFO"
                                       echo "error: $2 has a hole"; exit 1; }
mkdir -p .lake/build/bin && printf '#!/bin/sh\\n' > ".lake/build/bin/$2" && chmod +x ".lake/build/bin/$2"
echo "lake end $2" >> "$FAKE_LOG"
"""
"""A stand-in `lake build EXE`. FAKE_LAKE_HOLD's build, once the piece has stamped the builds' start, waits for a line
on FAKE_FIFO; FAKE_LAKE_FAILS's fails, with FAKE_FAIL_IN_SESSIONS (and that stamp) once the sessions have started (a line on
FAKE_FIFO)."""
TPY = """#!/usr/bin/env bash
echo "key start" >> "$FAKE_LOG"
! grep -qs "the Lean builds start" "$FAKE_STAMPS" || echo started > "$FAKE_FIFO"
printf '%0128d\\n' 0
"""
"""A stand-in for the tree's Python making the run's key, which releases FAKE_LAKE_HOLD's build."""
LEASE = """#!/usr/bin/env bash
echo "lease $*" >> "$FAKE_LOG"
while [ "$1" != -- ]; do shift; done
shift
GPU_LEASE_UUID=GPU-own0 exec "$@"
"""
PYTHON = """#!/usr/bin/env bash
if [ "$1" = -c ]; then  # the library's load: its path, and the cards this Python may see
  echo "load ${*: -1} [${CUDA_VISIBLE_DEVICES-unset}]" >> "$FAKE_LOG"
  [ -z "${FAKE_LOAD_FAILS:-}" ] && exit 0
  printf 'Traceback (most recent call last):\\n  File "<string>", line 1\\nOSError: %s\\n' "$FAKE_LOAD_FAILS" >&2
  exit 1
fi
part=${*: -1}
echo "piece $part ${CUDA_VISIBLE_DEVICES-unset}" >> "$FAKE_LOG"
printf '%s\\n' "$@" > "$FAKE_ARGV$part"
for a in "$@"; do case $a in */.lake/build/bin/*) [ -x "$a" ] && echo "$part $a" >> "$FAKE_BUILT" ;; esac; done
if [ "$part" = --sessions ] && [ -n "${FAKE_FAIL_IN_SESSIONS:-}" ]; then
  echo started > "$FAKE_FIFO"
  exec tail -f /dev/null  # a session that ends only when it is stopped
fi
[ "$part" = --answers ] || exit "${FAKE_SESSIONS_EXIT:-0}"
exit "${FAKE_ANSWERS_EXIT:-0}"
"""
STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ")


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def node(tmp_path) -> dict[str, str]:
    """The environment of a node with a stand-in `lake` and `gpu-lease` first on PATH, a stand-in for the piece's
    Python, a library built already, and a HOME with no elan, so the stand-in `lake` is the one found."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "lake", LAKE)
    _exe(bin_ / "gpu-lease", LEASE)
    _exe(bin_ / "systemd-run", "#!/bin/sh\nexit 1\n")
    _exe(bin_ / "prlimit", f'#!/bin/sh\necho "$1" >> {tmp_path / "limit"}; shift 2; exec "$@"\n')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GPU_LEASE_", "POUS_", "BW_", "CUDA_VISIBLE", "FAKE_"))
           and k not in ("LEASE_MIN", "MEM_GB", "ANSWERS_MEM_GB")}
    return env | {"PATH": f"{bin_}:{env['PATH']}", "HOME": str(tmp_path / "home"), "P2_LIB": str(tmp_path / "lib"),
                  "PY": str(_exe(tmp_path / "python", PYTHON)), "POUS_NODE": "gpu-node",
                  "FAKE_LOG": str(tmp_path / "log"), "FAKE_ARGV": str(tmp_path / "argv"),
                  "FAKE_BUILT": str(tmp_path / "built")}


def _piece(env: dict[str, str], d: Path, script: Path = SCRIPT) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), str(d.parent / "serve-run"), "3", str(d)], env=env, cwd=ROOT,
                          capture_output=True, text=True)


def _load(env: dict[str, str]) -> list[str]:
    """The log's line for the load of P2_LIB's library, by a Python that sees no card."""
    return ["load", env["P2_LIB"] + "/libp2v1.so", "[]"]


def _events(env: dict[str, str]) -> list[list[str]]:
    log = Path(env["FAKE_LOG"])
    return [line.split() for line in log.read_text().splitlines()] if log.exists() else []


def _log(env: dict[str, str]) -> list[list[str]]:
    """The stand-ins' log but the builds' ends, the builds first: they run beside the rest, in their own order."""
    events = [e for e in _events(env) if e[:2] != ["lake", "end"]]
    return [e for e in events if e[0] == "load"] + [e for e in events if e[0] == "lake"] + \
        [e for e in events if e[0] not in ("load", "lake")]


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _argv(env: dict[str, str], part: str = "--sessions") -> list[str]:
    return Path(env["FAKE_ARGV"] + part).read_text().splitlines()


def _stamps(d: Path) -> list[str]:
    """The events of ``d``'s ``stamps.txt`` in order, each line held to its form: a UTC second, then the event."""
    lines = (d / "stamps.txt").read_text().splitlines()
    assert all(STAMP.match(line) for line in lines), lines
    return [line.split(" ", 1)[1] for line in lines]


def test_the_two_leans_build_in_a_scratch_copy_beside_the_card_and_are_there_for_the_answers(node, tmp_path):
    """The library is loaded first, by the piece's Python with no card to see. Each verifier is built by `lake build`
    in its package of a copy of ``verity/`` outside the tree, whose ``.lake`` is never written, one after the other,
    started before the card is asked for, at the lease's defaults, for the piece's sessions alone; its answers run
    after the lease, with no card to see, once both builds have ended, and get the two built programs, which are
    removed with the scratch copy at the end."""
    d = tmp_path / "pieces" / "pous"
    r = _piece(node, d)
    assert r.returncode == 0, r.stdout + r.stderr
    load, (_, check, *run), (_, audit, *bw), lease, sessions, answers = _log(node)
    assert load == _load(node)
    assert run == ["build", RUN_EXE, "package"] and bw == ["build", BW_EXE, "package"]
    scratch = Path(check[:-len(CHECK)])
    assert check.endswith("/" + CHECK) and audit == str(scratch / AUDIT) and ROOT not in scratch.parents
    assert lease[:7] == ["lease", "1", "--wait", "--max-min", "90", "--mem-gb", "72"] and lease[-1] == "--sessions"
    assert sessions == ["piece", "--sessions", "GPU-own0"] and answers == ["piece", "--answers"]
    argv = _argv(node)
    exes = [str(scratch / CHECK / ".lake/build/bin" / RUN_EXE), str(scratch / AUDIT / ".lake/build/bin" / BW_EXE)]
    assert [_option(argv, "--case-exe"), _option(argv, "--bw-case-exe")] == exes
    built = Path(node["FAKE_BUILT"]).read_text().splitlines()
    assert [b for b in built if b.startswith("--answers ")] == ["--answers " + e for e in exes]
    assert not scratch.exists()
    assert argv[:4] == [str(ROOT / "benchmarks/pous/served_piece.py"), str(d.parent / "serve-run"), "3", str(d)]
    assert _option(argv, "--node") == "gpu-node" and (d / "lake-build.log").exists()
    assert _argv(node, "--answers")[:-1] == argv[:-1]
    stamps = _stamps(d)
    assert [s for s in stamps if not s.startswith("built ")] == [
        "start", "loaded libp2v1.so", "the Lean builds start", "asked for the card", "leased the card",
        "the sessions' part exit 0", "the answers' part exit 0"]
    assert stamps.index("the Lean builds start") < stamps.index(f"built {RUN_EXE}") < stamps.index(f"built {BW_EXE}") \
        < stamps.index("the answers' part exit 0")


def test_where_no_systemd_user_scope_is_made_the_answers_run_behind_prlimit_or_not_at_all(node, tmp_path):
    d, env = tmp_path / "pieces" / "pous", node | {"POUS_EXE": "/built/" + RUN_EXE, "BW_CASE_EXE": "/built/" + BW_EXE}
    assert _piece(env, d).returncode == 0
    assert (tmp_path / "limit").read_text().splitlines() == [f"--as={1 << 30}", f"--as={128 << 30}"]
    assert [line[1] for line in _log(node) if line[0] == "piece"] == ["--sessions", "--answers"]
    _exe(Path(node["PATH"].split(":")[0]) / "prlimit", "#!/bin/sh\nexit 1\n")
    Path(node["FAKE_LOG"]).unlink()
    r = _piece(env, d)
    assert r.returncode == 2 and r.stdout.splitlines()[-1] == (
        "POUS: no answer: this host makes no systemd user scope and runs no prlimit: the answers run only under a "
        "memory limit of their own"), r.stdout + r.stderr
    assert [line[1] for line in _log(node) if line[0] == "piece"] == ["--sessions"]


@pytest.mark.parametrize("exits, code", [({"FAKE_SESSIONS_EXIT": "1"}, 1), ({"FAKE_SESSIONS_EXIT": "2"}, 2),
                                         ({"FAKE_SESSIONS_EXIT": "124"}, 124), ({"FAKE_ANSWERS_EXIT": "1"}, 1)])
def test_the_answers_run_only_after_sessions_that_were_kept_and_their_exit_is_the_pieces(node, tmp_path, exits, code):
    """The sessions' part that exits other than 0 (an answer already, none, the lease's limit) ends the piece with its
    exit and nothing is answered; else the piece's exit is the answers' part's."""
    d = tmp_path / "pieces" / "pous"
    r = _piece(node | exits | {"POUS_EXE": "/built/" + RUN_EXE, "BW_CASE_EXE": "/built/" + BW_EXE}, d)
    assert r.returncode == code, r.stdout + r.stderr
    ran = [line[1] for line in _log(node) if line[0] == "piece"]
    assert ran == (["--sessions"] if "FAKE_SESSIONS_EXIT" in exits else ["--sessions", "--answers"])
    assert _stamps(d)[-1] == (f"the sessions' part exit {code}" if "FAKE_SESSIONS_EXIT" in exits
                              else f"the answers' part exit {code}")


def test_leans_built_already_are_passed_as_they_are_and_nothing_is_built(node, tmp_path):
    """POUS_EXE and BW_CASE_EXE name the two verifiers: no `lake` runs, and in the caller's lease no card is asked
    for. With one of them named, only the other is built."""
    d = tmp_path / "pieces" / "pous"
    named = {"POUS_EXE": "/built/" + RUN_EXE, "BW_CASE_EXE": "/built/" + BW_EXE}
    r = _piece(node | named | {"GPU_LEASE_UUID": "GPU-serve0"}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    both = [["piece", "--sessions", "GPU-serve0"], ["piece", "--answers"]]
    assert _log(node) == [_load(node), *both] and not (d / "lake-build.log").exists()
    argv = _argv(node)
    assert [_option(argv, "--case-exe"), _option(argv, "--bw-case-exe")] == list(named.values())
    assert _stamps(d) == ["start", "loaded libp2v1.so", "asked for the card", "the sessions' part exit 0",
                          "the answers' part exit 0"]

    Path(node["FAKE_LOG"]).unlink()
    r = _piece(node | {"POUS_EXE": named["POUS_EXE"], "GPU_LEASE_UUID": "GPU-serve0"}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    _, (_, audit, *bw), *pieces = _log(node)
    assert audit.endswith("/" + AUDIT) and bw == ["build", BW_EXE, "package"] and pieces == both
    argv = _argv(node)
    assert _option(argv, "--case-exe") == named["POUS_EXE"]
    assert _option(argv, "--bw-case-exe") == audit + "/.lake/build/bin/" + BW_EXE


@pytest.mark.parametrize("exe, builds", [(RUN_EXE, 1), (BW_EXE, 2)])
def test_a_lean_that_does_not_build_is_no_answer_and_its_own_error_ends_the_piece(node, tmp_path, exe, builds):
    """Exit 7, saying which verifier doesn't build, with the build's last lines, wherever the piece was when it failed:
    no answer is started, and the Python never decides in its place."""
    d = tmp_path / "pieces" / "pous"
    r = _piece(node | {"FAKE_LAKE_FAILS": exe}, d)
    assert r.returncode == 7 and r.stdout.count(f"POUS: no answer: {exe} doesn't build (lake-build.log)") == 1
    assert f"error: {exe} has a hole" in r.stdout and f"error: {exe} has a hole" in (d / "lake-build.log").read_text()
    log = _log(node)
    lakes = [line for line in log if line[0] == "lake"]
    assert len(lakes) == builds and ["piece", "--answers"] not in log
    assert not Path(lakes[0][1][:-len(CHECK)]).exists() and not (d / "answer.json").exists()
    stamps = _stamps(d)
    assert f"{exe} didn't build" in stamps and "the answers' part exit 0" not in stamps
    assert stamps[-1].startswith(("the piece stops: ", "the sessions' part stopped: ")), stamps


def test_a_lean_that_fails_while_the_sessions_run_stops_them_with_its_own_error(node, tmp_path):
    """The build of `poub-p2-protocol` fails once the sessions have started (a session that ends only when stopped):
    the sessions are stopped, and the piece ends with exit 7 and the build's own error, with no answer started."""
    d, fifo = tmp_path / "pieces" / "pous", tmp_path / "fifo"
    os.mkfifo(fifo)
    r = _piece(node | {"FAKE_LAKE_FAILS": BW_EXE, "FAKE_FAIL_IN_SESSIONS": "1", "FAKE_FIFO": str(fifo),
                       "FAKE_STAMPS": str(d / "stamps.txt")}, d)
    assert r.returncode == 7, r.stdout + r.stderr
    assert r.stdout.strip().splitlines()[-1] == f"POUS: no answer: {BW_EXE} doesn't build (lake-build.log)"
    assert f"error: {BW_EXE} has a hole" in r.stdout
    assert [line[:2] for line in _log(node) if line[0] == "piece"] == [["piece", "--sessions"]]
    stamps = _stamps(d)
    assert stamps[-1] == "the sessions' part stopped: a Lean verifier didn't build"
    assert stamps.index("leased the card") < stamps.index(f"{BW_EXE} didn't build")


def test_the_runs_key_is_made_while_the_leans_build_and_the_party_waits_for_its_own(node, tmp_path):
    """With --run-keys the piece makes the run's key while the verifiers build: the stand-in `lake`'s first build
    ends only after the key's maker has started. `poub-verifier`, the bandwidth party's program, is built too, and
    the party's command waits for it (`lean_ready.sh`) rather than the piece."""
    d, fifo = tmp_path / "pieces" / "pous", tmp_path / "fifo"
    os.mkfifo(fifo)
    env = node | {"POUS_ARGS": "--run-keys", "TPY": str(_exe(tmp_path / "tpy", TPY)), "FAKE_LAKE_HOLD": RUN_EXE,
                  "FAKE_FIFO": str(fifo), "FAKE_STAMPS": str(d / "stamps.txt")}
    r = _piece(env, d)
    assert r.returncode == 0, r.stdout + r.stderr
    events = [" ".join(e[:3]) if e[0] == "lake" else " ".join(e[:2]) for e in _events(node)]
    ends = [i for i, e in enumerate(events) if e.startswith("lake end")]
    assert len(ends) == 3 and events.index("key start") < min(ends), events
    assert [e.split()[2] for e in events if e.startswith("lake end")] == [RUN_EXE, "poub-verifier", BW_EXE]
    party = _option(_argv(node), "--bw-party").split()
    assert party[:2] == ["bash", str(READY)] and party[3:6] == ["poub-verifier", "bash", str(RUN_ATTESTER)]
    assert party[-1].endswith("/.lake/build/bin/poub-verifier")


def test_lean_ready_runs_its_command_once_the_build_ends_and_else_gives_the_builds_own_error(tmp_path):
    """`lean_ready.sh DIR EXE CMD...` waits on the build's lock: CMD runs once the build that holds it has ended
    and left DIR/EXE.ok; a build that failed (no EXE.ok) gives its own error (DIR/failed) and exit 7, and CMD never runs."""
    built, ran = tmp_path / "built", tmp_path / "ran"
    built.mkdir()
    with open(built / f"{RUN_EXE}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        waiting = subprocess.Popen(["bash", str(READY), str(built), RUN_EXE, "touch", str(ran)])
        (built / f"{RUN_EXE}.ok").touch()
    assert waiting.wait() == 0 and ran.exists()
    (built / f"{BW_EXE}.lock").touch()
    (built / "failed").write_text(f"error: {BW_EXE} has a hole\n")
    r = subprocess.run(["bash", str(READY), str(built), BW_EXE, "touch", str(tmp_path / "never")],
                       capture_output=True, text=True)
    assert r.returncode == 7 and r.stderr == f"error: {BW_EXE} has a hole\n" and not (tmp_path / "never").exists()


def test_a_library_the_pieces_python_does_not_load_ends_the_piece_before_anything_is_built(node, tmp_path):
    """The sessions open the library only once they hold the card, so the piece loads it first, in the same Python with
    no card to see: one that doesn't load is exit 3 with the loader's own last line, naming that Python and the
    library; no verifier is built, no card is asked for and no session starts."""
    d = tmp_path / "pieces" / "pous"
    why = "libstdc++.so.6: version `GLIBCXX_3.4.99' not found"
    r = _piece(node | {"FAKE_LOAD_FAILS": why}, d)
    assert r.returncode == 3, r.stdout + r.stderr
    assert r.stdout.strip() == f"POUS: no answer: {node['PY']} doesn't load {_load(node)[1]}: OSError: {why}"
    assert _log(node) == [_load(node)] and _stamps(d) == ["start"] and not (d / "lake-build.log").exists()


def test_where_setarch_r_is_refused_no_library_is_built_and_the_piece_names_p2_lib(node, tmp_path):
    """The library's build turns address randomization off (`setarch -R`). On a node that refuses that, a piece with no
    P2_LIB says so and says what to do, exit 3, before it builds, loads or asks for anything."""
    d = tmp_path / "pieces" / "pous"
    _exe(Path(node["PATH"].split(":")[0]) / "setarch", "#!/bin/sh\nexit 1\n")
    r = _piece({k: v for k, v in node.items() if k != "P2_LIB"}, d)
    assert r.returncode == 3, r.stdout + r.stderr
    said = r.stdout.strip()
    assert said.startswith("POUS: no answer: setarch -R is refused on this node, so libp2v1.so doesn't build here")
    assert said.endswith("(verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh) and name its directory in P2_LIB")
    assert _log(node) == [] and _stamps(d) == ["start"] and not (d / "build.log").exists()


def test_a_node_with_no_gpu_lease_gets_the_trees_own_and_only_the_callers_lease_goes_without_one(node, tmp_path):
    """Where no `gpu-lease` is installed the piece takes its card by the tree's own lease script, with the same
    request, and its sessions run on the card that hands out; they run with no lease of the piece's own only inside
    the caller's (GPU_LEASE_UUID), on its card. The script is a copy in a tree whose lease script is a stand-in; the
    tree's own is where the piece looks for it."""
    tree, d = tmp_path / "tree", tmp_path / "pieces" / "pous"
    script = tree / "benchmarks" / "pous" / "served_piece.sh"
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT.read_text())
    (tree / TREE_LEASE).parent.mkdir(parents=True)
    (tree / TREE_LEASE).write_text(LEASE.replace("GPU-own0", "GPU-tree0"))
    path = ":".join(p for p in node["PATH"].split(":")[1:] if not (Path(p) / "gpu-lease").exists())
    assert shutil.which("gpu-lease", path=path) is None and (ROOT / TREE_LEASE).is_file()
    env = node | {"PATH": path, "POUS_EXE": "/built/" + RUN_EXE, "BW_CASE_EXE": "/built/" + BW_EXE}
    r = _piece(env, d, script)
    assert r.returncode == 0, r.stdout + r.stderr
    load, lease, sessions, answers = _log(node)
    assert load == _load(node) and sessions == ["piece", "--sessions", "GPU-tree0"]
    assert lease[:7] == ["lease", "1", "--wait", "--max-min", "90", "--mem-gb", "72"] and lease[-1] == "--sessions"
    assert answers == ["piece", "--answers"]
    assert _stamps(d) == ["start", "loaded libp2v1.so", "asked for the card", "leased the card",
                          "the sessions' part exit 0", "the answers' part exit 0"]

    Path(node["FAKE_LOG"]).unlink()
    r = _piece(env | {"GPU_LEASE_UUID": "GPU-serve0"}, d, script)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _log(node) == [_load(node), ["piece", "--sessions", "GPU-serve0"], ["piece", "--answers"]]


@pytest.mark.parametrize("env, want", [
    ({}, "120m"), ({"PIECE_TIMEOUT": "45m"}, "45m"), ({"POUS_TIMEOUT": "3h", "PIECE_TIMEOUT": "45m"}, "3h"),
    ({"POUS_CMD": ""}, "")])
def test_the_composed_run_gives_the_piece_time_for_its_leans(tmp_path, env, want):
    """`onsite_piece_commands` sets the piece's limit with its command, exported for `served_verdicts.py pieces`:
    POUS_TIMEOUT where set, else PIECE_TIMEOUT's, else its own default; with no PoUS piece it sets none."""
    tree, home = tmp_path / "tree", tmp_path / "home"
    (tree / "benchmarks/pous").mkdir(parents=True)
    (tree / "benchmarks/pous/served_piece.sh").write_text("")
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh/vy-auditor2-pous").write_text("")
    got = subprocess.run(["bash", "-uc", f'. {SCRIPT.parent / "onsite_pieces.sh"}; onsite_piece_commands; '
                          'exec env'], env={"PATH": os.defpath, "HOME": str(home), "TREE": str(tree)} | env,
                         check=True, capture_output=True, text=True).stdout.splitlines()
    assert [line for line in got if line.startswith("POUS_TIMEOUT=")] == ([f"POUS_TIMEOUT={want}"] if want else [])


@pytest.mark.parametrize("repo, revision, rc", [
    ("verity-int8/Llama-3.2-1B-INT8", "15b0976d400d796ee589e7737d03a919526bf800", 0),
    ("verity-int8/Llama-3.2-1B-INT8", "0" * 40, 3),
    ("nobody/no-model", "0" * 40, 3),
])
def test_the_preflight_asks_the_piece_itself_whether_it_pins_a_checkpoint(tmp_path, repo, revision, rc):
    """`onsite_pous_pins REPO REVISION` is served_verdicts.sh's preflight question, answered by the piece's own lookup
    (served_piece.py's `_pin` over its `PINS`) and by no list of its own: 0 for a checkpoint the piece pins at that
    revision, 3 with the piece's refusal for one it doesn't, before a run serves a window the piece would then not
    answer. It runs as the piece does, here under this suite's Python."""
    got = subprocess.run(["bash", "-uc", f'. {SCRIPT.parent / "onsite_pieces.sh"}; onsite_pous_pins "$1" "$2"', "_",
                          repo, revision], env={"PATH": os.defpath, "HOME": str(tmp_path), "TREE": str(ROOT),
                                                "PY": sys.executable},
                         check=False, capture_output=True, text=True)
    assert got.returncode == rc, got.stderr[-600:]
    assert (got.stdout == "") if rc == 0 else (f"{repo}@{revision}" in got.stdout and got.stdout.count("\n") == 1)


def test_the_scratch_copy_holds_every_package_the_two_leans_require():
    """`served_piece.sh` copies ``verity/`` alone: each package the two verifiers' manifests require is a path
    requirement inside it, so the copy builds with no network and nothing of the tree outside it."""
    for package in (CHECK, AUDIT):
        manifest = json.loads((ROOT / package / "lake-manifest.json").read_text())
        for p in manifest["packages"]:
            assert p["type"] == "path", f"{package} requires {p['name']} from {p.get('url')}, which no copy holds"
            assert ROOT / "verity" in (ROOT / package / p["dir"]).resolve().parents, (package, p["name"], p["dir"])
