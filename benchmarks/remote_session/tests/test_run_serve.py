"""`run.sh`'s served sessions (SERVE=tiny, SERVE=row) where no GPU is: what `serve.py` decides for it (the windows
that fit a run, `fit`; how a session ended and the exit code that says so, `verdict`), and `run.sh` itself on a node
of stubs, which checks what it refuses before the lease, its exit codes, its port, its defaults, the order of what it
does around the lease, and that no verifier is left behind.

The node of stubs: a vLLM venv whose interpreter is this one, except that the verifier and the prover it would start
are `VERIFIER` and `PROVER` below (a verifier that listens, takes one connection and writes verifier.json as the word
it is sent says; a prover that connects, sends that word and writes its record), with `verity-vllm`, `gpu-lease` and
`nvcc` beside it, each writing what it was asked to `calls`, and an `ss` and an `nvidia-smi` that answer from files.
`taskset` and the rest are the system's own (`MACHINE_PATH`)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_serve():
    """`serve.py` as a module of its own name: the auditor host's challenger, which this suite's other tests run,
    imports its own `serve` by that name."""
    spec = importlib.util.spec_from_file_location("remote_session_serve", HERE.parent / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = load_serve()
RUN = HERE.parent / "run.sh"
BASH = shutil.which("bash")
TREE_LEASE = "tools/research/src/research/pods/sh/gpu_lease.sh"
MACHINE_PATH = ":".join(d for d in os.defpath.split(":") if d and not (Path(d) / "gpu-lease").exists())
"""The system's default search path, less any directory that holds a `gpu-lease`: a line whose stub lease a test took
away finds none, never the lease program of the machine the suite runs on.  The node's stubs come before it."""
ROW = "llama__bf16__h200__tp1__b1__i256__o32"

INTERPRETER = r"""#!/bin/bash
export STUB_INTERPRETER=$0
case " $* " in
  *" -c "*torch.version.cuda*) [ -n "${STUB_TORCH_CUDA:-}" ] || exit 1; echo "$STUB_TORCH_CUDA"; exit 0 ;;
  *" -c "*lean_verifier*)
    echo "lean-build" >> "$STUB_DIR/calls"
    [ -n "${STUB_LEAN:-}" ] || { echo "lake build remote-audit-verifier failed" >&2; exit 1; }
    echo "$STUB_LEAN"; exit 0 ;;
  *" -c "*config.row_file*)
    [ -n "${STUB_ROW_FILE:-}" ] || { echo "UsageError: the row has a file in two directories" >&2; exit 1; }
    echo "$STUB_ROW_FILE"; exit 0 ;;
  *verifier_process.py*) exec "$REAL_PY" "$STUB_DIR/verifier.py" "$@" ;;
  *"serve.py --connect"*) exec "$REAL_PY" "$STUB_DIR/prover.py" tiny "$@" ;;
esac
exec "$REAL_PY" "$@"
"""

VERIFIER = r"""
import json, os, signal, socket, sys
a = sys.argv[2:]
opt = lambda name: a[a.index(name) + 1]
stub, out, port = os.environ["STUB_DIR"], opt("--out"), int(opt("--port"))
with open(f"{stub}/calls", "a") as f:
    f.write("verifier\n")
with open(f"{stub}/verifier.seen.json", "w") as f:
    json.dump({"argv": a, "cpus": sorted(os.sched_getaffinity(0)), "pid": os.getpid(), "pgid": os.getpgid(0)}, f)
mode = os.environ.get("STUB_VERIFIER", "session")
if mode == "dead":
    print("verifier: the registry can't be read")
    sys.exit(2)
if mode == "deaf":
    print(f"verifier: listening on 127.0.0.1:{port}", flush=True)
    signal.pause()
server = socket.create_server(("127.0.0.1", port))
print(f"verifier: listening on 127.0.0.1:{port}", flush=True)
conn, _ = server.accept()
said = b""
while chunk := conn.recv(64):
    said += chunk
if not said:
    raise ConnectionError("the connection closed before a hello")
session = {"accepted": said == b"accepted", "failure": None if said == b"accepted" else {"failure": "late", "t": 2},
           "windows_seen": [{"t": 0, "passes": 4}, {"t": 1, "passes": 4}, {"t": 2, "passes": 1}]}
with open(f"{out}/verifier.json", "w") as f:
    json.dump({"sessions": [session]}, f)
print(f"verifier: session 0 accepted={session['accepted']}", flush=True)
"""

PROVER = r"""
import json, os, socket, sys
kind, a = sys.argv[1], sys.argv[2:]
stub, mode = os.environ["STUB_DIR"], os.environ.get("STUB_PROVER", "accepted")
if kind == "row" and a[:2] == ["epoch", "resolve"]:
    print(os.environ.get("STUB_EPOCH", "CHECKPOINT org/model 0123abcd"))
    sys.exit(0)
stages = a[len(a) - a[::-1].index("--stages")] if "--stages" in a else None
what = "tiny" if kind == "tiny" else f"row {stages}"
keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "GPU_LEASE", "REMOTE_VERIFIER", "CUDA_HOME",
        "EXTRA_COMMIT_ARGS", "CONFIG_RUN", "NATIVE_COLLECT_BUILD", "HF_HUB_OFFLINE", "STUB_INTERPRETER", "PYTHONPATH")
with open(f"{stub}/calls", "a") as f:
    f.write(what + "\n")
with open(f"{stub}/prover.jsonl", "a") as f:
    f.write(json.dumps({"what": what, "argv": a, "cpus": sorted(os.sched_getaffinity(0)),
                        "sweep_made": "--sweep-dir" in a and os.path.isdir(a[a.index("--sweep-dir") + 1]),
                        "env": {k: os.environ.get(k) for k in keys}}) + "\n")
if what == "row build":
    sys.exit(int(os.environ.get("STUB_BUILD_RC", "0")))
if kind == "tiny":
    address, record = a[a.index("--connect") + 1], a[a.index("--out") + 1] + "/serve.json"
    wrap = lambda accepted: {"prover": {"accepted": accepted, "failure": None if accepted else "receipt"}}
    red = 6
else:
    address = os.environ["REMOTE_VERIFIER"]
    record = f"{a[a.index('--sweep-dir') + 1]}/{a[2]}/commit/remote_session.json"
    wrap = lambda accepted: {"accepted": accepted, "error": None if accepted else
                             "RuntimeError: the run ended after 32 steps, before the session's last pass"}
    red = 12
    os.makedirs(os.path.dirname(record), exist_ok=True)
if mode in ("crash", "commitfail"):
    print("Traceback: the engine could not be built")
    sys.exit(1 if kind == "tiny" else red)
if mode == "quiet":
    sys.exit(0)
word = b"rejected" if mode == "rejected" else b"accepted"
host, port = address.split(":")
with socket.create_connection((host, int(port))) as s:
    s.sendall(word)
with open(record, "w") as f:
    json.dump(wrap(word == b"accepted"), f)
sys.exit({"accepted": 0, "rejected": red, "accepted-then-red": red}[mode])
"""

LEASE = r"""#!/bin/bash
echo "lease $0 $*" >> "$STUB_DIR/calls"
[ -n "${STUB_LEASE_RC:-}" ] && exit "$STUB_LEASE_RC"
while [ "$1" != -- ]; do shift; done
shift
exec "$@"
"""

LEASE_CORES = LEASE + "# gpu-lease N [--wait] [--on LIST] [--cores CPUS] [--max-min M] -- CMD...\n"
"""A lease program whose text names ``--cores``; `LEASE` names none, as a lease program older than the option."""

NVCC = r"""#!/bin/bash
echo "nvcc $*" >> "$STUB_DIR/calls"
while [ $# -gt 0 ]; do
  [ "$1" = -o ] && : > "$2"
  shift
done
exit "${STUB_NVCC_RC:-0}"
"""

SS = r"""#!/bin/bash
grep -qx "${2##*:}" "$STUB_DIR/busy" 2> /dev/null && echo "LISTEN 0 128 127.0.0.1:${2##*:} 0.0.0.0:*"
exit 0
"""

LEASE_BASH = r"""#!/bin/bash
case "$1" in
  */gpu_lease.sh)
    echo "lease bash $*" >> "$STUB_DIR/calls"
    while [ "$1" != -- ]; do shift; done
    shift
    exec "$@" ;;
esac
exec "$REAL_BASH" "$@"
"""

NVIDIA_SMI = r"""#!/bin/bash
n=$(cat "$STUB_DIR/smi.asked" 2> /dev/null || echo 0)
echo "$((n + 1))" > "$STUB_DIR/smi.asked"
echo "$*" > "$STUB_DIR/smi.argv"
cat "$STUB_DIR/apps.$n" 2> /dev/null
exit 0
"""


def program(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def row_file(path: Path, steps: int | None = 32, target: dict | None = None) -> Path:
    declared = {"compute_capability": [9, 0], "num_sms": 132, "async_scheduling": False}
    doc = {"row_id": ROW, "engine": {"target": declared if target is None else target}}
    path.write_text(json.dumps(doc if steps is None else doc | {"expected_engine_steps": steps}))
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gone(pid: int) -> bool:
    """Whether no process ``pid`` is left: `run.sh` waits for the verifier it started, or stopped, before it ends."""
    return not Path(f"/proc/{pid}").exists()


class Node:
    """A node of stubs under ``tmp``: its workspace (``ws``, the vLLM venv at ``venv``), its CUDA toolkits (``cuda``),
    its programs (``bin``, on the PATH) and what the stubs wrote (``stub``); `run` runs `run.sh` on it."""

    def __init__(self, tmp: Path, venv: str = "venv312", toolkits: tuple[str, ...] = ("cuda-12.9", "cuda-13.0", "cuda")):
        self.tmp, self.ws, self.stub, self.bin = tmp, tmp / "ws", tmp / "stub", tmp / "bin"
        self.out = tmp / "run" / "out"
        for d in (self.stub, tmp / "scratch", tmp / "home", *(tmp / "cuda" / t for t in toolkits)):
            d.mkdir(parents=True)
        (self.stub / "verifier.py").write_text(VERIFIER)
        (self.stub / "prover.py").write_text(PROVER)
        self.venv(venv)
        self.lease = program(self.stub / "gpu-lease", LEASE)
        self.lean = program(self.stub / "remote-audit-verifier", "#!/bin/bash\n")
        program(self.bin / "nvcc", NVCC)
        program(self.bin / "ss", SS)
        program(self.bin / "nvidia-smi", NVIDIA_SMI)
        self.row = row_file(tmp / f"{ROW}.json")

    def venv(self, name: str) -> Path:
        py = program(self.ws / name / "bin" / "python", INTERPRETER)
        program(py.parent / "verity-vllm", '#!/bin/bash\nexec "$REAL_PY" "$STUB_DIR/prover.py" row "$@"\n')
        return py

    def run(self, **env: str | None) -> subprocess.CompletedProcess:
        base = {"PATH": f"{self.bin}:{MACHINE_PATH}", "HOME": str(self.tmp / "home"), "TMPDIR": str(self.tmp / "scratch"),
                "RESEARCH_RUN_DIR": str(self.tmp / "run"), "NODE_WORKSPACE": str(self.ws), "CUDA_ROOT": str(self.tmp / "cuda"),
                "STUB_DIR": str(self.stub), "REAL_PY": sys.executable, "REAL_BASH": BASH, "STUB_TORCH_CUDA": "12.9",
                "GPU_LEASE": str(self.lease), "LEAN_EXE": str(self.lean), "TUNNEL_WAIT_S": "0"}
        env = {k: v for k, v in (base | env).items() if v is not None}
        return subprocess.run([BASH, str(RUN)], cwd=ROOT, env=env, capture_output=True, text=True)

    def both(self, serve_as: str, **env: str | None) -> subprocess.CompletedProcess:
        """ROLE=both SERVE=``serve_as`` on a free port; a row is `ROW`, its file this node's."""
        row = {"ROW": ROW, "ROW_FILE": str(self.row)} if serve_as == "row" else {}
        return self.run(**({"ROLE": "both", "SERVE": serve_as, "PORT": str(free_port())} | row | env))

    def calls(self) -> list[str]:
        f = self.stub / "calls"
        return f.read_text().splitlines() if f.exists() else []

    def provers(self) -> list[dict]:
        f = self.stub / "prover.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def verifier(self) -> dict:
        return json.loads((self.stub / "verifier.seen.json").read_text())

    def scratch(self) -> list[Path]:
        return list((self.tmp / "scratch").iterdir())


@pytest.fixture
def node(tmp_path):
    return Node(tmp_path)


def said(p: subprocess.CompletedProcess) -> str:
    return p.stdout + p.stderr[-3000:]


def cores() -> tuple[list[int], list[int]]:
    """(the verifier's, the prover's) of this process's cores under ROLE=both: the last quarter and the rest."""
    c = sorted(os.sched_getaffinity(0))
    v = max(1, len(c) // 4)
    return c[-v:], c[:len(c) - v] or c


# ---------------------------------------------------------------------------------------------- serve.py: the windows

def test_the_windows_are_derived_from_the_run_and_a_stated_one_is_checked_by_the_same_sum():
    assert serve.fit(32, 500) == (3, 2000) == serve.fit(None, 500)       # four passes a window, a row's run or no end
    assert serve.fit(10, 500) == (3, 1000)                               # half the room a short run leaves: two
    assert serve.fit(32, 500, windows=5) == (5, 1000)
    assert serve.fit(32, 580, 3, 3000) == (3, 3000)
    with pytest.raises(serve.Refused, match="the run is 32 steps.*WINDOW_MS under 5800 fits"):
        serve.fit(32, 580, 3, 11000)
    with pytest.raises(serve.Refused, match="holds no pass at 500 ms"):
        serve.fit(32, 500, 3, 400)
    with pytest.raises(serve.Refused, match="state fewer WINDOWS"):
        serve.fit(9, 500)
    with pytest.raises(serve.Refused, match="positive"):
        serve.fit(32, 0)


def test_every_plan_fit_returns_ends_inside_the_run_with_a_pass_in_every_window():
    """Whatever `fit` returns, derived or stated, the session's last pass comes within the run's steps even when every
    window holds one pass more than its length divides into, and no window is shorter than a pass."""
    kept = refused = 0
    for steps in range(1, 70):
        for windows in range(1, 7):
            for pass_ms in (100, 500, 580):
                for stated in (None, 250, 1000, 3000, 11000):
                    try:
                        w, t = serve.fit(steps, pass_ms, windows, stated)
                    except serve.Refused:
                        refused += 1
                        continue
                    kept += 1
                    assert w == windows and t // pass_ms >= 1 and stated in (None, t)
                    assert serve.session_steps(w, t, pass_ms) <= steps
    assert kept > 1000 and refused > 1000


def test_a_row_file_states_the_runs_steps_and_synchronous_scheduling(tmp_path):
    assert serve.row_steps(row_file(tmp_path / "a.json")) == 32
    with pytest.raises(serve.Refused, match="can't be read"):
        serve.row_steps(tmp_path / "absent.json")
    with pytest.raises(serve.Refused, match="states no expected_engine_steps"):
        serve.row_steps(row_file(tmp_path / "b.json", steps=None))
    for target in ({"ir": "boolean"}, {"async_scheduling": True}):
        with pytest.raises(serve.Refused, match="async_scheduling"):
            serve.row_steps(row_file(tmp_path / "c.json", target=target))


def test_fit_from_the_command_line(tmp_path, capsys):
    row = row_file(tmp_path / "row.json")
    assert serve.main(["--fit", "--pass-ms", "500", "--row-file", str(row)]) == 0
    assert capsys.readouterr().out == "3 2000\n"
    assert serve.main(["--fit", "--pass-ms", "580", "--row-file", str(row), "--window-ms", "11000"]) == serve.REFUSED == 3
    assert capsys.readouterr().out.startswith("refused: 3 windows of 11000 ms take up to 58 engine steps")


# ---------------------------------------------------------------------------------------------- serve.py: the verdict

ACCEPTED, REJECTED = {"accepted": True}, {"accepted": False, "failure": {"failure": "late", "t": 2},
                                           "windows_seen": [{"passes": 5}, {"passes": 6}, {"passes": 2}]}


@pytest.mark.parametrize("prover_rc, verifier, record, verifier_rc, code", [
    (0, {"sessions": [ACCEPTED]}, {"accepted": True}, 0, 0),
    (0, None, {"accepted": True}, 0, 0),                               # ROLE=prover: the verifier is not this run's
    (12, {"sessions": [REJECTED]}, {"accepted": False, "error": "late"}, 0, 6),
    (6, {"sessions": [REJECTED]}, {"accepted": False}, 0, 6),
    (0, {"sessions": [REJECTED]}, {"accepted": True}, 0, 6),          # whatever the prover says and exits with
    (12, None, {"accepted": False, "error": "ConnectionRefusedError"}, 0, 6),
    (12, {"sessions": [ACCEPTED]}, {"accepted": True}, 0, 12),        # a Commit that failed beside an accepted session
    (12, {}, None, 1, 12),                                             # and one that failed before any session
    (1, {}, None, 1, 1),
    (75, {}, None, 1, 75),
    (0, {}, {"accepted": True}, 0, 5),                                 # a green prover and no session on the verifier's record
    (0, {"refused": "the registry holds this store"}, {"accepted": True}, 3, 5),
    (0, {"sessions": [ACCEPTED]}, None, 0, 5),
    (0, None, None, 0, 5),
    (0, {"sessions": [ACCEPTED]}, {"accepted": True}, 1, 5),          # accepted on record beside a verifier that ended red
])
def test_the_verdict_ends_0_only_for_an_accepted_session_and_apart_for_a_rejected_one(
        prover_rc, verifier, record, verifier_rc, code):
    rc, line = serve.verdict(prover_rc, verifier, record, 3000, verifier_rc)
    assert rc == code, line
    assert (rc == 0) == line.startswith("session accepted")


def test_a_rejected_sessions_line_says_why_and_what_a_pass_took():
    rc, line = serve.verdict(12, {"sessions": [REJECTED]}, {"accepted": False, "error": "the run ended"}, 3000)
    assert rc == serve.REJECTED == 6
    assert 'the verifier rejected it, {"failure": "late", "t": 2}' in line and "the prover's record: the run ended" in line
    assert "its passes came every 545 ms: state PASS_MS=545" in line   # 11 passes in the two windows that ended


def test_verdict_from_the_command_line_reads_the_records(tmp_path, capsys):
    (tmp_path / "verifier.json").write_text(json.dumps({"sessions": [ACCEPTED]}))
    (tmp_path / "serve.json").write_text(json.dumps({"plan": "serve", "prover": {"accepted": True}}))
    args = ["--verdict", "--out", str(tmp_path), "--record", str(tmp_path / "serve.json"), "--window-ms", "2000"]
    assert serve.main([*args, "--prover-rc", "0", "--verifier-rc", "0"]) == 0
    assert capsys.readouterr().out == "remote session verdict: session accepted (1 on the verifier's record) (exit 0)\n"
    assert serve.main([*args, "--prover-rc", "0", "--verifier-rc", "1"]) == serve.VERIFIER_RED == 5
    (tmp_path / "verifier.json").write_text("{")                       # a record cut short is no record
    assert serve.main([*args, "--prover-rc", "0", "--verifier-rc", "0"]) == 5
    assert serve.main([*args, "--prover-rc", "0"]) == 0                # ROLE=prover: the prover's record alone
    assert "(exit 0)" in capsys.readouterr().out.splitlines()[-1]


# ------------------------------------------------------------------------------------------- run.sh: an accepted line

def test_one_line_of_tiny_ends_0_with_its_verifier_started_and_its_session_accepted(node):
    port = str(free_port())
    p = node.both("tiny", PORT=port)
    assert p.returncode == 0, said(p)
    assert "remote session verdict: session accepted (1 on the verifier's record) (exit 0)" in p.stdout
    assert (node.out / "verdict.txt").read_text().startswith("remote session verdict: session accepted")
    v, (prover,) = node.verifier(), node.provers()
    argv = " ".join(v["argv"])
    # the windows derived for a run with no end, ten passes of jitter, one session, the Lean verifier built before
    assert f"--host 127.0.0.1 --port {port} " in argv and f"--lean-exe {node.lean}" in argv
    assert "--windows 3 --window-ms 2000 --samples 2773 --jitter-ms 5000 --sessions 1" in argv
    assert prover["argv"][1:] == ["--connect", f"127.0.0.1:{port}", "--out", str(node.out), "--plan", "serve",
                                  "--codec", "p512/k2"]
    # the vLLM venv is node A's, found without VLLM_PY or PY; pinned, with thread counts; one lease, after the verifier
    assert prover["env"]["STUB_INTERPRETER"] == str(node.ws / "venv312/bin/python")
    assert (v["cpus"], prover["cpus"]) == cores()
    assert prover["env"]["OMP_NUM_THREADS"] == prover["env"]["MKL_NUM_THREADS"] == str(len(prover["cpus"]))
    assert prover["env"]["HF_HUB_OFFLINE"] == "1" and prover["env"]["GPU_LEASE"] is None
    assert prover["env"]["PYTHONPATH"].startswith(f"{ROOT}/adapters/vllm:{ROOT}")
    assert prover["env"]["CUDA_HOME"] == str(node.tmp / "cuda/cuda-12.9")
    lease, = [c for c in node.calls() if c.startswith("lease")]
    assert node.calls() == ["verifier", lease, "tiny"]
    assert lease.startswith(f"lease {node.lease} 1 --wait --max-min 30 -- taskset -c ")
    assert f"the lease program {node.lease} names no --cores: cores " in p.stdout      # and is told none
    assert gone(v["pid"]) and v["pid"] == v["pgid"] and node.scratch() == []


def test_one_line_of_a_row_builds_before_the_lease_and_commits_under_it(node):
    port = str(free_port())
    p = node.both("row", PORT=port, ROW_ARGS="--norm-tap-so /taps/norm.so")
    assert p.returncode == 0, said(p)
    assert "remote session verdict: session accepted (1 on the verifier's record) (exit 0)" in p.stdout
    build, commit = node.provers()
    calls = node.calls()
    lease, = [c for c in calls if c.startswith("lease")]
    nvcc, = [c for c in calls if c.startswith("nvcc")]
    # kv.cu's build, the Build with no GPU visible and the verifier all before the lease; the Commit alone under it
    assert calls == [nvcc, "row build", "verifier", lease, "row commit"]
    assert build["env"]["CUDA_VISIBLE_DEVICES"] == "" and commit["env"]["CUDA_VISIBLE_DEVICES"] is None
    sweep = build["argv"][build["argv"].index("--sweep-dir") + 1]
    assert sweep.startswith(str(node.tmp / "scratch")) and node.scratch() == []      # scratch, gone with the run
    assert build["sweep_made"]                                # and there before the row's first stage runs in it
    head = ["row", "run", ROW, "CHECKPOINT", "org/model", "0123abcd", "--py", str(node.ws / "venv312/bin/python"),
            "--sweep-dir", sweep, "--cuda-bin", str(node.tmp / "cuda/cuda-12.9/bin"), "--norm-tap-so", "/taps/norm.so",
            "--stages"]
    assert build["argv"] == [*head, "build"] and commit["argv"] == [*head, "commit"]
    assert lease.endswith(" ".join([str(node.ws / "venv312/bin/verity-vllm"), *head, "commit"]))
    assert build["env"]["REMOTE_VERIFIER"] is None           # the session is the Commit's: its Build sees no verifier
    # the registration is encoded on the card: kv.cu built for it at the session's codec, handed to the Commit
    assert nvcc.startswith("nvcc -O3 -arch=sm_90 ") and "-DKV_FIELD=1024 -DKV_MASK=1" in nvcc
    assert nvcc.endswith("benchmarks/word_codec/kv.cu") and (ROOT / "benchmarks/word_codec/kv.cu").is_file()
    assert commit["env"]["EXTRA_COMMIT_ARGS"].split() == [
        "--remote-plan", "serve", "--remote-kv-pages", "24", "--remote-weight-blocks", "4096", "--remote-kv-lib",
        f"{Path(sweep).parent}/kv.so"]
    assert commit["env"]["REMOTE_VERIFIER"] == f"127.0.0.1:{port}" and commit["env"]["CONFIG_RUN"] == "1"
    assert commit["env"]["GPU_LEASE"] is None                # the Commit asks for no lease of its own
    assert commit["env"]["NATIVE_COLLECT_BUILD"].startswith(str(node.tmp / "scratch"))
    assert build["cpus"] == commit["cpus"] == cores()[1] and commit["env"]["OMP_NUM_THREADS"] == str(len(commit["cpus"]))
    # the windows fit the row's 32 steps, and the row's records are kept
    assert "--windows 3 --window-ms 2000 " in " ".join(node.verifier()["argv"])
    assert json.loads((node.out / "row/commit/remote_session.json").read_text())["accepted"] is True


# ---------------------------------------------------------------------------------------------- run.sh: the exit codes

@pytest.mark.parametrize("serve_as, prover, code, line", [
    ("tiny", "rejected", 6, "session not accepted: the verifier rejected it"),
    ("row", "rejected", 6, "the prover's record: RuntimeError: the run ended after 32 steps"),
    ("row", "accepted-then-red", 12, "the prover ended 12 with its session accepted"),
    ("row", "commitfail", 12, "the prover ended 12 with no session on record"),
    ("tiny", "crash", 1, "the prover ended 1 with no session on record"),
])
def test_a_rejected_session_a_failed_commit_and_a_crash_end_apart(node, serve_as, prover, code, line):
    p = node.both(serve_as, STUB_PROVER=prover)
    assert p.returncode == code, said(p)
    verdict, = [x for x in p.stdout.splitlines() if x.startswith("remote session verdict: ")]
    assert line in verdict and verdict.endswith(f"(exit {code})")
    if prover == "rejected":
        assert "its passes came every 500 ms: state PASS_MS=500" in verdict
    # a red prover that never connected leaves no verifier listening: it ends at the connection the line drops, and
    # the line does not wait VERIFIER_WAIT_S to stop it
    assert gone(node.verifier()["pid"]) and "was still running" not in p.stdout


def test_a_green_prover_with_no_accepted_session_ends_5_and_its_verifier_is_stopped(node):
    p = node.both("tiny", STUB_PROVER="quiet", STUB_VERIFIER="deaf", VERIFIER_WAIT_S="0")
    assert p.returncode == 5, said(p)
    assert "the prover ended 0 and the verifier left no accepted session on record (exit 5)" in p.stdout
    v = node.verifier()
    assert f"the verifier (process group {v['pid']}) was still running 0 s after its prover ended: stopped" in p.stdout
    assert gone(v["pid"])


def test_no_lease_ends_75_and_a_lease_is_asked_as_the_line_says(node):
    p = node.both("tiny", STUB_LEASE_RC="75", NOWAIT="1", GPU="0", MAXMIN="20")
    assert p.returncode == 75, said(p)
    lease, = [c for c in node.calls() if c.startswith("lease")]
    assert lease.startswith(f"lease {node.lease} 1 --on 0 --max-min 20 -- ")
    assert node.provers() == [] and gone(node.verifier()["pid"])


def test_a_served_line_declares_its_cores_to_a_lease_program_that_takes_them(node):
    """The cores at work while the lease is held: the prover's and, under ROLE=both, its verifier's, which holds no
    lease of its own; under ROLE=prover the verifier is another line's, and every core of this one is the prover's."""
    verifier, prover = (",".join(map(str, c)) for c in cores())
    both = prover if verifier == prover else f"{prover},{verifier}"
    every = sorted(os.sched_getaffinity(0))
    program(node.lease, LEASE_CORES)
    p = node.both("tiny", NOWAIT="1", GPU="0")
    assert p.returncode == 0, said(p)
    assert "names no --cores" not in p.stdout
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = str(server.getsockname()[1])
        (node.stub / "busy").write_text(port + "\n")
        p = node.run(ROLE="prover", SERVE="row", ROW=ROW, ROW_FILE=str(node.row), PORT=port)
        assert p.returncode == 0, said(p)
    first, second = (c for c in node.calls() if c.startswith("lease"))
    assert first.startswith(f"lease {node.lease} 1 --on 0 --max-min 30 --cores {both} -- taskset -c {prover} ")
    alone = ",".join(map(str, every))
    assert second.startswith(f"lease {node.lease} 1 --wait --max-min 30 --cores {alone} -- taskset -c {alone} ")
    assert [x["cpus"] for x in node.provers() if x["what"] != "row build"] == [cores()[1], every]


def test_nowait_ends_75_for_a_card_with_a_process_on_it_before_a_build_and_before_the_lease(node):
    """The stub `nvidia-smi` answers its n-th question with the file apps.n: the first is asked before anything is
    built, the second before the lease."""
    (node.stub / "apps.0").write_text("4242\n")
    p = node.both("row", NOWAIT="1", GPU="0")
    assert p.returncode == 75 and "run.sh: NOWAIT=1: GPU 0 is not free: 1 compute process(es) on it" in p.stdout, said(p)
    assert node.calls() == [] and node.scratch() == []
    assert (node.stub / "smi.argv").read_text().split() == ["-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"]

    (node.stub / "apps.0").unlink()
    (node.stub / "smi.asked").unlink()
    (node.stub / "apps.1").write_text("4242\n4243\n")
    p = node.both("row", NOWAIT="1", GPU="0")
    assert p.returncode == 75 and "GPU 0 is not free: 2 compute process(es) on it" in p.stdout, said(p)
    assert "remote session verdict: the prover ended 75 with no session on record (exit 75)" in p.stdout
    assert [c.split()[0] for c in node.calls()] == ["nvcc", "row", "verifier"] and gone(node.verifier()["pid"])

    (node.stub / "apps.1").unlink()
    (node.stub / "smi.asked").unlink()
    assert node.both("row", NOWAIT="1", GPU="0").returncode == 0
    assert (node.stub / "smi.asked").read_text().split() == ["2"]
    (node.stub / "smi.asked").unlink()
    assert node.both("row", GPU="0").returncode == 0 and not (node.stub / "smi.asked").exists()     # it may queue


# ------------------------------------------------------------------------------- run.sh: refused before the lease

def refused(node: Node, p: subprocess.CompletedProcess, why: str) -> None:
    assert p.returncode == 3, said(p)
    assert why in p.stdout, said(p)
    assert node.calls() == [], "nothing ran: no Build, no verifier, no lease"
    assert node.scratch() == []


def test_windows_that_do_not_fit_the_rows_run_are_refused_before_the_lease(node):
    p = node.both("row", WINDOWS="3", WINDOW_MS="11000", PASS_MS="580")
    refused(node, p, "run.sh: refused: 3 windows of 11000 ms take up to 58 engine steps at a pass every 580 ms (PASS_MS), "
                     "and the run is 32 steps")
    assert "WINDOW_MS under 5800 fits" in p.stdout
    assert node.both("row", WINDOWS="3", WINDOW_MS="3000", PASS_MS="580").returncode == 0     # as accepted on an H200
    assert "--windows 3 --window-ms 3000 " in " ".join(node.verifier()["argv"])


def test_under_role_prover_the_windows_are_checked_only_when_the_line_states_both(node, tmp_path):
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = str(server.getsockname()[1])
        (node.stub / "busy").write_text(port + "\n")
        line = {"ROLE": "prover", "SERVE": "row", "ROW": ROW, "ROW_FILE": str(node.row), "PORT": port}
        refused(node, node.run(**line, WINDOWS="3", WINDOW_MS="11000", PASS_MS="580"), "and the run is 32 steps")
        p = node.run(**line, WINDOW_MS="11000", PASS_MS="580")        # the verifier's line holds the windows
        assert p.returncode == 0, said(p)
        assert "remote session verdict: session accepted (exit 0)" in p.stdout
        assert "verifier" not in node.calls()


@pytest.mark.parametrize("change, why", [
    ({"ROW_FILE": None, "STUB_ROW_FILE": "/no/such/row.json"}, "refused: the row file /no/such/row.json can't be read"),
    ({"ROW_FILE": None}, f"refused: row {ROW}: its file can't be found: UsageError: the row has a file in two directories"),
    ({"ROW": None}, "refused: SERVE=row takes ROW, a row id"),
    ({"PLAN": "honest"}, "refused: PLAN=honest is not a plan of a served session: SERVE=row takes serve"),
    ({"PLAN": "servedrop:1"}, "refused: PLAN=servedrop:1 is not a plan of a served session: SERVE=row takes serve"),
    ({"SESSIONS": "2"}, "refused: SESSIONS=2: a served store serves one session"),
    ({"STAGES": "build,match"}, "refused: STAGES=build,match holds no commit"),
    ({"SERVE": "rows"}, "refused: SERVE is tiny, row or node, not rows"),
    ({"KV_LIB": "/no/such/kv.so"}, "refused: KV_LIB=/no/such/kv.so is not build, cpu or a built kv.cu"),
    ({"LEAN_EXE": "/no/such/verifier"}, "refused: LEAN_EXE=/no/such/verifier is not an executable"),
    ({"VLLM_PY": "/no/such/python"}, "refused: VLLM_PY=/no/such/python is not an executable"),
    ({"GPU_LEASE": "/no/such/gpu-lease"}, "refused: GPU_LEASE=/no/such/gpu-lease is not a program"),
])
def test_a_line_that_cannot_run_is_refused_with_the_reason(node, change, why):
    refused(node, node.both("row", **change), why)


def test_a_row_whose_target_does_not_declare_synchronous_scheduling_is_refused(node):
    row_file(node.row, target={"ir": "boolean"})
    refused(node, node.both("row"), "target declares async_scheduling None")


def test_kv_cu_is_built_for_the_card_the_rows_target_declares_or_the_one_stated(node):
    row_file(node.row, target={"async_scheduling": False})
    refused(node, node.both("row"), "the row's target declares no compute_capability to build kv.cu for: state KV_ARCH, or "
                                    "KV_LIB=cpu")
    refused(node, node.both("row", KV_ARCH="h200"), "refused: KV_ARCH=h200 is no card's architecture (sm_90)")
    assert node.both("row", KV_ARCH="sm_120a").returncode == 0
    assert node.calls()[0].startswith("nvcc -O3 -arch=sm_120a ")


def test_tiny_takes_serve_or_servedrop(node):
    refused(node, node.both("tiny", PLAN="honest"), "SERVE=tiny takes serve or servedrop:K")
    refused(node, node.both("tiny", PLAN="servedrop:x"), "servedrop:K withholds the first K blocks appended")
    p = node.both("tiny", PLAN="servedrop:1", STUB_PROVER="rejected")
    assert p.returncode == 6 and node.provers()[0]["argv"][-3] == "servedrop:1"


def test_role_prover_takes_its_port_and_picks_none(node):
    """A prover with no PORT used to pick one and wait for a verifier on it; it is refused at once, served or not."""
    refused(node, node.run(ROLE="prover", SERVE="tiny"), "refused: ROLE=prover takes PORT")
    refused(node, node.run(ROLE="prover", PY=str(node.ws / "venv312/bin/python")), "refused: ROLE=prover takes PORT")
    p = node.run(ROLE="prover", SERVE="tiny", PORT="7711")
    assert p.returncode == 3 and "no verifier tunnel on 127.0.0.1:7711" in p.stdout and node.calls() == []


# ------------------------------------------------------------------------------------------------ run.sh: the defaults

def test_role_both_picks_a_port_nothing_listens_on(node):
    (node.stub / "busy").write_text("".join(f"{port}\n" for port in range(7700, 7900) if port != 7777))
    p = node.both("tiny", PORT=None, STUB_VERIFIER="deaf", STUB_PROVER="quiet", VERIFIER_WAIT_S="0")
    assert p.returncode == 5, said(p)
    assert "--port 7777 " in " ".join(node.verifier()["argv"])
    assert node.provers()[0]["argv"][2] == "127.0.0.1:7777"
    (node.stub / "busy").write_text("".join(f"{port}\n" for port in range(7700, 7900)))
    (node.stub / "calls").unlink()
    refused(node, node.both("tiny", PORT=None), "refused: no free port from 7700 to 7899")


def test_the_vllm_venv_is_the_first_there(tmp_path):
    node = Node(tmp_path, venv="jobs/venv312")
    node.venv("venv312")
    assert node.both("tiny").returncode == 0
    assert node.provers()[0]["env"]["STUB_INTERPRETER"] == str(node.ws / "jobs/venv312/bin/python")
    empty = Node(tmp_path / "empty", venv="elsewhere/venv")
    refused(empty, empty.both("tiny"),
            f"refused: no vLLM venv: VLLM_PY is not set, and neither {empty.ws}/jobs/venv312/bin/python nor "
            f"{empty.ws}/venv312/bin/python is there")
    assert empty.both("tiny", VLLM_PY=str(empty.ws / "elsewhere/venv/bin/python")).returncode == 0


@pytest.mark.parametrize("toolkits, torch, home", [
    (("cuda-12.9", "cuda-13.0", "cuda"), "12.9", "cuda-12.9"),     # the toolkit of the CUDA the prover's torch is built for
    (("cuda-13.0", "cuda"), "12.9", "cuda-13.0"),
    (("cuda-12.9", "cuda"), None, "cuda"),                         # a venv with no torch to ask
    (("cuda-12.9",), "13.0", None),
])
def test_cuda_home_is_a_toolkit_the_node_has(tmp_path, toolkits, torch, home):
    node = Node(tmp_path, toolkits=toolkits)
    p = node.both("row", STUB_TORCH_CUDA=torch)
    if home is None:
        return refused(node, p, f"refused: no CUDA toolkit: CUDA_HOME is not set, and none of {tmp_path}/cuda/cuda-13.0 "
                                f"{tmp_path}/cuda/cuda is a directory")
    assert p.returncode == 0, said(p)
    build, commit = node.provers()
    assert build["env"]["CUDA_HOME"] == commit["env"]["CUDA_HOME"] == f"{tmp_path}/cuda/{home}"
    assert commit["argv"][commit["argv"].index("--cuda-bin") + 1] == f"{tmp_path}/cuda/{home}/bin"


def test_the_lease_program_is_the_one_named_then_the_paths_then_the_nodes(node):
    assert node.both("tiny").returncode == 0                       # GPU_LEASE names it
    on_path = program(node.bin / "gpu-lease", LEASE)
    installed = program(node.ws / "pouw/infra/bin/gpu-lease", LEASE)
    assert node.both("tiny", GPU_LEASE=None).returncode == 0       # else the one on the PATH
    on_path.unlink()
    assert node.both("tiny", GPU_LEASE=None).returncode == 0       # else the one installed on the node
    installed.unlink()
    program(node.bin / "bash", LEASE_BASH)                         # a `bash` that runs what a lease script is handed
    assert node.both("tiny", GPU_LEASE=None).returncode == 0       # else the tree's script
    asked = [c for c in node.calls() if c.startswith("lease")]
    leases = [c.split()[1:3] for c in asked]
    assert [x[0] for x in leases[:3]] == [str(node.lease), str(on_path), str(installed)]
    assert leases[3] == ["bash", str(ROOT / TREE_LEASE)] and (ROOT / TREE_LEASE).is_file()
    assert [" --cores " in c for c in asked] == [False, False, False, True]     # the tree's script takes them


# ------------------------------------------------------------------------------------ run.sh: a row around its lease

def test_a_failed_build_or_kv_build_ends_before_the_lease_with_its_own_code(node):
    p = node.both("row", STUB_BUILD_RC="10")
    assert p.returncode == 10 and "the row's Build ended 10 with no GPU visible" in p.stdout, said(p)
    assert [c.split()[0] for c in node.calls()] == ["nvcc", "row"]
    (node.stub / "calls").unlink()
    p = node.both("row", STUB_NVCC_RC="1")
    assert p.returncode == 4 and "kv.cu's build for sm_90 failed (KV_LIB=cpu encodes on the CPU)" in p.stdout, said(p)
    assert [c.split()[0] for c in node.calls()] == ["nvcc"] and node.scratch() == []


def test_a_failed_lean_build_ends_4_before_the_lease(node):
    p = node.both("tiny", LEAN_EXE=None)
    assert p.returncode == 4 and "the Lean verifier's build failed" in p.stdout, said(p)
    assert node.calls() == ["lean-build"]
    (node.stub / "calls").unlink()
    assert node.both("tiny", LEAN_EXE=None, STUB_LEAN=str(node.lean)).returncode == 0
    assert node.calls()[:2] == ["lean-build", "verifier"] and f"--lean-exe {node.lean}" in " ".join(node.verifier()["argv"])


def test_a_verifier_that_never_listens_ends_the_line_before_the_lease(node):
    p = node.both("tiny", STUB_VERIFIER="dead")
    assert p.returncode == 3 and "refused: the verifier ended 2 before it listened" in p.stdout, said(p)
    assert node.calls() == ["verifier"] and gone(node.verifier()["pid"])


def test_the_rows_knobs_reach_its_build_and_its_commit(node, tmp_path):
    kv = tmp_path / "kept-kv.so"
    kv.write_text("")
    p = node.both("row", KV_LIB=str(kv), WEIGHT_BLOCKS="512", KV_PAGES="40", EXTRA_COMMIT_ARGS="--remote-filler-gib 1",
                  BUILD_GPU="1", STAGES="build,match,commit", SWEEP_DIR=str(tmp_path / "sweep"), CONFIG_RUN="0",
                  VERIFIER_CPUS=str(cores()[1][0]), PROVER_CPUS=str(cores()[1][0]), OMP_NUM_THREADS="3")
    assert p.returncode == 0, said(p)
    (commit,) = node.provers()                                  # BUILD_GPU: every stage under the lease, none before
    assert commit["what"] == "row build,match,commit" and [c.split()[0] for c in node.calls()] == ["verifier", "lease", "row"]
    assert commit["env"]["EXTRA_COMMIT_ARGS"].split() == [
        "--remote-plan", "serve", "--remote-kv-pages", "40", "--remote-weight-blocks", "512", "--remote-kv-lib", str(kv),
        "--remote-filler-gib", "1"]
    assert commit["argv"][commit["argv"].index("--sweep-dir") + 1] == str(tmp_path / "sweep")
    assert commit["env"]["CONFIG_RUN"] == "0" and commit["env"]["OMP_NUM_THREADS"] == "3" and commit["env"]["MKL_NUM_THREADS"] == "1"
    assert commit["cpus"] == node.verifier()["cpus"] == [cores()[1][0]]
    assert (tmp_path / "sweep" / ROW / "commit/remote_session.json").is_file()
    (node.stub / "calls").unlink()
    (node.stub / "prover.jsonl").unlink()
    assert node.both("row", KV_LIB="cpu").returncode == 0      # the reference on the CPU: no build, no library
    assert node.calls()[0] == "row build" and [c.split()[0] for c in node.calls()] == ["row", "verifier", "lease", "row"]
    assert "--remote-kv-lib" not in node.provers()[1]["env"]["EXTRA_COMMIT_ARGS"]


# ---------------------------------------------------------------------------- run.sh: ROLE=prover's verifier is told

@pytest.mark.parametrize("keep", [False, True])
def test_a_red_prover_under_role_prover_ends_a_verifier_still_waiting_for_it(node, keep):
    """The verifier here is this test's listening socket, as a tunnel's end would be: a prover that ends red without
    ever connecting opens and drops one connection to it, which a verifier in accept() ends at; KEEP_VERIFIER leaves
    it listening."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = str(server.getsockname()[1])
        (node.stub / "busy").write_text(port + "\n")
        p = node.run(ROLE="prover", SERVE="tiny", PORT=port, STUB_PROVER="crash", KEEP_VERIFIER="1" if keep else None)
        assert p.returncode == 1 and "the prover ended 1 with no session on record (exit 1)" in p.stdout, said(p)
        server.setblocking(False)
        if keep:
            with pytest.raises(BlockingIOError):
                server.accept()
        else:
            conn, _ = server.accept()
            conn.setblocking(True)
            assert conn.recv(16) == b""
            conn.close()


def test_a_red_prover_under_role_prover_ends_the_verifier_itself(node, tmp_path):
    """The same against the verifier's own process, started as by hand with the Lean verifier built: the connection
    `run.sh` opens and drops is the one it accepts, and it ends red at the closed stream rather than listening on.
    The verifier's output is read to its end, so one that listened on would hang here."""
    from verity.protocols.poub.verifier.session import lean_verifier as lv

    port = str(free_port())
    line = [sys.executable, str(ROOT / "verity/protocols/poub/verifier/remote_session/verifier_process.py"),
            "--host", "127.0.0.1", "--port", port, "--out", str(tmp_path / "verifier"), "--windows", "3",
            "--window-ms", "2000", "--samples", "2773", "--jitter-ms", "5000", "--sessions", "1", "--check-workers", "1",
            "--lean-exe", str(lv.build())]
    verifier = subprocess.Popen(line, cwd=ROOT, env=os.environ | {"HOME": str(tmp_path / "home")},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert any(x.startswith("verifier: listening") for x in iter(verifier.stdout.readline, ""))
        (node.stub / "busy").write_text(port + "\n")
        p = node.run(ROLE="prover", SERVE="tiny", PORT=port, STUB_PROVER="crash")
        assert p.returncode == 1 and "the prover ended 1 with no session on record (exit 1)" in p.stdout, said(p)
        rest = verifier.stdout.read()
        assert verifier.wait() != 0 and "the stream closed" in rest, rest[-3000:]
        assert not (tmp_path / "verifier" / "verifier.json").exists()       # no session, so no record of one
    finally:
        verifier.kill()
        verifier.wait()
