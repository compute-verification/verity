"""A leased Commit (`pipeline.gpu_lease`): only the Commit process holds the GPU, from right before its engine to the end of its last
run (or its exit, when it still needs the device), under the lease tool; the row and every other child see no CUDA device; a
single-GPU row's Match holds the lease for its GPU phase only; an unleased row is unchanged."""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline import row_driver as RD


def fake_smi(tmp_path: Path, looks: list[str]) -> Path:
    """A stand-in for nvidia-smi's compute-apps query in `tmp_path/bin`: each call lists the next of `looks` (UUIDs, comma-separated;
    "!" fails, as an nvidia-smi that can't answer; none left lists nothing) and notes in `smi.calls` whether 5.lock was held then."""
    (tmp_path / "looks").write_text("".join(f"{x}\n" for x in looks))
    (tmp_path / "bin").mkdir(exist_ok=True)
    p = tmp_path / "bin" / "nvidia-smi"
    p.write_text(f'#!/usr/bin/env bash\nflock -n "{tmp_path}/5.lock" true && s=free || s=held\necho "$s" >> "{tmp_path}/smi.calls"\n'
                 f'l=$(head -n 1 "{tmp_path}/looks"); sed -i 1d "{tmp_path}/looks"\n[ "$l" = "!" ] && exit 1\ntr , "\\n" <<< "$l"\n')
    p.chmod(0o755)
    return tmp_path / "bin"


def fake_lease(tmp_path: Path, gpu: str = "5") -> str:
    """A stand-in for gpu-lease: records its arguments, locks <g>.lock for each GPU g of ``gpu`` (comma-separated) on a descriptor CMD
    inherits, sets CUDA_VISIBLE_DEVICES and runs CMD as its child with its stdin, as gpu-lease does, with `fake_smi` (listing nothing)
    as its nvidia-smi."""
    if not (tmp_path / "bin" / "nvidia-smi").exists():
        fake_smi(tmp_path, [])
    locks = "".join(f'exec {{fd}}>>"{tmp_path}/{g}.lock"; flock -x $fd\n' for g in gpu.split(","))
    p = tmp_path / "gpu-lease"
    p.write_text(f'#!/usr/bin/env bash\necho "$*" > "{tmp_path}/lease.args"\nwhile [ "$1" != -- ]; do shift; done; shift\n'
                 f'{locks}export PATH="{tmp_path}/bin:$PATH"\nexport CUDA_VISIBLE_DEVICES={gpu} CUDA_DEVICE_ORDER=PCI_BUS_ID '
                 f'GPU_LEASE_UUID={",".join("GPU-" + g for g in gpu.split(","))}\n"$@" 0<&0 &\nwait $!\n')
    p.chmod(0o755)
    return str(p)


def commit(script: str) -> list[str]:
    """A Commit stand-in: `script` after the CLI's environment and `GL.acquire()`, with `before` the devices it saw first."""
    return [sys.executable, "-c", "import json, os, sys\nfrom verity_vllm.pipeline import cli, gpu_lease as GL\n"
            "before = os.environ['CUDA_VISIBLE_DEVICES']\ncli.process_settings(os.environ)\nGL.acquire()\n" + script]


def released(path: Path) -> bool:
    """Waits for the lock on `path` to be free (the holder ends on the Commit's exit)."""
    return subprocess.run(["flock", str(path), "true"]).returncode == 0


def test_only_an_item_with_a_lease_tool_is_leased_and_its_devices_are_hidden_outside_the_lease():
    assert not GL.leased({}) and not GL.leased({"GPU_LEASE": ""}) and GL.leased({"GPU_LEASE": "/x/gpu-lease"})
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    GL.hide_devices(env)
    assert env == {"CUDA_VISIBLE_DEVICES": ""}
    env = {"GPU_LEASE": "/x/gpu-lease", "CUDA_VISIBLE_DEVICES": ""}
    GL.acquire(environ=env)                            # not started by a leased row's Commit stage: nothing taken
    assert env == {"GPU_LEASE": "/x/gpu-lease", "CUDA_VISIBLE_DEVICES": ""}


def test_the_commit_requests_its_tasks_gpus_for_at_most_its_timeout():
    env = {"GPU_LEASE": "/l/gpu-lease", GL.STARTED: "/row/gpu_lease.started", GL.MAX_MIN: str(GL.max_min(2400))}
    argv = GL.holder_argv(env, 7)
    assert argv[:9] == ["/l/gpu-lease", "1", "--wait", "--max-min", "40", "--mem-gb", "0", "--", sys.executable]
    assert argv[-4:] == ["/row/gpu_lease.started", "7", str(GL.EMPTY_LOOKS), str(GL.EMPTY_EVERY_S)]
    assert GL.max_min(61) == 2 and GL.max_min(1) == 1
    assert GL.holder_argv({**env, "GPU_LEASE_N": "2"}, 7)[:2] == ["/l/gpu-lease", "2"]
    assert GL.STARTED != GL.TOOL_START and GL.tool_env({**env, GL.TOOL_START: "/row/x"}) == env    # the tool's own start override


def test_the_commit_takes_the_lease_itself_holds_its_lock_and_sees_only_the_leased_gpu(tmp_path):
    env = {"PATH": os.environ["PATH"], "GPU_LEASE": fake_lease(tmp_path)}
    GL.hide_devices(env)
    seen = tmp_path / "seen"
    cmd = commit(f"fds = [f for f in os.listdir('/proc/self/fd') if os.path.realpath('/proc/self/fd/' + f) == {str(tmp_path / '5.lock')!r}]\n"
                 f"json.dump(dict(before=before, after=os.environ['CUDA_VISIBLE_DEVICES'], uuid=os.environ['GPU_LEASE_UUID'], locks=len(fds)), "
                 f"open({str(seen)!r}, 'w'))\n")
    log = tmp_path / "commit.log"
    timeouts = []

    def run_tl(to: int, argv: list[str]) -> int:
        timeouts.append(to)
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    rc, rec = GL.run(str(tmp_path), str(log), cmd, 2400, env, run_tl, clock=lambda: 0.0)
    assert rc == 0, log.read_text()
    assert json.loads(seen.read_text()) == {"before": "", "after": "5", "uuid": "GPU-5", "locks": 1}
    assert env == {"PATH": os.environ["PATH"], "GPU_LEASE": env["GPU_LEASE"], "CUDA_VISIBLE_DEVICES": ""}
    assert timeouts == [4800] and (tmp_path / "lease.args").read_text().startswith(f"1 --wait --max-min 40 --mem-gb 0 -- {sys.executable} -c ")
    stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
    assert re.fullmatch(stamp, rec["started"]) and re.fullmatch(stamp, rec["requested"]) and rec["held_min"] is not None
    assert json.loads((tmp_path / GL.RECORD).read_text()) == rec and rec["max_min"] == 40 and rec["rc"] == 0
    assert released(tmp_path / "5.lock")


def test_the_lease_tool_ending_first_ends_the_commit(tmp_path):
    env = {"PATH": os.environ["PATH"], "GPU_LEASE": fake_lease(tmp_path), GL.STARTED: str(tmp_path / "gpu_lease.started"), GL.MAX_MIN: "1"}
    GL.hide_devices(env)
    r = subprocess.run(commit("GL._HELD[0][0].kill()\nimport threading\nthreading.Event().wait()\n"), env=env, capture_output=True, text=True)
    assert r.returncode == -9 and "GPU lease ended under the Commit" in r.stderr, r.stderr


def _holder(tmp_path: Path, looks: list[str], n: int) -> tuple[int, str, list[str]]:
    """The holder as the lease tool runs it (a lock descriptor inherited, GPU-5 leased), told that the Commit let go (EOF)."""
    path = fake_smi(tmp_path, looks)
    ours, theirs = socket.socketpair()
    with open(tmp_path / "5.lock", "a") as lock:
        p = subprocess.Popen([sys.executable, "-c", GL._HOLDER, str(tmp_path / "gpu_lease.started"), str(theirs.fileno()), str(n), "0"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             pass_fds=(theirs.fileno(), lock.fileno()), env={"PATH": f"{path}:{os.environ['PATH']}", "GPU_LEASE_UUID": "GPU-5"})
    theirs.close()
    for fd in socket.recv_fds(ours, 1, 8)[1]:
        os.close(fd)
    _, err = p.communicate(b"")
    return p.returncode, err.decode(), (tmp_path / "smi.calls").read_text().split()


@pytest.mark.parametrize("looks,n,calls,said", [
    ([], 240, 1, None),                                                    # a release: nothing is on the GPU at the first look
    (["GPU-5", "GPU-3,GPU-5", "GPU-3"], 240, 3, "(2 look(s) occupied); the lease ends"),     # another GPU's process doesn't count
    (["!", ""], 240, 2, "(1 look(s) occupied)"),                           # an nvidia-smi that can't answer counts as occupied
    (["GPU-5"] * 5, 3, 3, "nvidia-smi still lists a compute process on the leased GPU"),
])
def test_the_holder_ends_once_nothing_is_left_on_the_leased_gpu(tmp_path, looks, n, calls, said):
    rc, err, seen = _holder(tmp_path, looks, n)
    assert rc == 0 and len(seen) == calls and (said in err if said else err == ""), err


def test_a_commit_that_exits_holding_its_lease_keeps_it_until_its_gpu_is_empty(tmp_path):
    """A Commit's exit closes the lease's descriptors before the driver has freed its context (cov-gm388: nvidia-smi listed it on its
    GPU for about 8 s after the lease ended, and the node's stray probe blocked the pool): the lease ends only once nothing is left."""
    env = {"PATH": os.environ["PATH"], "GPU_LEASE": fake_lease(tmp_path)}
    fake_smi(tmp_path, ["GPU-5", "GPU-5", ""])
    GL.hide_devices(env)
    log = tmp_path / "commit.log"

    def run_tl(to: int, argv: list[str]) -> int:
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    rc, rec = GL.run(str(tmp_path), str(log), commit("pass\n"), 600, env, run_tl)
    assert rc == 0 and rec["released_at"] == "process.end" and released(tmp_path / "5.lock")
    assert (tmp_path / "smi.calls").read_text().split() == ["held", "held", "held"]
    assert "nothing is left on the leased GPU" in log.read_text() and "(2 look(s) occupied)" in log.read_text()


@pytest.mark.parametrize("looks,n,waited,left", [
    ([[]], 3, 0, []),                                                     # nothing else on the leased GPUs
    ([["GPU-5"], ["GPU-5"], []], 3, 2, []),                               # a cancelled Commit's engine workers leave after two looks
    ([None], 3, 0, []),                                                   # nvidia-smi can't say: the engine's own check stands
    ([["GPU-5"]] * 5, 3, 3, ["GPU-5"]),                                   # still there after the last look
])
def test_a_new_lease_waits_for_another_process_to_leave_its_gpus(monkeypatch, looks, n, waited, left):
    seq, slept = iter(looks), []
    monkeypatch.setattr(GL, "on_gpus", lambda uuids: next(seq))
    assert GL._foreign(["GPU-5"], slept.append, n) == (waited, left) and slept == [GL.CLEAR_EVERY_S] * waited


def test_a_commit_whose_leased_gpu_stays_occupied_ends_by_name_and_gives_the_lease_back(tmp_path):
    """A cancelled Commit's engine workers can outlive its lease, so the next Commit's engine can find its GPUs full."""
    fake_smi(tmp_path, ["GPU-5"] * 4)
    env = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "GPU_LEASE": fake_lease(tmp_path)}
    GL.hide_devices(env)
    cmd = commit("raise AssertionError('acquired')\n")
    cmd[2] = cmd[2].replace("GL.acquire()", "GL.CLEAR_LOOKS, GL.CLEAR_EVERY_S = 2, 0\nGL.acquire()")
    log = tmp_path / "commit.log"

    def run_tl(to: int, argv: list[str]) -> int:
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    rc, _ = GL.run(str(tmp_path), str(log), cmd, 600, env, run_tl)
    assert rc == 1 and "still lists another process on the leased GPU(s) ['GPU-5'] after 2 look(s)" in log.read_text(), log.read_text()
    assert "acquired" not in log.read_text() and released(tmp_path / "5.lock")


def test_a_commit_that_touched_cuda_before_its_lease_takes_none(tmp_path, monkeypatch):
    env = {"GPU_LEASE": "/l/missing-gpu-lease", GL.STARTED: str(tmp_path / "gpu_lease.started"), "CUDA_VISIBLE_DEVICES": ""}
    monkeypatch.setattr(GL, "cuda_initialised", lambda: True)
    try:
        GL.acquire(environ=env)
        raise AssertionError("acquired")
    except SystemExit as e:
        assert "CUDA was initialised before the lease" in str(e)
    assert not (tmp_path / "gpu_lease.requested").exists() and env["CUDA_VISIBLE_DEVICES"] == ""


def test_a_commit_that_never_got_its_gpu_records_no_start(tmp_path):
    rc, rec = GL.run(str(tmp_path), str(tmp_path / "commit.log"), ["true"], 60, {"GPU_LEASE": "/l/gpu-lease"},
                     lambda to, argv: 75, clock=lambda: 0.0)
    assert rc == 75 and rec["started"] is None and rec["requested"] is None and rec["held_min"] is None and rec["waited_min"] is None


@pytest.mark.parametrize("item,seen", [({}, "device"), ({GL.WEIGHTS_HASH: "host"}, "host"), ({GL.WEIGHTS_HASH: "device-checked"}, "device-checked")])
def test_a_leased_commit_hashes_its_weights_on_its_gpu_unless_the_item_says_how(tmp_path, item, seen):
    env, got = {"GPU_LEASE": "/l/gpu-lease", **item}, []
    GL.run(str(tmp_path), str(tmp_path / "commit.log"), ["true"], 60, env, lambda to, argv: got.append(env.get(GL.WEIGHTS_HASH)) or 0,
           clock=lambda: 0.0)
    assert got == [seen] and env == {"GPU_LEASE": "/l/gpu-lease", **item}


class _Ticks(threading.Event):
    """An event whose wait returns `n` times without being set (one heartbeat each), then as if set."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def wait(self, timeout: float | None = None) -> bool:
        self.n -= 1
        return self.n < 0


def test_a_queued_commit_writes_a_heartbeat_to_its_log_until_the_gpu_is_held(tmp_path):
    log, started, requested = tmp_path / "commit.log", tmp_path / "gpu_lease.started", tmp_path / "gpu_lease.requested"
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert not log.exists()                            # the Commit's CPU work before its request is not a wait
    requested.write_text("2026-10-01T16:00:00Z\n")
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert log.read_text().count("waiting for a GPU lease") == 3
    started.write_text("2026-10-01T16:00:00Z\n")
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert log.read_text().count("waiting for a GPU lease") == 3


def test_the_cuda_jit_cache_persists_beside_the_trees_triton_cache(tmp_path):
    env = {"TRITON_CACHE_DIR": str(tmp_path / "cache" / "triton" / "tree1")}
    RD.jit_cache(env)
    assert env["CUDA_CACHE_PATH"] == str(tmp_path / "cache" / "cuda" / "tree1") and Path(env["CUDA_CACHE_PATH"]).is_dir()
    assert env["CUDA_CACHE_MAXSIZE"] == str(4 << 30)
    kept = {"TRITON_CACHE_DIR": "/t/triton/x", "CUDA_CACHE_PATH": "/mine"}
    RD.jit_cache(kept)
    assert kept == {"TRITON_CACHE_DIR": "/t/triton/x", "CUDA_CACHE_PATH": "/mine"}
    none: dict[str, str] = {}
    RD.jit_cache(none)
    assert none == {}


PRO = "pleias-350m__bf16__rtxpro6000__tp1__b16__i256__o32__mixed__greedy__bi-eager"


def _toolchain(tmp_path: Path) -> tuple[str, str, Path]:
    """A venv with triton, torch and vllm, a CUDA toolkit, and `job_tree.sh`'s copy of a tree with one Triton kernel -> (the venv's
    python, CUDA_HOME, the copy's integrations/vllm)."""
    for d in ("triton-3.7.1", "torch-2.13.0+cu129", "vllm-0.28.1"):
        info = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / f"{d}.dist-info"
        info.mkdir(parents=True)
        (info / "RECORD").write_text(f"{d.split('-')[0]}/__init__.py,sha256=x,1\n")
    (tmp_path / "cuda" / "include").mkdir(parents=True)
    (tmp_path / "cuda" / "include" / "cuda.h").write_text("#define CUDA_VERSION 12090\n")
    tree = tmp_path / "src" / "0123456789abcdef" / "integrations" / "vllm"
    (tree / "verity_vllm" / "acquire").mkdir(parents=True)
    (tree / "verity_vllm" / "acquire" / "tap.py").write_text("import triton\n\n\n@triton.jit\ndef k(x):\n    pass\n")
    (tree / "verity_vllm" / "row.py").write_text("STAGES = 1\n")
    return str(tmp_path / "venv" / "bin" / "python"), str(tmp_path / "cuda"), tree


def test_the_jit_key_is_the_toolchain_the_device_class_and_the_trees_triton_kernels(tmp_path):
    py, cuda, a = _toolchain(tmp_path)
    k = RD.jit_key(py, cuda, PRO, str(a))
    assert re.fullmatch(r"[0-9a-f]{16}", k)
    b = tmp_path / "src" / "fedcba9876543210" / "integrations" / "vllm"
    shutil.copytree(a, b)
    (b / "verity_vllm" / "row.py").write_text("STAGES = 2\n")
    assert RD.jit_key(py, cuda, PRO, str(b)) == k                                  # another sync, its change outside the kernels
    assert RD.jit_key(py, cuda, PRO.replace("rtxpro6000", "l40s"), str(a)) not in (k, None)
    (b / "verity_vllm" / "acquire" / "tap.py").write_text("import triton\n\n\n@triton.jit\ndef k(x):\n    return x\n")
    assert RD.jit_key(py, cuda, PRO, str(b)) not in (k, None)
    (tmp_path / "cuda" / "include" / "cuda.h").write_text("#define CUDA_VERSION 13000\n")
    assert RD.jit_key(py, cuda, PRO, str(a)) not in (k, None)
    rec = next((tmp_path / "venv").rglob("torch-*.dist-info/RECORD"))
    rec.write_text("torch/__init__.py,sha256=y,1\n")
    k2 = RD.jit_key(py, cuda, PRO, str(a))
    assert k2 not in (k, None)
    assert RD.jit_key(py, cuda, "r", str(a)) is None                               # no device class, no key
    rec.unlink()
    assert RD.jit_key(py, cuda, PRO, str(a)) is None


def test_a_bi_eager_rows_jit_caches_move_from_its_job_tree_to_its_key_and_any_other_directory_is_kept(tmp_path):
    tree, c, k = str(tmp_path / "src" / "0123456789abcdef" / "integrations" / "vllm"), tmp_path / "cache", "00ff00ff00ff00ff"
    env = {"TRITON_CACHE_DIR": str(c / "triton" / "0123456789abcdef"), "VLLM_CACHE_ROOT": str(c / "vllm" / "0123456789abcdef")}
    RD.jit_cache(env, tree, lambda: k)
    assert env == {"TRITON_CACHE_DIR": str(c / "triton" / f"k-{k}"), "VLLM_CACHE_ROOT": str(c / "vllm" / f"k-{k}"),
                   "CUDA_CACHE_PATH": str(c / "cuda" / f"k-{k}"), "CUDA_CACHE_MAXSIZE": str(4 << 30)}
    own = {"TRITON_CACHE_DIR": str(c / "triton" / "0123456789abcdef")}
    RD.jit_cache(own, tree, lambda: None)                                           # a compiled row, or an input that can't be read
    assert own["TRITON_CACHE_DIR"] == str(c / "triton" / "0123456789abcdef") and own["CUDA_CACHE_PATH"] == str(c / "cuda" / "0123456789abcdef")

    def never() -> str:
        raise AssertionError("a key for a directory that isn't the job tree's")
    scratch = {"TRITON_CACHE_DIR": str(tmp_path / "scratch" / "rk" / "triton")}    # `research run --scratch triton`: cold on purpose
    RD.jit_cache(scratch, tree, never)
    assert scratch["TRITON_CACHE_DIR"] == str(tmp_path / "scratch" / "rk" / "triton")
    checkout = {"TRITON_CACHE_DIR": str(c / "triton" / "verity")}                 # a checkout, not `job_tree.sh`'s copy
    RD.jit_cache(checkout, str(tmp_path / "verity" / "integrations" / "vllm"), never)
    assert checkout["TRITON_CACHE_DIR"] == str(c / "triton" / "verity")


@pytest.mark.parametrize("row,workload,eager", [
    (PRO, {"sweep": {"execution": "bi-eager", "row_id": PRO}}, True),
    (PRO, {"sweep": {"row_id": PRO}}, True),
    (PRO, {"execution": "bi-compiled", "sweep": {"row_id": PRO}}, False),
    (PRO.replace("bi-eager", "bi-compiled"), {"sweep": {"row_id": PRO.replace("bi-eager", "bi-compiled")}}, False),
    (PRO.replace("bi-eager", "bi-eager-fa2"), {"sweep": {"execution": "bi-eager-fa2"}}, False),
    (PRO, None, False),
])
def test_only_a_bi_eager_row_shares_jit_caches(tmp_path, row, workload, eager):
    from verity_vllm.pipeline import cli
    o = cli.parse("row", ["run", row, "R", "o/r", "x", "--sweep-dir", str(tmp_path), "--py", sys.executable], {})
    r = RD.Row(row=row, role="R", repo="o/r", rev="x", stages="build", env=o.env, proc="row_pod")
    r.tree = str(tmp_path)
    if workload is not None:
        (tmp_path / "workloads").mkdir()
        (tmp_path / "workloads" / f"{row}.json").write_text(json.dumps(workload))
    assert r.bi_eager() is eager


def _args(**kw):
    return SimpleNamespace(**{"replay_deferred": True, "replay_bundle_overlap": 1, "open_after_release": True, "replay_openings": "auto",
                              "tensor_digests": False, **kw})


def _com(**kw):
    return SimpleNamespace(**{"retain": "host", "retain_exclude": None, "pool": SimpleNamespace(mode="exact-registered-shared"),
                              "_retained_device_bytes": lambda: 0, **kw})


ENG = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(engine_core=object()))))
ONE = {"CUDA_VISIBLE_DEVICES": "5", "GPU_LEASE_UUID": "GPU-5"}


@pytest.mark.parametrize("args,com,eng,env,why,bundle", [
    (_args(replay_deferred=False), _com(), ENG, ONE, "no --replay-deferred", False),
    (_args(replay_bundle_overlap=0), _com(), ENG, ONE, "--replay-bundle-overlap 0", False),
    (_args(), _com(), ENG, ONE, "replay_bundle.start began no write", False),
    (_args(serving_rows="request.json"), _com(), ENG, ONE, "--serving-rows commits its window at the C2 site", True),
    (_args(tensor_digests=True), _com(), ENG, ONE, "--tensor-digests", True),
    (_args(), _com(retain="none"), ENG, ONE, "replay the run on the engine", True),
    (_args(), _com(retain_exclude={"fa3_hidden_m1"}), ENG, ONE, "replay the run on the engine", True),
    (_args(replay_openings="always"), _com(), ENG, ONE, "replay the run on the engine", True),
    (_args(), _com(pool=SimpleNamespace(mode="torch-pinned-pow2")), ENG, ONE, "torch's pinned host memory", True),
    (_args(), _com(_retained_device_bytes=lambda: 3 << 20), ENG, ONE, "3 MiB of committed bytes", True),
    (_args(), _com(), SimpleNamespace(llm=None), ONE, "engine core is not in this process", True),
    (_args(), _com(), ENG, {"CUDA_VISIBLE_DEVICES": "4,5"}, "a lease of 2 GPUs", True),
])
def test_a_commit_that_still_needs_the_device_after_its_last_run_keeps_its_lease_and_says_why(args, com, eng, env, why, bundle):
    assert why in GL.kept_reason(args, [_com(), com], eng, env, bundle=bundle)


def test_a_commit_whose_checks_after_its_last_run_read_host_memory_only_may_release():
    assert GL.kept_reason(_args(), [_com()], ENG, ONE, bundle=True) is None
    assert GL.kept_reason(_args(open_after_release=False), [_com(retain="none")], ENG, ONE, bundle=True) is None    # no openings after release
    assert GL.kept_reason(_args(), [_com(pool=SimpleNamespace(mode=None))], ENG, ONE, bundle=True) is None          # a pool that never allocated


def test_the_commit_releases_its_lease_after_its_last_run_and_before_the_checks_that_follow():
    """In `commit.main`: the release follows the last run's records that read the device (the run, its openings, the replay bundle's
    device-resident tensors in `RB.start`) and precedes every check that reads host memory; nothing after it calls CUDA. A hot
    worker's job never releases; a grouped Commit's last row does (`commit_group`)."""
    src = Path(GL.__file__).with_name("commit.py").read_text()
    main = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
    marks = ["GL.acquire(TL)", "TL.end(_wstage", '"gpu_mem"', 'TL.start("validate.openings"', "RB.start(out, pair", "GL.release(TL",
             'TL.start("validate.binding_coverage"', 'TL.start("validate.manifest_coverage"', "RB.defer(", "TL.start(\"validate.weights_pin\"",
             "openings_after_release(com", "for c in coms.values():\n        c.close()", 'TL.start("closure.summary")']
    at = [main.index(m) for m in marks]
    assert at == sorted(at) and main.count("GL.release(") == 1, dict(zip(marks, at))
    call = main[at[5]:main.index("\n", at[5])]
    assert 'last=pair == args.pairs - 1 and k == len(order) - 1 and (_HOT or {}).get("last", _HOT is None)' in call and "after=_wstage" in call
    assert "bundle=_rb_started is not None" in call
    after = main[main.index("\n", at[5]):]
    assert not re.search(r"torch\.cuda|\.cuda\(|nvidia_smi|synchronize\(|empty_cache", after)


def test_the_lease_record_dates_the_release_and_names_its_stage_or_why_it_was_kept(tmp_path):
    def run_tl(marker):
        def go(to, argv):
            (tmp_path / "gpu_lease.requested").write_text("2026-10-01T20:00:00Z\n")
            (tmp_path / "gpu_lease.started").write_text("2026-10-01T20:01:00Z\n")
            if marker is not None:
                (tmp_path / "gpu_lease.released").write_text(json.dumps(marker))
            return 0
        return go

    t = 1790884800.0                                   # 2026-10-01T20:00:00Z
    env = {"GPU_LEASE": "/l/gpu-lease"}
    ticks = lambda: iter([t - 30, t + 600]).__next__
    _, rec = GL.run(str(tmp_path), str(tmp_path / "log"), ["x"], 600, env,
                    run_tl({"released": "2026-10-01T20:02:30Z", "t": t + 150.0, "released_at": "work.pair0.instrumented"}), clock=ticks())
    assert rec["released"] == "2026-10-01T20:02:30Z" and rec["released_at"] == "work.pair0.instrumented" and rec["kept"] is None
    assert rec["held_min"] == 1.5 and rec["waited_min"] == 1.0 and rec["process_ended"] == "2026-10-01T20:10:00Z"
    _, rec = GL.run(str(tmp_path), str(tmp_path / "log"), ["x"], 600, env, run_tl({"kept": "a lease of 2 GPUs"}), clock=ticks())
    assert rec["released"] == rec["process_ended"] and rec["released_at"] == "process.end" and rec["kept"] == "a lease of 2 GPUs"
    assert rec["held_min"] == 9.0
    _, rec = GL.run(str(tmp_path), str(tmp_path / "log"), ["x"], 600, env, run_tl(None), clock=ticks())
    assert rec["released_at"] == "process.end" and "ended before its release" in rec["kept"] and rec["held_min"] == 9.0


def _release_kept(tmp_path, monkeypatch, active, on):
    monkeypatch.setattr(GL, "_HELD", [(None, None, [], str(tmp_path / "gpu_lease.started"), dict(ONE))])
    monkeypatch.setattr(GL, "DRIVER", lambda: SimpleNamespace(reset=lambda: None, active=lambda: active))
    monkeypatch.setattr(GL, "on_gpus", lambda uuids: on)
    monkeypatch.setattr(GL, "_shut_down", lambda eng: None)
    waits: list[float] = []
    tl = SimpleNamespace(start=lambda *a, **k: None, end=lambda *a, **k: None)
    assert GL.release(tl, ENG, [_com()], _args(), last=True, after="work.pair0.instrumented", bundle=True, wait=waits.append) is False
    assert not GL._RELEASED.is_set() and len(GL._HELD) == 1 and waits == [0.5] * 9
    return json.loads((tmp_path / "gpu_lease.released").read_text())


def test_a_context_that_does_not_go_keeps_the_lease(tmp_path, monkeypatch):
    assert "still active on device(s) [0]" in _release_kept(tmp_path, monkeypatch, [0], [])["kept"]
    assert "still lists a compute process on ['GPU-5']" in _release_kept(tmp_path, monkeypatch, [], ["GPU-5"])["kept"]
    assert "can't list the compute processes" in _release_kept(tmp_path, monkeypatch, [], None)["kept"]


def test_the_commit_gives_its_gpu_back_after_its_last_run_and_ends_with_its_command(tmp_path):
    env = {"PATH": os.environ["PATH"], "GPU_LEASE": fake_lease(tmp_path)}
    GL.hide_devices(env)
    seen, exited, lock = tmp_path / "seen", tmp_path / "atexit", tmp_path / "5.lock"
    cmd = commit("import atexit, subprocess, types\ncalls = []\n"
                 "GL.DRIVER = lambda: types.SimpleNamespace(reset=lambda: calls.append('reset'), active=lambda: [])\n"
                 "GL.on_gpus = lambda uuids: calls.append(uuids) or []\n"
                 "GL._shut_down = lambda eng: calls.append('shutdown')\nGL._no_device = lambda: calls.append('no device')\n"
                 f"atexit.register(lambda: open({str(exited)!r}, 'w').write('ran'))\n"
                 "tl = types.SimpleNamespace(start=lambda n, **k: calls.append(n), end=lambda n, **k: calls.append(k.get('released')))\n"
                 "from tests.pipeline.test_gpu_lease import ENG, _args, _com\n"
                 "first = GL.release(tl, ENG, [_com()], _args(), last=False, after='work.pair0.control', bundle=True)\n"
                 "com = _com(_stop_worker=lambda: calls.append('worker stopped'))\n"
                 "ok = GL.release(tl, ENG, [com], _args(), last=True, after='work.pair0.instrumented', bundle=True, wait=lambda s: None)\n"
                 f"free = subprocess.run(['flock', '-n', {str(lock)!r}, 'true']).returncode == 0\n"
                 f"json.dump(dict(first=first, ok=ok, free=free, calls=calls, held=len(GL._HELD)), open({str(seen)!r}, 'w'))\n"
                 "GL.run_main(lambda: sys.exit(3))\n")
    log = tmp_path / "commit.log"

    def run_tl(to: int, argv: list[str]) -> int:
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=Path(__file__).parents[2]).returncode

    rc, rec = GL.run(str(tmp_path), str(log), cmd, 600, env, run_tl)
    assert rc == 3, log.read_text()                    # the holder's end after the release did not kill the Commit
    assert json.loads(seen.read_text()) == {"first": False, "ok": True, "free": True, "held": 0,
                                            "calls": ["gpu_lease.release", "worker stopped", "shutdown", "reset", ["GPU-5"], "no device", True]}
    assert not exited.exists(), "the released Commit ran its interpreter's teardown"
    assert "GPU LEASE released after work.pair0.instrumented" in log.read_text()
    assert rec["released_at"] == "work.pair0.instrumented" and rec["kept"] is None and rec["rc"] == 3 and rec["held_min"] is not None
    assert {"shutdown_s", "reset_s", "holder_s"} <= set(rec)


def test_a_context_that_reappears_after_the_release_ends_the_commit(tmp_path, monkeypatch):
    started = str(tmp_path / "gpu_lease.started")
    looks = iter([[], [], [0]])
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(GL.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    GL._watch(SimpleNamespace(active=lambda: next(looks)), started, wait=lambda s: None)
    assert killed == [(os.getpid(), signal.SIGKILL)]
    note = json.loads((tmp_path / "gpu_lease.released").read_text())
    assert "context_reappeared" in note and "MainThread" in note["threads"]


def test_the_holder_ending_after_the_release_leaves_the_commit_running(monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(GL.os, "kill", lambda pid, sig: killed.append(sig))
    GL._RELEASED.set()
    try:
        GL._end_with(subprocess.Popen(["true"]))
    finally:
        GL._RELEASED.clear()
    assert killed == []


def test_a_commit_that_has_not_released_returns_and_raises_as_its_command_does():
    assert GL.run_main(lambda: 5) == 5
    with pytest.raises(SystemExit):
        GL.run_main(lambda: sys.exit(4))


def test_after_the_release_torch_sees_no_device():
    torch = pytest.importorskip("torch")
    was = torch.cuda.is_available
    p = GL._no_device()
    try:
        assert torch.cuda.is_available() is False
    finally:
        p.uninstall()
    assert torch.cuda.is_available is was


#: A stand-in `match`: a GPU phase takes the lease as `match.main` does, and each run notes its phase, its devices and the lease
#: locks it holds in ``legs.jsonl``.
MATCH_LEG = '''import json, os, sys
from verity_vllm.pipeline import cli, gpu_lease as GL
args = sys.argv[1:]
phase = args[args.index("--phase") + 1]
if phase in ("gpu", "all"):
    cli.process_settings(os.environ)
    GL.acquire()
locks = sum(os.path.realpath("/proc/self/fd/" + f) == {lock!r} for f in os.listdir("/proc/self/fd"))
with open({legs!r}, "a") as f:
    f.write(json.dumps({{"phase": phase, "devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "locks": locks}}) + "\\n")
'''


@pytest.mark.parametrize("leased, legs", [
    (True, [{"phase": "gpu", "devices": "5", "locks": 1}, {"phase": "cpu", "devices": "", "locks": 0}]),
    (False, [{"phase": "all", "devices": "5", "locks": 0}]),
])
def test_a_single_gpu_match_holds_its_lease_for_its_gpu_phase_only(tmp_path, leased, legs):
    """A leased row's Match runs `match` as its GPU phase under the lease, holding it to the phase's exit (its record in
    ``match/lease/``), then its CPU phase with no device and no lease; an unleased row's runs both phases in one process, as before."""
    from tests.pipeline.test_row import _row
    t = _row(tmp_path / "sweep")
    t.B, t.I, t.O, t.snap_def, t.snap_steps, t.ck = 1, 256, 32, "all", None, str(tmp_path / "ck")
    (tmp_path / "leg.py").write_text(MATCH_LEG.format(lock=str(tmp_path / "5.lock"), legs=str(tmp_path / "legs.jsonl")))
    t.cli = lambda *a, py=None: [sys.executable, str(tmp_path / "leg.py"), *a]
    env = {k: v for k, v in t.child_env.items() if not k.startswith(("GPU_LEASE", "VERITY_LEASE"))}
    env["CUDA_VISIBLE_DEVICES"] = "5"
    if leased:
        env["GPU_LEASE"] = fake_lease(tmp_path)
        GL.hide_devices(env)
    t.child_env = env
    with pytest.raises(RD.Stop):              # the stand-in folds nothing, so the Match fails after its stages
        t.match()
    assert [json.loads(ln) for ln in (tmp_path / "legs.jsonl").read_text().splitlines()] == legs
    rec = Path(t.d, "match", "lease", GL.RECORD)
    assert rec.exists() is leased
    if leased:
        assert json.loads(rec.read_text())["kept"] == GL.MATCH_LEG_KEPT and released(tmp_path / "5.lock")


def test_the_match_takes_its_lease_in_a_gpu_phase_before_its_first_stage():
    from verity_vllm.pipeline import match as M
    src = Path(M.__file__).read_text()
    main = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
    assert main.count("GL.acquire()") == 1 and main.index("GL.acquire()") < main.index("run_stage(")
    assert re.search(r'if a\.phase in \("gpu", "all"\):\n\s+GL\.acquire\(\)', main)
