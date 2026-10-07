"""A leased TP Commit (`pipeline.tp.lease`): the driver takes every GPU of the task at once right before its engine; after the last
run's finalize has fixed each rank's root, every rank leaves its device and resets its context, the driver's context goes, and every
lock goes back at once, before the checks that follow.  A rank that still needs its device, or a step that fails, keeps the whole lease
to the Commit's exit (never part of it), and a Commit that ends before its release frees every lock at its exit.  A leased TP Match
leases every GPU for its capture and its control only, each to its exit; its check, served-shape derives and fold Match run with no
device and no lease, and its verdict is an unleased Match's.  CPU-only: the ranks are `TP2CaptureWorkerExtension`'s commit methods over
a stand-in committer and model, the LLM dispatches `collective_rpc` to them, the Match's legs are stand-in processes, and the lease is
`test_gpu_lease`'s stand-in tool."""
from __future__ import annotations

import ast
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.pipeline.test_gpu_lease import fake_lease, released
from tests.pipeline.test_row import TP_DENSE, _tp_row
from verity_vllm.engine import hooks
from verity_vllm.engine.rank_worker import TP2CaptureWorkerExtension as _cls
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline.tp import lease as TPL

GPUS = ("4", "5")
LAST = "work.pair0.instrumented"


class _T:
    """A tensor of ``n`` two-byte elements on the device until its ``.data`` is set to its host copy, as `torch.Tensor.data` moves it."""

    def __init__(self, n: int, device: str = "cuda") -> None:
        self.n, self.device = n, device

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def data(self) -> _T:
        return self

    @data.setter
    def data(self, v: _T) -> None:
        self.device = v.device

    def cpu(self) -> _T:
        return _T(self.n, "cpu")

    def numel(self) -> int:
        return self.n

    def element_size(self) -> int:
        return 2


class _Model:
    """Parameters on the device (one tied, listed twice), a buffer on the device and one on the host: 3 MiB and 16 bytes to move."""

    def __init__(self) -> None:
        w = _T(1 << 20)
        self.p, self.b = [w, _T(1 << 19), w], [_T(8), _T(4, "cpu")]

    def parameters(self):
        return iter(self.p)

    def buffers(self):
        return iter(self.b)


class _Com:
    """A committer that retains its bytes in registered shared host mappings, so nothing after its root reads the device
    (`gpu_lease._kept_by`); each call goes on the shared event list."""

    retain_exclude = None
    pool = SimpleNamespace(mode="exact-registered-shared")

    def __init__(self, rank: int, ev: list, retain: str = "host") -> None:
        self.rank, self.ev, self.retain = rank, ev, retain

    def _retained_device_bytes(self) -> int:
        return 0

    def reset(self) -> None:
        self.ev.append((self.rank, "reset"))

    def install(self) -> None:
        self.ev.append((self.rank, "install"))

    def uninstall(self) -> None:
        self.ev.append((self.rank, "uninstall"))

    def _stop_worker(self) -> None:
        self.ev.append((self.rank, "worker stopped"))

    def finalize(self):
        self.ev.append((self.rank, "root"))
        return SimpleNamespace(steps=[SimpleNamespace(n_leaves=3, n_bytes=12, n_tensors=2)], run_root=bytes([self.rank + 1]) * 32,
                               root_ready_delay_s=0.0, t_finalize_s=0.0, spans={"total_s": 0.0}, weight_registration=None, persistent_bytes=0,
                               temporary_bytes_peak=0, extra={"errors": []})

    def coverage_declaration(self, rc):
        return {"classes": []}


def free(lock: Path) -> bool:
    """Whether nothing holds ``lock`` now.  A shared probe: the lease holds its locks exclusively, and two ranks probing at once
    must not see each other's probe as a holder."""
    return subprocess.run(["flock", "-s", "-n", str(lock), "true"]).returncode == 0


class _Rank:
    """A rank's commit methods; its checks after the root note whether every lease lock was free then and the release was done."""

    tp2_commit_install = _cls.tp2_commit_install
    tp2_commit_finalize = _cls.tp2_commit_finalize
    tp2_commit_reset = _cls.tp2_commit_reset

    def __init__(self, rank: int, ev: list, locks: list[Path]) -> None:
        self._tp2_commit = {"com": _Com(rank, ev), "rank": rank, "eng": SimpleNamespace(model=_Model()), "base": None,
                            "replay": {"programs": {}}, "rc": None}
        self.ev, self.locks = ev, locks

    def _tp2_binding(self, rc):
        self.ev.append((self._tp2_commit["rank"], "checks", [free(p) for p in self.locks], GL._RELEASED.is_set()))
        return {"ok": True}

    def _tp2_attribution(self, rc):
        return None

    def _tp2_sampled_replay(self, rc):
        return {"sampled_replay": {"result": True}}


class _LLM:
    """`LLM.collective_rpc` over the ranks: a callable runs as `fn(worker, *args)` on each, a name as the worker's method."""

    def __init__(self, ranks: list[_Rank], ev: list) -> None:
        self.ranks, self.ev = ranks, ev

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.ev.append(("rpc", getattr(method, "__name__", method), kwargs or {}))
        return [method(r, *args) if callable(method) else getattr(r, method)(**(kwargs or {})) for r in self.ranks]


class _Cuda:
    """The CUDA driver (`gpu_lease.DRIVER`) of the driver and every rank: a reset goes on ``ev``; `active` lists the next of ``looks``
    (none left: nothing)."""

    def __init__(self, ev: list) -> None:
        self.ev, self.looks = ev, []

    def __call__(self) -> _Cuda:
        return self

    def reset(self) -> None:
        self.ev.append("context reset")

    def active(self) -> list[int]:
        return self.looks.pop(0) if self.looks else []


class _WorkerProc:
    """vLLM's `WorkerProc`, whose ``shutdown`` a rank that gave its GPU back replaces."""

    def shutdown(self) -> None:
        raise AssertionError("vLLM's worker teardown")


class _Now:
    """`threading.Thread` that runs its target at ``start``, so the watch a release starts is on the record at once."""

    def __init__(self, target, args=(), daemon=None, name=None) -> None:
        self.target, self.args = target, args

    def start(self) -> None:
        self.target(*self.args)


def device(ev: list, watched: list) -> list[tuple]:
    """The device side's stand-ins as (owner, name, value): the CUDA driver, torch's synchronize, the distributed groups, vLLM's
    `WorkerProc`, the threads and the watch they start (``watched``), the device's end in torch (``ev`` notes 'no device'), and the
    engine's pristine check after a rank's finalize."""
    import torch

    from verity_vllm.pipeline import commit as PC

    groups = SimpleNamespace(destroy_model_parallel=lambda: ev.append("groups destroyed"), destroy_distributed_environment=lambda: None)
    return [(PC, "assert_pristine", lambda eng, base: None),
            (GL, "DRIVER", _Cuda(ev)), (GL, "cuda_initialised", lambda: True), (GL, "_no_device", lambda: ev.append("no device")),
            (GL, "_watch", lambda drv, started: watched.append(started)), (GL, "threading", SimpleNamespace(Thread=_Now)),
            (TPL, "threading", SimpleNamespace(Thread=_Now)), (TPL, "_parallel_state", lambda: groups), (TPL, "_worker_proc", lambda: _WorkerProc),
            (torch.cuda, "synchronize", lambda: None), (torch.cuda, "empty_cache", lambda: None)]


def stand_ins(tmp: str) -> tuple[list, _LLM]:
    """In a Commit process of its own (the end-to-end tests): the device's stand-ins set, nvidia-smi the lease tool's stand-in; returns
    the event list and the LLM over two ranks."""
    ev: list = []
    for owner, name, value in device(ev, []):
        setattr(owner, name, value)
    return ev, _LLM([_Rank(r, ev, [Path(tmp) / f"{g}.lock" for g in GPUS]) for r in (0, 1)], ev)


@pytest.fixture
def lease(tmp_path, monkeypatch):
    """A held lease of GPUs 4 and 5 as `gpu_lease.acquire` leaves it (both locks held on this process's descriptors, a holder waiting
    for EOF, the leased devices), with the device's stand-ins."""
    pytest.importorskip("torch")
    started = tmp_path / "gpu_lease.started"
    locks = [tmp_path / f"{g}.lock" for g in GPUS]
    fds = []
    for p in locks:
        fds.append(os.open(p, os.O_CREAT | os.O_WRONLY | os.O_APPEND))
        fcntl.flock(fds[-1], fcntl.LOCK_EX)
    holder = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE)
    ours, theirs = socket.socketpair()
    theirs.close()
    held = [(holder, ours, fds, str(started), {"CUDA_VISIBLE_DEVICES": "4,5", "GPU_LEASE_UUID": "GPU-4,GPU-5"})]
    monkeypatch.setattr(GL, "_HELD", held)
    monkeypatch.setattr(GL, "_RELEASED", threading.Event())
    monkeypatch.setattr(TPL, "_OFF", threading.Event())
    ev: list = []
    watched: list = []
    for owner, name, value in device(ev, watched):
        monkeypatch.setattr(owner, name, value)
    smi: list = []
    monkeypatch.setattr(GL, "on_gpus", lambda uuids: smi.append(uuids) or [])
    ranks = [_Rank(r, ev, locks) for r in (0, 1)]
    yield SimpleNamespace(ev=ev, watched=watched, smi=smi, locks=locks, started=started, holder=holder, ranks=ranks, llm=_LLM(ranks, ev),
                          cuda=GL.DRIVER, note=lambda: json.loads((tmp_path / "gpu_lease.released").read_text()))
    for p in hooks.live():
        if p.owner is _WorkerProc:
            p.uninstall()
    for fd in fds if held else []:                    # a release closed them, and their numbers may be another file's by now
        os.close(fd)
    ours.close()
    holder.kill()
    holder.wait()


def _rpcs(ev: list) -> list[tuple[str, dict]]:
    return [(e[1], e[2]) for e in ev if e[0] == "rpc"]


def _last_run(lease, last: bool = True) -> list[dict]:
    lease.llm.collective_rpc("tp2_commit_install")
    fin = TPL.finalize(lease.llm, 60.0, LAST, last=last, wait=lambda s: None)
    TPL.reset(lease.llm, 60.0)
    return fin


def test_the_last_runs_finalize_fixes_every_root_then_gives_every_gpu_back_at_once_before_the_checks(lease):
    lease.cuda.looks = [[], [], [0]]                   # each rank's context gone after its reset; the driver's own context active
    fin = _last_run(lease)
    assert _rpcs(lease.ev) == [("tp2_commit_install", {}), ("tp2_commit_finalize", {"checks": False}), ("rank_kept", {}), ("rank_off", {}),
                               ("rank_reset", {}), ("tp2_commit_finalize", {})]          # no reset for another run
    assert [e for e in lease.ev if e[0] != "rpc"] == [
        (0, "reset"), (0, "install"), (1, "reset"), (1, "install"),
        (0, "root"), (0, "uninstall"), (1, "root"), (1, "uninstall"),                   # every root fixed before any rank leaves its device
        (0, "worker stopped"), "groups destroyed", (1, "worker stopped"), "groups destroyed",
        "context reset", "no device", "context reset", "no device",                    # every rank off before any context is reset
        "context reset",                                                               # the driver's
        "no device",
        (0, "checks", [True, True], True), (1, "checks", [True, True], True)]          # both locks free and the lease let go first
    assert lease.smi == [["GPU-4", "GPU-5"]] and lease.watched == [str(lease.started)] * 3 and GL._HELD == []
    assert lease.holder.wait() == 0 and _WorkerProc.shutdown is TPL._exit
    assert [(f["rank"], f["run_root"], f["binding"]) for f in fin] == [(0, "01" * 32, {"ok": True}), (1, "02" * 32, {"ok": True})]
    assert all(not t.is_cuda for r in lease.ranks for m in [r._tp2_commit["eng"].model] for t in [*m.p, *m.b])
    note = lease.note()
    assert note["released_at"] == LAST and "kept" not in note and note["shutdown_error"] is None and {"shutdown_s", "reset_s", "holder_s"} <= set(note)
    assert [(r["rank"], r["moved_bytes"], r["why"], r["teardown_error"]) for r in note["ranks"]] == [(0, (3 << 20) + 16, None, None),
                                                                                                   (1, (3 << 20) + 16, None, None)]


def _pinned(com: _Com) -> None:
    com.pool = SimpleNamespace(mode="torch-pinned-pow2")


@pytest.mark.parametrize("keep,why", [
    ({1: lambda c: setattr(c, "retain", "none")}, "rank 1: the openings after release replay the run on the engine"),
    ({0: _pinned, 1: lambda c: setattr(c, "retain", "none")}, "rank 0: the retained bytes are in torch's pinned host memory "
                                                              "(torch-pinned-pow2), which the CUDA context's end frees; rank 1: the openings"),
], ids=["one-rank", "both-ranks"])
def test_a_rank_that_still_needs_its_device_keeps_every_gpu_to_the_exit(lease, keep, why):
    for r, f in keep.items():
        f(lease.ranks[r]._tp2_commit["com"])
    _last_run(lease)
    assert _rpcs(lease.ev) == [("tp2_commit_install", {}), ("tp2_commit_finalize", {"checks": False}), ("rank_kept", {}),
                               ("tp2_commit_finalize", {}), ("tp2_commit_reset", {})]    # no rank left its device
    assert [e for e in lease.ev if e[1] == "checks"] == [(0, "checks", [False, False], False), (1, "checks", [False, False], False)]
    assert lease.note()["kept"].startswith(why) and lease.note()["after"] == LAST and len(GL._HELD) == 1 and lease.watched == []


def _context_stays(lease, monkeypatch):
    lease.cuda.looks = [[], [0]]
    return "rank 1: a CUDA context is still active on device(s) [0] after its reset"


def _off_fails(lease, monkeypatch):
    def stuck():
        raise RuntimeError("worker stuck")
    lease.ranks[0]._tp2_commit["com"]._stop_worker = stuck
    return "the ranks could not give their GPUs back: RuntimeError('worker stuck')"


def _smi_lists(lease, monkeypatch):
    monkeypatch.setattr(GL, "on_gpus", lambda uuids: ["GPU-5"])
    return "nvidia-smi still lists a compute process on ['GPU-5'] after the reset"


@pytest.mark.parametrize("fail", [_context_stays, _off_fails, _smi_lists])
def test_a_step_that_fails_or_leaves_a_context_keeps_every_gpu_to_the_exit(lease, monkeypatch, fail):
    why = fail(lease, monkeypatch)
    _last_run(lease)
    assert _rpcs(lease.ev)[-1] == ("tp2_commit_finalize", {}) and ("tp2_commit_reset", {}) not in _rpcs(lease.ev)
    assert [e for e in lease.ev if e[1] == "checks"] == [(0, "checks", [False, False], False), (1, "checks", [False, False], False)]
    assert lease.note()["kept"] == why and len(GL._HELD) == 1 and not GL._RELEASED.is_set() and lease.holder.poll() is None


def test_a_teardown_error_goes_on_the_record_and_every_gpu_still_goes_back(lease, monkeypatch):
    def broken():
        raise RuntimeError("nccl abort")
    monkeypatch.setattr(TPL, "_parallel_state", lambda: SimpleNamespace(destroy_model_parallel=broken, destroy_distributed_environment=broken))
    _last_run(lease)
    note = lease.note()
    assert note["released_at"] == LAST and note["shutdown_error"] == "rank 0: RuntimeError('nccl abort'); rank 1: RuntimeError('nccl abort')"
    assert GL._HELD == [] and all(free(p) for p in lease.locks)


@pytest.mark.parametrize("last,held", [(False, True), (True, False)])
def test_a_run_before_the_last_or_a_commit_without_a_lease_finalizes_and_resets_as_before(lease, monkeypatch, last, held):
    if not held:
        monkeypatch.setattr(GL, "_HELD", [])
    fin = _last_run(lease, last=last)
    assert _rpcs(lease.ev) == [("tp2_commit_install", {}), ("tp2_commit_finalize", {}), ("tp2_commit_reset", {})]
    assert [f["binding"] for f in fin] == [{"ok": True}] * 2 and lease.watched == [] and not (lease.started.parent / "gpu_lease.released").exists()


def test_the_driver_that_never_initialised_cuda_has_no_context_to_reset(monkeypatch):
    monkeypatch.setattr(GL, "cuda_initialised", lambda: False)
    monkeypatch.setattr(GL, "DRIVER", lambda: (_ for _ in ()).throw(RuntimeError("cuDeviceGetCount: CUDA driver error 3")))
    drv = TPL._Driver()
    assert drv.active() == [] and drv.reset() is None


def tp_commit(script: str) -> list[str]:
    """A TP Commit stand-in: `script` after the CLI's environment and `pipeline.tp.lease.acquire` (``held``: whether it took the lease;
    ``locks``: the lease lock descriptors this process holds then)."""
    return [sys.executable, "-c", "import json, os, sys\nfrom verity_vllm.pipeline import cli, gpu_lease as GL\nfrom verity_vllm.pipeline.tp import "
            "lease as TPL\ncli.process_settings(os.environ)\nheld = TPL.acquire()\n"
            "locks = sorted(os.path.basename(p) for p in (os.path.realpath(f'/proc/self/fd/{f}') for f in os.listdir('/proc/self/fd')) if p.endswith('.lock'))\n"
            + script]


def _run(tmp_path: Path, script: str) -> tuple[int, dict, str]:
    lease = fake_lease(tmp_path, ",".join(GPUS))
    env = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "GPU_LEASE": lease, "GPU_LEASE_N": str(len(GPUS))}   # nvidia-smi: `fake_smi`
    GL.hide_devices(env)
    log = tmp_path / "commit.log"

    def run_tl(to: int, argv: list[str]) -> int:
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=Path(__file__).parents[2]).returncode

    rc, rec = GL.run(str(tmp_path), str(log), tp_commit(script), 600, env, run_tl)
    return rc, rec, log.read_text()


def test_the_tp_commit_takes_every_gpu_at_once_and_gives_them_back_together_while_it_runs(tmp_path):
    seen = tmp_path / "seen"
    rc, rec, log = _run(tmp_path, f"from tests.pipeline.test_tp_gpu_lease import LAST, stand_ins\nev, llm = stand_ins({str(tmp_path)!r})\n"
                                  "llm.collective_rpc('tp2_commit_install')\nTPL.finalize(llm, 60.0, LAST, last=True, wait=lambda s: None)\n"
                                  "TPL.reset(llm, 60.0)\n"
                                  f"json.dump(dict(held=held, locks=locks, devices=os.environ['CUDA_VISIBLE_DEVICES'], spawn=os.environ.get(TPL.SPAWN), "
                                  f"checks=[e[2:] for e in ev if e[1] == 'checks'], resets=sum(e == ('rpc', 'tp2_commit_reset', {{}}) for e in ev)), "
                                  f"open({str(seen)!r}, 'w'))\nGL.run_main(lambda: 0)\n")
    assert rc == 0, log
    assert json.loads(seen.read_text()) == {"held": True, "locks": ["4.lock", "5.lock"], "devices": "4,5", "spawn": "spawn",
                                            "checks": [[[True, True], True], [[True, True], True]], "resets": 0}
    assert (tmp_path / "lease.args").read_text().startswith("2 --wait ") and f"GPU LEASE released after {LAST}: 2 ranks off their GPUs" in log
    assert rec["released_at"] == LAST and rec["kept"] is None and rec["held_min"] is not None and [r["rank"] for r in rec["ranks"]] == [0, 1]
    assert (tmp_path / "smi.calls").read_text().split()[0] == "held"      # nothing on the leased GPUs, seen before the locks went


@pytest.mark.parametrize("script,rc,kept", [
    ("raise SystemExit('the engine did not start')\n", 1, "the Commit ended before its release"),
    ("from tests.pipeline.test_tp_gpu_lease import LAST, stand_ins\n"
     "ev, llm = stand_ins(os.path.dirname(os.environ[GL.STARTED]))\n"
     "llm.ranks[1]._tp2_commit['com']._stop_worker = lambda: 1 / 0\nllm.collective_rpc('tp2_commit_install')\n"
     "TPL.finalize(llm, 60.0, LAST, last=True, wait=lambda s: None)\nprint('checks', [e[2:] for e in ev if e[1] == 'checks'], flush=True)\n"
     "GL.run_main(lambda: sys.exit(5))\n", 5, "the ranks could not give their GPUs back: ZeroDivisionError"),
], ids=["before-its-engine", "a-rank-stuck-on-its-device"])
def test_a_tp_commit_that_fails_holding_its_lease_frees_every_gpu_at_its_exit(tmp_path, script, rc, kept):
    got, rec, log = _run(tmp_path, script)
    assert got == rc and rec["kept"].startswith(kept) and rec["released_at"] == "process.end", log
    assert rc == 1 or "checks [([False, False], False), ([False, False], False)]" in log      # the checks ran holding both locks
    assert all(released(tmp_path / f"{g}.lock") for g in GPUS)


def test_the_tp_commit_takes_its_lease_before_the_device_and_gives_it_back_only_at_its_last_runs_finalize():
    """In `tp.commit.main`: the lease before the target reads the device and before the engine; the one release at the finalize of the
    last instrumented run (not under --value-check, whose run follows), before the openings and the cross-rank pass; and in the row,
    the Commit under the lease."""
    src = Path(TPL.__file__).with_name("commit.py").read_text()
    main = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
    marks = ["TPL.acquire()", "va.target_of_workload(wl)", "LLM(**kw)", '"tp2_commit_make"', '"tp2_commit_install"', "TPL.finalize(llm",
             '"tp2_commit_open"', '"tp2_commit_xrank_dump"', "TPL.reset(llm", "if args.value_check:"]
    at = [main.index(m) for m in marks]
    assert at == sorted(at) and main.count("TPL.finalize(") == 1, dict(zip(marks, at))
    assert '"tp2_commit_finalize"' not in main and '"tp2_commit_reset"' not in main and "torch.cuda" not in main
    call = main[at[5]:main.index("\n", at[5])]
    assert 'last=pair == args.pairs - 1 and k == len(order) - 1 and not args.value_check' in call and 'f"work.pair{pair}.{arm}"' in call
    assert 'self.run_gpu("commit.tp_commit", to_k' in Path(TPL.__file__).parents[1].joinpath("row_tp.py").read_text()


@pytest.mark.parametrize("item,leased,mode", [(None, False, "device"), (None, True, "device"), ("host", False, "host"),
                                              ("device-checked", True, "device-checked")])
def test_a_tp_commit_hashes_its_weights_on_the_device_unless_the_item_says_otherwise(tmp_path, monkeypatch, item, leased, mode):
    from verity_vllm.pipeline import row_records as R
    t = _tp_row(tmp_path, TP_DENSE)
    os.makedirs(f"{t.d}/commit")
    Path(f"{t.d}/commit/summary.json").write_text('{"pass": true}')
    t.hidden_so = str(tmp_path / "hidden.so")
    Path(t.hidden_so).touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_LEASE", "VERITY_LEASE", GL.WEIGHTS_HASH))}
    env.update({GL.WEIGHTS_HASH: item} if item else {})
    env.update({"GPU_LEASE": "/l/gpu-lease"} if leased else {})
    t.child_env = env
    for name, value in (("tp_rank_workload_digest", lambda bdir, r: f"wd{r}"), ("tp_manifest_complete", lambda d: True),
                        ("applicable_taps", lambda man, ck: [])):
        monkeypatch.setattr(R, name, value)
    seen: list[dict] = []
    monkeypatch.setattr(t, "manifest_global", lambda *a: 0)
    monkeypatch.setattr(t, "run_tl", lambda stage, to, log, cmd: seen.append(dict(t.child_env)) or 0)
    t.commit()
    assert [e.get(GL.WEIGHTS_HASH) for e in seen] == [mode] and (GL.STARTED in seen[0]) == leased
    assert f"weights hash {mode}" in Path(f"{t.d}/row.log").read_text() and "commit PASS" in Path(f"{t.d}/stages.txt").read_text()


#: Every Match leg's stand-in, run as ``leg.py <command> ARGS...`` from the dir of the lease tool's stand-in: it notes what it saw
#: (its devices, its lease, the lease locks it holds, whether each was free) in ``legs.jsonl`` and writes what the row reads after it.
#: A capture or control ends as its exit would end its lease (every lock and the holder's stdin closed, the holder gone) before it
#: returns, so the next leg sees the lease as the next process after it sees it.
LEG = '''\
import fcntl, json, os, sys
from verity_vllm.pipeline import cli, gpu_lease as GL
from verity_vllm.pipeline.tp import lease as TPL

TMP, cmd, args = os.path.dirname(os.path.abspath(__file__)), sys.argv[1], sys.argv[2:]
out = args[args.index("--out") + 1]


def free(g):
    # shared: the per-rank legs (build, global-program) run at once, and an exclusive probe would see the other leg's probe
    fd = os.open(os.path.join(TMP, g + ".lock"), os.O_CREAT | os.O_WRONLY | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


if cmd == "tp-capture":
    cli.process_settings(os.environ)
    TPL.acquire()
fds = (os.path.realpath(f"/proc/self/fd/{f}") for f in os.listdir("/proc/self/fd"))
seen = {"leg": os.path.basename(out) if cmd == "tp-capture" else cmd, "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lease": os.environ.get(GL.STARTED, "").partition("/match/")[2], "locks": sorted(os.path.basename(p) for p in fds if p.endswith(".lock"))}
seen["free"] = [free(g) for g in ("4", "5")]
with open(os.path.join(TMP, "legs.jsonl"), "a") as f:
    f.write(json.dumps(seen) + "\\n")
os.makedirs(out, exist_ok=True)
if cmd == "tp-capture":
    reqs = [{"prompt_token_ids": [1] * 16, "generated_token_ids": [2] * n} for n in (4, 2)]      # the second stopped before its cap
    json.dump({"requests": reqs}, open(os.path.join(out, "tokens.json"), "w"))
    if GL._HELD:
        holder, ours, held = GL._HELD[0][:3]
        GL._EXITING.set()
        for fd in held:
            os.close(fd)
        ours.close()
        holder.stdin.close()
        holder.wait()
elif cmd == "tp-match":
    json.dump({"pass": True, "structure": {"pass": True, "per_rank": {"0": {"n_collectives": 6, "build_signature_len": 6, "n_steps": 4}}},
               "provenance": {"pass": True, "collectives_checked": 6, "n_mismatches": 0}, "tokens": {"pass": True}},
              open(os.path.join(out, "tp2_match.json"), "w"))
elif cmd == "build":
    json.dump({"ok": True}, open(os.path.join(out, "result.json"), "w"))
elif cmd == "global-program":
    json.dump({"workload_digest": "wd"}, open(os.path.join(out, "workload_program.json"), "w"))
elif cmd == "tp-fold-match":
    json.dump({"pass": True, "legs": {"0": "PASS", "1": "PASS"}, "problems": []}, open(os.path.join(out, "tp2_fold_match.json"), "w"))
'''


def _match(tmp: Path, leased: bool) -> tuple[Any, list[dict]]:
    """A TP Match of a built row (step and declared request Programs on disk), every leg `LEG`; leased: the item's lease tool is
    `test_gpu_lease`'s stand-in over GPUs 4 and 5 and its devices are hidden, as `Row.setup` leaves them; else they are visible."""
    tmp.mkdir()
    t = _tp_row(tmp, TP_DENSE)
    t.shapes, t.MML, t.tgt_args, t.fold_profile = [(16, 3)], 64, [], "derived_test_tp2"
    for r in range(t.W):
        for d in ("build_step", "build_request_LP16_T3"):
            os.makedirs(f"{t.rank(r)}/{d}")
            Path(f"{t.rank(r)}/{d}/result.json").write_text('{"ok": true}')
    os.makedirs(t.M)
    (tmp / "leg.py").write_text(LEG)
    t.cli = lambda *a, py=None: [sys.executable, str(tmp / "leg.py"), *a]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_LEASE", "VERITY_LEASE", GL.WEIGHTS_HASH))}
    env.update(PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES=",".join(GPUS))
    if leased:
        env.update(GPU_LEASE=fake_lease(tmp, ",".join(GPUS)), GPU_LEASE_N=str(len(GPUS)))
        GL.hide_devices(env)
    t.child_env = env
    t.match()
    return t, [json.loads(ln) for ln in (tmp / "legs.jsonl").read_text().splitlines()]


def _stages(t: Any) -> list[str]:
    """The row's verdict lines without their times."""
    return [re.sub(r"wall=\d+s", "wall=", re.sub(r" \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", "", ln.rstrip("\n"))) for ln in open(f"{t.d}/stages.txt")]




def test_the_tp_capture_takes_its_lease_before_the_device_and_before_its_engine():
    """In `tp.capture.main`: the lease before anything reads the device or builds the engine (the case's environment, the engine's
    arguments, the LLM), and no CUDA call of its own before it."""
    src = Path(TPL.__file__).with_name("capture.py").read_text()
    main = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
    marks = ["TPL.acquire()", "case_of(", "LLM(**kw)"]
    at = [main.index(m) for m in marks]
    assert at == sorted(at) and main.count("TPL.acquire()") == 1, dict(zip(marks, at))
    assert "torch.cuda" not in main[:at[0]]
