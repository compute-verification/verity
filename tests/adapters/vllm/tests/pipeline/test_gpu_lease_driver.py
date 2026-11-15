"""`verity_vllm.pipeline.gpu_lease` in this process, on stand-ins of what only a card has: the CUDA driver as `_Driver` and
`cuda_initialised` call it, nvidia-smi as `on_gpus` asks it, torch and vLLM's engine as `_shut_down` ends them, and the lease
holder as `_end_with` and `let_go` wait for it. Also `run_main`'s exit once the lease is released, the environment `acquire`
reads (`use_environ`), what `_InProcess` says it did, and `acquire` itself over a stand-in lease tool that passes it lock
descriptors over the socket it is given. `build-match`'s `test_gpu_lease.py` runs the lease in child processes over a fake
lease tool; no test here starts a process or needs a GPU."""
from __future__ import annotations

import ctypes
import io
import json
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.gpu_lease as GL

LIBCUDA = "libcuda.so.1"


@pytest.fixture(autouse=True)
def events_as_they_were() -> Iterator[None]:
    """The module's two process-wide events are set by what these tests run; each is put back as the test found it."""
    was = [(e, e.is_set()) for e in (GL._RELEASED, GL._EXITING)]
    yield
    for e, on in was:
        if on:
            e.set()
        else:
            e.clear()


class Lib:
    """libcuda as `_Driver` calls it: ``count`` devices, numbered from 10, those in ``on`` with an active primary context. ``fail``
    is the return code of a call by its name, or of the state query by the device's number."""

    def __init__(self, count: int, on: tuple[int, ...] = (), fail: dict[Any, int] | None = None) -> None:
        self.count, self.on, self.fail = count, set(on), dict(fail or {})
        self.resets: list[int] = []

    def cuDeviceGetCount(self, n: Any) -> int:
        n._obj.value = self.count
        return self.fail.get("cuDeviceGetCount", 0)

    def cuDeviceGet(self, d: Any, i: int) -> int:
        d._obj.value = 10 + i
        return self.fail.get("cuDeviceGet", 0)

    def cuDevicePrimaryCtxGetState(self, d: ctypes.c_int, flags: Any, on: Any) -> int:
        on._obj.value = int(d.value in self.on)
        return self.fail.get(d.value, 0)

    def cuDevicePrimaryCtxReset_v2(self, d: ctypes.c_int) -> int:
        self.resets.append(d.value)
        return self.fail.get("cuDevicePrimaryCtxReset_v2", 0)


def libcuda(monkeypatch: pytest.MonkeyPatch, lib: Lib | None) -> None:
    """`gpu_lease`'s ctypes, with libcuda as ``lib`` (None: no driver on the machine)."""
    def cdll(name: str) -> Lib:
        assert name == LIBCUDA
        if lib is None:
            raise OSError(f"{name}: cannot open shared object file")
        return lib

    monkeypatch.setattr(GL, "ctypes", SimpleNamespace(CDLL=cdll, c_int=ctypes.c_int, c_uint=ctypes.c_uint, byref=ctypes.byref))


@pytest.mark.parametrize(("lib", "initialised"), [(None, False), (Lib(2), True), (Lib(2, fail={"cuDeviceGetCount": 3}), False)])
def test_cuda_is_initialised_where_the_driver_answers_its_device_count(monkeypatch, lib, initialised):
    libcuda(monkeypatch, lib)
    assert GL.cuda_initialised() is initialised


def test_the_driver_numbers_the_process_s_devices_and_lists_each_with_an_active_context_or_no_answer(monkeypatch):
    lib = Lib(4, on=(11,), fail={12: 201})
    libcuda(monkeypatch, lib)
    drv = GL._Driver()
    assert [d.value for d in drv.devs] == [10, 11, 12, 13]
    assert drv.active() == [1, 2]
    drv.reset()
    assert lib.resets == [10, 11, 12, 13]
    assert GL.DRIVER is GL._Driver


@pytest.mark.parametrize(("fail", "said"), [("cuDeviceGetCount", "cuDeviceGetCount: CUDA driver error 100"),
                                            ("cuDeviceGet", "cuDeviceGet: CUDA driver error 100")])
def test_a_driver_that_cannot_number_the_devices_is_refused_by_the_call(monkeypatch, fail, said):
    libcuda(monkeypatch, Lib(2, fail={fail: 100}))
    with pytest.raises(RuntimeError, match=f"^{said}$"):
        GL._Driver()


def test_a_reset_the_driver_refuses_stops_at_the_first_device_and_names_the_call(monkeypatch):
    lib = Lib(3, fail={"cuDevicePrimaryCtxReset_v2": 999})
    libcuda(monkeypatch, lib)
    with pytest.raises(RuntimeError, match="^cuDevicePrimaryCtxReset: CUDA driver error 999$"):
        GL._Driver().reset()
    assert lib.resets == [10]


SMI = ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"]


@pytest.mark.parametrize(("answer", "on"), [
    ("GPU-b\n GPU-a \nGPU-z\n", ["GPU-a", "GPU-b"]), ("", []),
    (OSError("nvidia-smi: not found"), None), (subprocess.CalledProcessError(9, SMI), None), (subprocess.TimeoutExpired(SMI, 60), None)])
def test_the_leased_gpus_a_compute_process_is_on_are_nvidia_smi_s_in_the_lease_s_order_and_none_when_it_cannot_say(monkeypatch, answer, on):
    asked: list[tuple[list[str], dict[str, Any]]] = []

    def run(argv: list[str], **kw: Any) -> SimpleNamespace:
        asked.append((argv, kw))
        if isinstance(answer, BaseException):
            raise answer
        return SimpleNamespace(stdout=answer)

    monkeypatch.setattr(GL, "subprocess", SimpleNamespace(run=run, SubprocessError=subprocess.SubprocessError))
    assert GL.on_gpus(["GPU-a", "GPU-c", "GPU-b"]) == on
    assert asked == [(SMI, {"capture_output": True, "text": True, "timeout": 60, "check": True})]


def engine(calls: list[str], error: BaseException | None = None) -> SimpleNamespace:
    """vLLM's engine as `_shut_down` reaches its core, with torch's CUDA calls recorded beside it."""
    def shutdown() -> None:
        calls.append("shutdown")
        if error is not None:
            raise error

    return SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown))))


@pytest.fixture
def torch_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    cuda = SimpleNamespace(synchronize=lambda: calls.append("synchronize"), empty_cache=lambda: calls.append("empty_cache"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    return calls


def test_the_engine_s_teardown_syncs_shuts_its_core_down_and_empties_torch_s_cache(torch_calls):
    assert GL._shut_down(engine(torch_calls)) is None
    assert torch_calls == ["synchronize", "shutdown", "synchronize", "empty_cache"]


def test_a_teardown_that_fails_stops_there_and_gives_its_error_cut_to_300_characters(torch_calls):
    error = RuntimeError("x" * 400)
    assert GL._shut_down(engine(torch_calls, error)) == repr(error)[:300]
    assert torch_calls == ["synchronize", "shutdown"]


class Holder:
    """The lease tool's process: it ends when told (its stdin closed), or, unless ``ends``, only when killed."""

    def __init__(self, ends: bool = True, returncode: int = 0) -> None:
        self.ends, self.returncode = ends, returncode
        self.calls: list[Any] = []
        self.stdin = SimpleNamespace(close=lambda: self.calls.append("stdin closed"))

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        if timeout is not None and not self.ends:
            raise subprocess.TimeoutExpired("gpu-lease", timeout)
        return self.returncode

    def kill(self) -> None:
        self.calls.append("kill")


@pytest.mark.parametrize("exiting", [False, True])
def test_a_holder_that_ends_under_a_running_commit_kills_it_by_name_and_not_one_that_is_exiting(monkeypatch, capsys, exiting):
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(GL.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(GL, "_utc", lambda t: "T")
    GL._RELEASED.clear()
    if exiting:
        GL._EXITING.set()
    else:
        GL._EXITING.clear()
    holder = Holder(returncode=-15)
    GL._end_with(holder)
    assert holder.calls == [("wait", None)]
    said = "" if exiting else "T GPU lease ended under the Commit (holder rc -15): the Commit ends now\n"
    assert capsys.readouterr().err == said
    assert killed == ([] if exiting else [(os.getpid(), signal.SIGKILL)])


@pytest.mark.parametrize("ends", [True, False])
def test_letting_go_closes_the_locks_tells_the_holder_and_kills_one_that_does_not_end(monkeypatch, capsys, tmp_path, ends):
    locks = list(os.pipe())
    ours = SimpleNamespace(closed=False)
    ours.close = lambda: setattr(ours, "closed", True)
    holder, started = Holder(ends=ends), str(tmp_path / "gpu_lease.started")
    monkeypatch.setattr(GL, "_HELD", [(holder, ours, locks, started, {"GPU_LEASE_UUID": "GPU-5"})])
    monkeypatch.setattr(GL, "_no_device", lambda: None)
    monkeypatch.setattr(GL, "_utc", lambda t: "T")
    threads: list[tuple[Any, tuple[Any, ...]]] = []

    class Thread:
        def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            assert daemon
            threads.append((target, args))

        def start(self) -> None:
            pass

    monkeypatch.setattr(GL, "threading", SimpleNamespace(Thread=Thread))
    ends_said: list[tuple[str, dict[str, Any]]] = []
    tl = SimpleNamespace(end=lambda name, **k: ends_said.append((name, k)))
    drv = object()
    GL.let_go(tl, drv, after="work.pair0.instrumented", shutdown_error=None, spans={"shutdown_s": 1.5},
              done="engine shut down, CUDA context reset")
    assert holder.calls == ["stdin closed", ("wait", 60)] + ([] if ends else ["kill", ("wait", None)])
    for fd in locks:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert ours.closed and GL._HELD == [] and GL._RELEASED.is_set()
    assert threads == [(GL._watch, (drv, started))]
    note = json.loads(Path(tmp_path / "gpu_lease.released").read_text())
    assert (note["released"], note["released_at"], note["shutdown_error"], note["shutdown_s"]) == ("T", "work.pair0.instrumented", None, 1.5)
    ((name, said),) = ends_said
    assert name == "gpu_lease.release" and said["released"] is True and said["holder_s"] == note["holder_s"]
    assert capsys.readouterr().out == ("[delta] GPU LEASE released after work.pair0.instrumented: engine shut down, CUDA context reset, "
                                       "nothing on GPU-5; the checks that follow read host memory\n")


class Exited(BaseException):
    """`os._exit`, which ends the process without its teardown, stood in: it carries the status."""


@pytest.fixture
def released(monkeypatch: pytest.MonkeyPatch) -> None:
    def _exit(code: int) -> None:
        raise Exited(code)

    monkeypatch.setattr(GL.os, "_exit", _exit)
    GL._RELEASED.set()


def _ends(code: Any) -> Any:
    def main() -> Any:
        raise SystemExit(code)

    return main


@pytest.mark.usefixtures("released")
@pytest.mark.parametrize(("main", "status", "said"), [
    (lambda: 3, 3, ""), (lambda: None, 0, ""), (_ends(4), 4, ""), (_ends(None), 0, ""), (_ends(0), 0, ""),
    (_ends("GPU lease: refused by name"), 1, "GPU lease: refused by name\n")])
def test_a_released_commit_ends_with_its_command_s_status_and_says_a_message_it_exits_with(capsys, main, status, said):
    with pytest.raises(Exited) as end:
        GL.run_main(main)
    assert end.value.args == (status,)
    assert capsys.readouterr().err == said


@pytest.mark.usefixtures("released")
def test_a_released_commit_whose_command_fails_prints_the_failure_and_ends_with_1(capsys):
    def main() -> None:
        raise RuntimeError("the checks after the release failed")

    with pytest.raises(Exited) as end:
        GL.run_main(main)
    assert end.value.args == (1,)
    err = capsys.readouterr().err
    assert err.startswith("Traceback (most recent call last):\n") and err.endswith("RuntimeError: the checks after the release failed\n")


def test_a_commit_that_has_not_released_raises_its_command_s_failure_as_it_is():
    GL._RELEASED.clear()
    failure = RuntimeError("the run failed before the release")

    def main() -> None:
        raise failure

    with pytest.raises(RuntimeError) as raised:
        GL.run_main(main)
    assert raised.value is failure


def test_acquire_reads_the_environment_the_command_names_and_none_before(monkeypatch):
    monkeypatch.setattr(GL, "_ENVIRON", [])
    monkeypatch.setattr(GL, "cuda_initialised", lambda: True)
    assert GL.acquire() is None
    env = {"GPU_LEASE": "/l/gpu-lease", GL.STARTED: "/nowhere/gpu_lease.started"}
    GL.use_environ(env)
    assert len(GL._ENVIRON) == 1 and GL._ENVIRON[0] is env
    with pytest.raises(SystemExit, match="^GPU lease: CUDA was initialised before the lease"):
        GL.acquire()


@pytest.mark.parametrize(("err", "done"), [(None, "engine shut down, CUDA context reset"),
                                           ("RuntimeError('core')", "engine shut down (RuntimeError('core')), CUDA context reset")])
def test_the_in_process_holders_say_what_they_did_and_reset_with_the_process_s_driver(monkeypatch, err, done):
    monkeypatch.setattr(GL, "DRIVER", lambda: "the driver")
    held = GL._InProcess(None, [], None, {}, True)
    held.err = err
    assert held.done() == done and held.error() == err and held.driver() == "the driver" and held.record() == {}


class LeaseTool:
    """The lease tool's process as `acquire` starts it: given the socket's end it is passed, it sends one lock descriptor for
    each of ``locks`` (files under the test's directory, as a pool's lock files are) and prints ``line``, the leased devices
    as JSON. With no line it has no GPU to give, and passes none."""

    def __init__(self, line: bytes, locks: list[Path], rc: int = 75) -> None:
        self.line, self.locks, self.rc = line, locks, rc
        self.calls: list[str] = []
        self.asked: dict[str, Any] = {}

    def __call__(self, argv: list[str], *, stdin: int, stdout: int, pass_fds: tuple[int, ...], env: dict[str, str]) -> LeaseTool:
        assert stdin == stdout == subprocess.PIPE
        (end,) = pass_fds
        self.asked = {"argv": argv, "end": end, "env": env}
        if self.line:
            sock = socket.socket(fileno=os.dup(end))
            fds = [os.open(p, os.O_RDONLY) for p in self.locks]
            socket.send_fds(sock, [b"L"], fds)
            for fd in fds:
                os.close(fd)
            sock.close()
        self.stdout = io.BytesIO(self.line)
        return self

    def wait(self) -> int:
        self.calls.append("wait")
        return self.rc

    def kill(self) -> None:
        self.calls.append("kill")


@pytest.fixture
def leasing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[SimpleNamespace]:
    """What `acquire` runs besides the lease tool, stood in and recorded: CUDA not yet initialised, the time, nvidia-smi's
    answers (``on``, one a look; the last repeats), the exit hook and the holder's watch. Every descriptor of a lock file
    left open in this process is closed after the test."""
    seen = SimpleNamespace(on=[[]], looks=[], waits=[], atexit=[], threads=[], timeline=[], held=[])

    def on_gpus(leased: list[str]) -> list[str]:
        seen.looks.append(leased)
        return seen.on.pop(0) if len(seen.on) > 1 else seen.on[0]

    class Thread:
        def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            assert daemon
            seen.threads.append((target, args))

        def start(self) -> None:
            pass

    monkeypatch.setattr(GL, "cuda_initialised", lambda: False)
    monkeypatch.setattr(GL, "_utc", lambda t: "T")
    monkeypatch.setattr(GL, "on_gpus", on_gpus)
    monkeypatch.setattr(GL, "atexit", SimpleNamespace(register=seen.atexit.append))
    monkeypatch.setattr(GL, "threading", SimpleNamespace(Thread=Thread))
    monkeypatch.setattr(GL, "_HELD", seen.held)
    seen.tl = SimpleNamespace(start=lambda name: seen.timeline.append(("start", name, {})),
                              end=lambda name, **k: seen.timeline.append(("end", name, k)))
    seen.env = {"GPU_LEASE": "/l/gpu-lease", GL.STARTED: str(tmp_path / "gpu_lease.started"), GL.TOOL_START: "0", "KEPT": "k"}
    seen.locks = [tmp_path / f"{n}.lock" for n in (5, 6)]
    for p in seen.locks:
        p.write_text("")
    yield seen
    for _, ours, _, _, _ in seen.held:
        ours.close()
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}").startswith(str(tmp_path)):
                os.close(int(fd))
        except OSError:
            pass


GRANTED = b'{"CUDA_VISIBLE_DEVICES": "5", "GPU_LEASE_UUID": "GPU-5"}\n'


def _tool(monkeypatch: pytest.MonkeyPatch, tool: LeaseTool) -> LeaseTool:
    monkeypatch.setattr(GL, "subprocess", SimpleNamespace(Popen=tool, PIPE=subprocess.PIPE))
    return tool


@pytest.mark.parametrize("timed", [True, False])
def test_acquire_takes_the_lease_tool_s_locks_and_devices_and_holds_them_to_the_exit(monkeypatch, capsys, tmp_path, leasing, timed):
    tool = _tool(monkeypatch, LeaseTool(GRANTED, leasing.locks[:1]))
    env = leasing.env
    before = dict(env)
    GL.acquire(leasing.tl if timed else None, env, wait=leasing.waits.append)
    assert (tmp_path / "gpu_lease.requested").read_text() == "T\n"
    assert tool.asked["argv"] == GL.holder_argv(before, tool.asked["end"]) and tool.asked["env"] == GL.tool_env(before)
    assert GL.TOOL_START not in tool.asked["env"] and tool.asked["env"]["KEPT"] == "k"
    assert (env["CUDA_VISIBLE_DEVICES"], env["GPU_LEASE_UUID"]) == ("5", "GPU-5")
    ((holder, ours, fds, started, held_env),) = leasing.held
    assert holder is tool and started == env[GL.STARTED] and held_env is env
    assert len(fds) == 1 and os.path.samefile(f"/proc/self/fd/{fds[0]}", leasing.locks[0])
    assert ours.fileno() >= 0 and tool.calls == []
    assert leasing.looks == [["GPU-5"]] and leasing.waits == []
    assert leasing.atexit == [GL._EXITING.set] and leasing.threads == [(GL._end_with, (tool,))]
    assert leasing.timeline == ([("start", "gpu_lease.wait", {}), ("end", "gpu_lease.wait", {"devices": "5", "locks": 1})] if timed else [])
    assert capsys.readouterr().err == ""


def test_acquire_waits_for_another_process_to_leave_the_leased_gpus_and_says_so(monkeypatch, capsys, leasing):
    _tool(monkeypatch, LeaseTool(GRANTED, leasing.locks[:1]))
    leasing.on = [["GPU-5"], ["GPU-5"], []]
    GL.acquire(None, leasing.env, wait=leasing.waits.append)
    assert leasing.waits == [GL.CLEAR_EVERY_S] * 2 and len(leasing.held) == 1
    assert capsys.readouterr().err == "T GPU lease: another process left the leased GPU(s) after 2 look(s)\n"


def test_a_lease_whose_gpus_stay_occupied_is_given_back_and_refused_by_name(monkeypatch, leasing):
    tool = _tool(monkeypatch, LeaseTool(GRANTED, leasing.locks[:1]))
    leasing.on = [["GPU-5"]]
    with pytest.raises(SystemExit) as refusal:
        GL.acquire(leasing.tl, leasing.env, wait=leasing.waits.append)
    assert str(refusal.value) == (f"GPU lease: nvidia-smi still lists another process on the leased GPU(s) ['GPU-5'] after {GL.CLEAR_LOOKS} "
                                  f"look(s) {GL.CLEAR_EVERY_S:g} s apart (a cancelled Commit's engine workers outlive its lease): no engine "
                                  "can start there")
    assert tool.calls == ["kill"] and leasing.waits == [GL.CLEAR_EVERY_S] * GL.CLEAR_LOOKS
    assert leasing.held == [] and leasing.atexit == [] and leasing.threads == []


def test_a_lease_tool_that_ends_without_a_gpu_is_refused_with_its_code(monkeypatch, leasing):
    tool = _tool(monkeypatch, LeaseTool(b"", [], rc=75))
    with pytest.raises(SystemExit, match=r"^GPU lease: /l/gpu-lease ended \(rc 75\) without a GPU$"):
        GL.acquire(leasing.tl, leasing.env, wait=leasing.waits.append)
    assert tool.calls == ["wait"] and leasing.held == [] and leasing.looks == []
    assert leasing.timeline == [("start", "gpu_lease.wait", {})]


@pytest.mark.parametrize(("n", "passed"), [("2", 1), ("3", 2)])
def test_a_holder_that_passes_fewer_locks_than_the_task_s_gpus_is_killed_and_the_lease_refused(monkeypatch, leasing, n, passed):
    tool = _tool(monkeypatch, LeaseTool(GRANTED, leasing.locks[:passed]))
    leasing.env["GPU_LEASE_N"] = n
    with pytest.raises(SystemExit) as refusal:
        GL.acquire(None, leasing.env, wait=leasing.waits.append)
    assert str(refusal.value) == (f"GPU lease: the holder passed {passed} lock descriptor(s) for {n} GPU(s); the Commit would not hold "
                                  "its lease")
    assert tool.calls == ["kill"] and leasing.held == [] and leasing.looks == []
    assert "CUDA_VISIBLE_DEVICES" not in leasing.env
