"""serve-openai's windows sealed off the engine loop (`serve_seal.Sealer` over `serve_seal.freeze`), bounded on disk
(`serve_seal.DiskFloor`), and its records copied into the run's ``$OUT`` (`serve_commit.Mirror`). CPU only."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.commit.committer import pinned_pool as PP
from verity_vllm.commit.served_record import served_rows
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_seal as ss
from verity_vllm.pipeline.serve_admit import Admission, Delta

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("could not import 'torch'", allow_module_level=True)

from tests.pipeline.test_serve_openai import CERTIFIER_BODY, Committer, Engine, Runner, Serving, events, records

A = {0: [("model.embed_tokens", bytes(range(256)) * 2)], 1: [("runner.sampler", b"\x01\x02\x03\x04")]}
B = {0: [("model.embed_tokens", bytes(255 - i for i in range(256)) * 2)], 1: [("runner.sampler", b"\x09\x08\x07\x06")]}
ROWS = {0: [("runner-a", 0, 3, 0, 3), ("runner-b", 3, 5, 0, 2)], 1: [("runner-a", 0, 1, 3, 3)]}
NAMES = {"runner-a": "alice", "runner-b": "cmpl-2"}


@pytest.fixture
def host_blocks(monkeypatch):
    """The pinned pool's blocks as plain host tensors (no CUDA), and the bundle's seal time fixed."""
    from verity_vllm.pipeline import replay_bundle as RB

    def block(n: int, _exact: bool, _threads: int, _times: Any = None) -> tuple[Any, Any, str]:
        b = torch.zeros(n, dtype=torch.uint8)
        return b, b, "test"

    monkeypatch.setattr(PP, "_pinned_block", block)
    monkeypatch.setattr(RB.time, "strftime", lambda *_a: "2026-10-25T00:00:00Z")


def _commit(com: Any, steps: dict[int, list[tuple[str, bytes]]]) -> Any:
    """``steps`` committed as `opened.commit_steps` commits them, its retained blocks and levels then moved into slices of the
    committer's pinned pool, where the windowed collector's D2H copies put them."""
    from verity_vllm.commit.committer.native_host import TensorMeta

    C = com.chunk
    for s in sorted(steps):
        metas, buf = [], bytearray()
        for i, (name, raw) in enumerate(steps[s]):
            off, n = len(buf), len(raw)
            m = TensorMeta(i, name, "uint8", (n,), n, C, off // C, -(-(off + n) // C) - off // C)
            m.dev_off = off
            metas.append(m)
            buf += raw + b"\0" * ((-n) % 4)
        com.commit_block_offline(s, bytes(buf), metas)
    rc = com.finalize()
    for s, (blk, used) in list(com._gpu_blocks.items()):
        v = com.pool.take(blk.numel())
        v.copy_(blk)
        com._gpu_blocks[s] = (v, used)
        levels = []
        for lv in com._gpu_levels[s]:
            p = com.pool.take(lv.numel() * lv.element_size()).view(lv.dtype).view(lv.shape)
            p.copy_(lv)
            levels.append(p)
        com._gpu_levels[s] = levels
    return rc


def _window_a() -> tuple[Any, Any, dict[str, Any]]:
    from tests.check.opened import committer
    from verity_vllm.commit import step_rows

    com = committer(commit_scheme=step_rows.NAME)
    com.pool = PP.PinnedPool(1 << 14, exact=False, populate_threads=0)
    rc = _commit(com, A)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    leds = {"alice": sc.LiveRequest("cmpl-1", "e1", [1, 2, 3], {"params": {"max_tokens": 4}}, "alice", streamed=[7, 9]),
            "cmpl-2": sc.LiveRequest("cmpl-2", "e2", [4, 5], {"params": {"max_tokens": 2}}, None, streamed=[8])}
    reqs, rows = ss.window_requests(ROWS, NAMES.get, leds.get)
    kw = dict(rec={"window": 5, "run_id": "t/w5", "steps": len(rc.steps), "passed": True}, reqs=reqs, request_rows=rows,
              tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t", code="c0de",
              engine_of_record={"vllm": "test"})
    return com, rc, kw


def _files(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def _sealed_after_the_next_window(d: Path, view_of: Any) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any], list[str]]:
    """Window A sealed on the loop into ``d``, then sealed again into ``d`` from ``view_of(committer)`` taken at A's close, on
    the sealer's thread, once the committer has been reset and has committed window B (its values in the pool's blocks, its salt
    keys in the committer's key map): what the first seal wrote, what each returned, and the sealer's errors."""
    com, rc, kw = _window_a()
    inline = ss.seal_window(com, rc, d, **kw)
    on_loop = _files(d)
    shutil.rmtree(d)
    view, extras = view_of(com), ss.extras_of(com, rc)
    com.reset()
    _commit(com, B)
    got: dict[str, Any] = {}
    sealer = ss.Sealer(1 << 40)
    sealer.submit(lambda _waited: got.update(s=ss.seal_window(view, rc, d, **kw, extras=extras)), 1)
    sealer.close()
    return on_loop, inline, got.get("s"), sealer.errors


def test_a_window_sealed_off_the_loop_writes_the_bytes_an_inline_seal_writes(tmp_path, host_blocks):
    d = tmp_path / "windows" / "w5"
    on_loop, inline, off, errors = _sealed_after_the_next_window(d, ss.freeze)
    assert not errors and _files(d) == on_loop, "the same files, byte for byte: store, salts, levels, binding map, context, manifest"
    assert off["manifest_sha256"] == inline["manifest_sha256"] and off["binding_map_digest"] == inline["binding_map_digest"]
    assert off["bytes"] == inline["bytes"] and "error" not in off


def test_without_its_own_state_and_lent_blocks_the_next_window_changes_what_a_seal_writes(tmp_path, host_blocks):
    """The control: a plain copy of the committer at the close shares its salt-key map and gives its blocks back at the reset,
    so window B's keys and values land in what A's seal reads."""
    d = tmp_path / "windows" / "w5"
    on_loop, _inline, _off, errors = _sealed_after_the_next_window(d, copy.copy)
    got = _files(d)
    assert not errors and got.keys() == on_loop.keys() and got != on_loop
    assert any(got[k] != on_loop[k] for k in got if k.startswith("commit/replay_bundle_p0/store/"))


def test_lent_blocks_come_back_free_at_the_reset_after_their_restore(host_blocks):
    pool = PP.PinnedPool(1 << 10, exact=False, populate_threads=0)
    a = pool.take(100)
    a.fill_(7)
    lent = pool.lend()
    assert len(lent) == 1 and pool.in_use == [] and pool.cur is None
    pool.reset()
    b = pool.take(100)
    b.fill_(9)
    assert int(a[0]) == 7 and len(pool.all) == 2, "the next run's takes come from a block of its own"
    pool.restore(lent)
    assert lent[0] not in pool.free, "free only from the next reset"
    pool.reset()
    assert any(f is lent[0] for f in pool.free) and len(pool.free) == 2


def test_the_pool_is_warmed_before_serving_in_blocks_that_hold_any_one_take(monkeypatch):
    from verity_vllm.commit.committer import native_collect as NC

    got: list[list[int]] = []
    fake = SimpleNamespace(reserve=lambda sizes: got.append(sizes) or len(sizes) * ss.WARM_BLOCK)
    t = iter([1.0, 3.5])
    assert ss.warm_pool(SimpleNamespace(pool=fake), 2.5, clock=lambda: next(t)) == {"gb": 3.2, "blocks": 3, "seconds": 2.5}
    assert got == [[ss.WARM_BLOCK] * 3] and ss.WARM_BLOCK >= NC.WINDOW_MB << 20
    assert ss.warm_pool(SimpleNamespace(pool=fake), 0) is None and ss.warm_pool(SimpleNamespace(), 8) is None
    with pytest.raises(UsageError, match="--seal-warm-gb"):
        so.preflight(cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--seal-warm-gb", "-1"], environ={}))

    monkeypatch.setattr(ss, "WARM_BLOCK", 1 << 20)
    monkeypatch.setattr(PP, "_shared_map", lambda n: (torch.zeros(n, dtype=torch.uint8), None))
    monkeypatch.setattr(PP, "_populate", lambda *_a: True)
    monkeypatch.setattr(PP, "_register", lambda *_a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pool = PP.PinnedPool(1 << 20, exact=True, populate_threads=1)
    assert ss.warm_pool(SimpleNamespace(pool=pool), 3 * (1 << 20) / ss.GB)["blocks"] == 3 and pool.allocs == 3
    pool.reset()
    for n in (1 << 18, 1 << 19, 1 << 20, 1 << 18):
        pool.take(n)
    assert pool.allocs == 3 and len(pool.in_use) == 3, "the first window's takes are served from the warmed blocks"
    pool.mode = None


def test_a_close_waits_for_room_only_while_the_windows_awaiting_their_seal_hold_the_bound():
    first, go, done = threading.Event(), threading.Event(), threading.Event()
    sealer = ss.Sealer(15, clock=lambda: (go.set(), 0.0)[1] if first.is_set() else 0.0)
    held: list[int] = []
    sealer.submit(lambda _w: go.wait(), 10)
    first.set()

    def second() -> None:
        sealer.submit(lambda _w: done.wait(), 10)
        held.append(sealer.held)

    t = threading.Thread(target=second)
    t.start()
    t.join()
    assert held == [10], "the second window was queued once the first one's seal had ended"
    assert sealer.submit(lambda _w: None, 0) == 0.0, "a window with nothing to seal never waits"
    done.set()
    sealer.close()
    assert sealer.held == 0 and not sealer.errors


def test_a_failed_seal_is_its_windows_and_the_next_one_still_runs():
    ran: list[int] = []

    def bad(_w: float) -> None:
        raise RuntimeError("disk")

    sealer = ss.Sealer(1 << 20)
    sealer.submit(bad, 1)
    sealer.submit(lambda _w: ran.append(1), 1)
    sealer.close()
    assert ran == [1] and sealer.errors == ["RuntimeError: disk"]


# ---------------------------------------------------------------------------------------------------------------- serving

class Counted(Committer):
    """The stand-in committer with its absorbed steps' byte counts, as `serve_seal.committed_bytes` reads them."""

    def reset(self) -> None:
        super().reset()
        self._done: dict[int, Any] = {}

    def _commit(self, step: int, name: str, values: list[int], dtype: str) -> None:
        super()._commit(step, name, values, dtype)
        self._done[step] = SimpleNamespace(n_bytes=sum(m.nbytes for m in self._layouts[step]))


class Detached(Counted):
    """`Counted` whose windows close off the loop as `NativeHostCommitter.detach` closes them: each into a view of its own
    steps, rows and byte counts (the next window's reset rebinds the committer's, its first step the rows), whose
    ``wait_absorbed`` and ``finalize`` run on the sealer's thread. ``gates``: window -> the event its finalize waits on,
    once it has set ``entered[window]``."""

    def __init__(self, runner: Any, gates: dict[int, threading.Event] | None = None) -> None:
        self.window_views, self.detached, self.gates = False, 0, dict(gates or {})
        self.entered = {k: threading.Event() for k in self.gates}
        super().__init__(runner)

    def enable_window_views(self) -> None:
        self.window_views = True

    def detach(self) -> Any:
        view = copy.copy(self)
        view.sources = [copy.copy(s) for s in self.sources]
        view.window, view.staged_bytes = self.detached, ss.committed_bytes(self)
        self.detached += 1
        return view

    def wait_absorbed(self) -> float:
        return 0.0

    def finalize(self) -> Any:
        k = getattr(self, "window", None)
        if k in self.gates:
            self.entered[k].set()
            self.gates[k].wait()
        return super().finalize()


class Sealed(Serving):
    """`Serving` with every window sealed by ``seal`` (in place of `seal_window`), off the loop on ``sealer``; ``com`` makes
    its committer over the runner."""

    def __init__(self, out: Path, *, sealer: Any, mirror: Any = None, disk: Any = None, window_bytes: int | None = None,
                 window_steps: int = 64, com: Any = Counted) -> None:
        super().__init__(out, window_steps=window_steps)
        w = self.windows
        w.com = com(self.runner)
        w.seal, w.sealer, w.mirror, w.disk, w.window_bytes, w.on_floor = {"label": "t"}, sealer, mirror, disk, window_bytes, self.loop.drain


def _body(i: int, n: int = 3) -> dict[str, Any]:
    return dict(CERTIFIER_BODY, prompt=[128000, 10 + i, 20 + i], max_tokens=n, min_tokens=n)


def test_the_engine_serves_the_next_windows_while_a_seal_is_held(tmp_path, monkeypatch):
    held, seen = threading.Event(), []

    def seal(com: Any, rc: Any, d: Path, **kw: Any) -> dict[str, Any]:
        seen.append((kw["rec"]["window"], sorted(com._layouts), sorted({r[0] for rows in served_rows(com).values() for r in rows})))
        if kw["rec"]["window"] == 0:
            held.wait()
        return {"row_dir": str(d), "requests": [q["request_id"] for q in kw["reqs"]]}

    monkeypatch.setattr(sc, "seal_window", seal)
    sealer = ss.Sealer(1 << 40)
    out = tmp_path / "serve"
    out.mkdir()
    mirror = sc.Mirror(out, tmp_path / "OUT")
    with Sealed(out, sealer=sealer, mirror=mirror) as s:
        raws = [s.post("/v1/completions", _body(i))[1] for i in range(3)]
        assert all(events(r)[-1] == "[DONE]" for r in raws), "every request served while window 0's seal is held"
    wins = records(out, "windows.jsonl")
    assert [w["window"] for w in wins] == [0, 1, 2] and all(w["seal"] == {"row_dir": str(out / "windows" / f"w{w['window']}"),
                                                                      "off_loop": True} for w in wins)
    assert not (out / "seals.jsonl").exists() and len(records(tmp_path / "OUT", "requests.jsonl")) == 3
    assert not s.windows.summary()["passed"] and s.windows.summary()["seals"]["awaiting"] == 3
    held.set()
    sealer.close()
    seals = records(out, "seals.jsonl")
    assert [x["window"] for x in seals] == [0, 1, 2] and all(x["run_root"] == wins[0]["run_root"] for x in seals)
    for (w, steps, rows), win in zip(seen, wins, strict=True):
        assert w == win["window"] and steps == list(range(win["steps"])), "each seal reads its own window's steps"
        assert len(rows) == 1 and rows[0].startswith(win["requests"][0] + "-"), "and its own window's rows"
    reqs = {r["http_id"]: r for r in records(out, "requests.jsonl")}
    assert len(reqs) == 3 and all(r["check"]["passed"] for r in reqs.values())
    assert s.windows.summary()["passed"] and s.windows.summary()["seals"]["sealed"] == 3
    assert records(tmp_path / "OUT", "seals.jsonl") == seals and (tmp_path / "OUT" / "windows.jsonl").read_bytes() == (out / "windows.jsonl").read_bytes()


def test_while_a_windows_finalize_blocks_the_loop_opens_the_next_window_and_serves_a_request_there(tmp_path, monkeypatch):
    """The finalize of a window detached at its close runs on the sealer's thread: while window 0's is held, the loop has
    closed window 0, opened window 1 and served a request in it, and window 0's records (its line, its requests' checks)
    wait for its finalize; once it returns, both windows land in order and every request passes."""
    monkeypatch.setattr(sc, "seal_window", lambda com, rc, d, **kw: {"row_dir": str(d)})
    gate, marks = threading.Event(), []
    sealer = ss.Sealer(1 << 40)
    s = Sealed(tmp_path, sealer=sealer, com=lambda r: Detached(r, {0: gate}))
    s.windows.marks = lambda k, ev: marks.append((k, ev))
    with s:
        assert events(s.post("/v1/completions", _body(0))[1])[-1] == "[DONE]"
        s.windows.com.entered[0].wait()
        status, raw = s.post("/v1/completions", _body(1))
        assert status == 200 and events(raw)[-1] == "[DONE]", "served while window 0's finalize is held"
        assert marks[:3] == [(0, "open"), (0, "close"), (1, "open")] and s.windows.window == 1
        assert not (tmp_path / "windows.jsonl").exists() and not (tmp_path / "requests.jsonl").exists()
        held = s.windows.summary()
        assert 0 in held["closing"] and not held["passed"] and held["unchecked"]
        gate.set()
    sealer.close()
    wins = records(tmp_path, "windows.jsonl")
    assert [w["window"] for w in wins] == [0, 1] and all(w["passed"] and w["seal"]["off_loop"] for w in wins)
    assert all(list(w["close_phases_s"])[:4] == ["on_loop", "finalize_wait", "finalize", "readback"]
               and list(w["on_loop_phases_s"]) == ["detach", "uninstall", "ledgers", "seal_wait"]
               and set(w["seconds"]) == {"open", "close", "on_loop", "finalize_wait", "seal_wait", "queued"} for w in wins)
    reqs = records(tmp_path, "requests.jsonl")
    assert len(reqs) == 2 and all(r["check"]["passed"] for r in reqs) and [r["windows"] for r in reqs] == [[0], [1]]
    assert [x["window"] for x in records(tmp_path, "seals.jsonl")] == [0, 1]
    got = s.windows.summary()
    assert got["passed"] and got["closing"] == [] and got["finalize_on_loop"] is None


def test_a_full_sealer_holds_the_loop_at_the_close_and_the_window_says_how_long(tmp_path, monkeypatch):
    """The windows queued for their finalize and seal hold the sealer's bound: a close past it waits on the loop for room,
    and its line (``seconds.seal_wait``) and its seal's (``seal_wait_s``) say how long."""
    monkeypatch.setattr(sc, "seal_window", lambda com, rc, d, **kw: {"row_dir": str(d)})
    gate, waiting, calls = threading.Event(), threading.Event(), [0]

    def clock() -> float:
        calls[0] += 1
        if calls[0] == 3:
            waiting.set()                       # window 1's close asks for room while window 0 holds the bound
        return 5.0 if calls[0] >= 4 else 0.0

    sealer = ss.Sealer(1, clock=clock)
    s = Sealed(tmp_path, sealer=sealer, com=lambda r: Detached(r, {0: gate}))
    with s:
        for i in range(2):
            assert events(s.post("/v1/completions", _body(i))[1])[-1] == "[DONE]"
        waiting.wait()
        gate.set()
    sealer.close()
    w0, w1 = records(tmp_path, "windows.jsonl")
    assert (w0["seconds"]["seal_wait"], w1["seconds"]["seal_wait"]) == (0.0, 5.0)
    assert [x["seal_wait_s"] for x in records(tmp_path, "seals.jsonl")] == [0.0, 5.0] and s.windows.summary()["passed"]


def test_with_a_seal_process_each_execution_block_is_sealed_there_and_its_refusal_is_the_blocks_error(tmp_path, monkeypatch):
    """A sealer with a ``process`` (`seal_process.ProcessSealer`) hands each execution block's seal to it, never to
    `seal_window`, with ``written``, which gives the block's lent memory blocks back once its store's files are written; a seal
    the process refused is the block's error by the name the process gave it, and the next block is still sealed there."""
    from verity_vllm.pipeline.seal_process import SealFailed

    monkeypatch.setattr(sc, "seal_window", lambda *_a, **_k: pytest.fail("sealed on the sealer's thread"))
    got: list[tuple[int, bool]] = []

    class Process:
        def seal(self, com: Any, rc: Any, d: Path, *, written: Any = None, **kw: Any) -> dict[str, Any]:
            got.append((kw["rec"]["window"], written is not None))
            written()
            if kw["rec"]["window"] == 0:
                raise SealFailed("ValueError: the step-input port's positions aren't the ones the window's rows ran")
            return {"row_dir": str(d)}

    sealer = ss.Sealer(1 << 40)
    sealer.process = Process()
    s = Sealed(tmp_path, sealer=sealer, com=Detached)
    with s:
        for i in range(2):
            assert events(s.post("/v1/completions", _body(i))[1])[-1] == "[DONE]"
    sealer.close()
    seals = records(tmp_path, "seals.jsonl")
    assert got == [(0, True), (1, True)] and [x["window"] for x in seals] == [0, 1]
    assert seals[0]["error"] == "ValueError: the step-input port's positions aren't the ones the window's rows ran"
    assert "error" not in seals[1] and seals[1]["row_dir"] == str(tmp_path / "windows" / "w1")


def test_serve_openais_seals_run_where_its_sealing_options_say():
    """`seal_process.sealer_of`: under ``--seal`` a `ProcessSealer`, whose summary names its process and, once closed, how it
    ended; a `serve_seal.Sealer` sealing on its own thread with ``--no-seal-process``; none, the seals on the engine loop, with
    ``--no-seal-off-loop`` or without ``--seal``."""
    from verity_vllm.pipeline import seal_process as sp

    def sealer(*args: str) -> Any:
        return sp.sealer_of(cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--seal-hold-gb", "2", *args], environ={}))

    assert sealer() is None and sealer("--seal", "--no-seal-off-loop") is None
    plain = sealer("--seal", "--no-seal-process")
    assert type(plain) is ss.Sealer and plain.process is None and plain.hold_bytes == 2 * ss.GB
    plain.close()
    spawned = sealer("--seal")
    assert isinstance(spawned, sp.ProcessSealer) and spawned.summary()["process"] == {"pid": spawned.process.pid}
    spawned.close()
    assert spawned.summary()["process"] == {"pid": spawned.process.pid, "exitcode": 0} and not spawned.thread.is_alive()


def test_a_window_closed_off_the_loop_writes_the_records_one_closed_on_it_writes(tmp_path, monkeypatch):
    """The same requests served twice, every window finalized on the loop (a committer that detaches none, frozen for its
    seal) and off it (detached): windows.jsonl, seals.jsonl, requests.jsonl and step_trace.jsonl hold the same lines in the
    same order, timings aside, and each seal read its own window's steps, rows, requests and tokens."""
    def seal(com: Any, rc: Any, d: Path, **kw: Any) -> dict[str, Any]:
        return {"row_dir": str(d), "steps": sorted(com._layouts), "rows": served_rows(com), "tokens": kw["tokens"],
                "requests": [q["request_id"] for q in kw["reqs"]], "run_root": rc.run_root.hex()}

    monkeypatch.setattr(sc, "seal_window", seal)
    timing = {"windows.jsonl": ("seconds", "close_phases_s", "on_loop_phases_s"), "seals.jsonl": ("queued_s", "seal_wait_s"),
              "requests.jsonl": ("times",), "step_trace.jsonl": ()}
    got = {}
    for name, com in (("on", Counted), ("off", Detached)):
        out = tmp_path / name
        out.mkdir()
        ids = iter(range(1 << 20))
        monkeypatch.setattr(so, "new_http_id", lambda: f"cmpl-{next(ids):04x}")
        sealer = ss.Sealer(1 << 40)
        with Sealed(out, sealer=sealer, com=com, window_steps=2) as s:
            for i in range(3):
                assert events(s.post("/v1/completions", _body(i, n=3 + i))[1])[-1] == "[DONE]"
        sealer.close()
        assert s.windows.summary()["passed"] and (s.windows.finalize_on_loop is None) == (name == "off")
        got[name] = {f: [json.dumps({k: v for k, v in x.items() if k not in drop}).replace(str(out), "OUT") for x in records(out, f)]
                     for f, drop in timing.items()}
    assert len(got["on"]["windows.jsonl"]) > 3 and len(got["on"]["requests.jsonl"]) == 3
    assert got["off"] == got["on"]


def test_a_detached_windows_late_steps_land_in_its_view_and_its_end_hands_its_blocks_over_in_the_workers_order(host_blocks, monkeypatch):
    """`NativeHostCommitter.detach`: the worker commits each step into the view of the window it began in, so a step it
    absorbs after its window was detached and the next one reset lands in the closed window's view. The window's end,
    queued behind its last step, lends the window's pinned blocks to its view and resets the pool there, so the next
    window's takes never reuse them; the view's ``wait_absorbed`` returns only then."""
    from tests.check.opened import committer
    from verity_vllm.commit.committer import native_host as NH

    com = committer()
    com.pool = PP.PinnedPool(1 << 10, exact=False, populate_threads=0)
    com.enable_window_views()
    at_1, gate = threading.Event(), threading.Event()

    def commit_step(self: Any, job: Any) -> None:
        if job.seq == 1:
            at_1.set()
            gate.wait()
        v = self.pool.take(64)
        v.fill_(job.seq + 1)
        self._done[job.step] = SimpleNamespace(blk=v, n_bytes=64)

    def steps(n: int) -> None:
        for _ in range(n):
            com._q.put(NH._StepJob(com._step_index, [], None, com=com._window_view(), seq=com._seq))
            com._step_index += 1
            com._seq += 1

    monkeypatch.setattr(NH.NativeHostCommitter, "_commit_step", commit_step)
    worker = threading.Thread(target=com._run_worker, daemon=True)
    worker.start()
    com.reset()
    steps(2)
    a = com.detach()
    com.reset()
    at_1.wait()
    assert com.pool.in_use and not a._ended.is_set(), "window 1's reset leaves the pool to window 0's end"
    steps(1)
    b = com.detach()
    assert (a._step_index, b._step_index) == (2, 1)
    gate.set()
    a.wait_absorbed()
    b.wait_absorbed()
    assert sorted(a._done) == [0, 1] and sorted(b._done) == [0]
    assert [int(a._done[s].blk[0]) for s in (0, 1)] == [1, 2] and int(b._done[0].blk[0]) == 3
    assert a.lent and b.lent and not {id(x) for x in a.lent} & {id(x) for x in b.lent}, "each window's blocks its own"
    assert com._windows_detached == com._windows_ended == 2 and com.detach() is None
    ss.thaw(a)
    assert not a.lent and com.pool._returned
    com._q.put(com._stop)
    worker.join()
    with pytest.raises(ValueError, match="arena staging"):
        NH.NativeHostCommitter.enable_window_views(SimpleNamespace(staging="direct", inline_d2h=False))


def test_a_detached_windows_sources_keep_their_stats_and_rows_as_they_stood_at_the_detach():
    """`window_views.window_source`: the next window's steps rebind the live source's rows and its native counters run on,
    so the detached window's copy is synced once at the detach and then frozen (`stats_frozen`, honored by both native
    hidden taps' ``sync_stats``), its stats and rows its own."""
    from verity_vllm.commit.committer import native_hidden as NHID
    from verity_vllm.commit.committer.window_views import window_source

    class Src:
        stats_frozen = False

        def __init__(self) -> None:
            self.stats, self.rows, self.live = {}, {"runner-a": 1}, 1

        def sync_stats(self) -> None:
            if not self.stats_frozen:
                self.stats["launches"] = self.live

    src = Src()
    view = window_source(src)
    src.live, src.rows["runner-a"] = 2, 2
    src.sync_stats()
    view.sync_stats()
    assert (view.stats, view.rows, src.stats) == ({"launches": 1}, {"runner-a": 1}, {"launches": 2})
    for tap in (NHID.NativeHiddenM1Source, NHID.NativeHiddenM1SourceFA3):
        tap.sync_stats(SimpleNamespace(stats_frozen=True))          # no `_w`: a frozen copy never reads the live counters


def test_the_serving_thread_counts_on_a_plan_only_if_it_survives_the_window_ends_ahead_of_its_step():
    """`NativeCollectCommitter._plan_held`: each detached window the worker has yet to end retires a generation of its
    plans (`PlanCache.end_run`) before it absorbs the step ending now, so with one such end ahead only the current
    generation's plans are still there for the step, and with two none is."""
    from verity_vllm.commit.committer import native_collect as NC
    from verity_vllm.commit.committer.plan_cache import PlanCache

    cache = PlanCache()
    cache[("old", 1)] = "o"
    cache.end_run()
    cache[("new", 1)] = "n"

    def held(behind: int, kv: tuple) -> bool:
        return NC.NativeCollectCommitter._plan_held(SimpleNamespace(_windows_detached=3 + behind, _windows_ended=3, _metas_cache=cache), kv)

    assert held(0, ("old", 1)) and held(0, ("new", 1)) and not held(0, ("gone", 1))
    assert not held(1, ("old", 1)) and held(1, ("new", 1))
    assert not held(2, ("new", 1))
    cache.end_run()
    assert ("new", 1) in cache and ("old", 1) not in cache, "what one window end leaves"


def test_a_window_closes_once_its_steps_have_committed_window_bytes(tmp_path):
    s = Sealed(tmp_path, sealer=None, window_bytes=1)
    s.windows.seal = None
    with s:
        s.post("/v1/completions", _body(0, n=4))
    wins = [w for w in records(tmp_path, "windows.jsonl") if not w.get("empty")]
    assert [w["steps"] for w in wins] == [1, 1, 1, 1]


def test_a_close_says_where_its_time_went(tmp_path):
    ticks = iter(range(1 << 20))
    s = Sealed(tmp_path, sealer=None, window_bytes=1)
    s.windows.seal, s.windows.clock = None, lambda: float(next(ticks))
    with s:
        s.post("/v1/completions", _body(0, n=2))
    wins = [w for w in records(tmp_path, "windows.jsonl") if not w.get("empty")]
    assert wins and all(list(w["close_phases_s"]) == ["finalize", "readback", "record", "step_trace", "requests"]
                        and min(w["close_phases_s"].values()) >= 1 and sum(w["close_phases_s"].values()) == w["seconds"]["close"]
                        for w in wins), [(w["close_phases_s"], w["seconds"]) for w in wins]


def test_a_size_bound_crossed_inside_a_step_closes_its_window_at_the_step_boundary(tmp_path):
    """The window's bytes pass ``window_bytes`` between a step's input ids and its sampled ids: the window closes only before
    the next step, so each step's schedule (between its window's marks, as `advice.windows_of` reads them), its committed
    values and its rows sit in one window, which runs past its bound."""
    s = Sealed(tmp_path, sealer=None, window_bytes=25)
    w, engine = s.windows, s.loop.engine
    w.seal = None
    scheduled: list[tuple[int, list[str]]] = []
    commits: list[tuple[int, int, str, int]] = []
    marks: list[tuple[int, str, int]] = []
    step, commit = engine.step, w.com._commit

    def traced_step() -> list[Any]:
        scheduled.append((w.window, sorted(r["internal"] for r in engine.reqs.values())))
        return step()

    def traced_commit(st: int, name: str, values: list[int], dtype: str) -> None:
        commit(st, name, values, dtype)
        commits.append((w.window, st, name, ss.committed_bytes(w.com)))

    w.com._commit, engine.step, w.marks = traced_commit, traced_step, lambda k, ev: marks.append((k, ev, len(scheduled)))
    with s:
        status, _ = s.post("/v1/completions", _body(0, n=3))
    assert status == 200
    names = [name for k, st, name, _b in commits if (k, st) == (0, 0)]
    crossed = next((i for i, (k, st, _n, b) in enumerate(commits) if (k, st) == (0, 0) and b >= 25), None)
    assert crossed is not None and 0 < crossed < len(names) - 1, "window 0 passes its bound part way through step 0's commits"
    opened = {k: n for k, ev, n in marks if ev == "open"}
    closed = {k: n for k, ev, n in marks if ev == "close"}
    traced = records(tmp_path, "step_trace.jsonl")
    wins = [x for x in records(tmp_path, "windows.jsonl") if not x.get("empty")]
    assert [x["steps"] for x in wins] == [1, 1, 1] and all(x["committed_bytes"] > 25 for x in wins), "each window runs past its bound"
    for x in wins:
        k, steps = x["window"], range(opened[x["window"]], closed[x["window"]])
        assert all(scheduled[j][0] == k for j in steps), "each step scheduled while its window was open"
        assert [scheduled[j][1] for j in steps] == [sorted(r[0] for r in t["rows"]) for t in traced if t["window"] == k]
        for j in steps:
            assert [name for kk, st, name, _b in commits if (kk, st) == (k, j - opened[k])] == names, "all of its values in its window"
    assert {(k, st) for k, st, _n, _b in commits} == {(x["window"], 0) for x in wins}


def test_below_the_disk_floor_a_window_is_not_sealed_and_the_server_stops_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "seal_window", lambda *_a, **_k: pytest.fail("a window under the floor is sealed"))
    disk = ss.DiskFloor(tmp_path, 1000, usage=lambda _p: SimpleNamespace(free=1001))
    sealer = ss.Sealer(1 << 40)
    s = Sealed(tmp_path, sealer=sealer, disk=disk)
    with s:
        status, raw = s.post("/v1/completions", _body(0))
        s.engine_thread.join()
        assert status == 200 and events(raw)[-1] == "[DONE]", "the request in flight is served whole"
    sealer.close()
    w0 = records(tmp_path, "windows.jsonl")[0]
    assert "not sealed" in w0["seal"]["error"] and w0["seal"]["disk"]["floor_gb"] == 0.0 and not w0["passed"]
    assert s.loop.draining == w0["seal"]["error"] and s.windows.floor_hit["window"] == 0
    assert not s.windows.summary()["passed"] and records(tmp_path, "requests.jsonl")[0]["check"]["passed"]


def test_a_draining_loop_refuses_new_requests_and_ends_once_idle():
    got: list[Delta] = []
    loop = so.EngineLoop(Engine(Runner()))
    loop.drain("the disk floor")
    loop.submit(Admission("cmpl-1", "cmpl-1-0", [1, 2], {"max_tokens": 2, "seed": 0}, got.append, {}))
    loop.run()
    assert [d.error for d in got] == ["the endpoint is stopping: the disk floor"] and loop.steps == 0


def test_the_disk_floor_refuses_to_start_below_one_window_beyond_it(tmp_path):
    (man := tmp_path / "manifest.json").write_text("{}")
    base = ["--model", "m", "--revision", "r", "--out", str(tmp_path / "new" / "dir"), "--seal", "--window-gb", "10", "--disk-floor-gb", "300",
            "--required-manifest", str(man)]
    opts = cli.parse("serve-openai", base, environ={})
    with pytest.raises(UsageError, match="disk floor .*--disk-floor-gb"):
        so.preflight(opts, which=lambda _: "/v/bin/ninja", usage=lambda _p: SimpleNamespace(free=309 * ss.GB))
    so.preflight(opts, which=lambda _: "/v/bin/ninja", usage=lambda _p: SimpleNamespace(free=310 * ss.GB))
    assert so.disk_floor_of(opts).path == tmp_path.resolve(), "the nearest directory --out is on"
    no_floor = cli.parse("serve-openai", [*base, "--disk-floor-gb", "0"], environ={})
    assert so.disk_floor_of(no_floor) is None
    assert ss.DiskFloor(tmp_path, 100, usage=lambda _p: SimpleNamespace(free=160)).short(50, pending=10) is None
    assert ss.DiskFloor(tmp_path, 100, usage=lambda _p: SimpleNamespace(free=160)).short(50, pending=11)["free_gb"] == 0.0


def test_the_records_mirror_copies_whole_lines_and_a_replaced_file_again(tmp_path):
    src, dst = tmp_path / "serve", tmp_path / "OUT"
    src.mkdir()
    (src / "requests.jsonl").write_bytes(b'{"a": 1}\n{"b": ')
    (src / "serve.json").write_text("{}")
    m = sc.Mirror(src, dst)
    m()
    assert (dst / "requests.jsonl").read_bytes() == b'{"a": 1}\n' and (dst / "serve.json").read_text() == "{}"
    with open(src / "requests.jsonl", "ab") as f:
        f.write(b'2}\n{"c": 3}\n')
    m()
    assert (dst / "requests.jsonl").read_bytes() == (src / "requests.jsonl").read_bytes()
    (src / "requests.jsonl").write_bytes(b'{"d": 4}\n')
    m()
    assert (dst / "requests.jsonl").read_bytes() == b'{"d": 4}\n' and not m.errors
    assert sc.records_mirror(src, str(src)) is None and sc.records_mirror(src, None) is None
    assert json.loads((dst / "serve.json").read_text()) == {} and not (dst / "windows.jsonl").exists()


def test_the_records_mirror_drains_the_step_traces_writer_before_it_copies(tmp_path):
    """serve-openai's step trace is written by a thread of its own (`advice_trace`): its lines so far reach the file only at a
    `drain`, which the mirror calls first."""
    src, dst = tmp_path / "serve", tmp_path / "OUT"
    src.mkdir()
    (src / "advice_trace.jsonl").write_bytes(b'{"kind": "header"}\n{"kind": "st')

    def drain() -> None:
        with open(src / "advice_trace.jsonl", "ab") as f:
            f.write(b'ep"}\n')

    m = sc.records_mirror(src, str(dst), before=drain)
    m()
    assert (dst / "advice_trace.jsonl").read_bytes() == b'{"kind": "header"}\n{"kind": "step"}\n' and not m.errors


def test_a_seal_writes_its_binding_map_in_pieces_to_the_bytes_and_digest_one_dumps_writes(tmp_path, host_blocks, monkeypatch):
    from verity_vllm.commit import binding as B, json_chunks as JC

    built: list[dict[str, Any]] = []
    real = B.build_binding_map

    def build(*a: Any, **k: Any) -> dict[str, Any]:
        m = real(*a, **k)
        built.append(copy.deepcopy(m))
        return m

    monkeypatch.setattr(B, "build_binding_map", build)
    monkeypatch.setattr(JC, "CHUNK", 1)
    com, rc, kw = _window_a()
    got = ss.seal_window(com, rc, tmp_path / "w5", **kw)
    (bmap,) = built
    assert bmap["entries_n"] > 1, "more than one piece"
    one = json.dumps({k: v for k, v in bmap.items() if k != "entries"} | {"entries": bmap["entries"]}, separators=(",", ":"), default=str)
    assert (tmp_path / "w5" / "commit" / "binding_map_p0.json").read_text() == one
    canon = json.dumps(bmap["entries"], sort_keys=True, separators=(",", ":"), default=str).encode()
    assert got["binding_map_digest"] == bmap["digest"] == hashlib.sha256(canon).hexdigest()
    assert got["binding_map_entries"] == bmap["entries_n"] and set(got["phases_s"]) == {"binding_map", "store", "bundle"}


def test_no_collection_runs_while_the_map_is_built_and_the_collector_comes_back_as_it_was():
    was = gc.isenabled()
    try:
        gc.enable()
        with ss.gc_paused():
            assert not gc.isenabled()
        assert gc.isenabled()
        gc.disable()
        with ss.gc_paused():
            pass
        assert not gc.isenabled()
    finally:
        (gc.enable if was else gc.disable)()
    items = list(range(10_001))
    ss.drop(items, chunk=4096)
    assert items == []


def test_the_watchdog_records_each_wake_late_by_its_floor_or_more(tmp_path):
    t = [0.0]
    late = iter([0.0, 0.5, 0.1, 2.0])
    wd = ss.Watchdog(tmp_path / "pauses.jsonl", period_s=0.05, min_s=0.2, clock=lambda: t[0], wall=lambda: 1000.0 + t[0])

    def sleep(s: float) -> None:
        d = next(late, None)
        if d is None:
            wd.stopped.set()
        t[0] += s + (d or 0.0)

    wd.sleep = sleep
    wd.run()
    lines = [json.loads(x) for x in (tmp_path / "pauses.jsonl").read_text().splitlines()]
    assert [x["late_s"] for x in lines] == [0.5, 2.0] and lines[1]["t_end"] == round(1000.0 + 4 * 0.05 + 0.5 + 0.1 + 2.0, 3)
    assert wd.summary() == {"period_s": 0.05, "min_s": 0.2, "pauses": 2, "late_s": {"max": 2.0, "total": 2.5}}
    wd.stop()


def test_a_windows_record_carries_its_collections_and_names_the_long_ones():
    rc = SimpleNamespace(extra={"gc": {"pauses": [(0, 1.0, 0.002, "MainThread"), (2, 5.0, 1.25, "serve-seal")], "freeze_s": [0.3]}})
    assert sc.gc_of(rc) == {"n": 2, "total_s": 1.252, "max_s": 1.25, "long": [[2, 5.0, 1.25, "serve-seal"]], "freeze_s": [0.3]}
    assert sc.gc_of(SimpleNamespace(extra={})) is None and sc.gc_of(None) is None
