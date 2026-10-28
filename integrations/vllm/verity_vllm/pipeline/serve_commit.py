"""The Commit side of `verity_vllm.pipeline.serve_openai`: the committer's windows over the serving engine, the map from
each HTTP request to its engine request, and each request's record held to what was served.

A window (`CommitWindows`) opens at the first engine step after the engine was idle and closes when the engine is idle
again (once it holds ``idle_close_steps`` steps), or after ``window_steps`` steps; each is one Commit, with its own run
root. The record is the served path's and COMMIT-ARB's one record, `commit.served_record`: the committer carries a
`served_record.SamplerInputsSource`, which binds each sampled row's seed and position beside the sampled ids and keeps
the step's rows, and at each close `served_record.Record.absorb` reads the window back from the committed bytes into one
`RequestRecord` per engine request. A finished request is checked by `Record.check` against what was streamed for it,
the seed it carried into the engine (the client's, or the served default, `engine.served_seed`) and how its stream
ended; one its client aborted ends at the k ids streamed before the abort, its finish ``abort`` with ``k``, and passes when
those are the committed ids' prefix. The finish says who ended it: ``by`` ``client``, or ``server`` for a request still
running when the server stopped or failed. Its ingress record, the sampling parameters it brought, is written beside it, with the body's ``user`` (null when it
sent none). Each committed step gets a line of ``step_trace.jsonl`` at its window's close: its root, when its engine step ended,
its rows and the committer's trace. ``marks`` hears each window's open and close (the scheduler step trace's window records,
`engine.step_trace.StepTrace.window`).
With ``seal`` each closed window is also sealed as a deferred Commit's row dir (`serve_seal.seal_window`); with a ``sealer``
(`serve_seal.Sealer`) off the engine loop. There the close itself runs off the loop too: the loop ends the window's last
step, detaches the window into a view of the committer (`NativeHostCommitter.detach`), takes the ledgers as they stand and
opens the next window, and the window's finalize, its readback, its records, its requests' checks and then its seal run on
the sealer's thread in window order (`CommitWindows._finish`); a committer that detaches no window is finalized at the
close and frozen for its seal (`serve_seal.freeze`). The window's line in ``windows.jsonl``, written once its finalize has
run and before its seal, names its row dir (``seal``); what its seal wrote is its line in ``seals.jsonl``, once written,
in window order; a sealer with a ``process`` (`seal_process.ProcessSealer`) writes only each seal's store files on its thread,
and the rest of the seal runs in that spawned process. With ``release`` (`serve_release.Draws`, ``--retain device
--release-after``) a window's seal, on either path, first waits for its draw-gated release, which keeps only the leaves its
subprotocols' drawn blocks open, and its ``seals.jsonl`` line carries that release's record (``release``). Each line says
where its close's time went: ``seconds`` (``close``, and off the loop ``on_loop``, the loop's part, ``finalize_wait``, for the
worker to absorb the window's steps, and ``seal_wait``, for the sealer's room) and ``close_phases_s``. ``window_bytes``
closes a window once its steps have committed that many
bytes, which bounds its bundle. Every close is between engine steps, so a step's schedule, its committed values and its rows
sit in one window, which runs past its bound by the step that crossed it (and by the steps the committer's worker had not
yet absorbed); a request may still span windows. ``disk`` (`serve_seal.DiskFloor`) refuses the seal of a window that would take the
filesystem under its floor, and ``on_floor`` then hears why (the server stops). ``mirror`` (`Mirror`) copies the records
into the run's ``$OUT`` after each close and each seal.

With ``served`` (the compile slice's statics, `served_statics`) each close also commits the step rows the window ran
(`verity.core.protocols.certifier.served_rows.register`, under the window's run root): each request named by its arrival
id, its index in the order the engine admitted the run's requests. The record's public part (the registration's
binding, root, count and row width) goes into the window's ``windows.jsonl`` line, beside its run root, written at the
close and so before any draw over it; the rows and their salts go to ``served_rows/w<k>.json``, the prover's.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier import served_rows as SRW
from verity_vllm.commit.served_record import Record, base_id, served_rows, window_commit
from verity_vllm.pipeline.seal_process import SealFailed
from verity_vllm.pipeline.serve_seal import GB, committed_bytes, extras_of, freeze, seal_window, thaw, window_requests

#: The compile slice's size class beside the engine's statics; R and P (with the engine's L and T, and IW) shape the rows.
SERVED_CLASS = ("R", "W", "A", "IW", "Q", "D", "P", "K")

#: The request fields that are sampling inputs, recorded as each request brought them.
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "min_tokens", "ignore_eos", "stop",
                   "stop_token_ids", "presence_penalty", "frequency_penalty", "repetition_penalty")


def user_of(body: Any) -> str | None:
    """A request body's ``user`` (a dict, or vLLM's ``CompletionRequest``), None when it sent none."""
    u = body.get("user") if isinstance(body, Mapping) else getattr(body, "user", None)
    return None if u is None else str(u)


def advice_trace(out: Path, engine: Any) -> Any:
    """The scheduler step trace (`engine.step_trace.StepTrace`) of ``engine`` for the server's lifetime, at ``out/advice_trace.jsonl``
    (`step_trace.SERVED`, beside the committer's step_trace.jsonl), under the engine ids requests.jsonl records (vLLM's internal
    suffix dropped): what `verity-vllm advice OUT` reads, with each window's open and close (``CommitWindows(marks=trace.window)``).
    Its writer thread writes the file (`StepTrace.writer`): the engine thread never waits on the serve dir's disk, which the seals
    keep busy; `Mirror(before=trace.drain)` copies it whole."""
    from verity_vllm.engine.engine_driver import external_id
    from verity_vllm.engine.step_trace import SERVED, StepTrace

    return StepTrace(Path(out) / SERVED, request_id=external_id, clock=time.time, writer=True).install(engine)


#: requests.jsonl's engine id to ``user`` map alone, a line each as requests.jsonl's: ``{"engine_id", "user"}``.  It holds no
#: request value (the network certifier's users are request indexes, ``user-<i>``), so it can leave the node with
#: advice_trace.jsonl, windows.jsonl and serve.json while requests.jsonl, with the prompt and output ids, stays.
REQUEST_USERS = "request_users.jsonl"

#: the records the gates' tie-run reads by run id, mirrored into the run's $OUT (the served rows' salts stay in the serve dir)
MIRRORED = ("serve.json", "requests.jsonl", REQUEST_USERS, "windows.jsonl", "seals.jsonl", "step_trace.jsonl",
            "advice_trace.jsonl", "pauses.jsonl", "summary.json", "remote_session.json", "memory_map.jsonl")


def gc_of(rc: Any) -> dict[str, Any] | None:
    """A window's Python collections (the committer's `GcPauses` record, while it was installed): how many, their total and
    longest, and each of 50 ms or more as (generation, start on `time.perf_counter`, seconds, thread)."""
    g = (getattr(rc, "extra", None) or {}).get("gc")
    if not g:
        return None
    p = list(g.get("pauses") or [])
    return {"n": len(p), "total_s": round(sum(x[2] for x in p), 4), "max_s": round(max((x[2] for x in p), default=0.0), 4),
            "long": [[x[0], round(x[1], 3), round(x[2], 3), x[3]] for x in p if x[2] >= 0.05], "freeze_s": g.get("freeze_s")}


class Mirror:
    """The serve dir's records (`MIRRORED`) copied into ``dst`` (the run's ``$OUT``, which custody uploads), so they outlive the serve
    dir: a JSON-lines file up to its last whole line, appended from where its copy ends; any other file whole. Any thread.
    ``before``: called first, each time (the step trace's `drain`, so its writer's lines are in the file)."""

    def __init__(self, src: str | Path, dst: str | Path, names: Sequence[str] = MIRRORED, *,
                 before: Callable[[], None] | None = None) -> None:
        self.src, self.dst, self.names, self.before = Path(src), Path(dst), tuple(names), before
        self.errors: list[str] = []
        self._lock = threading.Lock()

    def __call__(self) -> None:
        if self.before is not None:
            self.before()
        with self._lock:
            self.dst.mkdir(parents=True, exist_ok=True)
            for name in self.names:
                try:
                    self._one(self.src / name, self.dst / name)
                except OSError as e:
                    self.errors.append(f"{name}: {type(e).__name__}: {e}"[:300])

    @staticmethod
    def _one(s: Path, d: Path) -> None:
        if not s.is_file():
            return
        if s.suffix != ".jsonl":
            part = d.with_name(d.name + ".part")
            shutil.copyfile(s, part)
            os.replace(part, d)
            return
        have = d.stat().st_size if d.is_file() else 0
        with open(s, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            have = have if have <= size else 0          # the serve dir's file is not the one copied so far: copy it again whole
            f.seek(have)
            data = f.read(size - have)
        data = data[:data.rfind(b"\n") + 1]
        if data or not have:
            with open(d, "ab" if have else "wb") as g:
                g.write(data)


def records_mirror(out: Path, dst: str | None, before: Callable[[], None] | None = None) -> Mirror | None:
    """`Mirror` of ``out``'s records into ``--copy-records-to`` (None without one, or when it is ``out``)."""
    if not dst or Path(dst).resolve() == Path(out).resolve():
        return None
    return Mirror(out, dst, before=before)


def disk_record(windows: CommitWindows | None) -> dict[str, Any] | None:
    """serve.json's disk floor: the filesystem it watches, its floor and what was free at the start."""
    d = windows.disk if windows is not None else None
    return None if d is None else {"path": str(d.path), "floor_gb": d.floor / GB, "free_gb_at_start": round(d.free() / GB, 1)}


def param_of(params: Any, name: str) -> Any:
    """One field of the engine's sampling parameters: vLLM's ``SamplingParams`` or `PlainProtocol`'s dict."""
    return params.get(name) if isinstance(params, Mapping) else getattr(params, name, None)


def ingress_of(brought: Mapping[str, Any], params: Any) -> dict[str, Any]:
    """A request's ingress record: the sampling fields it sent, the engine's parameters they make (``params``, seeded),
    and the seed with its rule (``request``: the client's; ``default``: `engine.served_seed.DEFAULT_SEED`)."""
    made = {k: param_of(params, k) for k in SAMPLING_FIELDS}
    return {"brought": dict(brought), "params": made, "seed": made["seed"],
            "seed_rule": "request" if brought.get("seed") is not None else "default"}


def served_statics(engine: Mapping[str, Any], size_class: Mapping[str, int]) -> dict[str, int]:
    """The compile slice's statics (`compile_slice.statics`) for the served engine's scheduler configuration
    (`engine.step_trace.scheduler_config`) and the slice's size class (every name of `SERVED_CLASS`)."""
    if missing := [k for k in SERVED_CLASS if k not in size_class]:
        raise ValueError(f"the served rows' size class needs {missing}")
    c = {k: int(size_class[k]) for k in SERVED_CLASS}
    return CS.statics(c["R"], c["W"], c["A"], dict(engine), c["IW"], Q=c["Q"], D=c["D"], P=c["P"], K=c["K"])


def commit_served(statics: Mapping[str, int], by_step: Mapping[int, Sequence[tuple]], ids: Mapping[str, int], steps: int,
                  run_root: bytes, salts: Sequence[bytes] | None = None) -> tuple[SRW.Served, list]:
    """A window's served rows (`served_record.served_rows`' form, by step) committed under its run root
    (`served_rows.register`, fresh salts unless ``salts`` gives them), and the slice windows they make."""
    S = SimpleNamespace(**statics)
    session = SRW.windows(S, by_step, ids, steps)
    return SRW.register(S, session, run_root, salts=salts), session


@dataclass
class Ledger:
    """One request as served: its ids, the HTTP id -> engine id map, its ingress record and its stream."""

    http_id: str
    engine_id: str
    prompt_ids: list[int]
    ingress: dict[str, Any] = field(default_factory=dict)
    user: str | None = None                      # the body's `user`, the id the advice names the request by
    release: Any = None                          # (the network certifier's run, its place in the release order), under --admit release-order
    admitted: int = -1                           # its place in the order requests were added to the engine
    streamed: list[int] = field(default_factory=list)
    finish: dict[str, Any] | None = None
    finished: bool = False
    times: dict[str, float] = field(default_factory=dict)   # wall-clock stamps (telemetry): head, received, submitted, admitted, first_token


@dataclass
class _Closed:
    """A closed window as the rest of its close reads it, wherever that runs: its committer (on the loop) or the view it was
    detached into (off it; None for a window with no step), when its steps ended, every ledger as it stood at the close (a
    copy of each unfinished one's stream, off the loop) and the ledgers finished by then, which its readback checks."""

    window: int
    com: Any
    step_t: dict[int, float]
    ledgers: Mapping[str, Ledger]
    finished: list[Ledger]
    open_s: float
    off_loop: bool = False
    on_loop: dict[str, float] = field(default_factory=dict)    # off the loop: the close's phases on the loop
    t_close: float = 0.0                                       # ... when its job was queued
    seal_wait: float = 0.0                                     # ... and how long the loop waited for the sealer's room


def _phases(laps: Sequence[tuple[str, float]]) -> dict[str, float]:
    return {name: round(t - s, 4) for (_, s), (name, t) in zip(laps, laps[1:])}


def _lap(laps: list[tuple[str, float]], clock: Callable[[], float], name: str) -> None:
    """The end of a close's phase ``name``, marked in ``laps`` (`_phases` reads them)."""
    laps.append((name, clock()))


class CommitWindows:
    """The committer's windows over a serving engine, and every request's `Ledger`. ``committer`` is a
    `NativeHostCommitter` with a `served_record.SamplerInputsSource` attached (or its offline stand-in in the tests:
    ``reset``, ``install``, ``uninstall``, ``finalize``, ``_layouts``, ``_step_index``, ``sources``, and to close off the loop
    ``enable_window_views`` and ``detach``, whose view has ``wait_absorbed`` and ``finalize``); ``eos_ids`` and
    ``decode`` are the tokenizer's, for the finish check (`served_record.finish_problems`). ``arm`` (`serve_h1.H1Windows`)
    opens on each window once its committer is installed and closes first at its close, before the committer comes off,
    its hooks being on top of the committer's; what its ``close`` returns is the window line's ``pouw_h1``, and when that
    is a `Future` (its release off the loop) the line is written once it resolves, the loop and the sealer going on."""

    def __init__(self, committer: Any, out: Path, *, run_id: str, window_steps: int, idle_close_steps: int = 1,
                 eos_ids: Sequence[int] = (), decode: Callable[[Sequence[int]], str] | None = None,
                 clock: Callable[[], float] = time.perf_counter, wall: Callable[[], float] = time.time,
                 marks: Callable[[int, str], None] | None = None, seal: Mapping[str, Any] | None = None,
                 served: Mapping[str, int] | None = None, sealer: Any = None, window_bytes: int | None = None, disk: Any = None,
                 on_floor: Callable[[str], None] | None = None, mirror: Callable[[], None] | None = None,
                 request_cap: int = 0, arm: Any = None, release: Any = None) -> None:
        self.com, self.out, self.clock, self.wall, self.marks = committer, Path(out), clock, wall, marks
        self.release = release                           # `serve_release.Draws`: each window's draw-gated release, before its seal
        self.arm, self._arm_out = arm, {}                # each window's arm outcome, until its line is written
        self.arm_failed: list[int] = []                  # the windows whose arm failed to open or close
        self._arm_pending: set[int] = set()              # the windows whose arm is still closing off the loop
        self._awaiting: dict[int, dict[str, Any]] = {}   # ... and their lines, held until it ends
        self.request_cap = int(request_cap)              # --window-requests: at most this many distinct requests a window (0: any)
        self.in_window: set[str] = set()                 # the engine ids live at any step of the open window
        self.seal = dict(seal) if seal is not None else None     # `seal_window`'s label, code and engine_of_record: seal every window
        self.served = dict(served) if served is not None else None   # the compile slice's statics: commit every window's rows
        self.arrivals: dict[str, int] = {}               # every admitted engine id -> its arrival id, in admission order
        self.sealer, self.window_bytes, self.disk, self.on_floor, self.mirror = sealer, window_bytes, disk, on_floor, mirror
        self.floor_hit: dict[str, Any] | None = None     # the first window the disk floor left unsealed
        self.run_id, self.window_steps, self.idle_close_steps = run_id, int(window_steps), int(idle_close_steps)
        self.eos_ids, self.decode = [int(x) for x in eos_ids], decode
        self.known: dict[str, str] = {}                  # every admitted engine id -> its HTTP id; rows past a finish still map
        self.names: dict[str, str] = {}                  # every admitted engine id -> its name in a sealed window (`user`, else HTTP id)
        self.record = Record(known=self.known)
        self.window, self.open_s, self.is_open = -1, 0.0, False
        self.step_t: dict[int, float] = {}               # the open window's committed steps -> when their engine step ended
        self.ledgers: dict[str, Ledger] = {}
        self.records: list[dict[str, Any]] = []
        self.seals: list[dict[str, Any]] = []            # seals.jsonl's lines, appended on the sealer's thread
        self.checked: list[dict[str, Any]] = []
        self.off_loop = False                            # the open window's close finalizes off the loop (`open`)
        self.finalize_on_loop: str | None = None         # why a sealer's windows still finalize on the loop (the committer's refusal)
        self.closing: dict[int, list[str]] = {}          # windows closed whose finalize has not run: the engine ids each checks
        self._lock = threading.Lock()                    # records, seals, checked and closing, which the sealer's thread writes
        self._extras: dict[Any, Any] = {}                # the registered tensors' bytes (`extras_of`), by the weights root they are under
        self._views()

    # -- the engine loop's calls
    def admitted(self, adm: Any) -> None:
        """``adm`` a `verity_vllm.pipeline.serve_openai.Admission`."""
        self.known[adm.engine_id] = adm.http_id
        self.arrivals.setdefault(adm.engine_id, len(self.arrivals))
        self.ledgers[adm.engine_id] = Ledger(adm.http_id, adm.engine_id, list(adm.prompt_ids), dict(adm.ingress),
                                             getattr(adm, "user", None), getattr(adm, "release", None), len(self.known) - 1,
                                             times=dict(getattr(adm, "times", None) or {}))
        self.names[adm.engine_id] = self.ledgers[adm.engine_id].user or adm.http_id

    def streamed(self, engine_id: str, token_ids: list[int], finish_reason: str | None, stop_reason: Any) -> None:
        led = self.ledgers.get(engine_id)
        if led is None:
            return
        if token_ids and "first_token" not in led.times:
            led.times["first_token"] = self.wall()
        led.streamed.extend(int(t) for t in token_ids)
        if finish_reason is not None:
            led.finished = True
            led.finish = {"reason": finish_reason, "stop_reason": stop_reason}

    def aborted(self, engine_id: str, by: str) -> None:
        """The request ended unfinished, ``by`` its client (it went away) or the server (it stopped or failed)."""
        if by not in ("client", "server"):
            raise ValueError(f"an abort is by the client or the server, not {by!r}")
        led = self.ledgers.get(engine_id)
        if led is not None:
            led.finished = True
            led.finish = {"reason": "abort", "stop_reason": None, "k": len(led.streamed), "by": by}

    def idle(self) -> None:
        """The engine has nothing to run: close the open window once it holds ``idle_close_steps`` steps."""
        if self.is_open and int(self.com._step_index) >= self.idle_close_steps:
            self.close()

    def before_step(self, live: Iterable[str] = ()) -> None:
        """``live``: the engine ids the next step may run. With ``request_cap``, the open window closes before a step that
        would bring its distinct requests past the cap; the engine loop holds ``live`` to the cap, so the next window
        opens within it."""
        live = set(live)
        if self.is_open and (int(self.com._step_index) >= self.window_steps
                             or (self.window_bytes and committed_bytes(self.com) >= self.window_bytes)
                             or (self.request_cap and len(self.in_window | live) > self.request_cap)):
            self.close()
        if not self.is_open:
            self.open()
        self.in_window |= live

    # -- windows
    def open(self) -> None:
        t0 = self.clock()
        self.window += 1
        self.step_t = {}
        self.in_window = set()
        self.off_loop = self._views()
        self.com.reset()
        self.com.install()
        if self.arm is not None:
            try:
                self.arm.open(self.window)
            except (ValueError, RuntimeError, OSError) as e:      # the window's visible failure; serving goes on
                self._arm_out[self.window] = {"error": f"open: {type(e).__name__}: {e}"[:500]}
                self.arm_failed.append(self.window)
        self.open_s = self.clock() - t0
        self.is_open = True
        if self.marks is not None:
            self.marks(self.window, "open")

    def _close_arm(self) -> None:
        """The arm's window closed, before anything of the committer's: its outcome, or why it failed. An arm that
        releases off the loop hands back a `Future`, and the window's line waits for it (`_arm_closed`), not the loop."""
        window = self.window
        try:
            got = self.arm.close(window)
        except (ValueError, RuntimeError, OSError) as e:          # the window's visible failure; the Commit's close goes on
            got = {"error": f"close: {type(e).__name__}: {e}"[:500]}
        if callable(getattr(got, "add_done_callback", None)):
            with self._lock:
                self._arm_pending.add(window)
            got.add_done_callback(functools.partial(self._arm_closed, window))
        else:
            self._arm_closed(window, got)

    def _arm_closed(self, window: int, got: Any) -> None:
        """Window ``window``'s arm outcome, ``got`` or its `Future`'s: an error is a failed window's. Its line, if held for
        it, is written now."""
        if callable(getattr(got, "exception", None)):
            e = got.exception()
            got = got.result() if e is None else {"error": f"close: {type(e).__name__}: {e}"[:500]}
        with self._lock:
            if isinstance(got, Mapping) and "error" in got and window not in self.arm_failed:
                self.arm_failed.append(window)
            self._arm_out.setdefault(window, got)
            self._arm_pending.discard(window)
            rec = self._awaiting.pop(window, None)
        if rec is not None:
            self._write_window(rec)
            if self.mirror is not None:
                self.mirror()

    def stepped(self) -> None:
        """An engine step has returned: its committed steps ended now."""
        now = self.wall()
        for s in range(len(self.step_t), int(self.com._step_index)):
            self.step_t[s] = now

    def _views(self) -> bool:
        """Whether the window opening now closes off the loop: a sealer, and a committer that detaches each window into a view
        of its own (`NativeHostCommitter.enable_window_views`); a committer that refuses says why in ``finalize_on_loop``."""
        if self.sealer is None or self.finalize_on_loop is not None:
            return False
        if not callable(getattr(self.com, "detach", None)):
            self.finalize_on_loop = f"the committer ({type(self.com).__name__}) detaches no window"
            return False
        if not getattr(self.com, "window_views", False):
            try:
                self.com.enable_window_views()
            except ValueError as e:
                self.finalize_on_loop = str(e)
                return False
        return True

    def close(self) -> dict[str, Any] | None:
        """The open window closed, between engine steps; its record. Off the loop (`off_loop`) the loop only ends the window's
        last step, detaches the window into its view (`NativeHostCommitter.detach`), takes the ledgers as they stand and queues
        the rest on the sealer (`_finish`), waiting only while the windows queued there hold its bound (``seal_wait``); the rest,
        the window's finalize and everything that reads it, then runs on the sealer's thread in window order, and this returns None."""
        t0 = self.clock()
        laps = [("", t0)]
        lap = functools.partial(_lap, laps, self.clock)
        if self.arm is not None:
            self._close_arm()
            lap("arm")
        if self.off_loop:
            return self._close_off_loop(laps, lap)
        rc = self.com.finalize() if int(self.com._step_index) else None    # a window whose one step() only drained outputs
        lap("finalize")
        readback = self.record.absorb(self.com, self.window) if rc is not None else None
        lap("readback")
        self.com.uninstall()
        self.is_open = False
        c = _Closed(self.window, self.com, self.step_t, self.ledgers, [], self.open_s)
        rec, job = self._record(c, rc, readback, lap)
        self._no_release(rec, c.com, job)
        c.finished = self._pop_finished()
        self._write_step_trace(rec, rc, c)
        lap("step_trace")
        self._check_finished(c.finished)
        lap("requests")
        rec["seconds"] = {"open": round(self.open_s, 4), "close": round(laps[-1][1] - t0, 4)}
        rec["close_phases_s"] = _phases(laps)
        self._write_window(rec)
        if self.marks is not None:
            self.marks(self.window, "close")
        if self.mirror is not None:
            self.mirror()
        if isinstance(job, tuple):
            self._land(rec, job)
        return rec

    def _close_off_loop(self, laps: list[tuple[str, float]], lap: Callable[[str], None]) -> None:
        view = self.com.detach() if int(self.com._step_index) else None
        lap("detach")
        self.com.uninstall()
        self.is_open = False
        lap("uninstall")
        ledgers = {e: led if led.finished else replace(led, streamed=list(led.streamed)) for e, led in self.ledgers.items()}
        finished = self._pop_finished()
        with self._lock:
            self.closing[self.window] = [led.engine_id for led in finished]
        c = _Closed(self.window, view, self.step_t, ledgers, finished, self.open_s, off_loop=True)
        if self.marks is not None:
            self.marks(self.window, "close")
        lap("ledgers")
        nbytes = 0 if view is None else max(committed_bytes(view), int(getattr(view, "staged_bytes", 0) or 0))
        queued = threading.Event()
        c.t_close = self.clock()
        c.seal_wait = self.sealer.submit(functools.partial(self._finish, c, queued), nbytes)
        lap("seal_wait")
        c.on_loop = _phases(laps)
        queued.set()
        return None

    def _finish(self, c: _Closed, queued: threading.Event, _waited: float) -> None:
        """The sealer's thread: the rest of ``c``'s close, as `close` runs it on the loop, over the window's view."""
        queued.wait()
        t0 = self.clock()
        laps = [("", t0)]
        lap = functools.partial(_lap, laps, self.clock)
        job: Any = None
        waited, checking = 0.0, False
        try:
            waited = c.com.wait_absorbed() if c.com is not None else 0.0
            lap("finalize_wait")
            rc = c.com.finalize() if c.com is not None else None
            lap("finalize")
            readback = self.record.absorb(c.com, c.window) if rc is not None else None
            lap("readback")
            rec, job = self._record(c, rc, readback, lap)
            self._write_step_trace(rec, rc, c)
            lap("step_trace")
            checking = True
            self._check_finished(c.finished)
            lap("requests")
        except Exception as e:  # noqa: BLE001  (a window whose close fails is its own failure; the server goes on)
            rec = {"window": c.window, "run_id": f"{self.run_id}/w{c.window}", "steps": 0, "requests": [],
                   "error": f"{type(e).__name__}: {e}"[:500], "passed": False}
            job = None
            if not checking:
                self._check_finished(c.finished)        # against what the record holds of them: none of this window's rows
        finally:
            if c.com is not None and not isinstance(job, tuple):
                thaw(c.com)
        self._no_release(rec, c.com, job)
        on_loop = sum(c.on_loop.values())
        rec["seconds"] = {"open": round(c.open_s, 4), "close": round(on_loop + laps[-1][1] - t0, 4), "on_loop": round(on_loop, 4),
                          "finalize_wait": round(waited, 4), "seal_wait": round(c.seal_wait, 4), "queued": round(t0 - c.t_close, 4)}
        rec["close_phases_s"] = {"on_loop": round(on_loop, 4), **_phases(laps)}
        rec["on_loop_phases_s"] = c.on_loop
        self._write_window(rec)
        with self._lock:
            self.closing.pop(c.window, None)            # a close that failed before its line stays here, and fails the summary
        if self.mirror is not None:
            self.mirror()
        if isinstance(job, tuple):
            self._land(rec, job, here=(c.t_close, c.seal_wait))

    def _no_release(self, rec: dict[str, Any], com: Any, job: Any) -> None:
        """A window that reaches no seal, so no draw-gated release: `serve_release.Draws.discard`, its device pool pieces freed,
        so the next windows' steps don't wait on them (``rec["release"]``)."""
        if self.release is not None and com is not None and not isinstance(job, tuple):
            why = rec.get("error") or (job or {}).get("error") or "the window has no seal"
            if (got := self.release.discard(rec["window"], com, f"not sealed: {why}")) is not None:
                rec["release"] = got

    def _record(self, c: _Closed, rc: Any, readback: Any, lap: Callable[[str], None]) -> tuple[dict[str, Any], Any]:
        """The closed window's line of windows.jsonl as its finalize and readback make it, with its served rows and its seal's
        arguments (`_seal_job`)."""
        job: Any = None
        if rc is None:
            return {"window": c.window, "run_id": f"{self.run_id}/w{c.window}", "steps": 0, "empty": True, "requests": [],
                    "passed": True}, None
        commit = window_commit(rc)
        rec = {"window": c.window, "run_id": f"{self.run_id}/w{c.window}", "run_root": commit["run_root"],
               "steps": commit["steps"], "step_roots": [bytes(s.root).hex() for s in rc.steps],
               "committed_bytes": sum(int(s.n_bytes) for s in rc.steps), "ids_name": readback["ids_name"],
               "requests": readback["requests"], "commit": commit, "readback_errors": readback["readback_errors"],
               "readback_errors_n": readback["readback_errors_n"], "passed": commit["passed"] and readback["passed"],
               "gc": gc_of(rc)}
        lap("record")
        if self.served is not None:
            rec["served_rows"] = self._served_rows(rec, c.com)
            rec["passed"] = rec["passed"] and "error" not in rec["served_rows"]
            lap("served_rows")
        if self.seal is not None:
            job = self._seal_job(rec, rc, c)
            rec["seal"] = {"row_dir": str(self.out / "windows" / f"w{c.window}"), "off_loop": self.sealer is not None} \
                if isinstance(job, tuple) else job
            rec["passed"] = rec["passed"] and "error" not in rec["seal"]
            lap("seal_job")
        return rec, job

    def _write_window(self, rec: dict[str, Any]) -> None:
        """The window's line, with its arm's outcome; held while its arm is still closing (`_arm_closed` writes it)."""
        with self._lock:
            if rec["window"] in self._arm_pending:
                self._awaiting[rec["window"]] = rec
                return
            if self.arm is not None:
                rec["pouw_h1"] = self._arm_out.pop(rec["window"], None)
            self.records.append(rec)
            with open(self.out / "windows.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def _served_rows(self, rec: dict[str, Any], com: Any) -> dict[str, Any]:
        """The closed window's served rows committed under its run root (`commit_served`): the record's public part, with the
        statics it was packed under; the rows and salts written to ``served_rows/w<k>.json``. A failure is the window's."""
        try:
            by_step = served_rows(com)
            ids = {rid: self.arrivals[e] for rows in by_step.values() for rid, *_ in rows
                   if (e := base_id(rid, self.known)) in self.arrivals}
            served, session = commit_served(self.served, by_step, ids, rec["steps"], bytes.fromhex(rec["run_root"]))
        except Exception as e:  # noqa: BLE001  (a window whose rows can't be committed fails visibly; the server goes on)
            return {"error": f"{type(e).__name__}: {e}"[:500]}
        d = self.out / "served_rows"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"w{rec['window']}.json").write_text(json.dumps({
            "window": rec["window"], "run_root": rec["run_root"], "statics": self.served, "ids": ids,
            "session": [[None if s is None else [list(r) for r in s] for s in w] for w in session],
            "salts": [s.hex() for s in served.registration.salts]}))
        return {**served.record(), "statics": self.served}

    def _seal_job(self, rec: dict[str, Any], rc: Any, c: _Closed) -> Any:
        """What the window's seal reads that the next window changes (its requests, rows and tokens from the ledgers at its
        close, the registered tensors' bytes, and the committer: frozen when the seal runs after the next window's reset, its
        view off the loop) as `seal_window`'s arguments; or, by name, why it is not sealed: its failure, or the disk floor's
        refusal (`on_floor` hears it). Off the loop the seals of the windows before it have been written: none is pending."""
        by_name = {self.names[e]: led for e, led in c.ledgers.items() if e in self.names}

        def name_of(rid: str) -> str | None:
            return self.names.get(base_id(rid, self.known))

        try:
            pending = self.sealer.held if self.sealer is not None and not c.off_loop else 0
            short = self.disk.short(rec["committed_bytes"], pending) if self.disk else None
            if short is not None:
                self.floor_hit = self.floor_hit or {"window": rec["window"], **short}
                why = (f"not sealed: {short['free_gb']} GB free on {short['path']}, less {short['pending_gb']} GB the pending seals write, "
                       f"leaves under the disk floor ({short['floor_gb']} GB) for this window's {short['window_gb']} GB")
                if self.on_floor is not None:
                    self.on_floor(why)
                return {"error": why, "disk": short}
            reqs, request_rows = window_requests(served_rows(c.com), name_of, by_name.get)
            tokens = {q["request_id"]: list(by_name[q["request_id"]].streamed) for q in reqs if q["request_id"] in by_name}
            kw = dict(rec=rec, reqs=reqs, request_rows=request_rows, tokens=tokens, sampling={}, extras=self._extras_of(c.com, rc), **self.seal)
            if c.off_loop:
                if self.sealer.process is None:
                    for metas in c.com._layouts.values():
                        iter(metas)     # a plan's TensorMetas built once, here, as `freeze` builds them
                return c.com, rc, kw
            return (freeze(self.com) if self.sealer is not None else self.com), rc, kw
        except Exception as e:  # noqa: BLE001  (a window that can't be sealed fails visibly; the server goes on)
            return {"error": f"{type(e).__name__}: {e}"[:500]}

    def _extras_of(self, com: Any, rc: Any) -> Any:
        """`extras_of`, read from the device once for each weights root: the registered buffers and engine constants it carries
        are committed under that root, so every window under it carries the same bytes."""
        root = (getattr(rc, "weight_registration", None) or {}).get("root")
        if root is None or getattr(com, "weights_tree", None) is None:
            return extras_of(com, rc)
        if root not in self._extras:
            self._extras[root] = extras_of(com, rc)
        return self._extras[root]

    def read_extras(self) -> None:
        """`_extras_of` the committer's registered weights, read now rather than at the first seal: before a remote session
        holds the weights encoded where the engine holds them (`serve_remote`), so each seal carries the bytes committed."""
        reg = getattr(self.com, "weights", None)
        if self.seal is not None and reg is not None:
            self._extras_of(self.com, SimpleNamespace(weight_registration=reg))

    def _land(self, rec: Mapping[str, Any], job: tuple[Any, Any, dict[str, Any]], here: tuple[float, float] | None = None) -> None:
        """The window's seal into ``rec["seal"]["row_dir"]`` and then its line in seals.jsonl: on the sealer's thread, in window
        order (the engine loop waits here only while the windows awaiting their seal hold the sealer's bound), else here.
        ``here``: already on the sealer's thread (`_finish`), with when the window's close queued it and how long that waited."""
        t_close = self.clock() if here is None else here[0]

        def run(waited: float = 0.0) -> None:
            com, rc, kw = job
            t0 = self.clock()
            released: dict[str, Any] | None = None
            try:
                if self.release is not None:
                    released = self.release.release(rec["window"], com, self.clock)
                if self.sealer is not None and self.sealer.process is not None:
                    sealed = self.sealer.process.seal(com, rc, Path(rec["seal"]["row_dir"]),
                                                      written=(lambda: thaw(com)) if com is not self.com else None, **kw)
                else:
                    sealed = seal_window(com, rc, Path(rec["seal"]["row_dir"]), **kw)
            except SealFailed as e:
                sealed = {"error": str(e)[:500]}
            except Exception as e:  # noqa: BLE001  (a window that can't be sealed fails visibly; the server goes on)
                sealed = {"error": f"{type(e).__name__}: {e}"[:500]}
            finally:
                if com is not self.com:
                    thaw(com)
            line = {"window": rec["window"], "run_id": rec["run_id"], "run_root": rec.get("run_root"), **sealed,
                    "queued_s": round(t0 - t_close, 3), "seal_wait_s": round(waited, 4)}
            if self.release is not None:
                line["release"] = released
            with self._lock:
                self.seals.append(line)
            with open(self.out / "seals.jsonl", "a") as fh:
                fh.write(json.dumps(line) + "\n")
            if self.mirror is not None:
                self.mirror()

        if here is not None:
            run(here[1])
        elif self.sealer is None:
            run()
        else:
            self.sealer.submit(run, int(rec.get("committed_bytes") or 0))

    def check(self, led: Ledger) -> dict[str, Any]:
        """``led``'s record against what was served: `Record.check` with the seed it carried and its finish."""
        params = led.ingress.get("params") or {}
        finish = None if led.finish is None else {**led.finish, "max_tokens": params.get("max_tokens"),
                                                  "min_tokens": params.get("min_tokens") or 0, "eos_ids": self.eos_ids}
        got = self.record.check(led.engine_id, led.prompt_ids, led.streamed, seed=led.ingress.get("seed"), finish=finish,
                                decode=self.decode)
        if led.ingress.get("seed") is None:
            got = {**got, "passed": False, "problems": ["no seed carried into the engine", *got.get("problems", [])]}
        return got

    def _write_step_trace(self, rec: dict[str, Any], rc: Any, c: _Closed) -> None:
        """One ``step_trace.jsonl`` line a committed step: its window, its root, when its engine step ended (unix seconds),
        the request rows it ran (`served_record.served_rows`: runner id, lo, hi, start, prefill length), and the
        committer's own trace of it (tensors, bytes, leaves, layout digest)."""
        trace = {int(t["step"]): t for t in ((getattr(rc, "extra", None) or {}).get("step_trace") or [])} if rc is not None else {}
        rows = served_rows(c.com) if c.com is not None else {}
        with open(self.out / "step_trace.jsonl", "a") as fh:
            for s, root in enumerate(rec.get("step_roots") or []):
                own = {k: v for k, v in trace.get(s, {}).items() if k not in ("step", "root")}
                fh.write(json.dumps({"window": rec["window"], "run_id": rec["run_id"], "run_root": rec.get("run_root"), "step": s,
                                     "step_root": root, "t_end": c.step_t.get(s), "rows": [list(r) for r in rows.get(s, [])],
                                     **own}) + "\n")

    def _pop_finished(self) -> list[Ledger]:
        """The loop's: the ledgers finished by the close, out of the open ones (their requests ran no step after it)."""
        return [self.ledgers.pop(e) for e in [e for e, led in self.ledgers.items() if led.finished]]

    def _check_finished(self, finished: Sequence[Ledger]) -> None:
        """Each of ``finished`` checked against the record its window's readback completed, a line of requests.jsonl, and its
        engine id and ``user`` a line of `REQUEST_USERS`."""
        with open(self.out / "requests.jsonl", "a") as fh, open(self.out / REQUEST_USERS, "a") as fu:
            for led in finished:
                eid = led.engine_id
                rr = self.record.requests.get(eid)
                rec = {"http_id": led.http_id, "engine_id": led.engine_id, "user": led.user, "arrival": self.arrivals.get(eid),
                       "release": led.release, "admitted": led.admitted, "runner_ids": rr.runner_ids if rr else [],
                       "prompt_ids": led.prompt_ids, "streamed_ids": led.streamed, "ingress": led.ingress, "finish": led.finish,
                       "windows": sorted({r[0] for r in rr.rows}) if rr else [], "rows": rr.rows if rr else [],
                       "committed_seeds": sorted(rr.seeds) if rr else [], "check": self.check(led), "times": led.times}
                with self._lock:
                    self.checked.append({"http_id": led.http_id, "engine_id": led.engine_id, "passed": rec["check"]["passed"],
                                         "cache_hit": int(rec["check"].get("cache_hit") or 0)})
                fh.write(json.dumps(rec) + "\n")
                fu.write(json.dumps({"engine_id": rec["engine_id"], "user": rec["user"]}) + "\n")

    def summary(self) -> dict[str, Any]:
        """The windows closed so far; ``closing`` those whose finalize, off the loop, has yet to run (their requests unchecked)."""
        with self._lock:
            checked, records, seals = list(self.checked), list(self.records), list(self.seals)
            closing = {w: list(e) for w, e in self.closing.items()}
            arm_closing = sorted(self._arm_pending)
        bad_req = [c for c in checked if not c["passed"]]
        bad_win = sorted({r["window"] for r in records if not r["passed"]} | {s["window"] for s in seals if "error" in s})
        queued = sum(1 for r in records if "row_dir" in (r.get("seal") or {}))
        waits, took = [float(s["seal_wait_s"]) for s in seals], [float(s.get("seal_s") or 0) for s in seals]
        unchecked = sorted({*self.ledgers, *(e for es in closing.values() for e in es)})
        return {"windows": len(records), "windows_failed": bad_win, "steps": sum(r["steps"] for r in records),
                "closing": sorted(closing), "finalize_on_loop": self.finalize_on_loop,
                "requests_checked": len(checked), "requests_failed": [c["http_id"] for c in bad_req][:20],
                "seals": {"sealed": sum(1 for s in seals if "error" not in s), "awaiting": queued - len(seals),
                          "seal_s": {"max": max(took, default=0.0), "total": round(sum(took), 3)},
                          "wait_s": {"windows": sum(1 for w in waits if w > 0), "max": max(waits, default=0.0), "total": round(sum(waits), 3)}},
                "close_s_max": max((float(r["seconds"]["close"]) for r in records), default=0.0),
                "on_loop_s_max": max((float(r["seconds"].get("on_loop", r["seconds"]["close"])) for r in records), default=0.0),
                "disk_floor": self.floor_hit,
                "cache_hits": sum(1 for c in checked if c.get("cache_hit")),
                "cached_positions": sum(int(c.get("cache_hit") or 0) for c in checked),
                "unchecked": unchecked,
                **({"pouw_h1": {**self.arm.summary(), "failed": self.arm_failed[:20], "closing": arm_closing}}
                   if self.arm is not None else {}),
                "passed": bool(checked) and not bad_req and not bad_win and not unchecked and not closing and not arm_closing
                and queued == len(seals)}
