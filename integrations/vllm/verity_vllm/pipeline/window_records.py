"""A closed window's records, from plain data (`Recorder`): its readback into the requests' records
(`served_record.Record.absorb_read`), its served rows committed under its run root (`commit_served`), its steps' lines of
``step_trace.jsonl``, and each request it finished checked against what was served, a line of ``requests.jsonl`` and of
`REQUEST_USERS`.

The serving process reads what these take out of the window's committer or view (`serve_commit.CommitWindows._window_in`: the
committed ids, samples, seeds and positions as bytes, `served_record.window_read`, its rows, the committer's step trace, the
ledgers its close finished, the requests admitted since the window before). A `RecordProcess` runs the rest in a spawned process
of its own, beside the seal's (`seal_process.ProcessSealer`), so that none of its Python takes the serving process's GIL while
the engine loop runs; without one it runs on the thread that closes the window. Either way the windows go through it one at a
time, in order, and a request spanning windows is read back into one record.
"""
from __future__ import annotations

import json
import multiprocessing
import multiprocessing.connection
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity.core.protocols.certifier import served_rows as SRW
from verity_vllm.commit.served_record import Record, base_id

#: requests.jsonl's engine id to ``user`` map alone, a line each as requests.jsonl's: ``{"engine_id", "user"}``.  It holds no
#: request value (the network certifier's users are request indexes, ``user-<i>``), so it can leave the node with
#: advice_trace.jsonl, windows.jsonl and serve.json while requests.jsonl, with the prompt and output ids, stays.
REQUEST_USERS = "request_users.jsonl"

#: a ledger's fields as the recorder checks and records them (`serve_commit.Ledger`)
LEDGER_KEYS = ("http_id", "engine_id", "prompt_ids", "ingress", "user", "release", "admitted", "streamed", "finish", "times")


def commit_served(statics: Mapping[str, int], by_step: Mapping[int, Sequence[tuple]], ids: Mapping[str, int], steps: int,
                  run_root: bytes, salts: Sequence[bytes] | None = None) -> tuple[SRW.Served, list]:
    """A window's served rows (`served_record.served_rows`' form, by step) committed under its run root
    (`served_rows.register`, fresh salts unless ``salts`` gives them), and the slice windows they make."""
    S = SimpleNamespace(**statics)
    session = SRW.windows(S, by_step, ids, steps)
    return SRW.register(S, session, run_root, salts=salts), session


class RecordFailed(RuntimeError):
    """What a window's records raised, as ``<type>: <message>``."""


class Recorder:
    """Every window's records over one serve dir ``out``, in window order (the module docstring). ``served``: the compile
    slice's statics, to commit each window's rows; ``eos_ids``: the tokenizer's, for the finish check."""

    def __init__(self, out: str | Path, *, run_id: str, served: Mapping[str, int] | None = None, eos_ids: Sequence[int] = ()) -> None:
        self.out, self.run_id = Path(out), run_id
        self.served = dict(served) if served is not None else None
        self.eos_ids = [int(x) for x in eos_ids]
        self.known: dict[str, str] = {}                  # every admitted engine id -> its HTTP id; rows past a finish still map
        self.arrivals: dict[str, int] = {}               # ... -> its arrival id, in admission order
        self.record = Record(known=self.known)

    def window(self, w: Mapping[str, Any]) -> dict[str, Any]:
        """Window ``w``'s records (`serve_commit.CommitWindows._window_in`): its ``readback`` and ``served_rows`` records, and
        what its finished requests' checks found (``checked``), with where its time went (``phases``). A failure before the
        checks is the window's ``error``; its requests are still checked, against what the record holds of them."""
        for e, h in w["admitted"]:
            self.known[e] = h
            self.arrivals.setdefault(e, len(self.arrivals))
        laps = [time.perf_counter()]
        got: dict[str, Any] = {"checked": []}
        try:
            if w["read"] is not None:
                got["readback"] = self.record.absorb_read(w["window"], w["read"])
                if self.served is not None:
                    got["served_rows"] = self.served_rows(w)
            laps.append(time.perf_counter())
            self.step_trace(w)
            laps.append(time.perf_counter())
            self.check_finished(w["finished"], got["checked"])
        except Exception as e:  # noqa: BLE001  (the window's failure, in its line; serving goes on)
            got["error"] = f"{type(e).__name__}: {e}"[:500]
            if len(laps) < 3:
                self.check_finished(w["finished"], got["checked"])
        laps.append(time.perf_counter())
        got["phases"] = {k: round(b - a, 4) for k, a, b in zip(("readback", "step_trace", "requests"), laps, laps[1:])}
        return got

    def served_rows(self, w: Mapping[str, Any]) -> dict[str, Any]:
        """The window's served rows committed under its run root (`commit_served`): the record's public part, with the statics
        it was packed under; the rows and salts written to ``served_rows/w<k>.json``. A failure is the window's."""
        try:
            by_step = w["read"]["rows"]
            ids = {rid: self.arrivals[e] for rows in by_step.values() for rid, *_ in rows
                   if (e := base_id(rid, self.known)) in self.arrivals}
            served, session = commit_served(self.served, by_step, ids, w["steps"], bytes.fromhex(w["run_root"]))
        except Exception as e:  # noqa: BLE001  (a window whose rows can't be committed fails visibly; the server goes on)
            return {"error": f"{type(e).__name__}: {e}"[:500]}
        d = self.out / "served_rows"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"w{w['window']}.json").write_text(json.dumps({
            "window": w["window"], "run_root": w["run_root"], "statics": self.served, "ids": ids,
            "session": [[None if s is None else [list(r) for r in s] for s in win] for win in session],
            "salts": [s.hex() for s in served.registration.salts]}))
        return {**served.record(), "statics": self.served}

    def step_trace(self, w: Mapping[str, Any]) -> None:
        """One ``step_trace.jsonl`` line a committed step: its window, its root, when its engine step ended (unix seconds),
        the request rows it ran (`served_record.served_rows`: runner id, lo, hi, start, prefill length), and the
        committer's own trace of it (tensors, bytes, leaves, layout digest)."""
        trace = {int(t["step"]): t for t in w["trace"] or ()}
        rows = (w["read"] or {}).get("rows") or {}
        with open(self.out / "step_trace.jsonl", "a") as fh:
            for s, root in enumerate(w["step_roots"]):
                own = {k: v for k, v in trace.get(s, {}).items() if k not in ("step", "root")}
                fh.write(json.dumps({"window": w["window"], "run_id": f"{self.run_id}/w{w['window']}", "run_root": w["run_root"],
                                     "step": s, "step_root": root, "t_end": w["step_t"].get(s),
                                     "rows": [list(r) for r in rows.get(s, [])], **own}) + "\n")

    def check(self, led: Mapping[str, Any]) -> dict[str, Any]:
        """``led``'s record against what was served: `Record.check` with the seed it carried and its finish (its stop string's
        end, ``stop_end``, found where the tokenizer is)."""
        params = led["ingress"].get("params") or {}
        finish = None if led["finish"] is None else {**led["finish"], "max_tokens": params.get("max_tokens"),
                                                     "min_tokens": params.get("min_tokens") or 0, "eos_ids": self.eos_ids,
                                                     **({"stop_end": led["stop_end"]} if "stop_end" in led else {})}
        got = self.record.check(led["engine_id"], led["prompt_ids"], led["streamed"], seed=led["ingress"].get("seed"), finish=finish)
        if led["ingress"].get("seed") is None:
            got = {**got, "passed": False, "problems": ["no seed carried into the engine", *got.get("problems", [])]}
        return got

    def check_finished(self, finished: Sequence[Mapping[str, Any]], checked: list[dict[str, Any]]) -> None:
        """Each of ``finished`` checked against the record its window's readback completed, a line of requests.jsonl, its engine
        id and ``user`` a line of `REQUEST_USERS`, and what its check found in ``checked``."""
        with open(self.out / "requests.jsonl", "a") as fh, open(self.out / REQUEST_USERS, "a") as fu:
            for led in finished:
                eid = led["engine_id"]
                rr = self.record.requests.get(eid)
                rec = {"http_id": led["http_id"], "engine_id": eid, "user": led["user"], "arrival": self.arrivals.get(eid),
                       "release": led["release"], "admitted": led["admitted"], "runner_ids": rr.runner_ids if rr else [],
                       "prompt_ids": led["prompt_ids"], "streamed_ids": led["streamed"], "ingress": led["ingress"],
                       "finish": led["finish"], "windows": sorted({r[0] for r in rr.rows}) if rr else [],
                       "rows": rr.rows if rr else [], "committed_seeds": sorted(rr.seeds) if rr else [], "check": self.check(led),
                       "times": led["times"]}
                checked.append({"http_id": led["http_id"], "engine_id": eid, "passed": rec["check"]["passed"],
                                "cache_hit": int(rec["check"].get("cache_hit") or 0)})
                fh.write(json.dumps(rec) + "\n")
                fu.write(json.dumps({"engine_id": rec["engine_id"], "user": rec["user"]}) + "\n")


class Spawned:
    """A spawned process running ``target`` on the other end of a pipe, which first sends ``("ready", pid)``, called from one
    thread at a time. A process that has exited, killed or failed, raises at the next receive by name, and so does every one
    after it."""

    def __init__(self, target: Callable[[Any], None], name: str) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=target, args=(child,), name=name, daemon=True)
        self.process.start()
        child.close()
        self.pid = self._receive("ready")[1]

    def _receive(self, kind: str) -> tuple:
        multiprocessing.connection.wait([self.conn, self.process.sentinel])
        try:
            msg = self.conn.recv() if self.conn.poll() else None
        except EOFError:
            msg = None
        if msg is None:
            self.process.join()
            raise RuntimeError(f"the {self.process.name} process exited ({self.process.exitcode}) before its {kind}")
        return msg

    def close(self) -> dict[str, Any]:
        """The process ended once its calls are done: its pid and exit code."""
        try:
            self.conn.send(("close",))
        except OSError:
            pass
        self.process.join()
        self.conn.close()
        return {"pid": self.pid, "exitcode": self.process.exitcode}


class RecordProcess(Spawned):
    """One spawned process that runs a `Recorder` (`start`), called from `seal_stages.Sealer`'s first thread: each window's
    records (`window`) and a request's check (`check`), a message each."""

    def __init__(self) -> None:
        super().__init__(_main, "serve-records")

    def start(self, **config: Any) -> RecordProcess:
        """The process's `Recorder(**config)`; this, as it."""
        self.conn.send(("start", config))
        return self

    def window(self, w: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("window", w)

    def check(self, led: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("check", led)

    def _call(self, kind: str, arg: Any) -> Any:
        try:
            self.conn.send((kind, arg))
        except OSError:
            pass                                         # a process gone: the receive says how it ended
        got = self._receive(kind)
        if got[0] == "error":
            raise RecordFailed(got[1])
        return got[1]


def _main(conn: Any) -> None:
    conn.send(("ready", os.getpid()))
    rec: Recorder | None = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "close":
            return
        if msg[0] == "start":
            rec = Recorder(**msg[1])
            continue
        try:
            conn.send((msg[0], getattr(rec, msg[0])(msg[1])))
        except Exception as e:  # noqa: BLE001  (the call's failure, sent to the serving process, which records it)
            conn.send(("error", f"{type(e).__name__}: {e}"[:500]))
