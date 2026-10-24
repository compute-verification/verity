"""The Commit side of `verity_vllm.pipeline.serve_openai`: the committer's windows over the serving engine, the map from
each HTTP request to its engine request, and each request's record held to what was served.

A window (`CommitWindows`) opens at the first engine step after the engine was idle and closes when the engine is idle
again (once it holds ``idle_close_steps`` steps), or after ``window_steps`` steps; each is one Commit, with its own run
root. The record is the served path's and COMMIT-ARB's one record, `commit.served_record`: the committer carries a
`served_record.SamplerInputsSource`, which binds each sampled row's seed and position beside the sampled ids and keeps
the step's rows, and at each close `served_record.Record.absorb` reads the window back from the committed bytes into one
`RequestRecord` per engine request. A finished request is checked by `Record.check` against what was streamed for it,
the seed it carried into the engine (the client's, or the served default, `engine.served_seed`) and how its stream
ended. Its ingress record, the sampling parameters it brought, is written beside it, with the body's ``user`` (null when it
sent none). Each committed step gets a line of ``step_trace.jsonl`` at its window's close: its root, when its engine step ended,
its rows and the committer's trace. ``marks`` hears each window's open and close (the scheduler step trace's window records,
`engine.step_trace.StepTrace.window`).

With ``seal`` (`seal_window`), each closed window is also sealed as a deferred Commit's row dir, ``windows/w<k>/``, the way a
declared Commit with ``--replay-deferred`` seals its run (`pipeline.replay_bundle`): ``commit/replay_bundle_p0/`` (the committed
store, the weights tree, the binding map, the context, the manifest), and beside it the record that bundle is opened against
(`deferred_rows.of_record`): ``commit/verdict.json`` naming the bundle, ``commit/runs.jsonl`` with the window's run, and
``commit/binding_map_p0.json``. The window's requests are named by the body's ``user`` (the id the advice names them by), and the
bundle names no Programs: a served window's Programs are the registered compiler's for its advice, made after the window closed.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.commit.served_record import Record, base_id, served_rows, window_commit

#: what a sealed served window's context says it is (`deferred_rows.of_record` reads no Programs of record for it)
SERVED_WINDOW = "verity-vllm/served-window/v1"

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
    suffix dropped): what `verity-vllm advice OUT` reads, with each window's open and close (``CommitWindows(marks=trace.window)``)."""
    from verity_vllm.engine.engine_driver import external_id
    from verity_vllm.engine.step_trace import SERVED, StepTrace

    return StepTrace(Path(out) / SERVED, request_id=external_id, clock=time.time).install(engine)


def window_requests(rows: Mapping[int, Sequence[tuple]], name_of: Callable[[str], str | None],
                    ledger_of: Callable[[str], Any]) -> tuple[list[dict[str, Any]], dict[int, list[tuple[str, int, int]]]]:
    """A closed window's requests as a deferred Commit's context lists them (``request_id`` the body's ``user``, else the HTTP id;
    its index, prompt ids, ``max_tokens`` and first step in the window) and each step's request rows for the binding map
    (`binding.build_binding_map`'s ``request_rows``), from the rows the committer's `served_record.SamplerInputsSource` kept.
    ``name_of`` maps a runner id to the request's name (None: not a served request), ``ledger_of`` a name to its `Ledger`."""
    reqs: dict[str, dict[str, Any]] = {}
    request_rows: dict[int, list[tuple[str, int, int]]] = {}
    for step in sorted(rows):
        for rid, lo, hi, _start, _prefill in rows[step]:
            name = name_of(str(rid))
            if name is None:
                continue
            request_rows.setdefault(int(step), []).append((name, int(lo), int(hi)))
            if name not in reqs:
                led = ledger_of(name)
                params = (led.ingress.get("params") or {}) if led is not None else {}
                reqs[name] = {"request_id": name, "index": len(reqs), "arrive_step": int(step),
                              "prompt_token_ids": list(led.prompt_ids) if led is not None else [],
                              "max_tokens": params.get("max_tokens"), "http_id": led.http_id if led is not None else None,
                              "engine_id": led.engine_id if led is not None else None,
                              "ingress": dict(led.ingress) if led is not None else None}
    return list(reqs.values()), request_rows


def seal_window(com: Any, rc: Any, d: Path, *, rec: Mapping[str, Any], reqs: list[dict[str, Any]],
                request_rows: dict[int, list[tuple[str, int, int]]], tokens: Mapping[str, list[int]], sampling: Mapping[str, Any],
                label: str, code: str | None = None, engine_of_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Seal one closed window into the row dir ``d`` as a declared Commit with ``--replay-deferred`` seals its run: the binding map
    (`binding.build_binding_map` over the window's layouts and step roots, its request rows by name), the store and weights tree
    (`replay_bundle.write_store`), then `replay_bundle.seal_all` over the window's one run, its sampled replay and linkage deferred
    to the bundle; ``commit/runs.jsonl`` and ``commit/verdict.json`` are the record it is opened against. Returns what the
    window's line in windows.jsonl records of it."""
    from verity_vllm.commit import binding as B
    from verity_vllm.pipeline import replay_bundle as RB

    t0 = time.perf_counter()
    out = Path(d) / "commit"
    out.mkdir(parents=True, exist_ok=True)
    pd = bytes(com.program_digest).hex()
    bmap = B.build_binding_map(com._layouts, rc.steps, rc.run_root, pd, request_rows=request_rows, chunk=com.chunk,
                               weight_registration=rc.weight_registration)
    (out / "binding_map_p0.json").write_text(json.dumps({k: v for k, v in bmap.items() if k != "entries"} | {"entries": bmap["entries"]},
                                                        separators=(",", ":"), default=str))
    hashed: dict[str, str] = {}
    RB.write_store(out, 0, com, rc, digests=hashed)
    scheme = getattr(com, "commit_scheme", "vllm-v1")
    sr, bl = RB.deferred_records(RB.path(out, 0))
    row = {"pair": 0, "arm": "instrumented", "label": label, "served_window": {"schema": SERVED_WINDOW, **{k: rec.get(k) for k in ("window", "run_id")}},
           "tokens_equal_ref": None,
           "commit": {"run_root": rc.run_root.hex(), "commit_scheme": scheme, "weight_registration": rc.weight_registration,
                      "steps": rec.get("steps"), "committer": com.name, "passed": rec.get("passed"),
                      "binding": {"map_digest": bmap["digest"], "map_entries_n": bmap["entries_n"], "program_digest": pd},
                      "commit_pass": None, "commit_fail_reason": None, "sampled_replay": sr, "boundary_linkage": bl}}
    args = {k: None for k in RB.ARGS} | {"label": label, "retain": getattr(com, "retain", None)}
    ctx = {"pair": 0, "arm": "instrumented", "committer": com.name, "run_root": rc.run_root.hex(), "code": code, "args": args,
           "requests": reqs, "sampling": dict(sampling), "tokens": {k: list(v) for k, v in tokens.items()}, "stop_of_record": None,
           "ids_full": None, "retention": getattr(com, "retain", None), "required_manifest_sha256": None, "declaration": None,
           "required_classes": [], "min_openings": 0, "vc_result": None, "engine_of_record": dict(engine_of_record or {}), "slim": None,
           "served_window": {"schema": SERVED_WINDOW, "window": rec.get("window"), "run_id": rec.get("run_id"), "commit_scheme": scheme,
                             "programs": "none sealed: the registered compiler's Programs for the window's advice"},
           RB.HASHED: hashed}
    verdict: dict[str, Any] = {"schema": SERVED_WINDOW, "run_id": rec.get("run_id"), "window": rec.get("window"), "commit_pass": None,
                               "commit_scheme": scheme, "run_root": rc.run_root.hex()}
    tl = SimpleNamespace(span=lambda *_a, **_k: _Null())
    RB.seal_all(out, {0: ctx}, [row], verdict, [0], tl=tl)
    (out / "runs.jsonl").write_text(json.dumps(row, default=str) + "\n")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    b = (verdict.get("replay_deferred") or {}).get("bundles") or [{}]
    return {"row_dir": str(d), "bundle": b[0].get("path"), "manifest_sha256": b[0].get("manifest_sha256"), "bytes": b[0].get("bytes"),
            "commit_scheme": scheme, "binding_map_digest": bmap["digest"], "requests": [q["request_id"] for q in reqs],
            "seal_s": round(time.perf_counter() - t0, 3)}


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_a: object) -> None:
        return None


def param_of(params: Any, name: str) -> Any:
    """One field of the engine's sampling parameters: vLLM's ``SamplingParams`` or `PlainProtocol`'s dict."""
    return params.get(name) if isinstance(params, Mapping) else getattr(params, name, None)


def ingress_of(brought: Mapping[str, Any], params: Any) -> dict[str, Any]:
    """A request's ingress record: the sampling fields it sent, the engine's parameters they make (``params``, seeded),
    and the seed with its rule (``request``: the client's; ``default``: `engine.served_seed.DEFAULT_SEED`)."""
    made = {k: param_of(params, k) for k in SAMPLING_FIELDS}
    return {"brought": dict(brought), "params": made, "seed": made["seed"],
            "seed_rule": "request" if brought.get("seed") is not None else "default"}


@dataclass
class Ledger:
    """One request as served: its ids, the HTTP id -> engine id map, its ingress record and its stream."""

    http_id: str
    engine_id: str
    prompt_ids: list[int]
    ingress: dict[str, Any] = field(default_factory=dict)
    user: str | None = None                      # the body's `user`, the id the advice names the request by
    streamed: list[int] = field(default_factory=list)
    finish: dict[str, Any] | None = None
    finished: bool = False
    aborted: bool = False


class CommitWindows:
    """The committer's windows over a serving engine, and every request's `Ledger`. ``committer`` is a
    `NativeHostCommitter` with a `served_record.SamplerInputsSource` attached (or its offline stand-in in the tests:
    ``reset``, ``install``, ``uninstall``, ``finalize``, ``_layouts``, ``_step_index``, ``sources``); ``eos_ids`` and
    ``decode`` are the tokenizer's, for the finish check (`served_record.finish_problems`)."""

    def __init__(self, committer: Any, out: Path, *, run_id: str, window_steps: int, idle_close_steps: int = 1,
                 eos_ids: Sequence[int] = (), decode: Callable[[Sequence[int]], str] | None = None,
                 clock: Callable[[], float] = time.perf_counter, wall: Callable[[], float] = time.time,
                 marks: Callable[[int, str], None] | None = None, seal: Mapping[str, Any] | None = None) -> None:
        self.com, self.out, self.clock, self.wall, self.marks = committer, Path(out), clock, wall, marks
        self.seal = dict(seal) if seal is not None else None     # `seal_window`'s label, code and engine_of_record: seal every window
        self.run_id, self.window_steps, self.idle_close_steps = run_id, int(window_steps), int(idle_close_steps)
        self.eos_ids, self.decode = [int(x) for x in eos_ids], decode
        self.known: dict[str, str] = {}                  # every admitted engine id -> its HTTP id; rows past a finish still map
        self.names: dict[str, str] = {}                  # every admitted engine id -> its name in a sealed window (`user`, else HTTP id)
        self.record = Record(known=self.known)
        self.window, self.open_s, self.is_open = -1, 0.0, False
        self.step_t: dict[int, float] = {}               # the open window's committed steps -> when their engine step ended
        self.ledgers: dict[str, Ledger] = {}
        self.records: list[dict[str, Any]] = []
        self.checked: list[dict[str, Any]] = []

    # -- the engine loop's calls
    def admitted(self, adm: Any) -> None:
        """``adm`` a `verity_vllm.pipeline.serve_openai.Admission`."""
        self.known[adm.engine_id] = adm.http_id
        self.ledgers[adm.engine_id] = Ledger(adm.http_id, adm.engine_id, list(adm.prompt_ids), dict(adm.ingress),
                                             getattr(adm, "user", None))
        self.names[adm.engine_id] = self.ledgers[adm.engine_id].user or adm.http_id

    def streamed(self, engine_id: str, token_ids: list[int], finish_reason: str | None, stop_reason: Any) -> None:
        led = self.ledgers.get(engine_id)
        if led is None:
            return
        led.streamed.extend(int(t) for t in token_ids)
        if finish_reason is not None:
            led.finished = True
            led.finish = {"reason": finish_reason, "stop_reason": stop_reason}

    def aborted(self, engine_id: str) -> None:
        led = self.ledgers.get(engine_id)
        if led is not None:
            led.aborted = led.finished = True

    def idle(self) -> None:
        """The engine has nothing to run: close the open window once it holds ``idle_close_steps`` steps."""
        if self.is_open and int(self.com._step_index) >= self.idle_close_steps:
            self.close()

    def before_step(self) -> None:
        if self.is_open and int(self.com._step_index) >= self.window_steps:
            self.close()
        if not self.is_open:
            self.open()

    # -- windows
    def open(self) -> None:
        t0 = self.clock()
        self.window += 1
        self.step_t = {}
        self.com.reset()
        self.com.install()
        self.open_s = self.clock() - t0
        self.is_open = True
        if self.marks is not None:
            self.marks(self.window, "open")

    def stepped(self) -> None:
        """An engine step has returned: its committed steps ended now."""
        now = self.wall()
        for s in range(len(self.step_t), int(self.com._step_index)):
            self.step_t[s] = now

    def close(self) -> dict[str, Any]:
        t0 = self.clock()
        rc = self.com.finalize() if int(self.com._step_index) else None    # a window whose one step() only drained outputs
        readback = self.record.absorb(self.com, self.window) if rc is not None else None
        self.com.uninstall()
        self.is_open = False
        if rc is None:
            rec: dict[str, Any] = {"window": self.window, "run_id": f"{self.run_id}/w{self.window}", "steps": 0, "empty": True,
                                   "requests": [], "passed": True}
        else:
            commit = window_commit(rc)
            rec = {"window": self.window, "run_id": f"{self.run_id}/w{self.window}", "run_root": commit["run_root"],
                   "steps": commit["steps"], "step_roots": [bytes(s.root).hex() for s in rc.steps],
                   "committed_bytes": sum(int(s.n_bytes) for s in rc.steps), "ids_name": readback["ids_name"],
                   "requests": readback["requests"], "commit": commit, "readback_errors": readback["readback_errors"],
                   "readback_errors_n": readback["readback_errors_n"], "passed": commit["passed"] and readback["passed"]}
            if self.seal is not None:
                rec["sealed"] = self._seal(rec, rc)
                rec["passed"] = rec["passed"] and "error" not in rec["sealed"]
        rec["seconds"] = {"open": round(self.open_s, 4), "close": round(self.clock() - t0, 4)}
        self.records.append(rec)
        with open(self.out / "windows.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._write_step_trace(rec, rc)
        self._check_finished()
        if self.marks is not None:
            self.marks(self.window, "close")
        return rec

    def _seal(self, rec: dict[str, Any], rc: Any) -> dict[str, Any]:
        """The closed window sealed into ``out/windows/w<k>/`` (`seal_window`); a failure is the window's, by name."""
        by_name = {self.names[e]: led for e, led in self.ledgers.items() if e in self.names}

        def name_of(rid: str) -> str | None:
            return self.names.get(base_id(rid, self.known))

        try:
            reqs, request_rows = window_requests(served_rows(self.com), name_of, by_name.get)
            tokens = {q["request_id"]: list(by_name[q["request_id"]].streamed) for q in reqs if q["request_id"] in by_name}
            return seal_window(self.com, rc, self.out / "windows" / f"w{rec['window']}", rec=rec, reqs=reqs, request_rows=request_rows,
                               tokens=tokens, sampling={}, **self.seal)
        except Exception as e:  # noqa: BLE001  (a window that can't be sealed fails visibly; the server goes on)
            return {"error": f"{type(e).__name__}: {e}"[:500]}

    def check(self, led: Ledger) -> dict[str, Any]:
        """``led``'s record against what was served: `Record.check` with the seed it carried and its finish."""
        params = led.ingress.get("params") or {}
        finish = None if led.finish is None else {**led.finish, "max_tokens": params.get("max_tokens"),
                                                  "min_tokens": params.get("min_tokens") or 0, "eos_ids": self.eos_ids}
        got = self.record.check(led.engine_id, led.prompt_ids, led.streamed, seed=led.ingress.get("seed"), finish=finish,
                                decode=self.decode)
        if led.aborted:
            got = {**got, "passed": False, "problems": ["aborted before its finish", *got.get("problems", [])]}
        if led.ingress.get("seed") is None:
            got = {**got, "passed": False, "problems": ["no seed carried into the engine", *got.get("problems", [])]}
        return got

    def _write_step_trace(self, rec: dict[str, Any], rc: Any) -> None:
        """One ``step_trace.jsonl`` line a committed step: its window, its root, when its engine step ended (unix seconds),
        the request rows it ran (`served_record.served_rows`: runner id, lo, hi, start, prefill length), and the
        committer's own trace of it (tensors, bytes, leaves, layout digest)."""
        trace = {int(t["step"]): t for t in ((getattr(rc, "extra", None) or {}).get("step_trace") or [])} if rc is not None else {}
        rows = served_rows(self.com)
        with open(self.out / "step_trace.jsonl", "a") as fh:
            for s, root in enumerate(rec.get("step_roots") or []):
                own = {k: v for k, v in trace.get(s, {}).items() if k not in ("step", "root")}
                fh.write(json.dumps({"window": rec["window"], "run_id": rec["run_id"], "run_root": rec.get("run_root"), "step": s,
                                     "step_root": root, "t_end": self.step_t.get(s), "rows": [list(r) for r in rows.get(s, [])],
                                     **own}) + "\n")

    def _check_finished(self) -> None:
        done = [e for e, led in self.ledgers.items() if led.finished]
        with open(self.out / "requests.jsonl", "a") as fh:
            for eid in done:
                led = self.ledgers.pop(eid)
                rr = self.record.requests.get(eid)
                rec = {"http_id": led.http_id, "engine_id": led.engine_id, "user": led.user, "runner_ids": rr.runner_ids if rr else [],
                       "prompt_ids": led.prompt_ids, "streamed_ids": led.streamed, "ingress": led.ingress, "finish": led.finish,
                       "windows": sorted({r[0] for r in rr.rows}) if rr else [], "rows": rr.rows if rr else [],
                       "committed_seeds": sorted(rr.seeds) if rr else [], "check": self.check(led)}
                self.checked.append({"http_id": led.http_id, "engine_id": led.engine_id, "passed": rec["check"]["passed"]})
                fh.write(json.dumps(rec) + "\n")

    def summary(self) -> dict[str, Any]:
        bad_req = [c for c in self.checked if not c["passed"]]
        bad_win = [r["window"] for r in self.records if not r["passed"]]
        return {"windows": len(self.records), "windows_failed": bad_win, "steps": sum(r["steps"] for r in self.records),
                "requests_checked": len(self.checked), "requests_failed": [c["http_id"] for c in bad_req][:20],
                "unchecked": sorted(self.ledgers), "passed": bool(self.checked) and not bad_req and not bad_win and not self.ledgers}
