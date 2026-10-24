"""The Commit side of `verity_vllm.pipeline.serve_openai`: the committer's windows over the serving engine, and each
request's record of what was served against what the Commit holds of it.

A window (`CommitWindows`) opens at the first engine step after the engine was idle and closes when the engine is idle
again (once it holds ``idle_close_steps`` steps), or after ``window_steps`` steps; each is one Commit, with its own run
root. At each close it reads back, from the committed bytes, each request's rows of the embedding's input ids and of the
sampled ids, at the rows the model runner's input batch gave it at that step, and `check_request` holds a finished
request's committed ids to what was served.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The embedding's input ids, by execution: the compiled path's name, then the eager path's (`NativeHostCommitter`'s
#: ``<module>/in`` pre-hook on the ``VocabParallelEmbedding``).
IDS_NAMES = ("c00.input/ids", "model.embed_tokens/in")
SAMPLED_NAME = "runner.sampler/sampled_token_ids"
_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def known_id(internal: str, known: Mapping[str, Any]) -> str | None:
    """The engine id we admitted for a model runner's request id: vLLM runs a request under ``f"{id}-{8 hex}"``."""
    if internal in known:
        return internal
    base = _SUFFIX.sub("", internal, count=1)
    return base if base in known else None


def committed_ints(committer: Any, step: int, name: str) -> list[int] | None:
    """The integer tensor ``name`` as committed at ``step``, read back from the committed bytes the committer retains
    (its host copy, the packed step stream, or the retained GPU-tree block); None when the step committed no such tensor."""
    import numpy as np

    meta = next((m for m in committer._layouts.get(step, ()) if m.name == name), None)
    if meta is None:
        return None
    raw: Any = None
    blk = (getattr(committer, "_gpu_blocks", {}) or {}).get(step, (None, 0))[0]
    if blk is not None:
        off = int(meta.dev_off) if int(getattr(meta, "dev_off", -1)) >= 0 else int(meta.stream_off)
        raw = blk[off: off + meta.nbytes].cpu().numpy().tobytes()
    elif getattr(meta, "host", None) is not None:
        raw = bytes(meta.host)
    elif step in (getattr(committer, "_streams", {}) or {}) and int(meta.stream_off) >= 0:
        raw = bytes(committer._streams[step][meta.stream_off: meta.stream_off + meta.nbytes])
    if raw is None:
        return None
    return np.frombuffer(raw, dtype=np.dtype(str(meta.dtype).rsplit(".", 1)[-1])).astype(np.int64).tolist()


def step_rows(runner: Any) -> list[tuple[str, int, int]] | None:
    """The model runner's rows of the step it just ran (Model Runner V2's ``execute_model_state.input_batch``): per
    request, in the batch's order, its runner id and its token rows [lo, hi) of the flattened step; the sampled ids are
    indexed by the same order."""
    ib = getattr(getattr(runner, "execute_model_state", None), "input_batch", None)
    if ib is None:
        return None
    n = int(ib.num_reqs)
    qsl = [int(x) for x in list(ib.query_start_loc_np[: n + 1])]
    return [(str(ib.req_ids[i]), qsl[i], qsl[i + 1]) for i in range(n)]


@dataclass
class Ledger:
    """One request's record: what was served and what the Commit holds of it."""

    http_id: str
    engine_id: str
    prompt_ids: list[int]
    streamed: list[int] = field(default_factory=list)
    runner_ids: list[str] = field(default_factory=list)
    inputs: list[int] = field(default_factory=list)
    sampled: list[int] = field(default_factory=list)
    steps: list[list[int]] = field(default_factory=list)     # [window, step, lo, hi, row]
    finished: bool = False
    aborted: bool = False
    errors: list[str] = field(default_factory=list)


def check_request(led: Ledger) -> dict[str, Any]:
    """The committed ids against the served ones: the input ids are the prompt ids then every streamed id but the last
    (each sampled token is the next step's input; the last is never fed back), the sampled ids the streamed ids."""
    want_inputs = led.prompt_ids + led.streamed[:-1]
    first = None
    for what, got, want in (("inputs", led.inputs, want_inputs), ("sampled", led.sampled, led.streamed)):
        if got != want and first is None:
            k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            first = {"what": what, "at": k, "committed": got[k:k + 4], "served": want[k:k + 4], "lengths": [len(got), len(want)]}
    passed = first is None and not led.errors and not led.aborted and bool(led.streamed)
    return {"passed": passed, "prompt_ids": len(led.prompt_ids), "streamed_ids": len(led.streamed), "committed_inputs": len(led.inputs),
            "committed_sampled": len(led.sampled), "first_mismatch": first, "errors": led.errors[:5], "aborted": led.aborted}


class CommitWindows:
    """The committer's windows over a serving engine, and every request's `Ledger`. ``committer`` is a
    `NativeHostCommitter` (or its offline stand-in in the tests: ``reset``, ``install``, ``uninstall``, ``finalize``,
    ``_layouts``, ``_step_index``); ``runner`` the model runner whose ``execute_model`` it patches."""

    def __init__(self, committer: Any, runner: Any, out: Path, *, run_id: str, window_steps: int,
                 ids_names: Sequence[str] = IDS_NAMES, sampled_name: str = SAMPLED_NAME,
                 idle_close_steps: int = 1, clock: Callable[[], float] = time.perf_counter) -> None:
        from verity_vllm.engine.hooks import Hooks

        self.com, self.runner, self.out, self.clock = committer, runner, Path(out), clock
        self.idle_close_steps = int(idle_close_steps)
        self.open_s = 0.0
        self.run_id, self.window_steps = run_id, int(window_steps)
        self.ids_names, self.sampled_name = tuple(ids_names), sampled_name
        self._Hooks = Hooks
        self.hooks: Any = None
        self.window = -1
        self.rows: dict[int, list[tuple[str, int, int]]] = {}
        self.ledgers: dict[str, Ledger] = {}
        self.records: list[dict[str, Any]] = []
        self.checked: list[dict[str, Any]] = []
        self.is_open = False

    # -- the engine loop's calls
    def admitted(self, adm: Any) -> None:
        """``adm`` a `verity_vllm.pipeline.serve_openai.Admission`."""
        self.ledgers[adm.engine_id] = Ledger(adm.http_id, adm.engine_id, list(adm.prompt_ids))

    def streamed(self, engine_id: str, token_ids: list[int], finished: bool) -> None:
        led = self.ledgers.get(engine_id)
        if led is not None:
            led.streamed.extend(int(t) for t in token_ids)
            led.finished = led.finished or finished

    def aborted(self, engine_id: str) -> None:
        led = self.ledgers.get(engine_id)
        if led is not None:
            led.aborted = led.finished = True

    def idle(self) -> None:
        """The engine has nothing to run: close the open window once it holds ``idle_close_steps`` steps."""
        if self.is_open and len(self.rows) >= self.idle_close_steps:
            self.close()

    def before_step(self) -> None:
        if self.is_open and len(self.rows) >= self.window_steps:
            self.close()
        if not self.is_open:
            self.open()

    # -- windows
    def open(self) -> None:
        t0 = self.clock()
        self.window += 1
        self.rows = {}
        self.com.reset()
        self.com.install()
        self.open_s = self.clock() - t0
        self.hooks = self._Hooks(tag=f"serve-openai:{self.run_id}")
        me = self

        def make(orig: Callable[..., Any]) -> Callable[..., Any]:
            def execute_model(*a: Any, **kw: Any) -> Any:
                before = int(me.com._step_index)
                res = orig(*a, **kw)
                if int(me.com._step_index) != before:
                    rows = step_rows(me.runner)
                    me.rows[int(me.com._step_index) - 1] = rows if rows is not None else []
                return res
            return execute_model

        self.hooks.wrap(self.runner, "execute_model", make)
        self.is_open = True

    def close(self) -> dict[str, Any]:
        t0 = self.clock()
        self.hooks.uninstall()
        rc = self.com.finalize() if int(self.com._step_index) else None    # a window whose one step() only drained outputs
        self.com.uninstall()
        self.is_open = False
        rec = self._absorb(rc) if rc is not None else {"window": self.window, "run_id": f"{self.run_id}/w{self.window}", "steps": 0,
                                                       "empty": True, "requests": [], "passed": True}
        rec["seconds"] = {"open": round(self.open_s, 4), "close": round(self.clock() - t0, 4)}
        self.records.append(rec)
        with open(self.out / "windows.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._check_finished()
        return rec

    def _ids_name(self) -> str | None:
        names = {m.name for metas in self.com._layouts.values() for m in metas}
        return next((n for n in self.ids_names if n in names), None)

    def _absorb(self, rc: Any) -> dict[str, Any]:
        """The window's committed ids into each request's ledger, and the window's record."""
        ids_name, errors = self._ids_name(), []
        if ids_name is None and self.rows:
            errors.append(f"no committed input ids under any of {list(self.ids_names)}")
        held: list[str] = []
        for s in sorted(self.rows):
            rows = self.rows[s]
            if not rows:
                errors.append(f"step {s}: no input batch rows")
                continue
            ids = committed_ints(self.com, s, ids_name) if ids_name else None
            sampled = committed_ints(self.com, s, self.sampled_name)
            if ids is None or sampled is None:
                errors.append(f"step {s}: {'no ' + str(ids_name) if ids is None else ''} {'no ' + self.sampled_name if sampled is None else ''}".strip())
                continue
            per_row = len(sampled) // len(rows) if len(sampled) >= len(rows) else 0
            for i, (rid, lo, hi) in enumerate(rows):
                eid = known_id(rid, self.ledgers)
                if eid is None:
                    errors.append(f"step {s}: runner request {rid} is no admitted request")
                    continue
                led = self.ledgers[eid]
                if rid not in led.runner_ids:
                    led.runner_ids.append(rid)
                if hi > len(ids) or per_row == 0:
                    led.errors.append(f"window {self.window} step {s}: rows [{lo}, {hi}) / row {i} outside the committed ids")
                    continue
                led.inputs.extend(ids[lo:hi])
                if len(led.inputs) >= len(led.prompt_ids):       # a chunked prefill's earlier chunk samples nothing that is served
                    led.sampled.append(sampled[i * per_row])
                led.steps.append([self.window, s, lo, hi, i])
                if eid not in held:
                    held.append(eid)
        extra = getattr(rc, "extra", None) or {}
        ident = extra.get("identity_check") or {}
        com_errors = list(extra.get("errors") or [])
        root = bytes(rc.run_root or b"")
        commit_ok = bool(root) and not com_errors and int(extra.get("errors_n") or 0) == 0 and ident.get("ok") is not False
        return {"window": self.window, "run_id": f"{self.run_id}/w{self.window}", "run_root": root.hex(), "steps": len(rc.steps),
                "step_roots": [bytes(s.root).hex() for s in rc.steps], "committed_bytes": sum(int(s.n_bytes) for s in rc.steps),
                "ids_name": ids_name, "sampled_name": self.sampled_name, "requests": held,
                "rows": {str(s): [list(r) for r in rows] for s, rows in sorted(self.rows.items())},
                "commit": {"passed": commit_ok, "errors": com_errors[:10], "errors_n": int(extra.get("errors_n") or 0),
                           "identity_check": {k: ident.get(k) for k in ("ok", "steps_checked", "violations") if k in ident}},
                "readback_errors": errors[:10], "passed": commit_ok and not errors}

    def _check_finished(self) -> None:
        done = [e for e, led in self.ledgers.items() if led.finished]
        with open(self.out / "requests.jsonl", "a") as fh:
            for eid in done:
                led = self.ledgers.pop(eid)
                rec = {"http_id": led.http_id, "engine_id": led.engine_id, "runner_ids": led.runner_ids, "prompt_ids": led.prompt_ids,
                       "streamed_ids": led.streamed, "windows": sorted({w for w, *_ in led.steps}), "steps": led.steps,
                       "check": check_request(led)}
                self.checked.append({"http_id": led.http_id, "engine_id": led.engine_id, "passed": rec["check"]["passed"]})
                fh.write(json.dumps(rec) + "\n")

    def summary(self) -> dict[str, Any]:
        bad_req = [c for c in self.checked if not c["passed"]]
        bad_win = [r["window"] for r in self.records if not r["passed"]]
        return {"windows": len(self.records), "windows_failed": bad_win, "steps": sum(r["steps"] for r in self.records),
                "requests_checked": len(self.checked), "requests_failed": [c["http_id"] for c in bad_req][:20],
                "unchecked": sorted(self.ledgers), "passed": bool(self.checked) and not bad_req and not bad_win and not self.ledgers}
