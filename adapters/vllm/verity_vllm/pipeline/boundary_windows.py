"""A served window's boundary commitment as the engine core serves (`BoundaryWindows`): the hook on vLLM's scheduler
that takes each request in, keeps the ids each step serves it, closes windows, and commits each window's finished spans
class by class (`boundary_rows.commit`, `write`), under the serve's stop rule where it registers one
(`boundary_stop.filed`)."""
from __future__ import annotations

import atexit
import queue
import secrets
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from multiprocessing import util as mp_util
from pathlib import Path
from typing import Any

from verity.circuits.defs import SpecializedDefinition
from verity.commitments import history as HT

from verity_vllm.pipeline import boundary_tables as BT
from verity_vllm.pipeline.boundary_rows import (
    FINISH,
    FORMAT,
    TABLES_UNDER_A_RULE,
    Layout,
    Span,
    check_template,
    commit,
    run_root,
    sampling_of,
    table_roots,
    write,
)
from verity_vllm.pipeline.boundary_stop import Stop, filed

#: vLLM's RequestStatus names (past ``finished_``) for a `FINISH` name; its FinishReason names are `FINISH`'s own
_STATUS = {"stopped": "stop", "length_capped": "length", "aborted": "abort"}
#: the most instances of the check template the replay blocks one serve keeps hold in all, and the most counts of spans
#: one of them keeps its programs for (`_built`)
KEEP, COUNTS = 1 << 12, 64
#: the parts of a `Layout` that read its replay block and its count of spans alone, and a stop class's more of them
_PARTS, _STOP_PARTS = ("program", "grain", "partition"), ("stop_program", "stop_query", "stop_partition")


def _finish(reason: Any) -> str:
    name = str(getattr(reason, "name", reason)).lower().removeprefix("finished_")
    name = _STATUS.get(name, name)
    return name if name in FINISH else "error"


def _built(layout: Layout, kept: dict[int, tuple[SpecializedDefinition, dict[int, dict[str, Any]]]]) -> Layout:
    """``layout`` holding the parts of it that read no run, from ``kept`` where an earlier layout built them: its replay
    block, which reads its class through ``n_rho`` alone, by ``n_rho``; and its program, grain and partition, and in a
    stop class its stop population's, by that and its count of spans. So only the run and what binds it (each value's
    domain) are a window's own. ``kept`` is one serve's, whose layouts differ in class, whether it is a stop class,
    spans and run alone: its stop rule is one. It holds the replay blocks used last whose instances sum to `KEEP` at
    most, the least recently used leaving first, and for each its programs at the `COUNTS` counts of spans used last; a
    replay block of more than `KEEP` instances is built for its layout alone."""
    if layout.n_rho > KEEP:
        return layout
    replay, counts = kept.pop(layout.n_rho, None) or (layout.replay, {})
    kept[layout.n_rho] = replay, counts
    while sum(kept) > KEEP:
        del kept[next(iter(kept))]
    vars(layout)["replay"] = replay
    names = _PARTS + _STOP_PARTS if layout.stopped else _PARTS
    parts = counts.pop(layout.spans, None) or {}
    vars(layout).update({k: parts[k] for k in names if k in parts})
    counts[layout.spans] = parts | {k: getattr(layout, k) for k in names}
    while len(counts) > COUNTS:
        del counts[next(iter(counts))]
    return layout


class BoundaryWindows:
    """The serving path's boundary commitment, on one scheduler (`install`): every request it adds (`admit`), the ids
    each step serves it (`served`) and the requests it finishes from outside a step (`finished`: an abort). A window
    closes at the first schedule ``window_s`` after it opened (`tick`), at a step that leaves no request running once it
    holds a finished one (``drained``), and at `stop`; its finished spans are committed class by class (`commit`, `write`)
    on a thread of their own, under ``out``/w<window>/, before anything can sample them, and each class's
    ``commit.json`` goes to ``sink`` too. ``check`` is the check template, or a callable giving it on that thread, and
    ``inline`` its `Layout`'s. With ``tables`` (a `boundary_tables.Channel` or `LocalChannel` to rank 0), each class's
    commitment first assigns its spans their slots and waits for their history tables' roots at ``layers``, of
    ``capacity`` leaves. ``stop``: the serve's registered stop rule, with no tables; the spans it ends as they were served
    are committed in the stop classes, (P, T), and the rest by size class without a stop block, under
    ``p<P>-o<O>-nostop`` (`filed`). ``release``: for a serve behind a network certifier, a request's release by its engine
    id, (run, N) or None for a request the certifier didn't release (`serve_admit.release_of`'s, as the serve's admission
    keeps it); each span keeps its own, and one without is in no stop class. Under a rule every line of a window says
    how many of its spans were filed with a stop block and how many without (``filed``), so a rule that ends no response
    as the engine does shows as a count. A failed commit is reported to ``sink`` (``error``) and registers nothing."""

    def __init__(self, out: Path, check: SpecializedDefinition | Callable[[], SpecializedDefinition], per_prompt: int,
                 per_output: int, window_s: float, *, drained: bool = True, inline: bool = False,
                 sink: Callable[[dict], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 serve: str | None = None, tables: Any = None, layers: Sequence[int] = (),
                 capacity: int = HT.CAPACITY, stop: Stop | None = None,
                 release: Callable[[str], tuple[str, int] | None] | None = None) -> None:
        if (tables is None) != (not layers):
            raise ValueError("history tables need both rank 0's channel and their layers")
        if stop is not None and layers:
            raise ValueError(TABLES_UNDER_A_RULE)
        self.out, self.per_prompt, self.per_output = Path(out), int(per_prompt), int(per_output)
        self.window_s, self.drained, self.inline, self.stop_rule = float(window_s), drained, inline, stop
        self.tables, self.layers, self.capacity = tables, tuple(int(x) for x in layers), int(capacity)
        self.sink, self.clock, self.release = sink, clock, release
        self.serve = serve or secrets.token_hex(32)
        self._check, self._check_fn = (check, None) if isinstance(check, SpecializedDefinition) else (None, check)
        self.open: dict[str, dict[str, Any]] = {}
        self.window, self.opened, self.stopped = 0, clock(), False
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="boundary-commit", daemon=True)
        self.thread.start()

    @classmethod
    def of(cls, cfg: Mapping[str, Any], out: Path, sink: Callable[[dict], None] | None = None,
           release: Callable[[str], tuple[str, int] | None] | None = None, public: Sequence[Path] = ()) -> BoundaryWindows:
        """From a serve's configuration: ``check``, a serving template's descriptor id (`check_template`), and
        ``per_prompt``, ``per_output``, ``window_s``, ``drained`` and ``inline``; ``tables``, rank 0's history tables
        (`boundary_tables.Channel.of`: its private directory, never under ``out`` or ``public``); and ``stop`` (`Stop.of`:
        the registered model's stop ids and length and T, the most ids a response has). A stop rule the serve can't
        register (a T the word can't hold, or one beside history tables) is refused here, as the serve starts."""
        t, tc, stop = str(cfg["check"]), cfg.get("tables"), Stop.of(cfg.get("stop"))
        if stop is not None and tc:
            raise ValueError(TABLES_UNDER_A_RULE)
        ch = BT.Channel.of(tc, out, *public) if tc else None
        return cls(out, lambda: check_template(t), int(cfg.get("per_prompt", 0)), int(cfg.get("per_output", 1)),
                   float(cfg.get("window_s", 600)), drained=bool(cfg.get("drained", True)),
                   inline=bool(cfg.get("inline", False)), sink=sink, tables=ch,
                   layers=BT.layers_of(tc["layers"]) if tc else (), capacity=int((tc or {}).get("capacity", HT.CAPACITY)),
                   stop=stop, release=release)

    @property
    def check(self) -> SpecializedDefinition:
        if self._check is None:
            self._check = self._check_fn()
        return self._check

    def admit(self, request: Any) -> None:
        self.open[request.request_id] = {"prompt": tuple(request.prompt_token_ids or ()),
                                         "sampling": sampling_of(request.sampling_params), "served": [], "finish": None,
                                         "release": None if self.release is None else self.release(request.request_id)}

    def served(self, outputs: Any) -> None:
        """A step's outputs (the scheduler's ``update_from_output``'s: engine core outputs by client)."""
        for eco in (outputs or {}).values():
            for o in getattr(eco, "outputs", ()) or ():
                r = self.open.get(o.request_id)
                if r is None:
                    continue
                r["served"].extend(o.new_token_ids or ())
                if o.finish_reason is not None:
                    r["finish"] = _finish(o.finish_reason)
        self._drained()

    def finished(self, request_ids: Any, status: Any) -> None:
        """The scheduler's ``finish_requests``: requests finished from outside a step (a client's abort)."""
        for k in [request_ids] if isinstance(request_ids, str) else list(request_ids or ()):
            r = self.open.get(k)
            if r is not None and r["finish"] is None:
                r["finish"] = _finish(status)
        self._drained()

    def _drained(self) -> None:
        if self.drained and self.open and all(r["finish"] is not None for r in self.open.values()):
            self.close()

    def tick(self) -> None:
        if self.clock() - self.opened >= self.window_s:
            self.close()

    def close(self, every: bool = False) -> None:
        done = [k for k, r in self.open.items() if every or r["finish"] is not None]
        now = self.clock()
        if done:
            spans = [Span(k, self.open[k]["prompt"], self.open[k]["sampling"], tuple(self.open[k]["served"]),
                          self.open[k]["finish"], self.open[k]["release"]) for k in done]
            for k in done:
                del self.open[k]
            self.jobs.put((self.window, self.opened, now, len(self.open), spans))
            self.window += 1
        self.opened = now

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.close(every=True)
        self.jobs.put(None)
        self.thread.join()

    def _run(self) -> None:
        kept: dict = {}
        while (job := self.jobs.get()) is not None:
            window, opened, closed, carried, spans = job
            rule = self.stop_rule
            classes = filed(spans, rule, self.release is not None)
            n = [sum(len(mine) for k, mine in classes.items() if k[2] is held) for held in (True, False)]
            counts = {} if rule is None else {"filed": {"stop": n[0], "nostop": n[1]}}
            for (p, o, held), mine in classes.items():
                unheld = rule is not None and not held
                try:
                    layout = _built(Layout(self.check, self.per_prompt, self.per_output, p, o, len(mine),
                                           run_root(self.serve, window), self.inline, self.layers, self.capacity,
                                           stop=rule, held=held), kept)
                    c = commit(mine, layout, window, tables=self._tables(window, layout, mine))
                    write(c, self.out / f"w{window}" / f"p{p}-o{o}{'-nostop' if unheld else ''}")
                    line = {**c.public(), "serve": self.serve, "open_s": opened, "close_s": closed, "carried": carried,
                            **counts}
                except (ValueError, TypeError, KeyError, OverflowError, struct.error, OSError) as e:  # the sink says why
                    line = {"format": FORMAT, "window": window, "prompt": p, "outputs": o, "spans": len(mine),
                            **({} if rule is None else {"stop": rule.to_json(), "stopped": held}), **counts,
                            "error": f"{type(e).__name__}: {e}"}
                if self.sink is not None:
                    self.sink(line)

    def _tables(self, window: int, layout: Layout, spans: Sequence[Span]) -> dict[str, bytes] | None:
        """The class's history tables' roots from rank 0, once it has the spans' slots; None without tables."""
        if self.tables is None:
            return None
        self.tables.assign(BT.assignment(window, layout.run, layout.partition["program"], layout.prompt, layout.outputs,
                                         layout.layers, layout.capacity,
                                         [(s.engine_id, len(s.prompt), len(s.served), s.finish) for s in spans]))
        return table_roots(self.tables.roots(window, layout.run, layout.prompt, layout.outputs), layout)

    def install(self, sched: Any, hooks: Any) -> BoundaryWindows:
        """Wrap ``sched``'s ``add_request``, ``update_from_output``, ``finish_requests`` and ``schedule`` through ``hooks``
        (an `engine.hooks.Hooks`), and stop at the process's exit: at exit and at a `multiprocessing` child's, which runs
        no `atexit` hook (vLLM's engine core is one). A scheduler that caches prefixes commits the same rows: a span's
        boundary is its own prompt ids and the ids it was served, whichever KV pages its prefill read. With history
        tables it is refused: a table holds its request's latent rows from position 0 (`boundary_tables.Latents`), and
        a request served from the cache computes none before its hit's end. Refused under a stop rule whose model
        length is not the scheduler's ``max_model_len``: the engine ends a response at its own length and the stop block
        at the rule's, so every such response would be filed without a stop block. An honest serve's guard, which no
        verifier relies on: the verifier holds its own copy of the rule (`boundary_rows.layout_of`)."""
        if self.tables is not None and getattr(getattr(sched, "cache_config", None), "enable_prefix_caching", False):
            raise ValueError("the boundary commitment's history tables need prefix caching off "
                             "(--no-enable-prefix-caching): a table holds its request's latent rows from position 0, "
                             "and a request served from the cache computes none before its hit's end")
        served = getattr(sched, "max_model_len", None)
        if self.stop_rule is not None and served != self.stop_rule.max_model_len:
            raise ValueError(f"the stop rule's model length is {self.stop_rule.max_model_len} and the scheduler's "
                             f"max_model_len is {served!r}: serve at the rule's (--max-model-len "
                             f"{self.stop_rule.max_model_len}), or register the rule at the length served")

        def add_request(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(request: Any, *a: Any, **kw: Any) -> Any:
                self.admit(request)
                return orig(request, *a, **kw)
            return wrapped

        def update_from_output(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*a: Any, **kw: Any) -> Any:
                outs = orig(*a, **kw)
                self.served(outs)
                return outs
            return wrapped

        def finish_requests(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(request_ids: Any, finished_status: Any, *a: Any, **kw: Any) -> Any:
                ids = request_ids if isinstance(request_ids, str) else list(request_ids or ())
                out = orig(ids, finished_status, *a, **kw)
                self.finished(ids, finished_status)
                return out
            return wrapped

        def schedule(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*a: Any, **kw: Any) -> Any:
                self.tick()
                return orig(*a, **kw)
            return wrapped

        hooks.wrap(sched, "add_request", add_request)
        hooks.wrap(sched, "update_from_output", update_from_output)
        hooks.wrap(sched, "finish_requests", finish_requests)
        hooks.wrap(sched, "schedule", schedule)
        atexit.register(self.stop)
        mp_util.Finalize(None, self.stop, exitpriority=100)
        return self


__all__ = ["BoundaryWindows"]
