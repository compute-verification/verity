"""A two-stage window's history tables (`boundary_rows`): each finished request's latent rows at each layer, one
`commit.history_tables` table a (span, layer), committed at serve beside its ``b.ingress`` and ``b.egress``.

**Names.** A table is named from public items alone (`name`): its window's run, its span's class (P, O) and slot r in the
class, and its layer, never the engine's id for the request, which only the prover's index holds. So the challenger
derives every table's name and domain from the class's ``commit.json``, as it does each boundary value's.

**Rows.** On one rank (rank 0: the latent projections are replicated), `StepHook` follows the model runner's
``execute_model``: once a step's forward has written the KV cache, it reads each MLA layer's rows at the step's slots, the
layer's entry of the step's ``slot_mapping`` (the one `kv_slots_source` reads, which the layer's cache write read), and
appends each request's to its rows in `Latents`, by layer, at the positions the step computed. Those rows are
``DsaLatentRow_v1``'s output, ``[kv_c | k_pe]`` in bf16: the hook refuses a layer whose cache holds any other row.

**Tables.** A table's domain binds its name, and its name its class and slot, which are fixed only once the request's
window closes, so the tables are built then (`Builder.build`): the window's boundary commitment assigns each class's spans
their slots (`assignment`), and rank 0 builds each span's tables over its rows, under fresh salts, and answers with each
table's entry (`history.HistoryTable.entry`), which the class's registration lists before any coin. The two processes talk
through `Channel` (files in a private directory on the node) or, in one process, `LocalChannel`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.commitments import history as HT
from verity_vllm.commit import history_tables as H

__all__ = ["ASSIGN", "CAPACITY", "KEY", "ROOTS", "WORDS", "Builder", "Channel", "Latents", "LocalChannel", "StepHook",
           "assignment", "computed", "key", "mla_layers", "name", "private"]

#: a span's key in its tables' names (`history_tables.name`): the window's run, the class and the slot
KEY = "{run}/p{prompt}-o{outputs}/s{slot}"
WORDS, CAPACITY = H.WORDS, HT.CAPACITY
ASSIGN, ROOTS = "verity-vllm/boundary-tables/assign/v1", "verity-vllm/boundary-tables/roots/v1"
#: the KV cache dtypes under which an MLA layer's cache holds `WORDS` bf16 words a row (vLLM's ``fused_norm_rope``)
BF16_CACHES = ("auto", "bfloat16")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def key(run: str, prompt: int, outputs: int, slot: int) -> str:
    """Slot ``slot``'s key in class (``prompt``, ``outputs``) of the window whose run is ``run``."""
    for k, v in (("prompt", prompt), ("outputs", outputs), ("slot", slot)):
        if type(v) is not int or v < 0:
            raise ValueError(f"a table's {k} is a natural number, not {v!r}")
    if not isinstance(run, str) or not run:
        raise ValueError("a table's name binds its window's run: give it")
    return KEY.format(run=run, prompt=prompt, outputs=outputs, slot=slot)


def name(run: str, prompt: int, outputs: int, slot: int, layer: int) -> str:
    """The table of slot ``slot`` of class (``prompt``, ``outputs``) in the window of run ``run`` at layer ``layer``."""
    return H.name(key(run, prompt, outputs, slot), layer)


def computed(prompt: int, served: int, finish: str | None) -> tuple[int, int]:
    """The positions a request's tables hold, at least and at most: a request that stopped or hit its cap computed every
    position but its last id's; one aborted, errored or still running computed some prefix of its prompt and ids."""
    n = int(prompt) + int(served)
    return (n - 1, n - 1) if finish in ("stop", "length") and n else (0, n)


def assignment(window: int, run: str, program: str, prompt: int, outputs: int, layers: Sequence[int], capacity: int,
               spans: Sequence[tuple[str, int, int, str | None]]) -> dict[str, Any]:
    """A class's request for its tables: each span's slot, its engine id, prompt and served lengths and finish
    (``spans``, in slot order), and what every table's domain binds. Private: it names the engine's ids."""
    return {"format": ASSIGN, "window": int(window), "run": run, "program": program, "prompt": int(prompt),
            "outputs": int(outputs), "layers": [int(x) for x in layers], "capacity": int(capacity),
            "spans": [{"engine_id": str(e), "prompt": int(p), "served": int(s), "finish": f} for e, p, s, f in spans]}


def mla_layers(model: Any) -> dict[int, tuple[str, Any]]:
    """Layer index -> (layer name, KV cache) of each MLA layer of ``model`` (the modules the engine bound a cache to,
    `kv_slots_source.kv_layers`, with an MLA latent), refused unless its cache holds `WORDS` bf16 words a row."""
    from verity_vllm.acquire.sources.kv_slots_source import kv_layers

    out: dict[int, tuple[str, Any]] = {}
    for lname, m in kv_layers(model).items():
        if getattr(m, "kv_lora_rank", None) is None:
            continue
        layer, cache, dtype = _LAYER.search(lname), m.kv_cache, str(getattr(m, "kv_cache_dtype", "auto"))
        width = int(m.kv_lora_rank) + int(getattr(m, "qk_rope_head_dim", 0))
        if layer is None:
            raise ValueError(f"MLA layer {lname}: its name holds no layer index")
        if dtype not in BF16_CACHES or str(cache.dtype) != "torch.bfloat16" or width != WORDS or cache.shape[-1] != WORDS \
                or cache.dim() != 3:
            raise ValueError(f"MLA layer {lname}: its cache ({dtype}, {cache.dtype}, {tuple(cache.shape)}) doesn't hold "
                             f"DsaLatentRow_v1's {WORDS} bf16 words a row; serve with --kv-cache-dtype auto")
        if int(layer.group(1)) in out:
            raise ValueError(f"MLA layers {out[int(layer.group(1))][0]} and {lname} share layer {layer.group(1)}")
        out[int(layer.group(1))] = (lname, cache)
    if not out:
        raise ValueError("the model has no MLA layer with a bound KV cache")
    return dict(sorted(out.items()))


class Latents:
    """Rank 0's latent rows of each request it serves, by layer, in position order: `add` appends a step's, a step that
    starts below a request's length (a recompute after a preemption) replacing its rows from there; `take` hands a
    finished request's over. Thread-safe: the step hook adds, the builder takes. A hook failure (`fail`) is kept and
    refuses every later build."""

    def __init__(self, layers: Iterable[int], words: int = WORDS) -> None:
        self.layers, self.words = tuple(int(x) for x in layers), int(words)
        self._rows: dict[str, dict[int, list[np.ndarray]]] = {}
        self._length: dict[str, int] = {}
        self._lock = threading.Lock()
        self.failed: str | None = None

    def add(self, request: str, start: int, rows: Mapping[int, np.ndarray]) -> None:
        """Rows ``rows`` (by layer, each ``(k, words)`` u16) of request ``request`` at positions ``start`` on."""
        shapes = {np.shape(r) for r in rows.values()}
        if set(rows) != set(self.layers) or len(shapes) != 1 or next(iter(shapes))[1:] != (self.words,):
            raise ValueError(f"request {request}: one (k, {self.words}) block of rows at each layer of {list(self.layers)}, "
                             f"not {sorted(rows)} of shapes {sorted(shapes)}")
        k = next(iter(shapes))[0]
        with self._lock:
            have = self._length.get(request, 0)
            if start > have:
                raise ValueError(f"request {request}: rows at {start}, past the {have} it holds")
            mine = self._rows.setdefault(request, {x: [] for x in self.layers})
            if start < have:
                for x in self.layers:
                    mine[x] = [np.concatenate(mine[x])[:start]]
            for x in self.layers:
                mine[x].append(np.array(rows[x], dtype=np.uint16, copy=True))
            self._length[request] = start + k

    def length(self, request: str) -> int:
        with self._lock:
            return self._length.get(request, 0)

    def take(self, request: str) -> dict[int, np.ndarray]:
        """Request ``request``'s rows by layer, no longer held here."""
        with self._lock:
            mine = self._rows.pop(request, None)
            self._length.pop(request, None)
        empty = np.zeros((0, self.words), dtype=np.uint16)
        return {x: (np.concatenate(mine[x]) if mine and mine[x] else empty) for x in self.layers}

    def requests(self) -> list[str]:
        with self._lock:
            return sorted(self._rows)

    def fail(self, why: str) -> None:
        self.failed = self.failed or why


class StepHook:
    """`Latents.add` from each step of a model runner (vLLM's Model Runner V2: `install` wraps its ``execute_model``):
    after the forward, ``execute_model_state`` holds the step's batch (``input_batch``: its requests, query start
    locations and computed tokens) and each layer's slots (``slot_mappings_by_layer``); each request's rows at each MLA layer
    are that layer's cache rows at its tokens' slots, at positions from its computed tokens. A step it can't read fails
    ``latents`` (`Latents.fail`), and serving goes on."""

    def __init__(self, model: Any, latents: Latents | None = None) -> None:
        import torch

        self.torch, self.layers = torch, mla_layers(model)
        self.latents = latents if latents is not None else Latents(self.layers)
        if tuple(self.layers) != self.latents.layers:
            raise ValueError(f"the model's MLA layers {list(self.layers)} are not the tables' {list(self.latents.layers)}")
        self.steps = 0

    def after(self, runner: Any, scheduler_output: Any) -> None:
        if self.latents.failed or not int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0):
            return
        try:
            if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
                raise ValueError("a step with draft tokens: a rejected draft's row is no position the request keeps")
            self.read(runner.execute_model_state)
        except Exception as e:  # noqa: BLE001 - the builder refuses every table after it, and says why
            self.latents.fail(f"step {self.steps}: {type(e).__name__}: {e}")
        self.steps += 1

    def read(self, state: Any) -> None:
        torch, ib, slots = self.torch, state.input_batch, state.slot_mappings_by_layer
        n = len(ib.req_ids)
        qsl = np.asarray(ib.query_start_loc_np[:n + 1], dtype=np.int64)
        nct = np.asarray(ib.num_computed_tokens_np[:n], dtype=np.int64)
        got, bad = [], []
        for _layer, (lname, cache) in self.layers.items():
            s = slots[lname][:int(qsl[n])].to(torch.int64)
            bad.append((s < 0).any())
            got.append(cache[s.clamp(min=0) // cache.shape[1], s.clamp(min=0) % cache.shape[1]])
        if bool(torch.stack(bad).any()):
            raise ValueError("a scheduled token has no slot (-1): its row went to no cache position")
        host = torch.stack(got).view(torch.int16).cpu().numpy().view(np.uint16)
        for i, rid in enumerate(ib.req_ids[:n]):
            a, b = int(qsl[i]), int(qsl[i + 1])
            self.latents.add(str(rid), int(nct[i]), {x: host[j, a:b] for j, x in enumerate(self.layers)})

    def install(self, runner: Any, hooks: Any) -> StepHook:
        """Wrap ``runner``'s ``execute_model`` through ``hooks`` (an `engine.hooks.Hooks`); a dummy run reads nothing."""
        me = self

        def execute_model(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(scheduler_output: Any, *a: Any, **kw: Any) -> Any:
                out = orig(scheduler_output, *a, **kw)
                if not kw.get("dummy_run"):
                    me.after(runner, scheduler_output)
                return out
            return wrapped

        hooks.wrap(runner, "execute_model", execute_model)
        return self


class Builder:
    """Rank 0's tables: `build` answers a class's `assignment` with each span's tables over its rows in ``latents``, one a
    layer in slot order, each built by ``make`` (program, name, words, capacity; by default a `history_tables_device.Table`
    on ``device``, its salts under a row key fresh for the class)."""

    def __init__(self, latents: Latents, make: Callable[[str, str, int, int], Any] | None = None, device: Any = None) -> None:
        self.latents, self.make, self.device = latents, make, device

    def _maker(self) -> Callable[[str, str, int, int], Any]:
        if self.make is not None:
            return self.make
        from verity_vllm.commit import history_tables_device as HTD

        row_key = os.urandom(32)
        return lambda p, n, w, c: HTD.Table(p, n, w, c, key=row_key, device=self.device)

    def build(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """The class's tables' entries (`history.HistoryTable.entry`), or why there are none (``error``)."""
        head = {"format": ROOTS, **{k: doc.get(k) for k in ("window", "run", "prompt", "outputs")}}
        try:
            return {**head, "tables": self._tables(doc)}
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            for s in doc.get("spans") or ():
                self.latents.take(str(s.get("engine_id")))
            return {**head, "error": f"{type(e).__name__}: {e}"}

    def _tables(self, doc: Mapping[str, Any]) -> list[dict[str, Any]]:
        if doc.get("format") != ASSIGN:
            raise ValueError(f"not a {ASSIGN} assignment")
        if self.latents.failed:
            raise ValueError(f"rank 0's latent rows are incomplete: {self.latents.failed}")
        if [int(x) for x in doc["layers"]] != list(self.latents.layers):
            raise ValueError(f"the class's tables are at layers {doc['layers']}, rank 0 holds {list(self.latents.layers)}")
        make, run, p, o, cap = self._maker(), doc["run"], int(doc["prompt"]), int(doc["outputs"]), int(doc["capacity"])
        out = []
        for r, s in enumerate(doc["spans"]):
            rows = self.latents.take(str(s["engine_id"]))
            lo, hi = computed(s["prompt"], s["served"], s["finish"])
            n = len(rows[self.latents.layers[0]])
            if not lo <= n <= hi:
                raise ValueError(f"slot {r}: rank 0 holds {n} positions of it, not the {lo} to {hi} its lengths and finish "
                                 f"({s['finish']}) give")
            for x in self.latents.layers:
                t = make(doc["program"], name(run, p, o, r, x), WORDS, cap)
                t.append(rows[x])
                out.append(t.entry())
        return out


def _stem(window: int, run: str, prompt: int, outputs: int) -> str:
    return f"w{int(window)}-{run[:16]}-p{int(prompt)}-o{int(outputs)}.json"


def private(directory: Path) -> Path:
    """``directory``, made if need be, readable by its owner only."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    private(path.parent.parent)
    private(path.parent)
    tmp = path.with_name(f".{path.name}")
    tmp.write_text(json.dumps(doc) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


class Channel:
    """The window's boundary commitment and rank 0, in two processes on the node: `assign` writes a class's assignment
    under ``directory``/assign/, rank 0's `serve` answers each under ``directory``/roots/, and `roots` waits for the
    answer. ``directory`` is private: an assignment names the engine's ids."""

    def __init__(self, directory: Path, *, timeout_s: float = 900.0, poll_s: float = 0.05,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.dir, self.timeout_s, self.poll_s, self.sleep, self.clock = Path(directory), timeout_s, poll_s, sleep, clock

    def assign(self, doc: Mapping[str, Any]) -> None:
        _write(self.dir / "assign" / _stem(doc["window"], doc["run"], doc["prompt"], doc["outputs"]), doc)

    def roots(self, window: int, run: str, prompt: int, outputs: int) -> dict[str, Any]:
        path, t0 = self.dir / "roots" / _stem(window, run, prompt, outputs), self.clock()
        while not path.exists():
            if self.clock() - t0 > self.timeout_s:
                raise TimeoutError(f"rank 0 answered no tables for window {window}'s class ({prompt}, {outputs}) in "
                                   f"{self.timeout_s:g} s")
            self.sleep(self.poll_s)
        return json.loads(path.read_text())

    def serve(self, builder: Builder) -> int:
        """Answer every assignment not yet answered, oldest first. -> how many."""
        done = 0
        for a in sorted((self.dir / "assign").glob("w*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name)):
            out = self.dir / "roots" / a.name
            if not out.exists():
                _write(out, builder.build(json.loads(a.read_text())))
                done += 1
        return done

    def start(self, builder: Builder) -> threading.Event:
        """`serve` on a thread of its own every ``poll_s`` until the returned event is set."""
        stop = threading.Event()

        def run() -> None:
            while not stop.is_set():
                self.serve(builder)
                stop.wait(self.poll_s)

        threading.Thread(target=run, name="boundary-tables", daemon=True).start()
        return stop


class LocalChannel:
    """`Channel` in one process: `assign` builds at once."""

    def __init__(self, builder: Builder) -> None:
        self.builder, self.done = builder, {}

    def assign(self, doc: Mapping[str, Any]) -> None:
        self.done[_stem(doc["window"], doc["run"], doc["prompt"], doc["outputs"])] = self.builder.build(doc)

    def roots(self, window: int, run: str, prompt: int, outputs: int) -> dict[str, Any]:
        return self.done.pop(_stem(window, run, prompt, outputs))
