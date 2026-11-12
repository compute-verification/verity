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

**Length.** A request's exact length is private beyond its class, and a table's published strings would show it: the
verifier computes the placeholder past a held run's count, and the placeholder is public. So every table of a class holds
`class_rows` C rows, a function of the class alone: its request's own L rows, then zero rows to C, each under a fresh salt
like the rest (`fill`, which the interior's tables take too). A read past L returns the zero row.

**What round 2 opens** (`Kept`): each table's rows and salts as built, in files on the node readable by their owner only,
never under a run's directory or the serve's records, which leave it. `load` gives a served table by its name, for the
interior to open a drawn span's table under round 1's root; `Kept.span` reads a span's rows at a range of positions with
their salts and public strings, and `Kept.table` rebuilds a table from them, held to its root. The range is an argument:
the committed step schedule it is to come from isn't there yet.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments import history as HT
from verity_vllm.commit import history_tables as H
from verity_vllm.records.boundary_tables import ASSIGN, ROOTS, TableAssignment, TableRoots

__all__ = ["ASSIGN", "CAPACITY", "KEY", "ROOTS", "WORDS", "Builder", "Channel", "Kept", "Latents", "LocalChannel", "Served",
           "StepHook", "assignment", "class_rows", "computed", "fill", "key", "layers_of", "load", "mla_layers", "name",
           "private", "private_root"]

#: a span's key in its tables' names (`history_tables.name`): the window's run, the class and the slot
KEY = "{run}/p{prompt}-o{outputs}/s{slot}"
WORDS, CAPACITY = H.WORDS, HT.CAPACITY
#: the KV cache dtypes under which an MLA layer's cache holds `WORDS` bf16 words a row (vLLM's ``fused_norm_rope``)
BF16_CACHES = ("auto", "bfloat16")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_NAME = re.compile(r"history/(?P<run>[^/]+)/p(?P<prompt>\d+)-o(?P<outputs>\d+)/s(?P<slot>\d+)/"
                   r"model\.layers\.(?P<layer>\d+)\.self_attn\.latent")
_CLASS_DIR = re.compile(r"w(\d+)-")


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


def class_rows(prompt: int, outputs: int, capacity: int = CAPACITY) -> int:
    """C, the rows every table of class (``prompt``, ``outputs``) holds: the class's cap on positions, ``prompt`` +
    ``outputs``, since a request computes at most its prompt's and its ids' (`computed`), and at most the tables'
    ``capacity``. A function of the class alone, so a table's published strings show nothing of its request's length."""
    for k, v in (("prompt", prompt), ("outputs", outputs)):
        if type(v) is not int or v < 1:
            raise ValueError(f"a class's {k} cap is a positive integer, not {v!r}")
    if not HT.cap_ok(capacity):
        raise ValueError(f"a table's capacity is a power of two of at least 2, not {capacity!r}")
    return min(prompt + outputs, capacity)


def fill(rows: Any, prompt: int, outputs: int, capacity: int = CAPACITY) -> np.ndarray:
    """A span's table rows at one layer as its class publishes them: its own rows ``rows`` (``(L, words)`` u16), then zero
    rows to `class_rows` C, refused past C. Each table appends every one of them under a fresh salt, so a zero row's string
    is a hiding commitment like a computed row's; the placeholder at L instead is public, and would show L. A read past L
    returns the zero row."""
    r = np.asarray(rows, dtype=np.uint16)
    c = class_rows(prompt, outputs, capacity)
    if r.ndim != 2 or len(r) > c:
        raise ValueError(f"a span's rows are at most the {c} its class ({prompt}, {outputs}) holds at capacity {capacity}, "
                         f"not shape {r.shape}")
    return np.concatenate([r, np.zeros((c - len(r), r.shape[1]), dtype=np.uint16)])


def assignment(window: int, run: str, program: str, prompt: int, outputs: int, layers: Sequence[int], capacity: int,
               spans: Sequence[tuple[str, int, int, str | None]]) -> dict[str, Any]:
    """A class's request for its tables (`TableAssignment`): each span's slot, its engine id, prompt and served lengths
    and finish (``spans``, in slot order), and what every table's domain binds. Private: it names the engine's ids."""
    return TableAssignment(int(window), run, program, int(prompt), int(outputs), tuple(int(x) for x in layers),
                           int(capacity), tuple((str(e), int(p), int(s), f) for e, p, s, f in spans)).dump()


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
    are that layer's cache rows at its tokens' slots, at positions from its computed tokens; at ``layers`` (default:
    every MLA layer). A step it can't read fails ``latents`` (`Latents.fail`), and serving goes on."""

    def __init__(self, model: Any, latents: Latents | None = None, layers: Sequence[int] | None = None) -> None:
        import torch

        found = mla_layers(model)
        want = tuple(found) if layers is None else tuple(int(x) for x in layers)
        if missing := [x for x in want if x not in found]:
            raise ValueError(f"layers {missing} aren't MLA layers of the model, whose are {list(found)}")
        self.torch, self.layers = torch, {x: found[x] for x in want}
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
    """Rank 0's tables: `build` answers a class's `assignment` with each span's tables over its rows in ``latents`` filled
    to the class's C (`fill`), one a layer in slot order, each built by ``make`` (program, name, words, capacity; by default
    a `history_tables_device.Table` on ``device``, its salts under a row key fresh for the class), and kept in ``keep``
    before its entry is given."""

    def __init__(self, latents: Latents, make: Callable[[str, str, int, int], Any] | None = None, device: Any = None,
                 keep: Kept | None = None) -> None:
        self.latents, self.make, self.device, self.keep = latents, make, device, keep

    def _maker(self) -> Callable[[str, str, int, int], Any]:
        if self.make is not None:
            return self.make
        from verity_vllm.commit import history_tables_device as HTD

        row_key = os.urandom(32)
        return lambda p, n, w, c: HTD.Table(p, n, w, c, key=row_key, device=self.device)

    def build(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """The class's tables' entries (`history.HistoryTable.entry`), or why there are none (`TableRoots`)."""
        head = {k: doc.get(k) for k in ("window", "run", "prompt", "outputs")}
        try:
            return TableRoots(**head, tables=tuple(self._tables(TableAssignment.of(doc, ASSIGN))), error=None).dump()
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            for s in doc.get("spans") or ():
                self.latents.take(str(s.get("engine_id")))
            if self.keep is not None and isinstance(head["run"], str):
                self.keep.drop(*head.values())
            return TableRoots(**head, tables=None, error=f"{type(e).__name__}: {e}").dump()

    def _tables(self, a: TableAssignment) -> list[dict[str, Any]]:
        if self.latents.failed:
            raise ValueError(f"rank 0's latent rows are incomplete: {self.latents.failed}")
        if a.layers != self.latents.layers:
            raise ValueError(f"the class's tables are at layers {list(a.layers)}, rank 0 holds {list(self.latents.layers)}")
        make, out = self._maker(), []
        for r, (engine_id, prompt, served, finish) in enumerate(a.spans):
            rows = self.latents.take(engine_id)
            lo, hi = computed(prompt, served, finish)
            n = len(rows[self.latents.layers[0]])
            if not lo <= n <= hi:
                raise ValueError(f"slot {r}: rank 0 holds {n} positions of it, not the {lo} to {hi} its lengths and finish "
                                 f"({finish}) give")
            for x in self.latents.layers:
                t = make(a.program, name(a.run, a.prompt, a.outputs, r, x), WORDS, a.capacity)
                t.append(fill(rows[x], a.prompt, a.outputs, a.capacity))
                if self.keep is not None:
                    self.keep.write(a.window, a.run, a.prompt, a.outputs, r, x, a.program, t)
                out.append(t.entry())
        return out


def _stem(window: int, run: str, prompt: int, outputs: int) -> str:
    return f"w{int(window)}-{run[:16]}-p{int(prompt)}-o{int(outputs)}.json"


@dataclass(frozen=True)
class Served:
    """A table the serve committed, as rank 0 kept it (`load`): its entry (``value``, ``root``, ``leaves``, ``words``,
    ``domain``), the ``program`` its domain binds, its C ``rows`` (``(C, words)`` u16, its request's then the fill's) and
    their ``salts``, one 192-byte salt a row concatenated. Committed again under ``program``, ``value``, ``words`` and
    ``leaves`` with these salts, it has round 1's root. Private: rows and salts never leave the node."""

    value: str
    program: str
    root: str
    leaves: int
    words: int
    domain: str
    rows: np.ndarray = field(repr=False)
    salts: bytes = field(repr=False)

    def entry(self) -> dict[str, Any]:
        """Its registered entry, as round 1 listed it (`history.HistoryTable.entry`)."""
        return {"value": self.value, "root": self.root, "leaves": self.leaves, "words": self.words, "domain": self.domain}

    def strings(self) -> list[bytes]:
        """Its C rows' public strings (`history.string`), position by position: what the interior publishes of it."""
        y = len(self.salts) // max(1, len(self.rows))
        return [HT.string(r, self.words, self.salts[i * y:(i + 1) * y]) for i, r in enumerate(self.rows)]


class Kept:
    """Each table's rows and salts as built, under ``directory`` on the node: a class's in w<window>-<run>-p<P>-o<O>/, a
    span's in s<slot>/, and each of its tables in layer-<l>.rows (its rows, u16 little-endian), layer-<l>.salts (their
    salts) and layer-<l>.json (its entry, its program and how many rows it holds), written last. Every directory and
    file is readable by its owner only."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)

    @classmethod
    def of(cls, cfg: Mapping[str, Any], *public: Path | None) -> Kept:
        """From a serve's ``tables`` configuration: under its ``private`` directory (`private_root` of ``public``)."""
        return cls(private(private_root(cfg["private"], *public)) / "tables")

    def _span(self, window: int, run: str, prompt: int, outputs: int, slot: int) -> Path:
        key(run, prompt, outputs, slot)
        return self.dir / _stem(window, run, prompt, outputs).removesuffix(".json") / f"s{slot}"

    def write(self, window: int, run: str, prompt: int, outputs: int, slot: int, layer: int, program: str,
              table: Any) -> None:
        """Table ``table`` (a `history.HistoryTable` or `history_tables_device.Table`) of slot ``slot`` at ``layer``."""
        d = self._span(window, run, prompt, outputs, slot)
        for q in (self.dir, d.parent, d):
            private(q)
        rows = np.asarray(table.rows, dtype="<u2").reshape(len(table.rows), table.words)
        salts = [bytes(y) for y in table.salts]
        if len(salts) != len(rows) or len({len(y) for y in salts}) > 1:
            raise ValueError(f"{table.value}: one salt of one length a row, not {len(salts)} for {len(rows)} rows")
        for suffix, data in (("rows", rows.tobytes()), ("salts", b"".join(salts))):
            _bytes(d / f"layer-{int(layer)}.{suffix}", data)
        head = {**table.entry(), "program": program, "positions": len(rows), "salt_bytes": len(salts[0]) if salts else 0}
        _write(d / f"layer-{int(layer)}.json", head)

    def head(self, window: int, run: str, prompt: int, outputs: int, slot: int, layer: int) -> dict[str, Any]:
        return json.loads((self._span(window, run, prompt, outputs, slot) / f"layer-{int(layer)}.json").read_text())

    def rows(self, window: int, run: str, prompt: int, outputs: int, slot: int,
             layer: int) -> tuple[dict[str, Any], np.ndarray, list[bytes]]:
        """The table's head, its rows ``(n, words)`` and their salts, refused where the files disagree on how many."""
        h, d = self.head(window, run, prompt, outputs, slot, layer), self._span(window, run, prompt, outputs, slot)
        n, w, y = h["positions"], h["words"], h["salt_bytes"]
        raw, ys = (d / f"layer-{int(layer)}.rows").read_bytes(), (d / f"layer-{int(layer)}.salts").read_bytes()
        if len(raw) != 2 * n * w or len(ys) != n * y:
            raise ValueError(f"{h['value']}: its files hold {len(raw)} row bytes and {len(ys)} salt bytes, not {n} rows' "
                             f"{2 * n * w} and {n * y}")
        return h, np.frombuffer(raw, dtype="<u2").reshape(n, w), [ys[i * y:(i + 1) * y] for i in range(n)]

    def span(self, window: int, run: str, prompt: int, outputs: int, slot: int, layer: int,
             positions: range) -> dict[str, Any]:
        """What round 2 opens of slot ``slot``'s table at ``layer``: its rows at ``positions`` (a range within what it
        holds), their salts and their public strings (`history.string`), beside the table's entry."""
        h, rows, salts = self.rows(window, run, prompt, outputs, slot, layer)
        if not (isinstance(positions, range) and positions.step == 1 and 0 <= positions.start <= positions.stop <= len(rows)):
            raise ValueError(f"{h['value']}: positions {positions!r} aren't a range within the {len(rows)} it holds")
        lo, hi = positions.start, positions.stop
        return {"entry": {k: h[k] for k in ("value", "root", "leaves", "words", "domain")}, "positions": positions,
                "rows": rows[lo:hi], "salts": salts[lo:hi],
                "strings": [HT.string(r, h["words"], y) for r, y in zip(rows[lo:hi], salts[lo:hi], strict=True)]}

    def table(self, window: int, run: str, prompt: int, outputs: int, slot: int, layer: int) -> HT.HistoryTable:
        """The table rebuilt from its files (its paths for round 2's openings), refused unless it has the root it was
        kept with."""
        h, rows, salts = self.rows(window, run, prompt, outputs, slot, layer)
        t = HT.HistoryTable(h["program"], h["value"], h["words"], h["leaves"])
        t.append(rows, salts)
        if t.entry() != {k: h[k] for k in ("value", "root", "leaves", "words", "domain")}:
            raise ValueError(f"{h['value']}: its files rebuild to another table than the one kept")
        return t

    def load(self, table: str) -> Served:
        """The served table named ``table`` (`name`), refused unless exactly one table kept here has that name."""
        m = _NAME.fullmatch(table) if isinstance(table, str) else None
        parts = (m["run"], int(m["prompt"]), int(m["outputs"]), int(m["slot"]), int(m["layer"])) if m else None
        if parts is None or name(*parts) != table:
            raise ValueError(f"{table!r} is no served table's name, {H.NAME.format(key=KEY, layer='{layer}')}")
        run, prompt, outputs, slot, layer = parts
        found = []
        for d in sorted(self.dir.glob(f"w*-{run[:16]}-p{prompt}-o{outputs}")):
            w = _CLASS_DIR.match(d.name)
            if w and (d / f"s{slot}" / f"layer-{layer}.json").is_file() \
                    and self.head(int(w.group(1)), run, prompt, outputs, slot, layer).get("value") == table:
                found.append(int(w.group(1)))
        if len(found) != 1:
            raise ValueError(f"rank 0 kept {len(found)} tables named {table} under {self.dir}, not one")
        h, rows, salts = self.rows(found[0], run, prompt, outputs, slot, layer)
        return Served(h["value"], h["program"], h["root"], h["leaves"], h["words"], h["domain"], rows, b"".join(salts))

    def drop(self, window: int, run: str, prompt: int, outputs: int) -> None:
        """A class's files, for a class that registered nothing."""
        shutil.rmtree(self.dir / _stem(window, run, prompt, outputs).removesuffix(".json"), ignore_errors=True)


def load(directory: str | Path, table: str) -> Served:
    """The served table named ``table`` (`name`: its run, class, slot and layer) that rank 0 kept under a serve's private
    directory ``directory`` (its ``tables`` configuration's ``private``, as `Kept.of` puts it): what the interior opens
    for a drawn span, committed again under round 1's program, name, words and capacity so that its root is round 1's
    port root. On the node only."""
    return Kept(Path(directory).resolve() / "tables").load(table)


def private(directory: Path) -> Path:
    """``directory``, made if need be, readable by its owner only."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def private_root(directory: Path, *public: Path | None) -> Path:
    """``directory`` for files that stay on the node, refused inside any of ``public`` (a serve's records, its run's
    directory: what custody uploads)."""
    d = Path(directory).resolve()
    for p in public:
        if p is not None and d.is_relative_to(Path(p).resolve()):
            raise ValueError(f"the history tables' private directory {d} is inside {p}, which leaves the node: put it "
                             f"outside every run directory")
    return d


def layers_of(spec: int | Sequence[int]) -> tuple[int, ...]:
    """The tables' layers from a serve's configuration: a count (the first that many) or the layers themselves."""
    if type(spec) is int:
        return tuple(range(spec))
    return tuple(int(x) for x in spec)


def _bytes(path: Path, data: bytes) -> None:
    """``data`` at ``path``, readable by its owner only, whole or not at all."""
    tmp = path.with_name(f".{path.name}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    private(path.parent.parent)
    private(path.parent)
    _bytes(path, (json.dumps(doc) + "\n").encode())


class Channel:
    """The window's boundary commitment and rank 0, in two processes on the node: `assign` writes a class's assignment
    under ``directory``/assign/, rank 0's `serve` answers each under ``directory``/roots/, and `roots` waits for the
    answer. ``directory`` is private: an assignment names the engine's ids."""

    def __init__(self, directory: Path, *, timeout_s: float = 900.0, poll_s: float = 0.05,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.dir, self.timeout_s, self.poll_s, self.sleep, self.clock = Path(directory), timeout_s, poll_s, sleep, clock

    @classmethod
    def of(cls, cfg: Mapping[str, Any], *public: Path | None) -> Channel:
        """From a serve's ``tables`` configuration: ``private``, its directory (`private_root` of ``public``), and
        ``timeout_s``."""
        return cls(private(private_root(cfg["private"], *public)) / "channel", timeout_s=float(cfg.get("timeout_s", 900.0)))

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
