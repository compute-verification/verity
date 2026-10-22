"""H1's served tables (VLLM-64), committed as registered values of compute accounting's window Program
(`h1window.program`, one tile template per K) as the window serves them, so the Lean verifier opens every row a unit reads
(`flock-verify --zk --registered`, `h1window.table_reads_file`).

* **The Program** is declared before the window serves (`declared`): every step calls every linear ``(module, K, n)`` in
  order, call (s, w) reading A block ``s·B_K + i`` of its K (`layout`: ``i`` the place of ``blocks[w]``, each linear's
  input within a step, among the blocks its K's linears read, ``B_K`` their number), its A slot M `h1window.slot` of the
  window's T rows, over the served path's R × C tile (`pouw_h1.TILE`, 32 × 32) and the tile unit at each K
  (`h1window.tile_unit`, or ``unit``). Its parameters are named ``h1/<window>/`` and their role (`prefix`): ``<window>``
  the window's key (`Served.key`: the window-open root, `salt.window_open_root` of the weights' root, round 0's request
  table and the session, or the key the run is given, 64 bytes).
* **The tables** are the ones a unit's rows open at (`h1window.table`, `h1window.table_reads`): call c's A table
  ``a/<K>@<c>``, its M slot rows, each its K int8 codes then its f32 s_a (`h1zy.ROW`; the served rows, then zero codes of
  scale 0), and its z table ``zy@<c>``, one record a unit of the call in its tile order (`h1zy.RECORD`: its cells' s32 z,
  row-major, then their bf16 y; a slot past the served rows is z = 0 and y = +0, dq(0, +0, s_w) for a finite positive
  s_w); and the weights' ``b/<K>``, K's linears' rows linear after linear, each its K weight codes then its f32 s_w, once a
  window (Σn rows every step's units read).
* **Rows** are ``hm96-sha512/row/v2`` bit rows of 16-bit words (`V2`, `row_words`, `record_words`), committed by
  `registered.Registered.commit` under the registered-value domain of the Program's digest, the value, its words and rows,
  and the window's open root (`registered.domain`'s ``window``, `Served.root`, #3044), row j's salt `audit.hm96_salt` of
  the run's key at the value's nonce (`nonce`).
* **When.** `Served.commit` registers call c's two tables as `H1Run` records the call, and ``b/<K>`` at the call that
  completes K's weights (the last of K's linears in step 0). Each root goes into the call's manifest row
  (``tables.registered``), so its round's root commits it before the round's coin. A weight is taken once, and one that
  changes within the window is refused.
* **Refused** (no table from then on, and `trees` raises the reason): a window with no key of 64 bytes, blocks that aren't
  one per linear, a call off the declared Program's order, a served call with no z/y record, a weight that isn't the
  linear's n × K int8 codes and n scales or that changes, an odd K, what `h1window.program` refuses, and at `close` a
  window that isn't static or a table the Program reads that was never registered.
* **Statements.** A Lean ``--registered`` port holds one root, so a statement holds one call's units, and its file is
  `reads_file` of that call: `h1window.table_reads_file`, the verifier's own reads from its Program, with every port's
  domain bound to the window's open root as its tree was. `register` holds the window's trees to an earlier record of the
  log under the same domain (R7-held) and writes each call's file (`file_name`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G, registration as R
from verity.core.protocols.sampled_proofs.one_stage.served import ServedTree
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from verity_vllm.protocol_options import pouw_h1 as PH
from verity_vllm.protocol_options import pouw_h1_requests as RQ

#: the rows' format: the tile's port rows
V2 = G.SCHEMA_V2
#: a statement's ``--registered`` file, one a call
FILE = "pouw-h1-registered-c{call}.json"
PREFIX = "h1/{window}/"
#: the hm96 salt stream's first nonce of the window's registered tables, past every round's and call's tables
NONCE = RQ.REQUESTS_NONCE + (1 << 42)
#: ``b/<K>``'s nonces, past every call's two tables
B_NONCE = NONCE + (1 << 40)
DIGEST_TAG = b"verity/pouw/h1/served/v3\0"


def prefix(window: bytes) -> str:
    """The window Program's parameter prefix for the window keyed `window`: its first 16 bytes, hex."""
    return PREFIX.format(window=window[:16].hex())


def nonce(call: int, table: str) -> int:
    """The salt nonce of call ``call``'s table ``table`` (`h1window.TABLES`)."""
    return NONCE + 2 * int(call) + HW.TABLES.index(table)


def file_name(call: int) -> str:
    return FILE.format(call=int(call))


def row_words(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """int8 code rows ``(n, K)`` and their f32 scales ``(n,)`` as their bit rows' u16 words ``(n, K / 2 + 2)``: code t's
    bits are leaves 8t .. 8t + 7, then the scale's 32 bits, little-endian (`h1zy.ROW`)."""
    b = np.ascontiguousarray(np.asarray(codes, dtype=np.int8).view(np.uint8))
    b = b.reshape(len(b), -1)
    s = np.ascontiguousarray(np.asarray(scales, "<f4").reshape(len(b))).view(np.uint8).reshape(len(b), 4)
    return np.ascontiguousarray(np.concatenate([b, s], axis=1)).view("<u2").astype(np.uint16)


def record_words(z: np.ndarray, y: np.ndarray, slots: int, tile: tuple[int, int]) -> np.ndarray:
    """A call's z/y records over ``slots`` rows as their bit rows' u16 words ``(units, 3 R C)``, one a unit in tile order
    (row block major): its cells' s32 z, row-major, little-endian, then their bf16 y (`h1zy.RECORD`). ``z`` [m, n] s32 and
    ``y`` [m, n] bf16 bits are the served rows'; every later slot is z = 0, y = +0."""
    (r, c), (m, n) = tile, np.shape(z)
    zf, yf = np.zeros((slots, n), "<i4"), np.zeros((slots, n), "<u2")
    zf[:m], yf[:m] = z, np.asarray(y).astype(np.int16).view(np.uint16)

    def cells(v: np.ndarray) -> np.ndarray:
        return v.reshape(slots // r, r, n // c, c).transpose(0, 2, 1, 3).reshape(-1, r * c)
    return np.concatenate([np.ascontiguousarray(cells(zf)).view("<u2"), cells(yf)], axis=1).astype(np.uint16)


def layout(shapes: Sequence[tuple[int, int]], calls: Sequence[tuple[int, int]],
           blocks: Sequence[int] | None = None) -> list[int]:
    """Each call's A block of its K, ``calls`` each call's (step, linear) and ``blocks`` each linear's input block within a
    step (`Served`'s): call (s, w) reads block s·B_K + i, i the place of ``blocks[w]`` among the blocks K's linears read."""
    per = list(blocks) if blocks is not None else list(range(len(shapes)))
    if len(per) != len(shapes):
        raise ValueError(f"blocks {per} are not one input block for each of the window's {len(shapes)} linears")
    of_k: dict[int, list[int]] = {}
    for w, (_n, k) in enumerate(shapes):
        of_k.setdefault(k, [])
        if per[w] not in of_k[k]:
            of_k[k].append(per[w])
    of_k = {k: sorted(b) for k, b in of_k.items()}
    return [s * len(of_k[shapes[w][1]]) + of_k[shapes[w][1]].index(per[w]) for s, w in calls]


def declared(shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int,
             tile: tuple[int, int] = PH.TILE, blocks: Sequence[int] | None = None, prefix: str = "",
             unit: Callable[[int], Any] | None = None) -> Program:
    """The window Program a static window declares before it serves: ``steps`` steps, each calling every linear (``shapes``,
    each ``(n, K)``, named by ``modules``) in order, laid out by `layout`, its A slot for ``rows`` token rows. With no
    ``prefix`` it names no window, so it is one for every window of the declaration (a period's ``--circuit``, its
    `program_digest`); with a window's `prefix` it is that window's `Served.program`."""
    mods = [modules.get(w, str(w)) for w in range(len(shapes))]
    calls = [(s, w) for s in range(steps) for w in range(len(shapes))]
    return HW.program([(mods[w], k, n) for w, (n, k) in enumerate(shapes)],
                      [(mods[w], b) for (_s, w), b in zip(calls, layout(shapes, calls, blocks))], HW.slot(rows, tile), tile,
                      prefix, unit)


@dataclass
class Served:
    """How an `H1Run` commits its calls' tables and its weights as the window Program's registered values. ``tile``: the
    Program's R × C (`pouw_h1.TILE`). ``unit(k)``: the tile unit at contraction k (default `h1window.tile_unit`; a stand-in
    of its parameter shape in tests). ``blocks``: each linear's input block within a step (default each its own;
    ``(0, 0, 0, 1)`` for q, k, v and o). ``window``: the window's key, else the window-open root its commitment names.
    From the first call: ``program`` (`declared` under the window's `prefix`; None when refused), ``root`` (the window's
    open root its values' domains bind, hex), ``program_digest``, ``linears``, ``calls`` and ``m`` (the Program's); ``registered`` (each
    value's `registered.Registered`, in registration order) and ``refused`` (why the window has no trees)."""

    tile: tuple[int, int] = PH.TILE
    unit: Callable[[int], Any] | None = None
    blocks: Sequence[int] | None = None
    window: bytes | None = None
    program: Program | None = None
    program_digest: str | None = None
    root: str | None = None
    registered: dict[str, G.Registered] = field(default_factory=dict)
    linears: list[tuple[str, int, int]] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)
    m: int = 0
    refused: list[str] = field(default_factory=list)
    #: each linear's weight, the tensor and its codes, taken once, and its scales
    _weights: dict[int, tuple[Any, np.ndarray]] = field(default_factory=dict)
    _scales: dict[int, np.ndarray] = field(default_factory=dict)

    def key(self, commits: Mapping[str, Any]) -> bytes:
        if self.window is not None:
            return self.window
        session = commits.get("session") or {}
        need = (commits.get("weights_root"), commits.get("requests_root"), session.get("session"))
        if None in need:
            raise ValueError("the window has no key: neither one given nor a window-open root (its weights' root, round 0's "
                             "request table and the session)")
        return PS.window_open_root(bytes.fromhex(need[0]), bytes.fromhex(need[1]), need[2])

    def open(self, run: Any) -> None:
        """Declare the window's Program under its key, at its first call (after its salt opened)."""
        try:
            key = self.key(run.commits)
            self.root = G.window_root(key.hex())
            if any(k % 2 for _n, k in run.shapes_):
                raise ValueError(f"K {sorted({k for _n, k in run.shapes_})}: a row's codes are whole u16 words")
            self.program = declared(run.shapes_, run.modules, run.rows, run.steps, self.tile, self.blocks, prefix(key),
                                    self.unit)
        except ValueError as e:
            self.refused.append(str(e))
            return
        S, self.program_digest = self.program.fn.S(), P.program_digest(self.program)
        self.linears, self.calls, self.m = [tuple(x) for x in S.LINEARS], [tuple(x) for x in S.CALLS], int(S.M)

    def _register(self, run: Any, value: str, words: np.ndarray, first: int) -> str:
        reg = G.Registered.commit(self.program_digest, value, words.tolist(),
                                  [hm96_salt(run._key, first, j) for j in range(len(words))], schema=V2, window=self.root)
        self.registered[value] = reg
        return reg.root.hex()

    def _weight(self, call: Any, weight: Any, scale: Any) -> None:
        held = self._weights.get(call.layer)
        if weight is None or (held is not None and held[0] is weight):
            return
        w, s = PH.numpy_of(weight), None if scale is None else PH.numpy_of(scale).astype("<f4").reshape(-1)
        if w.dtype != np.int8 or w.shape != (call.n, call.k) or s is None or s.shape != (call.n,):
            self.refused.append(f"call {call.index}: its weight is not its {call.n} × {call.k} int8 codes and {call.n} scales")
        elif held is None:
            self._weights[call.layer], self._scales[call.layer] = (weight, w), s
        elif not (np.array_equal(held[1], w) and np.array_equal(self._scales[call.layer], s)):
            self.refused.append(f"call {call.index}: linear {call.layer}'s weight changed within the window")

    def commit(self, run: Any, call: Any, a: Any, s_a: Any, weight: Any, weight_scale: Any, y: Any, z: Any) -> dict[str, str]:
        """Register call ``call``'s A and z tables (its served rows ``a`` [m, K] and scales ``s_a`` [m], its exact z [m, n]
        and served y [m, n]), and ``b/<K>`` once K's linears' weights are all held: ``{value: root}``, what the call's
        manifest row commits under its round's root. Nothing once the window is refused."""
        if self.program is None and not self.refused:
            self.open(run)
        if not self.refused:
            self._weight(call, weight, weight_scale)
        if not self.refused and (call.index >= len(self.calls) or call.index != call.step * len(self.linears) + call.layer):
            self.refused.append(f"call {call.index} (step {call.step}, linear {call.layer}) is off the declared Program's "
                                "order")
        if not self.refused and call.m and (z is None or y is None):
            self.refused.append(f"call {call.index}: no z/y record (its exact z and served y)")
        if self.refused:
            return {}
        codes, scales = np.zeros((self.m, call.k), np.int8), np.zeros(self.m, "<f4")
        zv, yv = np.zeros((0, call.n), "<i4"), np.zeros((0, call.n), np.int16)
        if call.m:
            codes[:call.m] = PH.numpy_of(a).reshape(call.m, call.k)
            scales[:call.m] = PH.numpy_of(s_a).reshape(-1)
            zv, yv = PH._words(z, call.m, "<i4"), PH._words(y, call.m, "<i2")
        out = {}
        for name, words in (("a", row_words(codes, scales)), ("zy", record_words(zv, yv, self.m, self.tile))):
            value = HW.table(self.program, name, call.index)
            out[value] = self._register(run, value, words, nonce(call.index, name))
        b, ks = self.program.fn.bindings["PREFIX"] + f"b/{call.k}", sorted({k for _n, k in run.shapes_})
        mine = [w for w, (_n, k) in enumerate(run.shapes_) if k == call.k]
        if b not in self.registered and all(w in self._weights for w in mine):
            words = np.concatenate([row_words(self._weights[w][1], self._scales[w]) for w in mine])
            out[b] = self._register(run, b, words, B_NONCE + ks.index(call.k))
        return out

    def close(self, run: Any) -> None:
        """After the run's last round: refuse a window that isn't static, or that left a table its Program reads
        unregistered."""
        if self.refused:
            return
        if self.program is None:
            self.refused.append("no call was committed")
        elif not run.static():
            self.refused.append("the window isn't static: a call off its public position or refused, so the Program's calls "
                                "aren't the window's")
        elif missing := [v for v in self.tables() if v not in self.registered]:
            self.refused.append(f"{len(missing)} tables the Program reads were never registered: {missing[:4]}")

    def tables(self) -> list[str]:
        """Every value the Program's units open at: each K's ``b/<K>``, then each call's A and z tables."""
        pre, ks = self.program.fn.bindings["PREFIX"], sorted({k for _name, k, _n in self.linears})
        return [pre + f"b/{k}" for k in ks] + [HW.table(self.program, t, c) for c in range(len(self.calls))
                                                for t in HW.TABLES]

    def values(self) -> dict[str, int]:
        """Each registered value's words a row."""
        return {v: r.words for v, r in self.registered.items()}

    def roots(self) -> dict[str, str]:
        return {v: r.root.hex() for v, r in self.registered.items()}

    def ports(self) -> dict[str, str]:
        """The window's registered reads' circuit ports (`h1window.window_ports` of its Program), a lone row (R = 1 or
        C = 1) under both its keys: `registered.reads`' ``a_k<K>`` and `h1window.table_reads`' ``a_k<K>[0]``."""
        out = HW.window_ports(self.program)
        return out | {f"{key}[0]": port for key, port in out.items() if "[" not in key}

    def entries(self) -> list[dict[str, Any]]:
        return [r.entry() for r in self.registered.values()]

    def digest(self) -> bytes:
        """SHA-512 over the trees' entries and what was refused, canonical JSON: what `calls_root` binds."""
        return digest_of(self.entries(), self.refused)

    def query(self) -> dict[str, Any]:
        """The window Program's partition query: every tile of each K's template one unit (`h1window.query`)."""
        return HW.query(self.program)


def digest_of(entries: Sequence[Mapping[str, Any]], refused: Sequence[str]) -> bytes:
    """`Served.digest` of the trees' `entries` and `refused`, as a record carries them (the commitment's ``served``)."""
    return hashlib.sha512(DIGEST_TAG + canonical_json({"entries": list(entries), "refused": list(refused)})).digest()


def _served(run: Any) -> Served:
    served = getattr(run, "served", None)
    if not isinstance(served, Served):
        raise ValueError(f"{run.entry}: the window's rows aren't registered values (the run has no `Served`)")
    if not run.closed:
        raise ValueError(f"{run.entry}: the window isn't closed, so its tables aren't all committed")
    if served.refused or not served.registered:
        raise ValueError(f"{run.entry}: the window's trees were refused: {'; '.join(served.refused) or 'none committed'}")
    return served


def trees(run: Any) -> list[ServedTree]:
    """The window's registered trees, in registration order: refused unless `run` committed them as `Served` values."""
    return [ServedTree.of(r) for r in _served(run).registered.values()]


def reads_file(run: Any, call: int, ports: Mapping[str, str] | None = None, tp: Any = None) -> dict[str, Any]:
    """The verifier's ``--registered`` file of a statement over call ``call``'s units: `h1window.table_reads_file` over the
    window's Program (``tp`` its template partition, `partition.evaluate`'s, when the caller holds it) and its trees'
    roots, under the circuit ports ``ports`` (default `Served.ports`), each port's domain bound to the window's open root
    as its tree's is."""
    served = _served(run)
    prog, digest = served.program, served.program_digest
    tp = tp if tp is not None else P.evaluate(prog, served.query())
    doc = HW.table_reads_file(prog, tp, int(call), served.roots(), served.ports() if ports is None else dict(ports),
                              prog=digest)
    for p in doc["ports"].values():
        p["domain"] = G.domain(digest, p["value"], p["words"], p["leaves"], window=served.root).domain_id.hex()
    return doc


@dataclass(frozen=True)
class Registration:
    """What `register` returns: the record's ``registered`` entries (every tree of the window), the window's open root, and
    each call's ``--registered`` file."""

    registered: list[dict[str, Any]]
    window_root: str
    files: dict[int, dict[str, Any]]


def register(run: Any, calls: Sequence[int] | None = None, log: Sequence[Mapping[str, Any]] = (),
             directory: str | Path | None = None, ports: Mapping[str, str] | None = None) -> Registration:
    """The window's trees as one record's ``registered`` entries, held to this prover's earlier records ``log`` (R7-held:
    a value an earlier record registered under the same domain with another root is refused), and the ``--registered``
    file of each of ``calls`` (default every call; `reads_file`), written as `file_name` in ``directory`` (default the
    run's ``transcript_dir``)."""
    got, served = trees(run), run.served
    held = R.held_roots(log)
    for t in got:
        if held.get((t.value, t.domain), t.root) != t.root:
            raise ValueError(f"served tree {t.value!r}: an earlier record of the log registered it under this domain with "
                             f"root {held[(t.value, t.domain)][:16]}..., and a value is committed once a domain (R7-held)")
    tp = P.evaluate(served.program, served.query())
    files = {int(c): reads_file(run, c, ports, tp) for c in (range(len(served.calls)) if calls is None else calls)}
    d = directory if directory is not None else run.transcript_dir
    if d is not None:
        Path(d).mkdir(parents=True, exist_ok=True)
        for c, doc in files.items():
            (Path(d) / file_name(c)).write_bytes(canonical_json(doc))
    return Registration([t.entry() for t in got], served.root, files)


__all__ = ["B_NONCE", "FILE", "NONCE", "PREFIX", "Registration", "Served", "V2", "declared", "digest_of", "file_name",
           "layout", "nonce", "prefix", "reads_file", "record_words", "register", "row_words", "trees"]
