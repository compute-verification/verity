"""H1's served trees (VLLM-64): a window's A codes and B rows, committed as the registered parameters ``a/<K>`` and
``b/<K>`` of compute accounting's window Program (`h1window.program`, one tile template per K), so a served path registers
them as ports of the verifier's own ``--registered`` file (`one_stage.served.register_served`) and verifies with
``flock-verify --zk --registered``.

* **Values.** The Program's parameters are ``h1/<window>/`` and their role (`prefix`, `h1window.roles`): ``<window>`` the
  window's key (`Served.key`: the window-open root, `salt.window_open_root` of the weights' root, round 0's request table
  and the session, or the key the run is given, 64 bytes), so each window's values are its own, and every value's domain
  binds that key as the window's open root (`registered.domain`'s ``window``, `Served.root`): a value the run commits
  afresh each window, as it does its A and its B, is held by none (`register_served`'s ``held``).
* **The tile** is the served path's R × C, `pouw_h1.TILE` (32 × 32) unless given, over the tile unit at each K
  (`h1window.tile_unit`, or ``unit``), each K's template reading its A and B rows as ``a_k<K>`` and ``b_k<K>``
  (`h1window.template_params`). A unit of K's R A rows are ports ``a_k<K>_0`` .. ``a_k<K>_<R-1>`` of one value and its C
  B rows ``b_k<K>_0`` .. ``b_k<K>_<C-1>`` (`h1window.window_ports`, a lone row row 0's), so every K's ports are its own.
* **Linears, calls and blocks.** Linear w is the Program's linear ``(module, K, n)``, its weight once. Call (s, w) reads A
  block ``s·B_K + i`` of its K, ``i`` the place of ``blocks[w]`` (each linear's input within a step; q, k and v read one)
  among the blocks its K's linears read, ``B_K`` their number. A block's rows are its calls' T slots (`pouw_h1.slots`, the
  served rows first, empty slots zero codes), padded with zero rows to the Program's slot M (`h1window.slot`); every call
  sharing a block must hand over the same codes. ``b/<K>`` is the weight rows of K's linears, linear after linear, each
  once per window (Σn rows): every step's units read them.
* **Rows** are ``hm96-sha512/row/v2`` bit rows (`V2`): an A row is its K int8 codes only, a B row its K weight codes, each
  as K/2 u16 words, code t's bits leaves 8t .. 8t + 7 (`row_words`, the tile's port rows), committed by
  `registered.Registered.commit` in `V2` under the registered-value domain of the Program's digest, the value, K/2 words,
  its rows and the window's open root, row i's salt `audit.hm96_salt` of the run's key at the value's nonce (`nonce`). The A scale is no value
  of the window Program: no unit reads it, and it stays bound in each call's A table under its round root, with y and z
  (`pouw_h1.H1Run`).
* **When.** `Served.record` keeps each call's codes as the run records it and `Served.close` commits the trees at the
  run's `close`, after its last round; `calls_root` binds their entries (`digest`), so the draw's coin opens after them.
  What `close` refuses (a window that isn't static, no key of 64 bytes, blocks that aren't one per linear, a call whose codes aren't
  its shared block's, an odd K, a linear with no weight or whose weight isn't its n × K int8 codes or changes within the
  window, what `h1window.program` refuses) leaves no tree, and `trees` raises it.
* `trees` gives the window's `ServedTree`s, each K's A then B, and `register` registers them all at once: a window is one
  statement, one C-Flock session and one registration, so one ``--registered`` file (`FILE`) in the window's directory,
  whatever its K, under the window's root (`register_served`'s ``window_root``), which refuses a value an earlier record
  of the log registered under the same domain with another root (R7-held).
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
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G
from verity.core.protocols.sampled_proofs.one_stage.served import ServedRegistration, ServedTree, register_served
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from verity_vllm.protocol_options import pouw_h1 as PH
from verity_vllm.protocol_options import pouw_h1_requests as RQ

#: the rows' format: the tile's port rows
V2 = G.SCHEMA_V2
FILE = "pouw-h1-registered.json"
PREFIX = "h1/{window}/"
#: the hm96 salt stream's first nonce of the window's trees, past every round's and call's tables (`pouw_h1_requests`)
NONCE = RQ.REQUESTS_NONCE + (1 << 42)
DIGEST_TAG = b"verity/pouw/h1/served/v2\0"


def prefix(window: bytes) -> str:
    """The window Program's parameter prefix for the window keyed `window`: its first 16 bytes, hex."""
    return PREFIX.format(window=window[:16].hex())


def nonce(i: int) -> int:
    """The salt nonce of the window's i-th tree, in `roles` order (each K's A, then its B)."""
    return NONCE + int(i)


def row_words(codes: np.ndarray) -> np.ndarray:
    """int8 code rows ``(n, K)`` as their bit rows' u16 words ``(n, K / 2)``: code t's bits are leaves 8t .. 8t + 7."""
    b = np.ascontiguousarray(np.asarray(codes, dtype=np.int8).view(np.uint8))
    return b.reshape(len(b), -1).view("<u2").astype(np.uint16)


@dataclass
class Served:
    """How an `H1Run` commits its window as the window Program's registered values. ``tile``: the Program's R × C
    (`pouw_h1.TILE`). ``unit(k)``: the tile unit at contraction k (default `h1window.tile_unit`, the H1 tile; a stand-in
    of its parameter shape in tests). ``blocks``: each linear's input block within a step (default each its own;
    ``(0, 0, 0, 1)`` for q, k, v and o). ``window``: the window's key, else the window-open root its commitment names.
    After `close`: ``program`` (None when refused), ``root`` (the window's open root its values' domains bind, hex),
    ``registered`` (each K's A then B, `registered.Registered`), ``linears`` and ``calls`` (the Program's), ``m`` (its A
    slot) and ``refused`` (why there are no trees)."""

    tile: tuple[int, int] = PH.TILE
    unit: Callable[[int], Any] | None = None
    blocks: Sequence[int] | None = None
    window: bytes | None = None
    program: Program | None = None
    root: str | None = None
    registered: list[G.Registered] = field(default_factory=list)
    linears: list[tuple[str, int, int]] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)
    m: int = 0
    refused: list[str] = field(default_factory=list)
    #: each recorded call and its T slots' codes
    _got: list[tuple[Any, np.ndarray]] = field(default_factory=list)
    #: each linear's weight, the tensor and its codes, taken once
    _weights: dict[int, tuple[Any, np.ndarray]] = field(default_factory=dict)

    def key(self, commits: Mapping[str, Any]) -> bytes:
        if self.window is not None:
            return self.window
        session = commits.get("session") or {}
        need = (commits.get("weights_root"), commits.get("requests_root"), session.get("session"))
        if None in need:
            raise ValueError("the window has no key: neither one given nor a window-open root (its weights' root, round 0's "
                             "request table and the session)")
        return PS.window_open_root(bytes.fromhex(need[0]), bytes.fromhex(need[1]), need[2])

    def record(self, call: Any, a: Any, weight: Any) -> None:
        """Call `call`'s A codes (``a`` [m, k], its served rows) and its linear's weight (``weight`` [n, k]), as the run
        records it: a linear's weight is taken once, and a later call handing over another is refused."""
        codes = np.zeros((call.rows, call.k), np.int8)
        if a is not None and call.m:
            codes[:call.m] = PH.numpy_of(a).reshape(call.m, call.k)
        self._got.append((call, codes))
        held = self._weights.get(call.layer)
        if weight is None or (held is not None and held[0] is weight):
            return
        w = PH.numpy_of(weight)
        if w.dtype != np.int8 or w.shape != (call.n, call.k):
            self.refused.append(f"call {call.index}: its weight is not its {call.n} × {call.k} int8 codes")
        elif held is None:
            self._weights[call.layer] = (weight, w)
        elif not np.array_equal(held[1], w):
            self.refused.append(f"call {call.index}: linear {call.layer}'s weight changed within the window")

    def _layout(self, shapes: Sequence[tuple[int, int]]) -> list[int]:
        """Each recorded call's A block of its K (`Served`'s blocks)."""
        per = list(self.blocks) if self.blocks is not None else list(range(len(shapes)))
        if len(per) != len(shapes):
            raise ValueError(f"blocks {per} are not one input block for each of the window's {len(shapes)} linears")
        of_k: dict[int, list[int]] = {}
        for w, (_n, k) in enumerate(shapes):
            of_k.setdefault(k, [])
            if per[w] not in of_k[k]:
                of_k[k].append(per[w])
        of_k = {k: sorted(b) for k, b in of_k.items()}
        return [c.step * len(of_k[c.k]) + of_k[c.k].index(per[c.layer]) for c, _codes in self._got]

    def close(self, run: Any) -> None:
        """Commit the window's A and B trees under the window Program, after the run's last round."""
        try:
            key, blocks = self.key(run.commits), self._layout(run.shapes_)
            root = G.window_root(key.hex())
        except ValueError as e:
            self.refused.append(str(e))
        if not run.static():
            self.refused.append("the window isn't static: a call off its public position or refused, so the Program's calls "
                                "aren't the window's")
        ks = sorted({k for _n, k in run.shapes_})
        if any(k % 2 for k in ks):
            self.refused.append(f"K {ks}: a row's codes are whole u16 words")
        missing = [w for w in range(len(run.shapes_)) if w not in self._weights]
        if missing:
            self.refused.append(f"linears {missing} have no weight")
        a: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
        for i, (c, codes) in enumerate(self._got) if not self.refused else ():
            first = a.setdefault((c.k, blocks[i]), (c.index, codes))
            if not np.array_equal(first[1], codes):
                self.refused.append(f"call {c.index}'s A codes aren't those of its block {blocks[i]} of K = {c.k} (call "
                                    f"{first[0]}'s): the calls that share an input share its block")
        if self.refused:
            return
        self.m = HW.slot(run.rows, self.tile)
        mods = [run.modules.get(w, str(w)) for w in range(len(run.shapes_))]
        self.linears = [(mods[w], k, n) for w, (n, k) in enumerate(run.shapes_)]
        self.calls = [(mods[c.layer], blk) for (c, _codes), blk in zip(self._got, blocks)]
        try:
            self.program = HW.program(self.linears, self.calls, self.m, self.tile, prefix(key), self.unit)
        except ValueError as e:
            self.refused.append(str(e))
            return
        digest, pre = P.program_digest(self.program), prefix(key)
        rows: list[np.ndarray] = []
        for k in ks:
            nb = 1 + max(blk for (kk, blk) in a if kk == k)
            ak = np.zeros((nb * self.m, k), np.int8)
            for (kk, blk), (_c, codes) in a.items():
                if kk == k:
                    ak[blk * self.m:blk * self.m + len(codes)] = codes
            rows += [ak, np.concatenate([self._weights[w][1] for w, (_n, kw) in enumerate(run.shapes_) if kw == k])]
        self.root = root
        self.registered = [G.Registered.commit(digest, pre + role, row_words(r).tolist(),
                                               [hm96_salt(run._key, nonce(i), j) for j in range(len(r))], schema=V2,
                                               window=root)
                           for i, (role, r) in enumerate(zip(HW.roles(ks), rows))]
        self._got, self._weights = [], {}

    def ks(self) -> list[int]:
        """The window's contractions, ascending: one tile template, one A and one B value each."""
        return sorted({int(r.value.rsplit("/", 1)[1]) for r in self.registered})

    def values(self) -> dict[str, int]:
        """`registered.reads`' ``values``: each value's words a row."""
        return {r.value: r.words for r in self.registered}

    def schemas(self) -> dict[str, str]:
        """`registered.reads`' ``schemas``: every value's `V2`."""
        return {r.value: r.schema for r in self.registered}

    def ports(self) -> dict[str, str]:
        """The window's registered reads' circuit ports (`h1window.window_ports` of its Program)."""
        return HW.window_ports(self.program)

    def entries(self) -> list[dict[str, Any]]:
        return [r.entry() for r in self.registered]

    def digest(self) -> bytes:
        """SHA-512 over the trees' entries and what `close` refused, canonical JSON: what `calls_root` binds."""
        return hashlib.sha512(DIGEST_TAG + canonical_json({"entries": self.entries(), "refused": self.refused})).digest()

    def query(self) -> dict[str, Any]:
        """The window Program's partition query: every tile of each K's template one unit (`h1window.query`)."""
        return HW.query(self.program)


def trees(run: Any) -> list[ServedTree]:
    """The window's served trees, each K's A then B: refused unless `run` committed them as `Served` values at its `close`."""
    served = getattr(run, "served", None)
    if not isinstance(served, Served):
        raise ValueError(f"{run.entry}: the window's rows aren't registered values (the run has no `Served`)")
    if not run.closed:
        raise ValueError(f"{run.entry}: the window isn't closed, so its trees aren't committed")
    if served.refused or not served.registered:
        raise ValueError(f"{run.entry}: the window's trees were refused: {'; '.join(served.refused) or 'none committed'}")
    return [ServedTree.of(r) for r in served.registered]


def register(run: Any, log: Sequence[Mapping[str, Any]] = (), directory: str | Path | None = None,
             ports: Mapping[str, str] | None = None) -> ServedRegistration:
    """The window's trees registered on its window Program and query, every K's A and B in one registration
    (`register_served`, after this prover's earlier records `log`, under the circuit ports `ports`, default the window's
    own, `Served.ports`, and the window's open root `Served.root`), written as `FILE` in `directory` (default the run's
    ``transcript_dir``): the one file ``flock-verify --zk --registered`` reads for the window's session."""
    got = trees(run)
    served = run.served
    reg = register_served(served.program, served.query(), got, log=log, ports=served.ports() if ports is None else ports,
                          window_root=served.root)
    d = directory if directory is not None else run.transcript_dir
    if d is not None:
        Path(d).mkdir(parents=True, exist_ok=True)
        reg.write(Path(d) / FILE)
    return reg


__all__ = ["FILE", "NONCE", "PREFIX", "Served", "V2", "nonce", "prefix", "register", "row_words", "trees"]
