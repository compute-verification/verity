"""H1's served trees (VLLM-64): a window's A codes and B rows, committed as the registered parameters ``a/<K>`` and
``b/<K>`` of compute accounting's window Program (`h1window.program`, one tile template per K), so a served path registers
them as ports of the verifier's own ``--registered`` file (`one_stage.served.register_served`) and verifies with
``flock-verify --zk --registered``.

* **Values.** The Program's parameters are ``h1/<window>/`` and their role (`prefix`, `h1window.roles`): ``<window>`` the
  window's key (`Served.key`: the window-open root, `salt.window_open_root` of the weights' root, round 0's request table
  and the session, or the key the run is given), so each window's values are its own and a log's R7-held rule never meets
  one twice.
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
  `registered.Registered.commit` in `V2` under the registered-value domain of the Program's digest, the value, K/2 words
  and its rows, row i's salt `audit.hm96_salt` of the run's key at the value's nonce (`nonce`). The A scale is no value
  of the window Program: no unit reads it, and it stays bound in each call's A table under its round root, with y and z
  (`pouw_h1.H1Run`).
* **When.** `Served.record` keeps each call's codes as the run records it. At each round's close `Served.segments` hands
  the run step s's rows of each ``a/<K>`` as aligned subtrees (`credit.cover`), which round s's root commits before its
  coin (`pouw_h1_requests.round_row`, PRF-132); the first round fixes the values' domains under the `declared` window
  Program with the window's prefix. `Served.close` commits the trees at the run's `close`, after its last round, refused
  unless its Program is that one and each A tree is the one over the round roots' segments (`credit.a_link`);
  `calls_root` binds their entries (`digest`), so the draw's coin opens after them.
  What `close` refuses (a window that isn't static, no key, blocks that aren't one per linear, a call whose codes aren't
  its shared block's, an odd K, a linear with no weight or whose weight isn't its n × K int8 codes or changes within the
  window, what `h1window.program` refuses) leaves no tree, and `trees` raises it.
* `trees` gives the window's `ServedTree`s, each K's A then B, and `register` registers them all at once: a window is one
  statement, one C-Flock session and one registration, so one ``--registered`` file (`FILE`) in the window's directory,
  whatever its K. It refuses a value an earlier record of the log registered under its domain and another root (R7-held,
  `held`) before `register_served`.
* `declared_layout` and `Served.layout` give the window Program's `h1window.program` arguments, the layout the record
  commits and core's credit draws from (`credit.strata`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G, registration as R
from verity.core.protocols.sampled_proofs.one_stage.served import ServedRegistration, ServedTree, register_served
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import sha512_row_v2_digest

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


def served_tree(reg: G.Registered) -> ServedTree:
    e = reg.entry()
    return ServedTree(e["value"], e["root"], e["leaves"], e["words"], e["domain"], reg.word_bits, row_schema=reg.schema)


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
    each ``(n, K)``, named by ``modules``) in order, laid out as `Served.close` lays out the calls it records, its A slot
    for ``rows`` token rows. With no ``prefix`` it names no window, so it is one for every window of the declaration (a
    period's ``--circuit``, its `program_digest`); with a window's `prefix` it is that window's `Served.program`."""
    lay = declared_layout(shapes, modules, rows, steps, tile, blocks)
    return HW.program([tuple(x) for x in lay["linears"]], [tuple(x) for x in lay["calls"]], lay["m"], tile, prefix, unit)


def declared_layout(shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int,
                    tile: tuple[int, int] = PH.TILE, blocks: Sequence[int] | None = None) -> dict[str, Any]:
    """`declared`'s arguments to `h1window.program` as the window's layout, the form `credit.strata` reads (its units and
    their order, so the draw's population): ``{calls: [[linear, A block], ...], linears: [[name, K, n], ...], m, tile}``."""
    mods = [modules.get(w, str(w)) for w in range(len(shapes))]
    calls = [(s, w) for s in range(steps) for w in range(len(shapes))]
    return {"calls": [[mods[w], b] for (_s, w), b in zip(calls, layout(shapes, calls, blocks))],
            "linears": [[mods[w], k, n] for w, (n, k) in enumerate(shapes)], "m": HW.slot(rows, tile), "tile": list(tile)}


@dataclass
class Served:
    """How an `H1Run` commits its window as the window Program's registered values. ``tile``: the Program's R × C
    (`pouw_h1.TILE`). ``unit(k)``: the tile unit at contraction k (default `h1window.tile_unit`, the H1 tile; a stand-in
    of its parameter shape in tests). ``blocks``: each linear's input block within a step (default each its own;
    ``(0, 0, 0, 1)`` for q, k, v and o). ``window``: the window's key, else the window-open root its commitment names.
    ``prover``: the prover's private directory the run keeps its salt key and each call's values in (`pouw_h1_prover`), so a
    drawn unit opens after the close. After `close`: ``program`` (None when refused), ``registered`` (each K's A then B,
    `registered.Registered`), ``linears`` and ``calls`` (the Program's), ``m`` (its A slot) and ``refused`` (why there are
    no trees)."""

    tile: tuple[int, int] = PH.TILE
    unit: Callable[[int], Any] | None = None
    blocks: Sequence[int] | None = None
    window: bytes | None = None
    prover: str | None = None
    program: Program | None = None
    registered: list[G.Registered] = field(default_factory=list)
    linears: list[tuple[str, int, int]] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)
    m: int = 0
    refused: list[str] = field(default_factory=list)
    #: each recorded call and its T slots' codes
    _got: list[tuple[Any, np.ndarray]] = field(default_factory=list)
    #: each linear's weight, the tensor and its codes, taken once
    _weights: dict[int, tuple[Any, np.ndarray]] = field(default_factory=dict)
    #: from the first round's `segments`: the declared Program's digest, and each A value, its K, its blocks a step and
    #: its tree's index in `roles` order (`nonce`)
    _digest: str | None = None
    _a: list[tuple[str, int, int, int]] = field(default_factory=list)

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

    def _open(self, run: Any) -> None:
        key = self.key(run.commits)
        program = declared(run.shapes_, run.modules, run.rows, run.steps, self.tile, self.blocks, prefix(key), self.unit)
        self._digest, self.m = P.program_digest(program), HW.slot(run.rows, self.tile)
        first = layout(run.shapes_, [(0, w) for w in range(len(run.shapes_))], self.blocks)
        ks = sorted({k for _n, k in run.shapes_})
        roles = HW.roles(ks)
        self._a = [(prefix(key) + roles[2 * i], k, 1 + max(b for b, (_n, kw) in zip(first, run.shapes_) if kw == k), 2 * i)
                   for i, k in enumerate(ks)]

    def segments(self, run: Any, r: int) -> dict[str, list[tuple[int, int, bytes]]]:
        """Round r's rows of each A value, the blocks step r's calls read (`layout`), as the aligned subtrees
        (`credit.cover`, each ``(level, index, root)``) of the tree `close` registers, row j's leaf its codes' under
        `nonce`'s salt; the first round fixes the values' domains under the declared window Program (`declared`, the
        window's `prefix`). Nothing once the window is refused, and it then registers nothing."""
        if self.refused:
            return {}
        try:
            if self._digest is None:
                self._open(run)
            blocks = layout(run.shapes_, [(r, w) for w in range(len(run.shapes_))], self.blocks)
        except ValueError as e:
            self.refused.append(f"round {r}: {e}")
            return {}
        inst, out = hm96.SHA512, {}
        for value, k, per, i in self._a:
            codes, have, lo = np.zeros((per * self.m, k), np.int8), set(), r * per * self.m
            for c, got in self._got:
                b = blocks[c.layer] - r * per
                if c.step == r and c.k == k and b not in have:
                    have.add(b)
                    codes[b * self.m:b * self.m + len(got)] = got
            dom = G.domain(self._digest, value, k // 2, run.steps * per * self.m)
            leaves = [G.leaf(dom, lo + j, inst.commit_string(inst.default_key, sha512_row_v2_digest(row.tobytes(), 8 * k),
                                                            hm96_salt(run._key, nonce(i), lo + j)), V2)
                      for j, row in enumerate(codes)]
            out[value] = [(lv, ix, _subtree(dom, leaves[(ix << lv) - lo:((ix + 1) << lv) - lo], lv, ix))
                          for lv, ix in CR.cover(lo, lo + len(leaves))]
        return out

    def close(self, run: Any) -> None:
        """Commit the window's A and B trees under the window Program, after the run's last round."""
        try:
            key = self.key(run.commits)
            blocks = layout(run.shapes_, [(c.step, c.layer) for c, _codes in self._got], self.blocks)
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
        if digest != self._digest:
            self.refused.append("the window Program at the close is not the declared one its first round fixed its A "
                                "values' domains under")
            return
        rows: list[np.ndarray] = []
        for k in ks:
            nb = 1 + max(blk for (kk, blk) in a if kk == k)
            ak = np.zeros((nb * self.m, k), np.int8)
            for (kk, blk), (_c, codes) in a.items():
                if kk == k:
                    ak[blk * self.m:blk * self.m + len(codes)] = codes
            rows += [ak, np.concatenate([self._weights[w][1] for w, (_n, kw) in enumerate(run.shapes_) if kw == k])]
        self.registered = [G.Registered.commit(digest, pre + role, row_words(r).tolist(),
                                               [hm96_salt(run._key, nonce(i), j) for j in range(len(r))], schema=V2)
                           for i, (role, r) in enumerate(zip(HW.roles(ks), rows))]
        self._got, self._weights = [], {}
        why = CR.a_link(run.round_log, self.entries())
        if why is not None:
            self.refused.append(why[1])

    def layout(self) -> dict[str, Any] | None:
        """The committed Program's `h1window.program` arguments (`declared_layout`'s form), None when `close` refused it."""
        if self.program is None:
            return None
        return {"calls": [list(x) for x in self.calls], "linears": [list(x) for x in self.linears], "m": self.m,
                "tile": list(self.tile)}

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
        return digest_of(self.entries(), self.refused)

    def query(self) -> dict[str, Any]:
        """The window Program's partition query: every tile of each K's template one unit (`h1window.query`)."""
        return HW.query(self.program)


def _subtree(dom: Any, leaves: Sequence[bytes], level: int, index: int) -> bytes:
    """The root of subtree `index` at `level` over its leaves, hashed as `merkle.tree_levels` hashes the whole tree."""
    nodes = list(leaves)
    for d in range(level):
        base = index << (level - d - 1)
        nodes = [dom.node(d, base + j, nodes[2 * j], nodes[2 * j + 1]) for j in range(len(nodes) // 2)]
    return nodes[0]


def digest_of(entries: Sequence[Mapping[str, Any]], refused: Sequence[str]) -> bytes:
    """`Served.digest` of the trees' `entries` and `refused`, as a record carries them (the commitment's ``served``)."""
    return hashlib.sha512(DIGEST_TAG + canonical_json({"entries": list(entries), "refused": list(refused)})).digest()


def trees(run: Any) -> list[ServedTree]:
    """The window's served trees, each K's A then B: refused unless `run` committed them as `Served` values at its `close`."""
    served = getattr(run, "served", None)
    if not isinstance(served, Served):
        raise ValueError(f"{run.entry}: the window's rows aren't registered values (the run has no `Served`)")
    if not run.closed:
        raise ValueError(f"{run.entry}: the window isn't closed, so its trees aren't committed")
    if served.refused or not served.registered:
        raise ValueError(f"{run.entry}: the window's trees were refused: {'; '.join(served.refused) or 'none committed'}")
    return [served_tree(r) for r in served.registered]


def held(got: Sequence[ServedTree], log: Sequence[Mapping[str, Any]]) -> None:
    """R7-held: refused (ValueError) when an earlier record of the log registered one of `got`'s values under its domain
    and another root, as a later window keyed as an earlier one would (`registration.held_roots`, by value and domain)."""
    roots = R.held_roots(log)
    for t in got:
        if roots.get((t.value, t.domain), t.root) != t.root:
            raise ValueError(f"served tree {t.value!r}: an earlier record of the log registered it under root "
                             f"{roots[(t.value, t.domain)][:16]}..., and a value is committed once (R7-held)")


def register(run: Any, log: Sequence[Mapping[str, Any]] = (), directory: str | Path | None = None,
             ports: Mapping[str, str] | None = None) -> ServedRegistration:
    """The window's trees registered on its window Program and query, every K's A and B in one registration
    (`register_served`, after this prover's earlier records `log`, under the circuit ports `ports`, default the window's
    own, `Served.ports`), written as `FILE` in `directory` (default the run's ``transcript_dir``): the one file
    ``flock-verify --zk --registered`` reads for the window's session."""
    got = trees(run)
    held(got, log)
    served = run.served
    reg = register_served(served.program, served.query(), got, log=log, ports=served.ports() if ports is None else ports)
    d = directory if directory is not None else run.transcript_dir
    if d is not None:
        Path(d).mkdir(parents=True, exist_ok=True)
        reg.write(Path(d) / FILE)
    return reg


__all__ = ["FILE", "NONCE", "PREFIX", "Served", "V2", "declared", "digest_of", "held", "layout", "nonce", "prefix", "register",
           "row_words", "served_tree", "trees"]
