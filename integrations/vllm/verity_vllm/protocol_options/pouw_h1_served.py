"""H1's served trees (VLLM-64): a window's A codes and B rows, committed as the registered parameters ``a`` and ``b`` of
compute accounting's window Program (`h1window.program` over the tile unit), so a served path registers them as ports
of the verifier's own ``--registered`` file (`one_stage.served.register_served`) and verifies with ``flock-verify --zk
--registered``.

* **Values.** The Program's parameters are named ``h1/<window>/<role>`` (`names`, over `h1window.ROLES`): ``<window>``
  the window's key (`Served.key`: the window-open root, `salt.window_open_root` of the weights' root, round 0's request
  table and the session, or the key the run is given), so each window's values are its own and a log's R7-held rule never
  meets one twice.
* **Calls and blocks.** Call (s, w) of the run is the Program's call ``("s<s>/<module>", n, s·B + blocks[w])``: ``blocks``
  gives each linear's input within a step (q, k and v read one, so they share a block), B the blocks a step reads. A
  block's rows are its call's T slots (`pouw_h1.slots`, the served rows first, empty slots zero codes), padded with zero
  rows to the Program's slot M (`h1window.slot`); every call sharing a block must hand over the same codes. ``b`` is each
  call's n weight rows, call after call.
* **Rows** are ``hm96-sha512/row/v2`` bit rows (`V2`): an A row is its K int8 codes only, a B row its K weight codes, each
  as K/2 u16 words, code t's bits leaves 8t .. 8t + 7 (`row_words`, the tile's port rows). Row i's ``x`` is
  `rowleaf.sha512_row_v2_digest` over its bit row, its salt `audit.hm96_salt` of the run's key at the value's nonce
  (`A_NONCE`, `B_NONCE`), and its leaf `row_leaf(D, i, ..., V2)` under the registered-value domain D of the Program's
  digest, the value, K/2 words and its rows (`registered.domain`). The A scale is no value of the window Program: no
  unit reads it, and it stays bound in each call's A table under its round root, with y and z (`pouw_h1.H1Run`).
* **When.** `Served.record` keeps each call's codes as the run records it and `Served.close` commits both trees at the
  run's `close`, after its last round; `calls_root` binds their entries (`digest`), so the draw's coin opens after them.
  What `close` refuses (a window that isn't static, no key, a call whose codes aren't its shared block's, calls of two K
  or an odd K, a weight that isn't the call's n × K int8 codes, what `h1window.program` refuses) leaves no tree, and
  `trees` raises it. B repeats each step's weights: the Program has no B rows shared across steps.
* `trees` gives the window's two `ServedTree`s and `register` registers them and writes ``--registered`` (`FILE`) in the
  window's directory, refusing a value an earlier record of the log registered under another root (R7-held, `held`)
  before `register_served`, which reads ``hm96-sha512/row/v1`` rows alone until proofs' REGV2 and REGPORT land.
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
from verity.core.protocols.sampled_proofs.one_stage.served import ServedRegistration, ServedTree, register_served
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, bit_row, row_leaf, sha512_row_v2_digest

from verity_vllm.protocol_options import pouw_h1_requests as RQ

#: the rows' format: the tile's port rows
V2 = SCHEMA_HM96_SHA512_ROW_V2
FILE = "pouw-h1-registered.json"
VALUE = "h1/{window}/{role}"
#: the hm96 salt stream's nonces of the window's A and B rows, past every round's and call's tables (`pouw_h1_requests`)
A_NONCE = RQ.REQUESTS_NONCE + (1 << 42)
B_NONCE = A_NONCE + 1
DIGEST_TAG = b"verity/pouw/h1/served/v1\0"
_INST = hm96.SHA512


def names(window: bytes) -> tuple[str, ...]:
    """The window Program's parameters for the window keyed `window` (its first 16 bytes, hex), in `h1window.ROLES` order."""
    return tuple(VALUE.format(window=window[:16].hex(), role=r) for r in HW.ROLES)


def outputs(window: bytes) -> tuple[str, ...]:
    """The window Program's unit outputs (`h1window.OUTPUTS`), named as its parameters."""
    return tuple(VALUE.format(window=window[:16].hex(), role=r) for r in HW.OUTPUTS)


def row_words(codes: np.ndarray) -> np.ndarray:
    """int8 code rows ``(n, K)`` as their bit rows' u16 words ``(n, K / 2)``: code t's bits are leaves 8t .. 8t + 7."""
    b = np.ascontiguousarray(np.asarray(codes, dtype=np.int8).view(np.uint8))
    return b.reshape(len(b), -1).view("<u2").astype(np.uint16)


@dataclass(frozen=True)
class Tree:
    """One committed value: its rows' u16 words, their salts and `b || c`, and the tree under its registered domain."""

    value: str
    words: np.ndarray
    salts: tuple[bytes, ...]
    bcs: tuple[bytes, ...]
    levels: tuple[tuple[bytes, ...], ...]
    domain: str

    @classmethod
    def commit(cls, program: str, value: str, words: np.ndarray, salts: Sequence[bytes]) -> "Tree":
        n, w = words.shape
        dom = G.domain(program, value, w, n)
        bcs = tuple(_INST.commit_string(_INST.default_key, sha512_row_v2_digest(*bit_row((row.tolist(), 16))), s)
                    for row, s in zip(words, salts))
        lv = tree_levels(dom, [row_leaf(dom, i, _INST.tree_leaf(_INST.default_key, bc), V2) for i, bc in enumerate(bcs)])
        return cls(value, words, tuple(salts), bcs, tuple(tuple(x) for x in lv), dom.domain_id.hex())

    @property
    def root(self) -> str:
        return self.levels[-1][0].hex()

    def served(self) -> ServedTree:
        n, w = self.words.shape
        return ServedTree(self.value, self.root, int(n), int(w), self.domain, row_schema=V2)


@dataclass
class Served:
    """How an `H1Run` commits its window as the window Program's registered values. ``tile(k)``: the tile unit at
    contraction k (`h1tile.PouwH1Tile`'s binding, or a stand-in of its parameter shape). ``blocks``: each linear's input
    block within a step (default each its own; ``(0, 0, 0, 1)`` for q, k, v and o). ``window``: the window's key, else
    the window-open root its commitment names. After `close`: ``program`` (None when refused), ``trees`` (A then B),
    ``calls`` (the Program's), ``m`` (its A slot) and ``refused`` (why there are no trees)."""

    tile: Callable[[int], Any]
    blocks: Sequence[int] | None = None
    window: bytes | None = None
    program: Program | None = None
    trees: list[Tree] = field(default_factory=list)
    calls: list[tuple[str, int, int]] = field(default_factory=list)
    m: int = 0
    refused: list[str] = field(default_factory=list)
    #: each recorded call's (call, its T slots' codes, its weight codes)
    _got: list[tuple[Any, np.ndarray, np.ndarray]] = field(default_factory=list)
    #: each linear's last weight, the tensor and its codes, so a step reusing it copies nothing
    _weights: dict[int, tuple[Any, np.ndarray]] = field(default_factory=dict)
    _tile: Any = None

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
        """Call `call`'s A codes (``a`` [m, k], its served rows) and weights (``weight`` [n, k]), as the run records it."""
        from verity_vllm.protocol_options.pouw_h1 import numpy_of
        codes = np.zeros((call.rows, call.k), np.int8)
        if a is not None and call.m:
            codes[:call.m] = numpy_of(a).reshape(call.m, call.k)
        last = self._weights.get(call.layer)
        w = last[1] if last is not None and last[0] is weight else None if weight is None else numpy_of(weight)
        if weight is not None:
            self._weights[call.layer] = (weight, w)
        if w is None or w.dtype != np.int8 or w.shape != (call.n, call.k):
            self.refused.append(f"call {call.index}: its weight is not its {call.n} × {call.k} int8 codes")
        self._got.append((call, codes, w))

    def _layout(self, linears: int) -> list[int]:
        """Each recorded call's A block: step s's linear w reads block s·B + blocks[w]."""
        per = list(self.blocks) if self.blocks is not None else list(range(linears))
        if len(per) != linears or sorted(set(per)) != list(range(1 + max(per, default=-1))):
            raise ValueError(f"blocks {per} are no input block 0, 1, ... of each of the window's {linears} linears")
        return [c.step * (1 + max(per)) + per[c.layer] for c, _codes, _w in self._got]

    def close(self, run: Any) -> None:
        """Commit the window's A and B trees under the window Program, after the run's last round."""
        ks = sorted({c.k for c, _codes, _w in self._got})
        try:
            key, blocks = self.key(run.commits), self._layout(len(run.shapes_))
        except ValueError as e:
            self.refused.append(str(e))
        if not run.static():
            self.refused.append("the window isn't static: a call off its public position or refused, so the Program's calls "
                                "aren't the window's")
        if not self._got:
            self.refused.append("the window has no call")
        elif len(ks) > 1 or ks[0] % 2:
            self.refused.append(f"calls of K {ks}: a window Program's tile is one K, of whole u16 words of codes")
        a: dict[int, tuple[int, np.ndarray]] = {}
        for i, (c, codes, _w) in enumerate(self._got) if not self.refused else ():
            first = a.setdefault(blocks[i], (c.index, codes))
            if not np.array_equal(first[1], codes):
                self.refused.append(f"call {c.index}'s A codes aren't those of its block {blocks[i]} (call {first[0]}'s): "
                                    "the calls that share an input share its block")
        if self.refused:
            return
        self._tile = self.tile(ks[0])
        self.m = HW.slot(run.rows, self._tile.signature()[0][0][1].n)
        self.calls = [(f"s{c.step}/{c.module}", c.n, blk) for (c, _codes, _w), blk in zip(self._got, blocks)]
        try:
            self.program = HW.program(self._tile, self.calls, self.m, names(key))
        except ValueError as e:
            self.refused.append(str(e))
            return
        digest = P.program_digest(self.program)
        rows = np.zeros((len(a) * self.m, ks[0]), np.int8)
        for blk, (_c, codes) in a.items():
            rows[blk * self.m:blk * self.m + len(codes)] = codes
        b = np.concatenate([w for _c, _codes, w in self._got])
        va, vb = names(key)[:2]
        self.trees = [Tree.commit(digest, va, row_words(rows), [hm96_salt(run._key, A_NONCE, i) for i in range(len(rows))]),
                      Tree.commit(digest, vb, row_words(b), [hm96_salt(run._key, B_NONCE, i) for i in range(len(b))])]
        self._got, self._weights = [], {}

    def entries(self) -> list[dict[str, Any]]:
        return [t.served().entry() for t in self.trees]

    def digest(self) -> bytes:
        """SHA-512 over the trees' entries and what `close` refused, canonical JSON: what `calls_root` binds."""
        return hashlib.sha512(DIGEST_TAG + canonical_json({"entries": self.entries(), "refused": self.refused})).digest()

    def query(self) -> dict[str, Any]:
        """The window Program's partition query: every tile one unit (`h1window.query`)."""
        return HW.query(self._tile)


def trees(run: Any) -> list[ServedTree]:
    """The window's served trees, A then B: refused unless `run` committed them as `Served` values at its `close`."""
    served = getattr(run, "served", None)
    if not isinstance(served, Served):
        raise ValueError(f"{run.entry}: the window's rows aren't registered values (the run has no `Served`)")
    if not run.closed:
        raise ValueError(f"{run.entry}: the window isn't closed, so its trees aren't committed")
    if served.refused or not served.trees:
        raise ValueError(f"{run.entry}: the window's trees were refused: {'; '.join(served.refused) or 'none committed'}")
    return [t.served() for t in served.trees]


def held(got: Sequence[ServedTree], log: Sequence[Mapping[str, Any]]) -> None:
    """R7-held: refused (ValueError) when an earlier record of the log registered one of `got`'s values under another
    root, as a later window keyed as an earlier one would."""
    roots = R.held_roots(log)
    for t in got:
        if roots.get(t.value, t.root) != t.root:
            raise ValueError(f"served tree {t.value!r}: an earlier record of the log registered it under root "
                             f"{roots[t.value][:16]}..., and a value is committed once (R7-held)")


def register(run: Any, log: Sequence[Mapping[str, Any]] = (), ports: Mapping[str, str] | None = None,
             directory: str | Path | None = None) -> ServedRegistration:
    """The window's trees registered on its window Program and the tile's query (`register_served`, after this prover's
    earlier records `log`; `ports` the tile's circuit ports by template parameter), and the ``--registered`` file written
    as `FILE` in `directory` (default the run's ``transcript_dir``), the file ``flock-verify --zk --registered`` reads."""
    got = trees(run)
    held(got, log)
    reg = register_served(run.served.program, run.served.query(), got, log=log, ports=ports)
    d = directory if directory is not None else run.transcript_dir
    if d is not None:
        Path(d).mkdir(parents=True, exist_ok=True)
        reg.write(Path(d) / FILE)
    return reg


__all__ = ["A_NONCE", "B_NONCE", "FILE", "Served", "Tree", "V2", "VALUE", "held", "names", "outputs", "register",
           "row_words", "trees"]
