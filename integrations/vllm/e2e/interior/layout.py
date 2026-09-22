"""[E2, EOD 2026-09-11] The interior commitment layout ``interior-layout/chunks-4096/v2`` as code (brief interface 4; INTEGRATOR D13).

Two rules are selectable by name (``rule=``): ``SCHEMA_V2`` (the default; clause 4 below) and ``SCHEMA_V1`` (records written before
D13: a chunk closed on every width change -- 898,407 leaves for 4.62 M words under ArgmaxStep's 16/32-bit interleave).  A pin / index
names its rule in ``layout.rule`` (absent = v1); the verifier validates a record by the rule it names, writers use the default.

The rule (verbatim in ``out/eod/reports/e2.md``; E3 imports this module, never re-implements it):

1. **Tree.** One sibling tree per retained RU ``(req, step)``: tree id ``interior/<req>/<step>``; files
   ``adapter/selective/interior/<req>/<step>/{index.json, gates.bin, leaf-<rank:06d>.bin}``; root recorded in
   ``commitments.json#interior_roots["<req>/<step>"]``.  The phase-0 boundary/weights trees and the capture's RU trees are never
   rewritten.
2. **Calls.** The RU's root calls in *pinned call order* = increasing root-node index of the pinned program (= increasing gate offset).
   Serve_v1: the root calls of the step's positions ``t`` (:func:`serve_calls`); any other program (CORE): the caller's ordered
   ``(call_id, lo, hi)`` list, disjoint and increasing.
3. **Gates of a call.** The *required interior gates* of the call = the partition's VU output gates in ``[lo, hi)`` that have no
   boundary position (E3's oracle = E1's inventory), in increasing gate index.  Input and boundary gates are never interior.  Each
   gate is exactly one *word* = its output ``Value<w>`` as ``ceil(w/8)`` little-endian bytes.
4. **Chunks = leaves (v2).** Partition the call's gates by output width ``w``; order the width groups by increasing ``w``; within a
   group keep increasing gate index.  Cut each group every 4096 words (the last chunk of a group is shorter).  Chunk ``k`` (0-based,
   canonical decimal, restarting at 0 for every ``(call, w)``) of call ``c`` has leaf id ``<prefix>call=<c>/w=<w>/chunk=<k>``, dtype
   ``uint8|uint16|uint32|uint64`` for w in {8,16,32,64} else ``u<w>``, shape ``[n]``, bytes = the n words in gate order.  With ``r`` =
   the 0-based rank of ``g`` among the call's gates of width ``w`` (gate order): ``k = r // 4096``, ``flat = r % 4096``; byte offset
   ``flat * ceil(w/8)``; interface layout ``Array(n, Bits(w))``.  (v1: one walk in gate order, a chunk closed at 4096 words / a width
   change / the call's end, leaf id ``<prefix>call=<c>/chunk=<k>`` with ``k`` over the call's chunks.)
5. **Hashes.** leaf = ``H("verity-vllm/leaf/v1", leaf_id, dtype, "[n]", bytes)`` (``verity_vllm.hashing.leaf_hash``); rank = layout
   order (call order, then increasing width, then chunk index) -- not code-point order, not gate order; nodes ``verity-vllm/node/v1``,
   odd tail ``verity-vllm/lift/v1``, empty ``verity-vllm/empty/v1``; ``ids_digest = H("verity-vllm/ids/v1", ids in rank order)``.
   (:mod:`e2e.interior.tree`.)
6. **Pin.** ``protocol_instance.json#commitments.interior_layout = layout_pin()`` -- ``{"framing": "chunks-4096", "word_order":
   "gate", "tree": "interior/<req>/<step>", "digest": layout_digest(), "rule": "interior-layout/chunks-4096/v2"}``; the digest is a
   constant of the rule (canonical JSON of its parameters, including the rule name and the width-group clause), the per-tree
   ``ids_digest`` binds the realised leaf list.  ``word_order`` stays ``"gate"``: within a chunk the words are in gate order.
7. **Refusals** (:func:`validate_supplied`, E9 at step 4 before ``r_s``): ``INTERIOR_REGION_MISSING``, ``INTERIOR_DUPLICATE_GATE``,
   ``INTERIOR_UNEXPECTED_GATE``, ``INTERIOR_UNEXPECTED_LEAF``, ``INTERIOR_WIDTH_MISMATCH``, ``INTERIOR_ORDER``,
   ``INTERIOR_LAYOUT_DIGEST``, ``INTERIOR_INDEX_MALFORMED``.

Nothing here reads a value, hashes anything or consults the prover's index for *positions*: the layout is a pure function of
``(calls, required gates, widths)``, which the verifier derives from the pinned program + partition + its boundary oracle.
Program-agnostic: E8's CORE circuit passes its own ``CallSpan`` list.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CHUNK_WORDS",
    "FRAMING",
    "RULES",
    "SCHEMA",
    "SCHEMA_V1",
    "SCHEMA_V2",
    "TABLE_ORIGIN",
    "TABLE_ROWS_PER_LEAF",
    "TABLE_RULE",
    "TABLE_TREE_PATTERN",
    "TREE_PATTERN",
    "WORD_ORDER",
    "CallSpan",
    "InteriorLayout",
    "LazyInteriorLayout",
    "LeafSpec",
    "check_pin",
    "chunks",
    "derive_layout",
    "dtype_for_width",
    "layout_digest",
    "layout_pin",
    "leaf_id",
    "leaf_id_rule",
    "required_regions",
    "rule_of",
    "serve_calls",
    "tree_id",
    "validate_step4",
    "validate_supplied",
    "widths_from_program",
    "word_bytes",
]

SCHEMA_V1 = "interior-layout/chunks-4096/v1"  # chunk closes on width change (records written before D13 keep it, by name)
SCHEMA_V2 = "interior-layout/chunks-4096/v2"  # D13: words grouped by width within the call; leaf id carries w=<w>
SCHEMA = SCHEMA_V2
TABLE_RULE = "table-layout/v1"  # D23: the pinned MUFU tables (e2e.interior.tables); rows_per_leaf 2048, row = (index << 32) | T[index]
TABLE_ROWS_PER_LEAF = 2048
TABLE_TREE_PATTERN = "table/<name>"
RULES = (SCHEMA_V1, SCHEMA_V2, TABLE_RULE)
INDEX_SCHEMA = "verity-interior/index/v1"
FRAMING = "chunks-4096"
WORD_ORDER = "gate"
CHUNK_WORDS = 4096
TREE_PATTERN = "interior/<req>/<step>"
ORIGIN = "interior"  # brief interface 3: interior leaves carry origin "interior"; boundary leaves never do
TABLE_ORIGIN = "table"  # D23: table leaves carry origin "table"
LEAF_TAG = "verity-vllm/leaf/v1"

# refusal codes (the string a problem line starts with; E9 matches on the code)
INTERIOR_REGION_MISSING = "INTERIOR_REGION_MISSING"
INTERIOR_DUPLICATE_GATE = "INTERIOR_DUPLICATE_GATE"
INTERIOR_UNEXPECTED_GATE = "INTERIOR_UNEXPECTED_GATE"
INTERIOR_UNEXPECTED_LEAF = "INTERIOR_UNEXPECTED_LEAF"
INTERIOR_WIDTH_MISMATCH = "INTERIOR_WIDTH_MISMATCH"
INTERIOR_ORDER = "INTERIOR_ORDER"
INTERIOR_LAYOUT_DIGEST = "INTERIOR_LAYOUT_DIGEST"
INTERIOR_INDEX_MALFORMED = "INTERIOR_INDEX_MALFORMED"
INTERIOR_ROOT_MISMATCH = "INTERIOR_ROOT_MISMATCH"  # D35: the root recomputed from the shipped leaf files != index.json#root / interior_roots

_FIXED_DTYPES = {8: "uint8", 16: "uint16", 32: "uint32", 64: "uint64"}


def word_bytes(width: int) -> int:
    """``ceil(w/8)``: the bytes one word of a ``width``-bit gate occupies in its chunk."""
    w = int(width)
    if w <= 0:
        raise ValueError(f"width must be positive, got {width!r}")
    return (w + 7) // 8


def dtype_for_width(width: int) -> str:
    """The leaf dtype string hashed into the leaf: ``uint8/16/32/64`` for those widths, ``u<w>`` otherwise."""
    w = int(width)
    return _FIXED_DTYPES.get(w, f"u{w}")


def tree_id(request_id: str, step: int | str) -> str:
    """``interior/<req>/<step>``; ``step`` is spelled as the capture spells the RU step (``%06d`` for an int)."""
    s = f"{int(step):06d}" if isinstance(step, int) else str(step)
    return f"interior/{request_id}/{s}"


def _check_rule(rule: str | None) -> str:
    r = SCHEMA if rule is None else str(rule)
    if r not in RULES:
        raise ValueError(f"unknown interior layout rule {r!r} (known: {RULES})")
    return r


def leaf_id(prefix: str, call: str, chunk: int, width: int | None = None, *, rule: str | None = None) -> str:
    """v2 (default): ``<prefix>call=<call>/w=<width>/chunk=<k>`` -- ``k`` counts the chunks of THIS width group of the call;
    v1 (``rule=SCHEMA_V1``): ``<prefix>call=<call>/chunk=<k>`` -- ``k`` counts the call's chunks in gate order (``width`` ignored)."""
    if int(chunk) < 0:
        raise ValueError("chunk index must be >= 0")
    if _check_rule(rule) == SCHEMA_V1:
        return f"{prefix}call={call}/chunk={int(chunk)}"
    if width is None:
        raise ValueError(f"{SCHEMA}: leaf_id needs the chunk's width (w=<w> is part of the id)")
    return f"{prefix}call={call}/w={int(width)}/chunk={int(chunk)}"


def leaf_id_rule(pin: Mapping[str, Any] | None) -> Callable[..., str]:
    """E3 (``verifier_v3.locator.interior_layout_rule``): the ``leaf_id`` spelling of the rule a pin NAMES, as a callable
    ``(prefix, call, chunk, width=None) -> str`` -- v2 needs ``width``; a v1 pin (no ``rule`` field) ignores it.  ``None`` pin -> the current rule."""
    r = rule_of(pin) if pin is not None else SCHEMA

    def _id(prefix: str, call: Any, chunk: int, width: int | None = None) -> str:
        return leaf_id(prefix, str(call), int(chunk), width, rule=r)

    _id.rule = r  # type: ignore[attr-defined]
    return _id


# --- the rule's identity -----------------------------------------------------------------------------------------------


def _table_rule_document(rows_per_leaf: int = TABLE_ROWS_PER_LEAF) -> dict[str, Any]:
    """``table-layout/v1`` (D23): one tree per registered MUFU table, deterministic from the registry fixture alone."""
    return {
        "schema": TABLE_RULE,
        "framing": f"chunks-{int(rows_per_leaf)}",
        "rows_per_leaf": int(rows_per_leaf),
        "word_order": "index",
        "tables": "mufu/ex2 (2^23 rows), mufu/rcp (2^23), mufu/sqrt (2^24), mufu/rsq (2^24): verity_vllm.registry.prims MUFU tables, reconstructed from "
                  "the int8 delta fixtures (fa2_model.MufuTables.load / rms_triton_model.RmsTables.load)",
        "row": "row i = the 64-bit word (i << 32) | T[i] as 8 little-endian bytes (the reduce-mufu@1 / ftz-f32@1 table operand, arg width 64)",
        "leaf": "rows [k*rows_per_leaf, (k+1)*rows_per_leaf) in index order; dtype uint64, shape [rows_per_leaf]",
        "leaf_id": "table/<name>/chunk=<k>",
        "flat": "k = i // rows_per_leaf, flat = i % rows_per_leaf; byte offset flat * 8",
        "leaf_hash": f'{LEAF_TAG}(leaf_id, "uint64", "[rows_per_leaf]", bytes)',
        "rank": "k (index order)",
        "node": "verity-vllm/node/v1",
        "lift": "verity-vllm/lift/v1",
        "empty": "verity-vllm/empty/v1",
        "ids_digest": "verity-vllm/ids/v1 over the ids in rank order",
        "tree_id": TABLE_TREE_PATTERN,
        "openings": "verity_vllm.commit.Opening (single leaf, index = rank) / range over consecutive ranks",
    }


def _rule_document(chunk_words: int = CHUNK_WORDS, rule: str | None = None) -> dict[str, Any]:
    r = _check_rule(rule)
    if r == TABLE_RULE:
        return _table_rule_document(chunk_words)
    doc = {
        "schema": r,
        "framing": FRAMING,
        "chunk_words": int(chunk_words),
        "word_order": WORD_ORDER,
        "call_order": "pinned program root-node order (increasing gate offset); chunks never span calls",
        "gate_set": "partition VU output gates of the call without a boundary position, increasing gate index",
        "word": "one gate = ceil(w/8) little-endian bytes of its Value<w>",
        "dtype": "uint8|uint16|uint32|uint64 for w in {8,16,32,64}, else u<w>",
        "shape": "[n] (n = words in the chunk)",
        "leaf_hash": f'{LEAF_TAG}(leaf_id, dtype, "[n]", bytes)',
        "node": "verity-vllm/node/v1",
        "lift": "verity-vllm/lift/v1",
        "empty": "verity-vllm/empty/v1",
        "ids_digest": "verity-vllm/ids/v1 over the ids in rank order",
        "tree_id": TREE_PATTERN,
        "openings": "verity_vllm.commit.Opening (single leaf, index = rank) / range over consecutive ranks",
    }
    if r == SCHEMA_V1:
        doc.update({
            "split_on_width_change": True,
            "leaf_id": "<prefix>call=<call>/chunk=<k>",
            "flat": "rank of the gate among the chunk's gates; byte offset flat * ceil(w/8)",
            "rank": "layout order (call order, then chunk index)",
        })
    else:
        doc.update({
            "width_groups": "the call's gates are partitioned by width; groups ordered by increasing width; within a group gates in increasing "
                            "gate index; a group is cut every chunk_words words; chunk index k restarts at 0 per (call, width)",
            "leaf_id": "<prefix>call=<call>/w=<w>/chunk=<k>",
            "flat": "with r = rank of the gate among the call's gates of its width (gate order): k = r // chunk_words, flat = r % chunk_words; "
                    "byte offset flat * ceil(w/8)",
            "rank": "layout order (call order, then increasing width, then chunk index)",
        })
    return doc


def rule_of(pin: Mapping[str, Any] | None) -> str:
    """The rule a pin / index layout names: its ``rule`` field, else v1 (pins written before D13 carry no ``rule``)."""
    if not isinstance(pin, Mapping):
        return SCHEMA
    return _check_rule(pin.get("rule") or SCHEMA_V1)


def _chunk_words_of(pin_or_chunk_words: Mapping[str, Any] | int | None) -> int:
    """``chunk_words`` of a pin (``framing = "chunks-<n>"``) or of an int; a pin with a foreign framing / word order is a ``ValueError``."""
    if pin_or_chunk_words is None:
        return CHUNK_WORDS
    if isinstance(pin_or_chunk_words, Mapping):
        framing = str(pin_or_chunk_words.get("framing", ""))
        if not framing.startswith("chunks-") or not framing[len("chunks-"):].isdigit():
            raise ValueError(f"interior_layout.framing {framing!r} is not of this rule (chunks-<n>)")
        if pin_or_chunk_words.get("word_order", WORD_ORDER) not in (WORD_ORDER, "index"):
            raise ValueError(f"interior_layout.word_order {pin_or_chunk_words.get('word_order')!r} is not of this rule ({WORD_ORDER!r})")
        return int(framing[len("chunks-"):])
    return int(pin_or_chunk_words)


def layout_digest(pin: Mapping[str, Any] | int | None = None, *, chunk_words: int | None = None, rule: str | None = None) -> str:
    """sha256 of the canonical JSON of the rule's parameters -- what ``protocol_instance.json#commitments.interior_layout.digest`` pins.
    ``layout_digest()`` is the constant of the current rule (v2, 4096 words); ``layout_digest(chunk_words=n, rule=SCHEMA_V1)`` another;
    ``layout_digest(pin)`` (E9: ``== pin["digest"]``) the digest of the rule the pin NAMES (``rule`` field, absent = v1; ``framing``) --
    computed, never copied from the pin, so a pin carrying a foreign digest fails the comparison."""
    n = _chunk_words_of(pin if chunk_words is None else int(chunk_words))
    r = rule if rule is not None else (rule_of(pin) if isinstance(pin, Mapping) else SCHEMA)
    doc = json.dumps(_rule_document(n, r), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(doc).hexdigest()


def layout_pin(chunk_words: int = CHUNK_WORDS, *, rule: str | None = None) -> dict[str, Any]:
    """The pinned fields E6 writes into ``protocol_instance.json#commitments.interior_layout``: the brief's four (``framing``, ``word_order``,
    ``tree``, ``digest``) plus ``rule`` (the rule's name; v2 by default).  ``word_order`` stays ``"gate"``: within a chunk the words are in gate
    order under both rules."""
    r = _check_rule(rule)
    if r == TABLE_RULE:
        return {"framing": f"chunks-{int(chunk_words)}", "word_order": "index", "tree": TABLE_TREE_PATTERN,
                "digest": layout_digest(chunk_words=chunk_words, rule=r), "rule": r}
    return {"framing": FRAMING if chunk_words == CHUNK_WORDS else f"chunks-{int(chunk_words)}", "word_order": WORD_ORDER,
            "tree": TREE_PATTERN, "digest": layout_digest(chunk_words=chunk_words, rule=r), "rule": r}


def check_pin(pin: Mapping[str, Any] | None, *, chunk_words: int | None = None, rule: str | None = None) -> list[str]:
    """Refusals of a pinned ``interior_layout`` against a rule (empty = the pin names exactly that layout).  Default: the CURRENT rule (v2) --
    a v1 pin is refused by name so a writer cannot mix a v1 pin with v2 leaves; ``rule=rule_of(pin)`` validates a record by the rule it
    names (what the verifier does for records written under v1).  ``None`` (absent pin) is a refusal: ``V3.interior_layout`` fails closed."""
    if chunk_words is None:
        chunk_words = TABLE_ROWS_PER_LEAF if _check_rule(rule) == TABLE_RULE else CHUNK_WORDS
    want = layout_pin(chunk_words, rule=rule)
    if pin is None:
        return [f"{INTERIOR_LAYOUT_DIGEST}: protocol_instance.json#commitments.interior_layout is absent (expected {json.dumps(want, sort_keys=True)})"]
    p: list[str] = []
    for k in ("framing", "word_order", "tree", "digest"):
        if pin.get(k) != want[k]:
            p.append(f"{INTERIOR_LAYOUT_DIGEST}: interior_layout.{k} = {pin.get(k)!r}, rule {want['rule']} has {want[k]!r}")
    named = pin.get("rule") or SCHEMA_V1
    if named != want["rule"]:
        p.append(f"{INTERIOR_LAYOUT_DIGEST}: interior_layout.rule = {named!r} (absent = v1), expected {want['rule']!r}")
    return p


# --- the layout ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallSpan:
    """One root call of the pinned program: its structural name and gate interval ``[lo, hi)``."""

    call: str
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not (0 <= self.lo <= self.hi):
            raise ValueError(f"call {self.call!r}: bad interval [{self.lo}, {self.hi})")
        if "/chunk=" in self.call or not self.call:
            raise ValueError(f"call name {self.call!r} may not be empty or contain '/chunk='")


@dataclass(frozen=True, slots=True)
class LeafSpec:
    """One chunk of the layout: its rank, canonical id, call, chunk index, width, and the gates it holds (sorted, int64)."""

    rank: int
    canonical_id: str
    call: str
    chunk: int
    width: int
    gates: np.ndarray = field(repr=False, compare=False)
    origin: str = ORIGIN

    @property
    def words(self) -> int:
        return int(self.gates.shape[0])

    @property
    def dtype(self) -> str:
        return dtype_for_width(self.width)

    @property
    def word_nbytes(self) -> int:
        return word_bytes(self.width)

    @property
    def nbytes(self) -> int:
        return self.words * self.word_nbytes

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.words,)

    @property
    def g_first(self) -> int:
        return int(self.gates[0])

    @property
    def g_last(self) -> int:
        return int(self.gates[-1])

    def flat_of(self, g: int) -> int | None:
        """The word index of gate ``g`` in this chunk, or None."""
        k = int(np.searchsorted(self.gates, g))
        return k if k < self.words and int(self.gates[k]) == int(g) else None

    def layout_type(self):
        """The interface's ``Leaf.layout`` for this chunk: ``Array(n, Bits(w))`` (``src/veritor/interface/typed_obligation.py``)."""
        from veritor.interface.typed_obligation import Array, Bits

        return Array(Bits(self.width), self.words)

    def region(self) -> dict[str, Any]:
        """The index.json / required-regions row (no gate list; that is ``gates.bin``)."""
        return {"rank": self.rank, "canonical_id": self.canonical_id, "origin": self.origin, "call": self.call, "chunk": self.chunk, "width": self.width,
                "dtype": self.dtype, "shape": [self.words], "words": self.words, "nbytes": self.nbytes, "g_first": self.g_first, "g_last": self.g_last}


class InteriorLayout:
    """The ordered leaves of one RU's interior tree, from ``(calls, required gates, widths)`` -- deterministic, program-agnostic."""

    def __init__(self, tree: str, leaves: Sequence[LeafSpec], calls: Sequence[CallSpan], *, prefix: str = "", chunk_words: int = CHUNK_WORDS,
                 rule: str | None = None) -> None:
        self.tree = tree
        self.leaves = list(leaves)
        self.calls = list(calls)
        self.prefix = prefix
        self.chunk_words = int(chunk_words)
        self.rule = _check_rule(rule)
        self.ids = [lf.canonical_id for lf in self.leaves]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate leaf id in the layout (two calls with the same name?)")
        self._rank = {cid: i for i, cid in enumerate(self.ids)}
        # flat arrays for O(log n) locate: all gates in LAYOUT order (gates.bin), each leaf's first word position, and the sorted view
        # (v2 groups by width, so layout order is not gate order; v1's layout order is gate order and the sorted view is the identity)
        self._all_gates = np.concatenate([lf.gates for lf in self.leaves]) if self.leaves else np.zeros(0, dtype=np.int64)
        self._leaf_start = np.cumsum([0] + [lf.words for lf in self.leaves])[:-1].astype(np.int64) if self.leaves else np.zeros(0, dtype=np.int64)
        order = np.argsort(self._all_gates, kind="stable")
        self._sorted_gates = self._all_gates[order]
        self._sorted_pos = order.astype(np.int64)
        if self._sorted_gates.shape[0] and np.any(np.diff(self._sorted_gates) == 0):
            raise ValueError("a gate appears in two leaves of the layout")

    # -- construction ------------------------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, tree: str, calls: Sequence[CallSpan], gates: Sequence[int] | np.ndarray, widths: Sequence[int] | np.ndarray, *,
              prefix: str = "", chunk_words: int = CHUNK_WORDS, rule: str | None = None) -> InteriorLayout:
        """The layout of the required gates ``gates`` (sorted, unique) with output widths ``widths`` (bits, one per gate) over the
        RU's root calls ``calls`` (disjoint, increasing), under ``rule`` (default: the current rule, v2).  A gate outside every call is a
        caller error (``ValueError``): the required set of an RU lies in its calls by construction."""
        r = _check_rule(rule)
        if r == TABLE_RULE:
            raise ValueError(f"{TABLE_RULE} layouts are built by e2e.interior.tables.table_layout(name), not from gates")
        if int(chunk_words) < 1:
            raise ValueError("chunk_words must be >= 1")
        g = np.asarray(gates, dtype=np.int64).reshape(-1)
        w = np.asarray(widths, dtype=np.int64).reshape(-1)
        if g.shape != w.shape:
            raise ValueError(f"{g.shape[0]} gates but {w.shape[0]} widths")
        if g.shape[0] and (np.any(np.diff(g) <= 0)):
            raise ValueError("required gates must be strictly increasing (sorted, unique)")
        if np.any(w <= 0):
            raise ValueError("every width must be positive")
        spans = list(calls)
        for a, b in zip(spans, spans[1:], strict=False):
            if b.lo < a.hi:
                raise ValueError(f"calls {a.call!r} [{a.lo},{a.hi}) and {b.call!r} [{b.lo},{b.hi}) are not disjoint and increasing")
        names = [c.call for c in spans]
        if len(set(names)) != len(names):
            raise ValueError("two calls share a name")
        leaves: list[LeafSpec] = []
        covered = 0
        for span in spans:
            lo = int(np.searchsorted(g, span.lo, side="left"))
            hi = int(np.searchsorted(g, span.hi, side="left"))
            if hi <= lo:
                continue
            cg, cw = g[lo:hi], w[lo:hi]
            covered += hi - lo
            if r == SCHEMA_V1:
                # v1: chunk boundaries -- every width change starts a run; a run is cut every chunk_words words
                run_edges = np.concatenate(([0], np.flatnonzero(np.diff(cw) != 0) + 1, [hi - lo]))
                starts: list[int] = []
                for a, b in zip(run_edges[:-1], run_edges[1:], strict=False):
                    starts.extend(range(int(a), int(b), chunk_words))
                starts.append(hi - lo)
                for k, (a, b) in enumerate(zip(starts, starts[1:], strict=False)):
                    leaves.append(LeafSpec(len(leaves), leaf_id(prefix, span.call, k, rule=r), span.call, k, int(cw[a]), np.ascontiguousarray(cg[a:b])))
            else:
                # v2: width groups in increasing width; a group's gates (gate order) are cut every chunk_words words; k per (call, width)
                for width in np.unique(cw):
                    gw = np.ascontiguousarray(cg[cw == width])
                    for k, a in enumerate(range(0, gw.shape[0], chunk_words)):
                        leaves.append(LeafSpec(len(leaves), leaf_id(prefix, span.call, k, int(width), rule=r), span.call, k, int(width),
                                               np.ascontiguousarray(gw[a:a + chunk_words])))
        if covered != g.shape[0]:
            outside = _gates_outside(g, spans)
            raise ValueError(f"{g.shape[0] - covered} required gate(s) lie in no call of the RU (first: {outside[:5]})")
        return cls(tree, leaves, spans, prefix=prefix, chunk_words=chunk_words, rule=r)

    @classmethod
    def from_rows(cls, tree: str, calls: Sequence[CallSpan], rows: Iterable[tuple[int, int]], *, prefix: str = "", chunk_words: int = CHUNK_WORDS,
                  rule: str | None = None) -> InteriorLayout:
        """PROVER convenience: the layout over the gates the rows carry (``(g, w)`` pairs, any order) -- what the prover commits when
        no inventory is given.  The verifier never uses this: it derives the layout from its own required set and compares
        (:func:`validate_supplied`), so a missing region is detected there, not here."""
        pairs = np.asarray(sorted(set((int(g), int(w)) for g, w in rows)), dtype=np.int64).reshape(-1, 2)
        if pairs.shape[0] and np.any(np.diff(pairs[:, 0]) == 0):
            dup = pairs[1:, 0][np.diff(pairs[:, 0]) == 0]
            raise ValueError(f"{INTERIOR_WIDTH_MISMATCH}: gate(s) {dup[:5].tolist()} carry two widths")
        return cls.build(tree, calls, pairs[:, 0], pairs[:, 1], prefix=prefix, chunk_words=chunk_words, rule=rule)

    # -- queries -----------------------------------------------------------------------------------------------------------------
    @property
    def leaf_count(self) -> int:
        return len(self.leaves)

    @property
    def words(self) -> int:
        return int(self._all_gates.shape[0])

    @property
    def nbytes(self) -> int:
        return sum(lf.nbytes for lf in self.leaves)

    @property
    def gates(self) -> np.ndarray:
        """Every gate of the layout, in LAYOUT order (what ``gates.bin`` holds; gate order within a leaf, v1: gate order throughout)."""
        return self._all_gates

    @property
    def sorted_gates(self) -> np.ndarray:
        """Every gate of the layout in increasing gate index (the required set)."""
        return self._sorted_gates

    def rank_of(self, canonical_id: str) -> int | None:
        return self._rank.get(canonical_id)

    def leaf(self, rank: int) -> LeafSpec:
        return self.leaves[rank]

    @property
    def origin(self) -> str:
        return self.leaves[0].origin if self.leaves else ORIGIN

    def leaves_of(self, call: str) -> list[LeafSpec]:
        """The leaves of root call ``call`` in rank order (contiguous: call order, then width, then chunk)."""
        if not hasattr(self, "_by_call"):
            by_call: dict[str, list[LeafSpec]] = {}
            for lf in self.leaves:
                by_call.setdefault(lf.call, []).append(lf)
            self._by_call = by_call
        return self._by_call.get(call, [])

    def position(self, g: int) -> tuple[int, int] | None:
        """``(rank, flat)`` of gate ``g``, or None when the layout has no position for it."""
        if self.words == 0:
            return None
        k = int(np.searchsorted(self._sorted_gates, int(g)))
        if k >= self.words or int(self._sorted_gates[k]) != int(g):
            return None
        pos = int(self._sorted_pos[k])
        rank = int(np.searchsorted(self._leaf_start, pos, side="right")) - 1
        return rank, pos - int(self._leaf_start[rank])

    def locate(self, g: int) -> tuple[str, str, int] | None:
        """E3's resolution: ``(tree, leaf canonical id, flat)`` of gate ``g``, or None."""
        pos = self.position(g)
        if pos is None:
            return None
        rank, flat = pos
        return self.tree, self.leaves[rank].canonical_id, flat

    def positions(self, gates: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised :meth:`position`: ``(found mask, rank, flat)`` arrays for ``gates`` (rank/flat are -1 where not found)."""
        g = np.asarray(gates, dtype=np.int64).reshape(-1)
        rank = np.full(g.shape, -1, dtype=np.int64)
        flat = np.full(g.shape, -1, dtype=np.int64)
        if self.words == 0 or g.shape[0] == 0:
            return np.zeros(g.shape, dtype=bool), rank, flat
        k = np.searchsorted(self._sorted_gates, g)
        kk = np.minimum(k, self.words - 1)
        found = self._sorted_gates[kk] == g
        pos = self._sorted_pos[kk]
        r = np.searchsorted(self._leaf_start, pos, side="right") - 1
        rank[found] = r[found]
        flat[found] = pos[found] - self._leaf_start[r[found]]
        return found, rank, flat

    def required_regions(self) -> list[dict[str, Any]]:
        """The regions (leaves) the verifier requires of this RU, in rank order -- what ``validate_supplied`` compares the index against."""
        return [lf.region() for lf in self.leaves]

    def gates_bin(self) -> bytes:
        """``gates.bin``: every gate of the layout in layout order, int64 little-endian."""
        return np.ascontiguousarray(self._all_gates, dtype="<i8").tobytes()

    def summary(self) -> dict[str, Any]:
        return {"schema": self.rule, "tree": self.tree, "layout_digest": layout_digest(chunk_words=self.chunk_words, rule=self.rule), "chunk_words": self.chunk_words, "prefix": self.prefix,
                "calls": [{"call": c.call, "lo": c.lo, "hi": c.hi} for c in self.calls], "leaf_count": self.leaf_count, "words": self.words, "nbytes": self.nbytes}


def chunks(call: CallSpan, gates: Sequence[int] | np.ndarray, widths: Sequence[int] | np.ndarray, *, prefix: str = "", chunk_words: int = CHUNK_WORDS,
           tree: str = "", rule: str | None = None) -> list[LeafSpec]:
    """``chunks(call)`` (INTEGRATOR D2): the ordered leaves of ONE call over its required gates ``gates`` (sorted, unique, all in
    ``[call.lo, call.hi)``) with widths ``widths``.  Chunk ``k`` -> ``LeafSpec`` with ``canonical_id = leaf_id(prefix, call.call, k)``; the
    ``rank`` field is the rank WITHIN this call (the RU rank = the offset of the call's first leaf in the RU layout + k).  Same rule as
    :meth:`InteriorLayout.build`, which is what the RU-level layout uses."""
    lay = InteriorLayout.build(tree or f"interior/{call.call}", [call], gates, widths, prefix=prefix, chunk_words=chunk_words, rule=rule)
    return list(lay.leaves)


def _gates_outside(g: np.ndarray, spans: Sequence[CallSpan]) -> list[int]:
    inside = np.zeros(g.shape, dtype=bool)
    for s in spans:
        inside |= (g >= s.lo) & (g < s.hi)
    return g[~inside].tolist()


def required_regions(layouts: Mapping[str, InteriorLayout]) -> dict[str, list[dict[str, Any]]]:
    """``required_regions(J)``: for every retained RU key (``"<req>/<step>"``) the ordered regions its interior tree must hold."""
    return {ru: lay.required_regions() for ru, lay in layouts.items()}


# --- the per-call (lazy) layout: the verifier's required layout without O(words) memory ---------------------------------------------------


class LazyInteriorLayout:
    """[NIGHT 2026-09-12] The same layout as :class:`InteriorLayout` (rule clause 4), materialised ONE CALL AT A TIME from
    ``call_gw(span) -> (gates, widths)`` (E3's ``call_layout``: the call's required gates ascending + their widths).  Holds per call only its
    leaf / word counts (ints) and a small cache of built calls -- **memory ~ the largest call, not the RU** (an ``InteriorLayout`` of the
    128x64 prefill RU is 3.6 GB while building, 33 GB at 512x128; this is ~70 MB at 6500x1600).  Same ids, ranks, positions:
    ``leaves_of(call)`` == the eager layout's leaves of that call (asserted in ``tests/contract/e2_interior_test.py``).

    What the verifier's step-4 check (:func:`validate_supplied` / :func:`validate_step4`) needs: ``tree, calls, rule, chunk_words, prefix,
    origin, leaves_of(call), rank_of(id), locate(g), position(g), positions(gates), leaf_count, words, required_regions()`` (the last
    iterates -- O(leaves) output, not O(words) memory).  ``call_gw`` is called once per call for the counts (first pass, results discarded)
    and again per materialisation (cached, LRU of ``cache`` calls).

    **The required set, publicly (D40):** ``gates_by_call()`` (streaming, per call), ``sorted_gates`` (ascending array), ``gates`` / ``gates_bin``
    (layout order, == the eager class), ``materialize()`` (the eager layout).  The eager class's ``_all_gates`` is private and does not exist here."""

    def __init__(self, tree: str, calls: Sequence[CallSpan], call_gw: Callable[[CallSpan], tuple[Any, Any]], *, prefix: str = "",
                 chunk_words: int = CHUNK_WORDS, rule: str | None = None, cache: int = 8) -> None:
        self.tree = tree
        self.calls = list(calls)
        self.prefix = prefix
        self.chunk_words = int(chunk_words)
        self.rule = _check_rule(rule)
        if self.rule == TABLE_RULE:
            raise ValueError(f"{TABLE_RULE} layouts are built by e2e.interior.tables.table_layout(name)")
        for a, b in zip(self.calls, self.calls[1:], strict=False):
            if b.lo < a.hi:
                raise ValueError(f"calls {a.call!r} [{a.lo},{a.hi}) and {b.call!r} [{b.lo},{b.hi}) are not disjoint and increasing")
        names = [c.call for c in self.calls]
        if len(set(names)) != len(names):
            raise ValueError("two calls share a name")
        self._call_gw = call_gw
        self._index = {c.call: i for i, c in enumerate(self.calls)}
        self._lo = np.asarray([c.lo for c in self.calls], dtype=np.int64)
        self._cache_n = max(1, int(cache))
        self._cache: dict[int, list[LeafSpec]] = {}
        self._counts: list[int] | None = None  # leaves per call
        self._words: list[int] | None = None
        self._offsets: np.ndarray | None = None

    # -- counts (one pass over the calls, ints only) ------------------------------------------------------------------------------------
    def _ensure_counts(self) -> None:
        if self._counts is not None:
            return
        counts, words = [], []
        for span in self.calls:
            g, w = self._gw(span)
            words.append(int(g.shape[0]))
            if self.rule == SCHEMA_V1:
                counts.append(len(chunks(span, g, w, prefix=self.prefix, chunk_words=self.chunk_words, rule=self.rule)) if g.shape[0] else 0)
            else:
                _u, n = np.unique(w, return_counts=True) if w.shape[0] else (None, np.zeros(0, dtype=np.int64))
                counts.append(int(np.sum(-(-n // self.chunk_words))))
        self._counts, self._words = counts, words
        self._offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    def _gw(self, span: CallSpan) -> tuple[np.ndarray, np.ndarray]:
        g, w = self._call_gw(span)
        g = np.asarray(g, dtype=np.int64).reshape(-1)
        w = np.asarray(w, dtype=np.int64).reshape(-1)
        if g.shape != w.shape:
            raise ValueError(f"call {span.call!r}: {g.shape[0]} gates but {w.shape[0]} widths")
        if g.shape[0] and np.any(np.diff(g) <= 0):
            raise ValueError(f"call {span.call!r}: required gates must be strictly increasing (sorted, unique)")
        if g.shape[0] and (int(g[0]) < span.lo or int(g[-1]) >= span.hi):
            raise ValueError(f"call {span.call!r}: required gate(s) outside [{span.lo}, {span.hi})")
        return g, w

    def _leaves_i(self, i: int) -> list[LeafSpec]:
        hit = self._cache.get(i)
        if hit is not None:
            return hit
        self._ensure_counts()
        span = self.calls[i]
        g, w = self._gw(span)
        base = int(self._offsets[i])  # type: ignore[index]
        leaves = [LeafSpec(base + lf.rank, lf.canonical_id, lf.call, lf.chunk, lf.width, lf.gates, origin=lf.origin)
                  for lf in chunks(span, g, w, prefix=self.prefix, chunk_words=self.chunk_words, tree=self.tree, rule=self.rule)] if g.shape[0] else []
        if len(self._cache) >= self._cache_n:
            self._cache.pop(next(iter(self._cache)))
        self._cache[i] = leaves
        return leaves

    # -- queries ------------------------------------------------------------------------------------------------------------------------
    @property
    def origin(self) -> str:
        return ORIGIN

    @property
    def leaf_count(self) -> int:
        self._ensure_counts()
        return int(self._offsets[-1])  # type: ignore[index]

    @property
    def words(self) -> int:
        self._ensure_counts()
        return int(sum(self._words or []))

    @property
    def nbytes(self) -> int:
        return sum(lf.nbytes for span in self.calls for lf in self.leaves_of(span.call))

    def leaves_of(self, call: str) -> list[LeafSpec]:
        i = self._index.get(call)
        return self._leaves_i(i) if i is not None else []

    def iter_leaves(self) -> Iterable[LeafSpec]:
        for i in range(len(self.calls)):
            yield from self._leaves_i(i)

    @property
    def ids(self) -> list[str]:
        return [lf.canonical_id for lf in self.iter_leaves()]

    @property
    def leaves(self) -> list[LeafSpec]:
        """The eager layout's ``leaves`` -- MATERIALISES every call (O(words) memory, as the eager class); tests and tools only. The verifier
        uses ``leaves_of(call)`` / ``iter_leaves()``."""
        return list(self.iter_leaves())

    def materialize(self) -> InteriorLayout:
        """The equal eager :class:`InteriorLayout` (same ids / ranks / positions) -- what the PROVER's tree builder needs (it holds every word
        anyway); the verifier never calls this."""
        gs, ws = zip(*(self._gw(span) for span in self.calls), strict=True) if self.calls else ((), ())
        return InteriorLayout.build(self.tree, self.calls, np.concatenate(gs) if gs else np.zeros(0, dtype=np.int64),
                                    np.concatenate(ws) if ws else np.zeros(0, dtype=np.int64), prefix=self.prefix, chunk_words=self.chunk_words,
                                    rule=self.rule)

    # -- the REQUIRED SET, publicly (D40 / T-3: `_all_gates` is PRIVATE to the eager class and absent here; read these) ------------------
    def gates_by_call(self) -> Iterable[tuple[CallSpan, np.ndarray, np.ndarray]]:
        """STREAMING required set: ``(call, gates ascending, widths)`` per call in call order -- O(largest call) memory; what a prover-side
        required-set writer should iterate (E10's ``interior_required_v2``)."""
        for span in self.calls:
            yield (span, *self._gw(span))

    @property
    def gates(self) -> np.ndarray:
        """Every gate of the layout in LAYOUT order (= the eager ``InteriorLayout.gates`` / ``gates.bin`` order: leaf by leaf, gate order within a
        leaf) -- MATERIALISES (O(words)); the same array the eager class holds as ``_all_gates``."""
        parts = [lf.gates for lf in self.iter_leaves()]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    @property
    def sorted_gates(self) -> np.ndarray:
        """Every gate of the layout in increasing gate index (the required set as a sorted array) -- MATERIALISES (O(words))."""
        parts = [self._gw(span)[0] for span in self.calls]  # per call ascending; calls are disjoint and increasing -> concatenation is sorted
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def gates_bin(self) -> bytes:
        """``gates.bin``: every gate in layout order, int64 little-endian (== the eager ``gates_bin()``)."""
        return np.ascontiguousarray(self.gates, dtype="<i8").tobytes()

    def leaf(self, rank: int) -> LeafSpec:
        self._ensure_counts()
        assert self._offsets is not None
        i = int(np.searchsorted(self._offsets, rank, side="right")) - 1
        if i < 0 or i >= len(self.calls) or rank >= int(self._offsets[-1]):
            raise IndexError(rank)
        return self._leaves_i(i)[rank - int(self._offsets[i])]

    def required_regions(self) -> list[dict[str, Any]]:
        return [lf.region() for lf in self.iter_leaves()]

    def _call_of_gate(self, g: int) -> int | None:
        i = int(np.searchsorted(self._lo, g, side="right")) - 1
        return i if 0 <= i < len(self.calls) and self.calls[i].lo <= g < self.calls[i].hi else None

    def rank_of(self, canonical_id: str) -> int | None:
        # <prefix>call=<c>/w=<w>/chunk=<k> (v2) or <prefix>call=<c>/chunk=<k> (v1): the call is what follows 'call=' up to '/w=' or '/chunk='
        if not canonical_id.startswith(self.prefix):
            return None
        body = canonical_id[len(self.prefix):]
        if not body.startswith("call="):
            return None
        body = body[len("call="):]
        cut = body.rfind("/w=") if self.rule != SCHEMA_V1 else -1
        if cut < 0:
            cut = body.rfind("/chunk=")
        if cut < 0:
            return None
        for lf in self.leaves_of(body[:cut]):
            if lf.canonical_id == canonical_id:
                return lf.rank
        return None

    def position(self, g: int) -> tuple[int, int] | None:
        i = self._call_of_gate(int(g))
        if i is None:
            return None
        for lf in self._leaves_i(i):
            k = lf.flat_of(int(g))
            if k is not None:
                return lf.rank, k
        return None

    def locate(self, g: int) -> tuple[str, str, int] | None:
        pos = self.position(g)
        if pos is None:
            return None
        rank, flat = pos
        i = self._call_of_gate(int(g))
        lf = next(x for x in self._leaves_i(i) if x.rank == rank)  # type: ignore[arg-type]
        return self.tree, lf.canonical_id, flat

    def positions(self, gates: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = np.asarray(gates, dtype=np.int64).reshape(-1)
        found = np.zeros(g.shape, dtype=bool)
        rank = np.full(g.shape, -1, dtype=np.int64)
        flat = np.full(g.shape, -1, dtype=np.int64)
        if not g.shape[0] or not self.calls:
            return found, rank, flat
        ci = np.searchsorted(self._lo, g, side="right") - 1
        for i in np.unique(ci):
            if i < 0 or i >= len(self.calls):
                continue
            sel = np.flatnonzero(ci == i)
            for lf in self._leaves_i(int(i)):
                k = np.searchsorted(lf.gates, g[sel])
                kk = np.minimum(k, lf.words - 1)
                hit = lf.gates[kk] == g[sel]
                found[sel[hit]] = True
                rank[sel[hit]] = lf.rank
                flat[sel[hit]] = kk[hit]
        return found, rank, flat

    def summary(self) -> dict[str, Any]:
        return {"schema": self.rule, "tree": self.tree, "layout_digest": layout_digest(chunk_words=self.chunk_words, rule=self.rule), "chunk_words": self.chunk_words,
                "prefix": self.prefix, "calls": [{"call": c.call, "lo": c.lo, "hi": c.hi} for c in self.calls], "leaf_count": self.leaf_count, "words": self.words,
                "lazy": True}


# --- the verifier's check of a supplied index ----------------------------------------------------------------------------------------


def validate_supplied(index: Mapping[str, Any] | str | os.PathLike[str], required: Any, pin: Mapping[str, Any] | None = None,
                      commitments: Mapping[str, Any] | None = None, *, gates_bin: bytes | None = None, is_boundary: Callable[[int], bool] | None = None,
                      check_digest: bool = True, gates_reader: tuple[Callable[[int, int], np.ndarray], int] | None = None,
                      **step4_kw: Any) -> list[str] | dict[str, Any]:
    """Two call shapes:

    * ``validate_supplied(index_doc, required_layout, *, gates_bin=, is_boundary=) -> problems[]`` -- ONE tree's ``index.json`` against the
      verifier's derived :class:`InteriorLayout` (below);
    * ``validate_supplied(selective_dir, retained_request_ids, pin, commitments, *, required=, is_boundary=, rehash=) -> {"problems": [...], ...}``
      -- E9's step-4 form over the whole selective bundle (:func:`validate_step4`): every retained RU's interior tree present, its index
      consistent with ``commitments.json#interior_roots`` and with this rule, and -- with ``required`` (E3's derived layouts) -- every
      required region / gate supplied.

    Problems of the prover's ``index.json`` (``e2e.interior.tree.write_tree``) against the verifier's derived layout ``required``
    (empty = every required region is supplied at its position and nothing else is).  Called by E9 at step 4 BEFORE ``r_s``; a
    non-empty result is ``STEP4_REFUSED``.

    * ``INTERIOR_REGION_MISSING`` -- a leaf of the derived layout is absent, or is present with other words / gate range;
    * ``INTERIOR_UNEXPECTED_LEAF`` -- the index names a leaf the layout does not;
    * ``INTERIOR_ORDER`` -- a leaf sits at another rank than the layout gives it (rank = layout order), or ranks are not 0..n-1;
    * ``INTERIOR_WIDTH_MISMATCH`` -- a leaf's width / dtype / nbytes differ from the layout's;
    * with ``gates_bin`` (the tree's ``gates.bin``) or ``gates_reader=(read(offset_words, n), total_words)`` (the file read one call at a
      time; what :func:`validate_step4` passes): ``INTERIOR_DUPLICATE_GATE`` -- a gate committed twice, or (with ``is_boundary``) a
      gate that has a boundary position; ``INTERIOR_UNEXPECTED_GATE`` -- a gate the layout does not name; ``INTERIOR_REGION_MISSING``
      -- a required gate absent from its leaf (named by gate);
    * ``INTERIOR_LAYOUT_DIGEST`` -- the index was written under another rule (``layout.digest``), ``INTERIOR_INDEX_MALFORMED`` otherwise.

    Nothing here hashes or opens: the root / ids_digest of the tree are checked against ``commitments.json`` by
    :func:`e2e.interior.tree.check_index_against_commitments`."""
    if isinstance(index, (str, os.PathLike)):
        return validate_step4(index, list(required), pin, commitments or {}, is_boundary=is_boundary, **step4_kw)
    if not isinstance(required, (InteriorLayout, LazyInteriorLayout)):
        raise TypeError(f"validate_supplied(index_doc, required) needs an InteriorLayout / LazyInteriorLayout, got {type(required).__name__}")
    p: list[str] = []
    try:
        if index.get("schema") != INDEX_SCHEMA:
            p.append(f"{INTERIOR_INDEX_MALFORMED}: index schema {index.get('schema')!r}, expected {INDEX_SCHEMA!r}")
        if index.get("tree") != required.tree:
            p.append(f"{INTERIOR_INDEX_MALFORMED}: index is for tree {index.get('tree')!r}, the layout for {required.tree!r}")
        lay = index.get("layout") or {}
        want_digest = layout_digest(chunk_words=required.chunk_words, rule=required.rule)
        if check_digest and (lay.get("digest") != want_digest or rule_of(lay) != required.rule):
            p.append(f"{INTERIOR_LAYOUT_DIGEST}: index layout {rule_of(lay)} digest {str(lay.get('digest'))[:16]} != the required layout's {required.rule} {want_digest[:16]}")
        rows = list(index.get("leaves") or [])
        if int(index.get("leaf_count", -1)) != len(rows):
            p.append(f"{INTERIOR_INDEX_MALFORMED}: leaf_count {index.get('leaf_count')} but {len(rows)} leaves listed")
        by_id: dict[str, dict[str, Any]] = {}
        for i, row in enumerate(rows):
            cid = str(row.get("canonical_id"))
            if cid in by_id:
                p.append(f"{INTERIOR_INDEX_MALFORMED}: leaf {cid!r} listed twice")
                continue
            by_id[cid] = row
            if int(row.get("rank", -1)) != i:
                p.append(f"{INTERIOR_ORDER}: leaf {cid!r} listed at position {i} claims rank {row.get('rank')}")
            want_origin = required.origin
            if row.get("origin") != want_origin:
                p.append(f"{INTERIOR_INDEX_MALFORMED}: leaf {cid!r} carries origin {row.get('origin')!r}, these leaves carry {want_origin!r}")
        # every required region present, at its rank, with its width / words / gate range -- one call's leaves at a time
        for lf in (lf for span in required.calls for lf in required.leaves_of(span.call)):
            row = by_id.get(lf.canonical_id)
            if row is None:
                p.append(f"{INTERIOR_REGION_MISSING}: {required.tree} leaf {lf.canonical_id!r} (rank {lf.rank}, gates {lf.g_first}..{lf.g_last}, {lf.words} x {lf.dtype}) is not supplied")
                continue
            if int(row.get("rank", -1)) != lf.rank:
                p.append(f"{INTERIOR_ORDER}: leaf {lf.canonical_id!r} at rank {row.get('rank')}, the layout gives rank {lf.rank}")
            if int(row.get("width", -1)) != lf.width or str(row.get("dtype")) != lf.dtype or int(row.get("nbytes", -1)) != lf.nbytes:
                p.append(f"{INTERIOR_WIDTH_MISMATCH}: leaf {lf.canonical_id!r} is {row.get('words')} x {row.get('dtype')} (w={row.get('width')}, {row.get('nbytes')} B), "
                         f"the layout says {lf.words} x {lf.dtype} (w={lf.width}, {lf.nbytes} B)")
            if int(row.get("words", -1)) != lf.words or int(row.get("g_first", -1)) != lf.g_first or int(row.get("g_last", -1)) != lf.g_last:
                p.append(f"{INTERIOR_REGION_MISSING}: leaf {lf.canonical_id!r} supplied with {row.get('words')} words over gates {row.get('g_first')}..{row.get('g_last')}, "
                         f"required {lf.words} words over {lf.g_first}..{lf.g_last}")
        for cid in by_id:
            if required.rank_of(cid) is None:
                p.append(f"{INTERIOR_UNEXPECTED_LEAF}: leaf {cid!r} is not a region of the derived layout")
        # gate level (gates.bin): duplicates, boundary gates, unexpected gates, missing gates by name -- one call at a time
        if gates_bin is not None or gates_reader is not None:
            if gates_reader is not None:
                read, n_got = gates_reader
            else:
                got = np.frombuffer(gates_bin, dtype="<i8")
                n_got = int(got.shape[0])

                def read(a: int, n: int, _got: np.ndarray = got) -> np.ndarray:
                    return _got[a:a + n]

            total = sum(int(r.get("words", 0)) for r in rows)
            if n_got != total:
                p.append(f"{INTERIOR_INDEX_MALFORMED}: gates.bin holds {n_got} gates, the index {total} words")
            else:
                p.extend(_check_gates_by_call(rows, required.calls, read, tree=required.tree, required=required, is_boundary=is_boundary,
                                              rule=required.rule, chunk_words=required.chunk_words, prefix=required.prefix, rebuild=False))
    except (KeyError, TypeError, ValueError) as e:
        p.append(f"{INTERIOR_INDEX_MALFORMED}: {type(e).__name__}: {str(e)[:160]}")
    return p


# --- the gate-level check, streamed per call ------------------------------------------------------------------------------------------------
#
# [NIGHT 2026-09-12, E2 for E9] The whole-tree form (np.unique / setdiff1d over gates.bin, a claimed InteriorLayout of the whole tree and a
# second one rebuilt under the rule) held ~14 int64 copies of every committed word: 112 B/word -> 6.3 GB for the 128x64 prefill RU (55.9 M
# words), 58 GB at 512x128 -- laptop-side, BEFORE r_s.  The rule is per call (clause 4), so the check is too: a call's leaves are contiguous
# in rank order (call order, then width, then chunk), its gates are read as one slice of gates.bin, sorted, checked for duplicates and
# containment, rebuilt under the rule and compared leaf-by-leaf with the supplied rows and (with ``required``) with the derived layout's
# leaves of that call.  Peak memory ~13 x 8 B x the largest call's words (650 K words at 6500x1600 -> ~70 MB), whatever the context length;
# gates.bin is read call by call (``read(offset_words, n)``), never whole.  Global uniqueness follows: the calls are disjoint intervals
# and every gate of a call is checked to lie in the call's [lo, hi).

_NAME_LIMIT = 16


def _check_gates_by_call(rows: Sequence[Mapping[str, Any]], calls: Sequence[CallSpan], read: Callable[[int, int], np.ndarray], *, tree: str,
                         required: InteriorLayout | LazyInteriorLayout | None, is_boundary: Callable[[int], bool] | None, rule: str | None, chunk_words: int, prefix: str,
                         rebuild: bool = True, ru_label: str | None = None) -> list[str]:
    """Problems of the index ``rows`` (rank order) + the gate slices ``read(offset, n)`` (layout order) over the tree's ``calls``:

    * ``rebuild`` -- the supplied leaves of every call are exactly this rule's over the call's own gates / widths (``INTERIOR_ORDER``);
    * duplicates within a call, gates outside the call's interval (``INTERIOR_DUPLICATE_GATE`` / ``INTERIOR_UNEXPECTED_GATE``, boundary
      gates named as such with ``is_boundary``);
    * with ``required`` -- per call: the derived layout's leaves (ids, widths, gate lists) equal the supplied ones, else the missing /
      unexpected gates and the first order difference are named (bounded).

    Memory is proportional to the largest call, not the tree."""
    p: list[str] = []
    where = ru_label or tree
    rule_name = _check_rule(rule)
    # leaf rows grouped by call, in rank order; a call's leaves must be contiguous and the calls in call order
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for i, row in enumerate(rows):
        c = str(row.get("call"))
        if c not in groups:
            groups[c] = []
            order.append(c)
        groups[c].append(i)
    starts = np.cumsum([0] + [int(r.get("words", 0)) for r in rows])
    call_names = [c.call for c in calls]
    call_pos = {c: i for i, c in enumerate(call_names)}
    unknown = [c for c in order if c not in call_pos]
    for c in unknown[:_NAME_LIMIT]:
        p.append(f"{INTERIOR_UNEXPECTED_LEAF}: {where}: leaves of call {c!r}, which is not a call of the tree")
    seq = [call_pos[c] for c in order if c in call_pos]
    if any(b <= a for a, b in zip(seq, seq[1:], strict=False)):
        p.append(f"{INTERIOR_ORDER}: {where}: the leaves are not in call order (rule clause 2/4)")
    for c, idxs in groups.items():
        if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            p.append(f"{INTERIOR_ORDER}: {where}: the leaves of call {c!r} are not contiguous in rank order")
    named_dup = named_extra = named_missing = 0
    for span in calls:
        idxs = groups.get(span.call)
        req_leaves = required.leaves_of(span.call) if required is not None else []
        if not idxs:
            if req_leaves:
                p.append(f"{INTERIOR_REGION_MISSING}: {where}: call {span.call!r} supplies no leaf, the layout requires {len(req_leaves)} "
                         f"({sum(lf.words for lf in req_leaves)} words)")
            continue
        a, b = idxs[0], idxs[-1] + 1
        n = int(starts[b] - starts[a])
        got = np.asarray(read(int(starts[a]), n), dtype=np.int64)
        if got.shape[0] != n:
            raise ValueError(f"gates.bin ends inside call {span.call!r}")
        widths = np.repeat(np.asarray([int(rows[i].get("width", 0)) for i in range(a, b)], dtype=np.int64),
                           np.asarray([int(rows[i].get("words", 0)) for i in range(a, b)], dtype=np.int64))
        ordr = np.argsort(got, kind="stable")
        gs, ws = got[ordr], widths[ordr]
        # duplicates within the call
        if gs.shape[0] > 1:
            dup_mask = np.diff(gs) == 0
            if np.any(dup_mask):
                dups, counts = np.unique(gs[1:][dup_mask], return_counts=True)
                for g, k in zip(dups[: max(0, 8 - named_dup)], counts[: max(0, 8 - named_dup)], strict=True):
                    kk = int(k) + 1
                    p.append(f"{INTERIOR_DUPLICATE_GATE}: gate {int(g)} committed {'twice' if kk == 2 else f'{kk} times'} in {where}")
                    named_dup += 1
                keep = np.concatenate(([True], ~dup_mask))
                gs, ws = gs[keep], ws[keep]
        # gates outside the call's interval / (with required) not required: boundary or unexpected, by name
        inside = (gs >= span.lo) & (gs < span.hi)
        missing = np.zeros(0, dtype=np.int64)
        if required is not None:
            rg = np.concatenate([lf.gates for lf in req_leaves]) if req_leaves else np.zeros(0, dtype=np.int64)
            rg_sorted = np.sort(rg)
            k = np.searchsorted(rg_sorted, gs)
            kk = np.minimum(k, max(rg_sorted.shape[0] - 1, 0))
            wanted = inside & (rg_sorted[kk] == gs) if rg_sorted.shape[0] else np.zeros(gs.shape, dtype=bool)
        else:
            wanted = inside
        extra = gs[~wanted]
        for g in extra[: max(0, _NAME_LIMIT - named_extra)]:
            if is_boundary is not None and is_boundary(int(g)):
                p.append(f"{INTERIOR_DUPLICATE_GATE}: gate {int(g)} has a boundary position (step-1 value) and is emitted as interior in {where}")
            else:
                p.append(f"{INTERIOR_UNEXPECTED_GATE}: gate {int(g)} is not a required interior gate of {where}" if required is not None
                         else f"{INTERIOR_UNEXPECTED_GATE}: gate {int(g)} lies outside its call {span.call!r} [{span.lo}, {span.hi}) in {where}")
            named_extra += 1
        if extra.shape[0] > max(0, _NAME_LIMIT - named_extra) and named_extra >= _NAME_LIMIT:
            p.append(f"{INTERIOR_UNEXPECTED_GATE}: ... {int(extra.shape[0])} gates outside the layout in call {span.call!r}")
        if required is not None:
            missing = np.setdiff1d(rg_sorted, gs, assume_unique=True)
            for g in missing[: max(0, _NAME_LIMIT - named_missing)]:
                loc = required.locate(int(g))
                p.append(f"{INTERIOR_REGION_MISSING}: required gate {int(g)} absent from {where} (leaf {loc[1] if loc else '?'} flat {loc[2] if loc else '?'})")
                named_missing += 1
            if missing.shape[0] > _NAME_LIMIT and named_missing >= _NAME_LIMIT:
                p.append(f"{INTERIOR_REGION_MISSING}: ... {int(missing.shape[0])} required gates absent in call {span.call!r}")
        # the supplied leaves of this call are the rule's over its own gates (rebuild) and equal the derived layout's (required)
        gs_in, ws_in = gs[wanted], ws[wanted]
        if rebuild and gs_in.shape[0]:
            rebuilt = InteriorLayout.build(tree, [span], gs_in, ws_in, prefix=prefix, chunk_words=chunk_words, rule=rule_name)
            claimed_ids = [str(rows[i].get("canonical_id")) for i in range(a, b)]
            if rebuilt.ids != claimed_ids:
                first = next((i for i, (x, y) in enumerate(zip(rebuilt.ids, claimed_ids, strict=False)) if x != y), min(len(rebuilt.ids), len(claimed_ids)))
                p.append(f"{INTERIOR_ORDER}: {where}: the supplied leaves are not this rule's over their own gates (call {span.call!r}, first difference at "
                         f"rank {a + first}: rule {rebuilt.ids[first] if first < len(rebuilt.ids) else None!r}, supplied {claimed_ids[first] if first < len(claimed_ids) else None!r})")
            elif not extra.shape[0]:
                for j, lf in enumerate(rebuilt.leaves):
                    s0 = int(starts[a + j] - starts[a])
                    if not np.array_equal(lf.gates, got[s0:s0 + lf.words]):
                        p.append(f"{INTERIOR_ORDER}: {where}: leaf {lf.canonical_id!r} holds its gates in another order / other gates than the rule gives")
                        break
        if required is not None and not extra.shape[0] and not missing.shape[0]:
            # same gate set: the supplied layout order must be the derived layout's order, leaf by leaf
            want = rg
            if want.shape[0] == got.shape[0] and not np.array_equal(got, want):
                first = int(np.argmax(got != want))
                p.append(f"{INTERIOR_ORDER}: gates.bin differs from the layout order first at word {int(starts[a]) + first} (got gate {int(got[first])}, layout {int(want[first])})")
    return p


# --- E9's step-4 form: the whole selective bundle -------------------------------------------------------------------------------------------

INTERIOR_DIR = "interior"
RequiredSource = Mapping[str, "InteriorLayout | LazyInteriorLayout"] | Callable[[str, str], "InteriorLayout | LazyInteriorLayout | None"] | None


def _rehash_tree(d: Path, rows: Sequence[Mapping[str, Any]], where: str, rehash: bool, p: list[str]) -> tuple[str, bool]:
    """Leaf files present with the index's sizes (problems appended to ``p``); with ``rehash`` every leaf hashed from its bytes (a leaf whose
    bytes hash differently from its index row is named) and the root folded from those hashes with O(log n) memory.  Returns
    ``(root hex, files_ok)``."""
    from .tree import (  # the index's sha256 is the framed LEAF hash, not sha256(bytes)
        leaf_hash_of_row,
        root_of_leaf_hashes,
    )

    ok = [True]

    def hashes() -> Iterable[bytes]:
        for row in rows:
            f = d / str(row.get("file", f"leaf-{int(row['rank']):06d}.bin"))
            if not f.exists():
                p.append(f"{INTERIOR_REGION_MISSING}: {where} leaf {row.get('canonical_id')!r}: file {f.name} absent")
                ok[0] = False
                continue
            if f.stat().st_size != int(row.get("nbytes", -1)):
                p.append(f"{INTERIOR_WIDTH_MISMATCH}: {where} leaf {row.get('canonical_id')!r}: file {f.name} holds {f.stat().st_size} B, index says {row.get('nbytes')}")
                ok[0] = False
                continue
            if rehash:
                h = leaf_hash_of_row(row, f.read_bytes())
                if h.hex() != row.get("sha256"):
                    p.append(f"{INTERIOR_INDEX_MALFORMED}: {where} leaf {row.get('canonical_id')!r}: bytes do not hash to the index's leaf hash")
                yield h

    return root_of_leaf_hashes(hashes()).hex(), ok[0]


def validate_step4(selective_dir: str | os.PathLike[str], retained_request_ids: Sequence[str], pin: Mapping[str, Any] | None,
                   commitments: Mapping[str, Any], *, required: RequiredSource = None, is_boundary: Callable[[int], bool] | None = None,
                   rehash: bool = False) -> dict[str, Any]:
    """E9's step-4 check BEFORE ``r_s`` over the prover's selective bundle ``selective_dir`` (``adapter/selective/``):

    1. the pin is this rule's (:func:`check_pin`);
    2. every retained RU (``commitments.trees`` records of kind ``ru`` whose request is retained) has an ``interior_roots`` record and a tree
       directory ``interior/<req>/<step>/`` with ``index.json`` + ``gates.bin`` (else ``INTERIOR_REGION_MISSING``);
    3. per tree: ``index.json`` agrees with its ``interior_roots`` record (root, leaf_count, ids_digest, layout digest); the claimed leaves
       are exactly what this rule gives over the tree's own gates / calls / widths (chunk size, width split, ids, ranks -- ``INTERIOR_ORDER``
       / ``INTERIOR_UNEXPECTED_LEAF``); no gate twice and no boundary gate (``INTERIOR_DUPLICATE_GATE``); leaf files present with the right
       sizes (and, with ``rehash``, the index's sha256 -- the root itself is bound by the openings E7 verifies);
    4. with ``required`` -- E3's derived layouts, ``{"<req>/<step>": InteriorLayout}`` or ``(req, step) -> InteriorLayout | None`` (build them
       with :func:`derive_layout`) -- the full :func:`validate_supplied` per tree: a missing required region / gate is named
       (``INTERIOR_REGION_MISSING``).  Without it the missing-region check is NOT performed and the result says so (``required_source``).

    Returns ``{"problems": [...], "retained_rus": [...], "rus": {ru: facts}, "required_source": str, "trees_rehashed": {...}}``; ``problems == []``
    is the pass.  With ``rehash`` every tree's root is RECOMPUTED from the shipped leaf files (``facts.root_recomputed`` /
    ``root_recomputed_ok``; ``INTERIOR_ROOT_MISMATCH`` when it differs from ``index.json#root`` or the committed ``interior_roots`` root --
    INTEGRATOR D35, E9's ``V3.trees_rehashed``); ``trees_rehashed`` summarises ``{"rehash": bool, "recomputed": n, "ok": n, "mismatch": n}``."""
    sel = Path(selective_dir)
    retained = {str(r) for r in retained_request_ids}
    out: dict[str, Any] = {"problems": [], "retained_rus": [], "rus": {}, "required_source": None}
    p = out["problems"]
    p.extend(check_pin(pin, rule=rule_of(pin)))  # by the rule the pin NAMES (v1 records stay valid by name)
    roots = commitments.get("interior_roots") or {}
    rus: list[str] = []
    for rec in (commitments.get("trees") or {}).values():
        if isinstance(rec, Mapping) and rec.get("kind") == "ru" and str(rec.get("request_id")) in retained:
            step = rec.get("step")
            rus.append(f"{rec['request_id']}/{int(step):06d}" if isinstance(step, int) else f"{rec['request_id']}/{step}")
    if not rus:  # no RU records: fall back to the interior directories / roots of the retained requests
        rus = sorted({k for k in roots if k.partition('/')[0] in retained} | {f"{d.parent.name}/{d.name}" for r in retained for d in (sel / INTERIOR_DIR / r).glob("*") if d.is_dir()})
    out["retained_rus"] = sorted(set(rus))
    if callable(required):
        out["required_source"] = "callable (E3 derived layouts)"
    elif required is not None:
        out["required_source"] = f"mapping with {len(required)} derived layouts"
    else:
        out["required_source"] = "NONE: index self-consistency + commitments only; pass required= (derive_layout) for the missing-region check by gate"
    for ru in out["retained_rus"]:
        req, _, step = ru.partition("/")
        facts: dict[str, Any] = {"root": None, "leaves": None, "words": None, "checked_against_required": False, "files_ok": None,
                                 "root_recomputed": None, "root_recomputed_ok": None}
        out["rus"][ru] = facts
        d = sel / INTERIOR_DIR / req / step
        if ru not in roots:
            p.append(f"{INTERIOR_REGION_MISSING}: retained RU {ru} has no commitments.json#interior_roots record (no step-3 interior root)")
        if not (d / "index.json").exists():
            p.append(f"{INTERIOR_REGION_MISSING}: retained RU {ru} has no interior tree directory {d} (index.json absent)")
            continue
        try:
            index = json.loads((d / "index.json").read_text())
            gates_path = d / "gates.bin"
            n_gates = gates_path.stat().st_size // 8 if gates_path.exists() else None
        except (OSError, ValueError) as e:
            p.append(f"{INTERIOR_INDEX_MALFORMED}: {ru}: {type(e).__name__}: {str(e)[:120]}")
            continue
        facts.update(root=index.get("root"), leaves=index.get("leaf_count"), words=index.get("words"))
        if n_gates is None:
            p.append(f"{INTERIOR_INDEX_MALFORMED}: {ru}: gates.bin absent")
            continue

        def read_gates(offset: int, n: int, _path: Path = gates_path) -> np.ndarray:  # one call's slice of gates.bin, never the whole file
            return np.fromfile(_path, dtype="<i8", count=n, offset=8 * offset)

        idx_lay = index.get("layout") or {}
        if pin is not None and (rule_of(idx_lay) != rule_of(pin) or idx_lay.get("digest") != pin.get("digest")):
            p.append(f"{INTERIOR_LAYOUT_DIGEST}: interior/{ru} index.json was written under {rule_of(idx_lay)} digest {str(idx_lay.get('digest'))[:12]}, "
                     f"the instance pins {rule_of(pin)} digest {str(pin.get('digest'))[:12]}")
        if ru in roots:
            from .tree import check_index_against_commitments  # local import: tree imports layout

            p.extend(f"{ru}: {x}" for x in check_index_against_commitments(index, commitments, ru))
        # the claimed layout must be what the rule yields over its own inputs -- checked call by call (memory ~ the largest call)
        req_layout = required(req, step) if callable(required) else (required or {}).get(ru)
        try:
            rows = list(index.get("leaves") or [])
            total_words = sum(int(r.get("words", 0)) for r in rows)
            if total_words != n_gates:
                raise ValueError(f"gates.bin holds {n_gates} gates, the index {total_words} words")
            claimed_calls = [CallSpan(str(c["call"]), int(c["lo"]), int(c["hi"])) for c in index.get("calls") or []]
            idx_lay = index.get("layout") or {}
            p.extend(_check_gates_by_call(rows, claimed_calls, read_gates, tree=str(index.get("tree", f"interior/{ru}")), required=None, is_boundary=is_boundary,
                                          rule=rule_of(idx_lay), chunk_words=_chunk_words_of(idx_lay), prefix=str(index.get("prefix", "")), rebuild=True,
                                          ru_label=f"interior/{ru}"))
            # leaf files: present, right size; with rehash: every leaf hashed from its BYTES and the tree's root RECOMPUTED from those
            # hashes (INTEGRATOR D35: the root is never taken from the index / commitments alone) -- one leaf at a time, O(log n) memory
            root_recomputed, files_ok = _rehash_tree(d, rows, f"interior/{ru}", rehash, p)
            facts["files_ok"] = files_ok
            if rehash and files_ok:
                facts["root_recomputed"] = root_recomputed
                committed = (roots.get(ru) or {}).get("root") if isinstance(roots.get(ru), Mapping) else None
                if root_recomputed != index.get("root"):
                    p.append(f"{INTERIOR_ROOT_MISMATCH}: interior/{ru}: the root recomputed from the shipped leaf files {root_recomputed[:16]} != index.json#root "
                             f"{str(index.get('root'))[:16]}")
                if committed is not None and root_recomputed != committed:
                    p.append(f"{INTERIOR_ROOT_MISMATCH}: interior/{ru}: the root recomputed from the shipped leaf files {root_recomputed[:16]} != the committed "
                             f"interior_roots root {str(committed)[:16]}")
                facts["root_recomputed_ok"] = root_recomputed == index.get("root") and (committed is None or root_recomputed == committed)
            else:
                facts["root_recomputed"] = None
                facts["root_recomputed_ok"] = None if not rehash else False
        except (KeyError, TypeError, ValueError) as e:
            p.append(f"{INTERIOR_INDEX_MALFORMED}: {ru}: {type(e).__name__}: {str(e)[:160]}")
            continue
        # the required regions (E3's derivation), gates read call by call
        if req_layout is not None:
            facts["checked_against_required"] = True
            p.extend(f"{ru}: {x}" for x in validate_supplied(index, req_layout, gates_reader=(read_gates, n_gates), is_boundary=is_boundary))
    rec = [f for f in out["rus"].values() if f.get("root_recomputed") is not None]
    out["trees_rehashed"] = {"rehash": bool(rehash), "recomputed": len(rec), "ok": sum(1 for f in rec if f.get("root_recomputed_ok")),
                             "mismatch": sum(1 for f in rec if f.get("root_recomputed_ok") is False),
                             "label": "MEASURED: root = fold of leaf_hash(bytes) over the shipped leaf files, compared with index.json#root and interior_roots" if rehash
                             else "NOT_APPLICABLE: rehash=False (leaf files not re-hashed, roots taken from the index / commitments)"}
    return out


def derive_layout(program, request_id: str, step: int | str, positions: Iterable[int], *, prefix: str = "", boundary_gates: Sequence[int] | np.ndarray | None = None,
                  is_boundary: Callable[[int], bool] | None = None, widths: Callable[[np.ndarray], np.ndarray] | None = None,
                  chunk_words: int = CHUNK_WORDS, rule: str | None = None) -> InteriorLayout:
    """The VERIFIER's derivation of one Serve_v1 RU's interior layout from the pinned program: root calls = :func:`serve_calls` over the RU's
    positions; required gates = every gate of those calls minus the ones with a boundary position (``boundary_gates`` array or
    ``is_boundary`` oracle -- E3's ``RecordPartition.locate``); widths from ``widths(gates)`` (E3's kind-table oracle) or, slowly, from the
    program (:func:`widths_from_program`).  What E3/E9 pass as ``required`` to :func:`validate_step4`."""
    calls = serve_calls(program, positions)
    gates = np.concatenate([np.arange(c.lo, c.hi, dtype=np.int64) for c in calls]) if calls else np.zeros(0, dtype=np.int64)
    if boundary_gates is not None:
        gates = np.setdiff1d(gates, np.asarray(boundary_gates, dtype=np.int64), assume_unique=False)
    if is_boundary is not None and gates.shape[0]:
        gates = gates[~np.fromiter((is_boundary(int(g)) for g in gates), dtype=bool, count=gates.shape[0])]
    w = widths(gates) if widths is not None else widths_from_program(program, gates)
    return InteriorLayout.build(tree_id(request_id, step), calls, gates, np.asarray(w, dtype=np.int64), prefix=prefix, chunk_words=chunk_words, rule=rule)


# --- Serve_v1 helpers (vLLM adapter); anything else passes its own CallSpan list -------------------------------------------------------


def serve_calls(layout, positions: Iterable[int]) -> list[CallSpan]:
    """The root calls of a Serve_v1 RU holding positions ``positions`` (the capture's ``ru/<req>/<step>`` ``positions``), in pinned call
    order: for every ``t`` the ``embed``, per-layer ``pre|attn|post`` and (``t >= LP-1``) ``final`` root nodes, sorted by node index
    (``verity_vllm.ir.lazy.ServeLayout``: prefill roots layer-major, decode roots position-major).  ``layout`` is a ``ServeLayout`` (or a
    ``LazyProgram``, whose ``.layout`` is taken)."""
    lay = getattr(layout, "layout", layout)
    ks: list[int] = []
    for t in sorted({int(t) for t in positions}):
        if not (0 <= t < lay.P):
            raise ValueError(f"position {t} outside the program's {lay.P} positions")
        ks.append(lay.index(("embed", t)))
        for L in range(lay.NL):
            for kind in ("pre", "attn", "post"):
                ks.append(lay.index((kind, t, L)))
        if t >= lay.LP - 1:
            ks.append(lay.index(("final", t)))
    ks.sort()
    return [CallSpan(lay.name(k), lay.offset(k), lay.offset(k + 1)) for k in ks]


def widths_from_program(program, gates: Sequence[int] | np.ndarray) -> np.ndarray:
    """Output width (bits) of every gate from the pinned program itself (``Circuit.gate(g).prim.ret.width``).  Slow (a lookup per
    gate; fine for fixtures and spot checks) -- E3's oracle supplies widths from the population's kind table at scale."""
    circ = program.circuit
    return np.asarray([int(circ.gate(int(g)).prim.ret.w) for g in np.asarray(gates).reshape(-1)], dtype=np.int64)
