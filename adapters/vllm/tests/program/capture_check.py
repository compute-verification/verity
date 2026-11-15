"""The check `test_derive_negative` holds a derived Program's calls to, and `test_fp8`'s Input ranges: a call's region
evaluated gate by gate from value segments (captured or computed words by root gate index), its returns compared word for
word with the captured ones, and a captured tensor's shape and dtype held to the call's return gates."""
from __future__ import annotations

import bisect

import numpy as np

from verity.circuits.evaluation import MissingValue, evaluate_region
from verity.circuits.layout import Circuit, FunctionCall, resolve


class Segments:
    """Sorted disjoint [start, end) gate ranges, each backed by a flat words array and a location record.
    Additions are buffered and sorted on first lookup (Gemm returns are one gate per coordinate)."""

    def __init__(self) -> None:
        self._pending: list[tuple[int, np.ndarray, dict]] = []
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.words: list[np.ndarray] = []
        self.locs: list[dict] = []

    def add(self, start: int, words: np.ndarray, loc: dict) -> None:
        self._pending.append((start, words, loc))

    def _finalize(self) -> None:
        if not self._pending:
            return
        items = sorted(zip(self.starts, self.words, self.locs), key=lambda x: x[0]) + sorted(self._pending, key=lambda x: x[0])
        items.sort(key=lambda x: x[0])
        self._pending = []
        self.starts, self.ends, self.words, self.locs = [], [], [], []
        for start, words, loc in items:
            if self.ends and self.ends[-1] > start:
                raise ValueError(f"overlapping segment at gate {start} ({loc.get('leaf_id')} vs {self.locs[-1].get('leaf_id')})")
            self.starts.append(start)
            self.ends.append(start + len(words))
            self.words.append(words)
            self.locs.append(loc)

    def __len__(self) -> int:
        return len(self.starts) + len(self._pending)

    def find(self, g: int) -> tuple[int, int] | None:
        if self._pending:
            self._finalize()
        i = bisect.bisect_right(self.starts, g) - 1
        if i >= 0 and g < self.ends[i]:
            return i, g - self.starts[i]
        return None

    def value(self, g: int) -> int | None:
        f = self.find(g)
        if f is None:
            return None
        i, k = f
        return int(self.words[i][k])


class LazyTranscript(dict):
    """A `Trace` whose missing entries are read from value segments.  Values computed by `evaluate_region` are stored in
    the dict and take precedence."""

    def __init__(self, *sources: Segments) -> None:
        super().__init__()
        self.sources = sources

    def __missing__(self, g: int) -> int:
        for s in self.sources:
            v = s.value(g)
            if v is not None:
                return v
        raise KeyError(g)

    def __contains__(self, g: object) -> bool:  # type: ignore[override]
        return dict.__contains__(self, g) or any(s.find(g) is not None for s in self.sources)  # type: ignore[arg-type]


def coll_range(circuit: Circuit, coll) -> tuple[int, int]:
    """Root gate range [start, end) of a contiguous Input collection."""
    n = coll.refs.count if hasattr(coll.refs, "count") else coll.type.leaves
    first = resolve(circuit.root_scope, coll.refs[0])
    last = resolve(circuit.root_scope, coll.refs[n - 1])
    if last - first != n - 1:
        raise ValueError("input collection is not contiguous")
    return first, first + n


def shape_ok(call: FunctionCall, gates: tuple[int, ...], entry: dict) -> tuple[bool, str]:
    exp = int(np.prod(entry["shape"])) if entry["shape"] else 1
    if exp != len(gates):
        return False, f"captured {entry['shape']} has {exp} words, IR returns {len(gates)} gates"
    w = {call.circuit.gate(g).prim.ret.w for g in (gates[0], gates[-1])} if gates else set()
    want = {"bfloat16": 16, "int32": 32, "float32": 32, "float8_e4m3fn": 8}[entry["dtype"]]
    if w != {want}:
        return False, f"captured dtype {entry['dtype']} ({want} bits) vs gate widths {sorted(w)}"
    return True, "ok"


def check_call(call: FunctionCall, gates: tuple[int, ...], transcript: LazyTranscript, captured: np.ndarray, *, sub=None) -> dict:
    """Evaluate `sub` (default: the whole call) gate by gate, reading outside operands from the trace's sources, and
    compare the words at `gates` that lie inside `sub` with `captured`.  An operand no source holds is `missing-operand`."""
    region = call if sub is None else sub
    # a fresh overlay, so captured words of this call's own returns are never read as operands
    overlay = LazyTranscript(*transcript.sources)
    try:
        evaluate_region(call.circuit, region, overlay)
    except MissingValue as e:
        return {"status": "missing-operand", "detail": str(e)}
    lo, hi = region.lo, region.hi
    compared = 0
    mism: list[dict] = []
    for j, g in enumerate(gates):
        if not (lo <= g < hi):
            continue
        compared += 1
        v = dict.__getitem__(overlay, g)
        if v != int(captured[j]):
            mism.append({"word": j, "gate": g, "ir": int(v), "captured": int(captured[j])})
    return {"status": "exact" if not mism else "mismatch", "gates_evaluated": hi - lo, "words_compared": compared,
            "mismatches": len(mism), "first_mismatches": mism[:16]}
