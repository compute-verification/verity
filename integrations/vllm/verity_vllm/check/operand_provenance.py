"""Exhaustive operand provenance check (G5, D82) -- PROTECTED.

    uv run verity-vllm operand-provenance --log data/logs/m1.jsonl.gz \\
        --profile vllm_d9105ea80_sm89_eager --out out/capture/reports/wiring.json

What this establishes
---------------------
The fold decides, for every operand of every Program instance, which producer (node output leaves, weight
leaves, prompt tokens) supplied each element, by consulting its versioned memory model (`memory.py`).  This
module re-derives that answer for **every read** with a deliberately different, brute-force implementation and
compares the two exactly.  The observed facts both sides start from are the same: the storage identity and
generation of every operand (`ViewDesc.alloc`, one id per `(pointer range, generation)`), its byte range and
element geometry (`offset/shape/strides/dtype`), and the order of events on the single execution stream (the
log order).  Given those facts and the write set of every claimed launch, "which write did this read observe"
is a determined question; here it is answered twice, independently.

Checks (all exhaustive, over `FoldResult.accesses`, the fold's memory-access journal):

  wiring     every read's refs (what the fold wired into the instance) == the refs a last-writer byte map replays
             for the same view at the same moment (alloc identity + generation + byte range + order)
  coverage   a read never touches a byte that no journaled write (root, token, instance output, move) has
             produced in the allocation's current generation (the fold would have raised Unresolved; asserted again)
  stream     every Dispatch/Triton event of the run is on one stream (the log order is the execution order;
             multi-stream runs are outside what this check can establish and are reported as such)
  schema     for dispatcher ops the schema's mutable (`Tensor(a!)`) arguments and returned tensors are an
             independent statement of the launch's write set: every journaled write of a dispatcher event must
             lie inside a schema-declared written view (`writes_outside_schema` fails); schema-declared written
             views the pattern did not write are counted (`declared_unwritten`, stale-contents risk; a later read
             of them would be Unresolved).  Triton launches carry no schema: their write sets rest on the
             pattern contract and are counted in `triton_writes`.

What it does not establish: that the observer's `ViewDesc`s are the views the kernels used (observer facts),
and that a pattern's declared operand/write regions are what the kernel touched (pattern contract).  Both are
covered elsewhere: census + non-interference (G1), the sampled bit-exact evaluation of instances from their
operands (replay tier a), and byte evidence where recorded.

Output `wiring.json`: {"status": "pass"|"fail", "reads", "writes", "mismatches": [...first 50...], "n_mismatch",
"uncovered": n, "streams": {...}, "schema": {...}, "seconds"}.
"""

from __future__ import annotations

import bisect
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import option
from verity_vllm.observe.events import DispatchEnter, DispatchExit, TritonLaunch, ViewDesc
from verity_vllm.observe.log import read_log
from verity_vllm.observe.fold.views import byte_extent, collapsed_dims, element_runs, row_view
from verity_vllm.observe.fold.memory import access_runs

Run = tuple[str, int, int, int]      # (space, idx, lo, hi) — the same runs `verity.ml.circuits.refs.runs` yields


# ---------------------------------------------------------------------------------------------------------
# run arithmetic (independent of verity.ml.circuits.refs)
# ---------------------------------------------------------------------------------------------------------


def _canon(rs: list[Run]) -> list[Run]:
    out: list[Run] = []
    for sp, ix, lo, hi in rs:
        if hi <= lo:
            continue
        if out and out[-1][0] == sp and out[-1][1] == ix and out[-1][3] == lo:
            out[-1] = (sp, ix, out[-1][2], hi)
        else:
            out.append((sp, ix, lo, hi))
    return out


def _slice_runs(rs: list[Run], start: int, count: int) -> list[Run]:
    """Elements [start, start+count) of the sequence denoted by `rs`."""
    out: list[Run] = []
    pos = 0
    end = start + count
    for sp, ix, lo, hi in rs:
        n = hi - lo
        a, b = max(start, pos), min(end, pos + n)
        if a < b:
            out.append((sp, ix, lo + (a - pos), lo + (b - pos)))
        pos += n
        if pos >= end:
            break
    return out


def _count(rs: list[Run]) -> int:
    return sum(hi - lo for _, _, lo, hi in rs)


# ---------------------------------------------------------------------------------------------------------
# brute-force last-writer byte map
# ---------------------------------------------------------------------------------------------------------


class _Alloc:
    """Disjoint sorted byte intervals [lo, hi) -> (elem_bytes, refs runs for the elements in [lo, hi), writer seq)."""

    __slots__ = ("lo", "hi", "val")

    def __init__(self) -> None:
        self.lo: list[int] = []
        self.hi: list[int] = []
        self.val: list[tuple[int, list[Run], int]] = []

    def _cut(self, at: int) -> None:
        """Split the interval containing byte `at` (if any) so that `at` is a boundary.  A cut inside an element
        drops that element's bytes (the same discipline as memory.py: a partially clobbered element is unwritten)."""
        i = bisect.bisect_right(self.lo, at) - 1
        if i < 0 or self.hi[i] <= at or self.lo[i] == at:
            return
        lo, hi, (eb, rs, seq) = self.lo[i], self.hi[i], self.val[i]
        k, rem = divmod(at - lo, eb)
        left = rs and _slice_runs(rs, 0, k) or []
        right_start = k + (1 if rem else 0)
        n = (hi - lo) // eb
        right = _slice_runs(rs, right_start, n - right_start) if right_start < n else []
        del self.lo[i], self.hi[i], self.val[i]
        pieces = []
        if k:
            pieces.append((lo, lo + k * eb, (eb, left, seq)))
        if right_start < n:
            pieces.append((lo + right_start * eb, hi, (eb, right, seq)))
        for j, (a, b, v) in enumerate(pieces):
            self.lo.insert(i + j, a); self.hi.insert(i + j, b); self.val.insert(i + j, v)

    def write(self, lo: int, hi: int, eb: int, rs: list[Run], seq: int) -> None:
        if hi <= lo:
            return
        self._cut(lo); self._cut(hi)
        i = bisect.bisect_left(self.lo, lo)
        j = i
        while j < len(self.lo) and self.lo[j] < hi:
            j += 1
        del self.lo[i:j], self.hi[i:j], self.val[i:j]
        self.lo.insert(i, lo); self.hi.insert(i, hi); self.val.insert(i, (eb, rs, seq))

    def read(self, lo: int, hi: int, eb: int) -> tuple[list[Run], int]:
        """Refs for elements of width `eb` over bytes [lo, hi); returns (runs, uncovered_bytes)."""
        out: list[Run] = []
        missing = 0
        pos = lo
        i = bisect.bisect_right(self.lo, lo) - 1
        if i < 0:
            i = 0
        while pos < hi and i < len(self.lo):
            a, b, (web, rs, _) = self.lo[i], self.hi[i], self.val[i]
            if b <= pos:
                i += 1
                continue
            if a > pos:
                gap = min(a, hi) - pos
                missing += gap
                pos += gap
                continue
            end = min(b, hi)
            if web != eb or (pos - a) % web:
                missing += end - pos                      # width reinterpretation / misaligned: not resolvable
            else:
                out.extend(_slice_runs(rs, (pos - a) // eb, (end - pos) // eb))
            pos = end
            i += 1
        if pos < hi:
            missing += hi - pos
        return _canon(out), missing


class ProvenanceChecker:
    def __init__(self) -> None:
        self.allocs: dict[int, _Alloc] = {}
        self.reads = 0
        self.writes = 0
        self.mismatches: list[dict] = []
        self.n_mismatch = 0
        self.uncovered = 0
        self.by_kind: Counter = Counter()

    def _runs_of_view(self, view: ViewDesc, rows: list[int] | None) -> list[tuple[int, int]]:
        if rows is None:
            return element_runs(view)
        out: list[tuple[int, int]] = []
        for r in rows:
            out.extend(element_runs(row_view(view, r)))
        return out

    def apply(self, acc: dict) -> None:
        kind = acc["kind"]
        self.by_kind[kind] += 1
        if kind == "alloc":
            self.allocs[acc["alloc"]] = _Alloc()          # a new generation: previous contents unreachable
            return
        if kind == "free":
            self.allocs.pop(acc["alloc"], None)
            return
        view: ViewDesc = acc["view"]
        eb = view.itemsize
        rs: list[Run] = [tuple(r) for r in access_runs(acc["refs"])]
        eruns = self._runs_of_view(view, acc.get("rows"))
        if kind in ("write", "root", "token"):
            self.writes += 1
            a = self.allocs.get(view.alloc)
            if a is None:                                  # never announced: register implicitly (as memory.py does)
                a = self.allocs[view.alloc] = _Alloc()
            pos = 0
            for start, n in eruns:
                a.write(start * eb, (start + n) * eb, eb, _slice_runs(rs, pos, n), acc["seq"])
                pos += n
            return
        if kind == "read":
            self.reads += 1
            a = self.allocs.get(view.alloc)
            got: list[Run] = []
            missing = 0
            if a is None:
                missing = sum(n for _, n in eruns) * eb
            else:
                for start, n in eruns:
                    part, miss = a.read(start * eb, (start + n) * eb, eb)
                    got.extend(part)
                    missing += miss
            got = _canon(got)
            want = _canon(rs)
            if missing:
                self.uncovered += 1
            if got != want or missing:
                self.n_mismatch += 1
                if len(self.mismatches) < 50:
                    self.mismatches.append({"seq": acc["seq"], "event": acc["event"], "step": acc["step"], "what": acc["what"],
                                            "view": view.__dict__ if hasattr(view, "__dict__") else str(view),
                                            "uncovered_bytes": missing, "fold": want[:8], "recomputed": got[:8],
                                            "fold_runs": len(want), "recomputed_runs": len(got)})
            return
        raise ValueError(f"unknown access kind {kind!r}")


# ---------------------------------------------------------------------------------------------------------
# stream census + schema write-set cross-check (facts from the raw log)
# ---------------------------------------------------------------------------------------------------------

# `Tensor(a!) x`, `Tensor(a!)[] xs`, `Tensor($0! -> ) x` (torch.library custom ops), optional `?`
_MUTABLE = re.compile(r"Tensor\((?:[a-z]|\$\d+)!(?:\s*->\s*[^)]*)?\)(?:\[\])?\??\s+(\w+)")


def _schema_mutable_names(schema: str) -> list[str]:
    head = schema.rsplit("->", 1)[0]
    return _MUTABLE.findall(head)


def _schema_arg_names(schema: str) -> list[str]:
    head = schema.rsplit("->", 1)[0]
    inside = head[head.find("(") + 1: head.rfind(")")]
    names = []
    for part in inside.split(","):
        part = part.strip()
        if not part or part == "*":
            continue
        tok = part.split("=")[0].split()
        names.append(tok[-1] if tok else "")
    return names


def _views_in(x: Any) -> list[ViewDesc]:
    if isinstance(x, ViewDesc):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for y in x:
            out.extend(_views_in(y))
        return out
    return []


#: A declared view of more element runs than this is its byte extent (coarse, a superset), e.g. the whole KV cache.
DECLARED_RUNS_MAX = 20_000


def _declared_regions(v: ViewDesc) -> list[tuple[int, int, int]]:
    """`(alloc, byte lo, byte hi)` regions of a schema-declared written view: its element runs, not its byte extent (strided
    views have holes), or the extent above `DECLARED_RUNS_MAX` runs.  The runs are walked with the dims in descending-stride
    order, broadcast (stride-0) dims outermost: the same addresses, but a permuted-contiguous view such as a transposed weight
    is one run instead of one per element."""
    if v.numel == 0:
        return []
    dims = sorted(zip(v.shape, v.strides), key=lambda d: (d[1] != 0, -d[1]))
    w = ViewDesc(v.alloc, v.offset, tuple(n for n, _ in dims), tuple(st for _, st in dims), v.dtype, v.device)
    shape, strides = collapsed_dims(w)
    runs = w.numel // shape[-1] if shape and strides[-1] == 1 else w.numel     # the runs `element_runs(w)` walks
    if runs > DECLARED_RUNS_MAX:
        lo, hi = byte_extent(v)
        return [(v.alloc, lo, hi)]
    eb = v.itemsize
    return [(v.alloc, start * eb, (start + n) * eb) for start, n in element_runs(w)]


def stream_and_schema_facts(log_path: str | Path, writes_by_event: dict[int, list[tuple[int, int, int]]]) -> tuple[dict, dict]:
    """`writes_by_event`: event id -> [(alloc, byte lo, byte hi)] regions of its journaled writes (`_declared_regions` of each
    write's view, the rule the declared side uses).  Returns (streams, schema)."""
    streams: Counter = Counter()
    enters: dict[int, DispatchEnter] = {}
    parent: dict[int, int | None] = {}
    declared: dict[int, list[tuple[int, int, int]]] = {}
    triton_roots: set[int] = set()
    triton_writes = 0

    def root_of(eid: int) -> int | None:
        """The nearest enclosing event (itself included) that the fold attributed writes to (a resolution root)."""
        x: int | None = eid
        while x is not None:
            if x in writes_by_event:
                return x
            x = parent.get(x)
        return None

    for ev in read_log(log_path):
        if isinstance(ev, DispatchEnter):
            streams[("dispatch", ev.stream)] += 1
            enters[ev.id] = ev
            parent[ev.id] = ev.parent
        elif isinstance(ev, TritonLaunch):
            streams[("triton", ev.stream)] += 1
            parent[ev.id] = ev.parent
            r = root_of(ev.id)
            if r is not None:
                triton_roots.add(r)
        elif isinstance(ev, DispatchExit):
            en = enters.get(ev.id)
            r = root_of(ev.id) if en is not None else None
            if en is None or r is None:
                continue
            regions = declared.setdefault(r, [])
            names = _schema_arg_names(en.schema)
            mut = set(_schema_mutable_names(en.schema))
            for i, arg in enumerate(en.args):
                if i < len(names) and names[i] in mut:
                    for v in _views_in(arg):
                        regions.extend(_declared_regions(v))
            for k, arg in en.kwargs.items():
                if k in mut:
                    for v in _views_in(arg):
                        regions.extend(_declared_regions(v))
            for v in _views_in(ev.outs):
                regions.extend(_declared_regions(v))
    # Roots whose subtree contains a Triton launch: the write set rests on the pattern contract (no schema).
    outside = 0
    outside_examples: list[dict] = []
    unwritten_examples: list[dict] = []
    noschema_ops: Counter = Counter()
    unwritten_by_op: Counter = Counter()
    declared_unwritten = 0
    schema_roots = 0
    for eid, ws in writes_by_event.items():
        if eid in triton_roots:
            triton_writes += len(ws)
            continue
        regions = declared.get(eid, [])
        if not regions:
            noschema_ops[enters[eid].op if eid in enters else "?"] += 1
            continue
        schema_roots += 1
        for (al, lo, hi) in ws:
            if not any(al == a2 and lo >= l2 and hi <= h2 for a2, l2, h2 in regions):
                outside += 1
                if len(outside_examples) < 20:
                    outside_examples.append({"event": eid, "op": enters[eid].op if eid in enters else None, "write": (al, lo, hi), "declared": regions[:6]})
        merged: dict[int, list[tuple[int, int]]] = {}
        for al, lo, hi in sorted(ws):
            iv = merged.setdefault(al, [])
            if iv and lo <= iv[-1][1]:
                iv[-1] = (iv[-1][0], max(iv[-1][1], hi))
            else:
                iv.append((lo, hi))
        for (a2, l2, h2) in regions:
            if h2 > l2 and not any(lo <= l2 and hi >= h2 for lo, hi in merged.get(a2, [])):
                declared_unwritten += 1
                unwritten_by_op[enters[eid].op if eid in enters else "?"] += 1
                if len(unwritten_examples) < 20:
                    unwritten_examples.append({"event": eid, "op": enters[eid].op if eid in enters else None, "declared": (a2, l2, h2),
                                               "written": merged.get(a2, [])[:4]})
    distinct = sorted({s for (_, s) in streams})
    st = {"distinct_streams": distinct, "single_stream": len(distinct) <= 1, "events_by_stream": {str(k): v for k, v in streams.items()}}
    sc = {"roots_with_writes": len(writes_by_event), "schema_roots": schema_roots, "triton_roots": len(triton_roots),
          "writes_outside_schema": outside, "outside_examples": outside_examples,
          "declared_unwritten": declared_unwritten, "declared_unwritten_by_op": dict(unwritten_by_op.most_common(20)), "unwritten_examples": unwritten_examples, "triton_writes": triton_writes,
          "roots_without_schema_regions": dict(noschema_ops.most_common(20))}
    return st, sc


# ---------------------------------------------------------------------------------------------------------


def check(result, log_path: str | Path | None = None) -> dict:
    t0 = time.time()
    ck = ProvenanceChecker()
    writes_by_event: dict[int, list[tuple[int, int, int]]] = {}
    for acc in result.accesses:
        ck.apply(acc)
        if acc["kind"] == "write" and acc["event"] is not None and acc["event"] >= 0:
            v = acc["view"]
            rows = acc.get("rows")
            # A journaled write is listed by the declared side's own rule: a strided write's byte extent spans its holes.
            for u in ([v] if rows is None else [row_view(v, r) for r in rows]):
                writes_by_event.setdefault(acc["event"], []).extend(_declared_regions(u))
    rep: dict[str, Any] = {"reads": ck.reads, "writes": ck.writes, "accesses": dict(ck.by_kind), "n_mismatch": ck.n_mismatch,
                           "uncovered": ck.uncovered, "mismatches": ck.mismatches, "instances": len(result.instances)}
    fails = []
    if ck.n_mismatch:
        fails.append(f"{ck.n_mismatch} read(s) whose provenance the independent map does not reproduce")
    if log_path is not None:
        st, sc = stream_and_schema_facts(log_path, writes_by_event)
        rep["streams"], rep["schema"] = st, sc
        if not st["single_stream"]:
            fails.append(f"events on {len(st['distinct_streams'])} streams: log order is not execution order")
        if sc["writes_outside_schema"]:
            fails.append(f"{sc['writes_outside_schema']} journaled write(s) outside the schema-declared write set")
    rep["status"] = "fail" if fails else "pass"
    rep["failures"] = fails
    rep["seconds"] = round(time.time() - t0, 3)
    return rep


@dataclass(frozen=True, kw_only=True)
class Options:
    log: str = option("--log")
    profile: str = option("--profile")
    out: str = option("--out")
    no_facts: bool = option("--no-facts", default=False, action="store_true", help="skip the raw-log stream/schema pass")


def main(a: Options) -> int:
    from verity_vllm.check.replay.replay import fold_log, load_profile
    t0 = time.time()
    profile = load_profile(a.profile)
    result = fold_log(a.log, profile, with_program=True)
    fold_s = round(time.time() - t0, 3)
    rep = check(result, None if a.no_facts else a.log)
    rep = {"log": a.log, "profile": a.profile, "digest": result.digest, "fold_seconds": fold_s, **rep}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str))
    print(f"[provenance] {rep['status']}: {rep['reads']} reads, {rep['writes']} writes, {rep['n_mismatch']} mismatches, "
          f"uncovered={rep['uncovered']}, streams={rep.get('streams', {}).get('distinct_streams')}, "
          f"schema={ {k: v for k, v in rep.get('schema', {}).items() if not k.endswith('examples') and not k.endswith('_by_op') and k != 'roots_without_schema_regions'} } ({rep['seconds']}s) -> {out}")
    return 0 if rep["status"] == "pass" else 1
