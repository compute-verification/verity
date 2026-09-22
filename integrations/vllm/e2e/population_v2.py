"""[SAMPLER] ``veritor/population/v2``: the sampling population of a validated partition in ``O(runs)``, not ``O(N_V)``.

``population/v1`` (``e2e.population.write_maps``) lists ``ru_of[vu]`` for every verification unit -- 166,196,840 entries
for whole-B0 P1/cu=whole, which no verifier should have to hold.  A compact canonical plan already *is* a list of runs
(:class:`veritor.plans.runs.UnitRuns`: one :class:`~veritor.plans.runs.UnitRun` per stopped node per activation, units in
arithmetic progression over the gates), so the population is written as **rows** ``(first_vu, count, ru, kind, lo, stride,
unit_size)`` -- every run split at commitment-unit boundaries (refinement makes that a split *between* units, never inside
one) and coalesced with its arithmetic neighbours -- plus the **kind table** (one entry per distinct stopped-node callee
``(definition name, node form)``: the kind program the unit is an instance of) and the **commitment side** (the merged gate
intervals of every CU).  Everything a sampler needs is an ``O(log rows)`` lookup:

* :meth:`CompactPopulation.ru_of`, :meth:`CompactPopulation.kind_of`, :meth:`CompactPopulation.gate_interval` per VU index;
* :meth:`CompactPopulation.ru_size` / :meth:`CompactPopulation.vu_at` (the ``k``-th VU of an RU in ascending index order),
  which is what the second Bernoulli stage ranks over (``e2e.partition_sample``).

The file pins the same digests as v1 (``program_digest``, ``plan_digest`` + kind, ``partition_digest`` = SUMMARY-C's
``partition-intervals/v1``) and is **self-checking**: :meth:`CompactPopulation.check` re-derives the partition digest from the
rows and the CU intervals (``O(N_V)`` sha updates, ``O(1)`` memory -- 60 s for whole B0, MEASURED by COMPACT-P1 for the same
loop), checks that the rows tile ``[0, N_V)`` and the circuit's gates exactly, that every row lies inside its RU, and the
content sha256.  A tampered ``ru``, ``lo``, ``count`` or kind is therefore detected, not trusted.

Equality with the dense population (same ``vu_count``, ``ru_count``, every ``ru_of``, every gate interval) is asserted on TINY
in ``tests/test_sampler_partition.py``.  ``population/v1`` stays readable (``e2e.partition_sample.load_population``).

    python -m e2e.population_v2 build --case TINY --plan P1/cu=whole --out DIR      # DIR/population.compact.json (+ v1 for comparison on TINY)
    python -m e2e.population_v2 check DIR/population.compact.json [--fast]
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

POPULATION_V2 = "veritor/population/v2"
POPULATION_V1 = "veritor/population/v1"
KIND_SCHEMA = "veritor/population-kinds/v1"
COMPACT_FILE = "population.compact.json"

ROW_FIELDS = ("first_vu", "count", "ru", "kind", "lo", "stride", "unit_size")


class PopulationError(ValueError):
    """The population file is malformed, self-inconsistent, or not the partition it claims (REFUSE)."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


# -- kind table ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    """One kind program of the plan: the definition the stopped node calls (``fn``: a plan-IR ``FunctionDefinition`` or
    ``PrimitiveDefinition`` name, parameters included) under node form ``form`` (``batch`` for a member of an unrolled batch,
    ``call``/``primitive``/``scan`` otherwise).  ``out_bits`` is the callee's declared return width (a primitive's ``width``;
    a definition's returns summed); ``size`` its gates per unit."""

    id: int
    fn: str
    form: str
    primitive: bool
    out_bits: int
    size: int

    @property
    def name(self) -> str:
        return f"{self.fn}|{self.form}"

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "fn": self.fn, "form": self.form, "primitive": self.primitive, "out_bits": self.out_bits, "size": self.size}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Kind:
        return cls(int(d["id"]), str(d["fn"]), str(d["form"]), bool(d["primitive"]), int(d["out_bits"]), int(d["size"]))


def _callee_out_bits(fn) -> int:
    from veritor.plans.ir import PrimitiveDefinition

    if isinstance(fn, PrimitiveDefinition):
        return int(fn.width)
    # a definition: the sum of its return widths (the unit's declared interface; the validator's S.out may be narrower)
    total = 0
    for ref in fn.returns:
        prim = _return_prim(fn, ref)
        total += int(prim.width) if prim is not None else 0
    return total


def _return_prim(fn, ref):
    """The primitive producing a return reference of ``fn`` (``("n", node, leaf)``), if it is a node of ``fn``; else None."""

    from veritor.plans.ir import PrimitiveDefinition

    try:
        space, idx = ref[0], ref[1]
    except Exception:
        return None
    if space != "n" or not 0 <= idx < len(fn.nodes):
        return None
    callee = fn.nodes[idx].fn
    if isinstance(callee, PrimitiveDefinition):
        return callee
    return None


# -- construction --------------------------------------------------------------------------------------------------------------


def _cu_intervals(plan) -> list[list[list[int]]]:
    """Merged gate intervals per commitment unit, plan order (the CU side of ``partition_digest``)."""

    from veritor.plans.runs import UnitRuns

    side = plan.commitment_units
    out: list[list[list[int]]] = []
    if isinstance(side, UnitRuns):
        for run in side.runs:
            for j in range(run.count):
                lo, hi = run.interval(j)
                out.append([[lo, hi]])
        return out
    for unit in side:
        merged: list[list[int]] = []
        for lo, hi in unit.intervals.intervals():
            if merged and merged[-1][1] == lo:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        out.append(merged)
    return out


class _CuIndex:
    """``owner(gate)`` over the CU intervals, ``O(log intervals)``."""

    def __init__(self, cu_intervals: list[list[list[int]]]) -> None:
        entries = sorted((lo, hi, c) for c, ivs in enumerate(cu_intervals) for lo, hi in ivs)
        self.los = [e[0] for e in entries]
        self.his = [e[1] for e in entries]
        self.cus = [e[2] for e in entries]

    def owner(self, gate: int) -> int | None:
        k = bisect.bisect_right(self.los, gate) - 1
        if k >= 0 and gate < self.his[k]:
            return self.cus[k]
        return None

    def interval_at(self, gate: int) -> tuple[int, int, int] | None:
        k = bisect.bisect_right(self.los, gate) - 1
        if k >= 0 and gate < self.his[k]:
            return self.los[k], self.his[k], self.cus[k]
        return None


class RowSink:
    """A ``rows`` list that keeps only the row still open for coalescing and hands every finished row to ``emit`` (a file
    writer, a counter): whole-B0's rows (<= 6.1 M runs) without holding them in memory."""

    def __init__(self, emit=None) -> None:
        self._last: list[int] | None = None
        self.count = 0
        self._emit = emit

    def append(self, row: list[int]) -> None:
        if self._last is not None and self._emit is not None:
            self._emit(self._last)
        self._last = row
        self.count += 1

    def __bool__(self) -> bool:
        return self._last is not None

    def __getitem__(self, i: int) -> list[int]:
        if i != -1 or self._last is None:
            raise IndexError(i)
        return self._last

    def close(self) -> None:
        if self._last is not None and self._emit is not None:
            self._emit(self._last)
            self._last = None


def build_rows(plan, circuit, *, sink=None) -> tuple[list[Kind], list[list[int]], list[list[list[int]]], int]:
    """Kinds, rows, CU intervals and the VU count of a validated plan whose verification side is :class:`UnitRuns` (or a
    dense side, walked unit by unit: TINY / tests).  Rows: ``[first_vu, count, ru, kind, lo, stride, unit_size]``.  With a
    :class:`RowSink` as ``sink`` the rows are streamed to it (returned ``rows`` is then the sink)."""

    from veritor.plans.ir import PrimitiveDefinition
    from veritor.plans.runs import UnitRuns

    cu_ivs = _cu_intervals(plan)
    owners = _CuIndex(cu_ivs)
    kinds: list[Kind] = []
    kind_ids: dict[str, int] = {}

    def kind_id(fn, form: str, size: int) -> int:
        # (callee, form, unit size): at the coordinate level the size is the kind program's; a coarser plan's batch units
        # differ in size by member count, so the size is part of the entry (kind_counts aggregates by callee|form)
        key = f"{fn.name}|{form}|{size}"
        k = kind_ids.get(key)
        if k is None:
            k = kind_ids[key] = len(kinds)
            kinds.append(Kind(k, fn.name, form, isinstance(fn, PrimitiveDefinition), _callee_out_bits(fn), size))
        return k

    rows = [] if sink is None else sink

    def emit(first_vu: int, count: int, kid: int, lo: int, stride: int, unit_size: int) -> None:
        """Split ``count`` units at ``lo + j*stride`` across CU intervals; coalesce with the previous row when arithmetic."""

        j = 0
        while j < count:
            g = lo + j * stride
            iv = owners.interval_at(g)
            if iv is None:
                raise PopulationError(f"VU {first_vu + j} at gate {g} is owned by no commitment unit")
            clo, chi, cu = iv
            if g + unit_size > chi:
                raise PopulationError(f"VU {first_vu + j} [{g}, {g + unit_size}) straddles commitment unit {cu} [{clo}, {chi})")
            # units j .. j_end-1 start inside [clo, chi)
            j_end = min(count, j + (chi - g + stride - 1) // stride)
            n = j_end - j
            if rows:
                p = rows[-1]
                if p[2] == cu and p[3] == kid and p[6] == unit_size and p[0] + p[1] == first_vu + j:
                    if n == 1 and p[1] == 1 and g > p[4]:
                        # two consecutive singleton units of one kind (e.g. the same primitive repeated in a straight-line
                        # body): the second fixes the row's stride
                        p[1], p[5] = 2, g - p[4]
                        j = j_end
                        continue
                    if p[4] + p[1] * p[5] == g and (n == 1 or p[5] == stride):
                        p[1] += n
                        j = j_end
                        continue
            rows.append([first_vu + j, n, cu, kid, g, stride, unit_size])
            j = j_end

    vus = plan.verification_units
    if isinstance(vus, UnitRuns):
        first = 0
        for run in vus.runs:
            scope = circuit.scope_at(run.path)
            node = scope.fn.nodes[run.node_idx]
            form = "batch" if run.unrolled else node.form
            full = run.members // run.group  # units of exactly ``group`` members
            unit_size = run.group * run.member_size
            kid = kind_id(node.fn, form, unit_size)
            if full:
                emit(first, full, kid, run.lo, run.stride, unit_size)
            if run.count > full:  # ragged tail: one smaller unit
                tail_lo, tail_hi = run.interval(run.count - 1)
                emit(first + full, 1, kind_id(node.fn, form, tail_hi - tail_lo), tail_lo, run.stride, tail_hi - tail_lo)
            first += run.count
        n_v = first
    else:
        n_v = 0
        for v, unit in enumerate(vus):
            ivs = unit.intervals.intervals()
            if len(ivs) != 1:
                raise PopulationError(f"VU {v} is not one gate interval ({len(ivs)}): population/v2 describes run-shaped plans")
            lo, hi = ivs[0]
            parts = getattr(unit, "parts", None)
            if not parts or len(parts) != 1:
                raise PopulationError(f"VU {v} is not a single hierarchy part")
            part = parts[0]
            scope = circuit.scope_at(part.path)
            node = scope.fn.nodes[part.node_idx]
            kid = kind_id(node.fn, node.form, hi - lo)
            emit(v, 1, kid, lo, hi - lo, hi - lo)
            n_v = v + 1
    if sink is not None:
        sink.close()
    return kinds, rows, cu_ivs, n_v


def build(plan, circuit, *, program_digest: str | None, plan_name: str | None = None, label: str | None = None,
          prescribed_outside: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``population/v2`` document of a validated plan (validation is the caller's: pass only plans whose verdict is
    ``valid`` -- :func:`e2e.population.population` refuses otherwise; this builder does not re-validate).
    ``prescribed_outside`` (D2 PARTITION, decision 4): ``{"gate_intervals": [[lo, hi], ...], ...}`` -- root-Input gates in no
    VU (prescribed inputs consumed by nothing); recorded verbatim under ``prescribed_outside_partition`` and counted by
    :meth:`CompactPopulation.check` so the rows plus these gates tile the circuit.  Default ``None``: unchanged."""

    from veritor.plans.summary import PARTITION_DIGEST, partition_digest, plan_digest

    t = time.perf_counter()
    kinds, rows, cu_ivs, n_v = build_rows(plan, circuit)
    if n_v != len(plan.verification_units):
        raise PopulationError(f"rows cover {n_v} VUs but the plan has {len(plan.verification_units)}")
    pk, pd = plan_digest(plan)
    doc: dict[str, Any] = {
        "schema": POPULATION_V2, "plan_name": plan_name or plan.name, "instances": 1,
        "vu_count": n_v, "ru_count": len(plan.commitment_units), "per_instance": {"vu_count": n_v, "ru_count": len(plan.commitment_units)},
        "gates": circuit.size,
        "program_digest": program_digest, "plan_digest": str(pd), "plan_digest_kind": pk,
        "partition_digest": partition_digest(plan), "partition_digest_kind": PARTITION_DIGEST,
        "kinds": {"schema": KIND_SCHEMA, "key": "(stopped-node callee definition name, node form); 'batch' for a member of an adapter-unrolled batch",
                  "entries": [k.to_json() for k in kinds]},
        "row_fields": list(ROW_FIELDS),
        "rows": rows,
        "cu_intervals": cu_ivs,
        "label": label or "MEASURED (built from the validated compact plan: rows = UnitRuns split at CU boundaries and coalesced; ru = CU owning the unit's gates)",
        "seconds": round(time.perf_counter() - t, 3),
    }
    if prescribed_outside is not None:
        doc["prescribed_outside_partition"] = prescribed_outside
    doc["content_sha256"] = content_sha256(doc)
    return doc


def _outside_gates(doc: dict[str, Any]) -> int:
    """Gates of ``prescribed_outside_partition.gate_intervals`` (0 when absent)."""

    ivs = (doc.get("prescribed_outside_partition") or {}).get("gate_intervals") or []
    if not all(isinstance(iv, list) and len(iv) == 2 and all(type(x) is int for x in iv) and iv[0] < iv[1] for iv in ivs):
        raise PopulationError("prescribed_outside_partition.gate_intervals must be [lo, hi] int pairs")
    return sum(hi - lo for lo, hi in ivs)


def _outside_sides(doc: dict[str, Any]) -> frozenset[str]:
    """The sides ``prescribed_outside_partition`` applies to: its optional ``sides`` list (A1: ``["verification",
    "commitment"]`` for an ``inputs="prescribed"`` plan), default ``["verification"]`` (D2: the CU side covers every gate)."""

    sides = (doc.get("prescribed_outside_partition") or {}).get("sides") or ["verification"]
    if not isinstance(sides, list) or not set(sides) <= {"verification", "commitment"} or not sides:
        raise PopulationError("prescribed_outside_partition.sides must be a non-empty subset of [verification, commitment]")
    return frozenset(sides)


def prescribed_outside_record(intervals: Sequence[Sequence[int]], *, sides: Sequence[str] = ("verification", "commitment")) -> dict[str, Any]:
    """The ``prescribed_outside_partition`` block of an ``inputs="prescribed"`` canonical plan (A1, contract R2): the root
    ``Input`` gate intervals (``CanonicalPlan.prescribed_outside``), the sides that leave them uncovered, the gate count and
    the rule that produced them."""

    ivs = [[int(lo), int(hi)] for lo, hi in intervals]
    return {"gate_intervals": ivs, "gates": sum(hi - lo for lo, hi in ivs), "sides": list(sides),
            "rule": "every root Input batch (prompt token ids, weights incl. derived rope tables) -- canonical_plan(inputs='prescribed')",
            "check": "prescribed input values are checked against the declared input unconditionally (contract step 2), never sampled"}


def content_sha256(doc: dict[str, Any]) -> str:
    return sha256_hex(canonical_bytes({k: v for k, v in doc.items() if k not in ("content_sha256", "seconds")}))


# -- the compact population -----------------------------------------------------------------------------------------------------


class CompactPopulation:
    """``population/v2`` in memory: ``O(rows)`` state, ``O(log rows)`` lookups."""

    schema = POPULATION_V2

    def __init__(self, doc: dict[str, Any], *, path: Path | None = None) -> None:
        if doc.get("schema") != POPULATION_V2:
            raise PopulationError(f"schema {doc.get('schema')!r} != {POPULATION_V2!r}")
        for k in ("vu_count", "ru_count", "plan_digest", "program_digest", "partition_digest", "kinds", "rows", "cu_intervals", "gates", "content_sha256"):
            if k not in doc:
                raise PopulationError(f"population/v2 missing {k!r}")
        if doc.get("row_fields") != list(ROW_FIELDS):
            raise PopulationError(f"row_fields {doc.get('row_fields')!r} != {list(ROW_FIELDS)!r}")
        self.doc = doc
        self.path = path
        self.vu_count = int(doc["vu_count"])
        self.ru_count = int(doc["ru_count"])
        self.gates = int(doc["gates"])
        self.plan_digest = doc["plan_digest"]
        self.plan_digest_kind = doc.get("plan_digest_kind")
        self.program_digest = doc["program_digest"]
        self.partition_digest = doc["partition_digest"]
        self.instances = int(doc.get("instances", 1))
        self.content_sha256 = doc["content_sha256"]
        self.kinds = [Kind.from_json(k) for k in doc["kinds"]["entries"]]
        if [k.id for k in self.kinds] != list(range(len(self.kinds))):
            raise PopulationError("kind ids are not 0..n-1")
        rows = doc["rows"]
        if not all(isinstance(r, list) and len(r) == len(ROW_FIELDS) and all(type(x) is int for x in r) for r in rows):
            raise PopulationError("rows must be int lists of the row_fields")
        self.rows = rows
        self._first = [r[0] for r in rows]
        # per RU: its rows (VU order) and the cumulative unit counts inside the RU
        by_ru: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            by_ru.setdefault(r[2], []).append(i)
        self._ru_rows: list[list[int]] = [by_ru.get(c, []) for c in range(self.ru_count)]
        self._ru_prefix: list[list[int]] = []
        self._ru_sizes: list[int] = []
        for c in range(self.ru_count):
            pref = [0]
            for i in self._ru_rows[c]:
                pref.append(pref[-1] + rows[i][1])
            self._ru_prefix.append(pref)
            self._ru_sizes.append(pref[-1])
        self.cu_intervals: list[list[list[int]]] = doc["cu_intervals"]
        self._cu = _CuIndex(self.cu_intervals)

    # -- lookups -------------------------------------------------------------------------------------------------------------

    def _row(self, vu: int) -> tuple[list[int], int]:
        if not 0 <= vu < self.vu_count:
            raise IndexError(f"VU {vu} outside the population [0, {self.vu_count})")
        i = bisect.bisect_right(self._first, vu) - 1
        r = self.rows[i]
        j = vu - r[0]
        if j >= r[1]:
            raise PopulationError(f"VU {vu} is covered by no row (rows do not tile the population)")
        return r, j

    def ru_of(self, vu: int) -> int:
        return self._row(vu)[0][2]

    def kind_of(self, vu: int) -> Kind:
        return self.kinds[self._row(vu)[0][3]]

    def gate_interval(self, vu: int) -> tuple[int, int]:
        r, j = self._row(vu)
        lo = r[4] + j * r[5]
        return lo, lo + r[6]

    def ru_size(self, ru: int) -> int:
        return self._ru_sizes[ru]

    def vu_at(self, ru: int, k: int) -> int:
        """The ``k``-th VU (ascending index) of RU ``ru``; ``O(log rows of ru)``."""

        pref = self._ru_prefix[ru]
        if not 0 <= k < pref[-1]:
            raise IndexError(f"rank {k} outside RU {ru} ({pref[-1]} VUs)")
        i = bisect.bisect_right(pref, k) - 1
        row = self.rows[self._ru_rows[ru][i]]
        return row[0] + (k - pref[i])

    def vus_of_ru(self, ru: int) -> Iterator[int]:
        for i in self._ru_rows[ru]:
            r = self.rows[i]
            yield from range(r[0], r[0] + r[1])

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rows:
            name = self.kinds[r[3]].name
            counts[name] = counts.get(name, 0) + r[1]
        return counts

    def ru_of_all(self) -> Iterator[int]:
        """``ru_of[vu]`` streamed in VU order (tests: equality with population/v1)."""

        for r in self.rows:
            for _ in range(r[1]):
                yield r[2]

    # -- self-check ----------------------------------------------------------------------------------------------------------

    def check(self, *, digest: bool = True) -> dict[str, Any]:
        """Raise :class:`PopulationError` unless the file describes the partition it claims.  Always: content sha256, rows tile
        ``[0, vu_count)`` in order, rows are gate-monotone and tile ``[0, gates)`` (together with ``prescribed_outside_partition``
        when present) with the CU side, every row inside its RU's intervals, RU sizes sum to ``vu_count``.  ``digest=True``:
        re-derive ``partition_digest`` (``O(N_V)`` sha updates)."""

        problems: list[str] = []
        if content_sha256(self.doc) != self.content_sha256:
            problems.append("content_sha256 does not match the document (edited after it was written)")
        expected_vu = 0
        prev_lo = -1
        gates = 0
        for r in self.rows:
            first, count, ru, kind, lo, stride, size = r
            if first != expected_vu:
                problems.append(f"row at VU {first} does not follow VU {expected_vu}: rows do not tile the population")
                break
            if count < 1 or size < 1 or stride < size or not 0 <= kind < len(self.kinds) or not 0 <= ru < self.ru_count:
                problems.append(f"degenerate row {r}")
                break
            if lo <= prev_lo:
                problems.append(f"row {r} is not in gate order (a canonical plan lays its units out by gate)")
                break
            hi = lo + (count - 1) * stride + size
            iv = self._cu.interval_at(lo)
            if iv is None or iv[2] != ru or hi > iv[1]:
                problems.append(f"row {r} is not inside commitment unit {ru} (owner {None if iv is None else iv[2]})")
                break
            # strided rows interleave with other rows' units; disjointness + coverage are established by the total gate count
            # here and, in full mode, by the partition digest re-derived unit by unit
            gates += count * size
            prev_lo = lo
            expected_vu = first + count
        if expected_vu != self.vu_count:
            problems.append(f"rows cover {expected_vu} VUs, vu_count is {self.vu_count}")
        outside = _outside_gates(self.doc)
        if gates + outside != self.gates:
            problems.append(f"rows cover {gates} gates" + (f" + {outside} prescribed outside the partition" if outside else "")
                            + f", the circuit has {self.gates}: not a partition of the program")
        if sum(self._ru_sizes) != self.vu_count:
            problems.append("RU sizes do not sum to vu_count")
        cu_gates = sum(hi - lo for ivs in self.cu_intervals for lo, hi in ivs)
        outside_cu = outside if "commitment" in _outside_sides(self.doc) else 0
        if cu_gates + outside_cu != self.gates:
            problems.append(f"CU intervals cover {cu_gates} gates" + (f" + {outside_cu} prescribed outside the commitment side" if outside_cu else "")
                            + f", the circuit has {self.gates}")
        if len(self.cu_intervals) != self.ru_count:
            problems.append("cu_intervals count != ru_count")
        if problems:
            raise PopulationError("; ".join(problems))
        out: dict[str, Any] = {"rows": len(self.rows), "kinds": len(self.kinds), "vu_count": self.vu_count, "ru_count": self.ru_count, "digest_checked": digest,
                               "prescribed_outside_gates": outside}
        if digest:
            t = time.perf_counter()
            d = self.recompute_partition_digest()
            out["digest_seconds"] = round(time.perf_counter() - t, 3)
            if d != self.partition_digest:
                raise PopulationError(f"partition digest re-derived from the rows {d[:16]} != claimed {self.partition_digest[:16]}: not the partition it claims")
        return out

    def recompute_partition_digest(self) -> str:
        """``partition-intervals/v1`` from rows + CU intervals (the same stream as :func:`veritor.plans.summary.partition_digest`)."""

        h = hashlib.sha256(b"veritor/plans/partition-intervals/v1\0")
        h.update(b"v")
        for first, count, ru, kind, lo, stride, size in self.rows:
            for j in range(count):
                g = lo + j * stride
                h.update(f"{g}:{g + size};|".encode())
        h.update(b"c")
        for ivs in self.cu_intervals:
            for lo, hi in ivs:
                h.update(f"{lo}:{hi};".encode())
            h.update(b"|")
        return h.hexdigest()

    # -- io ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path, *, check: str = "fast") -> CompactPopulation:
        """``check``: ``"none"`` | ``"fast"`` (O(rows) structure + content sha) | ``"full"`` (+ partition digest re-derivation)."""

        path = Path(path)
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise PopulationError(f"{path}: {e}") from e
        pop = cls(doc, path=path)
        if check == "fast":
            pop.check(digest=False)
        elif check == "full":
            pop.check(digest=True)
        elif check != "none":
            raise ValueError(check)
        return pop

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.doc, sort_keys=True, separators=(",", ":")) + "\n")
        return path


def write(doc: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    return path


# -- equality with the dense population -----------------------------------------------------------------------------------------


def compare_with_v1(compact: CompactPopulation, v1: dict[str, Any], *, plan=None) -> dict[str, Any]:
    """Every field the sampler reads, dense vs compact: counts, digests, ``ru_of`` for every VU, and (given the plan) every
    unit's gate interval.  Returns the comparison; raises on the first disagreement."""

    if v1.get("schema") != POPULATION_V1:
        raise PopulationError(f"v1 schema {v1.get('schema')!r}")
    for k in ("vu_count", "ru_count", "plan_digest", "program_digest", "partition_digest"):
        if v1.get(k) != getattr(compact, k):
            raise PopulationError(f"{k}: v1 {v1.get(k)!r} != v2 {getattr(compact, k)!r}")
    ru_of = v1["ru_of"]
    if len(ru_of) != compact.vu_count:
        raise PopulationError("v1 ru_of length != vu_count")
    mism = 0
    for vu, (a, b) in enumerate(zip(ru_of, compact.ru_of_all(), strict=True)):
        if a != b:
            mism += 1
            if mism <= 5:
                raise PopulationError(f"ru_of[{vu}]: v1 {a} != v2 {b}")
    # random-access agrees with the stream
    for vu in range(0, compact.vu_count, max(1, compact.vu_count // 997)):
        if compact.ru_of(vu) != ru_of[vu]:
            raise PopulationError(f"ru_of({vu}) lookup {compact.ru_of(vu)} != v1 {ru_of[vu]}")
    intervals_checked = 0
    if plan is not None:
        from veritor.plans.runs import UnitRuns

        vus = plan.verification_units
        for vu in range(compact.vu_count):
            if isinstance(vus, UnitRuns):
                iv = vus.interval(vu)
            else:
                ivs = vus[vu].intervals.intervals()
                iv = (ivs[0][0], ivs[-1][1])
            if compact.gate_interval(vu) != tuple(iv):
                raise PopulationError(f"gate_interval({vu}) {compact.gate_interval(vu)} != plan {iv}")
            intervals_checked += 1
    return {"vu_count": compact.vu_count, "ru_count": compact.ru_count, "ru_of_equal": True, "intervals_checked": intervals_checked,
            "v1_bytes": len(canonical_bytes(v1)), "v2_bytes": len(canonical_bytes(compact.doc)), "rows": len(compact.rows), "kinds": len(compact.kinds)}


# -- CLI -----------------------------------------------------------------------------------------------------------------------------


def _build_case(case: str, lp: int, steps: int, spec: str, *, dense_too: bool, validate: bool = True):
    from e2e import population as popmod

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    if case == "TINY":
        config, lp, steps = dict(popmod.TINY), popmod.TINY_LP, popmod.TINY_STEPS
    else:
        from verity_vllm.registry import b1

        config = dict(getattr(b1, f"{case}_CONFIG"))
    program, adapted, circuit, pdigest = popmod.build(config, lp, steps)
    plan, meta = popmod.make_plan(spec, adapted, circuit)
    if not validate:
        # [L1 RECORD 2026-09-10] rows only: the verdict is NOT produced here (whole B0's streamed validation is a multi-pod job; its verdict is
        # cited from data/rec/p1-b0-lp32s15-validated.json by the caller).  The rows are the plan's UnitRuns; the partition digest is
        # re-derived from them and must equal the validated one -- that equality, not this flag, is what binds the file to the verdict.
        return program, adapted, circuit, pdigest, plan, meta, None
    from veritor.plans.validate import validate_plan

    v = validate_plan(circuit, plan, retain=False, **popmod.validate_kwargs(meta))
    if v.status != "valid":
        raise PopulationError(f"{spec}: {v.status} {v.codes}")
    return program, adapted, circuit, pdigest, plan, meta, v


def outside_record_from_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    """The ``prescribed_outside_partition`` block for a plan built by ``e2e.population.make_plan`` with ``/inputs=prescribed``
    (``None`` for the historical construction)."""

    if meta.get("inputs") != "prescribed":
        return None
    return prescribed_outside_record(meta["prescribed_outside"], sides=meta["prescribed_outside_sides"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build + validate (streamed) a plan and write population.compact.json")
    b.add_argument("--case", default="TINY")
    b.add_argument("--lp", type=int, default=4)
    b.add_argument("--steps", type=int, default=8)
    b.add_argument("--plan", default="P1/cu=whole")
    b.add_argument("--out", required=True)
    b.add_argument("--with-v1", action="store_true", help="also write population.json (v1, O(N_V)) from a retained validation and compare (TINY only)")
    b.add_argument("--no-validate", action="store_true", help="[L1 RECORD] rows only, no validate_plan here (whole B0: the verdict is cited from the "
                                                              "COMPACT-P1 artifact; the re-derived partition digest must equal its digest)")
    b.add_argument("--expect-partition-digest", default=None, help="[L1 RECORD] refuse unless the re-derived partition digest equals this hex")
    c = sub.add_parser("check", help="self-check a population.compact.json")
    c.add_argument("path")
    c.add_argument("--fast", action="store_true", help="skip the O(N_V) partition-digest re-derivation")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            program, adapted, circuit, pdigest, plan, meta, v = _build_case(a.case, a.lp, a.steps, a.plan, dense_too=a.with_v1, validate=not a.no_validate)
            doc = build(plan, circuit, program_digest=pdigest, prescribed_outside=outside_record_from_meta(meta))
            if a.expect_partition_digest and doc.get("partition_digest") != a.expect_partition_digest:
                raise PopulationError(f"re-derived partition digest {doc.get('partition_digest')} != expected {a.expect_partition_digest}")
            out = Path(a.out)
            write(doc, out / COMPACT_FILE)
            pop = CompactPopulation.load(out / COMPACT_FILE, check="full")
            print(f"[population_v2] {a.case} {a.plan}: N_V={pop.vu_count} N_R={pop.ru_count} rows={len(pop.rows)} kinds={len(pop.kinds)} "
                  f"partition {pop.partition_digest[:16]} content {pop.content_sha256[:16]} -> {out / COMPACT_FILE}")
            if a.with_v1:
                from e2e import population as popmod
                from veritor.plans.validate import validate_plan

                vr = validate_plan(circuit, plan)
                popmod.write_maps(out, adapted.program, circuit, plan, vr, pdigest, {"vu_level": meta["vu_level"], "cu_strategy": meta["cu_strategy"]}, tables=False)
                cmp = compare_with_v1(pop, json.loads((out / "population.json").read_text()), plan=plan)
                print(f"[population_v2] dense == compact: {cmp}")
            return 0
        pop = CompactPopulation.load(Path(a.path), check="none")
        res = pop.check(digest=not a.fast)
        print(f"[population_v2] OK {a.path}: {res}")
        return 0
    except PopulationError as e:
        print(f"[population_v2] REFUSED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
