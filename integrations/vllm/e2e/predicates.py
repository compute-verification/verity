"""[SAMPLER] Selected VU identity -> required proof predicate, through the plan/kind machinery (GUIDANCE §57.15, deliverable 3).

A selected VU (``selected.json`` of ``e2e.partition_sample``) is a unit of the validated scalar partition: a kind id (the kind
table of ``population/v2``: callee definition + node form + parameters), a gate interval, a commitment unit.  Its *required
predicate* is: the kind program (the callee definition) holds between the VU's boundary inputs and its boundary output, where
those values are read from the committed store.  This module answers, for one VU, three questions and refuses when it cannot:

1. **Which committed leaves must be opened?**  The VU's boundary (``veritor.plans.boundary.boundary_of`` of the plan's unit) --
   its input gates and output gates -- located in the *operator-level* partition of the same program (``P3/cu=block``: one unit
   per operator call ``Gemm_v1{..}``, ``RMSNorm*``, ``Attention_v1{..}``, ...).  Today's store commits operator boundaries (the
   adapter's rows: operand rows, weight rows, output rows) and nothing inside an operator; so a VU's leaf is *committed* iff it is
   a boundary gate of its operator unit (an input the operator reads from outside, or an output another operator / the root
   reads).  This is decided on the plans' boundaries, not on kind names.
2. **Which relation must be checked?**  The kind program itself (``kind.fn``): the callee definition of the stopped node, with
   its parameters -- the same object the census and the Table 3(a) kind table name.
3. **Which checker serves it?**  A verifier-side registry (:data:`CHECKERS`, the same kind of policy as ``approved_guests.json``)
   keyed by the kind program's *definition* (base name + parameter constraints) and by the interface status, naming the statement
   format / checker family and the approved ELF.  A kind program without an entry has no approved checker.

Outcomes (``Predicate.status``):

* ``SUPPORTED`` -- the interface is committed and an approved checker takes exactly this unit (today: a GEMM output coordinate,
  ``GemmCoordinate_v*{K}`` over the committed x row, weight row and output word: statement ``GemmCoordinate_v2``).
* ``ADAPTER`` -- the interface is committed (or its output is) and the existing path checks a *different* unit that maps
  one-to-one onto this VU (the RMSNorm output coordinate's cone; an embedding coordinate checked through its committed row).
  The adapter's unit, what it additionally checks and what it does NOT establish are spelled out in ``adapter``; the report says
  so explicitly.  Nothing is substituted silently.
* ``REFUSED`` -- ``code = PREDICATE_UNSUPPORTED``: the interface is not committed in today's store (every attention interior kind:
  ``DotBf16_v2{K=64}``, ``F32FmaSubFtz``, ``MufuEx2Ftz``, ``F32Max``, ``F32AddFtz``, ``F32MulFtz``, ``Fa2InvSum``, guards/consts
  inside attention -- ``interface not committed; pending S55``; the norm / rope / silu / argmax interiors likewise), or the
  interface is committed but no approved checker exists for the kind program.  A many-to-one "adapter" (checking an operator's
  row relation for one of its interior VUs) is NOT offered: it is a different sampling law (the user's 15:36 PT ruling).

Nothing here reads a capture or a record; nothing in ``run_composed`` is routed through it.  ``python -m e2e.predicates --case
TINY --plan P1/cu=operator --selected DIR/selected.json`` maps a selection; ``--table`` prints the per-kind table for §3.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e.population_v2 import CompactPopulation, Kind, PopulationError, build_rows  # noqa: E402

PREDICATE_UNSUPPORTED = "PREDICATE_UNSUPPORTED"
SCHEMA = "veritor/required-predicates/v1"


class PredicateError(Exception):
    """A REFUSAL of the mapping (``code`` names why); never a fallback."""

    def __init__(self, code: str, reason: str, vu: int | None = None) -> None:
        super().__init__(f"{code}: {reason}" + (f" (VU {vu})" if vu is not None else ""))
        self.code, self.reason, self.vu = code, reason, vu


# -- the checker registry (verifier-side policy) ---------------------------------------------------------------------------------

APPROVED_GUESTS = HERE / "approved_guests.json"
GEMM_COORDINATE_ELF = "73081b7b2c9305a93f243a2137f71c851fb24bddf5e6edfbeb398acb2f309029"  # approved 'reproduced' (approved_guests.json)


def _params(name: str) -> tuple[str, dict[str, str]]:
    """``'GemmCoordinate_v2{K=576,DOT=AmpereBF16TcDot16_v1}'`` -> ``('GemmCoordinate_v2', {'K': '576', 'DOT': '...'})``; the
    registry's ``{...}`` parameter block of a definition name (``Const32[0x..]_v1`` has none)."""

    m = re.match(r"^([^{]+)\{(.*)\}$", name)
    if not m:
        return name, {}
    params: dict[str, str] = {}
    for part in m.group(2).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip()] = v.strip()
    return m.group(1), params


def _versionless(base: str) -> str:
    return re.sub(r"_v\d+$", "", base)


@dataclass(frozen=True)
class Checker:
    """One registry entry: what an approved checker takes.  ``kind_base`` is the kind program's definition (versionless base name);
    ``operator_base`` restricts the operator-level unit the VU must lie in (adapters); ``requires`` is the interface status."""

    name: str
    kind_base: str
    statement: str
    elf_sha256: str | None
    statement_format: int | None
    requires: str  # "committed" (inputs + output) | "output-committed"
    unit: str  # "same" (the VU itself) | "cone" | "row"
    proof: str  # "zk" (an approved ELF proves it) | "verifier" (the verifier recomputes it over openings; not a proof)
    operator_base: str | None = None
    constraint: str | None = None
    note: str = ""

    def accepts(self, kind: Kind, operator_fn: str | None) -> tuple[bool, str | None]:
        base, params = _params(kind.fn)
        if _kind_base(kind) != self.kind_base:
            return False, None
        if self.operator_base is not None and (operator_fn is None or _versionless(_params(operator_fn)[0]) != self.operator_base):
            return False, None
        if self.constraint == "K%16==0":
            k = params.get("K")
            if k is None or int(k) % 16:
                return False, f"K={k} is not a multiple of 16 (the statement chains K/16 tc_dot16 steps)"
        return True, None


def _kind_base(kind: Kind) -> str:
    """The kind program's definition family: versionless base name, ``GatherBf16x<V>`` collapsed to ``GatherBf16x`` (V is a parameter
    written into the name by the registry)."""

    base = _versionless(_params(kind.fn)[0])
    m = re.match(r"^(GatherBf16x)\d+$", base)
    return m.group(1) if m else base


CHECKERS: tuple[Checker, ...] = (
    Checker("gemm-coordinate", "GemmCoordinate", "GemmCoordinate_v2 (statement format 2/3: committed x row, weight row, output word)",
            GEMM_COORDINATE_ELF, 2, requires="committed", unit="same", proof="zk", constraint="K%16==0",
            note="the unit proved IS the selected VU: one output coordinate of one GEMM call over its committed operand rows; approved ELF 73081b7b "
                 "(approved_guests.json, 'reproduced'); K=1536 was priced in Table 3(a) on the frozen ELF 08298f55 (NOT in approved_guests) -- "
                 "the approved ELF's K-generality is not verified here"),
    Checker("input-equation", "Input", "input equation: the committed leaf's opening equals its authorized source (weights: S8 weight_input_check against "
                                       "the authorized safetensors; prompt / boundary words: the first boundary) -- run_composed input_equations / "
                                       "boundary_input_equations",
            None, None, requires="committed", unit="same", proof="verifier", operator_base="Input",
            note="the unit checked IS the selected leaf; a verifier comparison of an opening, not a zk proof"),
    Checker("rmsnorm-cone", "F2fpBf16", "norm cone (statement format 2 over the committed input row and the output word; approved ELF 73081b7b, "
                                         "Table 3(a) row norm-out-cast)",
            GEMM_COORDINATE_ELF, 2, requires="output-committed", unit="cone", proof="zk", operator_base="RMSNormTriton",
            note="ADAPTER: the proved unit is the CONE of the selected output coordinate (its F32Mul/F32Add reduction over the whole row, the rsqrt, "
                 "the scale), not the selected F2fpBf16 unit alone; one cone per output coordinate, so the map is one-to-one on the norm's output "
                 "coordinates and the Bernoulli law over them is preserved; the cone check implies the VU's relation but also checks the interior "
                 "VUs of that row, whose own selection law it does not implement"),
    Checker("rmsnorm-cone", "F2fpBf16", "norm cone (statement format 2 over the committed input row and the output word; approved ELF 73081b7b, "
                                         "Table 3(a) row norm-out-cast)",
            GEMM_COORDINATE_ELF, 2, requires="output-committed", unit="cone", proof="zk", operator_base="RMSNormFusedCuda",
            note="ADAPTER (as for RMSNormTriton): the cone of the selected output coordinate"),
    Checker("embedding-row", "GatherBf16x", "verifier row check: the committed embedding row of the committed token id equals the committed output row "
                                            "(run_composed: 'checked by the verifier, not proved')",
            None, None, requires="committed", unit="row", proof="verifier", operator_base="Embedding",
            note="ADAPTER: the existing path opens and compares the WHOLE row; the selected coordinate is one of its words (one-to-one on coordinates: "
                 "(row, j)); not a zk proof -- a verifier recomputation"),
)

# operator-level units whose interior definition is an OPEN security choice: the refusal names it
OPERATOR_PENDING: dict[str, str] = {"Attention": "pending S55 (the attention VU definition is unresolved: security-choices-2026-09-09.md S55; no head-row cone is substituted)"}


# -- the operator-level partition: today's committed interface -------------------------------------------------------------------


class Operators:
    """The ``P3/cu=block`` partition of the same program (one unit per operator call) with the validator's boundaries: the
    committed interface of today's store, decided structurally (a gate is committed iff it is a boundary gate of its operator)."""

    def __init__(self, circuit, plan_p3, verdict_p3) -> None:
        if verdict_p3.status != "valid" or not verdict_p3.retained:
            raise PredicateError("OPERATORS_INVALID", f"operator-level plan {plan_p3.name!r} is {verdict_p3.status} / retained={verdict_p3.retained}")
        kinds, rows, _cu, n = build_rows(plan_p3, circuit)
        if n != len(plan_p3.verification_units) or any(r[1] != 1 for r in rows):
            raise PredicateError("OPERATORS_INVALID", "operator-level units are not one row each")
        self.plan, self.verdict, self.kinds = plan_p3, verdict_p3, kinds
        self.rows = rows  # [first_vu, 1, cu, kind, gate, stride, size]
        self._starts = [r[4] for r in rows]
        self._by_vu = sorted(range(len(rows)), key=lambda i: rows[i][0])
        self.name = plan_p3.name

    def unit_of_gate(self, g: int) -> int:
        i = bisect_right(self._starts, g) - 1
        if i < 0 or not (self.rows[i][4] <= g < self.rows[i][4] + self.rows[i][6]):
            raise PredicateError("GATE_UNOWNED", f"gate {g} lies in no operator-level unit")
        return self.rows[i][0]

    def kind(self, unit: int) -> Kind:
        return self.kinds[self.rows[self._by_vu[unit]][3]]

    def interval(self, unit: int) -> tuple[int, int]:
        r = self.rows[self._by_vu[unit]]
        return r[4], r[4] + r[6]

    def boundary(self, unit: int):
        return self.verdict.vu_boundaries[unit]


# -- predicates --------------------------------------------------------------------------------------------------------------------


@dataclass
class Predicate:
    vu_index: int
    ru_index: int
    kind: dict[str, Any]
    gate_interval: list[int]
    operator: dict[str, Any]
    interface: dict[str, Any]
    status: str
    code: str | None = None
    reason: str | None = None
    obligation: dict[str, Any] | None = None
    adapter: dict[str, Any] | None = None
    partition_digest: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class Predicates:
    """Maps VUs of a validated scalar plan (``population/v2`` + the plan) to required predicates."""

    def __init__(self, circuit, plan, pop: CompactPopulation, operators: Operators, *, verdict=None, checkers: tuple[Checker, ...] = CHECKERS) -> None:
        if pop.vu_count != len(plan.verification_units):
            raise PredicateError("POPULATION_MISMATCH", f"population has {pop.vu_count} VUs, plan {len(plan.verification_units)}")
        self.circuit, self.plan, self.pop, self.ops, self.checkers = circuit, plan, pop, operators, checkers
        self.verdict = verdict if (verdict is not None and getattr(verdict, "retained", False)) else None

    def vu_boundary(self, vu: int):
        if self.verdict is not None:
            return self.verdict.vu_boundaries[vu]
        from veritor.plans.boundary import boundary_of

        return boundary_of(self.plan.verification_units[vu])[0]

    def predicate(self, vu: int) -> Predicate:
        if not 0 <= vu < self.pop.vu_count:
            raise PredicateError("VU_OUT_OF_RANGE", f"VU {vu} outside [0, {self.pop.vu_count})", vu)
        kind = self.pop.kind_of(vu)
        lo, hi = self.pop.gate_interval(vu)
        b = self.vu_boundary(vu)
        op_unit = self.ops.unit_of_gate(lo)
        op_kind = self.ops.kind(op_unit)
        ob = self.ops.boundary(op_unit)
        op_lo, op_hi = self.ops.interval(op_unit)
        ins = list(b.inputs)
        outs = list(b.outputs)
        ins_c = [g for g in ins if g in ob.inputs or not (op_lo <= g < op_hi)]  # read from outside the operator (its committed operands / weights / inputs)
        outs_c = [g for g in outs if g in ob.outputs]  # exported by the operator (committed output rows / designated outputs)
        status_if = "committed" if len(ins_c) == len(ins) and len(outs_c) == len(outs) else ("output-committed" if len(outs_c) == len(outs) else "interior")
        pred = Predicate(vu, self.pop.ru_of(vu), kind.to_json(), [lo, hi],
                         {"unit": op_unit, "fn": op_kind.fn, "gate_interval": [op_lo, op_hi], "partition": self.ops.name},
                         {"inputs": len(ins), "inputs_committed": len(ins_c), "outputs": len(outs), "outputs_committed": len(outs_c), "status": status_if,
                          "in_bits": b.in_bits, "out_bits": b.out_bits},
                         status="REFUSED", partition_digest=self.pop.partition_digest)
        for c in self.checkers:
            ok, why = c.accepts(kind, op_kind.fn)
            if not ok:
                if why:
                    pred.code, pred.reason = PREDICATE_UNSUPPORTED, f"checker {c.name} does not take this kind program: {why}"
                continue
            if c.requires == "committed" and status_if != "committed":
                continue
            if c.requires == "output-committed" and status_if == "interior":
                continue
            pred.status = "SUPPORTED" if c.unit == "same" else "ADAPTER"
            pred.code = None
            pred.obligation = {
                "open": [{"gate": g, "width": b.inputs[g], "role": "operand", "committed_as": "operator input (operand / weight / input row)"} for g in ins]
                        + [{"gate": g, "width": b.outputs[g], "role": "output", "committed_as": "operator output word"} for g in outs],
                "relation": {"kind_program": kind.fn, "form": kind.form, "kind_id": kind.id, "parameters": _params(kind.fn)[1]},
                "checker": {"name": c.name, "statement": c.statement, "elf_sha256": c.elf_sha256, "statement_format": c.statement_format, "unit": c.unit,
                            "proof": c.proof},
            }
            if c.unit != "same":
                pred.adapter = {"unit": c.unit, "operator": op_kind.fn, "note": c.note,
                                "interior_inputs": [g for g in ins if g not in ins_c],
                                "caveat": "the checked unit differs from the selected VU; it maps one-to-one onto this VU's output coordinate, so the "
                                          "sampling law over selected VUs is preserved, but the predicate checked is the adapter's, not the VU's alone"}
            return pred
        # no checker took it
        base = _versionless(_params(op_kind.fn)[0])
        if status_if != "committed":
            pend = OPERATOR_PENDING.get(base)
            pred.reason = (f"interface not committed; {pend}" if pend else
                           f"interface not committed (the VU is interior to operator {op_kind.fn}: {len(ins) - len(ins_c)} of {len(ins)} inputs and "
                           f"{len(outs) - len(outs_c)} of {len(outs)} outputs are not boundary values of the operator; today's store commits operator "
                           f"boundaries only); no one-to-one adapter is registered")
        else:
            pred.reason = pred.reason or f"interface committed but no approved checker takes kind program {kind.fn!r} ({kind.form}) in operator {op_kind.fn}"
        pred.code = PREDICATE_UNSUPPORTED
        return pred

    def for_selected(self, selected: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
        """Map every VU of a ``selected.json``; with ``strict`` raise :class:`PredicateError` at the first REFUSED VU (the loud refusal)."""

        if selected.get("partition_digest") != self.pop.partition_digest:
            raise PredicateError("PARTITION_MISMATCH", f"selected.json partition {str(selected.get('partition_digest'))[:12]} != population {str(self.pop.partition_digest)[:12]}")
        preds = []
        for row in selected["selected"]:
            p = self.predicate(int(row["vu_index"]))
            if p.ru_index != row.get("ru_index"):
                raise PredicateError("RU_MISMATCH", f"selected.json names RU {row.get('ru_index')} for VU {p.vu_index}; the population says {p.ru_index}", p.vu_index)
            if strict and p.status == "REFUSED":
                raise PredicateError(p.code or PREDICATE_UNSUPPORTED, p.reason or "", p.vu_index)
            preds.append(p)
        by_status = Counter(p.status for p in preds)
        return {"schema": SCHEMA, "partition_digest": self.pop.partition_digest, "count": len(preds), "by_status": dict(by_status),
                "by_kind": kind_table(preds), "predicates": [p.to_json() for p in preds]}


def kind_table(preds: list[Predicate]) -> dict[str, dict[str, Any]]:
    """Per (kind, operator) outcome counts -- the §3 table's rows."""

    table: dict[str, dict[str, Any]] = {}
    for p in preds:
        key = f"{p.kind['fn']}|{p.kind['form']} in {_params(p.operator['fn'])[0]}"
        t = table.setdefault(key, {"kind": f"{p.kind['fn']}|{p.kind['form']}", "operator": _params(p.operator["fn"])[0], "count": 0, "outcomes": {}})
        t["count"] += 1
        o = t["outcomes"].setdefault(p.status, {"count": 0, "reason": p.reason, "checker": (p.obligation or {}).get("checker", {}).get("name"),
                                                 "proof": (p.obligation or {}).get("checker", {}).get("proof"), "interface": p.interface["status"]})
        o["count"] += 1
    return dict(sorted(table.items()))


def full_table(preds_obj: Predicates) -> dict[str, dict[str, Any]]:
    """Every VU of the plan (TINY: 53k, seconds) -> the per-(kind, operator) table."""

    return kind_table([preds_obj.predicate(vu) for vu in range(preds_obj.pop.vu_count)])


# -- CLI ---------------------------------------------------------------------------------------------------------------------------


def build_context(case: str, plan_spec: str):
    """TINY only locally (< 2 GB): program + scalar plan (validated, retained) + population/v2 + operator-level plan (validated)."""

    from e2e import population as popmod
    from e2e.population_v2 import build as build_v2
    from veritor.plans.validate import validate_plan

    if case != "TINY":
        raise SystemExit("predicates: only TINY is built locally (whole B0 is a pod job)")
    program, adapted, circuit, pdigest = popmod.build(popmod.TINY, popmod.TINY_LP, popmod.TINY_STEPS)
    plan, _ = popmod.make_plan(plan_spec, adapted, circuit)
    v = validate_plan(circuit, plan)
    pop = CompactPopulation(build_v2(plan, circuit, program_digest=pdigest))
    p3, _ = popmod.make_plan("P3/cu=block", adapted, circuit)
    v3 = validate_plan(circuit, p3)
    return Predicates(circuit, plan, pop, Operators(circuit, p3, v3), verdict=v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="TINY")
    ap.add_argument("--plan", default="P1/cu=operator")
    ap.add_argument("--selected", default=None, help="selected.json of e2e.partition_sample")
    ap.add_argument("--table", action="store_true", help="the per-kind table over EVERY VU of the plan")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-strict", action="store_true", help="list REFUSED predicates instead of stopping at the first")
    a = ap.parse_args(argv)
    P = build_context(a.case, a.plan)
    try:
        if a.table:
            doc: dict[str, Any] = {"schema": SCHEMA + "/kind-table", "case": a.case, "plan": a.plan, "partition_digest": P.pop.partition_digest,
                                   "vu_count": P.pop.vu_count, "table": full_table(P)}
            for k, t in doc["table"].items():
                outs = ", ".join(f"{s} x{o['count']}" + (f" [{o['checker']}]" if o.get("checker") else "") for s, o in t["outcomes"].items())
                print(f"{k:<70} n={t['count']:>6}  {outs}")
        else:
            if not a.selected:
                raise SystemExit("--selected or --table")
            doc = P.for_selected(json.loads(Path(a.selected).read_text()), strict=not a.no_strict)
            print(f"[predicates] {doc['count']} selected VUs: {doc['by_status']}")
    except PredicateError as e:
        print(f"[predicates] REFUSED {e}")
        return 1
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
