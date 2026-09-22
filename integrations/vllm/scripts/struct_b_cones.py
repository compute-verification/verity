#!/usr/bin/env python
"""Ancestor cones of attention OUTPUT coordinates in the compact plan IR (lane STRUCT-B, 2026-09-10).

S55 option (b) makes one attention output coordinate ``o[q, h, d]`` a verification unit.  Whether that is a
DISJOINT partition of the existing circuit is a structural fact of the program: two coordinates ``d``, ``d'`` of
the same (layer, query position, head) share the score / softmax gates, so their ancestor cones OVERLAP.  This
script measures the overlap on the actual IR (``verity_vllm.registry.b1`` bodies adapted by
``veritor.plans.verity_ir_adapter.adapt(compact=True)``): for every ``AttentionHead`` call instance selected, it
enumerates the ancestor cone of each requested output coordinate *inside the head composite* (gates strictly
inside the call's region; the walk stops at the composite's incoming values -- the q row, the K rows, the V rows
-- and at nothing else, so constants emitted inside the body are part of the cones), and reports

* gate count per cone, gates SHARED by every requested coordinate vs EXCLUSIVE to each, per primitive kind
  (``AmpereBF16TcDot16`` split into the q.k dots inside ``DotBf16`` and the P.V accumulate steps),
* the P1 (coordinate-level) units those gates belong to (``DotBf16{K}`` is one unit of K/16 + 1 gates),
* the incoming committed values of each cone: which q coordinates, which K rows/coordinates, which V rows and
  -- for coordinate ``d`` -- which V column(s),
* the outgoing value (one ``F2fpBf16`` gate, 16 bits),
* the (b2) owner interface: if one coordinate owned every shared gate, the shared values the other
  coordinates' exclusive gates read (count and bits, per kind), and
* the (b3) duplication factor: sum over coordinates of |cone_d| against |union of cones|.

Everything printed is MEASURED on the constructed program; nothing here changes the IR, a plan or a digest.

    PYTHONPATH=. python scripts/struct_b_cones.py --shape tiny --out docs/data/struct-b/two-coord-tiny.json
    PYTHONPATH=. python scripts/struct_b_cones.py --shape head:T=47,D=64 --coords 0,1,63 --out ...
    PYTHONPATH=. python scripts/struct_b_cones.py --shape head:T=128,D=64 --all-coords --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SCHEMA = "veritor/struct-b/two-coord-cones/v1"


# -- building -----------------------------------------------------------------------------------------------------------------


def build_head(T: int, D: int):
    """A bare ``AttentionHead_v1{T, D}`` program (q, K rows, V rows are Input gates)."""
    from verity.ir import Program, bind
    from verity_vllm.registry import b1
    from veritor.plans.verity_ir_adapter import adapt

    program = Program(bind(b1.AttentionHead, T=T, D=D))
    adapted = adapt(program, compact=True)
    return program, adapted, adapted.program.circuit()


def build_tiny():
    from e2e import population as popmod

    program, adapted, circuit, _ = popmod.build(popmod.TINY, popmod.TINY_LP, popmod.TINY_STEPS, digest=False)
    return program, adapted, circuit


def head_calls(circuit) -> list:
    """Every ``AttentionHead*`` call instance (batch members are call instances) in canonical order."""
    return [c for c in circuit.calls() if not c.is_primitive and c.name.startswith("AttentionHead")]


# -- cones ----------------------------------------------------------------------------------------------------------------------


def base_name(name: str) -> str:
    return name.split("{")[0]


class HeadCones:
    """Ancestor cones of the output coordinates of one ``AttentionHead`` composite: a call instance inside a larger
    program (``from_call``), or the root-inlined body of a bare ``AttentionHead`` program (``from_root``)."""

    def __init__(self, circuit, name: str, lo: int, hi: int, outputs: list[int], arg_gates: list[int], path=()) -> None:
        self.circuit, self.name, self.path = circuit, name, tuple(path)
        self.lo, self.hi = lo, hi
        self.outputs = list(outputs)
        self.D = len(self.outputs)
        self._gate_cache: dict[int, Any] = {}
        # AttentionHead(q[D], k[T][D], v[T][D]): the parameter leaves are flat in that order.
        D, n = self.D, len(arg_gates)
        T = (n - D) // (2 * D)
        assert D + 2 * T * D == n, (n, D)
        self.T = T
        self.param_leaves: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for i, g in enumerate(arg_gates):
            if i < D:
                self.param_leaves[g].append(("q", i))
            elif i < D + T * D:
                self.param_leaves[g].append(("k", i - D))
            else:
                self.param_leaves[g].append(("v", i - D - T * D))

    @classmethod
    def from_call(cls, circuit, call) -> "HeadCones":
        return cls(circuit, call.name, call.lo, call.hi, [r.gate for r in call.iter_returns()], [r.gate for r in call.iter_args()], call.path())

    @classmethod
    def from_root(cls, circuit, program) -> "HeadCones":
        """The bare program: the root body IS the head body, its parameters are the leading ``Input`` batches."""
        root = circuit.root
        n_in = sum(node.n for node in root.nodes if node.form == "batch" and node.fn.name == "Input")
        args = [g for pi in range(3) for (_, _, g) in program.input_coll(pi).refs]
        assert len(args) == n_in, (len(args), n_in)
        outputs = [r.gate for r in circuit.root_outputs()]
        return cls(circuit, program.fn.id, n_in, circuit.size, outputs, args, ())

    def gate(self, g: int):
        got = self._gate_cache.get(g)
        if got is None:
            got = self._gate_cache[g] = self.circuit.gate(g)
        return got

    def cone(self, out_gate: int) -> tuple[set[int], set[int]]:
        """(gates strictly inside the head region, incoming gates outside it) reachable backwards from ``out_gate``."""
        inside: set[int] = set()
        incoming: set[int] = set()
        stack = [out_gate]
        while stack:
            g = stack.pop()
            if g in inside or g in incoming:
                continue
            if not (self.lo <= g < self.hi):
                incoming.add(g)
                continue
            inside.add(g)
            stack.extend(self.gate(g).operands)
        return inside, incoming

    # -- kinds ---------------------------------------------------------------------------------------------------------------

    def kind(self, g: int) -> str:
        """Primitive name with the tensor-core step split by role (q.k dot inside ``DotBf16`` vs P.V accumulate)."""
        gt = self.gate(g)
        name = base_name(gt.prim.name)
        scope_fn = base_name(gt.scope.fn.name)
        if name.startswith("AmpereBF16TcDot16") or name.endswith("Dot16_v1") or "Dot16" in name:
            return f"{name}[qk]" if scope_fn.startswith("DotBf16") else f"{name}[pv]"
        if name.startswith("Const32[0x00000000]") and scope_fn.startswith("DotBf16"):
            return f"{name}[qk-acc0]"
        if name.startswith("F32MulFtz"):
            ops = [base_name(self.gate(o).prim.name) for o in gt.operands]
            if any(o.startswith("Fa2InvSum") or o.startswith("Fa3InvSum") for o in ops):
                return f"{name}[epilogue o*inv]"
            if any(o.startswith("Const32[") for o in ops):  # (m_use or m_prev-m_use) * scale_log2
                return f"{name}[*scale_log2]"
            if any(o.startswith("MufuEx2Ftz") for o in ops):
                return f"{name}[rescale o_prev*alpha]"
            return f"{name}[other]"
        return name

    def p1_units(self, gates: set[int]) -> int:
        """Coordinate-level (P1) units touched: every gate is its own unit except the gates of a ``DotBf16`` activation,
        which form one straight-line unit."""
        units = 0
        dot_scopes: set[int] = set()
        for g in gates:
            gt = self.gate(g)
            if base_name(gt.scope.fn.name).startswith("DotBf16"):
                dot_scopes.add(gt.scope.offset)
            else:
                units += 1
        return units + len(dot_scopes)

    def widths(self, gates: set[int]) -> int:
        return sum(self.gate(g).width for g in gates)

    def is_const(self, g: int) -> bool:
        return base_name(self.gate(g).prim.name).startswith("Const")

    # -- incoming classification --------------------------------------------------------------------------------------------

    def incoming_summary(self, incoming: set[int]) -> dict[str, Any]:
        D, T = self.D, self.T
        q_coords: set[int] = set()
        k_rows: set[int] = set()
        k_coords: set[int] = set()
        v_rows: set[int] = set()
        v_coords: set[int] = set()
        unknown = 0
        for g in incoming:
            leaves = self.param_leaves.get(g)
            if not leaves:
                unknown += 1
                continue
            for name, i in leaves:
                if name == "q":
                    q_coords.add(i)
                elif name == "k":
                    k_rows.add(i // D)
                    k_coords.add(i % D)
                else:
                    v_rows.add(i // D)
                    v_coords.add(i % D)
        return {
            "gates": len(incoming),
            "q_coords": _ranges(q_coords), "q_coord_count": len(q_coords),
            "k_rows": _ranges(k_rows), "k_row_count": len(k_rows), "k_coords": _ranges(k_coords), "k_coord_count": len(k_coords),
            "v_rows": _ranges(v_rows), "v_row_count": len(v_rows), "v_coords": sorted(v_coords), "v_coord_count": len(v_coords),
            "unclassified_gates": unknown,
            "T": T, "D": D,
        }

    # -- the analysis ----------------------------------------------------------------------------------------------------------

    def analyse(self, coords: list[int], owner: int | None = None) -> dict[str, Any]:
        t0 = time.time()
        cones = {d: self.cone(self.outputs[d]) for d in coords}
        inside = {d: c[0] for d, c in cones.items()}
        incoming = {d: c[1] for d, c in cones.items()}
        shared = set.intersection(*inside.values()) if len(inside) > 1 else set()
        union = set.union(*inside.values())
        exclusive = {d: inside[d] - shared for d in coords}
        pairwise: dict[str, int] = {}
        overlaps: list[int] = []
        for i, a in enumerate(coords):
            for b in coords[i + 1:]:
                n = len(inside[a] & inside[b])
                overlaps.append(n)
                if len(coords) <= 8:
                    pairwise[f"{a},{b}"] = n
        pairwise_summary = {"pairs": len(overlaps), "min": min(overlaps) if overlaps else None, "max": max(overlaps) if overlaps else None,
                            "all_equal_to_shared": bool(overlaps) and min(overlaps) == max(overlaps) == len(shared)}

        def by_kind(gates: set[int]) -> dict[str, int]:
            return dict(sorted(Counter(self.kind(g) for g in gates).items()))

        per_coord = {}
        for d in coords:
            og = self.gate(self.outputs[d])
            per_coord[str(d)] = {
                "output_gate": self.outputs[d], "output_prim": base_name(og.prim.name), "output_bits": og.width,
                "cone_gates": len(inside[d]), "cone_p1_units": self.p1_units(inside[d]),
                "exclusive_gates": len(exclusive[d]), "exclusive_p1_units": self.p1_units(exclusive[d]),
                "exclusive_by_kind": by_kind(exclusive[d]),
                "incoming": self.incoming_summary(incoming[d]),
                # the gate id set as intervals (kept for small selections; whole-head runs record the shared set and the
                # exclusive sets only, to keep the artifact small)
                "cone_intervals": _intervals(inside[d]) if len(coords) <= 4 else None,
                "exclusive_intervals": _intervals(exclusive[d]),
            }
        # (b2): one owner holds the shared gates; what do the others' exclusive gates read from it?
        own = coords[0] if owner is None else owner
        iface: set[int] = set()
        for d in coords:
            if d == own:
                continue
            for e in exclusive[d]:
                for op in self.gate(e).operands:
                    if op in shared:
                        iface.add(op)
        iface_nonconst = {g for g in iface if not self.is_const(g)}
        # gates of the shared set whose outputs are read by NO exclusive gate of another coordinate (interior to the owner)
        shared_nonconst = {g for g in shared if not self.is_const(g)}
        # gates of the head region that no requested coordinate reaches (dead within the head when all D are requested)
        dead = [g for g in range(self.lo, self.hi) if g not in union] if len(coords) == self.D else None
        report = {
            "head_call": {"name": self.name, "path": list(map(list, self.path)), "region": [self.lo, self.hi],
                          "gates_in_region": self.hi - self.lo, "D": self.D, "T": self.T},
            "coords": coords,
            "union_gates": len(union), "union_p1_units": self.p1_units(union),
            "union_by_kind": by_kind(union),
            "dead_gates_in_head": None if dead is None else len(dead),
            "dead_by_kind": None if dead is None else by_kind(set(dead)),
            "shared_gates": len(shared), "shared_p1_units": self.p1_units(shared), "shared_by_kind": by_kind(shared),
            "shared_nonconst_gates": len(shared_nonconst), "shared_nonconst_p1_units": self.p1_units(shared_nonconst),
            "shared_intervals": _intervals(shared),
            "pairwise_overlap_gates": pairwise,
            "pairwise_overlap_summary": pairwise_summary,
            "per_coord": per_coord,
            "b2_owner": {"owner": own,
                         "interface_gates": len(iface), "interface_bits": self.widths(iface), "interface_by_kind": by_kind(iface),
                         "interface_nonconst_gates": len(iface_nonconst), "interface_nonconst_bits": self.widths(iface_nonconst)},
            "b3_duplication": {"sum_cone_gates": sum(len(inside[d]) for d in coords), "union_gates": len(union),
                               "factor_over_selected": (sum(len(inside[d]) for d in coords) / len(union)) if union else None,
                               "note": "over the selected coordinates only; with --all-coords this is the whole head's factor"},
            "seconds": round(time.time() - t0, 3),
        }
        return report


def _ranges(xs: set[int]) -> list[list[int]]:
    """Sorted ints as closed-open [lo, hi) intervals."""
    out: list[list[int]] = []
    for x in sorted(xs):
        if out and out[-1][1] == x:
            out[-1][1] = x + 1
        else:
            out.append([x, x + 1])
    return out


def _intervals(gates: set[int]) -> list[list[int]]:
    return _ranges(gates)


# -- selection ----------------------------------------------------------------------------------------------------------------


def pick_tiny_heads(circuit) -> dict[str, Any]:
    """TINY (LP=4, STEPS=8, NL=2, NH=2, KVH=1, D=16): the two heads of layer 0 at the last prompt position (t=3, T=4), and
    the two heads of layer 1 at the last decode position (t=11, T=12) -- by the call's ``t=.../layers.L/attn`` name."""
    calls = head_calls(circuit)
    named = []
    for c in calls:
        # climb to the Attention call to read its name
        parent = c.parent
        attn_node = parent.fn.nodes[parent.node_idx] if parent.parent is not None else None
        label = attn_node.name if attn_node is not None and attn_node.name else "?"
        named.append((label, c.member, c))
    return {"calls": named, "count": len(calls)}


def sweep(T_lo: int, T_hi: int, D: int, mult: int, census_path: Path | None) -> dict[str, Any]:
    """Whole-head sharing structure of a bare ``AttentionHead_v1{T, D}`` for every ``T`` in ``[T_lo, T_hi]`` (all D
    coordinates), and the totals over ``mult`` (layers x heads) copies of each T -- the B0 census scope is
    ``mult = NL x NH = 270`` with ``T = 1..47`` (LP=32 + 15 decode steps).  Cross-checked kind by kind against a Table-3(a)
    census file when given (the sums must reproduce the census's attention rows exactly)."""
    per_T: list[dict[str, Any]] = []
    tot_union = Counter()
    tot_shared = Counter()
    tot_excl = Counter()
    tot_dead = Counter()
    sums = Counter()
    for T in range(T_lo, T_hi + 1):
        program, adapted, circuit = build_head(T, D)
        hc = HeadCones.from_root(circuit, program)
        r = hc.analyse(list(range(D)))
        excl_kinds = Counter()
        for d in range(D):
            excl_kinds.update(r["per_coord"][str(d)]["exclusive_by_kind"])
        row = {"T": T, "union_gates": r["union_gates"], "union_p1_units": r["union_p1_units"],
               "shared_gates": r["shared_gates"], "shared_p1_units": r["shared_p1_units"], "shared_nonconst_gates": r["shared_nonconst_gates"],
               "shared_nonconst_p1_units": r["shared_nonconst_p1_units"],
               "exclusive_gates_per_coord": r["per_coord"]["0"]["exclusive_gates"], "exclusive_units_per_coord": r["per_coord"]["0"]["exclusive_p1_units"],
               "b2_interface_bits": r["b2_owner"]["interface_bits"], "b2_interface_nonconst_bits": r["b2_owner"]["interface_nonconst_bits"],
               "b3_sum_cone_gates": r["b3_duplication"]["sum_cone_gates"], "b3_factor": r["b3_duplication"]["factor_over_selected"],
               "shared_by_kind": r["shared_by_kind"], "exclusive_by_kind_all_coords": dict(sorted(excl_kinds.items()))}
        row["dead_gates_in_head"] = r["dead_gates_in_head"]
        row["dead_by_kind"] = r["dead_by_kind"]
        per_T.append(row)
        tot_union.update(r["union_by_kind"])
        tot_shared.update(r["shared_by_kind"])
        tot_excl.update(excl_kinds)
        tot_dead.update(r["dead_by_kind"] or {})
        for key in ("union_gates", "union_p1_units", "shared_gates", "shared_p1_units", "shared_nonconst_gates", "shared_nonconst_p1_units", "b3_sum_cone_gates"):
            sums[key] += row[key]
        sums["exclusive_gates"] += row["exclusive_gates_per_coord"] * D
        sums["exclusive_units"] += row["exclusive_units_per_coord"] * D
        sums["b2_interface_bits"] += row["b2_interface_bits"]
        sums["coordinates"] += D
        sums["heads"] += 1
    totals = {k: v * mult for k, v in sums.items()}
    doc: dict[str, Any] = {"schema": SCHEMA + "/sweep", "D": D, "T_range": [T_lo, T_hi], "mult": mult,
                           "label": "MEASURED per T on the bare AttentionHead_v1 body; totals = per-T sums x mult (the B0 census scope when mult = NL x NH)",
                           "per_T": per_T,
                           "sum_over_T": dict(sums), "totals_x_mult": totals,
                           "totals_by_kind": {"union": {k: v * mult for k, v in sorted(tot_union.items())},
                                              "shared": {k: v * mult for k, v in sorted(tot_shared.items())},
                                              "exclusive": {k: v * mult for k, v in sorted(tot_excl.items())},
                                              "dead_in_head": {k: v * mult for k, v in sorted(tot_dead.items())}}}
    if census_path is not None:
        census = json.loads(census_path.read_text())
        rows = {r["row_id"]: r for r in census["rows"]}
        union = doc["totals_by_kind"]["union"]
        dead = doc["totals_by_kind"]["dead_in_head"]

        def u(prefix: str, table=union) -> int:
            return sum(v for k, v in table.items() if k.startswith(prefix))

        got = {
            "attn-qk-dot-K64": u("AmpereBF16TcDot16_v1[qk]") // (D // 16),
            "attn-pv-dot16": u("AmpereBF16TcDot16_v1[pv]"),
            "attn-fmasub": u("F32FmaSubFtz"), "attn-exp2": u("MufuEx2Ftz"), "attn-p-cast": u("F2fpBf16") - mult * (T_hi - T_lo + 1) * D,
            "attn-rowmax": u("F32Max"), "attn-lsum": u("F32AddFtz"),
            "attn-rescale-mul": u("F32MulFtz"), "attn-guard": u("GuardNegInfZero"), "attn-invsum": u("Fa2InvSum"),
            "attn-const": u("Const16") + u("Const32[0x3e") + u("Const16", dead) + u("Const32[0x3e", dead),
        }
        check = {}
        for row_id, mine in got.items():
            want = rows[row_id]["n_f"] if row_id in rows else None
            check[row_id] = {"sweep": mine, "census_n_f": want, "match": (want == mine)}
        doc["census_check"] = {"file": str(census_path), "workload": census.get("workload"), "rows": check,
                               "all_match": all(c["match"] for c in check.values()),
                               "dead_gates_total": sum(dead.values()),
                               "note": "attn-p-cast = F2fpBf16 gates minus the D output casts per head; attn-qk-dot-K64 counts DotBf16 units "
                                       "(tc_dot16[qk] / (D/16)); attn-const includes the head's DEAD constants (the two Const16 pads at T in {16, 32} "
                                       "are emitted but read by no gate: no partial k-step) -- the census counts every gate, the cones only reachable ones"}
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", default=None, help="'T_lo..T_hi' -- whole-head sweep on bare AttentionHead_v1{T, D} (uses --D, --mult, --census)")
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--mult", type=int, default=1, help="copies of each T (B0 census scope: NL x NH = 270)")
    ap.add_argument("--census", type=Path, default=None, help="Table-3(a) census JSON to cross-check the sweep totals against")
    ap.add_argument("--shape", default=None, help="'tiny' (the TINY Serve_v1 program) or 'head:T=<keys>,D=<dim>' (a bare AttentionHead_v1)")
    ap.add_argument("--coords", default="0,1", help="output coordinates d to compare (comma list); default 0,1")
    ap.add_argument("--all-coords", action="store_true", help="compare all D coordinates (whole-head sharing structure)")
    ap.add_argument("--head", type=int, default=0, help="which AttentionHead call instance (index in canonical order; tiny: see --list)")
    ap.add_argument("--list", action="store_true", help="list the AttentionHead call instances and exit")
    ap.add_argument("--other-head", type=int, default=None, help="a second head call instance: report the cross-head overlap of coordinate 0")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.sweep:
        lo, hi = (int(x) for x in a.sweep.split(".."))
        doc = sweep(lo, hi, a.D, a.mult, a.census)
        doc["seconds_total"] = round(time.time() - t0, 3)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(doc, indent=1) + "\n")
            print(f"wrote {a.out}")
        s = doc["totals_x_mult"]
        print(f"sweep T={lo}..{hi} D={a.D} x{a.mult}: heads {s['heads']}, coords {s['coordinates']}, union gates {s['union_gates']} "
              f"({s['union_p1_units']} P1 units), shared gates {s['shared_gates']} ({s['shared_p1_units']} units, {s['shared_nonconst_p1_units']} non-const), "
              f"exclusive gates {s['exclusive_gates']} ({s['exclusive_units']} units); (b3) duplicated gates {s['b3_sum_cone_gates']} = x{s['b3_sum_cone_gates'] / s['union_gates']:.2f}")
        if "census_check" in doc:
            print("census check:", "ALL MATCH" if doc["census_check"]["all_match"] else "MISMATCH")
            for k, v in doc["census_check"]["rows"].items():
                print(f"   {k:20s} sweep {v['sweep']:>10} census {v['census_n_f']!s:>10} {'ok' if v['match'] else 'DIFF'}")
        print(f"  {doc['seconds_total']} s")
        return 0
    if a.shape is None:
        ap.error("--shape or --sweep is required")
    if a.shape == "tiny":
        program, adapted, circuit = build_tiny()
        shape = {"kind": "tiny", "function": program.fn.id, "gates": circuit.size}
    elif a.shape.startswith("head:"):
        kv = dict(p.split("=") for p in a.shape[5:].split(","))
        T, D = int(kv["T"]), int(kv["D"])
        program, adapted, circuit = build_head(T, D)
        shape = {"kind": "head", "function": program.fn.id, "gates": circuit.size, "T": T, "D": D}
    else:
        ap.error("unknown --shape")
    calls = head_calls(circuit)

    def attn_name(c) -> str | None:
        """The name of the ``Attention`` call node whose body holds this head (``t=<pos>/layers.<L>/attn`` in Serve)."""
        parent = c.parent  # the Attention activation
        if parent.parent is None:
            return None
        node = parent.parent.fn.nodes[parent.node_idx]
        return node.name

    if a.list:
        for i, c in enumerate(calls):
            print(i, c.name, "member", c.member, "region", (c.lo, c.hi), "attn-call", attn_name(c))
        return 0
    if calls:
        call = calls[a.head]
        hc = HeadCones.from_call(circuit, call)
        label = attn_name(call)
    else:  # the bare head program: the root body is the head body
        if shape["kind"] != "head":
            ap.error("no AttentionHead call instance in the program")
        hc = HeadCones.from_root(circuit, program)
        label = "root"
    coords = list(range(hc.D)) if a.all_coords else [int(x) for x in a.coords.split(",")]
    report = hc.analyse(coords)
    report["head_call"]["attention_call_name"] = label
    report["head_call"]["index"] = a.head if calls else None
    doc: dict[str, Any] = {"schema": SCHEMA, "shape": shape, "label": "MEASURED (enumeration on the adapted compact plan IR)", "report": report}
    if a.other_head is not None and calls:
        other = HeadCones.from_call(circuit, calls[a.other_head])
        c0, _ = hc.cone(hc.outputs[coords[0]])
        c1, inc1 = other.cone(other.outputs[coords[0]])
        doc["cross_head"] = {"head_a": a.head, "head_b": a.other_head, "coord": coords[0],
                             "attention_call_b": attn_name(calls[a.other_head]), "region_b": [other.lo, other.hi],
                             "overlap_gates": len(c0 & c1), "cone_a_gates": len(c0), "cone_b_gates": len(c1),
                             "regions_disjoint": hc.hi <= other.lo or other.hi <= hc.lo,
                             "incoming_b": other.incoming_summary(inc1)}
    doc["seconds_total"] = round(time.time() - t0, 3)
    text = json.dumps(doc, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
        print(f"wrote {a.out}")
    r = report
    print(f"{shape['function']}: head {label} {hc.name} region {r['head_call']['region']} T={r['head_call']['T']} D={r['head_call']['D']}")
    print(f"coords {coords[:8]}{'...' if len(coords) > 8 else ''}: union {r['union_gates']} gates / {r['union_p1_units']} P1 units; "
          f"shared {r['shared_gates']} gates / {r['shared_p1_units']} units ({r['shared_nonconst_gates']} non-const)")
    for d in coords[:4]:
        pc = r["per_coord"][str(d)]
        inc = pc["incoming"]
        print(f"  d={d}: cone {pc['cone_gates']} gates, exclusive {pc['exclusive_gates']} {pc['exclusive_by_kind']}; "
              f"in: q {inc['q_coord_count']}/{inc['D']} coords, K {inc['k_row_count']} rows x {inc['k_coord_count']} coords, "
              f"V {inc['v_row_count']} rows x coords {inc['v_coords']}; out {pc['output_prim']} {pc['output_bits']} bits")
    b2 = r["b2_owner"]
    print(f"  (b2) owner d={b2['owner']}: interface {b2['interface_gates']} gates = {b2['interface_bits']} bits "
          f"({b2['interface_nonconst_gates']} non-const = {b2['interface_nonconst_bits']} bits) {b2['interface_by_kind']}")
    print(f"  (b3) sum of cones {r['b3_duplication']['sum_cone_gates']} vs union {r['union_gates']}: x{r['b3_duplication']['factor_over_selected']:.2f}")
    if "cross_head" in doc:
        ch = doc["cross_head"]
        print(f"  cross-head {ch['head_a']} vs {ch['head_b']} coord {ch['coord']}: overlap {ch['overlap_gates']} gates (regions disjoint: {ch['regions_disjoint']})")
    print(f"  shared by kind: {r['shared_by_kind']}")
    print(f"  {doc['seconds_total']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
