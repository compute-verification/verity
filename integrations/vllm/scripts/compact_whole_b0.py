#!/usr/bin/env python3
"""[COMPACT-2] Step (c): whole-B0 ``validate_plan`` + ``Bound`` on the program the certificate pins, with the
run-aware Boundary / reach memo (GUIDANCE §41.4 item 2, §42.2, §44.2), instrumented per stage for VBENCH, and
compared with (1) the certificate of record and (2) the symbolic PLAN/PAD-SPEC table.

    python scripts/compact_whole_b0.py --out /workspace/compact2/out/whole-b0 \
        --record fixtures/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z --plans root,P3/cu=whole

Stages (each: wall seconds, RSS before/after, max RSS so far, input sizes): ``program`` (verity.ir build + digest),
``adapt`` (compact plan IR), ``circuit``, ``plan``, ``validate`` (Boundary is computed inside; its sizes are
reported by the ``boundary`` stage that walks the verdict's boundaries), ``bound`` (``hierarchical_table`` +
the frozen fold at the certificate's declared policy), ``compare``.  The RSS sampler writes ``<out>.rss.tsv``
per line so a death under the cgroup limit leaves the stage, size, time and peak RSS behind (lead item 4).

``--oracle`` runs the equality oracle at the given size instead: the compact path against ``compact=False``
(the exact pre-§39 expanded adapter) and against the verbatim per-leaf algorithms of
``tests/veritor/plans/perleaf_reference.py`` (pre-COMPACT-2 ``hierarchical_boundary`` / ``Intrinsic`` /
``Reach``): Boundary views, ``used_params``, projections, kind-table rows/digest and the ``Bound`` value must
be equal.  Sizes are scaled with ``--nl`` / ``--set K=V`` (growth-profile knobs, as ``bridge_diag.py``).

Comparison = exactly the four BND-S obligations (GUIDANCE §41.5): (i) circuit / gate census, (ii) token-output
designation, (iii) shared dependencies (weights committed once), (iv) sampling policy.  Every item is PASS or
DIFF with both sides recorded; the Serve@3 mismatch is a DIFF (§42.2), never a failure of either side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SCHEMA = "veritor/compact-whole-b0/v1"
BOUND_SCHEMA = "veritor/bound-report/v1"  # body reused from BND-S's wip-bound_report (docs/data/bnd-s/), fields added only
STAGES_SCHEMA = "veritor/verifier-stages/v1"
RECORD_ID = "INTPAD2-B0-20260909T013519Z"
TOKEN_BITS = 32  # the shipped I32 slot convention (frontier.width W_OUT_CONVENTIONS["I32"])


# -- resources -------------------------------------------------------------------------------------------------------


def rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1 << 20), 1)
    except Exception:  # noqa: BLE001
        return maxrss_mb()


def maxrss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / 1024, 1) if platform.system() == "Linux" else round(ru / (1 << 20), 1)


def cpu_model() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Model name:\s*(.+)", out)
        if m:
            return m.group(1).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


class Sampler(threading.Thread):
    """Appends ``elapsed_s\\trss_mb\\tstage`` lines (flushed) so the timeline survives a SIGKILL."""

    def __init__(self, path: Path, period_s: float) -> None:
        super().__init__(daemon=True)
        self.path, self.period_s, self.stage = path, period_s, "start"
        self.t0, self.stop, self.peak = time.monotonic(), threading.Event(), 0.0

    def run(self) -> None:
        with self.path.open("a", buffering=1) as fh:
            fh.write(f"# pid={os.getpid()} platform={platform.platform()}\n")
            while not self.stop.wait(self.period_s):
                r = rss_mb()
                self.peak = max(self.peak, r)
                fh.write(f"{time.monotonic() - self.t0:.2f}\t{r}\t{self.stage}\n")


class Stages:
    """Per-stage wall / RSS / input-size rows (the VBENCH Table 3(c) instrument) plus the crash-safe dump."""

    def __init__(self, out: Path, sampler: Sampler, doc: dict[str, Any]) -> None:
        self.out, self.sampler, self.doc, self.rows = out, sampler, doc, []
        self.t_start = time.perf_counter()
        doc["stages"] = self.rows

    def dump(self, status: str, **extra: Any) -> None:
        self.doc.update(extra)
        self.doc["status"] = status
        self.doc["maxrss_mb"] = maxrss_mb()
        self.doc["sampler_peak_mb"] = self.sampler.peak
        self.doc["elapsed_s"] = round(time.perf_counter() - self.t_start, 2)
        self.out.write_text(json.dumps(self.doc, indent=1, default=str) + "\n")

    def run(self, name: str, fn, **inputs: Any) -> Any:
        self.sampler.stage = name
        before = rss_mb()
        t = time.perf_counter()
        row = {"stage": name, "inputs": inputs, "rss_before_mb": before, "started_s": round(t - self.t_start, 2)}
        self.rows.append(row)
        self.dump(f"running: {name}")
        try:
            result = fn(row)
        except BaseException as e:  # record the death, then re-raise
            row.update(seconds=round(time.perf_counter() - t, 2), rss_after_mb=rss_mb(), maxrss_mb=maxrss_mb(),
                       error=f"{type(e).__name__}: {e}"[:400])
            self.dump(f"died in {name}: {type(e).__name__}")
            raise
        row.update(seconds=round(time.perf_counter() - t, 2), rss_after_mb=rss_mb(), maxrss_mb=maxrss_mb())
        self.dump(f"done: {name}")
        print(f"[{row['started_s']:8.1f}s] {name:10s} {row['seconds']:9.2f}s  rss {before:.0f} -> {row['rss_after_mb']:.0f} MB "
              f"(max {row['maxrss_mb']:.0f})  {inputs}", flush=True)
        return result


# -- the program ------------------------------------------------------------------------------------------------------


def build_config(case: str, nl: int, sets: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from verity_vllm.registry import b1

    config = dict(getattr(b1, f"{case}_CONFIG"))
    overrides: dict[str, Any] = {}
    if nl:
        config["NL"] = nl
        overrides["NL"] = nl
    for kv in sets:
        k, _, v = kv.partition("=")
        config[k] = int(v)
        overrides[k] = int(v)
    return config, overrides


def build_program(config: dict[str, Any], lp: int, steps: int, digest: bool, row: dict[str, Any]):
    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    specialised = bind(b1.Serve, C=config, LP=lp, STEPS=steps)
    program = Program(specialised)
    row["gates"] = program.gates
    row["input_gates"] = program.input_gates
    row["root_nodes"] = len(program.root.body.nodes)
    row["function"] = specialised.id
    if digest:
        from verity.ir import encode_program, program_digest

        t = time.perf_counter()
        desc = encode_program(program)
        row["digest"] = program_digest(desc)
        row["digest_seconds"] = round(time.perf_counter() - t, 2)
        del desc
    return program


def definitions(program) -> list[Any]:
    from veritor.plans.ir import FunctionDefinition

    seen: dict[int, Any] = {}
    stack = [program.root]
    while stack:
        fn = stack.pop()
        if id(fn) in seen:
            continue
        seen[id(fn)] = fn
        for node in fn.nodes:
            if isinstance(node.fn, FunctionDefinition):
                stack.append(node.fn)
    return list(seen.values())


def ir_sizes(program) -> dict[str, int]:
    """Nodes, runs and leaves of the plan IR (what the compact representation stores vs. what it denotes)."""

    from veritor.plans.ir import RefSeq

    defs = definitions(program)
    nodes = runs = leaves = 0
    for fn in defs:
        seqs = [fn.returns]
        for node in fn.nodes:
            nodes += 1
            seqs += [node.args, node.init, node.shared]
        for seq in seqs:
            if isinstance(seq, RefSeq):
                runs += len(seq.runs)
                leaves += len(seq)
            else:
                runs += len(seq)
                leaves += len(seq)
    return {"definitions": len(defs), "nodes": nodes, "ref_runs": runs, "ref_leaves": leaves}


# -- plans ------------------------------------------------------------------------------------------------------------


def root_plan(program, circuit):
    """``bridge_diag.py``'s whole-program plan: one VU per root node (all members), CU(inputs) + CU(calls)."""

    from veritor.plans.boundary import Part, Parts
    from veritor.plans.plan import VerificationPlan

    root = program.root
    inputs = [k for k, n in enumerate(root.nodes) if n.form == "batch" and n.fn.name == "Input"]
    calls = [k for k in range(len(root.nodes)) if k not in inputs]

    def whole(k: int) -> Parts:
        return Parts(circuit, [Part((), k, 0, root.nodes[k].n)])

    vus = [whole(k) for k in inputs] + [whole(k) for k in calls]
    cus = [Parts(circuit, [Part((), k, 0, root.nodes[k].n) for k in group]) for group in (inputs, calls) if group]
    return VerificationPlan("root", vus, cus), {"constructor": "bridge_diag root P1", "vu_level": "root-node", "cu_strategy": "inputs|calls"}


def unrolled_runs(adapted) -> dict[str, tuple[tuple[int, int], ...]]:
    runs: dict[str, list[tuple[int, int]]] = {}
    for fid, original, members in adapted.unrolled:
        runs.setdefault(fid, []).append((adapted.node_index[(fid, original)], members))
    return {k: tuple(sorted(v)) for k, v in runs.items()}


def make_plan(spec: str, program, circuit, adapted):
    if spec == "root":
        return root_plan(program, circuit)
    from veritor.plans.canonical import PLAN_NAMES, Classifier, canonical_plan

    head, _, cu = spec.partition("/cu=")
    cp = canonical_plan(circuit, PLAN_NAMES[head], cu or "whole", Classifier(unrolled_runs(adapted)))
    return cp.plan, {"constructor": "veritor.plans.canonical", "vu_level": cp.vu_level, "cu_strategy": cp.cu_strategy}


# -- boundary statistics ------------------------------------------------------------------------------------------------


def boundary_stats(verdict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, boundaries in (("vu", verdict.vu_boundaries), ("cu", verdict.cu_boundaries)):
        gates = runs = 0
        biggest = 0
        for b in boundaries:
            size = b.boundary_size()
            gates += size
            biggest = max(biggest, size)
            runs += len(b.inputs.runs()) + len(b.outputs.runs())
        out[label] = {"units": len(boundaries), "boundary_gates": gates, "boundary_runs": runs, "largest_unit_boundary_gates": biggest}
    return out


# -- the fold --------------------------------------------------------------------------------------------------------


def fold(table, q: Fraction, s: Fraction, delta: Fraction):
    from veritor.analysis.bound import bound
    from veritor.core.policy import VerificationPolicy
    from veritor.frontier.bound_adapter import FOLD_OPTIONS

    return bound(table, VerificationPolicy(q=q, s=s), delta, FOLD_OPTIONS)


def bound_report_body(program_id: str, program, plan, plan_meta, verdict, result, q: Fraction, s: Fraction, delta: Fraction,
                      b, seconds: float, circuit) -> dict[str, Any]:
    """The ``veritor/bound-report/v1`` object (BND-S's frozen names; ``engine`` = ``legacy-runaware``)."""

    from veritor.frontier.bound_adapter import (
        FOLD_OPTIONS,
        Table,
        kind_table_to_json,
        options_to_json,
    )

    table = Table(result.table, program_id, plan.name, "COMPACT-2", "compact_whole_b0 engine=legacy-runaware")
    output_bits = table.output_bits
    vu_fb = sum(1 for u in result.vu_summaries if u.narrowing == "fallback")
    cu_fb = sum(1 for u in result.cu_summaries if u.narrowing == "fallback")
    extra = result.extra
    inf = float("inf")
    return {
        "schema": BOUND_SCHEMA,
        "program": {"name": program_id, "gates": program.gates, "input_gates": program.input_gates, "output_bits": output_bits,
                    "root_nodes": len(program.root.body.nodes)},
        "plan": {"name": plan.name, "vu_units": len(plan.verification_units), "cu_units": len(plan.commitment_units), **plan_meta},
        "delta": str(delta), "policy": {"q": str(q), "s": str(s)},
        "output_bits": output_bits, "U_bits": float(b.bits), "U_over_W": float(b.bits) / output_bits if output_bits else None,
        "capped": bool(b.capped),
        "terms": {"knapsack_bits": None if b.knapsack_bits == inf else float(b.knapsack_bits),
                  "laplace_bits": None if b.laplace_bits == inf else float(b.laplace_bits),
                  "dominant": table.dominant_term(delta, s, b)},
        "fold_options": options_to_json(FOLD_OPTIONS),
        "contextual": {"units": len(result.vu_summaries) - vu_fb, "kinds": result.vu_kinds - int(extra.get("vu_fallback_kinds", 0)),
                       "cu_units": len(result.cu_summaries) - cu_fb, "cu_kinds": result.cu_kinds - int(extra.get("cu_fallback_kinds", 0))},
        "fallback": {"units": vu_fb, "kinds": int(extra.get("vu_fallback_kinds", 0)), "cu_units": cu_fb,
                     "cu_kinds": int(extra.get("cu_fallback_kinds", 0))},
        "work": {"visits": result.visits, "hops": result.hops, "seconds": seconds, "reach_steps": result.steps,
                 "intrinsic_entries": result.intrinsic_entries, "circuit_visits_total": circuit.visits},
        "engine": "legacy-runaware", "kinds_by": "exact",
        "kind_table": kind_table_to_json(result.table),
        "label": "COMPACT-2", "provenance": f"scripts/compact_whole_b0.py python={sys.version.split()[0]}",
    }


# -- the symbolic side ----------------------------------------------------------------------------------------------------


def strip_digests(label: str) -> str:
    return re.sub(r"\{[0-9a-f]{12}\}", "{}", label)


def symbolic_side(config: dict[str, Any], lp: int, steps: int, q: Fraction, s: Fraction, delta: Fraction) -> dict[str, Any]:
    """The PLAN/PAD-SPEC table for the same (B0, LP, STEPS): ``Serve@2`` as the live branch with ``STEPS = D_MAX``
    (``veritor.frontier.width``), the source of ``docs/data/frontier/aggregate-export/u_table.json`` via
    ``aggregate.representative()`` (there at LP = 512, D_MAX = 2048 — a different workload, stated)."""

    from veritor.frontier.padded import PaddedWorkload
    from veritor.frontier.structure import VClassifier
    from veritor.frontier.width import derived_structure, n_out, width_table

    w = PaddedWorkload("B0", dict(config), lp, steps)
    cl = VClassifier()
    st = derived_structure(w, ("coordinate", "operator"), cl)
    out: dict[str, Any] = {
        "proxy": "Serve@2 (Serve_v2 body definitions) with STEPS = D_MAX as the live branch of the padded circuit — "
                 "veritor.frontier.width.derived_structure / width_table (padded-horizon structural proxy)",
        "workload": {"base": "B0", "LP": lp, "D_MAX": steps, "STEPS": w.STEPS, "positions": w.n_positions},
        "gates": st.gates, "gates_decomposition": st.verification["gates_decomposition"],
        "W_circuit_bits": w.W_circuit, "n_out": n_out(w),
        "serve_at_3_convention": {"D_MAX_as_max_tokens": steps + 1, "slots": steps + 2, "W_bits": TOKEN_BITS * (steps + 2),
                                  "note": "output-manifest-b0-v2 §2.2 (Serve@3): D_MAX = max_tokens gives D_MAX + 1 slots; the record's "
                                          "Serve_v1 runs STEPS = max_tokens - 1 (16 slots at max_tokens = 16) — §42.2 DIFF"},
        "census": {},
        "tables": {},
    }
    for lvl, census in st.censuses.items():
        if lvl == "whole":
            continue
        by_fn: Counter = Counter()
        for (fn_id, form, members), uk in census.kinds.items():
            by_fn[strip_digests(fn_id)] += uk.count
        out["census"][lvl] = {"units": census.units, "gates": census.gates, "by_function": dict(sorted(by_fn.items()))}
    for vu, cu in (("coordinate", "operator"), ("coordinate", "whole"), ("operator", "whole"), ("operator", "block")):
        try:
            t = width_table(w, vu, cu, st.gates, classifier=cl)
            b = fold(t.table, q, s, delta)
            rows = [{"kind": strip_digests(r.kind), "role": r.role, "copies": r.copies, "size": r.size, "out_bits": r.out_bits,
                     "reach_bits": r.reach_bits, "ancestor_bits": r.ancestor_bits, "source_inputs": r.source_inputs,
                     "source_weights": r.source_weights} for r in t.table.rows]
            inf = float("inf")
            out["tables"][f"{vu}/cu={cu}"] = {"plan": t.plan, "rows": len(rows), "output_bits": t.output_bits, "vu_units": t.vu_units,
                                              "cu_units": t.cu_units, "U_bits": float(b.bits), "capped": bool(b.capped),
                                              "knapsack_bits": None if b.knapsack_bits == inf else float(b.knapsack_bits),
                                              "laplace_bits": None if b.laplace_bits == inf else float(b.laplace_bits),
                                              "input_count": t.table.input_count, "weight_count": t.table.weight_count,
                                              "replay_unit_count": t.table.replay_unit_count,
                                              "root_row": next(r for r in rows if r["kind"] == strip_digests(t.table.root)),
                                              "kind_rows": rows}
        except Exception as e:  # noqa: BLE001
            out["tables"][f"{vu}/cu={cu}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


# -- the comparison ------------------------------------------------------------------------------------------------------


def load_record(record: Path) -> dict[str, Any]:
    def j(rel: str) -> Any:
        p = record / rel
        return json.loads(p.read_text()) if p.exists() else None

    return {"certificate": j("certificate.json"), "program": j("program/program.json"), "sampling_policy": j("sampling_policy.json"),
            "workload": j("workload.json"), "full_program": j("plan/full_program.json"), "unit_layer": j("plan/unit-layer.json")}


def item(items: list, obligation: str, name: str, ok: bool | None, program_side: Any, other_side: Any, note: str = "") -> None:
    items.append({"obligation": obligation, "item": name, "verdict": "PASS" if ok else ("DIFF" if ok is False else "N/A"),
                  "program_derived": program_side, "compared_with": other_side, "note": note})


def compare(doc: dict[str, Any], rec: dict[str, Any], sym: dict[str, Any] | None, program_row: dict[str, Any], plan_results: dict[str, Any],
            root_inputs: list[dict[str, Any]], q: Fraction, s: Fraction, delta: Fraction) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cert = rec["certificate"] or {}
    rprog = (rec["program"] or {}).get("program", {})
    # the plan-agnostic items (validated, output designation, the weight row) read the root plan when present, else the first plan
    root_spec = "root" if "root" in plan_results else next(iter(plan_results), "root")
    root = plan_results.get(root_spec, {})
    root_bound = (root.get("bound") or {})
    kt = ((root_bound.get("report") or {}).get("kind_table") or {})
    rows = kt.get("rows") or []

    # (i) circuit / gate census
    item(items, "i", "program identity (function id)", program_row.get("function") == rprog.get("function"),
         program_row.get("function"), rprog.get("function"), "the certificate names Serve_v1 (§42.1 G6); this run builds the same binding")
    if "digest" in program_row:
        item(items, "i", "program digest (descriptor codec)", program_row["digest"] == rprog.get("digest"), program_row["digest"], rprog.get("digest"),
             "program_digest(encode_program(.)) recomputed here vs program/program.json")
    item(items, "i", "gates vs certificate", program_row.get("gates") == rprog.get("gates"), program_row.get("gates"), rprog.get("gates"))
    item(items, "i", "input gates vs certificate", program_row.get("input_gates") == rprog.get("input_gates"), program_row.get("input_gates"),
         rprog.get("input_gates"))
    fpv = (cert.get("program") or {}).get("full_program_validation") or {}
    item(items, "i", "whole-program plan validated", root.get("status") == "valid",
         {"status": root.get("status"), "gates": root.get("circuit_size"), "seconds": root.get("validate_seconds"), "maxrss_mb": root.get("validate_maxrss_mb")},
         {"status": fpv.get("status"), "validated": fpv.get("validated"), "maxrss_mb": fpv.get("maxrss_mb"), "cap_gb": fpv.get("cap_gb")},
         "the record's own attempt (bridge P1 plan, per-gate structures) died under its 20 GB cap; this is the same plan on the run-aware structures")
    if sym is not None:
        item(items, "i", "total gates vs symbolic proxy", program_row.get("gates") == sym["gates"], program_row.get("gates"), sym["gates"],
             "symbolic = Serve@2 live-branch proxy (STEPS = D_MAX): its gate total has no Input gates (weights + prompt are not units there); "
             "the program's total includes its 135,039,328 input gates — see the next item")
        op_gates = (program_row.get("gates") or 0) - (program_row.get("input_gates") or 0)
        item(items, "i", "operator gates (gates - input gates) vs symbolic proxy gates", op_gates == sym["gates"],
             {"gates": program_row.get("gates"), "input_gates": program_row.get("input_gates"), "operator_gates": op_gates,
              "calls_CU_size": next((r.get("size") for r in rows if r.get("role") == "replay" and r.get("size") == op_gates), None)},
             {"gates": sym["gates"], "decomposition": sym.get("gates_decomposition")},
             "equal: Serve_v1's operator gates == the Serve@2 proxy's gate total at the same (LP, STEPS) — the v1/v2 body difference does not "
             "change the gate count; the `calls` CU of the root plan is exactly this set")
        census = doc.get("root_census") or {}
        if census and "operator" in sym["census"]:
            b = sym["census"]["operator"]["by_function"]
            def sym_count(pred):
                return sum(v for k, v in b.items() if pred(k))
            pairs = {
                "Embedding_v1 (root calls) vs Embedding_v1{V,H} operator units": (census.get("Embedding_v1"), sym_count(lambda k: k.startswith("Embedding_v1"))),
                "LayerPre_v1 vs Gemm_v2{K=576,N=960} (QKV projections)": (census.get("LayerPre_v1"), sym_count(lambda k: k.startswith("Gemm_v2{K=576,N=960"))),
                "Attention_v1 vs Attention_v2 (all T)": (census.get("Attention_v1"), sym_count(lambda k: k.startswith("Attention_v2"))),
                "LayerPost_v1 vs Gemm_v2{K=1536,N=576} (down projections)": (census.get("LayerPost_v1"), sym_count(lambda k: k.startswith("Gemm_v2{K=1536,N=576"))),
                "Final_v1 vs TokenSelect_v1": (census.get("Final_v1"), sym_count(lambda k: k.startswith("TokenSelect_v1"))),
                "Final_v1 vs Gemm_v2{K=576,N=49152} (lm_head)": (census.get("Final_v1"), sym_count(lambda k: k.startswith("Gemm_v2{K=576,N=49152"))),
            }
            diff = {k: v for k, v in pairs.items() if v[0] != v[1]}
            item(items, "i", "per-kind populations: root calls of Serve_v1 vs the proxy's operator census (LP + STEPS = 47 positions, 30 layers)",
                 not diff, {"root_census": census, "root_nodes": sum(census.values())},
                 {"proxy_operator_units": sym["census"]["operator"]["units"], "pairs": {k: {"program": v[0], "symbolic": v[1]} for k, v in pairs.items()}},
                 "one root call per (position, layer) in Serve_v1 = one operator unit per layer-op in the proxy: 47 embeddings, 1,410 = 47 x 30 layer "
                 "calls of each of LayerPre/Attention/LayerPost, 16 = STEPS + 1 final projections")
        op = plan_results.get("P3/cu=whole") or {}
        if op.get("kind_copies_by_function") and "operator" in sym["census"]:
            a = op["kind_copies_by_function"]
            b = sym["census"]["operator"]["by_function"]
            keys = sorted(set(a) | set(b))
            diff = {k: {"program": a.get(k), "symbolic": b.get(k)} for k in keys if a.get(k) != b.get(k)}
            item(items, "i", "operator-level unit census by function", not diff, {"units": sum(a.values()), "functions": len(a)},
                 {"units": sym["census"]["operator"]["units"], "functions": len(b)},
                 json.dumps(diff)[:1500] if diff else "every function's unit count equal (digests stripped)")

    # (ii) token outputs designation
    w_prog = root_bound.get("output_bits")
    n_out_prog = (w_prog // TOKEN_BITS) if w_prog else None
    enforcement = cert.get("enforcement") or {}
    linkage = enforcement.get("output_linkage") or {}
    per_request = (linkage.get("leaves_checked") or 0) // max(1, linkage.get("requests") or 1)
    item(items, "ii", "designated token outputs per request", n_out_prog == per_request if n_out_prog else None,
         {"root_output_bits": w_prog, "slots_I32": n_out_prog}, {"output_linkage_leaves_per_request": per_request, "declared_domain": enforcement.get("declared_domain")},
         "the program's root outputs are the designated token slots; the certificate checked output linkage on that many leaves per request")
    wl = rec["workload"] or {}
    max_tokens = (wl.get("max_tokens") or [None])[0]
    item(items, "ii", "STEPS = max_tokens - 1 (Serve_v1 convention)", (doc["steps"] == (max_tokens or 0) - 1) if max_tokens else None,
         {"STEPS": doc["steps"], "slots": n_out_prog}, {"max_tokens": max_tokens},
         "Serve_v1 emits STEPS + 1 slots = max_tokens slots")
    if sym is not None:
        item(items, "ii", "W_out vs symbolic proxy (same STEPS)", w_prog == sym["W_circuit_bits"] if w_prog else None, w_prog, sym["W_circuit_bits"],
             "both 32·(STEPS+1) under the I32 convention when equal")
        s3 = sym["serve_at_3_convention"]
        item(items, "ii", "Serve@3 convention (D_MAX = max_tokens)", False, {"slots": n_out_prog, "W_bits": w_prog}, s3,
             "§42.2: Serve@3 has no B0 instance/pin; its D_MAX = max_tokens convention gives one more slot than the certified Serve_v1 — "
             "recorded as DIFF, not as a failure of either")

    # (iii) shared dependencies — weights committed once
    weight_gates = sum(n["gates"] for n in root_inputs if n["kind"] == "weight")
    prompt_gates = sum(n["gates"] for n in root_inputs if n["kind"] == "prompt")
    other_gates = sum(n["gates"] for n in root_inputs if n["kind"] == "other")
    item(items, "iii", "input gates = weights + prompt (+ caches)", weight_gates + prompt_gates + other_gates == program_row.get("input_gates"),
         {"weights": weight_gates, "prompt": prompt_gates, "other": other_gates, "input_batches": len(root_inputs)},
         program_row.get("input_gates"), "root Input batches classified by name; B0's `weights` batch holds the checkpoint tensors AND the rope "
         "cos/sin cache (MAX_POS x D = 524,288 leaves, 47 positions of it read)")
    weight_rows = [r for r in rows if r.get("role") == "verification" and r.get("size") == weight_gates]
    wrow = weight_rows[0] if weight_rows else {}
    item(items, "iii", "weights are ONE root Input batch = one VU row (copies 1) inside one replay unit; every step reads it",
         bool(wrow) and wrow.get("copies") == 1 and kt.get("replay_unit_count") == 2,
         {"weight_batches": sum(1 for n in root_inputs if n["kind"] == "weight"), "weight_gates": weight_gates,
          "kind_row": {k: wrow.get(k) for k in ("kind", "copies", "size", "out_count", "out_bits", "closed")},
          "read_leaves": wrow.get("out_count"), "replay_units": kt.get("replay_unit_count")},
         {"certificate": "one weights_sha256 / one embed table_root / one h_L commitment for all four requests",
          "kind_coverage.embed.weights_sha256": ((cert.get("kind_coverage") or {}).get("embed") or {}).get("weights_sha256"),
          "commitments.h_L": (cert.get("commitments") or {}).get("h_L")},
         "structural: the checkpoint is a single root Input batch shared by all LP + STEPS positions and all 30 layers, so it is "
         "committed once (CU(inputs)); byte-level equality of the committed weights with the checkpoint is CAPCORR's L2 check, not this script's")
    item(items, "iii", "fold's source classification (kind_table.input_count / weight_count)", None,
         {"input_count": kt.get("input_count"), "weight_count": kt.get("weight_count")},
         {"symbolic": {k: {kk: v.get(kk) for kk in ("input_count", "weight_count", "replay_unit_count")} for k, v in (sym or {}).get("tables", {}).items() if "U_bits" in v}},
         "N/A: neither the hierarchical table nor the proxy table classifies source gates (both report 0); the fold's cuts do not use them "
         "(cut = min(out, reach, ancestor)); the shared-dependency evidence is the structural row above")
    if sym is not None:
        item(items, "iii", "weights in the symbolic proxy table", False, {"weight_row": bool(wrow), "weight_gates": weight_gates},
             {"proxy": "no Input rows: the padded proxy's censuses count operator/coordinate units; weights enter only as gate totals"},
             "representation DIFF, not a count mismatch: the proxy has no weight unit to compare against")

    # (iv) sampling policy
    sp = rec["sampling_policy"] or {}
    declared = {"q": sp.get("q_declared"), "s": sp.get("s_declared"), "k": sp.get("k"), "selector": sp.get("selector"), "coverage_size": sp.get("coverage_size")}
    used = {"q": str(q), "s": str(s), "delta": str(delta)}
    ok_pol = (Fraction(sp["q_declared"]) == q and Fraction(sp["s_declared"]) == s) if sp.get("q_declared") and sp.get("s_declared") else None
    item(items, "iv", "Bound evaluated at the certificate's declared (q, s)", ok_pol, used, declared,
         "declared lives in sampling_policy.json, bound before READY_PHASE0; δ is not a certificate field (record's Bound: bound_record DELTAS)")
    item(items, "iv", "whole-program Bound at (q, s, δ)", True if root_bound.get("U_bits") is not None else None,
         {"U_bits": root_bound.get("U_bits"), "W_out": w_prog, "capped": root_bound.get("capped"), "plan": "root"},
         {"record": "no whole-program Bound exists in the record (full_program_validation died); layer-program P1 Bound in plan/unit-layer.json",
          "unit_layer_P1_bound": ((rec["unit_layer"] or {}).get("partitions", {}).get("P1", {}) or {}).get("bound")},
         "a number the certificate could not produce; reported, not compared to a nonexistent record value")
    terms = ((root_bound.get("report") or {}).get("terms") or {})
    # canonical plans: the SAME plan in both constructions -> exact comparison of U and of the pre-cap terms (GUIDANCE §38.3)
    same_plan = {"P3/cu=whole": "operator/cu=whole", "P3/cu=block": "operator/cu=block", "P1/cu=operator": "coordinate/cu=operator",
                 "P1/cu=whole": "coordinate/cu=whole"}
    for spec, res in plan_results.items():
        if spec == "root" or not res.get("bound") or sym is None:
            continue
        b = res["bound"]
        t = b.get("report", {}).get("terms", {})
        st = sym["tables"].get(same_plan.get(spec, ""), {})
        prog_side = {"plan": spec, "status": res.get("status"), "vus": res.get("vus"), "cus": res.get("cus"), "U_bits": b.get("U_bits"),
                     "capped": b.get("capped"), "uncapped_knapsack_bits": t.get("knapsack_bits"), "uncapped_laplace_bits": t.get("laplace_bits"),
                     "kind_rows": b.get("rows")}
        if "U_bits" not in st:
            item(items, "iv", f"{spec}: U and pre-cap terms vs the proxy table for the same plan", None, prog_side,
                 {"plan": same_plan.get(spec), "error": st.get("error", "no proxy table for this plan")}, "the proxy has no table for this plan")
            continue
        sym_side = {"plan": same_plan[spec], "vu_units": st.get("vu_units"), "cu_units": st.get("cu_units"), "U_bits": st["U_bits"], "capped": st["capped"],
                    "uncapped_knapsack_bits": st.get("knapsack_bits"), "uncapped_laplace_bits": st.get("laplace_bits"), "kind_rows": st.get("rows")}
        item(items, "iv", f"{spec}: U_bits vs proxy {same_plan[spec]} at the same (q, s, δ)", b.get("U_bits") == st["U_bits"], prog_side, sym_side,
             ("both capped at W_out; " if b.get("capped") and st["capped"] else "") + "same canonical plan, program-derived vs formula table")
        pk, sk = t.get("knapsack_bits"), st.get("knapsack_bits")
        if pk is not None and sk is not None:
            rel = abs(pk - sk) / max(abs(sk), 1e-12)
            item(items, "iv", f"{spec}: uncapped knapsack term vs proxy {same_plan[spec]}", rel <= 1e-9,
                 {"knapsack_bits": pk, "laplace_bits": t.get("laplace_bits")}, {"knapsack_bits": sk, "laplace_bits": st.get("laplace_bits")},
                 f"difference {pk - sk:+.6f} bits ({rel:.3e} relative); " + ("equal" if rel <= 1e-9 else
                 "the proxy's operator units are Serve@2 bodies with no Input units and STEPS = D_MAX (16 slots here too, so not the Serve@3 "
                 "convention); a residual difference is the plan's unit census/out-widths on Serve_v1 vs the formula's — see the kind rows"))
        item(items, "iv", f"{spec}: unit counts vs proxy {same_plan[spec]}", res.get("vus") == st.get("vu_units") and res.get("cus") == st.get("cu_units"),
             {"vus": res.get("vus"), "cus": res.get("cus")}, {"vu_units": st.get("vu_units"), "cu_units": st.get("cu_units")},
             "the proxy census counts operator units of the Serve@2 body; Input batches are VUs in the program's plan but not units in the proxy")
    if sym is not None and root_spec == "root":
        for key, st in sym["tables"].items():
            if "U_bits" in st:
                both_capped = bool(root_bound.get("capped")) and bool(st["capped"])
                item(items, "iv", f"U_bits vs symbolic {key} at the same (q, s, δ)", root_bound.get("U_bits") == st["U_bits"],
                     {"plan": "root", "U_bits": root_bound.get("U_bits"), "capped": root_bound.get("capped"),
                      "uncapped_knapsack_bits": terms.get("knapsack_bits"), "uncapped_laplace_bits": terms.get("laplace_bits")},
                     {"plan": key, "U_bits": st["U_bits"], "capped": st["capped"], "uncapped_knapsack_bits": st.get("knapsack_bits"),
                      "uncapped_laplace_bits": st.get("laplace_bits")},
                     ("EQUAL ONLY BECAUSE BOTH ARE CAPPED at W_out: the uncapped fold terms differ (recorded); " if both_capped else "")
                     + "different plans over different programs (root-node VUs on Serve_v1 vs the proxy's structural table); the difference is the record")
                if both_capped:
                    item(items, "iv", f"uncapped fold term vs symbolic {key}", terms.get("knapsack_bits") == st.get("knapsack_bits"),
                         {"knapsack_bits": terms.get("knapsack_bits"), "laplace_bits": terms.get("laplace_bits")},
                         {"knapsack_bits": st.get("knapsack_bits"), "laplace_bits": st.get("laplace_bits")},
                         "the pre-cap sums the cap hides: root plan (4,295 coarse VUs) vs the proxy's fine-unit tables")
    return items


# -- the oracle ------------------------------------------------------------------------------------------------------------


def oracle(args, stages: Stages, doc: dict[str, Any]) -> int:
    """compact=True vs compact=False (expanded adapter) vs the per-leaf reference algorithms, at one size."""

    import importlib

    sys.path.insert(0, str(ROOT / "tests" / "veritor" / "plans"))
    tb = importlib.import_module("tests.veritor.plans.test_compact_boundary")
    pr = importlib.import_module("tests.veritor.plans.perleaf_reference")
    import pytest
    from veritor.plans import bound as bound_module
    from veritor.plans.bound import hierarchical_table
    from veritor.plans.validate import validate_plan
    from veritor.plans.verity_ir_adapter import adapt

    config, overrides = build_config(args.case, args.nl, args.set)
    doc["config_overrides"] = overrides
    program = stages.run("program", lambda row: build_program(config, args.lp, args.steps, False, row))
    compact = stages.run("adapt", lambda row: adapt(program, compact=True), compact=True)
    expanded = stages.run("adapt-expanded", lambda row: adapt(program, compact=False), compact=False)
    c1 = stages.run("circuit", lambda row: compact.program.circuit())
    c2 = stages.run("circuit-expanded", lambda row: expanded.program.circuit())
    doc["gates"] = c1.size
    checks: dict[str, Any] = {}

    def used_params(row):
        n = 0
        for fn in definitions(compact.program):
            assert set(fn.used_params) == set(fn.used_params_expanded()), fn.id
            n += 1
        row["definitions"] = n
        return n

    stages.run("used_params-oracle", used_params)
    stages.run("resolve_run-oracle", lambda row: row.update(runs=tb.assert_resolve_run_equal_perleaf(c1)))

    for spec in args.plans.split(","):
        plan1, _ = make_plan(spec, compact.program, c1, compact)
        plan2, _ = make_plan(spec, expanded.program, c2, expanded)
        v1 = stages.run(f"validate[{spec}]", lambda row: validate_plan(c1, plan1), vus=len(plan1.verification_units))
        v2 = stages.run(f"validate-expanded[{spec}]", lambda row: validate_plan(c2, plan2), vus=len(plan2.verification_units))
        assert v1.status == v2.status and list(v1.codes) == list(v2.codes), (v1.status, v2.status)
        for b1, b2 in zip((*v1.vu_boundaries, *v1.cu_boundaries), (*v2.vu_boundaries, *v2.cu_boundaries), strict=True):
            assert b1.inputs == b2.inputs and b1.outputs == b2.outputs
        stages.run(f"boundary-perleaf[{spec}]", lambda row: row.update(boundaries=tb.assert_boundaries_equal_perleaf(c1, plan1)))
        stages.run(f"escape-perleaf[{spec}]", lambda row: row.update(checked=tb.assert_escape_runs_equal_perleaf(c1, plan1, compact.program)))
        if v1.status != "valid":
            checks[spec] = {"status": v1.status, "boundaries_equal": True}
            continue
        stages.run(f"projection-perleaf[{spec}]", lambda row: row.update(checked=tb.assert_projections_equal_perleaf(c1, plan1, v1)))
        stages.run(f"reach-perleaf[{spec}]", lambda row: tb.assert_reach_equal_perleaf(c1, plan1, compact.program))
        t1 = stages.run(f"table[{spec}]", lambda row: hierarchical_table(c1, plan1, v1))
        t2 = stages.run(f"table-expanded[{spec}]", lambda row: hierarchical_table(c2, plan2, v2))

        def perleaf(row):
            mp = pytest.MonkeyPatch()
            with mp.context() as m:
                m.setattr(bound_module, "Intrinsic", pr.PerLeafIntrinsic)
                m.setattr(bound_module, "Reach", pr.PerLeafReach)
                return hierarchical_table(c1, plan1, v1)

        t3 = stages.run(f"table-perleaf[{spec}]", perleaf)
        rows_equal = t1.table.rows == t2.table.rows == t3.table.rows
        digest_equal = t1.table.digest == t2.table.digest == t3.table.digest
        bits = {}
        for name, (q, s, d) in {"cert": (args.q, args.s, args.delta), "bridge": (Fraction(1, 2), Fraction(1, 2), Fraction(1, 64))}.items():
            b = [fold(t.table, q, s, d) for t in (t1, t2, t3)]
            bits[name] = {"bits": [float(x.bits) for x in b], "equal": b[0].bits == b[1].bits == b[2].bits}
        checks[spec] = {"status": v1.status, "boundaries_equal": True, "rows_equal": rows_equal, "digest_equal": digest_equal,
                        "digest": t1.table.digest, "rows": len(t1.table.rows), "bound": bits}
        assert rows_equal and digest_equal and all(v["equal"] for v in bits.values()), checks[spec]
    doc["oracle"] = checks
    stages.dump("oracle: equal")
    print(json.dumps(checks, indent=1))
    return 0


# -- main ------------------------------------------------------------------------------------------------------------------


def classify_input(name: str | None, gates: int, lp: int) -> str:
    n = (name or "").lower()
    if gates == lp or "prompt" in n or "token" in n or "ids" in n:
        return "prompt"
    if "cos" in n or "sin" in n or "rope" in n or "cache" in n or "pos" in n:
        return "other"
    return "weight"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="B0")
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--nl", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=INT")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.json, <out>.rss.tsv, <out>.stages.json, <out>.passdiff.txt)")
    ap.add_argument("--record", default=str(POC / "results" / "INT-PAD-2" / RECORD_ID))
    ap.add_argument("--plans", default="root", help="comma list: root (bridge P1), P3/cu=whole, P3/cu=operator, ...")
    ap.add_argument("--q", type=Fraction, default=Fraction(2, 4), help="certificate's q_declared")
    ap.add_argument("--s", type=Fraction, default=Fraction(1, 4096), help="certificate's s_declared")
    ap.add_argument("--delta", type=Fraction, default=Fraction(1, 10**6), help="bound_record DELTAS[0]")
    ap.add_argument("--no-digest", action="store_true", help="skip encode_program/program_digest")
    ap.add_argument("--no-symbolic", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="equality oracle at this size instead of the run")
    ap.add_argument("--recompare", default="", help="path of a finished run's <out>.json: redo only the symbolic side + comparison")
    ap.add_argument("--sample-ms", type=int, default=1000)
    ap.add_argument("--recursion-limit", type=int, default=100_000)
    args = ap.parse_args(argv)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), args.recursion_limit))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(out.with_suffix(".rss.tsv"), args.sample_ms / 1000)
    sampler.start()
    doc: dict[str, Any] = {
        "schema": SCHEMA, "record": RECORD_ID, "case": args.case, "lp": args.lp, "steps": args.steps, "plans": args.plans,
        "policy": {"q": str(args.q), "s": str(args.s), "delta": str(args.delta)},
        "machine": {"cpu": cpu_model(), "nproc": os.cpu_count(), "platform": platform.platform(), "python": sys.version.split()[0],
                    "threads": 1, "state": "cold (fresh process; no caches persisted across runs)", "hostname": platform.node()},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stages = Stages(out.with_suffix(".json"), sampler, doc)
    try:
        if args.oracle:
            return oracle(args, stages, doc)
        if args.recompare:
            # reuse a finished run's evidence; redo only the (cheap) symbolic side and the comparison
            prior = json.loads(Path(args.recompare).read_text())
            doc.update({k: prior[k] for k in ("case", "lp", "steps", "plans", "config_overrides", "root_inputs", "plan_results") if k in prior})
            doc["recompare_of"] = args.recompare
            doc["prior_stages"] = prior.get("stages")
            config, _ = build_config(prior["case"], args.nl, args.set)
            program_row = next(r for r in prior["stages"] if r["stage"] == "program")
            if "root_census" in prior:
                doc["root_census"] = prior["root_census"]
            else:  # older run: rebuild the program (seconds, ~160 MB) for the root-node census only
                from veritor.plans.verity_ir_adapter import adapt

                def do_census(row):
                    program = build_program(config, prior["lp"], prior["steps"], False, {})
                    Q = adapt(program, compact=True).program
                    return dict(sorted(Counter(strip_digests(n.fn.name).split("{")[0] for n in Q.root.nodes).items()))

                doc["root_census"] = stages.run("root-census", do_census)
            sym = None if args.no_symbolic else stages.run("symbolic", lambda row: symbolic_side(config, prior["lp"], prior["steps"], args.q, args.s, args.delta))
            doc["symbolic"] = sym
            rec = load_record(Path(args.record))
            items = stages.run("compare", lambda row: compare(doc, rec, sym, program_row, prior["plan_results"], prior["root_inputs"], args.q, args.s, args.delta))
            doc["comparison"] = items
            lines = [f"{it['verdict']:4s} ({it['obligation']}) {it['item']}" for it in items]
            out.with_suffix(".passdiff.txt").write_text("\n".join(lines) + "\n")
            print("\n".join(lines))
            stages.dump("complete (recompare)")
            sampler.stop.set()
            return 0

        from veritor.plans.bound import hierarchical_table
        from veritor.plans.validate import validate_plan
        from veritor.plans.verity_ir_adapter import adapt

        config, overrides = build_config(args.case, args.nl, args.set)
        doc["config_overrides"] = overrides
        program_row: dict[str, Any] = {}
        program = stages.run("program", lambda row: (build_program(config, args.lp, args.steps, not args.no_digest, row), program_row.update(row))[0],
                             case=args.case, LP=args.lp, STEPS=args.steps)

        def do_adapt(row):
            a = adapt(program, compact=True)
            row.update(ir_sizes(a.program))
            row["compaction"] = a.compaction.to_json()
            row["unrolled"] = len(a.unrolled)
            return a

        adapted = stages.run("adapt", do_adapt, gates=program.gates)
        Q = adapted.program
        circuit = stages.run("circuit", lambda row: (lambda c: (row.update(size=c.size), c)[1])(Q.circuit()), root_nodes=len(Q.root.nodes))
        root_inputs = [{"index": k, "name": n.name, "gates": n.n, "kind": classify_input(n.name, n.n, args.lp)}
                       for k, n in enumerate(Q.root.nodes) if n.form == "batch" and n.fn.name == "Input"]
        doc["root_inputs"] = root_inputs
        doc["root_census"] = dict(sorted(Counter(strip_digests(n.fn.name).split("{")[0] for n in Q.root.nodes).items()))

        plan_results: dict[str, Any] = {}
        doc["plan_results"] = plan_results
        for spec in args.plans.split(","):
            res: dict[str, Any] = {"circuit_size": circuit.size}
            plan_results[spec] = res
            plan, meta = stages.run(f"plan[{spec}]", lambda row: (lambda pm: (row.update(vus=len(pm[0].verification_units), cus=len(pm[0].commitment_units)), pm)[1])(
                make_plan(spec, Q, circuit, adapted)), gates=circuit.size)
            res.update(plan=plan.name, vus=len(plan.verification_units), cus=len(plan.commitment_units), **meta)
            v0, h0 = circuit.visits, circuit.hops

            def do_validate(row):
                v = validate_plan(circuit, plan)
                row.update(status=v.status, codes=[str(x) for x in v.codes], visits=v.visits, hops=v.hops, escape_steps=v.escape_steps,
                           vu_edges=v.vu_edges, cu_edges=v.cu_edges, circuit_visits=circuit.visits - v0, circuit_hops=circuit.hops - h0)
                return v

            verdict = stages.run(f"validate[{spec}]", do_validate, gates=circuit.size, vus=len(plan.verification_units), cus=len(plan.commitment_units))
            res.update(status=verdict.status, codes=[str(x) for x in verdict.codes], reasons=[list(r) for r in verdict.reasons[:5]],
                       validate_seconds=stages.rows[-1]["seconds"], validate_maxrss_mb=stages.rows[-1]["maxrss_mb"])
            res["boundary"] = stages.run(f"boundary[{spec}]", lambda row: (lambda st: (row.update(st), st)[1])(boundary_stats(verdict)),
                                         units=len(verdict.vu_boundaries) + len(verdict.cu_boundaries))
            if verdict.status != "valid":
                stages.dump(f"{spec}: {verdict.status}")
                continue

            def do_bound(row):
                t = time.perf_counter()
                result = hierarchical_table(circuit, plan, verdict)
                b = fold(result.table, args.q, args.s, args.delta)
                seconds = round(time.perf_counter() - t, 3)
                body = bound_report_body(program_row.get("function", Q.name), program, plan, meta, verdict, result, args.q, args.s, args.delta, b, seconds, circuit)
                body["peak_rss_bytes"] = int(maxrss_mb() * (1 << 20))
                body["report_digest"] = hashlib.sha256(json.dumps({k: v for k, v in body.items() if k not in ("peak_rss_bytes", "work")},
                                                                  sort_keys=True, default=str).encode()).hexdigest()
                other = {}
                for name, (q, s, d) in {"cert-q1": (Fraction(1), args.s, args.delta), "bridge": (Fraction(1, 2), Fraction(1, 2), Fraction(1, 64)),
                                        "delta-1e-12": (args.q, args.s, Fraction(1, 10**12))}.items():
                    bb = fold(result.table, q, s, d)
                    other[name] = {"q": str(q), "s": str(s), "delta": str(d), "U_bits": float(bb.bits), "capped": bool(bb.capped)}
                rows = result.table.rows
                root_row = next((r for r in rows if r.kind == result.table.root), None)
                by_fn: Counter = Counter()
                for r in rows:
                    if r.role == "verification":
                        by_fn[strip_digests(r.kind).rsplit(":", 1)[0]] += r.copies
                row.update(rows=len(rows), vu_kinds=result.vu_kinds, cu_kinds=result.cu_kinds, U_bits=float(b.bits), capped=bool(b.capped),
                           output_bits=body["output_bits"], reach_steps=result.steps, intrinsic_entries=result.intrinsic_entries,
                           narrowed_units=result.narrowed_units, fallback_units=result.fallback_units)
                return {"U_bits": float(b.bits), "capped": bool(b.capped), "output_bits": body["output_bits"], "digest": result.table.digest,
                        "rows": len(rows), "other_policies": other,
                        "root_row": None if root_row is None else {"kind": root_row.kind, "size": root_row.size, "out_bits": root_row.out_bits,
                                                                   "source_inputs": root_row.source_inputs, "source_weights": root_row.source_weights},
                        "report": body}, dict(sorted(by_fn.items()))

            res["bound"], res["kind_copies_by_function"] = stages.run(f"bound[{spec}]", do_bound, gates=circuit.size, vus=len(plan.verification_units))
            stages.dump(f"{spec}: bounded")

        sym = None
        if not args.no_symbolic:
            def do_sym(row):
                s_ = symbolic_side(config, args.lp, args.steps, args.q, args.s, args.delta)
                row.update(gates=s_["gates"], tables=list(s_["tables"]))
                return s_

            try:
                sym = stages.run("symbolic", do_sym)
            except Exception as e:  # noqa: BLE001
                doc["symbolic_error"] = f"{type(e).__name__}: {e}"[:400] + "\n" + traceback.format_exc()[-1500:]
        doc["symbolic"] = sym
        rec = load_record(Path(args.record))
        items = stages.run("compare", lambda row: compare(doc, rec, sym, program_row, plan_results, root_inputs, args.q, args.s, args.delta))
        doc["comparison"] = items
        lines = [f"{it['verdict']:4s} ({it['obligation']}) {it['item']}" for it in items]
        out.with_suffix(".passdiff.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))

        vb = {"schema": STAGES_SCHEMA, "lane": "COMPACT-2", "record": RECORD_ID, "program": program_row.get("function"),
              "gates": program.gates, "machine": doc["machine"], "policy": doc["policy"],
              "stages": [{k: v for k, v in r.items() if k not in ("compaction",)} for r in stages.rows],
              "note": "wall seconds per stage on one Python thread; RSS = psutil rss before/after, maxrss = ru_maxrss so far; "
                      "input sizes in `inputs` and the stage's own row fields (gates, nodes, ref_runs, ref_leaves, vus, cus)"}
        out.with_suffix(".stages.json").write_text(json.dumps(vb, indent=1, default=str) + "\n")
        stages.dump("complete")
        sampler.stop.set()
        return 0
    except MemoryError:
        stages.dump("unsupported: MemoryError", error="MemoryError", traceback=traceback.format_exc()[-2000:])
        sampler.stop.set()
        return 3
    except Exception as e:  # noqa: BLE001
        stages.dump(f"error: {type(e).__name__}", error=f"{type(e).__name__}: {e}"[:400], traceback=traceback.format_exc()[-3000:])
        sampler.stop.set()
        print(traceback.format_exc(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
