"""Rows (contract section 9) for W3c's lanes, appended to an existing experiment directory.

Reads the experiment's Table 4 row (common fields, instrumentation mode), the coordinate-check
summary/JSONL, the layout breakdown and the synthesised-slice result, and appends:

* Table 2: one ``independent`` row per (GEMM family, phase) for the coordinate VUs (decomposition
  ``coordinate``; ``op_id`` is the family's op pattern, the per-op ids are in the JSONL), kept
  apart from the pipeline's ``same_runtime`` rows;
* Table 3: ``N02-coordinate``, ``15.2b-x``, ``15.2b-w`` and the executable partition's
  ``veritor`` negatives;
* Table 4: one row per operator family (``partition_id = row-default/family=<f>``) with the
  family's leaves and bytes, and one row per weights layout (v1 / v2) with the leaf counts and
  opened bytes per check in ``restrictions``.

    PYTHONPATH=. python -m verity_vllm_adapter.w3c_rows <experiment_dir> --coordinate <dir> --layout <json> --synth <json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from verity_vllm import rows as R

FAMILY_OP = {
    "qkv_proj": "layers.*.self_attn.qkv_proj.gemm",
    "o_proj": "layers.*.self_attn.o_proj.gemm",
    "gate_up_proj": "layers.*.mlp.gate_up_proj.gemm",
    "down_proj": "layers.*.mlp.down_proj.gemm",
    "lm_head": "lm_head.gemm",
}


def _append(path: Path, new_rows: list[dict[str, Any]]) -> None:
    existing = R.read_jsonl(path) if path.exists() else []
    R.write_jsonl(path, existing + new_rows)


def _synth_rows(common: R.Common, synth: dict[str, Any] | None, wanted: set[str] | None) -> list[dict[str, Any]]:
    """Table 3 rows for the tampered-interface-word run of every executed synthesised slice."""
    rows: list[dict[str, Any]] = []
    if synth is None:
        return rows
    for sname, sl in synth.get("slices", {}).items():
        if wanted is not None and sname not in wanted:
            continue
        for cap, ex in sl.get("executed", {}).items():
            if "error" in ex:
                continue
            tam = ex["protocol_tampered_interface_word"]
            rows.append(R.require_valid(3, {**common.fields(), "negative_id": f"veritor-N02-{sname}-cap{cap}", "attempted": 1, "rejected": int(not tam["accepted"]), "unexpectedly_accepted": int(tam["accepted"]), "reproducer": f"verity_vllm_synth.emit.protocol_run(tamper_gate=<first computed unit's output>): one bit of a committed interface word of the unit-aligned circuit ({sl.get('gemm_op', sname)}, K={sl.get('K')}, {sl.get('columns', sl.get('count'))} coordinates), s=1; veritor verifier code {tam.get('code')}"}))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment_dir")
    ap.add_argument("--coordinate", default=None, help="coordinate_check output dir")
    ap.add_argument("--layout", default=None, help="layout_breakdown.json")
    ap.add_argument("--synth", default=None, help="synth_slice.json")
    ap.add_argument("--negatives-dir", default=None, help="coordinate_check output dir whose negatives to report (default: --coordinate)")
    ap.add_argument("--timing-dir", default=None, help="coordinate_check output dir run with OMP_NUM_THREADS=1 (clean per-coordinate relation time)")
    ap.add_argument("--worker", default="W3c")
    ap.add_argument("--synth-only", action="store_true", help="append only the Table 3 rows of --synth (a later slice run)")
    ap.add_argument("--synth-slices", default=None, help="comma-separated slice names of --synth to report (default: all)")
    args = ap.parse_args(argv)
    exp = Path(args.experiment_dir)
    t4 = R.read_jsonl(exp / "table4.jsonl")
    base = t4[0]
    common = R.Common(base["experiment_id"], base["case"], base["profile_id"], args.worker, base["adapter_revision"])
    mode = base["instrumentation_mode"]

    synth = json.loads(Path(args.synth).read_text()) if args.synth else None
    wanted = {x.strip() for x in args.synth_slices.split(",")} if args.synth_slices else None
    if args.synth_only:
        t3s: list[dict[str, Any]] = _synth_rows(common, synth, wanted)
        _append(exp / "table3.jsonl", t3s)
        print(json.dumps({"table2": 0, "table3": len(t3s), "table4": 0}))
        return 0
    if not (args.coordinate and args.layout):
        ap.error("--coordinate and --layout are required unless --synth-only")

    coord = json.loads((Path(args.coordinate) / "coordinate_summary.json").read_text())
    checks = [json.loads(l) for l in (Path(args.coordinate) / "coordinate_checks.jsonl").read_text().splitlines() if l.strip()]
    neg_dir = Path(args.negatives_dir or args.coordinate)
    negs = [json.loads(l) for l in (neg_dir / "coordinate_negatives.jsonl").read_text().splitlines() if l.strip()]
    coord["negatives_summary"] = json.loads((neg_dir / "coordinate_summary.json").read_text())["negatives_summary"]
    layout = json.loads(Path(args.layout).read_text())
    timing: dict[str, float] = {}
    if args.timing_dir:
        tsum = json.loads((Path(args.timing_dir) / "coordinate_summary.json").read_text())
        timing = {k: float(v["checker_seconds_per_coordinate"]) for k, v in tsum["by_family_phase"].items()}

    # --- Table 2: independent coordinate rows per (family, phase) ---
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in checks:
        by[(c["family"], c["phase"])].append(c)
    neg_by: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"rejected": 0, "attempted": 0})
    fam_of_cid = {c["canonical_id"]: (c["family"], c["phase"]) for c in checks}
    for n in negs:
        key = fam_of_cid.get(n["canonical_id"])
        if key is None or n.get("status") == "skipped":
            continue
        neg_by[key]["attempted"] += 1
        neg_by[key]["rejected"] += int(bool(n.get("rejected")))
    t2: list[dict[str, Any]] = []
    for (fam, phase), rs in sorted(by.items()):
        exact = sum(1 for r in rs if r["status"] == "exact")
        attempted = sum(1 for r in rs if r["status"] in ("exact", "mismatch"))
        unresolved = sum(1 for r in rs if r["status"] not in ("exact", "mismatch"))
        opened = [r["opened_bytes"].get("total", 0) for r in rs if r.get("opened_bytes")]
        worst = "mismatch" if any(r["status"] == "mismatch" for r in rs) else ("unresolved" if attempted == 0 else "exact")
        seconds = sum(r["checker_seconds"] for r in rs)
        K, N = rs[0]["K"], rs[0]["N"]
        ops = sorted({r["op"] for r in rs})
        t2.append(R.table2_row(
            common,
            op_id=FAMILY_OP.get(fam, fam),
            phase=phase,
            shape=[N],
            dtype="bfloat16",
            layout_or_quant=f"weights v2 (one leaf per parameter row, {coord.get('weights_leaf_count')} leaves)",
            runtime_impl=rs[0].get("kernel_path") or "matmul_kernel_persistent",
            primitive_semantics="gemm.bf16.coordinate.v1 (W4b independent_gemm on a [1,K]x[1,K] problem: Ampere/Ada mma.sync chain over ascending K slices, RNE)",
            decomposition="coordinate",
            interface={"input_bytes": int(round(sum(opened) / max(1, len(opened)))), "output_bits": 16, "range": None},
            origin="captured",
            binding={"present": attempted, "unresolved": unresolved},
            checker="independent" if attempted else "none",
            natural_checks={"exact": exact, "attempted": attempted},
            stress_checks={"exact": 0, "attempted": 0, "domain": None},
            mutations=dict(neg_by[(fam, phase)]),
            checker_work={"seconds": float(seconds), "flops_or_bytes": float(sum(opened)), "includes_full_op": False},
            status=worst,
            limitation=(f"{len(rs)} coordinates over {len(ops)} op instances ({len({r['layer'] for r in rs})} layers); K={K}; opened bytes = x row + W row leaf + out row + Merkle paths; "
                        f"checker = C++ chain model ({coord.get('model_version', '')[:60]}...) on arch {coord.get('arch')}; per-op ids in coordinate_checks.jsonl"
                        + (f"; checker_work.seconds is the relation call under OpenMP thread contention (shared box); with OMP_NUM_THREADS=1 it is {timing[f'{fam}/{phase}'] * 1e6:.0f} us per coordinate (coordinate_timing/)" if f"{fam}/{phase}" in timing else "")),
        ))
    _append(exp / "table2.jsonl", t2)

    # --- Table 3 ---
    t3: list[dict[str, Any]] = []
    ns = coord.get("negatives_summary", {})
    n02 = ns.get("N02", {})
    t3.append(R.require_valid(3, {**common.fields(), "negative_id": "N02-coordinate", "attempted": int(n02.get("attempted", 0)), "rejected": int(n02.get("rejected", 0)), "unexpectedly_accepted": int(n02.get("attempted", 0)) - int(n02.get("rejected", 0)), "reproducer": "coordinate_check.n02_coordinate: one ulp flipped in the committed output word (t, j), RU tree rebuilt over the mutation (authenticates), independent relation rejects; coordinate_negatives.jsonl"}))
    for which in ("x", "w"):
        ms = [n for n in negs if n.get("negative_id") == "15.2b" and n.get("operand") == which and n.get("status") != "skipped"]
        t3.append(R.require_valid(3, {**common.fields(), "negative_id": f"15.2b-{which}", "attempted": len(ms), "rejected": sum(1 for n in ms if n.get("rejected")), "unexpectedly_accepted": sum(1 for n in ms if not n.get("output_changed")), "reproducer": f"coordinate_check.mutate_operand({which}): one authenticated operand word of the {'input row' if which == 'x' else 'W row leaf'} perturbed (sign flip, exponent +3), trees rebuilt; the predicted coordinate must change; coordinate_negatives.jsonl"}))
    t3.extend(_synth_rows(common, synth, wanted))
    _append(exp / "table3.jsonl", t3)

    # --- Table 4: per family + weights layouts ---
    t4_new: list[dict[str, Any]] = []
    opened = layout.get("opened_bytes_per_check", {})
    for fam, d in layout["by_family"].items():
        notes = [f"family breakdown of experiment {common.experiment_id}: {d['ops']} ops, phases {json.dumps(d['phases'])}, row bytes {json.dumps(d['row_bytes'])}, share of committed bytes {d['share_of_bytes']:.4f}"]
        o = opened.get(fam)
        if o:
            notes.append(f"opened bytes per selected check: same_runtime row v1 {o['same_runtime_row']['v1']['total']} / v2 {o['same_runtime_row']['v2']['total']}; independent coordinate v1 {o['independent_coordinate']['v1']['total']} / v2 {o['independent_coordinate']['v2']['total']} (x{o['reduction_independent_v1_to_v2']:.0f} less); W {o['weight_bytes']} B, K={o['K']}, N={o['N']}")
            if o.get("measured_independent_v2"):
                notes.append("measured (coordinate_check): " + json.dumps(o["measured_independent_v2"]))
        t4_new.append(R.table4_row(
            common,
            partition_id=f"row-default/family={fam}",
            instrumentation_mode=mode,
            production_overhead_ratio=None, replay_commit_overhead_ratio=None, replay_vs_production_gpu_work=None,
            transparent_check_cost={"cpu_s": None, "gpu_s": None},
            peak_replay_memory={"device_gb": None, "host_gb": None, "delta_vs_reference": None},
            committed_logical_bytes=int(d["hashed_bytes"]),
            processed_bytes={"hashed": int(d["hashed_bytes"]), "copied": int(d["copied_bytes"]), "retained": int(d["retained_bytes"]), "emitted": None},
            description_size_bytes=None, constructor_inputs_bytes=None,
            logical_size={"gates": int(d["ops"]), "vus": int(d["leaves"]), "rus": None},
            lookup_cost_s=None, validation_cost_s=None,
            moe_padding={"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None},
            restrictions=notes,
            spend={"gpu_hours": None, "usd": None},
        ))
    tot = layout["totals"]
    for ver, leaves in (("v1", tot["weights_leaves_v1"]), ("v2", tot["weights_leaves_v2"])):
        t4_new.append(R.table4_row(
            common,
            partition_id=f"weights/{ver}",
            instrumentation_mode=mode,
            production_overhead_ratio=None, replay_commit_overhead_ratio=None, replay_vs_production_gpu_work=None,
            transparent_check_cost={"cpu_s": None, "gpu_s": None},
            peak_replay_memory={"device_gb": None, "host_gb": None, "delta_vs_reference": None},
            committed_logical_bytes=int(tot["weights_bytes"]),
            processed_bytes={"hashed": int(tot["weights_bytes"]), "copied": int(tot["weights_bytes"]), "retained": int(tot["weights_bytes"]), "emitted": None},
            description_size_bytes=None, constructor_inputs_bytes=None,
            logical_size={"gates": None, "vus": int(leaves), "rus": 1},
            lookup_cost_s=None, validation_cost_s=None,
            moe_padding={"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None},
            restrictions=[
                "weights tree leaves: v1 one per tensor; v2 one per parameter row (rows_per_leaf=1) for rank>=2 tensors, vectors whole (capture-format.md section 11)",
                f"opened bytes per check by family in layout_breakdown.json; conversion: {json.dumps(layout.get('conversion', {}))}",
            ],
            spend={"gpu_hours": None, "usd": None},
        ))
    _append(exp / "table4.jsonl", t4_new)
    print(json.dumps({"table2": len(t2), "table3": len(t3), "table4": len(t4_new)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
