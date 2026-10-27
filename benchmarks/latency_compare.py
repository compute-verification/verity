#!/usr/bin/env python3
"""The latency table for two (or more) result envelopes that carry ``rounds.sequential_depth``.

    uv run python benchmarks/latency_compare.py RUN_A RUN_B [RUN_C ...] [--bw-gbps 25] [--capacity-gb 256] [--strict] [--json]

``RUN`` is a run id (or unique prefix) under ``~/.research/runs`` (``$RESEARCH_RUNS``), or a directory with a
``result.json`` (``benchmarks/judge.py`` conventions).  For every envelope: prover compute (``t.total``), verifier
compute, the measured depth, bytes, memory per batch; then the end-to-end wall time per batch at the contract's RTT sweep
from the model ``prove + verify + depth x RTT + bytes / bw`` (``verity_numerical.bench.latency``) beside the envelope's
own ``t.e2e.rtt_<x>ms`` (measured at the run's emulated RTT, predicted at the others -- the fingerprint's ``latency``
block says which), the in-flight batches that saturate the prover and whether they fit in memory, and the crossover RTT
between every deeper/shallower pair.

The judge's identity check (``contract.incomparable``) is run and printed: the latency axis is per design, so the table
is printed even for runs that are not the same campaign number, with the disagreement named; ``--strict`` refuses
instead (exit 2), as the judge would.  Exit 1: a run could not be read or lacks the depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for candidate, module in ((REPO, "verity"), (REPO / "benchmarks" / "numerical" / "python", "verity_numerical")):
    try:
        __import__(module)
    except ImportError:
        sys.path.insert(0, str(candidate))
sys.path.insert(0, str(REPO / "benchmarks"))

from judge import load, resolve, runs_root
from verity_numerical.bench import contract
from verity_numerical.bench import latency as L

EX_REFUSED = 2


def design_of(run_dir: Path, capacity_bytes: float) -> tuple[L.Design, dict[str, Any]]:
    doc = load(run_dir)
    fp = doc.get("workload_fingerprint") or {}
    label = f"{(fp.get('software') or {}).get('backend', {}).get('name', '?')} B={fp.get('B')} {doc.get('run_id')}"
    return L.from_envelope(label, doc, mem_capacity_bytes=capacity_bytes, source=str(run_dir)), doc


def table(designs: list[tuple[L.Design, dict[str, Any]]], bw: float, rtts=L.RTTS_MS) -> dict[str, Any]:
    rows = []
    for d, doc in designs:
        m = {x["name"]: x["value"] for x in doc.get("measurements", []) if isinstance(x, dict)}
        lat = (doc.get("workload_fingerprint") or {}).get("latency") or {}
        rows.append({
            "design": d.name, "proof_class": d.proof_class, "hardware": d.hardware, "B": d.B, "prove_s": d.prove_s, "verify_s": d.verify_s,
            "depth": d.depth, "protocol_rounds": d.protocol_rounds, "bytes": d.bytes_total, "mem_per_batch": d.mem_per_batch_bytes,
            "rtt_ms_run": lat.get("rtt_ms"), "e2e_measured_keys": lat.get("e2e_measured", []),
            "model": {f"{r:g}": L.e2e(d, r, bw)["total"] for r in rtts},
            "envelope": {f"{r:g}": m.get(f"t.e2e.rtt_{r:g}ms") for r in rtts},
            "pipeline": {f"{r:g}": L.pipeline(d, r, bw) for r in rtts},
            "memory_binding_rtt_ms": L.memory_binding_rtt(d, bw),
        })
    cross = []
    for a, _ in designs:
        for b, _ in designs:
            if a.depth > b.depth:
                cross.append({"deeper": a.name, "shallower": b.name, "crossover_rtt_ms": L.crossover_rtt(a, b, bw)})
    return {"bw_gbps": bw, "rows": rows, "crossovers": cross}


def render(rep: dict[str, Any], identity: list[dict[str, Any]]) -> str:
    out = ["[identity]  (the judge's fields; the latency axis is per design, so the table follows even when they differ)"]
    for row in identity:
        out.append(f"  {row['field']:<16} {'same' if row['same'] else 'DIFFERS: ' + ' vs '.join(str(v)[:40] for v in row['values'])}")
    out.append("")
    out.append(f"[latency]  per batch, link {rep['bw_gbps']:g} Gb/s; model = t.total + verifier.seconds + depth x RTT + bytes / bw; 'env' = the envelope's t.e2e.rtt_*ms (M = measured at that RTT)")
    hdr = f"  {'design':<58} {'B':>5} {'prove s':>9} {'verify s':>9} {'depth':>6} {'bytes':>10} " + " ".join(f"{'@' + k + 'ms':>11}" for k in rep["rows"][0]["model"])
    out.append(hdr)
    for r in rep["rows"]:
        meas = {k.split("rtt_")[1].rstrip("ms") for k in r["e2e_measured_keys"]}
        out.append(f"  {r['design'][:58]:<58} {r['B']:>5} {r['prove_s']:>9.3f} {r['verify_s']:>9.3f} {r['depth']:>6} {r['bytes']:>10.0f} "
                   + " ".join(f"{r['model'][k]:>11.3f}" for k in r["model"]) + "   model")
        out.append(f"  {'':<58} {'':>5} {'':>9} {'':>9} {'':>6} {'':>10} "
                   + " ".join((f"{r['envelope'][k]:>10.3f}" + ("M" if k in meas else " ")) if r["envelope"][k] is not None else f"{'-':>11}" for k in r["model"]) + "   env")
    out.append("")
    out.append("[pipelining]  N_sat = ceil(t.e2e / t.total) in-flight batches saturate the prover; memory = N_sat x mem/batch")
    for r in rep["rows"]:
        cells = []
        for k, p in r["pipeline"].items():
            cells.append(f"@{k}ms N={p['n_saturate']} ({p['n_saturate'] * r['mem_per_batch'] / 1e9:.0f} GB{'' if p['memory_fits'] or p['n_inflight'] == p['n_saturate'] else ' > capacity'}) {p['vu_per_s']:.3g} VU/s {p['binding']}")
        mb = r["memory_binding_rtt_ms"]
        out.append(f"  {r['design'][:58]:<58} mem/batch {r['mem_per_batch'] / 1e9:.1f} GB; memory binds above " + ("never" if mb is None else f"{mb:.1f} ms"))
        out += [f"      {c}" for c in cells]
    out.append("")
    out.append("[crossovers]  RTT above which the shallower protocol has the lower serial e2e")
    for c in rep["crossovers"]:
        out.append(f"  {c['deeper'][:50]:<50} -> {c['shallower'][:50]:<50} " + ("shallower already faster at 0 ms" if c["crossover_rtt_ms"] is None else f"{c['crossover_rtt_ms']:.3g} ms"))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--bw-gbps", type=float, default=25.0)
    ap.add_argument("--capacity-gb", type=float, default=256.0, help="prover memory for the pipelining rows (default the vy-cpu2 host)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--strict", action="store_true", help="refuse (exit 2) when the runs are not the same campaign number")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    root = runs_root(a.runs_dir)
    try:
        designs = [design_of(Path(r).parent if r.endswith("result.json") else resolve(r, root), a.capacity_gb * 1e9) for r in a.runs]
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"latency_compare: {e}", file=sys.stderr)
        return 1
    fps = [d[1].get("workload_fingerprint") or {} for d in designs]
    ids = [contract.identity(fp) for fp in fps]
    identity = [{"field": f, "same": all(i[f] is not contract.MISSING and i[f] == ids[0][f] for i in ids),
                 "values": ["<missing>" if i[f] is contract.MISSING else i[f] for i in ids]} for f in contract.IDENTITY_FIELDS]
    rep = table(designs, a.bw_gbps)
    rep["identity"] = identity
    refused = [row["field"] for row in identity if not row["same"]]
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(render(rep, identity))
        if refused:
            print(f"\nNOT the same campaign number ({', '.join(refused)} differ): the latency rows are per design, not a ratio between them" + ("; --strict refuses" if not a.strict else ""))
    return EX_REFUSED if (a.strict and refused) else 0


if __name__ == "__main__":
    sys.exit(main())
