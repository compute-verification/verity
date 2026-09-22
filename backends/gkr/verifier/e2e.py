"""A-GPU end-to-end latency with the independent Rust verifier.

Joins the a-gpu prover record (``gpu.run prove --json``: ``prover_s``, ``sequential_depth``, ``slots``, ``proof_bytes``,
``peak_mem_bytes``) with the Rust verifier's record (``verity-gkr-verify verify --json``: ``verify_seconds``,
``bytes_read``) into a ``result.json`` envelope that ``benchmarks/latency_compare.py`` consumes (``t.total``,
``verifier.seconds``, ``rounds.sequential_depth``, ``rounds.protocol``, ``transcript.bytes``, ``mem.peak_device_bytes``),
and prints the serial e2e at the contract's RTTs from ``verity_numerical.bench.latency`` (prove + verify + depth x RTT +
bytes / bw) plus the verifier's own share and ``overhead`` (seconds per VU over the native reference).

    python -m verifier.e2e --prover run4096.json --verifier verify4096_12t.json --out envelope.json [--python-verifier]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity_numerical.bench import latency as L

RTTS = (0.0, 1.0, 10.0, 50.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prover", required=True, help="gpu.run prove --json record (list of reps or one row)")
    ap.add_argument("--verifier", required=True, help="verity-gkr-verify verify --json record")
    ap.add_argument("--out", default=None, help="write the result.json envelope here")
    ap.add_argument("--name", default="A-GPU H100 BabyBear^6 + independent Rust verifier")
    ap.add_argument("--bw", type=float, default=10.0, help="link Gb/s")
    ap.add_argument("--gpu", default="H100 80 GB")
    args = ap.parse_args(argv)

    prov = json.loads(Path(args.prover).read_text())
    if isinstance(prov, dict) and "runs" in prov:
        prov = prov["runs"]
    if isinstance(prov, list):
        prov = sorted(prov, key=lambda r: r["prover_s"])[len(prov) // 2]  # median rep
    ver = json.loads(Path(args.verifier).read_text())
    if not ver["accepted"]:
        raise SystemExit("the verifier did not accept the proof")
    b = int(prov["vus"])
    depth = int(prov["sequential_depth"])
    slots = int(prov["slots"])
    mem = float(prov.get("peak_mem_bytes") or 0) or 18.6e9
    env = {
        "schema": "verity/result/v1",
        "workload_fingerprint": {"B": b, "K": 1536, "proof_class": "NON_ZK_PROOF_DIAGNOSTIC", "field": "babybear6",
                                 "hardware": {"gpu": {"name": args.gpu}, "verifier_cpu": ver.get("cpu")},
                                 "latency": {"rtt_ms": 0.0, "e2e_measured": []}},
        "measurements": [
            {"name": "t.total", "value": float(prov["prover_s"]), "unit": "s"},
            {"name": "verifier.seconds", "value": float(ver["verify_seconds"]), "unit": "s"},
            {"name": "verifier.threads", "value": int(ver["threads"]), "unit": "count"},
            {"name": "verifier.peak_rss_bytes", "value": ver.get("peak_rss_bytes"), "unit": "bytes"},
            {"name": "rounds.sequential_depth", "value": depth, "unit": "count"},
            {"name": "rounds.protocol", "value": slots, "unit": "count"},
            {"name": "transcript.bytes", "value": int(ver["bytes_read"]), "unit": "bytes"},
            {"name": "mem.peak_device_bytes", "value": mem, "unit": "bytes"},
        ],
        "sources": {"prover": args.prover, "verifier": args.verifier},
    }
    d = L.from_envelope(args.name, env, mem_capacity_bytes=L.GPU_H100_MEM, source=args.prover)
    rows = []
    for rtt in RTTS:
        e = L.e2e(d, rtt, args.bw)
        rows.append({"rtt_ms": rtt, **e, "verifier_share": e["verify"] / e["total"]})
    compute = d.prove_s + d.verify_s
    overhead = {
        "verifier_seconds_per_vu": d.verify_s / b,
        "verifier_overhead_vs_native": (d.verify_s / b) / L.NATIVE_S_PER_VU,
        "prover_plus_verifier_overhead_vs_native": (compute / b) / L.NATIVE_S_PER_VU,
        "verifier_share_of_compute": d.verify_s / compute,
    }
    env["latency"] = {"rtts_ms": list(RTTS), "bw_gbps": args.bw, "rows": rows, "overhead": overhead}
    if args.out:
        Path(args.out).write_text(json.dumps(env, indent=1) + "\n")
    print(f"{args.name}: B={b} prove {d.prove_s:.3f} s, verify {d.verify_s:.3f} s ({ver['threads']} threads), depth {depth}, "
          f"{slots} slots, {d.bytes_pv / 1e6:.1f} MB")
    print(f"  {'RTT ms':>7} {'e2e s':>9} {'network':>9} {'verify share':>13}")
    for r in rows:
        print(f"  {r['rtt_ms']:>7g} {r['total']:>9.3f} {r['network']:>9.3f} {r['verifier_share']:>12.1%}")
    print(f"  verifier: {overhead['verifier_seconds_per_vu'] * 1e3:.3f} ms/VU = {overhead['verifier_overhead_vs_native']:.3g}x native; "
          f"share of compute {overhead['verifier_share_of_compute']:.1%}; prover+verifier {overhead['prover_plus_verifier_overhead_vs_native']:.3g}x native")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
