"""The A GPU prover (gpu.run prove) + the independent Rust verifier on EVERY proof -> $RESEARCH_RUN_DIR/result.json.

    cd backends/gkr && python bench_result.py /workspace/bb/pos4096 --vus 4096 --checker v1 --path graphed --reps 3 --warmup 1 \
        --verifier /workspace/bin/verity-gkr-verify --threads 16

A thin wrapper (lane h100-bestvsbest) around `gpu.run`'s `load_instance` / `prove_once`: every recorded rep's proof is written as
`proofs/rep<i>.bin` (README "Proof file") and checked by `verity-gkr-verify verify` (backends/gkr/verifier: shares no code with the
prover; the Python verifier in `prover.verify` does), and one result.json (research/result/v0.1, kind bench-result/v1) is written
with the same measurement names as `backends/direct/ligero/bench_result.py` (prove_wall, verify_wall, verify_python_wall, proof_bytes,
depth, soundness_bits, e2e_0ms).  `--path` selects the kernel path by the prover's own environment switches (nothing in the prover
changes): `graphed` = packed LogUp + CUDA graphs (the default, a-gpu2 v6), `packed` = VERITY_GPU_NO_GRAPHS=1, `torch` =
VERITY_GPU_TORCH_ONLY=1 (the bit-exact reference).  `--checker` must agree with the export's manifest (`variant`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# security.accounting.account_candidate_a (gpu/README.md "Soundness"; ledger a-gpu2 / a-protocol-diff / a-kernel-a100): BabyBear^6,
# Ligero k = 4096, n = 16384, t = 192, SHA-256 Fiat-Shamir; 2^-127.7, dominated by the generic SHA-256 collision term.  The GPU
# prover has no live accountant; the figure is the campaign's recorded one for this exact configuration (both checker variants
# use the same Ligero parameters and field).
SOUNDNESS_LOG2 = -127.7
FIELD = "babybear6"          # F_p[X]/(X^6 - 22), p = 2^31 - 2^27 + 1 (gpu/field.py)


def _m(name: str, value, unit: str) -> dict:
    return {"name": name, "value": value, "unit": unit}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python backends/gkr/bench_result.py")
    ap.add_argument("dir")
    ap.add_argument("--vus", type=int, default=None)
    ap.add_argument("--checker", choices=("v1", "v2"), default="v1", help="must match DIR/manifest.json (v2 exports carry variant=v2)")
    ap.add_argument("--path", choices=("graphed", "packed", "torch"), default="graphed")
    ap.add_argument("--field", default=FIELD)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--verifier", default=None, help="verity-gkr-verify binary (independent Rust verifier)")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)

    if args.path == "torch":
        os.environ["VERITY_GPU_TORCH_ONLY"] = "1"
    elif args.path == "packed":
        os.environ["VERITY_GPU_NO_GRAPHS"] = "1"
    if args.field != FIELD:
        print(f"bench_result.py: this prover's field is {FIELD}", file=sys.stderr)
        return 2
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or "backends/gkr/gpu/results/bench_result")
    run_id = os.environ.get("RESEARCH_RUN_ID") or f"local-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    proofs_dir = run_dir / "proofs"
    proofs_dir.mkdir(exist_ok=True)
    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text())
    variant = "v2" if man.get("variant") == "v2" else "v1"
    if variant != args.checker:
        print(f"bench_result.py: --checker {args.checker} but {d}/manifest.json says {variant}", file=sys.stderr)
        return 2

    import torch

    from gpu import prover
    from gpu.run import _dev, load_instance, prove_once

    dev = _dev(args.device)
    inst = load_instance(d, dev, args.vus, tile=False)
    pcs = not args.clear
    rows = []
    for _ in range(args.warmup):
        _p, _st, tp, tv, err = prove_once(inst, pcs)
        print(json.dumps({"warmup": True, "prover_s": tp, "verified": err is None, "error": err}), flush=True)
    for rep in range(args.reps):
        proof, st, tp, tv, err = prove_once(inst, pcs)
        pf = proofs_dir / f"rep{rep}.bin"
        data = proof.to_bytes()
        pf.write_bytes(data)
        row = {"rep": rep, "vus": inst.vus(), "units": inst.segs[0].units, "prover_s": tp, "python_verifier_s": tv, "python_verified": err is None,
               "error": err, "proof_bytes": len(data), "proof_sha256": hashlib.sha256(data).hexdigest(), "proof_file": f"proofs/rep{rep}.bin",
               "t_commit": st.t_commit, "t_lookup": st.t_lookup, "t_arith": st.t_arith, "t_open": st.t_open, "slots": st.slots,
               "sequential_depth": st.sequential_depth(pcs), "depth_circuit": st.depth_circuit, "depth_lookup_max": st.depth_lookup_max,
               "committed_elements": st.committed_elements, "ligero_rows": st.ligero_rows, "peak_mem_bytes": st.peak_mem}
        rows.append(row)
        print(json.dumps(row), flush=True)

    # ---- the independent verifier on every recorded proof ------------------------------------------------------------------
    vbin = Path(args.verifier) if args.verifier else None
    verdicts = []
    for row in rows:
        if vbin is None or not vbin.is_file():
            print(f"bench_result.py: no independent verifier at {args.verifier!r}", file=sys.stderr)
            break
        out_json = run_dir / f"verify_rep{row['rep']}.json"
        cmd = [str(vbin), "verify", "--dir", str(d), "--proof", str(run_dir / row["proof_file"]), "--threads", str(args.threads), "--json", str(out_json)]
        if args.vus is not None:
            cmd += ["--vus", str(args.vus)]
        t1 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t1
        doc = None
        try:
            doc = json.loads(out_json.read_text()) if out_json.is_file() else None
        except json.JSONDecodeError:
            doc = None
        accepted = bool(doc and doc.get("accepted")) and r.returncode == 0
        verdicts.append({"rep": row["rep"], "proof_sha256": row["proof_sha256"], "accepted": accepted, "rc": r.returncode, "error": (doc or {}).get("error"),
                         "verify_seconds": (doc or {}).get("verify_seconds"), "process_wall": wall, "threads": (doc or {}).get("threads"),
                         "cpu": (doc or {}).get("cpu"), "json": out_json.name, "stderr_tail": r.stderr[-300:]})
        print(f"verify rep{row['rep']}: {'accept' if accepted else 'REJECT'} {wall:.3f}s ({(doc or {}).get('error')})", flush=True)
    (run_dir / "verify_independent.json").write_text(json.dumps({"verifier": str(vbin), "verifier_sha256": _sha256(vbin) if vbin and vbin.is_file() else None,
                                                                 "threads": args.threads, "items": verdicts}, indent=1))
    (run_dir / "prove_reps.json").write_text(json.dumps({"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                                                         "pcs": pcs, "path": args.path, "checker": args.checker, "runs": rows}, indent=1))
    n_acc = sum(1 for v in verdicts if v["accepted"])
    all_rust_ok = bool(verdicts) and n_acc == len(rows)
    python_ok = all(r["python_verified"] for r in rows)
    prove_reps = [r["prover_s"] for r in rows]
    pyv_reps = [r["python_verifier_s"] for r in rows]
    rust_reps = [v["verify_seconds"] if v["verify_seconds"] is not None else v["process_wall"] for v in verdicts]
    prove_med = statistics.median(prove_reps)
    rust_med = statistics.median(rust_reps) if rust_reps else None
    B = rows[0]["vus"]
    fp = {"candidate": "A", "backend": "backends/gkr/gpu (a-gpu2 prover: packed GKR + graphed LogUp + Ligero, torch/Triton)",
          "commit": os.environ.get("RESEARCH_SOURCE_COMMIT"), "protocol": "GKR + LogUp + Ligero over BabyBear^6, backends/gkr/PROTOCOL.md (4.2 low-bit-first LogUp)",
          "mode": "non-interactive (SHA-256 Fiat-Shamir), non-ZK", "zk": False, "proof_class": "NON_ZK_PROOF_DIAGNOSTIC",
          "checker": args.checker + (" (WP2 checker contracts)" if args.checker == "v1" else " (checker-min gadgets, limb epilogue)"),
          "path": args.path, "field": FIELD, "batch": B, "K": 1536, "steps": man.get("steps"), "export_manifest": {k: man.get(k) for k in ("tier", "range", "variant", "epilogue", "root_manifest_sha256")},
          "security": {"target": -128, "achieved_log2": SOUNDNESS_LOG2, "field": FIELD, "ligero": {"k": 4096, "n": 16384, "t": 192},
                       "hash": "SHA-256", "source": "security.accounting.account_candidate_a (recorded; no live accountant in the GPU prover)"},
          "hardware": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
          "software": {"torch": torch.__version__, "python": sys.version.split()[0]},
          "verifier": {"name": "verity-gkr-verify (backends/gkr/verifier, Rust, independent)", "path": str(vbin), "threads": args.threads,
                       "sha256": _sha256(vbin) if vbin and vbin.is_file() else None, "cpu": next((v["cpu"] for v in verdicts if v.get("cpu")), None)},
          "python_verifier": "gpu.prover.verify (shares field/transcript/logup helpers with the prover)", "warmup": args.warmup, "reps": args.reps}
    meas = [
        _m("prove_wall", prove_med, "s"), _m("prove_wall_min", min(prove_reps), "s"), _m("prove_wall_max", max(prove_reps), "s"),
        _m("prove_ms_per_vu", prove_med / B * 1e3, "ms"),
        _m("verify_wall", rust_med, "s"), _m("verify_python_wall", statistics.median(pyv_reps), "s"), _m("verifier_threads", args.threads, "count"),
        _m("proof_bytes", rows[0]["proof_bytes"], "B"), _m("depth", rows[0]["sequential_depth"], "count"), _m("soundness_bits", -SOUNDNESS_LOG2, "bits"),
        _m("e2e_0ms", (prove_med + rust_med) if rust_med is not None else None, "s"),
        _m("proofs_verified_independent", n_acc, "count"), _m("proofs_total", len(rows), "count"),
        _m("peak_device_bytes", max(r["peak_mem_bytes"] for r in rows), "B"),
        _m("split_lookup", statistics.median(r["t_lookup"] for r in rows), "s"), _m("split_arith", statistics.median(r["t_arith"] for r in rows), "s"),
        _m("split_open", statistics.median(r["t_open"] for r in rows), "s"), _m("split_commit", statistics.median(r["t_commit"] for r in rows), "s"),
    ]
    meas = [m for m in meas if m["value"] is not None]
    status = "passed" if (python_ok and all_rust_ok) else "failed"
    result = {"schema": "research/result/v0.1", "run_id": run_id, "kind": "bench-result/v1", "workload_fingerprint": fp,
              "validation": {"status": status, "checker": "backends/gkr/verifier verify (independent Rust) on every recorded rep's proof; gpu.prover.verify per rep",
                             "detail": f"python: {sum(r['python_verified'] for r in rows)}/{len(rows)}; rust: {n_acc}/{len(rows)} accepted; "
                                       f"reps prove {['%.3f' % v for v in prove_reps]} s, rust verify {['%.3f' % v for v in rust_reps]} s",
                             "evidence": {"reps": [{"prover_total": r["prover_s"], "verifier": r["python_verifier_s"]} for r in rows], "rust_reps": rust_reps}},
              "measurements": meas, "measurement_files": ["prove_reps.json", "verify_independent.json"] + [v["json"] for v in verdicts], "artifacts": []}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, run_dir / "result.json")
    print(json.dumps({"prove_wall": prove_med, "verify_wall": rust_med, "proof_bytes": rows[0]["proof_bytes"], "depth": rows[0]["sequential_depth"],
                      "rust_accepted": f"{n_acc}/{len(rows)}", "status": status}, indent=1))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
