"""`ligero-verify batch --jobs N` on a bench-vu proof dump -> $RESEARCH_RUN_DIR/result.json (research/result/v0.1, bench-result/v1).

    python archive/direct/ligero/verify_result.py --dir RUN/proofs --jobs 8 [--threads 1] [--reps 3] \
        --verifier /workspace/bin/ligero-verify [--prover-result RUN/result.json]

`--dir` is the `--dump-dir` tree `bench_result.py bench-vu` writes: `system.bin`, `manifest.json`, and one `repK/` of
`sub_XX.{stmt,proof[,coins]}` per recorded rep (the 25 independently committed sub-batch proofs of one 4096-VU batch).  For
every rep the batch is verified once as a warm-up and then `--reps` times; the batch verifier's in-process wall (shared system
parse + every sub-batch on `--jobs` OS threads, `--threads` inside each) is the measurement.  `verify_wall` = median over reps
of the per-rep median; `e2e_0ms` = the prover's `prove_wall` (from `--prover-result`, the bench-vu attempt whose dumps these
are) + `verify_wall`.  Nothing in the verifier is touched: the wrapper only launches the binary and reads its JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _m(name: str, value, unit: str) -> dict:
    return {"name": name, "value": value, "unit": unit}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def parse_args(argv: list[str]) -> dict:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--dir", required=True, help="the bench-vu --dump-dir tree (system.bin, manifest.json, rep*/)")
    ap.add_argument("--jobs", type=int, required=True, help="ligero-verify batch --jobs (0 = the binary's default: available parallelism)")
    ap.add_argument("--threads", type=int, default=1, help="ligero-verify --threads inside each sub-batch (default 1)")
    ap.add_argument("--reps", type=int, default=3, help="timed batch verifications per rep dir after one warm-up (default 3)")
    ap.add_argument("--verifier", required=True, help="path of the ligero-verify binary (where the check runs, not what is computed)")
    ap.add_argument("--prover-result", default=None, help="result.json of the bench-vu attempt that produced the dumps (prove_wall -> e2e_0ms)")
    ap.add_argument("--target-bits", type=float, default=128.0,
                    help="ligero-verify batch --target-bits: the batch union bound must meet 2^-target (PROTOCOL.md 8c; default 128)")
    return vars(ap.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    a = parse_args(list(sys.argv[1:] if argv is None else argv))
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or "backends/direct/ligero/results/verify_result")
    run_id = os.environ.get("RESEARCH_RUN_ID") or f"local-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(a["dir"])
    vbin = Path(a["verifier"])
    if not vbin.is_file():
        print(f"verify_result.py: no verifier at {vbin}", file=sys.stderr)
        return 2
    system = root / "system.bin"
    man = json.loads((root / "manifest.json").read_text())
    dset = man.get("set", {})
    rep_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("rep"))
    if not rep_dirs or not system.is_file():
        print(f"verify_result.py: {root} has no rep*/ dirs or no system.bin", file=sys.stderr)
        return 2
    prover = json.loads(Path(a["prover_result"]).read_text()) if a["prover_result"] else None
    prove_wall = None
    if prover:
        pm = {m["name"]: m["value"] for m in prover.get("measurements", [])}
        prove_wall = pm.get("prove_wall")

    v_dir = run_dir / "verify"
    v_dir.mkdir(exist_ok=True)
    jobs_arg = [] if a["jobs"] == 0 else ["--jobs", str(a["jobs"])]
    per_rep: list[dict] = []
    ok = True
    n_expected = int(dset.get("n_proofs") or 0)
    jobs_effective = None
    available = None
    batch_bits = None
    for rd in rep_dirs:
        walls, sums, procs = [], [], []
        for k in range(a["reps"] + 1):
            out = v_dir / f"batch_{rd.name}_{k}.json"
            cmd = [str(vbin), "batch", "--system", str(system), "--dir", str(rd), *jobs_arg, "--threads", str(a["threads"]),
                   "--target-bits", str(a["target_bits"]), "--json", str(out)]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            pw = time.perf_counter() - t0
            doc = json.loads(out.read_text()) if out.is_file() else {}
            accepted = bool(doc.get("batch_accepted")) and r.returncode == 0 and (n_expected == 0 or doc.get("n") == n_expected)
            ok = ok and accepted
            jobs_effective, available = doc.get("jobs"), doc.get("available_parallelism")
            batch_bits = doc.get("batch_bits", batch_bits)          # the batch union bound (binaries after b-batch-bound)
            print(f"{rd.name} run {k}{' (warm-up)' if k == 0 else ''}: {'ACCEPT' if accepted else 'REJECT'} n={doc.get('n')} jobs={doc.get('jobs')} "
                  f"wall={doc.get('wall_seconds')} sum={doc.get('verify_seconds_sum')} shared={doc.get('shared_seconds')} proc={pw:.3f}s rc={r.returncode} "
                  f"batch_bits={doc.get('batch_bits')} reason={doc.get('batch_reason')!r}", flush=True)
            if not accepted:
                print(r.stderr[-2000:], file=sys.stderr)
            if k == 0:
                continue                                    # warm-up: page cache, allocator
            walls.append(float(doc.get("wall_seconds", pw)))
            sums.append(float(doc.get("verify_seconds_sum", 0.0)))
            procs.append(pw)
        per_rep.append({"rep": rd.name, "n": doc.get("n"), "accepted": doc.get("accepted"), "rejected": doc.get("rejected"), "own_coins": doc.get("own_coins"),
                        "wall_seconds": walls, "verify_seconds_sum": sums, "process_wall": procs,
                        "wall_median": statistics.median(walls) if walls else None, "sum_median": statistics.median(sums) if sums else None,
                        "shared_seconds": doc.get("shared_seconds")})
    verify_wall = statistics.median([r["wall_median"] for r in per_rep])
    verify_sum = statistics.median([r["sum_median"] for r in per_rep])
    verify_wall_min = min(min(r["wall_seconds"]) for r in per_rep)
    proc_wall = statistics.median([statistics.median(r["process_wall"]) for r in per_rep])
    n_proofs = per_rep[0]["n"]

    fp = {"candidate": "B-Ligero", "component": "verifier", "backend": "archive/ligero-verify (independent Rust verifier, batch mode)",
          "commit": os.environ.get("RESEARCH_SOURCE_COMMIT") or (Path.cwd().name if len(Path.cwd().name) == 40 else None),
          "protocol": "Ligero direct-constraint chain proof, BabyBear, PROTOCOL.md 3-8c",
          "mode": dset.get("mode"), "zk": dset.get("zk"), "batch": dset.get("total_vus"), "K": 1536, "n_subbatches": n_proofs,
          "per_proof_vus": dset.get("per_proof"),
          "verifier": {"name": "ligero-verify batch (archive/ligero-verify, Rust, independent)", "path": str(vbin), "sha256": _sha256(vbin),
                       "jobs": jobs_effective, "jobs_requested": a["jobs"], "threads": a["threads"], "available_parallelism": available,
                       "target_bits": a["target_bits"], "batch_bits": batch_bits,
                       "coins": "each sub-batch's own step-0 coins (sub_XX.coins)" if dset.get("mode") == "interactive" else "n/a (Fiat-Shamir)"},
          "dumps": {"dir": str(root), "manifest_set": dset, "prover_run_id": prover.get("run_id") if prover else None,
                    "prover_commit": (prover or {}).get("workload_fingerprint", {}).get("commit"), "reps": [r.name for r in rep_dirs]},
          "hardware": {"cpu": _cpu_model(), "cpu_count": os.cpu_count(), "machine": platform.machine()},
          "software": {"python": platform.python_version()}, "warmup": "one batch per rep (excluded)", "reps": a["reps"]}
    meas = [
        _m("verify_wall", verify_wall, "s"), _m("verify_wall_min", verify_wall_min, "s"), _m("verify_seconds_sum", verify_sum, "s"),
        _m("verify_process_wall", proc_wall, "s"), _m("verifier_jobs", jobs_effective, "count"), _m("verifier_threads", a["threads"], "count"),
        _m("available_parallelism", available, "count"), _m("proofs_total", n_proofs, "count"),
        _m("proofs_verified_independent", sum(r["accepted"] or 0 for r in per_rep) // max(len(per_rep), 1), "count"),
    ]
    if batch_bits is not None:
        meas += [_m("batch_bits", batch_bits, "bits"), _m("target_bits", a["target_bits"], "bits")]
    if prove_wall is not None:
        meas += [_m("prove_wall", prove_wall, "s"), _m("e2e_0ms", prove_wall + verify_wall, "s")]
    result = {"schema": "research/result/v0.1", "run_id": run_id, "kind": "bench-result/v1", "workload_fingerprint": fp,
              "validation": {"status": "passed" if ok else "failed",
                             "checker": "archive/ligero-verify batch (independent Rust) on every recorded rep's 25 sub-batch proofs, every run",
                             "detail": f"{len(rep_dirs)} reps x (1 warm-up + {a['reps']}) batch verifications at jobs={jobs_effective} threads={a['threads']}; "
                                       f"all accepted: {ok}; wall medians {[round(r['wall_median'], 4) for r in per_rep]} s",
                             "evidence": {"per_rep": per_rep, "prove_wall": prove_wall}},
              "measurements": meas, "measurement_files": [f"verify/{p.name}" for p in sorted(v_dir.iterdir())], "artifacts": []}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, run_dir / "result.json")
    print(json.dumps({"verify_wall": verify_wall, "verify_seconds_sum": verify_sum, "jobs": jobs_effective, "threads": a["threads"], "prove_wall": prove_wall,
                      "e2e_0ms": (prove_wall + verify_wall) if prove_wall is not None else None, "status": result["validation"]["status"]}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
