"""bench-vu + the independent Rust verifier on every proof -> $RESEARCH_RUN_DIR/result.json (research/result/v0.1, bench-result/v1).

    python backends/direct/ligero/bench_result.py bench-vu [--zk] --mode interactive|fiat-shamir --batch 16384 --total-vus 4096 \
        --reps 3 --root /workspace/bench-instances/v1 --device cuda --verifier /workspace/bin/ligero-verify --threads 16

A thin wrapper (lane h100-bestvsbest): it runs `backends.direct.ligero.run bench-vu` in-process with `--out`, `--dump-dir` and
`--run-id` pointed at the run directory, then runs `ligero-verify verify` (backends/ligero-verify, shares no code with the prover)
on every recorded rep's sub-batch proof -- with `--coins` = the runner's step-0 coins in interactive mode (DISCREPANCIES D7: a
file proof is sound only against the verifier's own coins) -- and writes one result.json whose measurement names are the same as
`backends/gkr/bench_result.py`'s (prove_wall, verify_wall, verify_python_wall, proof_bytes, depth, soundness_bits, e2e_0ms) so the
two candidates are `research data select`-able side by side.  Nothing in the prover is touched.

Lane b-sweep: `--keep-proofs N` deletes every dumped sub-batch proof but the first N of each rep once the verifier has
judged them (the manifest keeps each proof's sha256 and size, `verify/independent.json` every verdict) -- a 2^20-VU rep is
~14 GB of proofs, more than a run tree should ship; `--dump-reps` (bench-vu) limits how many reps are written at all.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROOF_CLASS = {(True, "interactive"): "COMPLETE_ZK_BACKEND", (True, "fiat-shamir"): "COMPLETE_HVZK_BACKEND",
               (False, "interactive"): "NON_ZK_PROOF_DIAGNOSTIC", (False, "fiat-shamir"): "NON_ZK_PROOF_DIAGNOSTIC"}


def _m(name: str, value, unit: str) -> dict:
    return {"name": name, "value": value, "unit": unit}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commit(fallback: str | None = None) -> str | None:
    """The commit of the tree that is running: git when there is one; else `research run --on --source` ships `git archive HEAD`
    to `<root>/src/<commit sha>/` and makes it the cwd, so the cwd's name is the commit; else `RESEARCH_SOURCE_COMMIT`."""
    if fallback:
        return fallback
    env = os.environ.get("RESEARCH_SOURCE_COMMIT")
    if env:
        return env
    name = Path.cwd().name
    return name if len(name) == 40 and all(c in "0123456789abcdef" for c in name) else None


def _pop(argv: list[str], flag: str, default=None):
    if flag in argv:
        i = argv.index(flag)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def prune_proofs(proofs_dir: Path, manifest: dict, keep: int) -> dict:
    """After the independent verifier has run: delete the dumped .stmt/.proof/.coins of every sub-batch but the first
    ``keep`` of each rep (a 2^20-VU rep is ~14 GB; the manifest keeps every proof's sha256 and byte count, the verdicts
    stay in verify/independent.json).  Returns {"kept", "deleted", "bytes_deleted"}; manifest entries get "retained"."""
    kept = deleted = bytes_deleted = 0
    for e in manifest["files"]:
        retain = e["sub"] < keep
        e["retained"] = retain
        if retain:
            kept += 1
            continue
        for key in ("stmt", "proof", "coins"):
            p = proofs_dir / e[key] if e.get(key) else None
            if p is not None and p.is_file():
                bytes_deleted += p.stat().st_size
                p.unlink()
        deleted += 1
    return {"kept": kept, "deleted": deleted, "bytes_deleted": bytes_deleted}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or "backends/direct/ligero/results/bench_result")
    run_id = os.environ.get("RESEARCH_RUN_ID") or f"local-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    verifier = _pop(argv, "--verifier")
    threads = int(_pop(argv, "--threads", "1"))
    keep_proofs = _pop(argv, "--keep-proofs")          # retain only this many sub-batch proofs per dumped rep after verification
    keep_proofs = None if keep_proofs is None else int(keep_proofs)
    if "bench-vu" not in argv:
        print("bench_result.py: the subcommand must be bench-vu", file=sys.stderr)
        return 2
    proofs_dir = run_dir / "proofs"
    bench_json = run_dir / "bench_vu.json"
    for flag in ("--out", "--dump-dir", "--run-id"):
        if flag in argv:
            print(f"bench_result.py: {flag} is set by the wrapper", file=sys.stderr)
            return 2
    bench_argv = [*argv, "--out", str(bench_json), "--dump-dir", str(proofs_dir), "--run-id", run_id]
    print("bench_result.py: run.main", bench_argv, flush=True)

    from backends.direct.ligero.run import main as run_main
    t0 = time.perf_counter()
    rc = run_main(bench_argv)
    bench_wall = time.perf_counter() - t0
    if rc != 0 or not bench_json.is_file():
        print(f"bench_result.py: bench-vu failed rc={rc}", file=sys.stderr)
        return rc or 1
    bench = json.loads(bench_json.read_text())
    man = json.loads((proofs_dir / "manifest.json").read_text())
    fp_in = bench["workload_fingerprint"]
    sec = fp_in.get("security", {})
    mode, zk = sec.get("mode"), bool(sec.get("zk"))
    meas_in = {m["name"]: m["value"] for m in bench["measurements"]}
    reps = bench["validation"]["evidence"]["reps"]

    # ---- the independent verifier on every dumped proof --------------------------------------------------------------------
    verdicts = []
    v_dir = run_dir / "verify"
    v_dir.mkdir(exist_ok=True)
    vbin = Path(verifier) if verifier else None
    if vbin is None or not vbin.is_file():
        print(f"bench_result.py: no independent verifier at {verifier!r}; verify_wall not measured", file=sys.stderr)
    system = proofs_dir / "system.bin"
    for e in man["files"]:
        if vbin is None or not vbin.is_file():
            break
        stmt, proof = proofs_dir / e["stmt"], proofs_dir / e["proof"]
        cmd = [str(vbin), "verify", "--system", str(system), "--statement", str(stmt), "--proof", str(proof), "--threads", str(threads)]
        if e.get("coins"):
            cmd += ["--coins", str(proofs_dir / e["coins"])]
        t1 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t1
        out = r.stdout.strip().splitlines()
        doc = None
        for line in reversed(out):
            if line.startswith("{"):
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    doc = None
                break
        accepted = bool(doc and doc.get("accepted")) and r.returncode == 0
        verdicts.append({"rep": e["rep"], "sub": e["sub"], "proof": e["proof"], "proof_sha256": e["proof_sha256"], "proof_bytes": e["proof_bytes"],
                         "accepted": accepted, "rc": r.returncode, "reason": (doc or {}).get("reason"), "verify_seconds": (doc or {}).get("timing", {}).get("total"),
                         "process_wall": wall, "coins": bool(e.get("coins")), "stdout_tail": out[-1][:300] if out else r.stderr[-300:]})
        print(f"verify rep{e['rep']} sub{e['sub']:02d}: {'accept' if accepted else 'REJECT'} {wall:.3f}s ({(doc or {}).get('reason')})", flush=True)
    (v_dir / "independent.json").write_text(json.dumps({"verifier": str(vbin), "verifier_sha256": _sha256(vbin) if vbin and vbin.is_file() else None,
                                                       "threads": threads, "items": verdicts}, indent=1))
    pruned = None
    if keep_proofs is not None:
        pruned = prune_proofs(proofs_dir, man, keep_proofs)
        man["pruned"] = pruned
        (proofs_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
        print(f"bench_result.py: kept {pruned['kept']} sub-batch proofs per rep, deleted {pruned['deleted']} ({pruned['bytes_deleted'] / 1e9:.2f} GB) after verification", flush=True)
    n_acc = sum(1 for v in verdicts if v["accepted"])
    by_rep: dict[int, float] = {}
    for v in verdicts:
        by_rep[v["rep"]] = by_rep.get(v["rep"], 0.0) + (v["verify_seconds"] or v["process_wall"])
    rust_rep_sums = [by_rep[k] for k in sorted(by_rep)]
    all_rust_ok = bool(verdicts) and n_acc == len(verdicts) and len(verdicts) == len(man["files"])
    python_ok = bench["validation"]["status"] == "passed"

    prove_reps = [r["prover_total"] for r in reps]
    pyv_reps = [r["verifier"] for r in reps]
    prove_med = statistics.median(prove_reps)
    rust_med = statistics.median(rust_rep_sums) if rust_rep_sums else None
    pyv_med = statistics.median(pyv_reps)
    proof_bytes = int(meas_in["transcript.bytes"])
    depth = int(meas_in["rounds.sequential_depth"])
    soundness_bits = -float(sec["achieved_log2"])
    n_proofs = int(meas_in["split.subbatches"])
    B = int(meas_in["vu.count"])
    proof_class = PROOF_CLASS[(zk, mode)] if fp_in.get("proof_class") is None else fp_in["proof_class"]

    fp = {"candidate": "B-Ligero", "backend": "backends/direct/ligero (b-zk-fix prover)", "commit": source_commit(fp_in.get("software", {}).get("backend", {}).get("commit")),
          "protocol": "Ligero direct-constraint chain proof, BabyBear, PROTOCOL.md 3-8c", "mode": mode, "zk": zk, "proof_class": proof_class,
          "checker": "v1 (WP2 checker: bits + one-hot selectors)", "batch": B, "K": 1536, "n_subbatches": n_proofs,
          "per_proof_vus": int(meas_in["split.per_proof_vus"]), "security": sec, "hardware": fp_in.get("hardware"), "software": fp_in.get("software"),
          "verifier": {"name": "ligero-verify (backends/ligero-verify, Rust, independent)", "path": str(vbin), "threads": threads,
                       "sha256": _sha256(vbin) if vbin and vbin.is_file() else None, "coins": "runner's step-0 coins (--coins)" if mode == "interactive" else "n/a (Fiat-Shamir)"},
          "python_verifier": "backends.direct.ligero (live, every rep, every sub-batch; shares code with the prover)",
          "warmup": "one sub-batch (excluded)", "reps": len(reps),
          "dumped_reps": man.get("set", {}).get("dump_reps", len(reps)),    # reps whose proofs were written out and Rust-verified
          "proofs_retained": pruned, "instances": fp_in.get("instances"), "B_real": fp_in.get("B_real")}
    meas = [
        _m("prove_wall", prove_med, "s"), _m("prove_wall_min", min(prove_reps), "s"), _m("prove_wall_max", max(prove_reps), "s"),
        _m("prove_ms_per_vu", prove_med / B * 1e3, "ms"),
        _m("verify_wall", rust_med, "s"), _m("verify_python_wall", pyv_med, "s"), _m("verifier_threads", threads, "count"),
        _m("proof_bytes", proof_bytes, "B"), _m("depth", depth, "count"), _m("soundness_bits", soundness_bits, "bits"),
        _m("e2e_0ms", (prove_med + rust_med) if rust_med is not None else None, "s"),
        _m("proofs_verified_independent", n_acc, "count"), _m("proofs_total", len(man["files"]), "count"),
        _m("peak_device_bytes", meas_in["mem.peak_device_bytes"], "B"), _m("bench_wall", bench_wall, "s"),
    ]
    meas = [m for m in meas if m["value"] is not None]
    status = "passed" if (python_ok and all_rust_ok) else "failed"
    result = {"schema": "research/result/v0.1", "run_id": run_id, "kind": "bench-result/v1", "workload_fingerprint": fp,
              "validation": {"status": status,
                             "checker": "backends/ligero-verify verify (independent Rust) on every recorded rep's sub-batch proof; Python verifier live per rep",
                             "detail": f"python: {bench['validation']['status']}; rust: {n_acc}/{len(man['files'])} accepted "
                                       f"({'coins from the runner' if mode == 'interactive' else 'transferable Fiat-Shamir files'}); "
                                       f"reps prove {['%.3f' % v for v in prove_reps]} s, rust verify {['%.3f' % v for v in rust_rep_sums]} s",
                             "evidence": {"reps": reps, "rust_rep_sums": rust_rep_sums, "python_verifier_reps": pyv_reps}},
              "measurements": meas, "measurement_files": ["bench_vu.json", "verify/independent.json", "proofs/manifest.json"], "artifacts": []}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, run_dir / "result.json")
    print(json.dumps({"prove_wall": prove_med, "verify_wall": rust_med, "verify_python_wall": pyv_med, "proof_bytes": proof_bytes, "depth": depth,
                      "soundness_bits": soundness_bits, "rust_accepted": f"{n_acc}/{len(man['files'])}", "status": status}, indent=1))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
