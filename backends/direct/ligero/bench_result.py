"""bench-vu + the independent Rust verifier on every proof -> $RESEARCH_RUN_DIR/result.json (research/result/v0.1, bench-result/v1).

    python backends/direct/ligero/bench_result.py bench-vu [--zk] --mode interactive|fiat-shamir --batch 16384 --total-vus 4096 \
        --reps 3 --root /workspace/bench-instances/v1 --device cuda --verifier /workspace/bin/ligero-verify --threads 16

A thin wrapper (lane h100-bestvsbest): it runs `backends.direct.ligero.run bench-vu` in-process with `--out`, `--dump-dir` and
`--run-id` pointed at the run directory, then runs `ligero-verify verify` (backends/ligero-verify, shares no code with the prover)
on every recorded rep's sub-batch proof -- with `--coins` = the runner's step-0 coins in interactive mode (DISCREPANCIES D7: a
file proof is sound only against the verifier's own coins) -- and writes one result.json whose measurement names are the same as
`backends/gkr/bench_result.py`'s (prove_wall, verify_wall, verify_python_wall, proof_bytes, depth, soundness_bits, e2e_0ms) so the
two candidates are `research data select`-able side by side.  Nothing in the prover is touched.

Lane b-hopper: `--relation <name>` (before `bench-vu`, as run.py takes it) selects a registered relation (relations.py); its
proofs are dumped and verified the same way.  `--verifier python` runs the Python file verifier (`serialize.py verify`) in a
SEPARATE PROCESS instead of the Rust binary -- the acceptance check for relations the Rust verifier cannot read (the FP8
relations: 8-bit operands, 22-bit public words), labelled as such in `fp["verifier"]` and `independently_verified=false`.  A
non-Ampere BF16 relation (bf16-hopper) is checked by the Rust binary with `--allow-any-system` (its system id is not the
pinned Ampere one; the statement digest binds the system id).  The inner bench's `profile`, `relation`, `model`,
`instances`, `census` and the rate / overhead measurements are carried into this result.
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or "backends/direct/ligero/results/bench_result")
    run_id = os.environ.get("RESEARCH_RUN_ID") or f"local-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    verifier = _pop(argv, "--verifier")
    threads = int(_pop(argv, "--threads", "1"))
    relation = argv[argv.index("--relation") + 1] if "--relation" in argv else "bf16"
    python_verifier = verifier == "python"
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
    vbin = None if (verifier is None or python_verifier) else Path(verifier)
    have_verifier = python_verifier or (vbin is not None and vbin.is_file())
    if not have_verifier:
        print(f"bench_result.py: no independent verifier at {verifier!r}; verify_wall not measured", file=sys.stderr)
    system = proofs_dir / "system.bin"
    allow_any = relation != "bf16"            # the pinned system id is the Ampere relation's
    for e in man["files"]:
        if not have_verifier:
            break
        stmt, proof = proofs_dir / e["stmt"], proofs_dir / e["proof"]
        if python_verifier:
            vjson = v_dir / f"rep{e['rep']}_sub{e['sub']:02d}.json"
            cmd = [sys.executable, "-m", "backends.direct.ligero.serialize", "verify", "--statement", str(stmt), "--proof", str(proof),
                   "--device", "cpu", "--relation", relation, "--json", str(vjson)]
        else:
            cmd = [str(vbin), "verify", "--system", str(system), "--statement", str(stmt), "--proof", str(proof), "--threads", str(threads)]
            if allow_any:
                cmd.append("--allow-any-system")
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
        secs = (doc or {}).get("seconds") if python_verifier else (doc or {}).get("timing", {}).get("total")
        verdicts.append({"rep": e["rep"], "sub": e["sub"], "proof": e["proof"], "proof_sha256": e["proof_sha256"], "proof_bytes": e["proof_bytes"],
                         "accepted": accepted, "rc": r.returncode, "reason": (doc or {}).get("reason"), "verify_seconds": secs,
                         "process_wall": wall, "coins": bool(e.get("coins")), "stdout_tail": out[-1][:300] if out else r.stderr[-300:]})
        print(f"verify rep{e['rep']} sub{e['sub']:02d}: {'accept' if accepted else 'REJECT'} {wall:.3f}s ({(doc or {}).get('reason')})", flush=True)
    (v_dir / "independent.json").write_text(json.dumps({"verifier": "python (backends.direct.ligero.serialize verify, separate process)" if python_verifier else str(vbin),
                                                       "verifier_sha256": _sha256(vbin) if vbin and vbin.is_file() else None,
                                                       "relation": relation, "allow_any_system": allow_any and not python_verifier,
                                                       "threads": threads, "items": verdicts}, indent=1))
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

    if python_verifier:
        verifier_fp = {"name": "python file verifier (backends.direct.ligero.serialize verify), separate process; shares code with the prover",
                       "independent": False, "relation": relation,
                       "why_not_rust": "backends/ligero-verify reads ligero-statement/v2 (16-bit words) and decodes BF16 operands only",
                       "coins": "transcript's own coins replayed (file verifier)" if mode == "interactive" else "n/a (Fiat-Shamir)"}
    else:
        verifier_fp = {"name": "ligero-verify (backends/ligero-verify, Rust, independent)", "path": str(vbin), "threads": threads,
                       "independent": True, "relation": relation, "allow_any_system": allow_any,
                       "sha256": _sha256(vbin) if vbin and vbin.is_file() else None,
                       "coins": "runner's step-0 coins (--coins)" if mode == "interactive" else "n/a (Fiat-Shamir)"}
    fp = {"candidate": "B-Ligero", "backend": fp_in.get("software", {}).get("backend", {}).get("name") or "backends/direct/ligero (b-zk-fix prover)",
          "commit": source_commit(fp_in.get("software", {}).get("backend", {}).get("commit")),
          "protocol": "Ligero direct-constraint chain proof, BabyBear, PROTOCOL.md 3-8c", "mode": mode, "zk": zk, "proof_class": proof_class,
          "checker": "v1 (WP2 checker: bits + one-hot selectors)", "batch": B, "K": int(fp_in.get("K", 1536)), "n_subbatches": n_proofs,
          "per_proof_vus": int(meas_in["split.per_proof_vus"]), "security": sec, "hardware": fp_in.get("hardware"), "software": fp_in.get("software"),
          "profile": fp_in.get("profile"), "relation_id": relation,
          **{k: fp_in[k] for k in ("relation", "model", "instances", "census", "rows", "step_k", "steps_per_vu", "domain") if k in fp_in},
          "verifier": verifier_fp,
          "python_verifier": "backends.direct.ligero (live, every rep, every sub-batch; shares code with the prover)",
          "warmup": "one sub-batch (excluded)", "reps": len(reps)}
    meas = [
        _m("prove_wall", prove_med, "s"), _m("prove_wall_min", min(prove_reps), "s"), _m("prove_wall_max", max(prove_reps), "s"),
        _m("prove_ms_per_vu", prove_med / B * 1e3, "ms"),
        _m("verify_wall", rust_med, "s"), _m("verify_python_wall", pyv_med, "s"), _m("verifier_threads", threads, "count"),
        _m("proof_bytes", proof_bytes, "B"), _m("depth", depth, "count"), _m("soundness_bits", soundness_bits, "bits"),
        _m("e2e_0ms", (prove_med + rust_med) if rust_med is not None else None, "s"),
        _m("proofs_verified_independent", n_acc, "count"), _m("proofs_total", len(man["files"]), "count"),
        _m("peak_device_bytes", meas_in["mem.peak_device_bytes"], "B"), _m("bench_wall", bench_wall, "s"),
    ]
    meas += [m for m in bench["measurements"] if m["name"].startswith(("rate.", "overhead.", "t.", "vu.", "rows_per_unit", "units_per_vu"))]
    meas = [m for m in meas if m["value"] is not None]
    status = "passed" if (python_ok and all_rust_ok) else "failed"
    result = {"schema": "research/result/v0.1", "run_id": run_id, "kind": "bench-result/v1", "workload_fingerprint": fp,
              "validation": {"status": status,
                             "checker": ("backends.direct.ligero.serialize verify (Python file verifier, separate process; NOT independent of the prover) "
                                         "on every recorded rep's sub-batch proof; Python verifier live per rep" if python_verifier else
                                         "backends/ligero-verify verify (independent Rust) on every recorded rep's sub-batch proof; Python verifier live per rep"),
                             "detail": f"python: {bench['validation']['status']}; {'python-file' if python_verifier else 'rust'}: {n_acc}/{len(man['files'])} accepted "
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
