"""bench-vu + the independent Rust verifier on every proof -> $RESEARCH_RUN_DIR/result.json (research/result/v0.1, bench-result/v1).

    python archive/direct/ligero/bench_result.py bench-vu [--zk] --mode interactive|fiat-shamir --batch 16384 --total-vus 4096 \
        --reps 3 --root /workspace/bench-instances/v1 --device cuda --verifier /workspace/bin/ligero-verify --threads 16

A thin wrapper (lane h100-bestvsbest): it runs `archive.direct.ligero.run bench-vu` in-process with `--out`, `--dump-dir` and
`--run-id` pointed at the run directory, then runs `ligero-verify verify` (archive/ligero-verify, shares no code with the prover)
on every recorded rep's sub-batch proof -- with `--coins` = the runner's step-0 coins in interactive mode (D7, archive/ligero-verify/discrepancies/d7.md: a
file proof is sound only against the verifier's own coins) -- and writes one result.json whose measurement names are the same as
`archive/gkr/bench_result.py`'s (prove_wall, verify_wall, verify_python_wall, proof_bytes, depth, soundness_bits, e2e_0ms) so the
two candidates are `research data select`-able side by side.  Nothing in the prover is touched.

Lane b-hopper: `--relation <name>` (before `bench-vu`, as run.py takes it) selects a registered relation (relations.py); its
proofs are dumped and verified the same way.  `--verifier python` runs the Python file verifier (`serialize.py verify`) in a
SEPARATE PROCESS instead of the Rust binary, labelled as such in `fp["verifier"]` (`independent: false`).  Since the merge of
lane b-hopper with main (lane b-merge-h100) every registered relation's statement is the relation-named `ligero-statement/v4`,
which archive/ligero-verify does not read yet (it reads v2 and v3: the Ampere BF16 relation's formats), so `--relation
<registered>` always uses the Python file verifier -- a self-check, not an independent verification; the wall of that
separate process is dominated by importing torch, not by verifying.  The inner bench's `profile`, `relation`, `model`,
`instances`, `census` and the rate / overhead measurements are carried into this result.

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
WHY_NOT_RUST = ("archive/ligero-verify reads ligero-statement/v2 and v3 (the Ampere BF16 relation: 16-bit words, pinned system id) only; "
                "the registered relations write the relation-named ligero-statement/v4")


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


def verify_cmd(vbin: Path, system: Path, stmt: Path, proof: Path, threads: int, coins: Path | None, target_log2: float) -> list[str]:
    """`ligero-verify verify` for one sub-batch proof.  `--soundness-bits` is the bench's target (`security.target`, e.g. -80
    -> 80): the verifier defaults to 128 and rejects any proof whose parameters do not reach the bits it was asked for,
    so a soundness sweep below 2^-128 must tell it the target it is judging against (lane b-sweep)."""
    cmd = [str(vbin), "verify", "--system", str(system), "--statement", str(stmt), "--proof", str(proof), "--threads", str(threads),
           "--soundness-bits", f"{-float(target_log2):g}"]
    if coins is not None:
        cmd += ["--coins", str(coins)]
    return cmd


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
    relation = argv[argv.index("--relation") + 1] if "--relation" in argv else "bf16"
    python_verifier = verifier == "python" or relation != "bf16"
    if python_verifier and verifier not in (None, "python"):
        print(f"bench_result.py: --relation {relation} writes ligero-statement/v4, which {verifier} does not read: using the Python file "
              "verifier (a self-check, not independent)", file=sys.stderr)
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

    from archive.direct.ligero.run import main as run_main
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

    def run_verifier(e: dict) -> tuple[bool, dict, float, list[str], int, str]:
        stmt, proof = proofs_dir / e["stmt"], proofs_dir / e["proof"]
        if python_verifier:
            vjson = v_dir / f"{Path(e['proof']).stem}.json"
            cmd = [sys.executable, "-m", "archive.direct.ligero.serialize", "verify", "--statement", str(stmt), "--proof", str(proof),
                   "--device", "cpu", "--relation", relation, "--json", str(vjson)]
        else:
            cmd = verify_cmd(vbin, system, stmt, proof, threads, proofs_dir / e["coins"] if e.get("coins") else None, sec.get("target", -128.0))
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
        return accepted, doc or {}, wall, out, r.returncode, r.stderr

    def _seconds(doc: dict):
        return doc.get("seconds") if python_verifier else doc.get("timing", {}).get("total")

    for e in man["files"]:
        if not have_verifier:
            break
        accepted, doc, wall, out, rc_v, err = run_verifier(e)
        verdicts.append({"rep": e["rep"], "sub": e["sub"], "proof": e["proof"], "proof_sha256": e["proof_sha256"], "proof_bytes": e["proof_bytes"],
                         "accepted": accepted, "rc": rc_v, "reason": doc.get("reason"), "verify_seconds": _seconds(doc),
                         "auth_seconds": doc.get("timing", {}).get("auth"),
                         "process_wall": wall, "coins": bool(e.get("coins")), "stdout_tail": out[-1][:300] if out else err[-300:]})
        print(f"verify rep{e['rep']} sub{e['sub']:02d}: {'accept' if accepted else 'REJECT'} {wall:.3f}s ({doc.get('reason')})", flush=True)
    # authentication = included: the dumped negatives (prover-side + file-level) through the same binary; every one must be
    # rejected (the honest included proof among them accepted) -- soundness evidence, not a timing
    neg_verdicts = []
    for e in man.get("auth_negatives", []):
        if not have_verifier:
            break
        accepted, doc, wall, out, rc_v, err = run_verifier(e)
        neg_verdicts.append({"name": e["name"], "kind": e["kind"], "expected_accepted": e["expected_accepted"], "accepted": accepted, "rc": rc_v,
                             "reason": doc.get("reason"), "python_accepted": e["python_verdict"]["accepted"], "python_reason": e["python_verdict"]["reason"],
                             "mutation": e.get("mutation"), "process_wall": wall})
        print(f"verify auth {e['kind']} {e['name']}: rust {'accept' if accepted else 'REJECT'} ({doc.get('reason')}); python "
              f"{'accept' if e['python_verdict']['accepted'] else 'REJECT'} ({e['python_verdict']['reason']})", flush=True)
    verifier_label = "python (archive.direct.ligero.serialize verify, separate process)" if python_verifier else str(vbin)
    (v_dir / "independent.json").write_text(json.dumps({"verifier": verifier_label, "verifier_sha256": _sha256(vbin) if vbin and vbin.is_file() else None,
                                                       "relation": relation, "independent": not python_verifier,
                                                       "threads": threads, "items": verdicts, "auth_negatives": neg_verdicts}, indent=1))
    pruned = None
    if keep_proofs is not None:
        pruned = prune_proofs(proofs_dir, man, keep_proofs)
        man["pruned"] = pruned
        (proofs_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
        print(f"bench_result.py: kept {pruned['kept']} sub-batch proofs per rep, deleted {pruned['deleted']} ({pruned['bytes_deleted'] / 1e9:.2f} GB) after verification", flush=True)
    n_acc = sum(1 for v in verdicts if v["accepted"])
    by_rep: dict[int, float] = {}
    auth_by_rep: dict[int, float] = {}
    for v in verdicts:
        by_rep[v["rep"]] = by_rep.get(v["rep"], 0.0) + (v["verify_seconds"] or v["process_wall"])
        auth_by_rep[v["rep"]] = auth_by_rep.get(v["rep"], 0.0) + (v["auth_seconds"] or 0.0)
    rust_rep_sums = [by_rep[k] for k in sorted(by_rep)]
    rust_auth_sums = [auth_by_rep[k] for k in sorted(auth_by_rep)]
    all_rust_ok = bool(verdicts) and n_acc == len(verdicts) and len(verdicts) == len(man["files"])
    negs_expected = [v for v in neg_verdicts if not v["expected_accepted"]]
    negs_rust_rejected = sum(1 for v in negs_expected if not v["accepted"])
    negs_ok = (negs_rust_rejected == len(negs_expected) and all(v["accepted"] for v in neg_verdicts if v["expected_accepted"])
               and len(neg_verdicts) == len(man.get("auth_negatives", [])))
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
    authentication = fp_in.get("authentication", "excluded")
    with_auth = authentication == "included"
    file_verifier = "python-file" if python_verifier else "rust"

    if python_verifier:
        verifier_fp = {"name": "python file verifier (archive.direct.ligero.serialize verify), separate process; shares code with the prover",
                       "independent": False, "relation": relation, "why_not_rust": WHY_NOT_RUST,
                       "wall_caveat": "process_wall of each verdict is a fresh Python process importing torch; verify_seconds is the verifier alone",
                       "coins": "transcript's own coins replayed (file verifier)" if mode == "interactive" else "n/a (Fiat-Shamir)"}
    else:
        verifier_fp = {"name": "ligero-verify (archive/ligero-verify, Rust, independent)", "path": str(vbin), "threads": threads,
                       "independent": True, "relation": relation,
                       "sha256": _sha256(vbin) if vbin and vbin.is_file() else None,
                       "coins": "runner's step-0 coins (--coins)" if mode == "interactive" else "n/a (Fiat-Shamir)",
                       "soundness_bits_requested": -float(sec.get("target", -128.0)),
                       "authentication": ("SHA-256 Merkle multiproofs of every boundary word folded to the statement's roots (auth.rs), after the Ligero tests"
                                          if with_auth else "n/a (authentication = excluded)")}
    fp = {"candidate": "B-Ligero", "backend": fp_in.get("software", {}).get("backend", {}).get("name") or "archive/direct/ligero (b-zk-fix prover)",
          "commit": source_commit(fp_in.get("software", {}).get("backend", {}).get("commit")),
          "protocol": "Ligero direct-constraint chain proof, BabyBear, PROTOCOL.md 3-8c", "mode": mode, "zk": zk, "proof_class": proof_class,
          "authentication": authentication,
          "checker": "v1 (WP2 checker: bits + one-hot selectors)", "batch": B, "K": int(fp_in.get("K", 1536)), "n_subbatches": n_proofs,
          "per_proof_vus": int(meas_in["split.per_proof_vus"]), "security": sec, "hardware": fp_in.get("hardware"), "software": fp_in.get("software"),
          "preprocessing": fp_in.get("preprocessing"), "instances": fp_in.get("instances"),
          "profile": fp_in.get("profile"), "relation_id": relation, "statement_format": man.get("statement_format"),
          **{k: fp_in[k] for k in ("relation", "model", "census", "rows", "step_k", "steps_per_vu", "domain") if k in fp_in},
          "verifier": verifier_fp,
          "python_verifier": "archive.direct.ligero (live, every rep, every sub-batch; shares code with the prover)",
          "warmup": "one sub-batch (excluded)", "reps": len(reps),
          "dumped_reps": man.get("set", {}).get("dump_reps", len(reps)),    # reps whose proofs were written out and file-verified
          "proofs_retained": pruned, "B_real": fp_in.get("B_real")}
    meas = [
        _m("prove_wall", prove_med, "s"), _m("prove_wall_min", min(prove_reps), "s"), _m("prove_wall_max", max(prove_reps), "s"),
        _m("prove_ms_per_vu", prove_med / B * 1e3, "ms"),
        _m("verify_wall", rust_med, "s"), _m("verify_python_wall", pyv_med, "s"), _m("verifier_threads", threads, "count"),
        _m("proof_bytes", proof_bytes, "B"), _m("depth", depth, "count"), _m("soundness_bits", soundness_bits, "bits"),
        _m("e2e_0ms", (prove_med + rust_med) if rust_med is not None else None, "s"),
        _m("proofs_verified_independent" if not python_verifier else "proofs_verified_python_file", n_acc, "count"),
        _m("proofs_total", len(man["files"]), "count"),
        _m("peak_device_bytes", meas_in["mem.peak_device_bytes"], "B"), _m("bench_wall", bench_wall, "s"),
    ]
    if with_auth:
        meas += [
            _m("verify_auth_wall", statistics.median(rust_auth_sums) if rust_auth_sums else None, "s"),      # inside verify_wall
            _m("verify_auth_python_wall", meas_in.get("auth.verifier_seconds"), "s"),                        # inside verify_python_wall
            _m("prove_auth_openings_wall", meas_in.get("auth.openings_seconds"), "s"),                       # inside prove_wall
            _m("auth_multiproof_bytes", meas_in.get("auth.multiproof_bytes"), "B"),                           # inside proof_bytes
            _m("auth_commit_seconds", meas_in.get("auth.commit_seconds"), "s"),                              # once per object
            _m("auth_opened_leaves", meas_in.get("auth.opened_leaves"), "count"),
            _m("auth_negatives_total", len(negs_expected), "count"),
            _m("auth_negatives_rejected_rust", negs_rust_rejected, "count"),
            _m("auth_negatives_rejected_python", sum(1 for v in negs_expected if not v["python_accepted"]), "count"),
        ]
    meas += [m for m in bench["measurements"] if m["name"].startswith(("rate.", "overhead.", "t.", "vu.", "rows_per_unit", "units_per_vu"))]
    meas = [m for m in meas if m["value"] is not None]
    status = "passed" if (python_ok and all_rust_ok and (negs_ok or not with_auth)) else "failed"
    detail = (f"python: {bench['validation']['status']}; {file_verifier}: {n_acc}/{len(man['files'])} accepted "
              f"({'coins from the runner' if mode == 'interactive' else 'transferable Fiat-Shamir files'}); "
              f"reps prove {['%.3f' % v for v in prove_reps]} s, {file_verifier} verify {['%.3f' % v for v in rust_rep_sums]} s")
    if python_verifier:
        detail += "; the Python file verifier shares code with the prover (a self-check, not an independent verification)"
    if with_auth:
        detail += (f"; authentication negatives: rust rejected {negs_rust_rejected}/{len(negs_expected)}, python rejected "
                   f"{sum(1 for v in negs_expected if not v['python_accepted'])}/{len(negs_expected)}; honest included proof accepted by both: "
                   f"{all(v['accepted'] and v['python_accepted'] for v in neg_verdicts if v['expected_accepted'])}")
    checker = (("archive.direct.ligero.serialize verify (Python file verifier, separate process; NOT independent of the prover) "
                "on every recorded rep's sub-batch proof; Python verifier live per rep") if python_verifier else
               "archive/ligero-verify verify (independent Rust) on every recorded rep's sub-batch proof; Python verifier live per rep")
    result = {"schema": "research/result/v0.1", "run_id": run_id, "kind": "bench-result/v1", "workload_fingerprint": fp,
              "validation": {"status": status,
                             "checker": checker + ("; both verifiers on every authentication negative" if with_auth else ""),
                             "detail": detail,
                             "evidence": {"reps": reps, "rust_rep_sums": rust_rep_sums, "python_verifier_reps": pyv_reps,
                                          **({"rust_auth_sums": rust_auth_sums, "auth_negatives": neg_verdicts,
                                              "auth_negatives_python": bench["validation"]["evidence"].get("auth_negatives"),
                                              "auth_roots": bench["validation"]["evidence"].get("auth_roots")} if with_auth else {})}},
              "measurements": meas, "measurement_files": ["bench_vu.json", "verify/independent.json", "proofs/manifest.json"], "artifacts": []}
    if "contention" in bench:
        result["contention"] = bench["contention"]
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, run_dir / "result.json")
    print(json.dumps({"prove_wall": prove_med, "verify_wall": rust_med, "verify_python_wall": pyv_med, "proof_bytes": proof_bytes, "depth": depth,
                      "soundness_bits": soundness_bits, f"{file_verifier}_accepted": f"{n_acc}/{len(man['files'])}", "authentication": authentication,
                      "auth_negatives_rust_rejected": f"{negs_rust_rejected}/{len(negs_expected)}", "status": status}, indent=1))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
