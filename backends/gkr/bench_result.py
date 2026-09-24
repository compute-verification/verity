"""A-GKR on the GPU as a Table 2 cell: the frozen instances' operands -> the device witness generator (gpu/v2/witness.py)
-> the prover (gpu/prover.py) -> proof.bin; the Python verifier and the independent Rust verifier (backends/gkr/verifier)
on every recorded proof -> $RESEARCH_RUN_DIR/result.json (research/result/v0.1, bench-result/v1, the campaign contract
verity_numerical.bench.contract).

    research run --on vy-agkr-a100 --project verity --source . --cwd source/backends/gkr --tool a_gpu_prove --scratch triton \
        -- /workspace/venv312/bin/python bench_result.py /workspace/agkr-table/bb/stmt --instances /workspace/bench-instances/v1 \
        --vus 4096 --reps 3 --warmup 1 --verifier /workspace/bin/verity-gkr-verify --threads 15

DIR holds the statement's Params-only files of a checker-v2 export (circuit.txt, epilogue.txt, chain.txt, manifest.json;
`python -m verity_numerical.gkr_export.vu export --field babybear --variant v2`); the witness is never read from it.  The
operands and public words come from --instances after every file of the tier is checked against the committed frozen
manifest (--manifest, default fixtures/bench-instances/v1/manifest.json of the source tree: its sha256 is the
`instances.manifest_sha256` Table 2 requires).  The run directory gets `statement/` (DIR's four files + public.bin written
from the frozen words) and `proofs/rep<i>.bin`: `verity-gkr-verify verify --dir statement --proof proofs/rep0.bin --vus 4096`.

Per timed rep (wall clock; the device is synchronised at every boundary; the buckets sum to t.total up to the Python
between the timers, which tables.phases shows as Other):

    t.witness              H2D of the operands + the witness generator (chain, unit rows, limb epilogue) + wire evaluation
    t.encoding_commitment  x assembly + RS encode (SIMT NTT) + SHA-512 Merkle tree
    t.lookup               LogUp multiplicities + transcript start + every table's LogUp GKR
    t.arithmetic           the circuit GKR layers + the Ligero functional (input / lookup / chain claims) + w, q (INT8 GEMMs)
    t.serialization        the opened columns + Merkle paths + proof.to_bytes()
    t.zk_additional        0 (non-ZK)

B-Ligero books its Ligero tests as t.arithmetic and its column openings as t.serialization (backends/direct/ligero/vu.py);
this is the same split.  The buckets reported are the median rep's (by t.total), so they sum to its t.total.  Soundness is
computed live on the instance proved (backends/gkr/soundness.py -> security.accounting.account_candidate_a).  `--path`
selects the kernel path by the prover's own environment switches: graphed (default) | packed | torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

FIELD = "babybear6"          # F_p[X]/(X^6 - 22), p = 2^31 - 2^27 + 1 (gpu/field.py)
TIER = "vu-k1536"
K_VU = 1536
BACKEND = "verity-gkr gpu (backends/gkr/gpu: checker v2 + LogUp + Ligero over BabyBear^6, SHA-512 Merkle; torch/Triton)"
STATEMENT_FILES = ("circuit.txt", "epilogue.txt", "chain.txt", "manifest.json")


def _m(name: str, value, unit: str) -> dict:
    return {"name": name, "value": value, "unit": unit}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commit() -> str | None:
    """git when there is one; else `research run --on --source` ships `git archive HEAD` to `<root>/src/<commit sha>/`,
    so an ancestor directory's name is the commit; else `RESEARCH_SOURCE_COMMIT`."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        if sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    env = os.environ.get("RESEARCH_SOURCE_COMMIT")
    if env:
        return env
    for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if len(p.name) == 40 and all(c in "0123456789abcdef" for c in p.name):
            return p.name
    return None


def load_frozen(root: Path, manifest: Path, tier: str, lo: int, n: int):
    """``(x, w, y, check)``: the tier's operands ``(n, K)`` uint16 and words ``(n,)`` uint16 for ids ``[lo, lo + n)``, after
    every file of the tier under ``root`` matched the sha256 the frozen ``manifest`` records for it."""
    import numpy as np
    from verity_numerical.bench import instances as I

    man = json.loads(manifest.read_text())
    files = man["tiers"][tier]["files"]
    check = {}
    for name, ent in files.items():
        got = _sha256(root / name)
        if got != ent["sha256"]:
            raise SystemExit(f"bench_result.py: {root / name} sha256 {got} != the frozen manifest's {ent['sha256']}")
        check[name] = got
    td = I.load(tier, root)
    if lo + n > td.n:
        raise SystemExit(f"bench_result.py: {tier} has {td.n} ids, [{lo}, {lo + n}) requested")
    x = np.frombuffer(td.x, dtype=np.uint16).reshape(td.n, td.K)[lo:lo + n]
    w = np.frombuffer(td.w, dtype=np.uint16).reshape(td.n, td.K)[lo:lo + n]
    y = np.frombuffer(td.y, dtype=np.uint16)[lo:lo + n]
    return x, w, y, check


def write_statement(d: Path, out: Path, y) -> None:
    """``out/``: DIR's Params-only files and ``public.bin`` holding the frozen words (``bb_export.write_rows`` layout)."""
    from gpu.bb_export import write_rows

    out.mkdir(parents=True, exist_ok=True)
    for name in STATEMENT_FILES:
        shutil.copyfile(d / name, out / name)
    write_rows([[int(v)] for v in y], out / "public.bin")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python backends/gkr/bench_result.py")
    ap.add_argument("dir", help="a checker-v2 export directory (only its Params-only statement files are read)")
    ap.add_argument("--instances", required=True, help="bench-instances/v1 data directory (the tier's files)")
    ap.add_argument("--manifest", default=None, help="the frozen manifest.json (default: <source>/fixtures/bench-instances/v1)")
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--vus", type=int, default=4096)
    ap.add_argument("--checker", choices=("v2",), default="v2")
    ap.add_argument("--path", choices=("graphed", "packed", "torch"), default="graphed")
    ap.add_argument("--field", default=FIELD)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--verifier", default=None, help="verity-gkr-verify binary (independent Rust verifier)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--target", type=int, default=-128)
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
    if man.get("variant") != "v2":
        print(f"bench_result.py: {d}/manifest.json is not a checker-v2 export", file=sys.stderr)
        return 2
    src_root = Path(__file__).resolve().parents[2]
    frozen = Path(args.manifest) if args.manifest else src_root / "fixtures" / "bench-instances" / "v1" / "manifest.json"

    import numpy as np
    import torch
    from verity_numerical.bench import contract

    from gpu import ligero, prover
    from gpu.circuit import layers, load_circuit
    from gpu.run import _dev, read_chain
    from gpu.v2.witness import Generator, Ops
    import soundness

    # ---- preprocessing (Params-only; not in the buckets) -------------------------------------------------------------
    tp0 = time.perf_counter()
    dev = _dev(args.device)
    x, w, y, inst_check = load_frozen(Path(args.instances), frozen, args.tier, args.lo, args.vus)
    t_load = time.perf_counter() - tp0
    ucirc, ecirc = load_circuit(d / "circuit.txt"), load_circuit(d / "epilogue.txt")
    ulayers, elayers = layers(ucirc), layers(ecirc)
    steps = man["steps"]
    if steps * 16 != x.shape[1]:
        raise SystemExit(f"bench_result.py: DIR has {steps} steps, the tier has K = {x.shape[1]}")
    public = [int(v) for v in y]
    ops = Ops(str(dev))
    gen = Generator(ops)
    torch.cuda.synchronize(dev)
    t_prep = time.perf_counter() - tp0
    write_statement(d, run_dir / "statement", y)
    encoder = "SIMT (backends/direct/encode)" if ligero.simt_encoder(ligero.DEFAULT.k, ligero.DEFAULT.n) is not None else "torch NTT"

    def one_rep():
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        A, B, Y = ops.asarray(x), ops.asarray(w), ops.asarray(y)
        rows = gen.run(A, B, Y)
        bad = int(rows.bad.sum())
        t1 = time.perf_counter()
        if bad:
            raise SystemExit(f"bench_result.py: {bad} VUs rejected at witness generation (the frozen positives are all valid)")
        gen_peak = torch.cuda.max_memory_allocated(dev)
        del A, B, Y
        inst = prover.Instance([prover.Segment("unit", ucirc, ulayers, rows.units, ucirc.hash),
                                prover.Segment("epilogue", ecirc, elayers, rows.epilogue, ecirc.hash)],
                               read_chain(d / man.get("chain_file", "chain.txt"), ucirc, ecirc, steps, public))
        proof, st = prover.prove(inst, True)
        torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        data = proof.to_bytes()
        t3 = time.perf_counter()
        return inst, proof, st, data, {"witness_gen": t1 - t0, "prove": t2 - t1, "bytes": t3 - t2, "total": t3 - t0,
                                       "gen_chain": rows.seconds["chain"], "peak": max(gen_peak, st.peak_mem)}

    def verify_py(inst, proof):
        t = time.perf_counter()
        try:
            prover.verify(inst, proof, True)
            err = None
        except prover.VerifyError as e:
            err = str(e)
        torch.cuda.synchronize(dev)
        return err, time.perf_counter() - t

    warm = []
    for _ in range(args.warmup):
        tw = time.perf_counter()
        inst, proof, _st, _data, tm = one_rep()
        err, _ = verify_py(inst, proof)
        warm.append(time.perf_counter() - tw)
        print(json.dumps({"warmup": True, "seconds": warm[-1], "total": tm["total"], "verified": err is None, "error": err}), flush=True)
        del inst, proof, _st, _data
    rows = []
    budget = None
    for rep in range(args.reps):
        inst, proof, st, data, tm = one_rep()
        err, tv = verify_py(inst, proof)
        if budget is None:
            budget = soundness.summary(soundness.budget(inst, st.ligero_rows, st.linkage_constraints))
        pf = proofs_dir / f"rep{rep}.bin"
        pf.write_bytes(data)
        row = {"rep": rep, "vus": inst.vus(), "units": inst.segs[0].units, "t_total": tm["total"], "witness_gen": tm["witness_gen"],
               "gen_chain": tm["gen_chain"], "prove": tm["prove"], "to_bytes": tm["bytes"],
               "t_witness_wires": st.t_witness_wires, "t_mults": st.t_mults, "t_commit": st.t_commit, "t_lookup": st.t_lookup,
               "t_arith": st.t_arith, "t_open": st.t_open, "t_open_acc": st.t_open_acc, "t_open_wq": st.t_open_wq,
               "t_open_cols": st.t_open_cols, "python_verifier_s": tv, "python_verified": err is None, "error": err,
               "proof_bytes": len(data), "proof_sha256": hashlib.sha256(data).hexdigest(), "proof_file": f"proofs/rep{rep}.bin",
               "slots": st.slots, "sequential_depth": st.sequential_depth(True), "depth_circuit": st.depth_circuit,
               "depth_lookup_max": st.depth_lookup_max, "committed_elements": st.committed_elements, "ligero_rows": st.ligero_rows,
               "witness_elements": sum(int(s.cols.numel()) for s in inst.segs), "peak_mem_bytes": tm["peak"]}
        row["buckets"] = {
            "t.witness": tm["witness_gen"] + st.t_witness_wires,
            "t.encoding_commitment": st.t_commit,
            "t.arithmetic": st.t_arith + st.t_open_acc + st.t_open_wq,
            "t.lookup": st.t_mults + st.t_lookup,
            "t.zk_additional": 0.0,
            "t.serialization": st.t_open_cols + tm["bytes"],
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "buckets"}), flush=True)
        del inst, proof, st, data

    # ---- the independent verifier on every recorded proof ------------------------------------------------------------
    vbin = Path(args.verifier) if args.verifier else None
    verdicts = []
    for row in rows:
        if vbin is None or not vbin.is_file():
            print(f"bench_result.py: no independent verifier at {args.verifier!r}", file=sys.stderr)
            break
        out_json = run_dir / f"verify_rep{row['rep']}.json"
        cmd = [str(vbin), "verify", "--dir", str(run_dir / "statement"), "--proof", str(run_dir / row["proof_file"]),
               "--vus", str(row["vus"]), "--threads", str(args.threads), "--json", str(out_json)]
        t1 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t1
        try:
            doc = json.loads(out_json.read_text()) if out_json.is_file() else None
        except json.JSONDecodeError:
            doc = None
        accepted = bool(doc and doc.get("accepted")) and r.returncode == 0
        verdicts.append({"rep": row["rep"], "proof_sha256": row["proof_sha256"], "accepted": accepted, "rc": r.returncode,
                         "error": (doc or {}).get("error"), "verify_seconds": (doc or {}).get("verify_seconds"), "process_wall": wall,
                         "threads": (doc or {}).get("threads"), "cpu": (doc or {}).get("cpu"), "json": out_json.name,
                         "command": " ".join(cmd), "stderr_tail": r.stderr[-300:]})
        print(f"verify rep{row['rep']}: {'accept' if accepted else 'REJECT'} {wall:.3f}s ({(doc or {}).get('error')})", flush=True)
    vsha = _sha256(vbin) if vbin and vbin.is_file() else None
    (run_dir / "verify_independent.json").write_text(json.dumps({"verifier": str(vbin), "verifier_sha256": vsha, "threads": args.threads,
                                                                 "items": verdicts}, indent=1))
    (run_dir / "prove_reps.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(dev), "path": args.path, "checker": "v2",
                                                         "encoder": encoder, "runs": rows, "warmup_seconds": warm}, indent=1))

    # ---- the envelope ---------------------------------------------------------------------------------------------------
    n_acc = sum(1 for v in verdicts if v["accepted"])
    rust_ok = bool(verdicts) and n_acc == len(rows)
    python_ok = all(r["python_verified"] for r in rows)
    same_proof = len({r["proof_sha256"] for r in rows}) == 1
    med = sorted(rows, key=lambda r: r["t_total"])[(len(rows) - 1) // 2]
    t_total = med["t_total"]
    Bn = med["vus"]
    rust_reps = [v["verify_seconds"] if v["verify_seconds"] is not None else v["process_wall"] for v in verdicts]
    rust_med = statistics.median(rust_reps) if rust_reps else None
    ach = budget["achieved_log2"]
    hw = contract.probe_hardware()
    import triton

    try:
        import cupy
        cupy_v = cupy.__version__
    except ImportError:
        cupy_v = None
    sw = {"backend": {"name": BACKEND, "commit": source_commit(), "version": "checker v2 (limb epilogue), proof.bin v2",
                      "path": args.path, "encoder": encoder, "opening": "INT8 tensor-core limb GEMMs" if ligero.use_int8(dev) else "fp64 limbs",
                      "witness": "gpu/v2/witness.py (device generator; byte-exact with the CPU export)"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "triton": triton.__version__, "cupy": cupy_v, "numpy": np.__version__}}
    security = {"target": args.target, "achieved_log2": ach, "achieved_bits": -ach, "field": "BabyBear^6 = F_p[X]/(X^6 - 22)",
                "trace_field": "BabyBear", "challenge_coords": 6,
                "ligero": {"n": ligero.DEFAULT.n, "k": ligero.DEFAULT.k, "t": ligero.DEFAULT.t, "rows": med["ligero_rows"], "zk": False,
                           "row_combination": "powers of one challenge (BCIKS20 curve bound)"},
                "hash": ligero.MERKLE_HASH, "mode": "interactive", "fiat_shamir_queries_log2": None,
                "bound_model": budget["model"], "hash_assumption": "collision resistance of SHA-512 (Merkle, q^2/2^512 at q = 2^64)",
                "zk_statement": "none (no masking)", "by_component_log2": budget["by_component_log2"],
                "dominant_term": budget["dominant_term"], "source": budget["source"], "union_over": 1}
    fp = contract.fingerprint(
        profile=contract.PROFILE, domain=contract.DOMAIN, K=K_VU, B=Bn, N_subbatches=1,
        proof_class="NON_ZK_PROOF_DIAGNOSTIC", authentication="excluded", zk_mode=contract.ZK_NONE,
        security=security, hardware=hw, software=sw,
        preprocessing=(f"circuit compilation (unit + limb-epilogue circuits, GKR layers, chain spec: functions of Params only), "
                       f"the witness generator's dense tables on the device, loading the frozen operands from disk "
                       f"({t_load:.2f}s incl. the sha256 check of every tier file); {t_prep:.2f}s in all.  Triton / CUDA-graph / "
                       f"NVRTC compilation happens in {args.warmup} untimed warm-up rep(s) ({', '.join('%.1fs' % v for v in warm)})"),
        retention="retained",
        instances=contract.instances_ref(args.tier, args.lo, args.lo + Bn, _sha256(frozen)),
        scope="vu", candidate="A-GKR", checker="v2", field=FIELD,
        instances_check={"manifest": str(frozen), "files_sha256_match": inst_check, "data_root": str(Path(args.instances).resolve())},
        witness_generation="device generator (gpu/v2/witness.py) inside t.witness, from the frozen operands (H2D included)",
        omitted="ZK masking",
        steps=steps, units=med["units"], reps=args.reps, warmup=args.warmup,
        verifier={"name": "verity-gkr-verify (backends/gkr/verifier, Rust, shares no code with the prover)", "path": str(vbin),
                  "sha256": vsha, "threads": args.threads, "cpu": next((v["cpu"] for v in verdicts if v.get("cpu")), None)},
        python_verifier="gpu.prover.verify (shares field / transcript / LogUp helpers with the prover)",
        note=f"one proof of all {Bn} VUs (96 chained units each + the limb epilogue); buckets of the median rep "
             f"(rep {med['rep']}) by t.total",
    )
    b = med["buckets"]
    meas = [contract.measurement(name, b[name]) for name in contract.BUCKET_NAMES]
    meas.append(contract.measurement("t.total", t_total))
    if rust_med is not None:
        meas.append(contract.measurement("verifier.seconds", rust_med))
    meas += [
        contract.measurement("transcript.bytes", med["proof_bytes"]),
        contract.measurement("rounds.protocol", med["sequential_depth"]),
        contract.measurement("rounds.sequential_depth", med["sequential_depth"]),
        contract.measurement("mem.peak_device_bytes", max(r["peak_mem_bytes"] for r in rows)),
        contract.measurement("census.witness_elements", med["witness_elements"]),
        contract.measurement("census.committed_elements", med["committed_elements"]),
        contract.measurement("census.bytes_encoded", med["ligero_rows"] * ligero.DEFAULT.n * 4),
        contract.measurement("fixed.setup_seconds", t_prep),
        contract.measurement("vu.count", Bn),
        contract.measurement("vu.seconds_per_vu", t_total / Bn),
        contract.measurement("vu.per_second", Bn / t_total),
        *contract.rate_measurements(t_total, Bn, K_VU, contract.PROFILE),
        _m("t_total_min", min(r["t_total"] for r in rows), "s"),
        _m("t_total_max", max(r["t_total"] for r in rows), "s"),
        _m("verify_python_seconds", statistics.median(r["python_verifier_s"] for r in rows), "s"),
        _m("proofs_verified_independent", n_acc, "count"),
        _m("proofs_total", len(rows), "count"),
        _m("soundness_bits", -ach, "bits"),
        _m("warmup_seconds", warm[0] if warm else 0.0, "s"),
    ]
    for key in ("witness_gen", "gen_chain", "t_witness_wires", "t_mults", "t_commit", "t_lookup", "t_arith", "t_open_acc", "t_open_wq",
                "t_open_cols", "to_bytes"):
        meas.append(_m(f"split.{key}_seconds", med[key], "s"))
    ok = python_ok and rust_ok and same_proof
    detail = (f"python: {sum(r['python_verified'] for r in rows)}/{len(rows)}; rust: {n_acc}/{len(verdicts)} accepted"
              f"{'' if same_proof else '; the reps produced DIFFERENT proofs'}; reps t.total "
              f"{['%.3f' % r['t_total'] for r in rows]} s, rust verify {['%.3f' % v for v in rust_reps]} s")
    result = {"schema": contract.RESULT_SCHEMA, "run_id": run_id, "kind": "bench-result/v1", "workload_fingerprint": fp,
              "validation": {"status": "passed" if ok else "failed",
                             "checker": "backends/gkr/verifier `verify` (independent Rust) on every recorded rep's proof + gpu.prover.verify per rep",
                             "detail": detail,
                             "evidence": {"reps": [{"t_total": r["t_total"], "buckets": r["buckets"], "python_verifier": r["python_verifier_s"],
                                                    "proof_sha256": r["proof_sha256"]} for r in rows],
                                          "rust": [{k: v[k] for k in ("rep", "accepted", "verify_seconds", "error", "command")} for v in verdicts],
                                          "soundness": budget}},
              "measurements": meas,
              "measurement_files": ["prove_reps.json", "verify_independent.json"] + [v["json"] for v in verdicts],
              "artifacts": [r["proof_file"] for r in rows] + [f"statement/{n}" for n in (*STATEMENT_FILES, "public.bin")]}
    problems = contract.validate(result)
    result["contract_problems"] = problems
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, run_dir / "result.json")
    print(json.dumps({"t_total": t_total, "buckets": b, "verify_wall": rust_med, "proof_bytes": med["proof_bytes"],
                      "soundness_log2": ach, "rust_accepted": f"{n_acc}/{len(rows)}", "status": result["validation"]["status"],
                      "contract_problems": problems}, indent=1))
    return 0 if ok and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
