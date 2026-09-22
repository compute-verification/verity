"""The verification unit (chain of 96 units + epilogue) on the Ligero prover: gate and measurement.

    python -m backends.direct.ligero.run gate-vu  --root fixtures/bench-instances/v1 --device cuda --vus 64
    python -m backends.direct.ligero.run bench-vu --root fixtures/bench-instances/v1 --device cuda --reps 5 --out result.json

A proof of l columns holds floor(l / 96) VUs (l = 4096: 42 VUs, 64 pad columns).  Public per proof: the operands
of every column and the claimed BF16 word of every VU; the accumulators are witness (linked across columns by the
chain test), c_0 = 0.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from verity_numerical.bench import contract, instances
from verity_numerical.checker.params import REAL
from verity_numerical.checker.step import step_output

from .chain import Layout
from .compile import P
from .hints_torch import hints_torch
from .protocol import Config, config_for, prove, soundness, verify
from .unit import compile_unit, public_vectors
from .witness import hints_tensor, tables

STEP, STEPS = instances.STEP, instances.STEPS


def _pad_unit():
    return (0, [0] * STEP, [0] * STEP, 0)


def vu_units(x: list[int], w: list[int], accs: list[int]) -> list[tuple]:
    """The 96 units of one VU with their recorded output words."""
    units, c = [], 0
    for s in range(STEPS):
        units.append((c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16], accs[s]))
        c = accs[s]
    return units


class ChainRunner:
    def __init__(self, device: str, target: float, rate_log2: int = 2, hints: str = "torch"):
        self.p = REAL
        self.device = device
        self.hints_mode = hints
        self.target = target
        self.rate_log2 = rate_log2
        t0 = time.perf_counter()
        self.sys = compile_unit(self.p, chain=True)
        self.tb = tables(self.sys, self.device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg: dict[int, Config] = {}

    def cfg(self, l: int) -> Config:
        if l not in self._cfg:
            self._cfg[l] = config_for(l, self.target, self.rate_log2)
        return self._cfg[l]

    def layout(self, n_vus: int, min_l: int = 256) -> Layout:
        l = max(min_l, 1 << (n_vus * STEPS - 1).bit_length())
        return Layout(l, n_vus, STEPS)

    def prove_vus(self, vus: list[tuple[list, list, list, int]], check=True, mutate=None, min_l: int = 256):
        """``vus``: (x, w, accs, y16) per VU (honest data: the recorded accumulators)."""
        lay = self.layout(len(vus), min_l)
        units, y16 = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (x, w, accs, y) in enumerate(vus):
            units += vu_units(x, w, accs)
            y16[STEPS * v + STEPS - 1] = y
        while len(units) < lay.l:
            units.append(_pad_unit())
        a, b = [u[1] for u in units], [u[2] for u in units]
        t0 = time.perf_counter()
        pub = public_vectors(self.p, None, a, b, None, device=self.device)
        c_in, y_out = [u[0] for u in units], [u[3] for u in units]
        if self.hints_mode == "torch":
            hints = hints_torch(self.p, self.sys, c_in, a, b, y_out, self.device)
        else:
            hints = hints_tensor(self.p, self.sys, units, self.device)
        if check:      # the gate: the torch hint generator must reproduce the checker's gen_step exactly
            ref = hints_tensor(self.p, self.sys, units, self.device)
            alt = hints_torch(self.p, self.sys, c_in, a, b, y_out, self.device)
            assert torch.equal(ref, alt), "hints_torch != gen_step"
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        proof = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                      chain=(lay, y16), words={"a": a, "b": b})
        proof.timings["hints_host"] = hints_seconds
        return proof, (a, b, y16), lay

    def verify_vus(self, proof, pubs, lay: Layout):
        a, b, y16 = pubs
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, y16))


def _vu_data(vu: instances.VuTier, i: int):
    x, w, y16 = vu.instance(i)
    return (x, w, vu.accs(i), int(y16))


def gate_vu(args) -> int:
    R = ChainRunner(args.device, args.target, args.rate_log2, args.hints)
    log = print
    log(f"chain system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; device {args.device}")
    root = Path(args.root)
    report = {"system": R.sys.summary(), "positives": [], "negatives": [], "mutations": []}
    failures = 0
    per_proof = args.batch // STEPS

    # (1) the VUs, recorded accumulators and output words
    vu = instances.load("vu-k1536", root)
    for lo in range(0, args.vus, per_proof):
        vus = [_vu_data(vu, i) for i in range(lo, min(lo + per_proof, args.vus))]
        proof, pubs, lay = R.prove_vus(vus, check=True)
        ok, why = R.verify_vus(proof, pubs, lay)
        log(f"vu-k1536 [{lo}, {lo + len(vus)}): l={lay.l} prover {proof.timings['total']:.2f}s -> {ok} {why}")
        report["positives"].append({"tier": "vu-k1536", "range": [lo, lo + len(vus)], "l": lay.l, "ok": ok, "why": why,
                                    "timings": proof.timings})
        failures += not ok

    # (2) the 52 negatives, each as a VU claim on its own (l = 256: 2 VUs' worth of columns)
    neg = instances.load("vu-k1536-neg", root)
    cols = {n: i for i, n in enumerate(neg.columns)}
    foreign, _, _ = R.prove_vus([([0] * (STEP * STEPS), [0] * (STEP * STEPS), [0] * STEPS, 0)], check=False)
    for i in range(neg.n):
        row = neg.index[i]
        x, w, y16 = neg.instance(i)
        meta = {"id": i, "mistake": row[cols["mistake"]], "verdict": row[cols["verdict"]]}
        try:
            # the honest chain's accumulators (may Reject on off-domain operands or an infinite accumulator)
            accs, c = [], 0
            for s in range(STEPS):
                c = step_output(R.p, c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
                accs.append(c)
            proof, pubs, lay = R.prove_vus([(x, w, accs, int(y16))], check=False)
            ok, why = R.verify_vus(proof, pubs, lay)
        except Exception as e:  # noqa: BLE001
            # no honest witness exists: the verifier must reject any proof of this statement -- run it on a proof
            # of a different statement (the trivial VU)
            lay = R.layout(1)
            a = [x[16 * s:16 * s + 16] for s in range(STEPS)] + [[0] * STEP] * (lay.l - STEPS)
            b = [w[16 * s:16 * s + 16] for s in range(STEPS)] + [[0] * STEP] * (lay.l - STEPS)
            yv = torch.zeros(lay.l, dtype=torch.int64)
            yv[STEPS - 1] = int(y16)
            ok, why = R.verify_vus(foreign, (a, b, yv), lay)
            why = f"prover raised {type(e).__name__}; verifier on a foreign proof: {why}"
        report["negatives"].append({**meta, "accepted": ok, "why": why})
        if ok:
            log(f"!! NEGATIVE ACCEPTED: {meta} -- soundness bug")
            failures += 1
    log(f"negatives: {sum(not r['accepted'] for r in report['negatives'])}/{len(report['negatives'])} rejected")

    # (3) mutated witnesses: one row per (gadget scope, row class), flipped in one column of a 2-VU proof
    seen, picks = set(), []
    for r in R.sys.rows:
        key = (".".join(r.name.split(".")[:2]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    base = [_vu_data(vu, 0), _vu_data(vu, 1)]
    rng = np.random.default_rng(11)
    for r in picks:
        col = int(rng.integers(0, 2 * STEPS))
        delta = int(rng.integers(1, 3))

        def mutate(W, idx=r.idx, col=col, delta=delta):
            W[idx, col] = (W[idx, col] + delta) % P

        proof, pubs, lay = R.prove_vus(base, check=False, mutate=mutate)
        ok, why = R.verify_vus(proof, pubs, lay)
        report["mutations"].append({"row": r.name, "kind": r.kind, "col": col, "delta": delta, "accepted": ok, "why": why})
        if ok:
            log(f"!! MUTATION ACCEPTED: row {r.name} ({r.kind}) -- soundness bug")
            failures += 1
    # (3b) chain-specific: a wrong claimed word, a broken link, a nonzero c_0
    extra = []
    vus = base
    proof, (a, b, y16v), lay = R.prove_vus(vus, check=False)
    y_bad = y16v.clone()
    y_bad[STEPS - 1] ^= 1
    extra.append(("claimed y16 flipped", R.verify_vus(proof, (a, b, y_bad), lay)))
    c_rows = R.sys.chain["c"]

    def break_link(W):
        W[c_rows[2], 40] = (W[c_rows[2], 40] + 1) % P          # unit 40's c.f no longer equals unit 39's y.f
    proof, pubs, lay = R.prove_vus(vus, check=False, mutate=break_link)
    extra.append(("link c.f[40] != y.f[39]", R.verify_vus(proof, pubs, lay)))

    def nonzero_c0(W):
        W[c_rows[1], STEPS] = 1                                 # VU 1 starts from t = 1 instead of +0
    proof, pubs, lay = R.prove_vus(vus, check=False, mutate=nonzero_c0)
    extra.append(("c_0 != 0 at a chain start", R.verify_vus(proof, pubs, lay)))
    for name, (ok, why) in extra:
        report["mutations"].append({"row": name, "kind": "chain", "accepted": ok, "why": why})
        if ok:
            log(f"!! CHAIN NEGATIVE ACCEPTED: {name} -- soundness bug")
            failures += 1
    log(f"mutations: {sum(not r['accepted'] for r in report['mutations'])}/{len(report['mutations'])} rejected "
        f"(one per gadget scope x row class + {len(extra)} chain-specific, l={lay.l})")
    report["failures"] = failures
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


def bench_vu(args) -> int:
    from .run import _git_commit

    R = ChainRunner(args.device, args.target, args.rate_log2, args.hints)
    root = Path(args.root)
    vu = instances.load("vu-k1536", root)
    n_vus = args.batch // STEPS
    vus = [_vu_data(vu, i % vu.n) for i in range(n_vus)]
    lay = R.layout(n_vus)
    l = lay.l
    cfg = R.cfg(l)
    print(f"config {cfg}; layout {lay}; system {R.sys.summary()}")
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for rep in range(args.reps + 1):
        proof, pubs, lay = R.prove_vus(vus, check=False)
        t0 = time.perf_counter()
        ok, why = R.verify_vus(proof, pubs, lay)
        vsec = time.perf_counter() - t0
        assert ok, why
        if rep == 0:
            print("warm-up:", {k: round(v, 3) for k, v in proof.timings.items()})
            continue
        runs.append((proof.timings, vsec))
        print(f"rep {rep}: prover {proof.timings['total']:.3f}s (+hints {proof.timings['hints_host']:.3f}s) verifier {vsec:.3f}s")
    keys = runs[0][0].keys()
    med = {k: statistics.median(r[0][k] for r in runs) for k in keys}
    vmed = statistics.median(r[1] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = proof.n_bytes()
    snd = soundness(cfg, R.tb.L, R.tb.Q)
    total = med["total"] + med["hints_host"]
    per_vu = total / n_vus
    hw = contract.probe_hardware()
    sw = {"backend": {"name": "b-ligero chain (backends/direct/ligero)", "commit": _git_commit(), "hash": proof.hash_name,
                      "challenge_expander": "SHAKE-256 (placeholder)"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    fp = contract.fingerprint(
        profile=contract.PROFILE, domain=contract.DOMAIN, K=STEP * STEPS, B=n_vus, proof_class="NON_ZK_PROOF_DIAGNOSTIC",
        authentication="excluded", zk_mode="none",
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_l": cfg.l,
                  "opened_columns": cfg.t, "achieved_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation (a function of Params only, {R.compile_seconds:.2f}s)",
        retention="retained",
        instances=contract.instances_ref("vu-k1536", 0, n_vus, instances.manifest_sha256(root)),
        scope="vu", omitted="external authentication, ZK masking",
        rows=R.sys.m, linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        witness_generation=f"{R.hints_mode} hint generator (gen_step semantics) + torch program (derived rows)",
        note=f"{n_vus} VUs per proof of l = {l} columns ({l - n_vus * STEPS} pad columns)",
    )
    meas = [
        contract.measurement("t.witness", med["witness"] + med["hints_host"]),
        contract.measurement("t.encoding_commitment", med["encode"] + med["transfer"] + med["merkle"]),
        contract.measurement("t.arithmetic", med["tests"]),
        contract.measurement("t.lookup", 0.0),
        contract.measurement("t.zk_additional", 0.0),
        contract.measurement("t.serialization", med["openings"] + med.get("statement", 0.0)),
        contract.measurement("t.total", total),
        contract.measurement("verifier.seconds", vmed),
        contract.measurement("transcript.bytes", nbytes["total"]),
        contract.measurement("rounds.protocol", 2),
        contract.measurement("rounds.sequential_depth", 2),
        contract.measurement("mem.peak_device_bytes", peak_dev),
        contract.measurement("census.witness_elements", R.sys.m * l),
        contract.measurement("census.committed_elements", R.sys.m * cfg.n),
        contract.measurement("census.bytes_encoded", R.sys.m * cfg.n * 4),
        contract.measurement("vu.seconds_per_vu", per_vu, "s"),
        contract.measurement("vu.count", n_vus, "count"),
        contract.measurement("split.hints_seconds", med["hints_host"], "s"),
        contract.measurement("split.witness_torch_seconds", med["witness"], "s"),
        contract.measurement("split.encode_seconds", med["encode"], "s"),
        contract.measurement("split.transfer_seconds", med["transfer"], "s"),
        contract.measurement("split.merkle_seconds", med["merkle"], "s"),
        contract.measurement("split.tests_seconds", med["tests"], "s"),
        contract.measurement("split.openings_seconds", med["openings"], "s"),
        contract.measurement("proof.opened_columns_bytes", nbytes["opened_columns"], "B"),
        contract.measurement("proof.merkle_paths_bytes", nbytes["merkle_paths"], "B"),
        contract.measurement("proof.w_h_q_bytes", nbytes["w"] + nbytes["h"] + nbytes["q_chain"], "B"),
    ]
    result = {"schema": contract.RESULT_SCHEMA, "run_id": args.run_id or f"b-ligero-vu-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": "passed", "checker": "backends.direct.ligero.run bench-vu (verifier accepted every rep)",
                             "detail": f"{args.reps} reps, {n_vus} VUs at l={l}", "evidence": {"soundness": snd}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"seconds_per_vu": per_vu, "prover_total": total, "verifier": vmed, "bytes": nbytes,
                      "peak_device_bytes": peak_dev, "soundness_log2": snd["total_log2"], "medians": med}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    return 0
