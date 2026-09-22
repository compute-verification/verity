"""Correctness gate and measurement for the Ligero unit prover on the frozen instance set.

    python -m backends.direct.ligero.run gate  --root fixtures/bench-instances/v1 --device cuda --vus 64
    python -m backends.direct.ligero.run bench --root fixtures/bench-instances/v1 --device cuda --reps 5 --out result.json

Gate: (1) the 4096 ``tu-k16`` units in one proof of l = 4096 -> accept; (2) the units of the first ``--vus`` VUs of
``vu-k1536`` (every recorded intermediate accumulator is a unit's public output) -> accept; (3) every ``vu-k1536-neg``
instance as a unit-level negative (its last unit claims an FP32 word whose BF16 cast is the wrong claimed output;
``reject``-class negatives fail the verifier's domain check) -> reject; (4) mutated witnesses: one row of every gadget
scope and row class flipped in one column before commitment -> reject.  An accepted negative aborts with exit 2.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys as _sys
import time
from pathlib import Path

import numpy as np
import torch

from verity_numerical.bench import contract, instances
from verity_numerical.checker.params import REAL
from verity_numerical.checker.step import step_output

from . import merkle
from .compile import P
from .protocol import Config, config_for, prove, soundness, verify
from .unit import compile_unit, public_vectors
from .witness import hints_tensor, tables

STEP, STEPS = instances.STEP, instances.STEPS


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def pad_units(units: list, ys: list, l: int):
    """Pad to l units with the trivial unit (c = +0, a = b = 0 -> y = +0)."""
    units = list(units)
    ys = list(ys)
    while len(units) < l:
        units.append((0, [0] * STEP, [0] * STEP))
        ys.append(0)
    return units, ys


def units_of_vu(vu: instances.VuTier, i: int) -> tuple[list, list]:
    x, w, _ = vu.instance(i)
    acc = vu.accs(i)
    units, ys = [], []
    c = 0
    for s in range(STEPS):
        units.append((c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16]))
        ys.append(acc[s])
        c = acc[s]
    return units, ys


def split_pub(units, ys):
    return [u[0] for u in units], [u[1] for u in units], [u[2] for u in units], list(ys)


class Runner:
    def __init__(self, device: str, target: float, rate_log2: int = 2):
        self.p = REAL
        self.device = device
        self.target = target
        self.rate_log2 = rate_log2
        t0 = time.perf_counter()
        self.sys = compile_unit(self.p)
        self.tb = tables(self.sys, device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg: dict[int, Config] = {}

    def cfg(self, l: int) -> Config:
        if l not in self._cfg:
            self._cfg[l] = config_for(l, self.target, self.rate_log2)
        return self._cfg[l]

    def prove_units(self, units, ys, check=True, mutate=None, min_l: int = 256):
        l = max(min_l, 1 << (len(units) - 1).bit_length())
        units, ys = pad_units(units, ys, l)
        c, a, b, y = split_pub(units, ys)
        t0 = time.perf_counter()
        pub = public_vectors(self.p, c, a, b, y, device=self.device)
        hints = hints_tensor(self.p, self.sys, units, self.device)
        hints_seconds = time.perf_counter() - t0
        proof = prove(self.p, self.sys, self.tb, self.cfg(l), pub, hints, self.device, check=check, mutate=mutate)
        proof.timings["hints_host"] = hints_seconds
        return proof, (c, a, b, y), l

    def verify_units(self, proof, pubs, l):
        c, a, b, y = pubs
        return verify(self.p, self.sys, self.tb, self.cfg(l), c, a, b, y, proof, self.device)


# ---------------------------------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------------------------------

def gate(args) -> int:
    R = Runner(args.device, args.target, args.rate_log2)
    log = print
    log(f"system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; device {args.device}")
    root = Path(args.root)
    report = {"system": R.sys.summary(), "positives": [], "negatives": [], "mutations": []}
    failures = 0

    # (1) tu-k16
    tu = instances.load("tu-k16", root)
    units = [tu.instance(i)[:3] for i in range(tu.n)]
    ys = [tu.instance(i)[3] for i in range(tu.n)]
    for lo in range(0, tu.n, args.batch):
        proof, pubs, l = R.prove_units(units[lo:lo + args.batch], ys[lo:lo + args.batch], check=True)
        ok, why = R.verify_units(proof, pubs, l)
        log(f"tu-k16 [{lo}, {lo + args.batch}): l={l} prover {proof.timings['total']:.2f}s -> {ok} {why}")
        report["positives"].append({"tier": "tu-k16", "range": [lo, lo + args.batch], "ok": ok, "why": why, "timings": proof.timings})
        failures += not ok

    # (2) VUs' units
    vu = instances.load("vu-k1536", root)
    all_units, all_ys = [], []
    for i in range(args.vus):
        u, y = units_of_vu(vu, i)
        all_units += u
        all_ys += y
    for lo in range(0, len(all_units), args.batch):
        proof, pubs, l = R.prove_units(all_units[lo:lo + args.batch], all_ys[lo:lo + args.batch], check=True)
        ok, why = R.verify_units(proof, pubs, l)
        log(f"vu-k1536 units [{lo}, {lo + args.batch}) of {args.vus} VUs: l={l} prover {proof.timings['total']:.2f}s -> {ok} {why}")
        report["positives"].append({"tier": "vu-k1536", "units": [lo, min(lo + args.batch, len(all_units))], "ok": ok, "why": why})
        failures += not ok

    # (3) negatives: unit-level image of every vu-k1536-neg instance
    neg = instances.load("vu-k1536-neg", root)
    cols = {n: i for i, n in enumerate(neg.columns)}
    neg_units, neg_ys, neg_meta = [], [], []
    for i in range(neg.n):
        row = neg.index[i]
        x, w, y16 = neg.instance(i)
        verdict = row[cols["verdict"]]
        mistake = row[cols["mistake"]]
        # honest chain (may fail on out-of-domain operands: those are the 'reject' class)
        try:
            c = 0
            for s in range(STEPS):
                c = step_output(R.p, c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
            honest_last = c
        except Exception as e:                        # Reject: the operands are off the domain
            honest_last = None
        cs = 0
        if honest_last is None:
            # last unit with the offending operands; any claimed word; the verifier must reject on the domain check
            unit = (0, x[-16:], w[-16:])
            claimed = int(y16) << 16
        else:
            # rebuild c_95 and claim the FP32 word claimed_y16 << 16 (its BF16 cast is the wrong claimed word)
            c = 0
            for s in range(STEPS - 1):
                c = step_output(R.p, c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
            unit = (c, x[-16:], w[-16:])
            claimed = int(y16) << 16
            if claimed == honest_last:
                # the wrong BF16 word's widening equals the honest FP32 word only if the honest cast is not that word
                raise AssertionError("negative maps to the honest word")
        neg_units.append(unit)
        neg_ys.append(claimed)
        neg_meta.append({"id": i, "mistake": mistake, "verdict": verdict})
    # each negative alone, padded to l = 256 with trivial units (t must fit in n; a proof at l = 256 is quick)
    trivial_proof, _, _ = R.prove_units([(0, [0] * STEP, [0] * STEP)], [0], check=False)
    for meta, unit, claimed in zip(neg_meta, neg_units, neg_ys):
        try:
            proof, pubs, l = R.prove_units([unit], [claimed], check=False)
            ok, why = R.verify_units(proof, pubs, l)
        except Exception as e:
            # the prover cannot even form hints (operands off the domain): the verifier must reject any proof for
            # this statement on its own domain check -- run it on the trivial units' proof
            u_pad, y_pad = pad_units([unit], [claimed], 256)
            ok, why = R.verify_units(trivial_proof, split_pub(u_pad, y_pad), 256)
            why = f"prover raised {type(e).__name__}; verifier on a foreign proof: {why}"
        report["negatives"].append({**meta, "accepted": ok, "why": why})
        if ok:
            log(f"!! NEGATIVE ACCEPTED: {meta} -- soundness bug")
            failures += 1
    log(f"negatives: {sum(not r['accepted'] for r in report['negatives'])}/{len(report['negatives'])} rejected")

    # (4) mutated witnesses: one row per (scope, kind), flipped in one column of a 4096-unit proof
    seen = set()
    picks = []
    for r in R.sys.rows:
        key = (".".join(r.name.split(".")[:2]), r.kind)
        if key in seen:
            continue
        seen.add(key)
        picks.append(r)
    l_mut = min(args.batch, 256)
    base_units, base_ys = units[:l_mut], ys[:l_mut]
    rng = np.random.default_rng(7)
    for r in picks:
        col = int(rng.integers(0, l_mut))
        delta = int(rng.integers(1, 3))

        def mutate(W, idx=r.idx, col=col, delta=delta):
            W[idx, col] = (W[idx, col] + delta) % P

        proof, pubs, l = R.prove_units(base_units, base_ys, check=False, mutate=mutate)
        ok, why = R.verify_units(proof, pubs, l)
        report["mutations"].append({"row": r.name, "kind": r.kind, "col": col, "delta": delta, "accepted": ok, "why": why})
        if ok:
            log(f"!! MUTATION ACCEPTED: row {r.name} ({r.kind}) -- soundness bug")
            failures += 1
    log(f"mutations: {sum(not r['accepted'] for r in report['mutations'])}/{len(report['mutations'])} rejected "
        f"(one per gadget scope x row class, l={l_mut})")
    report["failures"] = failures
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


# ---------------------------------------------------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------------------------------------------------

def bench(args) -> int:
    R = Runner(args.device, args.target, args.rate_log2)
    root = Path(args.root)
    tu = instances.load("tu-k16", root)
    units = [tu.instance(i)[:3] for i in range(args.batch)]
    ys = [tu.instance(i)[3] for i in range(args.batch)]
    l = args.batch
    cfg = R.cfg(l)
    print(f"config {cfg}; system {R.sys.summary()}")
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for rep in range(args.reps + 1):
        proof, pubs, _ = R.prove_units(units, ys, check=False)
        t0 = time.perf_counter()
        ok, why = R.verify_units(proof, pubs, l)
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
    per_unit = total / l
    hw = contract.probe_hardware()
    sw = {"backend": {"name": "b-ligero (backends/direct/ligero)", "commit": _git_commit(), "hash": merkle.HASH_NAME,
                      "challenge_expander": "SHAKE-256 (placeholder)"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    fp = contract.fingerprint(
        profile=contract.PROFILE, domain=contract.DOMAIN, K=16, B=l, proof_class="NON_ZK_PROOF_DIAGNOSTIC",
        authentication="excluded", zk_mode="none",
        security={"target": int(round(args.target)), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_l": cfg.l,
                  "opened_columns": cfg.t, "achieved_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing="constraint compilation (a function of Params only, %.2fs)" % R.compile_seconds,
        retention="retained",
        instances=contract.instances_ref("tu-k16", 0, l, instances.manifest_sha256(root)),
        scope="unit", omitted="chain linkage, epilogue, external authentication, ZK masking",
        rows=R.sys.m, linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        witness_generation="host Python gen_step (hints) + torch program (derived rows)",
    )
    meas = [
        contract.measurement("t.witness", med["witness"] + med["hints_host"]),
        contract.measurement("t.encoding_commitment", med["encode"] + med["transfer"] + med["merkle"]),
        contract.measurement("t.arithmetic", med["tests"]),
        contract.measurement("t.lookup", 0.0),
        contract.measurement("t.zk_additional", 0.0),
        contract.measurement("t.serialization", med["openings"]),
        contract.measurement("t.total", total),
        contract.measurement("verifier.seconds", vmed),
        contract.measurement("transcript.bytes", nbytes["total"]),
        contract.measurement("rounds.protocol", 2),
        contract.measurement("rounds.sequential_depth", 2),
        contract.measurement("mem.peak_device_bytes", peak_dev),
        contract.measurement("census.witness_elements", R.sys.m * l),
        contract.measurement("census.committed_elements", R.sys.m * cfg.n),
        contract.measurement("census.bytes_encoded", R.sys.m * cfg.n * 4),
        contract.measurement("census.hash_bytes", R.sys.m * cfg.n * 4 + 64 * (cfg.n - 1)),
        contract.measurement("unit.seconds_per_unit", per_unit, "s"),
        contract.measurement("unit.count", l, "count"),
        contract.measurement("split.hints_host_seconds", med["hints_host"], "s"),
        contract.measurement("split.witness_torch_seconds", med["witness"], "s"),
        contract.measurement("split.encode_seconds", med["encode"], "s"),
        contract.measurement("split.transfer_seconds", med["transfer"], "s"),
        contract.measurement("split.merkle_seconds", med["merkle"], "s"),
        contract.measurement("split.tests_seconds", med["tests"], "s"),
        contract.measurement("split.openings_seconds", med["openings"], "s"),
        contract.measurement("proof.opened_columns_bytes", nbytes["opened_columns"], "B"),
        contract.measurement("proof.merkle_paths_bytes", nbytes["merkle_paths"], "B"),
        contract.measurement("proof.w_h_bytes", nbytes["w"] + nbytes["h"], "B"),
    ]
    result = {"schema": contract.RESULT_SCHEMA, "run_id": args.run_id or f"b-ligero-unit-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": "passed", "checker": "backends.direct.ligero.run bench (verifier accepted every rep)",
                             "detail": f"{args.reps} reps at l={l}, tu-k16 [0, {l})", "evidence": {"soundness": snd}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"seconds_per_unit": per_unit, "prover_total": total, "verifier": vmed, "bytes": nbytes,
                      "peak_device_bytes": peak_dev, "soundness_log2": snd["total_log2"], "medians": med}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("gate", gate), ("bench", bench)):
        s = sub.add_parser(name)
        s.add_argument("--root", default=str(instances.DEFAULT_ROOT))
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        s.add_argument("--target", type=float, default=-128.0)
        s.add_argument("--rate-log2", type=int, default=2)
        s.add_argument("--batch", type=int, default=4096)
        s.add_argument("--out", default=None)
        s.set_defaults(fn=fn)
    sub.choices["gate"].add_argument("--vus", type=int, default=64)
    sub.choices["bench"].add_argument("--reps", type=int, default=5)
    sub.choices["bench"].add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    _sys.exit(main())
