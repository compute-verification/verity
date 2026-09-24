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
import os
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

from . import protocol as protocol_mod
from .compile import P
from .hints_torch import hints_torch
from .protocol import Config, config_for, prove, soundness, verify
from .unit import compile_unit, public_vectors
from .witness import hints_tensor, tables

STEP, STEPS = instances.STEP, instances.STEPS


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
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
    def __init__(self, device: str, target: float, rate_log2: int = 2, hints: str = "torch", zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE):
        self.zk = zk
        self.mode = mode
        self.p = REAL
        self.device = device
        self.hints_mode = hints
        self.target = target
        self.rate_log2 = rate_log2
        t0 = time.perf_counter()
        self.sys = compile_unit(self.p)
        self.tb = tables(self.sys, device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg: dict[int, Config] = {}

    def cfg(self, l: int) -> Config:
        if l not in self._cfg:
            self._cfg[l] = config_for(l, self.target, self.rate_log2, zk=self.zk, mode=self.mode)
        return self._cfg[l]

    @property
    def interactive(self) -> bool:
        return self.mode == protocol_mod.MODE_INTERACTIVE

    def prove_units(self, units, ys, check=True, mutate=None, min_l: int = 256):
        l = max(min_l, 1 << (len(units) - 1).bit_length())
        units, ys = pad_units(units, ys, l)
        c, a, b, y = split_pub(units, ys)
        t0 = time.perf_counter()
        pub = public_vectors(self.p, c, a, b, y, device=self.device)
        if self.hints_mode == "torch":
            hints = hints_torch(self.p, self.sys, c, a, b, None, self.device)
        else:
            hints = hints_tensor(self.p, self.sys, units, self.device)
        if check:      # the gate: the torch hint generator must reproduce the checker's gen_step exactly
            ref = hints_tensor(self.p, self.sys, units, self.device)
            alt = hints_torch(self.p, self.sys, c, a, b, None, self.device)
            assert torch.equal(ref, alt), "hints_torch != gen_step"
        _sync(self.device)
        hints_seconds = time.perf_counter() - t0
        coins = protocol_mod.Coins.sample() if self.interactive else None   # the runner plays the verifier's step 0
        proof = prove(self.p, self.sys, self.tb, self.cfg(l), pub, hints, self.device, check=check, mutate=mutate,
                      words={"a": a, "b": b, "c": c, "y": y}, coins=coins)
        proof.timings["hints_host"] = hints_seconds
        return proof, (c, a, b, y), l

    def verify_units(self, proof, pubs, l):
        c, a, b, y = pubs
        return verify(self.p, self.sys, self.tb, self.cfg(l), c, a, b, y, proof, self.device,
                      coins=proof.coins if self.interactive else None)


# ---------------------------------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------------------------------

def gate(args) -> int:
    R = Runner(args.device, args.target, args.rate_log2, args.hints, zk=args.zk, mode=args.mode)
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
        except Exception:  # noqa: BLE001 -- Reject or any failure: the operands are off the domain
            honest_last = None
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
        except Exception as e:  # noqa: BLE001
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
    R = Runner(args.device, args.target, args.rate_log2, args.hints, zk=args.zk, mode=args.mode)
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
    sw = {"backend": {"name": "b-ligero (backends/direct/ligero)", "commit": _git_commit(), "hash": proof.hash_name,
                      "challenge_expander": "SHAKE-256 (placeholder)"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    fp = contract.fingerprint(
        profile=contract.PROFILE, domain=contract.DOMAIN, K=16, B=l, proof_class="NON_ZK_PROOF_DIAGNOSTIC",
        authentication="excluded", zk_mode="none",
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_l": cfg.l,
                  "opened_columns": cfg.t, "achieved_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation (a function of Params only, {R.compile_seconds:.2f}s)",
        retention="retained",
        instances=contract.instances_ref("tu-k16", 0, l, instances.manifest_sha256(root)),
        scope="unit", omitted="chain linkage, epilogue, external authentication, ZK masking",
        rows=R.sys.m, linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        witness_generation=f"{R.hints_mode} hint generator (gen_step semantics) + torch program (derived rows)",
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
        contract.measurement("rounds.protocol", 3 if R.interactive else 1),
        contract.measurement("rounds.sequential_depth", 3 if R.interactive else 1),     # PROTOCOL.md 8c
        contract.measurement("mem.peak_device_bytes", peak_dev),
        contract.measurement("census.witness_elements", R.sys.m * l),
        contract.measurement("census.committed_elements", R.sys.m * cfg.n),
        contract.measurement("census.bytes_encoded", R.sys.m * cfg.n * 4),
        contract.measurement("census.hash_bytes", R.sys.m * cfg.n * 4 + 64 * (cfg.n - 1)),
        contract.measurement("unit.seconds_per_unit", per_unit, "s"),
        contract.measurement("unit.count", l, "count"),
        contract.measurement("split.hints_seconds", med["hints_host"], "s"),
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


def set_impl(impl: str) -> None:
    """``--impl``: which witness program runs (witness.IMPL); the transcripts are identical either way."""
    from . import witness as witness_mod
    if impl not in ("device", "legacy"):
        raise ValueError(impl)
    witness_mod.IMPL = impl


_LEAF_SUFFIX: list[str] = []


def _relation_arg(value: str) -> str:
    """``--relation <rel>[+<leaf>]``: the registered relation for argparse's choices; a leaf suffix is remembered for
    ``--auth included-hash`` (``leaf.registry.split_relation_name``: ``+hash`` is Poseidon2)."""
    base, _, leaf = value.partition("+")
    _LEAF_SUFFIX.clear()
    if leaf:
        _LEAF_SUFFIX.append(leaf)
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    from .vu import bench_vu, gate_vu

    # --relation <registered>: relations.py on the generic chain runner (relchain.py); fp4-nvf4: the sm_120 NVFP4 unit
    # (backends/direct/ligero/fp4, its own chain runner); gate-vu / bench-vu dispatch there, nothing else changes for the
    # Ampere BF16 relation (vu.py on the frozen instance set)
    ap.add_argument("--relation", default="bf16", type=_relation_arg, choices=("bf16", "bf16-hopper", "fp8-hopper", "fp8-ada", "fp4-nvf4",
                                                             "bf16-ampere",
                                                             "bf16-hopper-v2", "bf16-ampere-v2", "fp8-hopper-v2", "fp8-ada-v2",
                                                             "bf16-hopper-v3", "bf16-ampere-v3", "fp8-hopper-v3", "fp8-ada-v3",
                                                             "bf16-hopper-v2x4", "bf16-ampere-v2x4", "fp8-hopper-v2x4",
                                                             "fp8-ada-v2x4",
                                                             "bf16-hopper-v3x4", "bf16-ampere-v3x4", "fp8-hopper-v3x4",
                                                             "fp8-ada-v3x4",
                                                             "bf16-hopper-x4", "bf16-ampere-x4", "fp8-hopper-x4", "fp8-ada-x4"),
                    help="bf16: the Ampere relation on the frozen instance set (vu.py); bf16-ampere: the same unit and frozen set "
                         "(--root) on the generic chain runner (relations.py; its --auth included-hash column); fp4-nvf4: "
                         "fp4/chain.py; the others: relations.py on synthetic instances.  A `+<leaf>` suffix (fp8-ada+poseidon2, "
                         "fp8-ada+blake3) selects the leaf scheme of --auth included-hash, as --leaf does")
    for name, fn in (("gate", gate), ("bench", bench), ("gate-vu", gate_vu), ("bench-vu", bench_vu)):
        s = sub.add_parser(name)
        s.add_argument("--root", default=str(instances.DEFAULT_ROOT))
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        s.add_argument("--target", type=float, default=-128.0)
        s.add_argument("--rate-log2", type=int, default=2)
        s.add_argument("--batch", type=int, default=4096)
        s.add_argument("--out", default=None)
        s.add_argument("--zk", action="store_true", help="HVZK masking (PROTOCOL.md 8b): blinded rows, mask rows, coset domain")
        s.add_argument("--mode", default=protocol_mod.MODE_INTERACTIVE, choices=protocol_mod.MODES,
                       help="interactive: verifier-supplied step-0 coins, challenges from the coins alone, statistical bound "
                            "(PROTOCOL.md 8c); fiat-shamir: no step 0, challenges = H(transcript), q x eps accounting (t ~ 288, D = 7)")
        s.add_argument("--hints", default="torch", choices=("torch", "python"), help="hint generator: torch (vectorised) or the checker's Python gen_step")
        s.add_argument("--impl", default="device", choices=("device", "legacy"),
                       help="prover implementation (lane hostphase; bit-identical transcripts): device = the witness program as one "
                            "fused CUDA kernel; legacy = the torch program (CUDA graph / eager). Recorded in the dump manifest")
        s.set_defaults(fn=fn)
    for name in ("gate", "gate-vu"):
        sub.choices[name].add_argument("--vus", type=int, default=64)
    sub.choices["gate-vu"].add_argument("--dump-proof", default=None, metavar="PATH",
                                        help="write each honest sub-batch proof as ligero-proof/v1 (PATH, or PATH_<lo> for several)")
    sub.choices["gate-vu"].add_argument("--dump-statement", default=None, metavar="PATH",
                                        help="write the matching ligero-statement/v1 (the verifier's public inputs + configuration)")
    for name in ("bench", "bench-vu"):
        sub.choices[name].add_argument("--reps", type=int, default=5)
        sub.choices[name].add_argument("--run-id", default=None)
    sub.choices["bench-vu"].add_argument("--total-vus", type=int, default=None,
                                         help="VUs in the whole batch, proved as ceil(total / (batch // 96)) sub-batches (union-bounded)")
    sub.choices["bench-vu"].add_argument("--hvzk-label", action="store_true",
                                         help="label a --zk run COMPLETE_HVZK_BACKEND even with the coin commitment on (conservative)")
    sub.choices["bench-vu"].add_argument("--dump-dir", default=None, metavar="DIR",
                                         help="write every recorded rep's sub-batch proofs (.stmt/.proof[/.coins]) + system.bin for backends/ligero-verify")
    sub.choices["bench-vu"].add_argument("--pipeline", type=int, default=4, metavar="N",
                                         help="registered relations and bf16/bf16-ampere (vu.py, lane live-2) on CUDA: prove the sub-batches N-deep on their own CUDA streams (lane hp2-host; "
                                              "bit-identical transcripts at every depth; default 4 -- 9-17 %% faster than 3 on the H100 and 4090, "
                                              "+1.2 GB of device memory per extra slot; 0/1 = the sequential schedule, for A/B). Recorded in the "
                                              "result's backend block")
    sub.choices["bench-vu"].add_argument("--dump-reps", type=int, default=None, metavar="N",
                                         help="with --dump-dir: dump only the first N recorded reps (default all; a 2^20-VU rep is ~14 GB of proofs)")
    for name in ("gate-vu", "bench-vu"):
        sub.choices[name].add_argument("--auth", nargs="?", const="included", default=None, choices=("included", "included-hash", "included-hash-shared"),
                                       help="authentication: `--auth` / `--auth included` (bf16 relation, bench-vu): commit the set's words as three "
                                            "SHA-256 word trees (auth.py), bind roots and leaf indices into the statement, ship and verify Merkle "
                                            "multiproofs of every boundary word.  `--auth included-hash` (registered relations): the operands are "
                                            "private witness wires hashed inside the relation with Poseidon2 (hashchain.py); the statement carries "
                                            "the roots, leaf indices, published digests and SHA-256 multiproofs of the ROW leaves (leaf/v2h).  "
                                            "`--auth included-hash-shared`: every DISTINCT operand row of a sub-batch is hashed once in a paired "
                                            "hash-side proof and each VU proves equality to its row by a post-commitment fingerprint "
                                            "(relchain.SharedHashedRunner; chain.py \"Fingerprint constraints\")")
        sub.choices[name].add_argument("--leaf", default=None, metavar="SCHEME",
                                       help="--auth included-hash: the leaf scheme hashing the operand rows (backends/direct/ligero/leaf/ "
                                            "registry: poseidon2 = the default and the historical `+hash`; `hash` is its alias)")
        sub.choices[name].add_argument("--tile", default=None, metavar="NXxNW",
                                       help="--auth included-hash: the committed workload is a GEMM tile of NX distinct x rows by NW distinct W "
                                            "columns (every pair one VU; sharing = tileNXxNW) instead of every VU its own rows (sharing = none)")
    sub.choices["bench-vu"].add_argument("--auth-cache", default=None, metavar="DIR", help="--auth: directory holding the built trees (reused when the bindings match)")
    sub.choices["bench-vu"].add_argument("--auth-procs", type=int, default=None, help="--auth: processes for leaf hashing (default: all cores)")
    sub.choices["bench-vu"].add_argument("--verifier", default=None, metavar="tcp://HOST:PORT",
                                         help="interactive mode: take the step-0 coins of every recorded rep from a LIVE verifier "
                                              "(backends.direct.ligero.live serve) on another machine and send it every prover message as "
                                              "it is produced; same protocol (8c), independent coins; adds t.total_live / net.* / verify.*")
    sub.choices["bench-vu"].add_argument("--target-bits", type=float, default=128.0,
                                         help="the VERIFIER's batch target: reject unless sum_i 2^-b_i <= 2^-target over the sub-batches "
                                              "presented and every proof's n_proofs is their count (PROTOCOL.md 8c; --target is the prover's)")
    sub.choices["gate-vu"].add_argument("--row-negatives", type=int, default=60, help="non-bf16 gate: witness-row mutations to try")
    for name in ("gate-vu", "bench-vu"):
        sub.choices[name].add_argument("--instance-procs", type=int, default=1,
                                       help="non-bf16 relations: processes recording the synthetic instances' accumulators")
        sub.choices[name].add_argument("--instances-cache", default=os.environ.get("LIGERO_INSTANCES_CACHE"), metavar="DIR",
                                       help="non-bf16 relations: keep the synthetic instance set as an .npz under DIR "
                                            "(default $LIGERO_INSTANCES_CACHE: pod_bootstrap.sh's env.sh sets it)")
    sub.choices["gate-vu"].add_argument("--unit-negatives", type=int, default=48,
                                        help="fp4-nvf4 gate: units per edge family for the wrong-rule witness battery (tu-k64-neg)")
    sub.choices["bench-vu"].add_argument("--warmup-subs", type=int, default=1,
                                         help="fp4-nvf4: sub-batches proven in the untimed warm-up rep 0 (default 1, main's harness; "
                                              "n_proofs = a full rep). Recorded in the result's note")
    for name in ("gate-vu", "bench-vu"):
        sub.choices[name].add_argument("--fp4-hints", default="device", choices=("device", "numpy"),
                                       help="fp4-nvf4: the decode + hint generator -- device (fp4/hints_device.py: torch on the prover's "
                                            "device, one CUDA graph per (system, l); bit-identical) or numpy (fp4/witness.py on the host, "
                                            "lane fp4-proof's path). Recorded in the result and the dump manifest")
    args = ap.parse_args(argv)
    set_impl(getattr(args, "impl", "device"))
    if _LEAF_SUFFIX:
        # --relation <rel>+<leaf>: the suffix names the leaf scheme (--leaf must agree when both are given)
        if getattr(args, "leaf", None) not in (None, _LEAF_SUFFIX[0]):
            ap.error(f"--relation names leaf scheme {_LEAF_SUFFIX[0]!r}, --leaf {args.leaf!r}")
        args.leaf = _LEAF_SUFFIX[0]
    if getattr(args, "leaf", None) is not None:
        from .leaf import registry as leaf_registry
        if getattr(args, "auth", None) != "included-hash":
            ap.error("--leaf / --relation <rel>+<leaf> select the leaf scheme of --auth included-hash")
        try:
            args.leaf = leaf_registry.canonical(args.leaf)
        except KeyError as e:
            ap.error(str(e))
    if args.relation != "bf16":
        from .relations import relation
        from .relchain import bench_vu_rel, gate_vu_rel
        if args.cmd not in ("gate-vu", "bench-vu"):
            ap.error(f"--relation {args.relation} supports gate-vu and bench-vu")
        if getattr(args, "auth", None) == "included":
            ap.error(f"--auth included (verifier-side word openings) is implemented for the bf16 relation only, not --relation {args.relation}")
        if getattr(args, "auth", None) in ("included-hash", "included-hash-shared") and args.relation == "fp4-nvf4":
            ap.error("--auth included-hash: the fp4-nvf4 relation's packed-nibble operands are not composed yet (see hashchain.py)")
        if args.relation == "fp4-nvf4":
            from .fp4.chain import bench_vu_fp4, gate_vu_fp4
            return (gate_vu_fp4 if args.cmd == "gate-vu" else bench_vu_fp4)(args)
        rel = relation(args.relation)
        return (gate_vu_rel if args.cmd == "gate-vu" else bench_vu_rel)(args, rel)
    if getattr(args, "auth", None) in ("included-hash", "included-hash-shared"):
        ap.error("--auth included-hash[-shared] is a mode of the registered relations (--relation bf16-hopper | fp8-hopper | fp8-ada)")
    return args.fn(args)


if __name__ == "__main__":
    _sys.exit(main())
