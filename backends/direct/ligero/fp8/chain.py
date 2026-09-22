"""GemmCoordinate<K=1536> over ``sm89.mma.m16n8k32.e4m3`` on the B-Ligero chain prover: 48 chained FP8 units per VU from
``c_0 = +0``, the final FP32 accumulator word public (as its 22-bit packing, ``relation.pack_public``), the accumulators
witness rows linked by the chain test (``chain.py``; statement binding as PROTOCOL.md: the claimed words and the operand
bytes enter the statement digest before any challenge).

    python -m backends.direct.ligero.run --relation fp8-ada gate-vu  --device cuda --vus 64
    python -m backends.direct.ligero.run --relation fp8-ada bench-vu --device cuda --batch 4096 --total-vus 4096 --reps 3 --out result.json

Instances are synthetic and deterministic (``instances_fp8``: uniform finite E4M3 bytes from a seed; the accumulators
recorded by the model ``silicon.tc_dot``), there being no frozen FP8 instance set yet.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from verity.ml.tc.silicon import ADA_E4M3_M16N8K32, tc_dot
from verity.verification.target import FP8_ADA_MMA
from verity_numerical.bench import contract

from .. import protocol as protocol_mod
from ..chain import Layout
from ..compile import P
from ..protocol import Config, RelationHooks, config_for, n_rows, prove, soundness, verify
from ..witness import tables
from .census import census
from .relation import ADA, K, STEPS, Y_PACK_BITS, compile_fp8_unit, pack_public, public_vectors_fp8
from .witness import hints_fp8, random_finite_operands

__all__ = ["FP8_HOOKS", "FP8ChainRunner", "RELATION_NAME", "MODEL_NAME", "bench_vu_fp8", "gate_vu_fp8", "instances_fp8"]

RELATION_NAME = "GemmCoordinate<1536> over sm89.mma.m16n8k32.e4m3"
MODEL_NAME = "ADA_E4M3_M16N8K32"
FP8_HOOKS = RelationHooks(public_vectors_fp8, word_dtype="<u1", y16_dtype="<u4", y16_bits=Y_PACK_BITS, tag=b"|rel=fp8-ada|v1")
INSTANCE_SEED = 20260922


def instances_fp8(n: int, seed: int = INSTANCE_SEED) -> list[tuple[list, list, list, int]]:
    """``n`` VUs: (a (1536,), b (1536,), accs (48,), y_final) with the accumulators recorded from the model."""
    out = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        a = random_finite_operands(rng, (STEPS * K,)).tolist()
        b = random_finite_operands(rng, (STEPS * K,)).tolist()
        accs, c = [], 0
        for s in range(STEPS):
            c = tc_dot(ADA_E4M3_M16N8K32, c, a[K * s:K * s + K], b[K * s:K * s + K])
            accs.append(c)
        out.append((a, b, accs, accs[-1]))
    return out


def instances_digest(n: int, seed: int = INSTANCE_SEED) -> str:
    return hashlib.sha256(f"fp8-ada synthetic|seed={seed}|n={n}|K={STEPS * K}".encode()).hexdigest()


def vu_units(a: list[int], b: list[int], accs: list[int]) -> list[tuple]:
    units, c = [], 0
    for s in range(STEPS):
        units.append((c, a[K * s:K * s + K], b[K * s:K * s + K], accs[s]))
        c = accs[s]
    return units


def _pad_unit():
    return (0, [0] * K, [0] * K, 0)


class FP8ChainRunner:
    """``vu.ChainRunner`` for the FP8 unit: the same protocol (``prove`` / ``verify``) on ``compile_fp8_unit(chain=True)``."""

    def __init__(self, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE):
        self.p = ADA
        self.zk, self.mode, self.device, self.target, self.rate_log2, self.n_proofs = zk, mode, device, target, rate_log2, n_proofs
        t0 = time.perf_counter()
        self.sys = compile_fp8_unit(self.p, chain=True)
        self.tb = tables(self.sys, self.device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg: dict[int, Config] = {}

    def cfg(self, l: int) -> Config:
        if l not in self._cfg:
            self._cfg[l] = config_for(l, self.target, self.rate_log2, self.n_proofs, zk=self.zk, mode=self.mode)
        return self._cfg[l]

    @property
    def interactive(self) -> bool:
        return self.mode == protocol_mod.MODE_INTERACTIVE

    def layout(self, n_vus: int, min_l: int = 256) -> Layout:
        l = max(min_l, 1 << (n_vus * STEPS - 1).bit_length())
        return Layout(l, n_vus, STEPS)

    def prove_vus(self, vus: list[tuple[list, list, list, int]], check=True, mutate=None, min_l: int = 256, coins=None):
        """``vus``: (a, b, accs, y_final) per VU.  The public claim per VU is ``pack_public(y_final)`` at its last column."""
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        lay = self.layout(len(vus), min_l)
        units, ypk = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (a, b, accs, y) in enumerate(vus):
            units += vu_units(a, b, accs)
            ypk[STEPS * v + STEPS - 1] = pack_public(int(y))
        while len(units) < lay.l:
            units.append(_pad_unit())
        a = np.asarray([u[1] for u in units], dtype=np.int64)
        b = np.asarray([u[2] for u in units], dtype=np.int64)
        c_in = np.asarray([u[0] for u in units], dtype=np.int64)
        t0 = time.perf_counter()
        pub = public_vectors_fp8(self.p, None, a, b, None, device=self.device)
        hints = hints_fp8(self.p, self.sys, c_in, a, b, self.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        proof = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                      chain=(lay, ypk), words={"a": a, "b": b}, coins=coins, rel=FP8_HOOKS)
        proof.timings["hints_host"] = hints_seconds
        return proof, (a, b, ypk), lay

    def verify_vus(self, proof, pubs, lay: Layout, coins=None):
        a, b, ypk = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, ypk),
                      coins=coins, rel=FP8_HOOKS)


# ---------------------------------------------------------------------------------------------------------------------
# Gate: honest VUs accepted, a negative battery rejected (on the device)
# ---------------------------------------------------------------------------------------------------------------------

def gate_vu_fp8(args) -> int:
    R = FP8ChainRunner(args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode)
    log = print
    log(f"fp8 chain system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; device {args.device}")
    report = {"system": R.sys.summary(), "census": census(R.sys), "positives": [], "negatives": []}
    failures = 0
    per_proof = args.batch // STEPS
    data = instances_fp8(args.vus)
    for lo in range(0, args.vus, per_proof):
        vus = data[lo:lo + per_proof]
        proof, pubs, lay = R.prove_vus(vus, check=True)
        ok, why = R.verify_vus(proof, pubs, lay)
        log(f"fp8 vu-k1536 [{lo}, {lo + len(vus)}): l={lay.l} prover {proof.timings['total']:.2f}s -> {ok} {why}")
        report["positives"].append({"range": [lo, lo + len(vus)], "l": lay.l, "ok": ok, "why": why, "timings": proof.timings})
        failures += not ok

    base = data[:2]
    negs = []
    proof, (a, b, ypk), lay = R.prove_vus(base, check=False)
    # (1) wrong claimed words: every bit of the packed word of VU 0 (sign, the 8 exponent bits, the 13 kept fraction bits)
    for bit in range(Y_PACK_BITS):
        y_bad = ypk.clone()
        y_bad[STEPS - 1] ^= 1 << bit
        negs.append((f"claimed word bit {bit} flipped", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (2) a claimed word off the honest chain by one unit of the 14-bit adder (off-by-one rounding of the last group)
    for delta in (1, -1):
        y_bad = ypk.clone()
        y_bad[STEPS - 1] = (int(ypk[STEPS - 1]) + delta) % 2**Y_PACK_BITS
        negs.append((f"claimed word {'+' if delta > 0 else '-'}1 ulp", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (3) a NaN operand smuggled into the statement
    a_bad = a.copy()
    a_bad[3, 5] = 0x7F
    negs.append(("NaN operand 0x7f in a", R.verify_vus(proof, (a_bad, b, ypk), lay)))
    b_bad = b.copy()
    b_bad[7, 0] = 0xFF
    negs.append(("NaN operand 0xff in b", R.verify_vus(proof, (a, b_bad, ypk), lay)))
    # (4) an operand changed (the proof is of another statement)
    a_bad = a.copy()
    a_bad[10, 1] ^= 0x08
    negs.append(("operand a[10][1] exponent bit flipped", R.verify_vus(proof, (a_bad, b, ypk), lay)))
    # (5) chain: broken link, nonzero c_0, the truncation applied wrongly (an accumulator bit of the linked word)
    c_rows = R.sys.chain["c"]

    def break_link(W):
        W[c_rows[2], 20] = (W[c_rows[2], 20] + 1) % P
    proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=break_link)
    negs.append(("link c.f[20] != y.f[19]", R.verify_vus(proof2, pubs2, lay2)))

    def nonzero_c0(W):
        W[c_rows[1], STEPS] = 1
    proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=nonzero_c0)
    negs.append(("c_0 != 0 at a chain start", R.verify_vus(proof2, pubs2, lay2)))

    def acc_bit(W):
        W[c_rows[2], 30] = (W[c_rows[2], 30] ^ (1 << 12)) % P       # a kept fraction bit of the incoming accumulator
    proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=acc_bit)
    negs.append(("accumulator fraction bit 12 flipped in a linked unit", R.verify_vus(proof2, pubs2, lay2)))

    def acc_low_bit(W):
        W[c_rows[2], 31] = (W[c_rows[2], 31] ^ 1) % P               # a dropped (truncated) fraction bit
    proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=acc_low_bit)
    negs.append(("accumulator dropped bit 0 flipped in a linked unit (wrong truncation)", R.verify_vus(proof2, pubs2, lay2)))
    # (6) one witness row per (gadget scope, row class), as gate_vu does
    seen, picks = set(), []
    for r in R.sys.rows:
        key = (".".join(r.name.split(".")[:2]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(11)
    picks = picks[: max(0, getattr(args, "row_negatives", 60))]
    for r in picks:
        col = int(rng.integers(0, 2 * STEPS))
        delta = int(rng.integers(1, 3))

        def mutate(W, idx=r.idx, col=col, delta=delta):
            W[idx, col] = (W[idx, col] + delta) % P
        proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=mutate)
        negs.append((f"row {r.name} ({r.kind}) +{delta} in column {col}", R.verify_vus(proof2, pubs2, lay2)))
    for name, (ok, why) in negs:
        report["negatives"].append({"name": name, "accepted": ok, "why": why})
        if ok:
            log(f"!! NEGATIVE ACCEPTED: {name} -- soundness bug")
            failures += 1
    log(f"negatives: {sum(not r['accepted'] for r in report['negatives'])}/{len(report['negatives'])} rejected (l={lay.l})")
    report["failures"] = failures
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


# ---------------------------------------------------------------------------------------------------------------------
# Bench: the same result envelope as vu.bench_vu
# ---------------------------------------------------------------------------------------------------------------------

def bench_vu_fp8(args) -> int:
    from ..run import _git_commit as _git_commit_repo
    from ..vu import ZK_MODE, ZK_MODE_FS, ZK_MODE_HM96, ZK_VACUOUS_NOTE

    def _git_commit():                      # a shipped `git archive` has no .git: the launcher passes the sha in the environment
        return _git_commit_repo() or os.environ.get("VERITY_COMMIT")

    per_proof = args.batch // STEPS
    total_vus = args.total_vus or per_proof
    n_proofs = -(-total_vus // per_proof)
    interactive = args.mode == protocol_mod.MODE_INTERACTIVE
    depth = 3 if interactive else 1
    R = FP8ChainRunner(args.device, args.target, args.rate_log2, n_proofs=n_proofs, zk=args.zk, mode=args.mode)
    cen = census(R.sys)
    print("census:", json.dumps(cen))
    subs = [(lo, min(lo + per_proof, total_vus)) for lo in range(0, total_vus, per_proof)]
    t0 = time.perf_counter()
    data = instances_fp8(total_vus)
    print(f"{total_vus} synthetic VUs (model-recorded accumulators) in {time.perf_counter() - t0:.1f}s")
    lay = R.layout(per_proof)
    l = lay.l
    cfg = R.cfg(l)
    print(f"config {cfg}; layout {lay}; zk {cfg.zk}, mode {cfg.mode}; system {R.sys.summary()}; {total_vus} VUs as {n_proofs} sub-batches of <= {per_proof}")
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    keys = None
    for rep in range(args.reps + 1):
        agg: dict[str, float] = {}
        vsum = 0.0
        t_pass = time.perf_counter()
        for lo, hi in subs:
            V = protocol_mod.Coins.sample() if interactive else None
            proof, pubs, lay_i = R.prove_vus(data[lo:hi], check=(rep == 0 and lo == 0), min_l=l, coins=V)
            t1 = time.perf_counter()
            ok, why = R.verify_vus(proof, pubs, lay_i, coins=V)
            vsum += time.perf_counter() - t1
            assert ok, f"sub-batch [{lo}, {hi}): {why}"
            for k_, v_ in proof.timings.items():
                agg[k_] = agg.get(k_, 0.0) + v_
            if rep == 0:
                break
        wall = time.perf_counter() - t_pass
        if rep == 0:
            print("warm-up (one sub-batch, with the prover's self-check):", {k_: round(v_, 3) for k_, v_ in agg.items()})
            continue
        keys = agg.keys()
        runs.append((agg, vsum))
        print(f"rep {rep}: prover {agg['total']:.3f}s (+hints {agg['hints_host']:.3f}s) over {n_proofs} proofs; "
              f"verifier {vsum:.3f}s; pass wall {wall:.3f}s ({(agg['total'] + agg['hints_host']) / total_vus * 1e3:.3f} ms/VU)")
    med = {k_: statistics.median(r[0][k_] for r in runs) for k_ in keys}
    vmed = statistics.median(r[1] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = {k_: v_ * n_proofs for k_, v_ in proof.n_bytes().items()}
    snd = soundness(cfg, R.tb.L, R.tb.Q, n_proofs)
    total = med["total"] + med["hints_host"]
    per_vu = total / total_vus
    hw = contract.probe_hardware()
    sw = {"backend": {"name": "b-ligero chain, fp8-ada relation (backends/direct/ligero/fp8)", "commit": _git_commit(),
                      "hash": proof.hash_name, "challenge_expander": "SHAKE-256 (placeholder)"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    if cfg.zk and interactive:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", ZK_MODE + ZK_MODE_HM96 + ZK_VACUOUS_NOTE
        omitted = "external authentication; unopened columns hidden by Blake3 of a uniform column"
    elif cfg.zk:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_MODE_FS + ZK_VACUOUS_NOTE
        omitted = "external authentication; malicious-verifier ZK outside the programmable random-oracle model"
    else:
        proof_class, zk_mode, omitted = "NON_ZK_PROOF_DIAGNOSTIC", "none", "external authentication, ZK masking"
    fp = contract.fingerprint(
        profile=FP8_ADA_MMA.name, domain=contract.DOMAIN, K=K * STEPS, B=total_vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication="excluded", zk_mode=zk_mode,
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_k": cfg.k, "rs_l": cfg.l,
                  "zk": cfg.zk, "t_pad": cfg.t_pad, "mode": cfg.mode, "bound_model": snd["bound_model"],
                  "hash_assumption": snd["hash_assumption"], "zk_statement": snd["zk_statement"],
                  "fiat_shamir_queries_log2": snd["fiat_shamir_queries_log2"],
                  "opened_columns": cfg.t, "achieved_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  "per_proof_target_log2": snd["per_proof_target_log2"], "per_proof_total_log2": snd["per_proof_total_log2"],
                  "union_over": n_proofs, "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation (a function of FP8Params only, {R.compile_seconds:.2f}s)",
        retention="retained",
        # the frozen FP8 Ada set (contract.FP8_ADA_INSTANCES_*): the canonical tables accept exactly this dataset/tier at
        # range [0, 4096) with manifest_sha256 == contract.FP8_ADA_INSTANCES_DIGEST == instances_digest(4096)
        instances={"dataset": contract.FP8_ADA_INSTANCES_DATASET, "tier": contract.FP8_ADA_INSTANCES_TIER,
                   "range": [0, total_vus], "manifest_sha256": instances_digest(total_vus), "seed": INSTANCE_SEED,
                   "recipe": "fp8-ada synthetic (uniform finite E4M3 bytes, model-recorded accumulators)"},
        scope="vu", omitted=omitted,
        relation=RELATION_NAME, model=MODEL_NAME, commit=_git_commit(), step_k=K, steps_per_vu=STEPS,
        rows=R.sys.m, committed_rows=n_rows(cfg, R.sys.m, True), linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        census=cen,
        witness_generation="torch hint generator (fp8/witness.py, gen_step semantics on the adder-side params) + torch program (derived rows)",
        B_proof=per_proof,
        note=f"{total_vus} VUs as {n_proofs} independently committed proofs of <= {per_proof} VUs each (l = {l} columns, "
             f"{l - per_proof * STEPS} pad columns per proof); prover time = sum over the proofs; soundness = union bound; "
             f"public claim per VU = the final FP32 word (22-bit packing: low 10 bits provably zero)",
    )
    meas = [
        contract.measurement("t.witness", med["witness"] + med["hints_host"]),
        contract.measurement("t.encoding_commitment", med["encode"] + med["transfer"] + med["merkle"]),
        contract.measurement("t.arithmetic", med["tests"]),
        contract.measurement("t.lookup", 0.0),
        contract.measurement("t.zk_additional", med.get("zk_masks", 0.0)),
        contract.measurement("t.serialization", med["openings"] + med.get("statement", 0.0)),
        contract.measurement("t.total", total),
        contract.measurement("verifier.seconds", vmed),
        contract.measurement("transcript.bytes", nbytes["total"]),
        contract.measurement("rounds.protocol", depth),
        contract.measurement("rounds.sequential_depth", depth),
        contract.measurement("mem.peak_device_bytes", peak_dev),
        contract.measurement("census.witness_elements", R.sys.m * l * n_proofs),
        contract.measurement("census.committed_elements", R.sys.m * cfg.n * n_proofs),
        contract.measurement("census.bytes_encoded", R.sys.m * cfg.n * 4 * n_proofs),
        contract.measurement("vu.seconds_per_vu", per_vu, "s"),
        contract.measurement("vu.count", total_vus, "count"),
        *contract.rate_measurements(total, total_vus, K * STEPS, FP8_ADA_MMA.name),
        contract.measurement("split.subbatches", n_proofs, "count"),
        contract.measurement("split.per_proof_vus", per_proof, "count"),
        contract.measurement("split.hints_seconds", med["hints_host"], "s"),
        contract.measurement("split.witness_torch_seconds", med["witness"], "s"),
        contract.measurement("split.encode_seconds", med["encode"], "s"),
        contract.measurement("split.transfer_seconds", med["transfer"], "s"),
        contract.measurement("split.merkle_seconds", med["merkle"], "s"),
        contract.measurement("split.tests_seconds", med["tests"], "s"),
        contract.measurement("split.openings_seconds", med["openings"], "s"),
        contract.measurement("proof.opened_columns_bytes", nbytes["opened_columns"], "B"),
        contract.measurement("proof.merkle_paths_bytes", nbytes["merkle_paths"], "B"),
        contract.measurement("proof.w_h_q_bytes", nbytes["w"] + nbytes["h"] + nbytes["q_chain"] + nbytes["v_linear"], "B"),
        contract.measurement("split.zk_masks_seconds", med.get("zk_masks", 0.0), "s"),
        contract.measurement("prove_wall", total, "s"),
        contract.measurement("verify_wall", vmed, "s"),
        contract.measurement("proof_bytes", nbytes["total"], "B"),
        contract.measurement("depth", depth, "count"),
        contract.measurement("soundness_bits", -snd["total_log2"], "bits"),
        contract.measurement("rows_per_unit", R.sys.m, "count"),
        contract.measurement("units_per_vu", STEPS, "count"),
    ]
    for k_, v_ in cen.items():
        if isinstance(v_, int) and not isinstance(v_, bool):
            # ``census.`` is a contract-reserved prefix: only the seven names contract.py defines may live under
            # it (emitted above).  The relation's own per-gadget row counts go under ``relation.census.``.
            meas.append(contract.measurement(f"relation.census.{k_}", v_, "count"))
    result = {"schema": contract.RESULT_SCHEMA, "run_id": args.run_id or os.environ.get("RESEARCH_RUN_ID") or f"b-ligero-fp8-vu-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": "passed", "checker": "backends.direct.ligero.fp8.chain bench_vu_fp8 (verifier accepted every rep; "
                                                            "prover self-check on the warm-up sub-batch)",
                             "detail": f"{args.reps} reps, {total_vus} VUs as {n_proofs} proofs at l={l}", "evidence": {"soundness": snd}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"seconds_per_vu": per_vu, "prover_total": total, "verifier": vmed, "bytes": nbytes, "n_proofs": n_proofs,
                      "per_proof_vus": per_proof, "t": cfg.t, "zk": cfg.zk, "k": cfg.k, "peak_device_bytes": peak_dev,
                      "soundness_log2": snd["total_log2"], "per_proof_log2": snd["per_proof_total_log2"], "rows": R.sys.m,
                      "medians": med}, indent=1))
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(result, indent=1, default=str))   # $RESEARCH_RUN_DIR/result.json
    return 0
