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

from . import protocol as protocol_mod
from .chain import Layout
from .compile import P
from .hints_torch import hints_torch
from .protocol import Config, config_for, n_rows, prove, soundness, verify

ZK_MODE = ("Ligero HVZK masking: t_pad uniform coefficients (Z_H s_j) per row, 6D committed mask rows "
           "(prox/lin/quad x2/chain x2: q masked by a uniform zero-sum polynomial, AHIV22 u_add -- red team F1 fixed), "
           "evaluation on the coset g<w_n> disjoint from H (PROTOCOL.md 8b)")
ZK_MODE_HM96 = ("; malicious-verifier: HM96/GK step-0 commitment c_i = H(r_i || s_i) to both verifier coins before root, "
                "opened after root / after the test messages, prover aborts on a mismatched opening; challenges are "
                "functions of the coins alone (F2 fixed), simulator by rewinding with a CRH only (PROTOCOL.md 8c; "
                "redteam/simulator.py against the real verify())")
ZK_MODE_FS = ("; Fiat-Shamir mode: no step 0, challenges = H(statement, transcript); zero knowledge is HVZK, "
              "NIZK only in the programmable random-oracle model")
ZK_VACUOUS_NOTE = ("; NOTE: this statement publishes the operands a, b and the words y16 (statement_digest), so privacy "
                   "is vacuous for it today -- the ZK claim is about the construction (masking + coins) and becomes "
                   "meaningful once operands are committed (auth-integration LEAF_V2)")
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
    def __init__(self, device: str, target: float, rate_log2: int = 2, hints: str = "torch", n_proofs: int = 1,
                 zk: bool = False, mode: str = protocol_mod.MODE_INTERACTIVE):
        self.p = REAL
        self.zk = zk
        self.mode = mode
        self.device = device
        self.hints_mode = hints
        self.target = target
        self.rate_log2 = rate_log2
        self.n_proofs = n_proofs            # sub-batches the target is union-bounded over (config_for)
        t0 = time.perf_counter()
        self.sys = compile_unit(self.p, chain=True)
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
        """``vus``: (x, w, accs, y16) per VU (honest data: the recorded accumulators).  ``coins``: the verifier's
        step-0 coin source.  Interactive mode with ``coins=None``: the RUNNER plays the honest verifier's step 0
        (``Coins.sample()``; the same object is what ``verify_vus`` then uses as its own coins -- the runner is both
        parties, as in every gate).  Fiat-Shamir mode: no coins."""
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        lay = self.layout(len(vus), min_l)
        units, y16 = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (x, w, accs, y) in enumerate(vus):
            units += vu_units(x, w, accs)
            y16[STEPS * v + STEPS - 1] = y
        while len(units) < lay.l:
            units.append(_pad_unit())
        # operands as arrays (the loader's marshaling of Python lists is not prover work; everything after t0 is)
        a = np.asarray([u[1] for u in units], dtype=np.int64)
        b = np.asarray([u[2] for u in units], dtype=np.int64)
        t0 = time.perf_counter()
        pub = public_vectors(self.p, None, a, b, None, device=self.device)
        c_in, y_out = np.asarray([u[0] for u in units], dtype=np.int64), np.asarray([u[3] for u in units], dtype=np.int64)
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
                      chain=(lay, y16), words={"a": a, "b": b}, coins=coins)
        proof.timings["hints_host"] = hints_seconds
        return proof, (a, b, y16), lay

    def verify_vus(self, proof, pubs, lay: Layout, coins=None):
        """Interactive mode with ``coins=None``: the verifier's own coins are the ones the runner sampled in
        ``prove_vus`` (``proof.coins``, i.e. the honest step 0 the runner played); pass ``coins`` explicitly to check
        a transcript against a specific verifier's session."""
        a, b, y16 = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, y16),
                      coins=coins)


def _vu_data(vu: instances.VuTier, i: int):
    x, w, y16 = vu.instance(i)
    return (x, w, vu.accs(i), int(y16))


def gate_vu(args) -> int:
    R = ChainRunner(args.device, args.target, args.rate_log2, args.hints, zk=args.zk, mode=args.mode)
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
        if getattr(args, "dump_proof", None) or getattr(args, "dump_statement", None):
            # ligero-proof/v1 + ligero-statement/v1 (backends/ligero-verify reads them; serialize.py documents them)
            from .serialize import _statement_of, proof_bytes, statement_bytes
            st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
            suffix = f"_{lo:05d}" if args.vus > per_proof else ""
            if args.dump_proof:
                Path(args.dump_proof).with_name(Path(args.dump_proof).stem + suffix + Path(args.dump_proof).suffix).write_bytes(proof_bytes(proof))
            if args.dump_statement:
                Path(args.dump_statement).with_name(Path(args.dump_statement).stem + suffix + Path(args.dump_statement).suffix).write_bytes(statement_bytes(st))
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
    """``--batch`` units per proof (``B_proof = batch // 96`` VUs); ``--total-vus`` (default ``B_proof``) VUs proved as
    ``N_subbatches = ceil(total / B_proof)`` independently committed proofs whose soundness is union-bounded
    (``config_for(..., n_proofs)`` raises ``t`` so the union meets the target).  Every sub-batch is proved AND
    verified in every rep; the reported prover time is the sum over the sub-batches (per_vu = total / total_vus)."""
    from .run import _git_commit

    root = Path(args.root)
    vu = instances.load("vu-k1536", root)
    per_proof = args.batch // STEPS
    total_vus = args.total_vus or per_proof
    n_proofs = -(-total_vus // per_proof)
    interactive = args.mode == protocol_mod.MODE_INTERACTIVE
    depth = 3 if interactive else 1                 # interactive: c, r1, r2 on the critical path; Fiat-Shamir: one message
    R = ChainRunner(args.device, args.target, args.rate_log2, args.hints, n_proofs=n_proofs, zk=args.zk, mode=args.mode)
    subs = [(lo, min(lo + per_proof, total_vus)) for lo in range(0, total_vus, per_proof)]
    data = {i: _vu_data(vu, i % vu.n) for i in range(total_vus)}
    lay = R.layout(per_proof)
    l = lay.l
    cfg = R.cfg(l)
    print(f"config {cfg}; layout {lay}; zk {cfg.zk}, mode {cfg.mode}; system {R.sys.summary()}; {total_vus} VUs as {n_proofs} sub-batches of <= {per_proof}")
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    keys = None
    # --dump-dir: every RECORDED rep's sub-batch proofs as ligero-statement/v2 + ligero-proof/v1 files (+ the runner's
    # step-0 coins, 128 raw bytes, interactive mode) for the independent verifier (backends/ligero-verify); written after
    # the verifier timer, outside every measured interval.  system.bin + manifest.json once.
    dump_dir = Path(args.dump_dir) if getattr(args, "dump_dir", None) else None
    dump_manifest = None
    if dump_dir is not None:
        from .serialize import _manifest, _statement_of, _write_system, proof_bytes, sha256_file, statement_bytes
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_manifest = _manifest(dump_dir, args, R)
        _write_system(dump_dir, R, dump_manifest)
        dump_manifest["set"] = {"total_vus": total_vus, "batch": args.batch, "per_proof": per_proof, "n_proofs": n_proofs,
                                "zk": args.zk, "mode": args.mode, "reps": args.reps}
    for rep in range(args.reps + 1):
        agg: dict[str, float] = {}
        vsum = 0.0
        t_pass = time.perf_counter()
        for si, (lo, hi) in enumerate(subs):
            vus = [data[i] for i in range(lo, hi)]
            V = protocol_mod.Coins.sample() if interactive else None      # the verifier's step 0 (interactive only)
            proof, pubs, lay_i = R.prove_vus(vus, check=False, min_l=l, coins=V)
            t0 = time.perf_counter()
            ok, why = R.verify_vus(proof, pubs, lay_i, coins=V)
            vsum += time.perf_counter() - t0
            assert ok, f"sub-batch [{lo}, {hi}): {why}"
            for k, v in proof.timings.items():
                agg[k] = agg.get(k, 0.0) + v
            if dump_dir is not None and rep > 0:
                rd = dump_dir / f"rep{rep}"
                rd.mkdir(exist_ok=True)
                stem = f"sub_{si:02d}"
                st = _statement_of(R, pubs, lay_i, R.cfg(lay_i.l), n_proofs)
                (rd / f"{stem}.stmt").write_bytes(statement_bytes(st))
                (rd / f"{stem}.proof").write_bytes(proof_bytes(proof))
                entry = {"rep": rep, "sub": si, "vus": [lo, hi], "stmt": f"rep{rep}/{stem}.stmt", "proof": f"rep{rep}/{stem}.proof",
                         "proof_bytes": (rd / f"{stem}.proof").stat().st_size, "proof_sha256": sha256_file(rd / f"{stem}.proof"),
                         "python_verdict": {"accepted": ok, "reason": why}, "prover_seconds": proof.timings.get("total")}
                if V is not None:
                    (rd / f"{stem}.coins").write_bytes(V.r1 + V.s1 + V.r2 + V.s2)
                    entry["coins"] = f"rep{rep}/{stem}.coins"
                dump_manifest["files"].append(entry)
            if rep == 0:
                break                                        # warm-up: one sub-batch (graph capture, compile)
        wall = time.perf_counter() - t_pass
        if rep == 0:
            print("warm-up (one sub-batch):", {k: round(v, 3) for k, v in agg.items()})
            continue
        keys = agg.keys()
        runs.append((agg, vsum))
        print(f"rep {rep}: prover {agg['total']:.3f}s (+hints {agg['hints_host']:.3f}s) over {n_proofs} proofs; "
              f"verifier {vsum:.3f}s; pass wall {wall:.3f}s ({(agg['total'] + agg['hints_host']) / total_vus * 1e3:.3f} ms/VU)")
    if dump_dir is not None:
        (dump_dir / "manifest.json").write_text(json.dumps(dump_manifest, indent=1, default=str))
        print(f"dumped {len(dump_manifest['files'])} proofs to {dump_dir}", flush=True)
    med = {k: statistics.median(r[0][k] for r in runs) for k in keys}
    vmed = statistics.median(r[1] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = proof.n_bytes()
    nbytes = {k: v * n_proofs for k, v in nbytes.items()}
    snd = soundness(cfg, R.tb.L, R.tb.Q, n_proofs)
    total = med["total"] + med["hints_host"]
    n_vus = total_vus
    per_vu = total / n_vus
    hw = contract.probe_hardware()
    sw = {"backend": {"name": "b-ligero chain (backends/direct/ligero)", "commit": _git_commit(), "hash": proof.hash_name,
                      "challenge_expander": "SHAKE-256 (placeholder)"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    if cfg.zk and interactive and not args.hvzk_label:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", ZK_MODE + ZK_MODE_HM96 + ZK_VACUOUS_NOTE
        omitted = "external authentication; unopened columns hidden by Blake3 of a uniform column (random-oracle hiding, not HM96 salted leaves)"
    elif cfg.zk and interactive:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_VACUOUS_NOTE
        omitted = "external authentication (malicious-verifier ZK holds but the conservative label was requested)"
    elif cfg.zk:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_MODE_FS + ZK_VACUOUS_NOTE
        omitted = "external authentication; malicious-verifier ZK outside the programmable random-oracle model (Fiat-Shamir mode: NIZK in PROM only)"
    else:
        proof_class, zk_mode, omitted = "NON_ZK_PROOF_DIAGNOSTIC", "none", "external authentication, ZK masking"
    fp = contract.fingerprint(
        profile=contract.PROFILE, domain=contract.DOMAIN, K=STEP * STEPS, B=n_vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication="excluded", zk_mode=zk_mode,
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_k": cfg.k, "rs_l": cfg.l,
                  "zk": cfg.zk, "t_pad": cfg.t_pad, "mode": cfg.mode, "bound_model": snd["bound_model"],
                  "hash_assumption": snd["hash_assumption"], "zk_statement": snd["zk_statement"],
                  "fiat_shamir_queries_log2": snd["fiat_shamir_queries_log2"],
                  "opened_columns": cfg.t, "achieved_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  "per_proof_target_log2": snd["per_proof_target_log2"], "per_proof_total_log2": snd["per_proof_total_log2"],
                  "union_over": n_proofs, "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation (a function of Params only, {R.compile_seconds:.2f}s)",
        retention="retained",
        instances=contract.instances_ref("vu-k1536", 0, n_vus, instances.manifest_sha256(root)),
        scope="vu", omitted=omitted,
        rows=R.sys.m, committed_rows=n_rows(cfg, R.sys.m, True), linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        witness_generation=f"{R.hints_mode} hint generator (gen_step semantics) + torch program (derived rows)",
        B_proof=per_proof,
        note=f"{total_vus} VUs as {n_proofs} independently committed proofs of <= {per_proof} VUs each (l = {l} columns, "
             f"{l - per_proof * STEPS} pad columns per proof); prover time = sum over the proofs; soundness = union bound",
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
        contract.measurement("rounds.protocol", depth),         # 3: step-0 coin commitments, r1 opening, r2 opening
        contract.measurement("rounds.sequential_depth", depth), # PROTOCOL.md 8c (2 without the HM96 step)
        contract.measurement("mem.peak_device_bytes", peak_dev),
        contract.measurement("census.witness_elements", R.sys.m * l * n_proofs),
        contract.measurement("census.committed_elements", R.sys.m * cfg.n * n_proofs),
        contract.measurement("census.bytes_encoded", R.sys.m * cfg.n * 4 * n_proofs),
        contract.measurement("vu.seconds_per_vu", per_vu, "s"),
        contract.measurement("vu.count", n_vus, "count"),
        *contract.rate_measurements(total, n_vus, STEP * STEPS, contract.PROFILE),
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
    ]
    result = {"schema": contract.RESULT_SCHEMA, "run_id": args.run_id or f"b-ligero-vu-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": "passed", "checker": "backends.direct.ligero.run bench-vu (verifier accepted every rep)",
                             "detail": f"{args.reps} reps, {n_vus} VUs as {n_proofs} proofs at l={l}",
                             "evidence": {"soundness": snd, "reps": [{"prover_total": r[0]["total"] + r[0]["hints_host"], "verifier": r[1]}
                                                                     for r in runs]}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"seconds_per_vu": per_vu, "prover_total": total, "verifier": vmed, "bytes": nbytes, "n_proofs": n_proofs,
                      "per_proof_vus": per_proof, "t": cfg.t, "zk": cfg.zk, "k": cfg.k, "peak_device_bytes": peak_dev, "soundness_log2": snd["total_log2"],
                      "per_proof_log2": snd["per_proof_total_log2"], "medians": med}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    return 0
