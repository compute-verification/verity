"""The B-Ligero chain prover for any registered relation (``relations.py``): gate, bench, proof dumps.

    python -m backends.direct.ligero.run --relation bf16-hopper gate-vu  --device cuda --vus 64
    python -m backends.direct.ligero.run --relation fp8-hopper  bench-vu --device cuda --batch 4096 --total-vus 4096 --reps 3 \
        --dump-dir $RESEARCH_RUN_DIR/proofs --out $RESEARCH_RUN_DIR/result.json

One VU is ``GemmCoordinate<1536>`` (BF16: cast at the end) or ``GemmAccumulator<1536>`` (FP8: the FP32 word) as
``1536 / k`` chained units from ``c_0 = +0``; the accumulators are witness rows linked by the chain test (``chain.py``),
the operands and the claimed chain-end words enter the statement digest before any challenge (PROTOCOL.md).  Instances
are synthetic and deterministic (``instances``: the relation's finite operand recipe from ``(seed, index)``, the
accumulators recorded by the model ``tc_dot``); ``--instances-cache DIR`` keeps them as ``.npz`` between runs.

``fp8/chain.py`` (the fp8-proof lane's runner for ``--relation fp8-ada``) is this module at ``RELATIONS["fp8-ada"]``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from verity.ml.tc.silicon import tc_dot
from verity_numerical.bench import contract

from . import protocol as protocol_mod
from .chain import Layout
from .compile import P
from .fp8.census import census
from .protocol import Config, config_for, n_rows, prove, soundness, verify
from .relations import K_VU, Relation, relation
from .witness import tables

__all__ = ["RelationChainRunner", "bench_vu_rel", "gate_vu_rel", "instances", "instances_digest", "vu_units"]


# ---------------------------------------------------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------------------------------------------------

def _instance(args) -> tuple[list, list, list, int]:
    name, seed, i = args
    rel = relation(name)
    rng = np.random.default_rng([seed, i])
    a = rel.operands(rng, (K_VU,)).tolist()
    b = rel.operands(rng, (K_VU,)).tolist()
    k, accs, c = rel.k, [], 0
    for s in range(rel.steps):
        c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
        accs.append(c)
    return a, b, accs, accs[-1]


def instances(rel: Relation, n: int, seed: int | None = None, procs: int = 1, cache: str | Path | None = None):
    """``n`` VUs: (a (1536,), b (1536,), accs (steps,), y_final) with the accumulators recorded from the model.
    Deterministic in ``(seed, index)``; ``procs > 1`` records them in parallel; ``cache`` keeps the set as an npz."""
    seed = rel.instance_seed if seed is None else seed
    path = Path(cache) / f"{rel.name}-{seed}-{n}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as z:  # materialise each array once (indexing the lazy NpzFile re-reads the member every time)
            a, b, accs = z["a"], z["b"], z["accs"]
        return [(a[i].tolist(), b[i].tolist(), accs[i].tolist(), int(accs[i][-1])) for i in range(n)]
    jobs = [(rel.name, seed, i) for i in range(n)]
    if procs > 1 and n > 1:
        with ProcessPoolExecutor(procs) as ex:
            out = list(ex.map(_instance, jobs, chunksize=max(1, n // (8 * procs))))
    else:
        out = [_instance(j) for j in jobs]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, a=np.asarray([o[0] for o in out], dtype=np.int64), b=np.asarray([o[1] for o in out], dtype=np.int64),
                 accs=np.asarray([o[2] for o in out], dtype=np.int64))
    return out


def instances_digest(rel: Relation, n: int, seed: int | None = None) -> str:
    """The synthetic set's manifest digest: a function of (relation, seed, n, K) -- the set itself is."""
    seed = rel.instance_seed if seed is None else seed
    return hashlib.sha256(f"{rel.name} synthetic|seed={seed}|n={n}|K={K_VU}".encode()).hexdigest()


def vu_units(rel: Relation, a: list[int], b: list[int], accs: list[int]) -> list[tuple]:
    """The ``steps`` units of one VU with their recorded output words."""
    k, units, c = rel.k, [], 0
    for s in range(rel.steps):
        units.append((c, a[k * s:k * s + k], b[k * s:k * s + k], accs[s]))
        c = accs[s]
    return units


# ---------------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------------

class RelationChainRunner:
    """``vu.ChainRunner`` for a registered relation: the same protocol (``prove`` / ``verify``) on ``rel.compile(p, chain=True)``
    with the relation's hint generator, public decode and statement hooks."""

    def __init__(self, rel: Relation, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE):
        self.rel = rel
        self.p = rel.params
        self.zk, self.mode, self.device, self.target, self.rate_log2, self.n_proofs = zk, mode, device, target, rate_log2, n_proofs
        self.steps = rel.steps
        self.hooks = rel.hooks
        t0 = time.perf_counter()
        self.sys = rel.compile(self.p, True)
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

    @property
    def relation(self) -> str:
        """The registered relation's name: ``serialize._relation_of`` reads it, so every registered relation's statement
        file is the relation-named ``ligero-statement/v4`` (the Ampere BF16 relation, ``vu.ChainRunner``, has no such
        property and keeps v2 / v3 for the Rust verifier)."""
        return self.rel.name

    def layout(self, n_vus: int, min_l: int = 256) -> Layout:
        l = max(min_l, 1 << (n_vus * self.steps - 1).bit_length())
        return Layout(l, n_vus, self.steps)

    def _pad_unit(self):
        return (0, [0] * self.rel.k, [0] * self.rel.k, 0)

    def prove_vus(self, vus: list[tuple[list, list, list, int]], check=True, mutate=None, min_l: int = 256, coins=None):
        """``vus``: (a, b, accs, y_final) per VU.  The public claim per VU is ``rel.y_public(y_final)`` at its last column.
        Interactive mode with ``coins=None``: the runner plays the honest verifier's step 0 (``Coins.sample()``)."""
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        lay = self.layout(len(vus), min_l)
        units, ypub = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (a, b, accs, y) in enumerate(vus):
            units += vu_units(self.rel, a, b, accs)
            ypub[self.steps * v + self.steps - 1] = self.rel.y_public(int(y))
        while len(units) < lay.l:
            units.append(self._pad_unit())
        a = np.asarray([u[1] for u in units], dtype=np.int64)
        b = np.asarray([u[2] for u in units], dtype=np.int64)
        c_in = np.asarray([u[0] for u in units], dtype=np.int64)
        y_out = np.asarray([u[3] for u in units], dtype=np.int64)
        t0 = time.perf_counter()
        pub = self.rel.public_vectors(self.p, None, a, b, None, device=self.device)
        hints = self.rel.hints(self.p, self.sys, c_in, a, b, y_out, self.device)
        if check and self.rel.reference_hints is not None:      # the gate: the torch generator must equal the checker's gen_step
            ref = self.rel.reference_hints(self.p, self.sys, units, self.device)
            assert torch.equal(ref, hints), f"{self.rel.name}: torch hints != gen_step"
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        proof = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                      chain=(lay, ypub), words={"a": a, "b": b}, coins=coins, rel=self.hooks)
        proof.timings["hints_host"] = hints_seconds
        return proof, (a, b, ypub), lay

    def verify_vus(self, proof, pubs, lay: Layout, coins=None):
        a, b, ypub = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, ypub),
                      coins=coins, rel=self.hooks)


# ---------------------------------------------------------------------------------------------------------------------
# Gate: honest VUs accepted, a negative battery rejected (on the device)
# ---------------------------------------------------------------------------------------------------------------------

def negatives(R: RelationChainRunner, base: list, row_negatives: int = 60) -> list[tuple[str, tuple[bool, str]]]:
    """The negative battery on two honest VUs: wrong claims, off-domain operands, another statement, broken chain links,
    a nonzero chain start, an accumulator bit, and one witness row per (gadget scope, row class)."""
    rel, steps = R.rel, R.steps
    negs = []
    proof, (a, b, ypub), lay = R.prove_vus(base, check=False)
    # (1) wrong claimed words: every bit of the public word of VU 0
    for bit in range(rel.y_bits):
        y_bad = ypub.clone()
        y_bad[steps - 1] ^= 1 << bit
        negs.append((f"claimed word bit {bit} flipped", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (2) off by one unit of the public word (the neighbouring representable value)
    for delta in (1, -1):
        y_bad = ypub.clone()
        y_bad[steps - 1] = (int(ypub[steps - 1]) + delta) % 2**rel.y_bits
        negs.append((f"claimed word {'+' if delta > 0 else '-'}1 ulp", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (3) an operand off the finite domain smuggled into the statement
    for j, w in enumerate(rel.off_domain):
        a_bad = a.copy()
        a_bad[3 + j, 5] = w
        negs.append((f"off-domain operand 0x{w:x} in a", R.verify_vus(proof, (a_bad, b, ypub), lay)))
    b_bad = b.copy()
    b_bad[7, 0] = rel.off_domain[-1]
    negs.append((f"off-domain operand 0x{rel.off_domain[-1]:x} in b", R.verify_vus(proof, (a, b_bad, ypub), lay)))
    # (4) an operand changed (the proof is of another statement)
    a_bad = a.copy()
    a_bad[10, 1] ^= 1 << (rel.params.frac if rel.params.word_bits == 16 else 3)   # an exponent bit
    negs.append(("operand a[10][1] exponent bit flipped", R.verify_vus(proof, (a_bad, b, ypub), lay)))
    # (5) chain: broken link, nonzero c_0, accumulator bits of a linked word
    c_rows = R.sys.chain["c"]

    def break_link(W):
        W[c_rows[2], 20] = (W[c_rows[2], 20] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=break_link)
    negs.append(("link c.f[20] != y.f[19]", R.verify_vus(pf, pb, ly)))

    def nonzero_c0(W):
        W[c_rows[1], steps] = 1
    pf, pb, ly = R.prove_vus(base, check=False, mutate=nonzero_c0)
    negs.append(("c_0 != 0 at a chain start", R.verify_vus(pf, pb, ly)))

    def acc_bit(W):
        W[c_rows[2], 30] = (W[c_rows[2], 30] ^ (1 << 12)) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=acc_bit)
    negs.append(("accumulator fraction bit 12 flipped in a linked unit", R.verify_vus(pf, pb, ly)))

    def acc_low_bit(W):
        W[c_rows[2], 31] = (W[c_rows[2], 31] ^ 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=acc_low_bit)
    negs.append(("accumulator fraction bit 0 flipped in a linked unit", R.verify_vus(pf, pb, ly)))
    # (6) one witness row per (gadget scope, row class), as gate_vu does
    seen, picks = set(), []
    for r in R.sys.rows:
        key = (".".join(r.name.split(".")[:2]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(11)
    picks = picks[: max(0, row_negatives)]
    for r in picks:
        col = int(rng.integers(0, 2 * steps))
        delta = int(rng.integers(1, 3))

        def mutate(W, idx=r.idx, col=col, delta=delta):
            W[idx, col] = (W[idx, col] + delta) % P
        pf, pb, ly = R.prove_vus(base, check=False, mutate=mutate)
        negs.append((f"row {r.name} ({r.kind}) +{delta} in column {col}", R.verify_vus(pf, pb, ly)))
    return negs


def gate_vu_rel(args, rel: Relation) -> int:
    R = RelationChainRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode)
    log = print
    log(f"{rel.name} chain system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; device {args.device}")
    report = {"relation": rel.name, "model": rel.model_name, "system": R.sys.summary(), "census": census(R.sys),
              "positives": [], "negatives": []}
    failures = 0
    per_proof = args.batch // R.steps
    data = instances(rel, args.vus, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None))
    for lo in range(0, args.vus, per_proof):
        vus = data[lo:lo + per_proof]
        proof, pubs, lay = R.prove_vus(vus, check=True)
        ok, why = R.verify_vus(proof, pubs, lay)
        log(f"{rel.name} vu-k1536 [{lo}, {lo + len(vus)}): l={lay.l} prover {proof.timings['total']:.2f}s -> {ok} {why}")
        report["positives"].append({"range": [lo, lo + len(vus)], "l": lay.l, "ok": ok, "why": why, "timings": proof.timings})
        failures += not ok
    negs = negatives(R, data[:2], getattr(args, "row_negatives", 60))
    for name, (ok, why) in negs:
        log(f"  negative {name}: {'REJECTED' if not ok else 'ACCEPTED (BUG)'} {why}")
        report["negatives"].append({"name": name, "accepted": ok, "why": why})
        failures += ok
    report["failures"] = failures
    log(f"{rel.name} gate: {len(report['positives'])} honest sub-batches, {len(negs)} negatives, {failures} failures")
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


# ---------------------------------------------------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------------------------------------------------

def bench_vu_rel(args, rel: Relation) -> int:
    """``vu.bench_vu`` for a registered relation: ``--batch`` units per proof, ``--total-vus`` VUs as ceil(total / B_proof)
    independently committed proofs (union-bounded soundness), every sub-batch of a rep proved first and then every one
    verified (the verifier's host work never sits between two timed prover sub-batches), the reported prover time the sum
    over the sub-batches.  ``--dump-dir`` writes the first ``--dump-reps`` recorded reps' statements (``ligero-statement/v4``:
    the relation is named in the file) + proofs (+ coins in interactive mode) and ``system.bin`` + ``manifest.json``, then
    re-reads and verifies every dumped file cold (``_verify_dumps``, the Python file verifier: a self-check); a dump inside
    the run directory is listed in ``artifacts`` so it travels with the Attempt as run-files (``vu.dump_artifacts``)."""
    from .run import _git_commit as _git_commit_repo
    from .vu import ZK_MODE, ZK_MODE_FS, ZK_MODE_HM96, ZK_VACUOUS_NOTE, dump_artifacts

    def _git_commit():
        return os.environ.get("RESEARCH_GIT_COMMIT") or _git_commit_repo()

    steps = rel.steps
    per_proof = args.batch // steps
    total_vus = args.total_vus or per_proof
    n_proofs = -(-total_vus // per_proof)
    interactive = args.mode == protocol_mod.MODE_INTERACTIVE
    depth = 3 if interactive else 1
    R = RelationChainRunner(rel, args.device, args.target, args.rate_log2, n_proofs=n_proofs, zk=args.zk, mode=args.mode)
    cen = census(R.sys)
    print(f"{rel.name} census:", json.dumps(cen))
    subs = [(lo, min(lo + per_proof, total_vus)) for lo in range(0, total_vus, per_proof)]
    t0 = time.perf_counter()
    data = instances(rel, total_vus, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None))
    print(f"{total_vus} synthetic VUs (model-recorded accumulators) in {time.perf_counter() - t0:.1f}s")
    lay = R.layout(per_proof)
    l = lay.l
    cfg = R.cfg(l)
    print(f"config {cfg}; layout {lay}; zk {cfg.zk}, mode {cfg.mode}; system {R.sys.summary()}; {total_vus} VUs as {n_proofs} sub-batches of <= {per_proof}")
    # --dump-dir: the first --dump-reps RECORDED reps' sub-batch proofs as ligero-statement/v4 + ligero-proof/v1 files (+ the
    # runner's step-0 coins, 128 raw bytes, interactive mode), system.bin + manifest.json once -- the layout vu.bench_vu
    # writes for BF16 (repK/sub_XX.{stmt,proof[,coins]}); written after the verifier timer, outside every measured interval.
    dump_dir = Path(os.path.expandvars(args.dump_dir)) if getattr(args, "dump_dir", None) else None
    dump_reps = getattr(args, "dump_reps", None)
    dump_reps = args.reps if dump_reps is None else max(0, min(int(dump_reps), args.reps))
    dump_manifest = None
    if dump_dir is not None:
        from .serialize import _manifest, _statement_of, _write_system, proof_bytes, sha256_file, statement_bytes
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_manifest = _manifest(dump_dir, args, R)
        dump_manifest["relation"] = {"name": rel.name, "model": rel.model_name, "profile": rel.target.name,
                                     "word_dtype": rel.word_dtype, "y_dtype": rel.y_dtype, "y_bits": rel.y_bits,
                                     "rust_verifiable": rel.rust_verifiable}
        _write_system(dump_dir, R, dump_manifest)
        dump_manifest["set"] = {"total_vus": total_vus, "batch": args.batch, "per_proof": per_proof, "n_proofs": n_proofs,
                                "zk": args.zk, "mode": args.mode, "reps": args.reps, "dump_reps": dump_reps,
                                "instances": instances_digest(rel, total_vus)}
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    keys = None
    for rep in range(args.reps + 1):
        agg: dict[str, float] = {}
        vsum = 0.0
        t_pass = time.perf_counter()
        # Prove every sub-batch of the rep first, then verify (and dump) them: the verifier's host work must not sit
        # between two timed prover sub-batches (vu.bench_vu since f28c13d; the same discipline here).
        proved = []
        for si, (lo, hi) in enumerate(subs):
            V = protocol_mod.Coins.sample() if interactive else None
            proof, pubs, lay_i = R.prove_vus(data[lo:hi], check=(rep == 0 and lo == 0), min_l=l, coins=V)
            for k_, v_ in proof.timings.items():
                agg[k_] = agg.get(k_, 0.0) + v_
            proved.append((si, lo, hi, V, proof, pubs, lay_i))
            if rep == 0:
                break                                        # warm-up: one sub-batch (compile, graph capture, self-check)
        for si, lo, hi, V, proof, pubs, lay_i in proved:
            t1 = time.perf_counter()
            ok, why = R.verify_vus(proof, pubs, lay_i, coins=V)
            vsum += time.perf_counter() - t1
            assert ok, f"sub-batch [{lo}, {hi}): {why}"
            if dump_dir is not None and 0 < rep <= dump_reps:
                rd = dump_dir / f"rep{rep}"
                rd.mkdir(exist_ok=True)
                stem = f"sub_{si:02d}"
                st = _statement_of(R, pubs, lay_i, R.cfg(lay_i.l), n_proofs)          # ligero-statement/v4: R.relation names it
                (rd / f"{stem}.stmt").write_bytes(statement_bytes(st))
                (rd / f"{stem}.proof").write_bytes(proof_bytes(proof))
                entry = {"rep": rep, "sub": si, "vus": [lo, hi], "stmt": f"rep{rep}/{stem}.stmt", "proof": f"rep{rep}/{stem}.proof",
                         "proof_bytes": (rd / f"{stem}.proof").stat().st_size, "proof_sha256": sha256_file(rd / f"{stem}.proof"),
                         "python_verdict": {"accepted": ok, "reason": why}, "prover_seconds": proof.timings.get("total")}
                if V is not None:
                    (rd / f"{stem}.coins").write_bytes(V.r1 + V.s1 + V.r2 + V.s2)
                    entry["coins"] = f"rep{rep}/{stem}.coins"
                dump_manifest["files"].append(entry)
        del proved
        wall = time.perf_counter() - t_pass
        if rep == 0:
            print("warm-up (one sub-batch, with the prover's self-check):", {k_: round(v_, 3) for k_, v_ in agg.items()})
            continue
        keys = agg.keys()
        runs.append((agg, vsum))
        print(f"rep {rep}: prover {agg['total']:.3f}s (+hints {agg['hints_host']:.3f}s) over {n_proofs} proofs; "
              f"verifier {vsum:.3f}s; pass wall {wall:.3f}s ({(agg['total'] + agg['hints_host']) / total_vus * 1e3:.3f} ms/VU)")
    dumps = None
    if dump_dir is not None:
        (dump_dir / "manifest.json").write_text(json.dumps(dump_manifest, indent=1, default=str))
        print(f"dumped {len(dump_manifest['files'])} proofs to {dump_dir}", flush=True)
        dumps = _verify_dumps(dump_dir, dump_manifest, R, args.device)
    med = {k_: statistics.median(r[0][k_] for r in runs) for k_ in keys}
    vmed = statistics.median(r[1] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = {k_: v_ * n_proofs for k_, v_ in proof.n_bytes().items()}
    snd = soundness(cfg, R.tb.L, R.tb.Q, n_proofs)
    total = med["total"] + med["hints_host"]
    per_vu = total / total_vus
    hw = contract.probe_hardware()
    sw = {"backend": {"name": rel.backend_name, "commit": _git_commit(), "hash": proof.hash_name,
                      "challenge_expander": "SHAKE-256 (placeholder)"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    if cfg.zk and interactive:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", ZK_MODE + ZK_MODE_HM96 + ZK_VACUOUS_NOTE
        omitted = "external authentication; unopened columns hidden by Blake3 of a uniform column"
    elif cfg.zk:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_MODE_FS + ZK_VACUOUS_NOTE
        omitted = "external authentication; malicious-verifier ZK outside the programmable random-oracle model"
    else:
        proof_class, zk_mode, omitted = "NON_ZK_PROOF_DIAGNOSTIC", "none", "external authentication, ZK masking"
    inst = {"dataset": rel.instances_dataset, "tier": rel.instances_tier, "range": [0, total_vus],
            "manifest_sha256": instances_digest(rel, total_vus), "seed": rel.instance_seed}
    if rel.instances_recipe:
        inst["recipe"] = rel.instances_recipe
    fp = contract.fingerprint(
        profile=rel.target.name, domain=contract.DOMAIN, K=K_VU, B=total_vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication="excluded", zk_mode=zk_mode,
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_k": cfg.k, "rs_l": cfg.l,
                  "zk": cfg.zk, "t_pad": cfg.t_pad, "mode": cfg.mode, "bound_model": snd["bound_model"],
                  "hash_assumption": snd["hash_assumption"], "zk_statement": snd["zk_statement"],
                  "fiat_shamir_queries_log2": snd["fiat_shamir_queries_log2"],
                  "opened_columns": cfg.t, "achieved_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  "per_proof_target_log2": snd["per_proof_target_log2"], "per_proof_total_log2": snd["per_proof_total_log2"],
                  "union_over": n_proofs, "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation (a function of the relation's Params only, {R.compile_seconds:.2f}s)",
        retention="retained",
        instances=inst,
        scope="vu", omitted=omitted,
        relation=rel.relation_name, model=rel.model_name, commit=_git_commit(), step_k=rel.k, steps_per_vu=steps,
        rows=R.sys.m, committed_rows=n_rows(cfg, R.sys.m, True), linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        census=cen, witness_generation=rel.witness_generation, B_proof=per_proof,
        statement_format="ligero-statement/v4",              # what --dump-dir writes: the relation-named statement (serialize.py)
        note=f"{total_vus} VUs as {n_proofs} independently committed proofs of <= {per_proof} VUs each (l = {l} columns, "
             f"{l - per_proof * steps} pad columns per proof); prover time = sum over the proofs; soundness = union bound; "
             f"{rel.claim_note}",
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
        *contract.rate_measurements(total, total_vus, K_VU, rel.target.name),
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
        contract.measurement("units_per_vu", steps, "count"),
    ]
    for k_, v_ in cen.items():
        if isinstance(v_, int) and not isinstance(v_, bool):
            # ``census.`` is a contract-reserved prefix: only the seven names contract.py defines may live under
            # it (emitted above).  The relation's own per-gadget row counts go under ``relation.census.``.
            meas.append(contract.measurement(f"relation.census.{k_}", v_, "count"))
    checker = (f"backends.direct.ligero.relchain bench_vu_rel[{rel.name}] (verifier accepted every rep; prover self-check on the "
               "warm-up sub-batch")
    status = "passed"
    evidence = {"soundness": snd, "reps": [{"prover_total": r[0]["total"] + r[0]["hints_host"], "verifier": r[1]} for r in runs],
                "verify_wall": "the live in-process Python verifier, median over the reps of the sum over the sub-batches (shares code "
                               "with the prover); not a file verifier's process wall"}
    detail = f"{args.reps} reps, {total_vus} VUs as {n_proofs} proofs at l={l}"
    if dumps is not None:
        # the same code as the prover (the Python file verifier, cold from the dumped bytes): a self-check of the dump
        # round trip, NOT an independent verification -- that is the `verify` lane's label, never this producer's
        checker += "; Python file verifier (serialize.verify_files, shares code with the prover) on every dumped proof, cold from bytes"
        evidence["dumps"] = dumps
        detail += (f"; dumps: {dumps['accepted']}/{dumps['total']} accepted cold from bytes ({dumps['statement_format']}) by the Python file "
                   f"verifier in {dumps['wall']:.2f}s wall on {args.device}, in-process (a self-check: it shares code with the prover; "
                   "an independent verifier has to judge the same bytes)")
        if dumps["accepted"] != dumps["total"]:
            status = "failed"
    checker += ")"
    result = {"schema": contract.RESULT_SCHEMA,
              "run_id": args.run_id or os.environ.get("RESEARCH_RUN_ID") or f"b-ligero-{rel.name}-vu-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": status, "checker": checker, "detail": detail, "evidence": evidence},
              # the dump tree (relative to the run dir) travels with the result as its run-files/v1 artifact (store README §10)
              "measurements": meas, "measurement_files": [], "artifacts": dump_artifacts(dump_dir, args.out)}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"relation": rel.name, "seconds_per_vu": per_vu, "prover_total": total, "verifier": vmed, "bytes": nbytes,
                      "n_proofs": n_proofs, "per_proof_vus": per_proof, "t": cfg.t, "zk": cfg.zk, "k": cfg.k,
                      "peak_device_bytes": peak_dev, "soundness_log2": snd["total_log2"], "per_proof_log2": snd["per_proof_total_log2"],
                      "rows": R.sys.m, "medians": med, "dumps": dumps, "validation": status}, indent=1))
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(result, indent=1, default=str))   # $RESEARCH_RUN_DIR/result.json
    return 0 if status == "passed" else 1


def _verify_dumps(dump_dir: Path, manifest: dict, R: RelationChainRunner, device: str) -> dict:
    """``serialize.verify_files`` on every dumped (statement, proof[, coins]) of ``manifest`` with the prover's own runner:
    the files parse and verify cold from bytes (interactive mode: against the coins in the transcript -- a self-check, D7).
    The verdicts go to ``<dump_dir>/verify_python.json``; returns the summary that enters ``validation.evidence.dumps``."""
    from .serialize import verify_files
    items, t0 = [], time.perf_counter()
    for e in manifest["files"]:
        ok, why, secs = verify_files(dump_dir / e["stmt"], dump_dir / e["proof"], device, runner=R)
        items.append({"rep": e["rep"], "sub": e["sub"], "proof": e["proof"], "accepted": ok, "reason": why, "seconds": secs})
        if not ok:
            print(f"!! dumped proof {e['proof']} REJECTED cold from bytes: {why}", flush=True)
    n_ok = sum(1 for it in items if it["accepted"])
    summary = {"verifier": "python (backends.direct.ligero.serialize.verify_files; shares code with the prover: a self-check)",
               "independent": False, "statement_format": manifest["format"]["statement"], "accepted": n_ok, "total": len(items),
               "seconds": sum(it["seconds"] for it in items), "wall": time.perf_counter() - t0,
               "coins": "the transcript's own coins replayed (file verifier)" if R.interactive else "n/a (Fiat-Shamir)"}
    (dump_dir / "verify_python.json").write_text(json.dumps({**summary, "items": items}, indent=1))
    print(f"dumps: {n_ok}/{len(items)} accepted cold from bytes by the Python file verifier in {summary['wall']:.2f}s", flush=True)
    return summary
