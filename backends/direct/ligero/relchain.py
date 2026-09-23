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
        self._marshal_cache: dict = {}

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

    def marshal(self, vus: list[tuple[list, list, list, int]], min_l: int = 256) -> dict:
        """The sub-batch's public words as arrays, once per sub-batch for the whole run (cached on the identity of the
        VU tuples): host ``a, b, c_in, y_out`` (int64 numpy: the statement, the dumps, the reference hints) and their
        device-resident copies + the verifier's public decode ``pub`` (``rel.public_vectors``, device) -- so a timed
        sub-batch does no list -> numpy -> device marshalling and no host round trip (lane hp2-host, "instances
        device-resident")."""
        key = (len(vus), id(vus[0]), id(vus[-1]), id(vus[len(vus) // 2]), min_l)
        hit = self._marshal_cache.get(key)
        if hit is not None:
            return hit
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
        dev = self.device
        to_dev = lambda x: torch.as_tensor(x).to(dev)  # noqa: E731
        bundle = {"lay": lay, "units": units, "a": a, "b": b, "c_in": c_in, "y_out": y_out, "ypub": ypub,
                  "a_d": to_dev(a), "b_d": to_dev(b), "c_d": to_dev(c_in), "y_d": to_dev(y_out), "ypub_d": to_dev(ypub)}
        bundle["pub"] = self.rel.public_vectors(self.p, None, bundle["a_d"], bundle["b_d"], None, device=dev)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize(dev)
        self._marshal_cache[key] = bundle
        return bundle

    def prove_vus(self, vus: list[tuple[list, list, list, int]], check=True, mutate=None, min_l: int = 256, coins=None):
        """``vus``: (a, b, accs, y_final) per VU.  The public claim per VU is ``rel.y_public(y_final)`` at its last column.
        Interactive mode with ``coins=None``: the runner plays the honest verifier's step 0 (``Coins.sample()``)."""
        from .pipeline import run_to_completion
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        bd = self.marshal(vus, min_l)
        if hasattr(coins, "set_statement"):
            # a live verifier's coin source (live.LiveCoins): the statement travels with the root, so the claim is fixed
            # before the verifier opens its first coin.  Built outside prove's clock; the protocol is unchanged.
            from .live import set_statement
            set_statement(coins, self, (bd["a"], bd["b"], bd["ypub"]), bd["lay"], self.cfg(bd["lay"].l), self.n_proofs)
        gen = self._prove_stages(bd, coins, check=check, mutate=mutate, slot=None)
        proof = run_to_completion(gen)
        return proof, (bd["a"], bd["b"], bd["ypub"]), bd["lay"]

    def _prove_stages(self, bd: dict, coins, check=False, mutate=None, slot=None):
        """The prover generator for one marshalled sub-batch (``pipeline.prove_many`` drives several at once): the hints
        (device, timed ``hints_host`` inside the prover's clock) then ``protocol.prove_stages``."""
        lay = bd["lay"]

        def hints_fn():
            hints = self.rel.hints(self.p, self.sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], self.device)
            if check and self.rel.reference_hints is not None:  # the gate: the torch generator must equal the checker's gen_step
                ref = self.rel.reference_hints(self.p, self.sys, bd["units"], self.device)
                assert torch.equal(ref, hints), f"{self.rel.name}: torch hints != gen_step"
            return hints

        return protocol_mod.prove_stages(self.p, self.sys, self.tb, self.cfg(lay.l), bd["pub"], None, self.device, check=check,
                                         mutate=mutate, chain=(lay, bd["ypub_d"]), words={"a": bd["a"], "b": bd["b"], "y16": bd["ypub"]},
                                         coins=coins, rel=self.hooks, hints_fn=hints_fn, slot=slot)

    def prove_vus_many(self, batches: list[list[tuple]], coins_factory, depth: int = 2, min_l: int = 256):
        """``len(batches)`` sub-batches proved with up to ``depth`` in flight on their own CUDA streams (``pipeline.prove_many``):
        the host work of one sub-batch (challenge expansion, transcript copies, launch overhead) hides under the device work
        of the next.  ``coins_factory(i)`` is called per sub-batch when it starts, in order (the honest verifier's step 0);
        with a live verifier (``coins`` exposing ``set_statement``) the statement is fixed per sub-batch before proving, and
        the coin exchange is non-blocking (the stage yields a Future the pipeline driver polls -- ``live.LiveCoins``).
        Returns ``[(coins, proof, pubs, lay)]`` in sub-batch order and the pass wall; each proof's ``timings`` is the
        sub-batch's share of the wall (see ``pipeline._rescale``)."""
        from .pipeline import prove_many
        bundles = [self.marshal(vus, min_l) for vus in batches]
        coins_of: list = [None] * len(batches)

        def starter(i):
            def start(slot):
                c = coins_factory(i)
                coins_of[i] = c
                if hasattr(c, "set_statement"):
                    # live verifier: the claim (a, b, ypub) is fixed before its first coin is opened; built outside prove's clock
                    from .live import set_statement
                    bd = bundles[i]
                    set_statement(c, self, (bd["a"], bd["b"], bd["ypub"]), bd["lay"], self.cfg(bd["lay"].l), self.n_proofs)
                return self._prove_stages(bundles[i], c, check=False, mutate=None, slot=slot)
            return start

        proofs, wall = prove_many([starter(i) for i in range(len(batches))], self.device, depth=depth)
        return [(coins_of[i], pf, (bd["a"], bd["b"], bd["ypub"]), bd["lay"]) for i, (pf, bd) in enumerate(zip(proofs, bundles))], wall

    def verify_vus(self, proof, pubs, lay: Layout, coins=None):
        a, b, ypub = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, ypub),
                      coins=coins, rel=self.hooks)

    def batch_bound(self, cfgs: list[Config], n_proofs_claimed: list[int], target_bits: float = 128.0,
                    names: list[str] | None = None) -> dict:
        """The batch accounting gate (PROTOCOL.md 8c, "Batch verification"; lane b-batch-bound, D10) for this relation's
        sub-batch proofs that each verified: the verifier's OWN bound of every proof from its configuration (one commitment
        each), union-bounded over the ``len(cfgs)`` presented, against ``target_bits``; every proof's ``n_proofs`` (the count
        its ``t`` was sized for) against that number.  The same rule and reason strings as ``vu.ChainRunner.batch_bound``
        and ``ligero-verify batch``: the union bound is enforced for every relation, not only Ampere BF16."""
        per = [soundness(c, self.tb.L, self.tb.Q, 1)["per_proof_total_log2"] for c in cfgs]
        return protocol_mod.batch_bound(per, n_proofs_claimed, target_bits, names)

    def verify_batch(self, items: list[tuple], target_bits: float = 128.0, n_proofs_claimed: list[int] | None = None) -> dict:
        """The Python batch verifier on in-memory proofs of this relation: ``items`` = ``(proof, pubs, lay[, coins])`` per
        sub-batch presented together, each verified as :meth:`verify_vus` does (own statement, own coins); a rejected
        sub-batch rejects the batch with ``"<sub_XX>: <reason>"``; then :meth:`batch_bound` with ``n_proofs_claimed``
        (default: this runner's ``n_proofs`` for every proof)."""
        names = [f"sub_{i:02d}" for i in range(len(items))]
        cfgs, verdicts = [], []
        for name, it in zip(names, items):
            proof, pubs, lay = it[0], it[1], it[2]
            coins = it[3] if len(it) > 3 else None
            ok, why = self.verify_vus(proof, pubs, lay, coins=coins)
            verdicts.append({"name": name, "accepted": ok, "reason": why})
            cfgs.append(self.cfg(lay.l))
        claimed = list(n_proofs_claimed) if n_proofs_claimed is not None else [self.n_proofs] * len(items)
        for v in verdicts:
            if not v["accepted"]:
                per = [soundness(c, self.tb.L, self.tb.Q, 1)["per_proof_total_log2"] for c in cfgs]
                return {"accepted": False, "reason": f"{v['name']}: {v['reason']}", "n_subbatches": len(items),
                        "target_bits": float(target_bits), "batch_log2": protocol_mod.log2_sum(per), "per_proof_log2": per,
                        "n_proofs_claimed": claimed, "items": verdicts}
        return {**self.batch_bound(cfgs, claimed, target_bits, names), "items": verdicts}


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
    pipeline_depth = int(getattr(args, "pipeline", 0) or 0)
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
    target_bits = float(getattr(args, "target_bits", 128.0) or 128.0)   # the VERIFIER's parameter (PROTOCOL.md 8c)
    # --verifier tcp://host:port: the step-0 coins of every RECORDED rep come from a LIVE verifier on another machine
    # (live.py) and every prover message is sent to it as it is produced -- the protocol is unchanged (8c), only the coin
    # source is; the warm-up rep keeps local coins (it presents one sub-batch, not n_proofs).
    live = None
    if getattr(args, "verifier", None):
        from .live import BenchLive
        live = BenchLive(args, R, n_proofs, total_vus, rel.name, interactive)
    batch = None
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
        cfgs_seen: list[Config] = []
        if live is not None:
            live.begin_rep(rep)
        if pipeline_depth > 1 and rep > 0 and args.device.startswith("cuda"):
            # --pipeline N: the sub-batches interleaved N-deep on their own streams (lane hp2-host); the timings are each
            # sub-batch's share of the pass wall, so their sum is the wall.  Coins are sampled per sub-batch as it starts.
            # With --verifier the factory hands out the LIVE verifier's coins in sub-batch order and the coin exchange is
            # non-blocking (lane live-pipeline): the network waits of sub-batch i overlap the device work of i +- 1, so
            # net_wait falls from n_proofs x 2 x RTT to ~2 x RTT.  PROOF i is sent on the data channel after the pass.
            local_factory = lambda i: (protocol_mod.Coins.sample() if interactive else None)  # noqa: E731
            factory = live.pipeline_factory(local_factory) if live is not None else local_factory
            many, _wall = R.prove_vus_many([data[lo:hi] for lo, hi in subs], factory, depth=pipeline_depth, min_l=l)
            for si, ((lo, hi), (V, proof, pubs, lay_i)) in enumerate(zip(subs, many)):
                if live is not None:
                    V = live.proved(si, proof, V)            # PROOF si on the data channel (async); coins -> plain Coins
                for k_, v_ in proof.timings.items():
                    agg[k_] = agg.get(k_, 0.0) + v_
                proved.append((si, lo, hi, V, proof, pubs, lay_i))
                cfgs_seen.append(R.cfg(lay_i.l))
        for si, (lo, hi) in enumerate(subs if not proved else []):
            V = protocol_mod.Coins.sample() if interactive else None
            if live is not None:
                V = live.coins(si, V)                        # a recorded rep: the live verifier's step 0 for sub-batch si
            proof, pubs, lay_i = R.prove_vus(data[lo:hi], check=(rep == 0 and lo == 0), min_l=l, coins=V)
            if live is not None:
                V = live.proved(si, proof, V)                # PROOF si on the data channel (async); waits out of the clocks
            for k_, v_ in proof.timings.items():
                agg[k_] = agg.get(k_, 0.0) + v_
            proved.append((si, lo, hi, V, proof, pubs, lay_i))
            cfgs_seen.append(R.cfg(lay_i.l))
            if rep == 0:
                if pipeline_depth > 1 and args.device.startswith("cuda"):
                    # every pipeline slot once (its stream's hint graph capture, pinned buffers): untimed, like the warm-up
                    warm = subs[:pipeline_depth] + (subs[-1:] if len(subs) > pipeline_depth else [])   # + the ragged last layout
                    R.prove_vus_many([data[lo_:hi_] for lo_, hi_ in warm],
                                     (lambda i: protocol_mod.Coins.sample() if interactive else None), depth=pipeline_depth, min_l=l)
                break                                        # warm-up: one sub-batch (compile, graph capture, self-check)
        if live is not None:
            live.end_rep(rep, agg)                           # END; the verifier's session verdict (waits for its Rust verify)
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
                    entry["coins_bytes"] = (rd / f"{stem}.coins").stat().st_size
                    entry["coins_sha256"] = sha256_file(rd / f"{stem}.coins")     # D7: the step-0 coins' custody, like proof_sha256
                dump_manifest["files"].append(entry)
        del proved
        if rep > 0:
            # the batch verdict (PROTOCOL.md 8c, "Batch verification"; D10): every sub-batch verified above; now the union
            # bound of this verifier's own per-proof bounds over the n_proofs presented, and each proof's n_proofs (what
            # config_for sized t for) against that count.  Arithmetic on n_proofs numbers, inside the verifier time.
            t0 = time.perf_counter()
            batch = R.batch_bound(cfgs_seen, [n_proofs] * len(cfgs_seen), target_bits)
            vsum += time.perf_counter() - t0
            assert batch["accepted"], batch["reason"]
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
        dumps = _verify_dumps(dump_dir, dump_manifest, R, args.device, target_bits)
    med = {k_: statistics.median(r[0][k_] for r in runs) for k_ in keys}
    vmed = statistics.median(r[1] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = {k_: v_ * n_proofs for k_, v_ in proof.n_bytes().items()}
    snd = soundness(cfg, R.tb.L, R.tb.Q, n_proofs)
    # security.achieved_log2 is the BATCH bound -log2 sum_i 2^b_i as the verifier's batch_bound computed it (vu.bench_vu since
    # lane b-batch-bound); the prover-side soundness() of the same n_proofs must agree with it to the last bit.
    assert batch is not None and abs(batch["batch_log2"] - snd["total_log2"]) < 1e-6, (batch, snd["total_log2"])
    total = med["total"] + med["hints_host"]
    per_vu = total / total_vus
    hw = contract.probe_hardware()
    from . import witness as witness_mod
    sw = {"backend": {"name": rel.backend_name, "commit": _git_commit(), "hash": proof.hash_name,
                      "challenge_expander": "SHAKE-256 (placeholder)", "impl": witness_mod.IMPL, "pipeline": pipeline_depth},
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
                  "opened_columns": cfg.t, "achieved_log2": batch["batch_log2"], "achieved_bits": -batch["batch_log2"],
                  "verifier_target_bits": target_bits, "terms_log2": snd["terms_log2"],
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
    live_fields = live.fields(contract, total, med) if live is not None else None
    if live_fields is not None:
        meas += live_fields[0]           # t.total_live / net.* / verify.* (lane live-verifier; additive)
    for k_, v_ in cen.items():
        if isinstance(v_, int) and not isinstance(v_, bool):
            # ``census.`` is a contract-reserved prefix: only the seven names contract.py defines may live under
            # it (emitted above).  The relation's own per-gadget row counts go under ``relation.census.``.
            meas.append(contract.measurement(f"relation.census.{k_}", v_, "count"))
    checker = (f"backends.direct.ligero.relchain bench_vu_rel[{rel.name}] (verifier accepted every rep; prover self-check on the "
               "warm-up sub-batch")
    status = "passed"
    evidence = {"soundness": snd, "batch": batch,
                "reps": [{"prover_total": r[0]["total"] + r[0]["hints_host"], "verifier": r[1]} for r in runs],
                "verify_wall": "the live in-process Python verifier, median over the reps of the sum over the sub-batches (shares code "
                               "with the prover), batch verdict included; not a file verifier's process wall"}
    detail = f"{args.reps} reps, {total_vus} VUs as {n_proofs} proofs at l={l}; batch {batch['reason']}"
    if dumps is not None:
        # the same code as the prover (the Python file verifier, cold from the dumped bytes): a self-check of the dump
        # round trip, NOT an independent verification -- that is the `verify` lane's label, never this producer's
        checker += "; Python file verifier (serialize.verify_files, shares code with the prover) on every dumped proof, cold from bytes"
        evidence["dumps"] = dumps
        detail += (f"; dumps: {dumps['accepted']}/{dumps['total']} accepted cold from bytes ({dumps['statement_format']}) by the Python file "
                   f"verifier in {dumps['wall']:.2f}s wall on {args.device}, in-process (a self-check: it shares code with the prover; "
                   "an independent verifier has to judge the same bytes)")
        detail += f"; dumped batch verdict per rep: {[b['reason'] for b in dumps['batch'].values()]}"
        if dumps["accepted"] != dumps["total"] or not dumps["batch_accepted"]:
            status = "failed"
    if live_fields is not None:
        # the live verifier's verdicts are reported here as the PROVER received them; the verifier's own dump + verdict
        # artifact (live.py record) is what carries the independent `verified=` label, never this producer's result
        checker += f"; live verifier at {args.verifier}"
        evidence["live_verifier"] = live_fields[1]
        detail += live_fields[2]
        if not live_fields[3]:
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


def _verify_dumps(dump_dir: Path, manifest: dict, R: RelationChainRunner, device: str, target_bits: float = 128.0) -> dict:
    """``serialize.verify_files`` on every dumped (statement, proof[, coins]) of ``manifest`` with the prover's own runner:
    the files parse and verify cold from bytes (interactive mode: against the coins in the transcript -- a self-check, D7);
    then, per dumped rep, the batch verdict (PROTOCOL.md 8c, D10) on the STATEMENTS' configurations and ``n_proofs`` --
    what ``serialize.verify_batch_dir`` / ``ligero-verify batch`` recompute on the same tree.  The verdicts go to
    ``<dump_dir>/verify_python.json``; returns the summary that enters ``validation.evidence.dumps``."""
    from .serialize import read_statement, verify_files
    items, t0 = [], time.perf_counter()
    per_rep: dict[int, list] = {}
    for e in manifest["files"]:
        ok, why, secs = verify_files(dump_dir / e["stmt"], dump_dir / e["proof"], device, runner=R)
        items.append({"rep": e["rep"], "sub": e["sub"], "proof": e["proof"], "accepted": ok, "reason": why, "seconds": secs})
        if not ok:
            print(f"!! dumped proof {e['proof']} REJECTED cold from bytes: {why}", flush=True)
        per_rep.setdefault(e["rep"], []).append(read_statement((dump_dir / e["stmt"]).read_bytes()))
    batches = {}
    for rep, sts in sorted(per_rep.items()):
        b = R.batch_bound([st.config() for st in sts], [st.n_proofs for st in sts], target_bits, [f"sub_{st_i:02d}" for st_i in range(len(sts))])
        batches[str(rep)] = b
        if not b["accepted"]:
            print(f"!! dumped rep {rep}: batch verdict REJECTED: {b['reason']}", flush=True)
    n_ok = sum(1 for it in items if it["accepted"])
    batch_ok = all(b["accepted"] for b in batches.values())
    summary = {"verifier": "python (backends.direct.ligero.serialize.verify_files; shares code with the prover: a self-check)",
               "independent": False, "statement_format": manifest["format"]["statement"],
               "accepted": n_ok, "total": len(items),
               "batch": batches, "batch_accepted": batch_ok,
               "seconds": sum(it["seconds"] for it in items), "wall": time.perf_counter() - t0,
               "coins": "the transcript's own coins replayed (file verifier)" if R.interactive else "n/a (Fiat-Shamir)"}
    (dump_dir / "verify_python.json").write_text(json.dumps({**summary, "items": items}, indent=1))
    print(f"dumps: {n_ok}/{len(items)} accepted cold from bytes by the Python file verifier in {summary['wall']:.2f}s; "
          f"batch verdict per rep: {[b['reason'] for b in batches.values()]}", flush=True)
    return summary
