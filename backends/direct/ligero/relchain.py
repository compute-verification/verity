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

import dataclasses
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


def _frozen_root(root) -> Path:
    from verity_numerical.bench import instances as frozen
    return Path(root) if root else frozen.DEFAULT_ROOT


def frozen_instances(rel: Relation, n: int, root=None):
    """The first ``n`` VUs of the relation's FROZEN tier (``rel.frozen_tier``: ``bench-instances/v1`` ``vu-k1536`` under
    ``root``, the set ``vu.py`` proves) in the runner's shape ``(a, b, accs, y_final)``: the recorded FP32 accumulators are
    the chain's public words, the final one casts to the tier's expected BF16 word (checked).  Beyond the tier's ids the
    instances are recycled (``i mod n_tier``), as ``vu.bench_vu`` does."""
    from verity_numerical.bench import instances as frozen
    vu = frozen.load(rel.frozen_tier, _frozen_root(root))
    if vu.acc is None:
        raise ValueError(f"{rel.frozen_tier}: the tier carries no accumulators")
    out = []
    for i in range(n):
        j = i % vu.n
        a, b, y16 = vu.instance(j)
        accs = vu.accs(j)
        if rel.fold > 1:
            # a folded relation's columns end on every ``fold``-th recorded accumulator (the intermediate ones are private)
            accs = accs[rel.fold - 1::rel.fold]
        if len(accs) != rel.steps or rel.y_public(int(accs[-1])) != y16:
            raise ValueError(f"{rel.frozen_tier} instance {j}: {len(accs)} accumulators / final word does not cast to the expected y")
        out.append((a, b, accs, int(accs[-1])))
    return out


def instances(rel: Relation, n: int, seed: int | None = None, procs: int = 1, cache: str | Path | None = None, root=None):
    """``n`` VUs: (a (1536,), b (1536,), accs (steps,), y_final) with the accumulators recorded from the model.
    Deterministic in ``(seed, index)``; ``procs > 1`` records them in parallel; ``cache`` keeps the set as an npz.
    A relation with a ``frozen_tier`` reads its frozen set from ``root`` instead (:func:`frozen_instances`)."""
    if rel.frozen_tier:
        return frozen_instances(rel, n, root)
    seed = rel.instance_seed if seed is None else seed
    path = Path(cache) / f"{rel.name}-{seed}-{n}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as z:  # materialise each array once (indexing the lazy NpzFile re-reads the member every time)
            a, b, accs = z["a"], z["b"], z["accs"]
        return [(a[i].tolist(), b[i].tolist(), accs[i].tolist(), int(accs[i][-1])) for i in range(n)]
    if path is not None and rel.fold > 1 and rel.folded_from and (Path(cache) / f"{rel.folded_from}-{seed}-{n}.npz").exists():
        # the fold's VUs ARE the base's (same operands from the same (seed, i) streams, the same chain): every ``fold``-th
        # recorded accumulator is a column's end, the others are the words the fold keeps private
        with np.load(Path(cache) / f"{rel.folded_from}-{seed}-{n}.npz") as z:
            a, b, accs = z["a"], z["b"], z["accs"][:, rel.fold - 1::rel.fold]
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


def instances_digest(rel: Relation, n: int, seed: int | None = None, root=None) -> str:
    """The synthetic set's manifest digest: a function of (relation, seed, n, K) -- the set itself is.  A frozen tier's is
    the sha256 of its ``manifest.json`` (``instances.manifest_sha256``), the value the bare ``vu.py`` results carry."""
    if rel.frozen_tier:
        from verity_numerical.bench import instances as frozen
        return frozen.manifest_sha256(_frozen_root(root))
    seed = rel.instance_seed if seed is None else seed
    return hashlib.sha256(f"{rel.name} synthetic|seed={seed}|n={n}|K={K_VU}".encode()).hexdigest()


def vu_units(rel: Relation, a: list[int], b: list[int], accs: list[int]) -> list[tuple]:
    """The ``steps`` units of one VU with their recorded output words."""
    k, units, c = rel.k, [], 0
    for s in range(rel.steps):
        units.append((c, a[k * s:k * s + k], b[k * s:k * s + k], accs[s]))
        c = accs[s]
    return units


def _tile_accs(args) -> list[int]:
    name, a, b = args
    rel = relation(name)
    k, accs, c = rel.k, [], 0
    for s in range(rel.steps):
        c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
        accs.append(c)
    return accs


def parse_tile(spec: str | None) -> tuple[int, int] | None:
    """``--tile 64x64`` -> (n_x_rows, n_w_cols); None for the worst case (every VU its own rows)."""
    if not spec or spec == "none":
        return None
    nx, nw = spec.lower().split("x")
    return int(nx), int(nw)


def tile_instances(rel: Relation, nx: int, nw: int, seed: int | None = None, procs: int = 1, cache: str | Path | None = None):
    """The committed GEMM-tile workload (``sharing = tile<nx>x<nw>``): ``nx`` distinct activation rows and ``nw`` distinct
    weight columns (the relation's finite operand recipe from ``(seed, "tile", ...)``), every (row, column) pair one VU
    -- VU ``v`` is row ``v // nw`` against column ``v % nw`` -- with the accumulators recorded from the model.  Same FLOPs
    and relation as ``instances(rel, nx * nw)``; the hashed statement commits ``nx + nw`` rows instead of ``2 nx nw``.
    Returns ``(vus, x_rows (nx, K), w_cols (nw, K))``."""
    seed = rel.instance_seed if seed is None else seed
    path = Path(cache) / f"{rel.name}-{seed}-tile{nx}x{nw}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as z:
            x, w, accs = z["x"], z["w"], z["accs"]
    else:
        rng = np.random.default_rng([seed, 0x7115, nx, nw])
        x = np.asarray([rel.operands(rng, (K_VU,)) for _ in range(nx)], dtype=np.int64)
        w = np.asarray([rel.operands(rng, (K_VU,)) for _ in range(nw)], dtype=np.int64)
        jobs = [(rel.name, x[v // nw].tolist(), w[v % nw].tolist()) for v in range(nx * nw)]
        if procs > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(procs) as ex:
                out = list(ex.map(_tile_accs, jobs, chunksize=max(1, len(jobs) // (8 * procs))))
        else:
            out = [_tile_accs(j) for j in jobs]
        accs = np.asarray(out, dtype=np.int64)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, x=x, w=w, accs=accs)
    vus = [(x[v // nw].tolist(), w[v % nw].tolist(), accs[v].tolist(), int(accs[v][-1])) for v in range(nx * nw)]
    return vus, x, w


def tile_indices(nx: int, nw: int, lo: int, hi: int) -> tuple[list[int], list[int], list[int]]:
    """(vu_index, x_index, w_index) of the tile's VUs ``[lo, hi)``."""
    vus = list(range(lo, hi))
    return vus, [v // nw for v in vus], [v % nw for v in vus]


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

    def prove_vus(self, vus: list[tuple[list, list, list, int]], check=True, mutate=None, min_l: int = 256, coins=None, vu_ids=None):
        """``vus``: (a, b, accs, y_final) per VU.  The public claim per VU is ``rel.y_public(y_final)`` at its last column.
        Interactive mode with ``coins=None``: the runner plays the honest verifier's step 0 (``Coins.sample()``).
        ``vu_ids`` (the VUs' indices in the committed set) is what the hashed runner binds; the bare statement ignores it."""
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


class HashedRelationRunner(RelationChainRunner):
    """``authentication = included-hash`` (``hashchain.py``, ``hashauth.py``): the relation's chain system composed with
    private operands and the two Poseidon2 sponges; the statement is the digest table + leaf indices + roots, and the
    verifier checks the published digests' Merkle multiproofs after the Ligero verdict.

    ``pubs`` of :meth:`prove_vus` / :meth:`verify_vus` is ``(table (l, 17), indices (n_vus, 3), ypub (l,), auth, siblings)``:
    the public table (``hash.is_end | hash.d[0..15]`` per column), the ``(vu, x, W)`` leaf triples, the chain-end words,
    the :class:`hashauth.HashAuth` block and the three multiproofs.  ``commit(...)`` sets the trees the prover opens."""

    AUTH = "included-hash"

    def __init__(self, rel: Relation, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE, leaf=None):
        """``leaf``: a ``leaf/`` scheme object (``leaf.registry.leaf_for``) replacing the Poseidon2 sponges -- the relation
        is then named ``<rel>+<leaf tag>`` (statements, runners, the Rust verifier); ``None`` is Poseidon2, unchanged."""
        from . import hashauth, hashchain
        from .protocol import RelationHooks
        self.rel = rel
        self.p = rel.params
        self.zk, self.mode, self.device, self.target, self.rate_log2, self.n_proofs = zk, mode, device, target, rate_log2, n_proofs
        self.steps = rel.steps
        t0 = time.perf_counter()
        self.hr = hashchain.compose(rel, leaf)
        self.leaf = leaf
        self.leafp = hashauth.leaf_params(leaf)
        self.sys = self.hr.sys
        self.tb = tables(self.sys, self.device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg = {}
        self._base_sys = dataclasses.replace(self.sys, hints=self.sys.hints[:self.hr.n_base_hints])
        self.base_tag = b"hash/v2h|" + rel.tag
        self.hooks = RelationHooks(hashchain.public_vectors_for(self.hr), word_dtype="<u4", y16_dtype=rel.y_dtype,
                                   y16_bits=rel.y_bits, tag=self.base_tag)
        self.trees = None
        self.tile = None                    # (nx, nw) of the tile workload, None for the worst case
        self.rows, self.sponges, self.sponge_seconds = None, None, 0.0

    @property
    def relation(self) -> str:
        return self.hr.relation_name

    def hooks_for(self, auth):
        from . import hashauth
        return dataclasses.replace(self.hooks, tag=self.base_tag + hashauth.digest_bytes(auth, self.leafp))

    def commit(self, *, x_rows, w_cols, y_words, bindings, cache=None, tile=None):
        """Build (or load) the three trees of the committed set; ``tile = (nx, nw)`` records the sharing layout."""
        from . import hashauth, hashchain
        x_rows, w_cols = np.asarray(x_rows, dtype=np.int64), np.asarray(w_cols, dtype=np.int64)
        self.trees, self.digests, cached = hashauth.load_or_build_trees(
            cache, x_rows=x_rows, w_cols=w_cols, leaf=self.leafp,
            y_words=np.asarray(y_words, dtype=np.int64), word_bits=self.hr.word_bits, y_bits=self.rel.y_bits, bindings=bindings)
        self.tile = tile
        # the sponges of every committed row, once (the prover's cost of hashing; gathered per sub-batch in prove_vus).
        # A leaf/ scheme's states are linear in the bits and recomputed per sub-batch on the device (hash_hints).
        t0 = time.perf_counter()
        self.rows = {"a": x_rows, "b": w_cols}
        if self.leaf is None:
            self.sponges = {nm: hashchain.row_sponges(self.hr, self.rows[nm], role, self.device) for _, nm, role in hashchain.ROLES}
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.sponge_seconds = time.perf_counter() - t0
        return cached

    def commit_vus(self, data: list, dataset: str = "synthetic", tier: str = "dev", manifest_sha256: str = "", cache=None, tile=None,
                   x_rows=None, w_cols=None):
        """Commit an instance list: the worst case (every VU its own x row and W column: ``x_rows = a``, ``w_cols = b``) or a
        tile (``x_rows`` / ``w_cols`` given, ``tile = (nx, nw)``)."""
        from . import hashauth
        y = [self.rel.y_public(int(v[3])) for v in data]
        if tile is None:
            x_rows = [v[0] for v in data]
            w_cols = [v[1] for v in data]
        ysch, _ = hashauth.word_schema(self.rel.y_bits)
        bindings = {n: hashauth.binding_digest(dataset=dataset, tier=tier, manifest_sha256=manifest_sha256, lo=0, hi=len(data),
                                               K=K_VU, tree=n, schema=(self.leafp.schema if n != "y" else ysch)) for n in ("a", "b", "y")}
        return self.commit(x_rows=x_rows, w_cols=w_cols, y_words=y, bindings=bindings, cache=cache, tile=tile)

    def auth_for(self, vu_ids: list[int]):
        from . import hashauth
        if self.trees is None:
            raise RuntimeError("commit(...) first")
        if self.tile is None:
            xi, wi = list(vu_ids), list(vu_ids)
        else:
            _, xi, wi = tile_indices(*self.tile, min(vu_ids), max(vu_ids) + 1)
            if list(range(min(vu_ids), max(vu_ids) + 1)) != list(vu_ids):
                xi = [v // self.tile[1] for v in vu_ids]
                wi = [v % self.tile[1] for v in vu_ids]
        return hashauth.HashAuth.for_vus(self.trees, vu_ids, xi, wi)

    def _sponges_for(self, auth, a, b, n_vus: int) -> dict | None:
        """The committed rows' precomputed sponges for this sub-batch, per operand whose words ARE the committed rows
        (else None for that operand: the sponge runs on the words the prover computes on -- negatives, overrides)."""
        if self.sponges is None:
            return None
        out = {}
        for nm, idx in (("a", auth.x_index), ("b", auth.w_index)):
            idx = np.asarray(idx, dtype=np.int64)
            words = (a if nm == "a" else b)[:n_vus * self.steps].reshape(n_vus, self.steps * self.hr.k)
            if len(idx) == n_vus and (idx < len(self.rows[nm])).all() and np.array_equal(self.rows[nm][idx], words):
                cap, dg = self.sponges[nm]
                t = torch.as_tensor(idx, device=cap.device)
                out[nm] = (cap[t], dg[t])
        return out or None

    def prove_vus(self, vus, check=True, mutate=None, min_l: int = 256, coins=None, vu_ids=None, auth=None, words_override=None):
        """As the base runner; ``vu_ids`` (default ``range(len(vus))``) are the committed set's VU indices.  ``auth`` overrides
        the block (negatives); ``words_override(a, b)`` edits the operand words the prover computes on AFTER the units
        are formed (negatives: "computed on another row")."""
        from . import hashauth, hashchain
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        vu_ids = list(range(len(vus))) if vu_ids is None else list(vu_ids)
        auth = self.auth_for(vu_ids) if auth is None else auth
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
        if words_override is not None:
            a, b = words_override(a, b)
        t0 = time.perf_counter()
        base = self.rel.hints(self.p, self._base_sys, c_in, a, b, y_out, self.device)
        extra, digests = hashchain.hash_hints(self.hr, a, b, len(vus), lay.l, self.device,
                                              sponges=self._sponges_for(auth, a, b, len(vus)))
        hints = torch.cat([base, extra], 0)
        table = hashchain.public_table(digests.cpu().numpy(), len(vus), self.steps, lay.l, self.hr.n_digest)
        idx = auth.indices()
        pub = self.hooks.public_vectors(self.p, None, table, idx, None, device=self.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        hooks = self.hooks_for(auth)
        sibs = None
        if hasattr(coins, "set_statement"):
            # a live verifier's coin source (live.LiveCoins): the v5 statement -- multiproofs included -- travels with the
            # root, so the openings are taken before prove here (outside prove's clock either way).
            t1 = time.perf_counter()
            sibs = hashauth.open_trees(self.trees, auth)
            openings = time.perf_counter() - t1
            from .live import set_statement
            set_statement(coins, self, (table, idx, ypub, auth, sibs), lay, self.cfg(lay.l), self.n_proofs)
        proof = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                      chain=(lay, ypub), words={"a": table, "b": idx}, coins=coins, rel=hooks)
        if sibs is None:
            t1 = time.perf_counter()
            sibs = hashauth.open_trees(self.trees, auth)
            openings = time.perf_counter() - t1
        proof.timings["auth_openings"] = openings
        proof.timings["hints_host"] = hints_seconds
        return proof, (table, idx, ypub, auth, sibs), lay

    def verify_vus(self, proof, pubs, lay: Layout, coins=None, auth_timing: dict | None = None):
        from . import hashauth, hashchain
        table, idx, ypub, auth, sibs = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        if not np.array_equal(np.asarray(idx, dtype=np.int64), auth.indices()) or auth.n_vus != lay.n_vus:
            return False, "authentication block does not match the statement's indices / layout"
        ok, why = verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, table, idx, None, proof, self.device,
                         chain=(lay, ypub), coins=coins, rel=self.hooks_for(auth))
        if not ok:
            return ok, why
        t0 = time.perf_counter()
        ends = np.arange(lay.n_vus) * lay.steps + lay.steps - 1
        tab = np.asarray(table, dtype=np.int64)
        if (tab[ends, 0] != 1).any() or tab[:, 0].sum() != lay.n_vus or tab[np.setdiff1d(np.arange(lay.l), ends)].any():
            return False, "public table: is_end / digests off the chain ends"
        y_words = np.asarray(ypub, dtype=np.int64)[ends]
        ok, why = hashauth.verify_hash_auth(auth, sibs, digests=tab[ends, 1:1 + self.hr.n_digest], y_words=y_words,
                                            y_bits=self.rel.y_bits, leaf=self.leafp)
        if auth_timing is not None:
            auth_timing["seconds"] = auth_timing.get("seconds", 0.0) + (time.perf_counter() - t0)
        return ok, why


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
    # (4b) folded relations (``rel.fold`` instruction steps per column): the column boundary is where the fold could leak --
    # claim the intermediate word of the last column (the state after fold - 1 of its steps, which the base relation would
    # publish and the folded one keeps private), and swap the two operand words on either side of a column boundary while
    # keeping the original claim (a different chain, the same statement layout)
    if rel.fold > 1:
        # relation-level: the prover PROVES the bad VU (check=False), so the constraints -- not the statement digest --
        # have to reject it (the proof's transcript is consistent with the bad statement)
        from dataclasses import replace as _replace
        base_model = _replace(rel.model, groups=tuple(rel.model.groups)[: len(rel.model.groups) // rel.fold])
        kb, K = base_model.k, rel.k
        # the boundary VU: VU 0's words in its LAST column only (zeros before it), so the accumulator entering that column
        # is +0 and each of the column's fold steps moves the word -- a K = 1536 chain's accumulator has grown past the
        # adder's window by its last column and later products truncate away, which would make every negative below vacuous
        a_vu, b_vu = [0] * (K * (steps - 1)) + list(base[0][0][K * (steps - 1):]), [0] * (K * (steps - 1)) + list(base[0][1][K * (steps - 1):])
        accs, c_in = [], 0
        for s in range(steps):
            c_in = tc_dot(rel.model, c_in, a_vu[K * s:K * s + K], b_vu[K * s:K * s + K])
            accs.append(c_in)
        # NOTE: ``marshal`` caches sub-batch bundles by ``id()`` of the VU tuples, so every VU built here is kept alive in
        # ``keep`` until the battery is done -- a freed temporary's id can be reused and would replay the previous bundle
        keep = []

        def prove(vu):
            keep.append(vu)
            pf, pb, ly = R.prove_vus([vu, base[1]], check=False)
            return R.verify_vus(pf, pb, ly)
        ok, why = prove((a_vu, b_vu, accs, accs[-1]))
        negs.append(("folded boundary VU (honest control: must ACCEPT)", (not ok, "honest boundary VU accepted" if ok else why)))
        y_end, mid = rel.y_public(accs[-1]), accs[-2] if steps > 1 else 0
        for s in range(rel.fold - 1):                       # the state after 1 .. fold - 1 of the last column's steps: the
            lo = K * (steps - 1) + kb * s                   # words the base relation publishes and the fold keeps private
            mid = tc_dot(base_model, mid, a_vu[lo:lo + kb], b_vu[lo:lo + kb])
            if rel.y_public(mid) != y_end:
                negs.append((f"folded boundary: proved the {s + 1}-step intermediate of the last column as the final word",
                             prove((a_vu, b_vu, accs[:-1] + [mid], mid))))
        for nm, words in (("a", a_vu), ("b", b_vu)):
            for s in range(1, rel.fold):                    # across a base-step boundary inside the last column
                w2 = list(words)
                i = K * (steps - 1) + kb * s
                w2[i - 1], w2[i] = w2[i], w2[i - 1]
                vu2 = (w2, b_vu, accs, accs[-1]) if nm == "a" else (a_vu, w2, accs, accs[-1])
                if rel.y_public(tc_dot(rel.model, accs[-2] if steps > 1 else 0, vu2[0][K * (steps - 1):], vu2[1][K * (steps - 1):])) != y_end:
                    negs.append((f"folded boundary: {nm} words {i - 1} and {i} swapped across base step {s} of the last column, original claim",
                                 prove(vu2)))
                    break
        c2 = accs[-3] if steps > 2 else 0
        for d in range(1, kb + 1):                          # across the column boundary (a swap of two dead / zero products
            w2 = list(a_vu)                                 # changes nothing: take the first offset whose swap moves the word)
            i, j = K * (steps - 1) - d, K * (steps - 1) + d - 1
            w2[i], w2[j] = w2[j], w2[i]
            m2 = tc_dot(rel.model, c2, w2[K * (steps - 2):K * (steps - 1)], b_vu[K * (steps - 2):K * (steps - 1)])
            m3 = tc_dot(rel.model, m2, w2[K * (steps - 1):K * steps], b_vu[K * (steps - 1):K * steps])
            if rel.y_public(m3) != y_end:
                negs.append((f"folded boundary: a words {i} and {j} swapped across columns {steps - 2}|{steps - 1}, original claim",
                             prove((w2, b_vu, accs, accs[-1]))))
                break
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


def hashed_negatives(R: HashedRelationRunner, data: list, row_negatives: int = 40) -> list[tuple[str, tuple[bool, str]]]:
    """The committed-operand negatives (both verifiers must reject): a wrong digest published, digests swapped between VUs,
    a wrong path, a path to another root, a word of a committed row flipped (the prover computed on another row than the
    one it claims), the W column of another VU, the hashed word changed under the computation (a bit row of the sponge
    input), a broken sponge link, a nonzero sponge start, an index out of range, two VUs claiming one leaf with different
    digests; then one witness row per (scope, class) of the added gadget."""
    from . import hashchain
    from .auth import TreeRef
    rel, steps = R.rel, R.steps
    # the second VU is the first whose x row AND W column both differ from VU 0's (in a tile VUs 0 and 1 share the x row,
    # so "computed on VU 1's x row" would be an honest proof there)
    j = next((i for i in range(1, len(data)) if list(data[i][0]) != list(data[0][0]) and list(data[i][1]) != list(data[0][1])), 1)
    base = [data[0], data[j]]
    ids = [0, j]
    negs = []
    proof, (table, idx, ypub, auth, sibs), lay = R.prove_vus(base, check=False, vu_ids=ids)
    ends = [steps - 1, 2 * steps - 1]

    def V(pubs, name):
        negs.append((name, R.verify_vus(proof, pubs, lay)))
    t = table.copy(); t[ends[0], 1] ^= 1
    V((t, idx, ypub, auth, sibs), "x digest lane 0 of VU 0 flipped")
    t = table.copy(); t[ends[1], 9] = (t[ends[1], 9] + 1) % P
    V((t, idx, ypub, auth, sibs), "W digest lane 0 of VU 1 changed")
    t = table.copy(); t[ends[0], 1:], t[ends[1], 1:] = table[ends[1], 1:].copy(), table[ends[0], 1:].copy()
    V((t, idx, ypub, auth, sibs), "digests of VU 0 and VU 1 swapped")
    t = table.copy(); t[ends[0], 0] = 0; t[ends[0] - 1, 0] = 1
    V((t, idx, ypub, auth, sibs), "is_end moved one column early")
    for ti, tn in enumerate(("a", "b", "y")):
        if sibs[ti]:
            s2 = [list(s) for s in sibs]
            s2[ti][0] = bytes([s2[ti][0][0] ^ 1]) + s2[ti][0][1:]
            V((table, idx, ypub, auth, s2), f"multiproof sibling of tree {tn} corrupted")
            s2 = [list(s) for s in sibs]
            s2[ti] = s2[ti][:-1]
            V((table, idx, ypub, auth, s2), f"multiproof of tree {tn} truncated")
    ra, rb, ry = auth.trees
    other = dataclasses.replace(auth, trees=(TreeRef("a", ra.binding, ra.owner, ra.count, rb.root), rb, ry))
    V((table, idx, ypub, other, sibs), "x root replaced by the W root (verifier's statement)")
    other = dataclasses.replace(auth, trees=(TreeRef("a", rb.binding, ra.owner, ra.count, ra.root), rb, ry))
    V((table, idx, ypub, other, sibs), "x tree binding changed (verifier's statement)")
    idx2 = idx.copy(); idx2[0, 1] = auth.x_index[1]
    other = dataclasses.replace(auth, x_index=(auth.x_index[1], auth.x_index[1]))
    V((table, idx2, ypub, other, sibs), "VU 0 claims VU 1's x row (two VUs, one leaf, different digests)")
    other = dataclasses.replace(auth, w_index=(auth.w_index[0], ra.count))
    idx2 = idx.copy(); idx2[1, 2] = ra.count
    V((table, idx2, ypub, other, sibs), "W leaf index out of range")
    V((table, idx, ypub, auth, sibs[:2]), "a multiproof missing")
    # the prover computed on other rows than the committed ones (a self-consistent proof of the wrong statement)
    def recomputed(a, b):
        k, accs, c = rel.k, [], 0
        for s in range(rel.steps):
            c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
            accs.append(c)
        return (a, b, accs, accs[-1])
    a0, b0 = list(base[0][0]), list(base[0][1])
    a_flip = list(a0); a_flip[5] ^= 1
    pf, pb, ly = R.prove_vus([recomputed(a_flip, b0), base[1]], check=False, vu_ids=ids)
    negs.append(("one word of the committed x row flipped (computed on the flipped row)", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus([recomputed(a0, list(base[1][1])), base[1]], check=False, vu_ids=ids)
    negs.append(("VU 0 computed on VU 1's W column, claims its own", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus([recomputed(list(base[1][0]), b0), base[1]], check=False, vu_ids=ids)
    negs.append(("VU 0 computed on VU 1's x row, claims its own", R.verify_vus(pf, pb, ly)))
    # witness-level: the hashed word differs from the computed word; the carried hash state's links (sponge capacity /
    # leaf accumulator)
    rows = {r.name: r.idx for r in R.sys.rows}
    hb0 = rows["hash.b[0].b0"]
    def flip_hashed_bit(W):
        W[hb0, 3] = (W[hb0, 3] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=flip_hashed_bit, vu_ids=ids)
    negs.append(("bit 0 of hashed word b[0] flipped in unit 3 (computation unchanged)", R.verify_vus(pf, pb, ly)))
    cap_rows = R.sys.chain["c"]
    state = "sponge capacity" if R.leaf is None else f"{R.leaf.name} state"
    def break_cap(W):
        W[cap_rows[5], 7] = (W[cap_rows[5], 7] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=break_cap, vu_ids=ids)
    negs.append((f"{state} link broken at unit 7", R.verify_vus(pf, pb, ly)))
    def cap_start(W):
        W[cap_rows[12], steps] = 5
    pf, pb, ly = R.prove_vus(base, check=False, mutate=cap_start, vu_ids=ids)
    negs.append((f"nonzero {state} at a chain start", R.verify_vus(pf, pb, ly)))
    # one witness row per (scope, class) of the added rows
    seen, picks = set(), []
    for r in R.sys.rows[R.hr.census["rows_base"]:]:
        key = (".".join(r.name.split(".")[:3]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(13)
    for r in picks[:max(0, row_negatives)]:
        col = int(rng.integers(0, 2 * steps))
        delta = int(rng.integers(1, 3))
        def mutate(W, idx_=r.idx, col=col, delta=delta, kind=r.kind):
            if kind == "inv":
                # t * inv = nz leaves inv FREE where t = 0 (inv = 0 there): mutate a column where it is constrained
                nz = (W[idx_, :2 * steps] != 0).nonzero().flatten()
                col = int(nz[col % len(nz)]) if len(nz) else col
            W[idx_, col] = (W[idx_, col] + delta) % P
        pf, pb, ly = R.prove_vus(base, check=False, mutate=mutate, vu_ids=ids)
        negs.append((f"row {r.name} ({r.kind}) +{delta} in column {col}", R.verify_vus(pf, pb, ly)))
    return negs


def leaf_negatives(R: HashedRelationRunner, data: list) -> list[tuple[str, tuple[bool, str]]]:
    """The brief's three negatives of a linear ``leaf/`` digest (§3.4), each a self-consistent proof of the wrong thing:
    the digest of ANOTHER committed row published (the tree opens it, the relation's digest of the row computed on does
    not match); one operand bit of the hashed row flipped in the witness (every bit row is a committed wire; the
    accumulator rows are recomputed so the digest changes and no longer opens under the root); a non-binary ``s``: a bit
    row set to 2 with the accumulator recomputed consistently (the booleanity constraint, not the digest, must reject)."""
    from . import hashchain
    leaf, rel, steps, k, nb = R.leaf, R.rel, R.steps, R.hr.k, R.hr.word_bits
    j = next((i for i in range(1, len(data)) if list(data[i][0]) != list(data[0][0]) and list(data[i][1]) != list(data[0][1])), 1)
    base, ids = [data[0], data[j]], [0, j]
    negs = []
    n = leaf.digest_elems
    proof, (table, idx, ypub, auth, sibs), lay = R.prove_vus(base, check=False, vu_ids=ids)
    # 1. the digest of a different row: VU 0 publishes VU j's x digest (the tree opens VU j's leaf only under VU j's rank)
    t = table.copy()
    t[steps - 1, 1:1 + n] = table[2 * steps - 1, 1:1 + n]
    negs.append(("digest of a different row published for VU 0's x row", R.verify_vus(proof, (t, idx, ypub, auth, sibs), lay)))
    # ... and claiming VU j's rank for it too (then the multiproof opens, but the relation's digest is of VU 0's bits)
    idx2 = idx.copy(); idx2[0, 1] = auth.x_index[1]
    other = dataclasses.replace(auth, x_index=(auth.x_index[1], auth.x_index[1]))
    negs.append(("digest of a different row, its rank claimed (relation digest != published)",
                 R.verify_vus(proof, (t, idx2, ypub, other, sibs), lay)))
    rows = {r.name: r.idx for r in R.sys.rows}
    acc_rows = {nm: [rows[f"hash.{nm}.acc[{x}]"] for x in range(n)] for _, nm, _ in hashchain.ROLES}
    out_rows = {nm: [rows[f"hash.{nm}.acc_out[{x}]"] for x in range(n)] for _, nm, _ in hashchain.ROLES}
    bit_rows = {nm: [rows[f"hash.{nm}[{i}].b{j_}"] for i in range(k) for j_ in range(nb)] for _, nm, _ in hashchain.ROLES}

    def recompute(W, nm: str, col0: int):
        """Recompute the accumulator rows of the VU holding column ``col0`` from its (mutated) bit rows: a prover that keeps
        its own state consistent, so only the digest pin / booleanity can catch it."""
        v0 = (col0 // steps) * steps
        bits = W[bit_rows[nm]][:, v0:v0 + steps].T.to(torch.int64)             # (steps, m)
        acc = torch.zeros(n, dtype=torch.int64, device=W.device)
        for s in range(steps):
            W[acc_rows[nm], v0 + s] = acc.to(W.dtype)                          # W is int32 on the device, int64 on the CPU
            acc = leaf.step(acc, bits[s])
            W[out_rows[nm], v0 + s] = acc.to(W.dtype)

    # 2. one bit flipped in the hashed x row of VU 0 (column 2, word 3, bit 1), state recomputed
    def flip_bit(W):
        r = bit_rows["a"][3 * nb + 1]
        W[r, 2] = 1 - W[r, 2]
        recompute(W, "a", 2)
    pf, pb, ly = R.prove_vus(base, check=False, mutate=flip_bit, vu_ids=ids)
    negs.append(("one bit of the hashed x row flipped (digest recomputed consistently)", R.verify_vus(pf, pb, ly)))
    # 3. non-binary s: a bit row of the W column of VU j set to 2, state recomputed (booleanity must reject)
    def non_binary(W):
        r = bit_rows["b"][5 * nb]
        W[r, steps + 1] = 2
        recompute(W, "b", steps + 1)
    pf, pb, ly = R.prove_vus(base, check=False, mutate=non_binary, vu_ids=ids)
    negs.append(("non-binary operand bit s = 2 (digest recomputed consistently)", R.verify_vus(pf, pb, ly)))
    # 3b. non-binary s AND the digest pin honoured: bit 0 of one word set to 2 while bit 1 is cleared (same word value,
    # different bit vector -> a different digest unless caught): the decode constraint word = sum b_j 2^j still holds
    def non_binary_same_word(W):
        r0, r1 = bit_rows["b"][7 * nb], bit_rows["b"][7 * nb + 1]
        if W[r1, steps + 2] == 1 and W[r0, steps + 2] == 0:
            W[r0, steps + 2], W[r1, steps + 2] = 2, 0
        else:
            W[r0, steps + 2], W[r1, steps + 2] = (W[r0, steps + 2] + 2) % P, (W[r1, steps + 2] + P - 1) % P
        recompute(W, "b", steps + 2)
    pf, pb, ly = R.prove_vus(base, check=False, mutate=non_binary_same_word, vu_ids=ids)
    negs.append(("non-binary bit vector of the same word value (2*1 + 0*2 for 0*1 + 1*2)", R.verify_vus(pf, pb, ly)))
    # 4. the carried state tampered without recomputation: acc_out[3] + 1 at one column (the linear row must reject)
    def tamper_out(W):
        W[out_rows["a"][3], 1] = (W[out_rows["a"][3], 1] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=tamper_out, vu_ids=ids)
    negs.append(("leaf accumulator output row tampered", R.verify_vus(pf, pb, ly)))
    return negs


def gate_hashed(args, rel: Relation) -> int:
    """``gate-vu --auth included-hash``: honest sub-batches of the hashed relation accepted (worst case, or the tile of
    ``--tile``), the relation's own negative families still rejected (claimed word, off-domain word now caught by the
    private decode, broken accumulator link, accumulator bits) and the committed-operand negatives of ``hashed_negatives``."""
    R = HashedRelationRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode, leaf=leaf_of(args))
    log = print
    log(f"{R.relation} hashed chain system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; census {R.hr.census}; device {args.device}")
    report = {"relation": R.relation, "model": rel.model_name, "authentication": R.AUTH, "hash": R.hr.hash_name, "system": R.sys.summary(),
              "census": census(R.sys), "hash_census": R.hr.census, "positives": [], "negatives": []}
    failures = 0
    per_proof = args.batch // R.steps
    tile = parse_tile(getattr(args, "tile", None))
    procs = getattr(args, "instance_procs", 1)
    cache = getattr(args, "instances_cache", None)
    root = getattr(args, "root", None)
    if tile is None:
        data = instances(rel, args.vus, procs=procs, cache=cache, root=root)
        R.commit_vus(data, manifest_sha256=instances_digest(rel, args.vus, root=root))
    else:
        if rel.frozen_tier:
            raise SystemExit(f"--tile: the frozen {rel.frozen_tier} set has no tile layout (sharing = none only for {rel.name})")
        data, x_rows, w_cols = tile_instances(rel, *tile, procs=procs, cache=cache)
        data = data[:args.vus]
        R.commit_vus(data, manifest_sha256=instances_digest(rel, len(data)), tile=tile, x_rows=x_rows, w_cols=w_cols)
    for lo in range(0, len(data), per_proof):
        vus = data[lo:lo + per_proof]
        proof, pubs, lay = R.prove_vus(vus, check=True, vu_ids=range(lo, lo + len(vus)))
        ok, why = R.verify_vus(proof, pubs, lay)
        sib_bytes = 32 * sum(len(s) for s in pubs[4])
        log(f"{rel.name} hashed vu-k1536 [{lo}, {lo + len(vus)}): l={lay.l} prover {proof.timings['total']:.2f}s "
            f"(+hints {proof.timings['hints_host']:.2f}s) multiproofs {sib_bytes} B -> {ok} {why}")
        report["positives"].append({"range": [lo, lo + len(vus)], "l": lay.l, "ok": ok, "why": why, "timings": proof.timings,
                                    "multiproof_bytes": sib_bytes})
        failures += not ok
    # the relation's own negatives on the hashed system: claimed word, links, accumulator bits (the operand words are
    # private now: the off-domain family becomes "a word off the finite domain is proved" -> rejected by the decode)
    base = data[:2]
    proof, (table, idx, ypub, auth, sibs), lay = R.prove_vus(base, check=False, vu_ids=[0, 1])
    negs = []
    for bit in (0, rel.y_bits - 1):
        y_bad = ypub.clone(); y_bad[R.steps - 1] ^= 1 << bit
        negs.append((f"claimed word bit {bit} flipped", R.verify_vus(proof, (table, idx, y_bad, auth, sibs), lay)))
    c_rows = R.sys.chain["c"]
    def break_link(W):
        W[c_rows[2], 20] = (W[c_rows[2], 20] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=break_link, vu_ids=[0, 1])
    negs.append(("accumulator link c.f[20] != y.f[19]", R.verify_vus(pf, pb, ly)))
    def acc_bit(W):
        W[c_rows[2], 30] = (W[c_rows[2], 30] ^ (1 << 12)) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=acc_bit, vu_ids=[0, 1])
    negs.append(("accumulator fraction bit 12 flipped in a linked unit", R.verify_vus(pf, pb, ly)))
    off = list(base[0][0]); off[4] = rel.off_domain[0]
    k, accs, c = rel.k, [], 0
    for s in range(rel.steps):     # the model on an off-domain word: whatever it records, the private decode must reject the word
        try:
            c = tc_dot(rel.model, c, off[k * s:k * s + k], base[0][1][k * s:k * s + k])
        except Exception:
            c = 0
        accs.append(c)
    try:
        pf, pb, ly = R.prove_vus([(off, base[0][1], accs, accs[-1]), base[1]], check=False, vu_ids=[0, 1])
        negs.append((f"off-domain operand 0x{rel.off_domain[0]:x} computed on", R.verify_vus(pf, pb, ly)))
    except Exception as e:                                  # the hint generator itself refuses the word: also a rejection
        negs.append((f"off-domain operand 0x{rel.off_domain[0]:x} computed on", (False, f"prover: {type(e).__name__}: {e}")))
    negs += hashed_negatives(R, data, getattr(args, "row_negatives", 40))
    if R.leaf is not None:
        negs += leaf_negatives(R, data)
    for name, (ok, why) in negs:
        log(f"  negative {name}: {'REJECTED' if not ok else 'ACCEPTED (BUG)'} {why}")
        report["negatives"].append({"name": name, "accepted": ok, "why": why})
        failures += ok
    report["failures"] = failures
    log(f"{rel.name} hashed gate: {len(report['positives'])} honest sub-batches, {len(negs)} negatives, {failures} failures")
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


def leaf_of(args):
    """``--leaf`` (``--auth included-hash``): the ``leaf/`` scheme object, ``None`` for Poseidon2 (the default, ``hash``)."""
    from .leaf.registry import leaf_for
    return leaf_for(getattr(args, "leaf", None))


def make_runner(args, rel: Relation, **kw):
    """The runner ``--auth`` selects: ``excluded`` (the bare v4 statement) or ``included-hash`` (``HashedRelationRunner``,
    with the ``--leaf`` scheme)."""
    auth = getattr(args, "auth", "excluded") or "excluded"
    if auth == "excluded":
        return RelationChainRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode, **kw)
    if auth == HashedRelationRunner.AUTH:
        return HashedRelationRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode, leaf=leaf_of(args), **kw)
    raise SystemExit(f"--auth {auth}: not a mode of the relation runner (excluded | included-hash)")


def gate_vu_rel(args, rel: Relation) -> int:
    if getattr(args, "auth", "excluded") == HashedRelationRunner.AUTH:
        return gate_hashed(args, rel)
    R = RelationChainRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode)
    log = print
    log(f"{rel.name} chain system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; device {args.device}")
    report = {"relation": rel.name, "model": rel.model_name, "system": R.sys.summary(), "census": census(R.sys),
              "positives": [], "negatives": []}
    failures = 0
    per_proof = args.batch // R.steps
    data = instances(rel, args.vus, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None),
                     root=getattr(args, "root", None))
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
    ZK_HASHED_NOTE = ("; NOTE: authentication = included-hash: the operand rows x, W are NOT in this statement (private witness, "
                      "hashed in-circuit); the statement publishes their Poseidon2-BabyBear row digests (deterministic, unsalted: "
                      "hiding only up to preimage search on a row), the Merkle roots + (vu, x_index, w_index), and the words y16")

    def _git_commit():
        return os.environ.get("RESEARCH_GIT_COMMIT") or _git_commit_repo()

    steps = rel.steps
    per_proof = args.batch // steps
    total_vus = args.total_vus or per_proof
    n_proofs = -(-total_vus // per_proof)
    interactive = args.mode == protocol_mod.MODE_INTERACTIVE
    depth = 3 if interactive else 1
    R = make_runner(args, rel, n_proofs=n_proofs)
    hashed = isinstance(R, HashedRelationRunner)
    # --pipeline N (lane hp2-host): the staged prover; the hashed runner (lane hash-relation) has no stage generator yet
    pipeline_depth = int(getattr(args, "pipeline", 0) or 0) if not hashed else 0
    tile = parse_tile(getattr(args, "tile", None)) if hashed else None
    if tile is not None and tile[0] * tile[1] != total_vus:
        raise SystemExit(f"--tile {tile[0]}x{tile[1]} is {tile[0] * tile[1]} VUs; --total-vus {total_vus}")
    cen = census(R.sys)
    print(f"{rel.name} census:", json.dumps(cen))
    if hashed:
        print(f"{rel.name} hashed ({R.AUTH}, sharing {'none' if tile is None else 'tile%dx%d' % tile}):", json.dumps(R.hr.census))
    subs = [(lo, min(lo + per_proof, total_vus)) for lo in range(0, total_vus, per_proof)]
    root = getattr(args, "root", None)
    t0 = time.perf_counter()
    if tile is None:
        data = instances(rel, total_vus, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None), root=root)
        x_rows = w_cols = None
    else:
        if rel.frozen_tier:
            raise SystemExit(f"--tile: the frozen {rel.frozen_tier} set has no tile layout (sharing = none only for {rel.name})")
        data, x_rows, w_cols = tile_instances(rel, *tile, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None))
    inst_digest = instances_digest(rel, total_vus, root=root)
    print(f"{total_vus} {'frozen ' + rel.frozen_tier if rel.frozen_tier else 'synthetic'} VUs (recorded accumulators) in {time.perf_counter() - t0:.1f}s")
    commit_seconds = 0.0
    if hashed:
        t0 = time.perf_counter()
        R.commit_vus(data, dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=inst_digest,
                     cache=getattr(args, "auth_cache", None), tile=tile, x_rows=x_rows, w_cols=w_cols)
        commit_seconds = time.perf_counter() - t0
        print(f"committed: x tree {R.trees['a'].count} rows, W tree {R.trees['b'].count} columns, y tree {R.trees['y'].count} words "
              f"(leaf/v2h Poseidon2 + SHA-256) in {commit_seconds:.1f}s (row sponges on {args.device}: {R.sponge_seconds:.2f}s); "
              f"roots a={R.trees['a'].root.hex()[:16]} b={R.trees['b'].root.hex()[:16]}")
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
                                "instances": inst_digest}
    target_bits = float(getattr(args, "target_bits", 128.0) or 128.0)   # the VERIFIER's parameter (PROTOCOL.md 8c)
    # --verifier tcp://host:port: the step-0 coins of every RECORDED rep come from a LIVE verifier on another machine
    # (live.py) and every prover message is sent to it as it is produced -- the protocol is unchanged (8c), only the coin
    # source is; the warm-up rep keeps local coins (it presents one sub-batch, not n_proofs).
    live = None
    if getattr(args, "verifier", None):
        from .live import BenchLive
        live = BenchLive(args, R, n_proofs, total_vus, rel.name, interactive)
        if pipeline_depth > 1 and args.device.startswith("cuda"):
            live.window = pipeline_depth                 # the verifier commits N sub-batches ahead (N openings in flight)
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
            # the prover's work hides under the wire time and t.total_live -> the session's network critical path (the
            # verifier opens the sub-batches strictly in order, 2 x RTT each, on one control connection).  PROOF i is
            # sent on the data channel after the pass.
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
            proof, pubs, lay_i = R.prove_vus(data[lo:hi], check=(rep == 0 and lo == 0), min_l=l, coins=V, vu_ids=range(lo, hi))
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
                         "stmt_bytes": (rd / f"{stem}.stmt").stat().st_size,
                         "python_verdict": {"accepted": ok, "reason": why}, "prover_seconds": proof.timings.get("total")}
                if hashed:
                    entry["multiproof_bytes"] = 32 * sum(len(x) for x in pubs[4])
                    entry["auth_openings_seconds"] = proof.timings.get("auth_openings")
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
    # The bare v4 statement publishes the operands, so vu.py's ZK_VACUOUS_NOTE applies; the hashed v5 statement does not
    # (hash-relation final report §9): it carries the operand rows' Poseidon2 digests + the Merkle roots + y16 instead.
    privacy_note = (ZK_HASHED_NOTE if hashed else ZK_VACUOUS_NOTE)
    auth_omitted = ("" if hashed else "external authentication; ")
    if cfg.zk and interactive:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", ZK_MODE + ZK_MODE_HM96 + privacy_note
        omitted = auth_omitted + "unopened columns hidden by Blake3 of a uniform column"
    elif cfg.zk:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_MODE_FS + privacy_note
        omitted = auth_omitted + "malicious-verifier ZK outside the programmable random-oracle model"
    else:
        proof_class, zk_mode, omitted = "NON_ZK_PROOF_DIAGNOSTIC", "none", auth_omitted + "ZK masking"
    zk_statement = snd["zk_statement"]
    if hashed and cfg.zk:
        zk_statement = zk_statement.split("; operands a, b")[0] + ZK_HASHED_NOTE.replace("; NOTE: ", "; ", 1)
    if rel.frozen_tier:
        # the frozen set: the same ``instances`` block as vu.bench_vu (no seed, no recipe), so the committed column of the
        # A100 cell names exactly the bare cell's instances
        inst = contract.instances_ref(rel.instances_tier, 0, total_vus, inst_digest, dataset=rel.instances_dataset)
    else:
        inst = {"dataset": rel.instances_dataset, "tier": rel.instances_tier, "range": [0, total_vus],
                "manifest_sha256": inst_digest, "seed": rel.instance_seed}
        if rel.instances_recipe:
            inst["recipe"] = rel.instances_recipe
    fp = contract.fingerprint(
        profile=rel.target.name, domain=contract.DOMAIN, K=K_VU, B=total_vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication=(R.AUTH if hashed else "excluded"), zk_mode=zk_mode,
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_k": cfg.k, "rs_l": cfg.l,
                  "zk": cfg.zk, "t_pad": cfg.t_pad, "mode": cfg.mode, "bound_model": snd["bound_model"],
                  "hash_assumption": snd["hash_assumption"], "zk_statement": zk_statement,
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
        statement_format="ligero-statement/v5" if hashed else "ligero-statement/v4",   # what --dump-dir writes (serialize.py)
        note=f"{total_vus} VUs as {n_proofs} independently committed proofs of <= {per_proof} VUs each (l = {l} columns, "
             f"{l - per_proof * steps} pad columns per proof); prover time = sum over the proofs; soundness = union bound; "
             f"{rel.claim_note}"
             + (f"; authentication = {R.AUTH}: the operand words are private witness wires, each unit hashes its 16 lanes of "
                f"x and of W with {_hash_note(R)} along the chain, the digests are chain-end pins and leaf/v2h leaves under the "
                f"x / W roots; sharing = {'none' if tile is None else 'tile%dx%d' % tile} ({R.trees['a'].count} x rows + "
                f"{R.trees['b'].count} W columns committed for {total_vus} VUs); the statement carries digests, leaf indices, "
                f"roots and SHA-256 multiproofs instead of the operand words; hash = {R.hr.hash_name}" if hashed else ""),
    )
    if hashed:
        fp["hash"] = R.hr.hash_name
        fp["relation_hashed"] = R.relation
        fp["digest_elems_per_row"] = R.hr.lanes
        fp["digest_bytes_per_row"] = 4 * R.hr.lanes
        fp["sharing"] = "none" if tile is None else f"tile{tile[0]}x{tile[1]}"
        fp["committed_rows"] = {"x": R.trees["a"].count, "w": R.trees["b"].count, "y": R.trees["y"].count}
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
    if hashed:
        meas += [contract.measurement("relation.hash.rows_added_per_unit", R.hr.census["rows_added"], "count"),
                 contract.measurement("relation.hash.permutations_per_unit", R.hr.census.get("permutations_per_unit", 0), "count"),
                 contract.measurement("relation.hash.linked_rows", R.hr.census["linked_rows"], "count"),
                 contract.measurement("relation.hash.digest_elems_per_row", R.hr.lanes, "count"),
                 contract.measurement("relation.hash.digest_bytes_per_row", 4 * R.hr.lanes, "B"),
                 contract.measurement("relation.hash.commit_seconds", commit_seconds, "s"),
                 contract.measurement("relation.hash.sponge_seconds", R.sponge_seconds, "s"),
                 contract.measurement("relation.hash.auth_openings_seconds", med.get("auth_openings", 0.0), "s"),
                 contract.measurement("relation.hash.committed_x_rows", R.trees["a"].count, "count"),
                 contract.measurement("relation.hash.committed_w_cols", R.trees["b"].count, "count")]
        if dump_manifest is not None and dump_manifest["files"]:
            fs = [e for e in dump_manifest["files"] if e["rep"] == 1]
            meas += [contract.measurement("relation.hash.statement_bytes_total", sum(e["stmt_bytes"] for e in fs), "B"),
                     contract.measurement("relation.hash.multiproof_bytes_total", sum(e["multiproof_bytes"] for e in fs), "B")]
    elif dump_manifest is not None and dump_manifest["files"]:
        fs = [e for e in dump_manifest["files"] if e["rep"] == 1]
        meas += [contract.measurement("relation.statement_bytes_total", sum(e["stmt_bytes"] for e in fs), "B")]
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


def _hash_note(R) -> str:
    if R.leaf is None:
        return (f"Poseidon2 (BabyBear, width 24, rate 16, x^7; Plonky3 constants, {R.hr.census['sbox_rows_per_permutation']} "
                "product rows per permutation)")
    return f"the linear {R.leaf.tag} leaf ({R.leaf.assumption}; {R.hr.lanes} BabyBear elements per row digest, {R.hr.census['linear_added']} linear rows per unit)"


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
