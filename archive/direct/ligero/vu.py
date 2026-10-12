"""The proof unit (chain of 96 units + epilogue) on the Ligero prover: gate and measurement.

    python -m archive.direct.ligero.run gate-vu  --root fixtures/bench-instances/v1 --device cuda --vus 64
    python -m archive.direct.ligero.run bench-vu --root fixtures/bench-instances/v1 --device cuda --reps 5 --out result.json

A proof of l columns holds floor(l / 96) VUs (l = 4096: 42 VUs, 64 pad columns).  Public per proof: the operands
of every column and the claimed BF16 word of every VU; the accumulators are witness (linked across columns by the
chain test), c_0 = 0.
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from verity_numerical.bench import contract, instances
from verity_numerical.checker.params import REAL
from verity_numerical.checker.step import step_output

from . import auth as auth_mod
from . import phases
from . import protocol as protocol_mod
from .chain import Layout
from .compile import P
from .hints_torch import hints_torch
from .protocol import Config, config_for, n_rows, prove, prove_stages, soundness, verify

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
        self.hooks = dataclasses.replace(protocol_mod.BF16_HOOKS, steps=STEPS)   # the verifier's VU shape (96 columns)

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

    def prove_vus(self, vus: list[tuple[list, list, list, int]], check=True, mutate=None, min_l: int = 256, coins=None,
                  auth: tuple | None = None):
        """``vus``: (x, w, accs, y16) per VU (honest data: the recorded accumulators).  ``coins``: the verifier's
        step-0 coin source.  Interactive mode with ``coins=None``: the RUNNER plays the honest verifier's step 0
        (``Coins.sample()``; the same object is what ``verify_vus`` then uses as its own coins -- the runner is both
        parties, as in every gate).  Fiat-Shamir mode: no coins.
        ``auth = (trees, vu_index)`` (authentication = included): the committed word trees (``auth.WordTree`` by
        name) and the index of every VU of ``vus`` in them; the statement then carries the roots and indices and the
        transcript the three multiproofs (``auth.py``); returns ``(proof, pubs, lay, auth_stmt)`` then."""
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
        c_in, y_out = np.asarray([u[0] for u in units], dtype=np.int64), np.asarray([u[3] for u in units], dtype=np.int64)
        return self.prove_arrays(a, b, c_in, y_out, y16, lay, check=check, mutate=mutate, coins=coins, units=units, auth=auth)

    def prove_arrays(self, a, b, c_in, y_out, y16, lay: Layout, check=True, mutate=None, coins=None, units=None,
                     auth: tuple | None = None):
        """The timed part of ``prove_vus`` on already-marshaled arrays: ``a``/``b`` (l, STEP) int64 operands, ``c_in``/
        ``y_out`` (l,) int64 accumulator words in and out of every unit, ``y16`` (l,) int64 tensor with the claimed BF16
        word at each VU's last unit (0 elsewhere), ``lay`` the layout they were padded to.  ``units`` (the tuple form)
        is needed only for ``check=True`` / the Python hint generator.  ``auth`` as in ``prove_vus`` (the authentication
        statement -- the verifier's public input -- is built before the timer; the openings are prover work inside it).
        Everything after ``t0`` is prover work."""
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        auth_stmt = auth_open = None
        if auth is not None:
            trees, vu_index = auth
            auth_stmt = auth_mod.AuthStatement.for_vus(trees, vu_index)
            auth_open = lambda: auth_mod.open_trees(trees, auth_stmt, K=STEP * STEPS, steps=STEPS)
        t0 = time.perf_counter()
        pub = public_vectors(self.p, None, a, b, None, device=self.device)
        if self.hints_mode == "torch":
            hints = hints_torch(self.p, self.sys, c_in, a, b, y_out, self.device)
        else:
            assert units is not None, "the python hint generator needs the unit tuples"
            hints = hints_tensor(self.p, self.sys, units, self.device)
        if check:      # the gate: the torch hint generator must reproduce the checker's gen_step exactly
            assert units is not None, "check=True needs the unit tuples"
            ref = hints_tensor(self.p, self.sys, units, self.device)
            alt = hints_torch(self.p, self.sys, c_in, a, b, y_out, self.device)
            assert torch.equal(ref, alt), "hints_torch != gen_step"
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        if hasattr(coins, "set_statement"):
            # a live verifier's coin source (live.LiveCoins): the statement travels with the root, so the claim is fixed
            # before the verifier opens its first coin.  Built outside prove's clock; the protocol is unchanged.
            from .live import set_statement
            set_statement(coins, self, (a, b, y16), lay, self.cfg(lay.l), self.n_proofs, auth=auth_stmt)
        proof = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                      chain=(lay, y16), words={"a": a, "b": b}, coins=coins, auth=auth_stmt, auth_open=auth_open)
        proof.timings["hints_host"] = hints_seconds
        if auth_stmt is not None:
            return proof, (a, b, y16), lay, auth_stmt
        return proof, (a, b, y16), lay

    def _prove_stages(self, arrays: tuple, lay: Layout, coins, auth: tuple | None = None, slot=None):
        """The prover generator for one marshalled sub-batch (``pipeline.prove_many`` drives several at once; lane live-2,
        the Ampere BF16 counterpart of ``relchain.RelChainRunner._prove_stages``): the public vectors and the torch hints
        (device, timed ``hints_host`` inside the prover's clock, as ``prove_arrays`` reports them) then
        ``protocol.prove_stages``.  Returns ``(generator, auth_stmt)``; the caller sets the statement before starting it."""
        a, b, c_in, y_out, y16 = arrays
        auth_stmt = auth_open = None
        if auth is not None:
            trees, vu_index = auth
            auth_stmt = auth_mod.AuthStatement.for_vus(trees, vu_index)
            auth_open = lambda: auth_mod.open_trees(trees, auth_stmt, K=STEP * STEPS, steps=STEPS)  # noqa: E731

        def hints_fn():
            if self.hints_mode != "torch":
                raise ValueError("--pipeline needs the torch hint generator (hints_mode='torch')")
            return hints_torch(self.p, self.sys, c_in, a, b, y_out, self.device)

        pub = public_vectors(self.p, None, a, b, None, device=self.device)
        gen = prove_stages(self.p, self.sys, self.tb, self.cfg(lay.l), pub, None, self.device, check=False, mutate=None,
                           chain=(lay, y16), words={"a": a, "b": b}, coins=coins, auth=auth_stmt, auth_open=auth_open,
                           hints_fn=hints_fn, slot=slot)
        return gen, auth_stmt

    def prove_arrays_many(self, batches: list[tuple], lays: list[Layout], coins_factory, depth: int = 2,
                          auths: list | None = None):
        """``len(batches)`` marshalled sub-batches (``marshal_ids`` arrays each, layouts ``lays``) proved with up to ``depth``
        in flight on their own CUDA streams (``pipeline.prove_many``; lane live-2 -- ``vu.py --pipeline``).  ``coins_factory(i)``
        is called per sub-batch as it starts, in order; a live coin source (``set_statement``) gets its statement built off
        the driver thread and its coin exchanges are non-blocking (``live.LiveCoins``).  ``auths[i]``: the ``auth`` tuple of
        ``prove_arrays`` per sub-batch (or None).  Returns ``[(coins, proof, pubs, lay, auth_stmt)]`` in order and the pass
        wall; each proof's ``timings`` is its share of the wall (``pipeline._rescale``)."""
        from .pipeline import prove_many
        coins_of: list = [None] * len(batches)
        auth_of: list = [None] * len(batches)

        def starter(i):
            def start(slot):
                c = coins_factory(i)
                coins_of[i] = c
                gen, st_auth = self._prove_stages(batches[i], lays[i], c, auth=(auths[i] if auths else None), slot=slot)
                auth_of[i] = st_auth
                if hasattr(c, "set_statement"):
                    from .live import set_statement
                    a, b, _c, _y, y16 = batches[i]
                    set_statement(c, self, (a, b, y16), lays[i], self.cfg(lays[i].l), self.n_proofs, auth=st_auth)
                return gen
            return start

        proofs, wall = prove_many([starter(i) for i in range(len(batches))], self.device, depth=depth)
        out = []
        for i, pf in enumerate(proofs):
            a, b, _c, _y, y16 = batches[i]
            out.append((coins_of[i], pf, (a, b, y16), lays[i], auth_of[i]))
        return out, wall

    def verify_vus(self, proof, pubs, lay: Layout, coins=None, auth: auth_mod.AuthStatement | None = None,
                   auth_procs: int | None = None, auth_timing: dict | None = None):
        """Interactive mode with ``coins=None``: the verifier's own coins are the ones the runner sampled in
        ``prove_vus`` (``proof.coins``, i.e. the honest step 0 the runner played); pass ``coins`` explicitly to check
        a transcript against a specific verifier's session.  ``auth``: the verifier's own committed-object block."""
        a, b, y16 = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, y16),
                      coins=coins, rel=self.hooks, auth=auth, auth_procs=auth_procs, auth_timing=auth_timing)

    def batch_bound(self, cfgs: list[Config], n_proofs_claimed: list[int], target_bits: float = 128.0,
                    names: list[str] | None = None) -> dict:
        """The batch accounting gate (PROTOCOL.md 8c, "Batch verification") for sub-batch proofs that each verified:
        this verifier's OWN bound of every proof from its configuration (``soundness(cfg_i)``, one commitment each),
        union-bounded over the ``len(cfgs)`` presented, against ``target_bits``; and every proof's ``n_proofs`` (the
        count its ``t`` was sized for) against that number.  Reason strings shared with ``ligero-verify batch``."""
        per = [soundness(c, self.tb.L, self.tb.Q, 1)["per_proof_total_log2"] for c in cfgs]
        return protocol_mod.batch_bound(per, n_proofs_claimed, target_bits, names)

    def verify_batch(self, items: list[tuple], target_bits: float = 128.0, n_proofs_claimed: list[int] | None = None) -> dict:
        """The Python batch verifier on in-memory proofs: ``items`` = ``(proof, pubs, lay[, coins])`` per sub-batch
        presented together.  Every one is verified as ``verify_vus`` does (own statement, own coins); a rejected
        sub-batch rejects the batch with ``"<sub_XX>: <reason>"``; then :meth:`batch_bound` with ``n_proofs_claimed``
        (default: this runner's ``n_proofs`` for every proof -- the count ``config_for`` sized ``t`` for)."""
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


def marshal_ids(vu: instances.VuTier, ids: list[int], lay: Layout):
    """Vectorised loader for ``bench_vu``: the arrays ``prove_vus`` builds from ``[_vu_data(vu, i) for i in ids]`` --
    ``a``, ``b`` (l, STEP) int64, ``c_in``, ``y_out`` (l,) int64, ``y16`` (l,) int64 tensor -- straight from the tier's
    flat operand arrays, padded to ``lay.l`` units with the trivial unit.  Identical to the tuple path (``bench_vu_test.py``
    checks it); it exists because a million VUs as Python lists cost more than the proof does."""
    n_vus, l = len(ids), lay.l
    if n_vus != lay.n_vus or n_vus * STEPS > l:
        raise ValueError(f"{n_vus} VUs do not match layout {lay}")
    idx = np.asarray(ids, dtype=np.int64)
    x = np.frombuffer(vu.x, dtype=np.uint16).reshape(vu.n, STEPS, STEP)[idx]
    w = np.frombuffer(vu.w, dtype=np.uint16).reshape(vu.n, STEPS, STEP)[idx]
    acc = np.frombuffer(vu.acc, dtype=np.uint32).reshape(vu.n, STEPS)[idx]
    a = np.zeros((l, STEP), dtype=np.int64)
    b = np.zeros((l, STEP), dtype=np.int64)
    c_in = np.zeros(l, dtype=np.int64)
    y_out = np.zeros(l, dtype=np.int64)
    a[:n_vus * STEPS] = x.reshape(-1, STEP)
    b[:n_vus * STEPS] = w.reshape(-1, STEP)
    y_out[:n_vus * STEPS] = acc.reshape(-1)
    c = np.zeros_like(acc)
    c[:, 1:] = acc[:, :-1]                      # unit s starts from the recorded accumulator of unit s - 1; c_0 = +0
    c_in[:n_vus * STEPS] = c.reshape(-1)
    y16 = torch.zeros(l, dtype=torch.int64)
    y16[STEPS * torch.arange(n_vus) + STEPS - 1] = torch.from_numpy(
        np.frombuffer(vu.y, dtype=np.uint16)[idx].astype(np.int64))
    return a, b, c_in, y_out, y16


def _vu_data(vu: instances.VuTier, i: int):
    x, w, y16 = vu.instance(i)
    return (x, w, vu.accs(i), int(y16))


# ---------------------------------------------------------------------------------------------------------------------
# authentication = included: the frozen set's words as committed objects (auth.py)
# ---------------------------------------------------------------------------------------------------------------------

def auth_trees_for(vu: instances.VuTier, root: Path, total_vus: int, cache: str | Path | None, procs: int | None):
    """The three word trees over VUs ``[0, total_vus)`` of the frozen ``vu-k1536`` set (from ``cache`` when the
    bindings match) and the fingerprint's description of what was committed and how."""
    if not 0 < total_vus <= vu.n:
        raise ValueError(f"the committed object is a prefix of the frozen set: 0 < total_vus <= {vu.n}")
    K = vu.K
    msha = instances.manifest_sha256(root)
    bindings = {n: auth_mod.binding_digest(dataset=contract.INSTANCES_DATASET, tier=vu.tier, manifest_sha256=msha, lo=0,
                                           hi=total_vus, K=K, tree=n) for n in auth_mod.TREE_NAMES}
    xa = np.frombuffer(vu.x, dtype=np.uint16)[:total_vus * K]
    wb = np.frombuffer(vu.w, dtype=np.uint16)[:total_vus * K]
    yy = np.frombuffer(vu.y, dtype=np.uint16)[:total_vus]
    t0 = time.perf_counter()
    trees, cached = auth_mod.load_or_build_trees(cache, a_words=xa, b_words=wb, y_words=yy, bindings=bindings, procs=procs)
    secs = time.perf_counter() - t0
    from verity.ml.commitments import merkle_depth
    info = {"version": auth_mod.AUTH_VERSION, "scheme": "verifier-side openings (auth-integration option b): Merkle "
            "multiproofs of every boundary word beside the Ligero proof; both verifiers recompute the leaves from the "
            "published words and fold them to the roots",
            "hash": "SHA-256 with the veritor/protocol/merkle/frame/v3 framing (verity.ml.commitments.merkle)",
            "tree": "binary Merkle tree, arity 2, one 16-bit word per leaf (leaf rank == position), padded to 2^depth "
                    "with domain-bound pad leaves; depth = merkle_depth(count)",
            "leaf": "H('leaf' | domain_id | uint(rank) | uint(position) | 'u16' | word as 2 big-endian bytes)",
            "domain": "CommitmentDomain(binding, owner, RangeIndexedDomain(0, count)); binding = identity_digest("
                      f"{auth_mod.BINDING_TAG!r}, {{dataset, tier, manifest_sha256, vu_lo, vu_hi, K, tree, schema}})",
            "multiproof": "canonical shared-path multiproof of verity.ml.commitments.multiproof (per level, ascending, the "
                          "sibling of every known node whose sibling is not known)",
            "trees": {n: {"count": t.count, "depth": merkle_depth(t.count), "root": t.root.hex(),
                          "binding": t.ref.binding.hex(), "owner": t.ref.owner} for n, t in trees.items()},
            "committed_vus": [0, total_vus], "K": K, "commit_seconds": secs, "cached": cached,
            "procs": procs or (__import__("os").cpu_count() or 1)}
    return trees, info


def auth_negatives(R: ChainRunner, trees: dict, vu: instances.VuTier, n_vus: int = 2, procs: int | None = None):
    """The prover-side ``authentication = included`` negatives on the real trees: every proof is a VALID Ligero
    proof (honest witness, the statement it was made for), so only the authentication check can reject it.
    Returns ``[(name, proof, pubs, lay, auth_stmt, meta)]``; the first entry is the honest included proof."""
    from verity_numerical.checker.step import step_output, vu_output
    out = []
    ids = list(range(n_vus))
    vus = [_vu_data(vu, i) for i in ids]
    K = STEP * STEPS

    def run(name, vus_, vu_index, trees_=trees, expect=False, **meta):
        proof, pubs, lay, st = R.prove_vus(vus_, check=False, auth=(trees_, vu_index))
        tm = {}
        ok, why = R.verify_vus(proof, pubs, lay, auth=st, auth_procs=procs, auth_timing=tm)
        out.append((name, proof, pubs, lay, st, {"vu_index": list(vu_index), "python_accepted": ok, "python_reason": why,
                                                 "python_auth_seconds": tm.get("seconds"), "expected_accepted": expect,
                                                 **meta}))
        return ok

    run("auth_honest", vus, ids, expect=True)
    # (1) words not under the root: one operand word of VU 0 swapped (a valid dot product of the changed words; the
    #     accumulators and the output word recomputed by the checker's own relation), claimed at VU 0's indices
    x, w, _accs, _y = vus[0]
    x2 = list(x); x2[5] ^= 0x0001                      # a different BF16 word (mantissa LSB) at k = 5
    accs2, c = [], 0
    for s in range(STEPS):
        c = step_output(R.p, c, x2[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
        accs2.append(c)
    y2 = vu_output(R.p, x2, w)
    run("auth_word_not_under_root", [(x2, w, accs2, y2)] + vus[1:], ids, mutation="a-word k=5 of VU 0 changed; accs/y16 recomputed")
    # (1b) the same on a B-word (weights tree)
    w2 = list(w); w2[K - 1] ^= 0x0001
    accs3, c = [], 0
    for s in range(STEPS):
        c = step_output(R.p, c, x[16 * s:16 * s + 16], w2[16 * s:16 * s + 16])
        accs3.append(c)
    y3 = vu_output(R.p, x, w2)
    run("auth_weight_not_under_root", [(x, w2, accs3, y3)] + vus[1:], ids, mutation=f"b-word k={K - 1} of VU 0 changed; accs/y16 recomputed")
    # (2) wrong index: the honest words of VUs [0, n) claimed at the leaf indices of VUs [1, n + 1)
    run("auth_wrong_index", vus, [i + 1 for i in ids], mutation="vu_index shifted by one (committed words of other VUs)")
    # (2b) wrong index, same words: VU 0 and VU 1 swapped in the proof but not in the index
    if n_vus >= 2:
        run("auth_vus_swapped", [vus[1], vus[0]] + vus[2:], ids, mutation="VUs 0 and 1 swapped against a fixed vu_index")
    # (3) right words, different root: the a tree replaced by a tree over the same words with one bit flipped
    xa = np.frombuffer(vu.x, dtype=np.uint16)[:trees["a"].count].copy()
    xa[7] ^= 0x8000
    other = dict(trees)
    other["a"] = auth_mod.build_word_tree("a", trees["a"].ref.binding, xa, procs=procs)
    run("auth_different_root", vus, ids, trees_=other, mutation="a root of a tree whose leaf 7 differs (the prover opens against it)")
    # (3b) right words, right multiproofs, the y tree's reference carrying the b tree's root: the prover absorbs the
    #      forged block into its statement digest (so the challenge binding holds) and opens against the real y tree
    st_ok = auth_mod.AuthStatement.for_vus(trees, ids)
    forged = auth_mod.AuthStatement(st_ok.vu_index, (st_ok.trees[0], st_ok.trees[1],
                                                     auth_mod.TreeRef("y", st_ok.trees[2].binding, st_ok.trees[2].owner,
                                                                      st_ok.trees[2].count, st_ok.trees[1].root)))
    lay = R.layout(len(vus))
    made = prove_with_auth(R, vus, forged, lambda: auth_mod.open_trees(trees, st_ok, K=K, steps=STEPS))
    pf, pubs = made["proof"], made["pubs"]
    tm = {}
    ok, why = R.verify_vus(pf, pubs, lay, auth=forged, auth_procs=procs, auth_timing=tm)
    out.append(("auth_root_of_other_tree", pf, pubs, lay, forged, {"vu_index": ids, "python_accepted": ok, "python_reason": why,
                                                                  "python_auth_seconds": tm.get("seconds"), "expected_accepted": False,
                                                                  "mutation": "y tree reference carries the b tree's root"}))
    return out


def prove_with_auth(R: ChainRunner, vus, auth_stmt: auth_mod.AuthStatement, auth_open):
    """A proof whose statement digest absorbs ``auth_stmt`` as given (for negatives that forge the block)."""
    lay = R.layout(len(vus))
    units, y16 = [], torch.zeros(lay.l, dtype=torch.int64)
    for v, (x, w, accs, y) in enumerate(vus):
        units += vu_units(x, w, accs)
        y16[STEPS * v + STEPS - 1] = y
    while len(units) < lay.l:
        units.append(_pad_unit())
    a = np.asarray([u[1] for u in units], dtype=np.int64)
    b = np.asarray([u[2] for u in units], dtype=np.int64)
    pub = public_vectors(R.p, None, a, b, None, device=R.device)
    c_in, y_out = np.asarray([u[0] for u in units], dtype=np.int64), np.asarray([u[3] for u in units], dtype=np.int64)
    hints = hints_torch(R.p, R.sys, c_in, a, b, y_out, R.device)
    coins = protocol_mod.Coins.sample() if R.interactive else None
    proof = prove(R.p, R.sys, R.tb, R.cfg(lay.l), pub, hints, R.device, check=False, chain=(lay, y16),
                  words={"a": a, "b": b}, coins=coins, auth=auth_stmt, auth_open=auth_open)
    return {"proof": proof, "pubs": (a, b, y16)}


def dump_artifacts(dump_dir: Path | None, out: str | None) -> list[str]:
    """The result's ``artifacts`` (research/result/v0.1: paths relative to the run directory, which the store preserves as the
    attempt's ``run-files/v1`` tree): the ``--dump-dir`` tree when it lies inside the run directory (``$RESEARCH_RUN_DIR``, else
    the directory ``--out`` is written to), so the proofs, statements, coins and system.bin travel with the result.  No dump, a
    dump elsewhere, or the run directory itself gives ``[]``."""
    if dump_dir is None:
        return []
    run_dir = os.environ.get("RESEARCH_RUN_DIR") or (Path(out).parent if out else None)
    if run_dir is None:
        return []
    try:
        rel = Path(dump_dir).resolve().relative_to(Path(run_dir).resolve())
    except ValueError:
        return []
    return [rel.as_posix()] if rel.parts else []


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
            # ligero-proof/v1 + ligero-statement/v1 (archive/ligero-verify reads them; serialize.py documents them)
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
    # VU i of the batch is tier instance i % vu.n: beyond the tier's 4096 ids the frozen instances are recycled (the
    # fingerprint says so: B_real = vu.n, instances.range = [0, vu.n)).  The bench has no synthetic VU generator.
    recycled = total_vus > vu.n
    lay = R.layout(per_proof)
    l = lay.l
    cfg = R.cfg(l)
    print(f"config {cfg}; layout {lay}; zk {cfg.zk}, mode {cfg.mode}; system {R.sys.summary()}; {total_vus} VUs as {n_proofs} sub-batches of <= {per_proof}"
          + (f"; tier ids recycled mod {vu.n}" if recycled else ""))
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    keys = None
    # --dump-dir: every RECORDED rep's sub-batch proofs (the first --dump-reps of them, default all) as
    # ligero-statement/v2 + ligero-proof/v1 files (+ the runner's step-0 coins, 128 raw bytes, interactive mode) for the
    # independent verifier (archive/ligero-verify); written after the verifier timer, outside every measured interval.
    # system.bin + manifest.json once.
    dump_dir = Path(os.path.expandvars(args.dump_dir)) if getattr(args, "dump_dir", None) else None   # $RESEARCH_RUN_DIR/proofs, as the other runners
    dump_reps = getattr(args, "dump_reps", None)
    dump_reps = args.reps if dump_reps is None else min(dump_reps, args.reps)
    dump_manifest = None
    with_auth = bool(getattr(args, "auth", False))
    if dump_dir is not None:
        from .serialize import (
            _manifest,
            _statement_of,
            _write_system,
            proof_bytes,
            sha256_file,
            statement_bytes,
        )
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_manifest = _manifest(dump_dir, args, R, auth=with_auth)
        _write_system(dump_dir, R, dump_manifest)
        dump_manifest["set"] = {"total_vus": total_vus, "batch": args.batch, "per_proof": per_proof, "n_proofs": n_proofs,
                                "zk": args.zk, "mode": args.mode, "reps": args.reps, "dump_reps": dump_reps,
                                "tier_n": vu.n, "ids_recycled_mod_tier_n": recycled,
                                "authentication": "included" if with_auth else "excluded"}
    target_bits = float(getattr(args, "target_bits", 128.0) or 128.0)   # the VERIFIER's parameter (PROTOCOL.md 8c)
    batch = None
    # authentication = included: the frozen set's words [0, total_vus) as three committed word trees (auth.py); the
    # negatives on the real trees must all be rejected BEFORE anything is measured
    trees = auth_info = None
    auth_neg_report = None
    if with_auth:
        trees, auth_info = auth_trees_for(vu, root, total_vus, getattr(args, "auth_cache", None), getattr(args, "auth_procs", None))
        print(f"auth trees: {'loaded' if auth_info['cached'] else 'built'} in {auth_info['commit_seconds']:.2f}s; roots "
              + ", ".join(f"{n}={t['root'][:16]}.. (count {t['count']}, depth {t['depth']})" for n, t in auth_info["trees"].items()), flush=True)
        negs = auth_negatives(R, trees, vu, n_vus=2, procs=getattr(args, "auth_procs", None))
        auth_neg_report = {"cases": [{"name": n, **m} for n, _pf, _pubs, _lay, _st, m in negs]}
        auth_neg_report["total"] = sum(1 for c in auth_neg_report["cases"] if not c["expected_accepted"])
        auth_neg_report["rejected"] = sum(1 for c in auth_neg_report["cases"] if not c["expected_accepted"] and not c["python_accepted"])
        auth_neg_report["honest_accepted"] = all(c["python_accepted"] for c in auth_neg_report["cases"] if c["expected_accepted"])
        for c in auth_neg_report["cases"]:
            print(f"auth {'negative' if not c['expected_accepted'] else 'honest'} {c['name']}: python {c['python_accepted']} ({c['python_reason']})", flush=True)
        if auth_neg_report["rejected"] != auth_neg_report["total"] or not auth_neg_report["honest_accepted"]:
            raise SystemExit("authentication negatives not all rejected (or the honest included proof rejected) -- refusing to bench")
        if dump_dir is not None:
            from .serialize import (
                auth_mutations,
                read_proof,
                read_statement,
                verify_files,
            )
            nd = dump_dir / "auth_negatives"
            nd.mkdir(exist_ok=True)
            dump_manifest["auth_negatives"] = []
            for name, pf, pubs, lay_n, st_n, meta in negs:
                st = _statement_of(R, pubs, lay_n, R.cfg(lay_n.l), 1, auth=st_n)
                (nd / f"{name}.stmt").write_bytes(statement_bytes(st))
                (nd / f"{name}.proof").write_bytes(proof_bytes(pf))
                entry = {"kind": "honest" if meta["expected_accepted"] else "auth_negative", "name": name,
                         "stmt": f"auth_negatives/{name}.stmt", "proof": f"auth_negatives/{name}.proof",
                         "proof_sha256": sha256_file(nd / f"{name}.proof"), "expected_accepted": meta["expected_accepted"],
                         "python_verdict": {"accepted": meta["python_accepted"], "reason": meta["python_reason"]}, "mutation": meta.get("mutation")}
                if pf.coins is not None:
                    (nd / f"{name}.coins").write_bytes(pf.coins.r1 + pf.coins.s1 + pf.coins.r2 + pf.coins.s2)
                    entry["coins"] = f"auth_negatives/{name}.coins"
                dump_manifest["auth_negatives"].append(entry)
            st_h = read_statement((nd / "auth_honest.stmt").read_bytes())
            pf_h = read_proof((nd / "auth_honest.proof").read_bytes())
            for name, p2, st2 in auth_mutations(pf_h, st_h):
                (nd / f"mut_{name}.stmt").write_bytes(statement_bytes(st2))
                (nd / f"mut_{name}.proof").write_bytes(proof_bytes(p2))
                ok_m, why_m, _s = verify_files(nd / f"mut_{name}.stmt", nd / f"mut_{name}.proof", args.device, runner=R)
                entry = {"kind": "auth_mutation", "name": f"mut_{name}", "stmt": f"auth_negatives/mut_{name}.stmt",
                         "proof": f"auth_negatives/mut_{name}.proof", "expected_accepted": False,
                         "python_verdict": {"accepted": ok_m, "reason": why_m}, "mutation": name}
                if pf_h.coins is not None:
                    entry["coins"] = "auth_negatives/auth_honest.coins"
                dump_manifest["auth_negatives"].append(entry)
                print(f"auth mutation {name}: python {ok_m} ({why_m})", flush=True)
                if ok_m:
                    raise SystemExit(f"authentication file mutation {name} ACCEPTED -- refusing to bench")
            auth_neg_report["file_mutations"] = sum(1 for e in dump_manifest["auth_negatives"] if e["kind"] == "auth_mutation")
            auth_neg_report["file_mutations_rejected"] = sum(1 for e in dump_manifest["auth_negatives"]
                                                             if e["kind"] == "auth_mutation" and not e["python_verdict"]["accepted"])
    # --verifier tcp://host:port: the step-0 coins of every RECORDED rep come from a LIVE verifier on another machine
    # (live.py) and every prover message is sent to it as it is produced -- the protocol is unchanged (8c), only the coin
    # source is; the warm-up rep keeps local coins (it presents one sub-batch, not n_proofs).
    live = None
    # --pipeline N (lane live-2, the Ampere BF16 cell path): the sub-batches proved N-deep on their own CUDA streams
    # (``ChainRunner.prove_arrays_many`` -> ``pipeline.prove_many``), as ``relchain.bench_vu_rel`` does for the registered
    # relations; with --verifier the coin exchanges are non-blocking (``live.LiveCoins`` futures) and the verifier commits
    # N sub-batches ahead (HELLO window).  0/1 = the sequential schedule.
    pipeline_depth = int(getattr(args, "pipeline", 0) or 0) if args.device.startswith("cuda") else 0
    if pipeline_depth > 1 and R.hints_mode != "torch":
        raise SystemExit("--pipeline needs the torch hint generator")
    if getattr(args, "verifier", None):
        from .live import BenchLive
        live = BenchLive(args, R, n_proofs, total_vus, "bf16-ampere", interactive)
        if pipeline_depth > 1:
            live.window = pipeline_depth                 # the verifier commits N sub-batches ahead (N openings in flight)
    local_coins = lambda i: (protocol_mod.Coins.sample() if interactive else None)   # noqa: E731  the honest step 0
    for rep in range(args.reps + 1):
        agg: dict[str, float] = {}
        vsum = 0.0
        vauth = 0.0
        t_pass = time.perf_counter()
        # Prove every sub-batch of the rep first, then verify them: the verifier's host work (with authentication, ~1M
        # SHA-256 per proof across a process pool) must not sit between two timed prover sub-batches -- interleaved, the
        # prover's `tests` lap ran 3x slower on the A100 pod after each verification while every device lap was unchanged.
        proved = []
        cfgs_seen: list[Config] = []
        if live is not None:
            live.begin_rep(rep)
        if pipeline_depth > 1 and rep > 0:
            lays = [Layout(l, hi - lo, STEPS) for lo, hi in subs]
            arrays_all = [marshal_ids(vu, [i % vu.n for i in range(lo, hi)], lay_s) for (lo, hi), lay_s in zip(subs, lays)]
            auths = [(trees, list(range(lo, hi))) for lo, hi in subs] if with_auth else None
            factory = live.pipeline_factory(local_coins) if live is not None else local_coins
            many, _wall = R.prove_arrays_many(arrays_all, lays, factory, depth=pipeline_depth, auths=auths)
            del arrays_all
            for si, ((lo, hi), (V, proof, pubs, lay_i, st_auth)) in enumerate(zip(subs, many)):
                if live is not None:
                    V = live.proved(si, proof, V)            # PROOF si on the data channel (async); coins -> plain Coins
                for k, v in proof.timings.items():
                    agg[k] = agg.get(k, 0.0) + v
                proved.append((si, lo, hi, V, proof, pubs, lay_i, st_auth))
                cfgs_seen.append(R.cfg(lay_i.l))
        for si, (lo, hi) in enumerate(subs if not proved else []):
            lay_s = Layout(l, hi - lo, STEPS)
            arrays = marshal_ids(vu, [i % vu.n for i in range(lo, hi)], lay_s)     # loader work, outside the prover timer
            V = protocol_mod.Coins.sample() if interactive else None      # the verifier's step 0 (interactive only)
            if live is not None:
                V = live.coins(si, V)                        # a recorded rep: the live verifier's step 0 for sub-batch si
            st_auth = None
            if with_auth:
                proof, pubs, lay_i, st_auth = R.prove_arrays(*arrays, lay_s, check=False, coins=V, auth=(trees, list(range(lo, hi))))
            else:
                proof, pubs, lay_i = R.prove_arrays(*arrays, lay_s, check=False, coins=V)
            if live is not None:
                V = live.proved(si, proof, V)                # PROOF si on the data channel (async); waits out of the clocks
            for k, v in proof.timings.items():
                agg[k] = agg.get(k, 0.0) + v
            proved.append((si, lo, hi, V, proof, pubs, lay_i, st_auth))
            cfgs_seen.append(R.cfg(lay_i.l))
            if rep == 0:
                if pipeline_depth > 1:
                    # every pipeline slot once (its stream's graph capture, pinned buffers): untimed, like the warm-up
                    warm = subs[:pipeline_depth] + (subs[-1:] if len(subs) > pipeline_depth else [])   # + the ragged last layout
                    w_lays = [Layout(l, hi_ - lo_, STEPS) for lo_, hi_ in warm]
                    w_arrays = [marshal_ids(vu, [i % vu.n for i in range(lo_, hi_)], ly) for (lo_, hi_), ly in zip(warm, w_lays)]
                    R.prove_arrays_many(w_arrays, w_lays, local_coins, depth=pipeline_depth,
                                        auths=([(trees, list(range(lo_, hi_))) for lo_, hi_ in warm] if with_auth else None))
                    del w_arrays
                break                                        # warm-up: one sub-batch (graph capture, compile)
        if live is not None:
            live.end_rep(rep, agg)                           # END; the verifier's session verdict (waits for its Rust verify)
        for si, lo, hi, V, proof, pubs, lay_i, st_auth in proved:
            tm_auth: dict = {}
            t0 = time.perf_counter()
            ok, why = R.verify_vus(proof, pubs, lay_i, coins=V, auth=st_auth, auth_procs=getattr(args, "auth_procs", None),
                                   auth_timing=tm_auth)
            vsum += time.perf_counter() - t0
            vauth += tm_auth.get("seconds", 0.0)
            assert ok, f"sub-batch [{lo}, {hi}): {why}"
            if dump_dir is not None and 0 < rep <= dump_reps:
                rd = dump_dir / f"rep{rep}"
                rd.mkdir(exist_ok=True)
                stem = f"sub_{si:02d}"
                st = _statement_of(R, pubs, lay_i, R.cfg(lay_i.l), n_proofs, auth=st_auth)
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
        if rep > 0:
            # the batch verdict (PROTOCOL.md 8c, "Batch verification"): every sub-batch verified above; now the union
            # bound of this verifier's own per-proof bounds over the n_proofs presented, and each proof's n_proofs
            # (what config_for sized t for) against that count.  Arithmetic on n_proofs numbers, inside the verifier time.
            t0 = time.perf_counter()
            batch = R.batch_bound(cfgs_seen, [n_proofs] * len(cfgs_seen), target_bits)
            vsum += time.perf_counter() - t0
            assert batch["accepted"], batch["reason"]
        wall = time.perf_counter() - t_pass
        if rep == 0:
            print("warm-up (one sub-batch):", {k: round(v, 3) for k, v in agg.items()})
            continue
        keys = agg.keys()
        runs.append((agg, vsum, vauth))
        print(f"rep {rep}: prover {agg['total']:.3f}s (+hints {agg['hints_host']:.3f}s) over {n_proofs} proofs; "
              f"verifier {vsum:.3f}s" + (f" (auth {vauth:.3f}s)" if with_auth else "") +
              f"; pass wall {wall:.3f}s ({(agg['total'] + agg['hints_host']) / total_vus * 1e3:.3f} ms/VU)")
    if dump_dir is not None:
        (dump_dir / "manifest.json").write_text(json.dumps(dump_manifest, indent=1, default=str))
        print(f"dumped {len(dump_manifest['files'])} proofs to {dump_dir}", flush=True)
    # t.total and every bucket come from one rep (phases.median_rep), as relchain.bench_vu_rel
    med_rep = phases.median_rep(runs)
    med = dict(runs[med_rep][0])
    vmed = statistics.median(r[1] for r in runs)
    vauth_med = statistics.median(r[2] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = proof.n_bytes()
    nbytes = {k: v * n_proofs for k, v in nbytes.items()}
    snd = soundness(cfg, R.tb.L, R.tb.Q, n_proofs)
    # security.achieved_log2 (the contract's field; N_subbatches and B_proof are its siblings) is the BATCH bound
    # -log2 sum_i 2^b_i as the verifier's batch_bound computed it; achieved_bits is the same number as positive bits.
    assert batch is not None and abs(batch["batch_log2"] - snd["total_log2"]) < 1e-6, (batch, snd["total_log2"])
    total = med["total"] + med["hints_host"]
    n_vus = total_vus
    per_vu = total / n_vus
    hw = contract.probe_hardware()
    sw = {"backend": {"name": "b-ligero chain (archive/direct/ligero)", "commit": _git_commit(), "hash": proof.hash_name,
                      "challenge_expander": "SHAKE-256 (placeholder)", "pipeline": pipeline_depth},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    auth_note = ("; the operands and words are published AND authenticated: each is a leaf the statement's roots commit to "
                 "(auth.py), so the ZK claim is about the construction, the privacy of the words is not claimed" if with_auth else "")
    ext = "" if with_auth else "external authentication"
    if cfg.zk and interactive and not args.hvzk_label:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", ZK_MODE + ZK_MODE_HM96 + ZK_VACUOUS_NOTE + auth_note
        omitted = (ext + "; " if ext else "") + "unopened columns hidden by Blake3 of a uniform column (random-oracle hiding, not HM96 salted leaves)"
    elif cfg.zk and interactive:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_VACUOUS_NOTE + auth_note
        omitted = (ext + " " if ext else "") + "(malicious-verifier ZK holds but the conservative label was requested)"
    elif cfg.zk:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_MODE_FS + ZK_VACUOUS_NOTE + auth_note
        omitted = (ext + "; " if ext else "") + "malicious-verifier ZK outside the programmable random-oracle model (Fiat-Shamir mode: NIZK in PROM only)"
    else:
        proof_class, zk_mode, omitted = "NON_ZK_PROOF_DIAGNOSTIC", "none", (ext + ", " if ext else "") + "ZK masking"
    if with_auth:
        sw["authentication"] = {k: v for k, v in auth_info.items() if k not in ("commit_seconds", "cached", "procs")}
        sw["authentication"]["multiproof_bytes_per_proof"] = int(proof.n_bytes()["auth_multiproofs"])
        sw["authentication"]["multiproof_siblings_per_proof"] = [len(s) for s in proof.auth]
        sw["authentication"]["opened_leaves_per_proof"] = {"a": per_proof * STEP * STEPS, "b": per_proof * STEP * STEPS, "y": per_proof}
        sw["authentication"]["verifier"] = ("python: protocol.verify -> auth.verify_auth (leaf digests + verity.ml.commitments.verify_multiproof, "
                                            f"{auth_info['procs']} processes for the leaves); rust: archive/ligero-verify (sha256.rs, auth fold)")
    preprocessing = f"constraint compilation (a function of Params only, {R.compile_seconds:.2f}s)"
    if with_auth:
        preprocessing += (f"; object commitment: three SHA-256 word trees (a: activations, b: weights, y: outputs) over the frozen "
                          f"set's VUs [0, {total_vus}) -- {auth_info['trees']['a']['count']} + {auth_info['trees']['b']['count']} + "
                          f"{auth_info['trees']['y']['count']} leaves, built once per object in {auth_info['commit_seconds']:.2f}s "
                          f"({auth_info['procs']} processes{', loaded from cache' if auth_info['cached'] else ''}); the roots are "
                          f"public inputs; the per-proof multiproof openings are prover work (t.serialization)")
    fp = contract.fingerprint(
        profile=contract.PROFILE, domain=contract.DOMAIN, K=STEP * STEPS, B=n_vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication="included" if with_auth else "excluded", zk_mode=zk_mode,
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_k": cfg.k, "rs_l": cfg.l,
                  "zk": cfg.zk, "t_pad": cfg.t_pad, "mode": cfg.mode, "bound_model": snd["bound_model"],
                  "hash_assumption": snd["hash_assumption"], "zk_statement": snd["zk_statement"],
                  "fiat_shamir_queries_log2": snd["fiat_shamir_queries_log2"],
                  "opened_columns": cfg.t, "achieved_log2": batch["batch_log2"], "achieved_bits": -batch["batch_log2"],
                  "verifier_target_bits": target_bits, "terms_log2": snd["terms_log2"],
                  "per_proof_target_log2": snd["per_proof_target_log2"], "per_proof_total_log2": snd["per_proof_total_log2"],
                  "union_over": n_proofs, "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing=preprocessing,
        retention="retained",
        instances=contract.instances_ref("vu-k1536", 0, min(n_vus, vu.n), instances.manifest_sha256(root)),
        **({"B_real": vu.n} if recycled else {}),
        scope="vu", omitted=omitted,
        rows=R.sys.m, committed_rows=n_rows(cfg, R.sys.m, True), linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        witness_generation=f"{R.hints_mode} hint generator (gen_step semantics) + torch program (derived rows)",
        B_proof=per_proof,
        note=f"{total_vus} VUs as {n_proofs} independently committed proofs of <= {per_proof} VUs each (l = {l} columns, "
             f"{l - per_proof * STEPS} pad columns per proof); prover time = sum over the proofs; soundness = union bound"
             + (f"; VU i of the batch is tier instance i mod {vu.n} (the frozen set has {vu.n} ids, so B_real = {vu.n} distinct "
                f"instances are proved {total_vus / vu.n:.1f}x over -- the bench has no synthetic VU generator)" if recycled else ""),
    )
    meas = [
        *(contract.measurement(name, v) for name, v in phases.buckets(med, serialization_extra=("auth_openings",)).items()),
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
    if with_auth:
        meas += [
            contract.measurement("auth.multiproof_bytes", nbytes["auth_multiproofs"], "B"),
            contract.measurement("auth.opened_leaves", 2 * n_vus * STEP * STEPS + n_vus, "count"),
            contract.measurement("auth.openings_seconds", med.get("auth_openings", 0.0), "s"),   # prover: in t.serialization
            contract.measurement("auth.verifier_seconds", vauth_med, "s"),                       # verifier: in verifier.seconds
            contract.measurement("auth.commit_seconds", auth_info["commit_seconds"], "s"),       # once per object (preprocessing)
            contract.measurement("auth.negatives_rejected", auth_neg_report["rejected"], "count"),
            contract.measurement("auth.negatives_total", auth_neg_report["total"], "count"),
        ]
    validation = {"status": "passed", "checker": "archive.direct.ligero.run bench-vu (verifier accepted every sub-batch and the batch "
                                                 "union bound every rep)",
                  "detail": f"{args.reps} reps, {n_vus} VUs as {n_proofs} proofs at l={l}; batch {batch['reason']}",
                  "evidence": {"soundness": snd, "batch": batch,
                               "reps": [{"prover_total": r[0]["total"] + r[0]["hints_host"], "verifier": r[1]} for r in runs],
                               "phase_rep": {"index": med_rep, "rule": "t.total and the t.* buckets are this rep's (the median "
                                                                       "prover_total of reps); verifier.seconds is the median over the reps"}}}
    if with_auth:
        validation["checker"] += " + auth.verify_auth on every sub-batch (multiproofs of every boundary word against the roots)"
        validation["detail"] += (f"; authentication negatives: {auth_neg_report['rejected']}/{auth_neg_report['total']} prover-side "
                                 f"rejected" + (f", {auth_neg_report['file_mutations_rejected']}/{auth_neg_report['file_mutations']} "
                                                f"file-level rejected" if "file_mutations" in auth_neg_report else "") +
                                 ", honest included proof accepted")
        validation["evidence"]["auth_negatives"] = auth_neg_report
        validation["evidence"]["auth_roots"] = {n: t["root"] for n, t in auth_info["trees"].items()}
    if live is not None:
        # t.total_live / net.* / verify.* (lane live-verifier; additive).  The live verifier's verdicts are reported as the
        # PROVER received them; the independent record (`verified=` label) is the verifier's own dump + verdict artifact.
        lmeas, lev, ldetail, lok = live.fields(contract, total, med)
        meas += lmeas
        validation["checker"] += f"; live verifier at {args.verifier}"
        validation["detail"] += ldetail
        validation["evidence"]["live_verifier"] = lev
        if not lok:
            validation["status"] = "failed"
    result = {"schema": contract.RESULT_SCHEMA,
              "run_id": args.run_id or os.environ.get("RESEARCH_RUN_ID") or f"b-ligero-vu-{int(time.time())}",   # as relchain.py
              "workload_fingerprint": fp, "validation": validation,
              "measurements": meas, "measurement_files": [], "artifacts": dump_artifacts(dump_dir, args.out)}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"seconds_per_vu": per_vu, "prover_total": total, "verifier": vmed, "bytes": nbytes, "n_proofs": n_proofs,
                      "per_proof_vus": per_proof, "t": cfg.t, "zk": cfg.zk, "k": cfg.k, "peak_device_bytes": peak_dev, "soundness_log2": snd["total_log2"],
                      "per_proof_log2": snd["per_proof_total_log2"], "batch_bits": -batch["batch_log2"], "target_bits": target_bits,
                      "batch_reason": batch["reason"], "medians": med}, indent=1))
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(result, indent=1, default=str))   # $RESEARCH_RUN_DIR/result.json
    return 0
