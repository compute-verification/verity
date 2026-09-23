"""End-to-end Ligerito prover / verifier for one batch of a registered relation (lane ligerito-relation).

    witness (Ligero's generator, relchain's runner: marshal -> hints -> build_witness; nothing of Ligero's proof runs)
    -> ``layout.build_z`` -> commit ``w`` (``z`` with the virtual rows zeroed, row-major, ``proof.py``)
    -> ``sumcheck.prove`` (zero-check + rows + shift) -> 1 + n_links evaluation claims on ``w``
    -> ONE Ligerito evaluation proof for all of them (``pcs_open``: the claims batched by a coin ``beta``).

The transcript order is in ``proof.py``.  ``pcs_commit`` / ``pcs_open`` / ``pcs_verify`` are the commit / multi-claim-open
split of ``proto/pcs.prove`` / ``verify``, here until lane ligerito-pcs-fast (owner of ``pcs.py``) exposes
``commit(f, dims)`` / ``open(state, claims, coins)`` (handoff 20260923T1850Z): they call proto's encoder, Merkle tree,
contractions and claim algebra unchanged; the only protocol change is J eq terms with coefficients ``beta_j`` in place of
proto's single eq term with coefficient 1.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from backends.direct.ligero import protocol as protocol_mod
from backends.direct.ligero.compile import P
from backends.direct.ligero.relations import relation as get_relation
from backends.direct.ligero.relchain import RelationChainRunner
from backends.direct.ligero.witness import build_witness

from . import params as params_mod
from . import sumcheck as sc_mod
from .layout import Constraints, Layout, SubBatch, build_z, constraints, layout_for, public_rows
from .proof import Key, PcsProof, Proof, Statement, params_bytes
from .proto import claims as cl
from .proto import ext, merkle_multi
from .proto._ligero import merkle as ligero_merkle
from .proto.dims import Dims
from .proto.kernels import col_contract, row_contract
from .proto.linalg import _powers_rows, contract_columns_ext, contract_rows_ext, eq_planes, matmul_mod_p_np, powers_matrix
from .proto.ntt_gpu import Encoder, split
from .proto.pcs import Timer, _commit, _ext_bytes, _HostTree, _open_rows, _row_claims
from .transcript import FiatShamirCoins, LocalCoins

D = 6
#: parameter set B (ligerito-design FINAL) at the pow2 shape: level-1 rate 1/2, later levels 1/4, 2^-128 union bound
SET_B_RATES = (1, 2, 2, 2, 2)

__all__ = ["Committed", "Prover", "Reject", "dims_for", "make_coins", "pcs_commit", "pcs_open", "pcs_verify"]


class Reject(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Parameters and coins
# ---------------------------------------------------------------------------------------------------------------------

def dims_for(n: int, rel_name: str | None = None, L: int | None = None) -> tuple[Dims, params_mod.LigeritoParams]:
    """Parameter set B for ``N = 2^n`` cells: L = 5 at batch scale (n >= 26), fewer levels on the gate's small batches;
    |S_i| allocated by ``params.allocate_queries`` for the 2^-128 union bound."""
    shape = params_mod.RELATIONS.get(rel_name) if rel_name else None
    for L_ in ([L] if L is not None else range(5, 1, -1)):
        try:
            p = params_mod.optimize(1 << n, L_, rate_log2=SET_B_RATES[:L_], relation=shape, zk=False, min_wide_log2=3)
        except ValueError:
            continue
        return Dims(n, list(p.wide_log2), list(p.queries), list(p.rate_log2)), p
    raise ValueError(f"no parameter-set-B dims for N = 2^{n} (the queries do not fit the codewords)")


def make_coins(kind: str, seed: int = 0, slots=None):
    if kind == "fiat-shamir":
        return FiatShamirCoins()
    if kind == "local":
        return LocalCoins(seed)
    if kind == "live":
        from .transcript import LiveCoins
        return LiveCoins(slots)
    raise ValueError(f"coins {kind!r}")


# ---------------------------------------------------------------------------------------------------------------------
# PCS: commit / multi-claim open / verify (proto/pcs.py split; see the module docstring)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Committed:
    dims: Dims
    M1: torch.Tensor                    # (C1, R1) int32 view of f
    U1: torch.Tensor                    # (C1, n_1) codeword
    tree1: object
    enc1: Encoder
    root1: bytes


def _gpu(device) -> bool:
    return torch.device(device).type == "cuda" and ligero_merkle._gpu_merkle is not None


def _tree(U: torch.Tensor):
    if _gpu(U.device):
        return _commit(U)
    t = _HostTree(U)
    return t.root, t


def _open(U: torch.Tensor, tree, S: list[int], depth: int) -> tuple[np.ndarray, bytes]:
    if _gpu(U.device):
        op = _open_rows(U, tree, S, depth)
        return op.rows, op.siblings
    idx = torch.as_tensor(np.asarray(S, dtype=np.int64))
    return U[:, idx].t().contiguous().to(torch.int64).numpy(), tree.gather(S, depth)


def pcs_commit(f: torch.Tensor, dims: Dims, tm: Timer) -> Committed:
    """Round 1 of proto's ``prove``: encode the columns of ``M1 = f.view(C1, R1)`` and Merkle-commit the codeword rows."""
    ks, kps, rates = dims.k, dims.kprime, dims.rates
    R1, C1 = 1 << ks[0], 1 << kps[0]
    assert f.numel() == 1 << dims.n and f.dtype == torch.int32
    M1 = f.view(C1, R1)
    enc1 = Encoder(R1, f.device, rates[0])
    tm.start("commit_r1_alloc")
    U1 = torch.empty((C1, enc1.n), dtype=torch.int32, device=f.device)
    tm.stop()
    tm.start("commit_r1_encode")
    U1 = enc1.encode(M1, out=U1)
    tm.stop()
    tm.start("commit_r1_merkle")
    root1, tree1 = _tree(U1)
    tm.stop()
    return Committed(dims, M1, U1, tree1, enc1, root1)


def pcs_open(com: Committed, points: np.ndarray, values: np.ndarray, coins, tm: Timer) -> PcsProof:
    """The evaluation proof of ``w~(points[j]) = values[j]`` for all ``j`` (points in PCS variable order, (J, n, 6))."""
    dims, device = com.dims, com.M1.device
    n, ks, kps, qs, L, rates = dims.n, dims.k, dims.kprime, dims.queries, dims.rounds, dims.rates
    J = points.shape[0]
    assert points.shape == (J, n, D) and values.shape == (J, D)
    tr = coins
    tr.absorb(b"z", _ext_bytes(points))
    beta = tr.challenge(b"beta", J)
    roots, sumcheck, rows_out, sibs = [com.root1], [], [], []
    Us, trees, encs = [com.U1], [com.tree1], [com.enc1]
    # ---------------- round 1: J eq terms
    tm.start("sumcheck_r1_contract")
    terms = [cl.Term(beta[j] % P, "eq", z=points[j] % P, nvars=n) for j in range(J)]
    us = []
    for j in range(J):
        zrow = eq_planes(points[j][:ks[0]], device)                     # (6, R1)
        us.append(col_contract(com.M1, zrow))                            # (C1, 6)
        del zrow
    tm.stop()
    tm.start("sumcheck_host")
    sc = cl.ColumnSumcheckProver(terms, us, ks[0], kps[0])
    msgs = []
    for _ in range(kps[0]):
        g = sc.round_message()
        tr.absorb(b"g", _ext_bytes(g))
        r = tr.challenge(b"r", 1)[0]
        sc.fold(r)
        msgs.append(g)
    sumcheck.append(msgs)
    r1 = sc.challenges()
    rbar = cl.rbar(r1)
    terms = cl.restrict_terms(terms, ks[0], r1)
    running = sc.claim()
    tm.stop()
    tm.start("sumcheck_r1_fold")
    y = row_contract(com.M1, rbar)                                       # (6, R1) int32 planes
    tm.stop()
    rbars = [rbar]
    # ---------------- rounds 2..L (proto's loop; several eq terms in the contraction)
    for i in range(1, L):
        R, C = 1 << ks[i], 1 << kps[i]
        Mi = y.view(D, C, R)
        tm.start("recursion_encode")
        enc = Encoder(R, device, rates[i])
        U = enc.encode(Mi.reshape(D * C, R))
        tm.stop()
        tm.start("recursion_merkle")
        root, tree = _tree(U)
        tm.stop()
        roots.append(root)
        Us.append(U)
        trees.append(tree)
        encs.append(enc)
        tr.absorb(b"root", root)
        tm.start("serialize_openings")
        prev = i - 1
        n_prev = encs[prev].n
        S = tr.indices(b"S", n_prev, qs[prev])
        rows, sib = _open(Us[prev], trees[prev], S, n_prev.bit_length() - 1)
        rows_out.append(rows)
        sibs.append(sib)
        tr.absorb(b"rows", np.ascontiguousarray(rows.astype(np.uint32)).tobytes())
        tr.absorb(b"sib", sib)
        tm.stop()
        tm.start("sumcheck_host")
        etas = encs[prev].etas(S)
        rhs = _row_claims(rows, rbars[prev], base=(prev == 0))
        new_terms = [cl.Term(ext.one(), "geom", eta=int(e), nvars=ks[prev]) for e in etas]
        alpha = tr.challenge(b"alpha", 1 + len(new_terms))
        for t in terms:
            t.coef = ext.mul(t.coef, alpha[0])
        for j, t in enumerate(new_terms):
            t.coef = alpha[1 + j]
        running = ext.add(ext.mul(alpha[0], running), ext.sum_ext(ext.mul(alpha[1:], rhs), axis=0))
        terms = terms + new_terms
        tm.stop()
        tm.start("recursion_contract")
        geom = [t for t in terms if t.kind == "geom"]
        eqs = [t for t in terms if t.kind == "eq"]
        G = powers_matrix(np.array([t.eta for t in geom], dtype=np.int64), R, device) if geom else None
        u_geom, u_eq0 = contract_columns_ext(Mi, G, eq_planes(eqs[0].z[:ks[i]], device))
        u_eqs = [u_eq0] + [contract_columns_ext(Mi, None, eq_planes(t.z[:ks[i]], device))[1] for t in eqs[1:]]
        tm.stop()
        tm.start("sumcheck_host")
        us, gi, ei = [], 0, 0
        for t in terms:
            if t.kind == "geom":
                us.append(u_geom[gi])
                gi += 1
            else:
                us.append(u_eqs[ei])
                ei += 1
        sc = cl.ColumnSumcheckProver(terms, us, ks[i], kps[i])
        msgs = []
        for _ in range(kps[i]):
            g = sc.round_message()
            tr.absorb(b"g", _ext_bytes(g))
            r = tr.challenge(b"r", 1)[0]
            sc.fold(r)
            msgs.append(g)
        sumcheck.append(msgs)
        ri = sc.challenges()
        rbar = cl.rbar(ri)
        terms = cl.restrict_terms(terms, ks[i], ri)
        running = sc.claim()
        rbars.append(rbar)
        tm.stop()
        tm.start("recursion_fold")
        y = contract_rows_ext(Mi, rbar)
        tm.stop()
    # ---------------- final vector, open U_L
    tm.start("serialize_final")
    final_y = y.to(torch.int64).t().contiguous().cpu().numpy() % P
    tr.absorb(b"y", _ext_bytes(final_y))
    n_last = encs[-1].n
    S = tr.indices(b"S", n_last, qs[-1])
    rows, sib = _open(Us[-1], trees[-1], S, n_last.bit_length() - 1)
    rows_out.append(rows)
    sibs.append(sib)
    tm.stop()
    return PcsProof(roots, sumcheck, rows_out, sibs, final_y)


class _Map:
    """The verifier's codeword position -> evaluation point map of one round (proto's 4-step split)."""

    def __init__(self, R: int, split_: tuple[int, int], rate: int):
        self.n = R << rate
        self.n1, self.n2 = split_
        if self.n1 * self.n2 != self.n:
            raise Reject("bad encoder split")

    def etas(self, s):
        return Encoder.etas(self, s)

    def eta_index(self, s):
        return Encoder.eta_index(self, s)


def pcs_verify(pf: PcsProof, dims: Dims, splits: list, points: np.ndarray, values: np.ndarray, coins) -> None:
    """proto's ``_verify`` from round 1 on, with J eq terms batched by ``beta``; raises ``Reject``.  The caller has
    absorbed root_1 (before the sumcheck)."""
    n, ks, kps, qs, L = dims.n, dims.k, dims.kprime, dims.queries, dims.rounds
    J = points.shape[0]
    if points.shape != (J, n, D) or values.shape != (J, D):
        raise Reject("bad claim shape")
    if len(pf.roots) != L or len(pf.rows) != L or len(pf.siblings) != L or len(pf.sumcheck) != L or len(splits) != L:
        raise Reject("bad proof shape")
    maps = [_Map(1 << ks[i], tuple(splits[i]), dims.rates[i]) for i in range(L)]
    if any(m.n1 != split(m.n)[0] for m in maps):
        raise Reject("non-canonical encoder split")
    tr = coins
    tr.absorb(b"z", _ext_bytes(points))
    beta = tr.challenge(b"beta", J)
    running = ext.sum_ext(ext.mul(beta, values % P), axis=0)
    terms = [cl.Term(beta[j] % P, "eq", z=points[j] % P, nvars=n) for j in range(J)]
    rs = []
    for g in pf.sumcheck[0]:
        if g.shape != (3, D):
            raise Reject("bad sumcheck message")
        tr.absorb(b"g", _ext_bytes(g))
        rs.append(tr.challenge(b"r", 1)[0])
    ok, running = cl.verify_column_rounds(running, pf.sumcheck[0], rs)
    if not ok:
        raise Reject("PCS sumcheck round 1 failed")
    r1 = np.stack(rs[::-1], axis=0)
    rbars = [cl.rbar(r1)]
    terms = cl.restrict_terms(terms, ks[0], r1)
    for i in range(1, L):
        tr.absorb(b"root", pf.roots[i])
        prev = i - 1
        S = tr.indices(b"S", maps[prev].n, qs[prev])
        rows = pf.rows[prev]
        W = (1 << kps[prev]) * (1 if prev == 0 else D)
        if rows.shape != (qs[prev], W):
            raise Reject(f"opened rows of round {prev + 1} have shape {rows.shape}, expected {(qs[prev], W)}")
        if (rows < 0).any() or (rows >= P).any():
            raise Reject("opened row word out of range")
        tr.absorb(b"rows", np.ascontiguousarray(rows.astype(np.uint32)).tobytes())
        tr.absorb(b"sib", pf.siblings[prev])
        leaves = {s: merkle_multi.leaf_digest(rows[j]) for j, s in enumerate(S)}
        if not merkle_multi.verify(pf.roots[prev], leaves, pf.siblings[prev], maps[prev].n.bit_length() - 1):
            raise Reject(f"Merkle multi-path of round {prev + 1} failed")
        etas = maps[prev].etas(S)
        rhs = _row_claims(rows, rbars[prev], base=(prev == 0))
        new_terms = [cl.Term(ext.one(), "geom", eta=int(e), nvars=ks[prev]) for e in etas]
        alpha = tr.challenge(b"alpha", 1 + len(new_terms))
        for t in terms:
            t.coef = ext.mul(t.coef, alpha[0])
        for j, t in enumerate(new_terms):
            t.coef = alpha[1 + j]
        running = ext.add(ext.mul(alpha[0], running), ext.sum_ext(ext.mul(alpha[1:], rhs), axis=0))
        terms = terms + new_terms
        rs = []
        for g in pf.sumcheck[i]:
            if g.shape != (3, D):
                raise Reject("bad sumcheck message")
            tr.absorb(b"g", _ext_bytes(g))
            rs.append(tr.challenge(b"r", 1)[0])
        ok, running = cl.verify_column_rounds(running, pf.sumcheck[i], rs)
        if not ok:
            raise Reject(f"PCS sumcheck round {i + 1} failed")
        ri = np.stack(rs[::-1], axis=0)
        rbars.append(cl.rbar(ri))
        terms = cl.restrict_terms(terms, ks[i], ri)
    y = pf.final_y % P
    RL = 1 << ks[-1]
    if y.shape != (RL, D):
        raise Reject("bad final vector shape")
    tr.absorb(b"y", _ext_bytes(y))
    S = tr.indices(b"S", maps[-1].n, qs[-1])
    rows = pf.rows[-1]
    W = (1 << kps[-1]) * (1 if L == 1 else D)
    if rows.shape != (qs[-1], W):
        raise Reject("opened rows of the last round have a bad shape")
    if (rows < 0).any() or (rows >= P).any():
        raise Reject("opened row word out of range")
    leaves = {s: merkle_multi.leaf_digest(rows[j]) for j, s in enumerate(S)}
    if not merkle_multi.verify(pf.roots[-1], leaves, pf.siblings[-1], maps[-1].n.bit_length() - 1):
        raise Reject("Merkle multi-path of the last round failed")
    etas = maps[-1].etas(S)
    lhs = matmul_mod_p_np(_powers_rows(np.asarray(etas, dtype=np.int64), RL), y)
    if not np.array_equal(lhs, _row_claims(rows, rbars[-1], base=(L == 1))):
        raise Reject("final vector is inconsistent with the opened rows")
    geom = [t for t in terms if t.kind == "geom"]
    eqs = [t for t in terms if t.kind == "eq"]
    total = ext.zero()
    if geom:
        ip = matmul_mod_p_np(_powers_rows(np.array([t.eta for t in geom], dtype=np.int64), RL), y)
        total = ext.add(total, ext.sum_ext(ext.mul(np.stack([t.coef for t in geom], axis=0), ip), axis=0))
    for t in eqs:
        total = ext.add(total, ext.mul(t.coef, ext.sum_ext(ext.mul(t.rowpart_eq(ks[-1]), y), axis=0)))
    if not np.array_equal(total, running):
        raise Reject("final PCS claim failed")


# ---------------------------------------------------------------------------------------------------------------------
# The batch prover / verifier
# ---------------------------------------------------------------------------------------------------------------------

def pcs_points(lay: Layout, claims: list) -> tuple[np.ndarray, np.ndarray]:
    """The sumcheck's claims ``(p_i || p_c)`` -> PCS order ``(p_c || p_i)`` (row-major commitment), numpy."""
    pts = [torch.cat((c.point[lay.n_i:], c.point[:lay.n_i])).cpu().numpy() for c in claims]
    vals = [c.value.cpu().numpy() for c in claims]
    return np.stack(pts).astype(np.int64) % P, np.stack(vals).astype(np.int64) % P


@dataclass
class Batch:
    groups: list                        # per real sub-batch: the VU tuples
    lay: Layout
    stmt: Statement
    dims: Dims
    lparams: params_mod.LigeritoParams
    params: dict


@dataclass
class Result:
    proof: Proof
    timings: dict
    coins_rounds: int
    peak_device_bytes: int = 0
    extra: dict = field(default_factory=dict)


class Prover:
    """Owns the relation's compiled system, Ligero's witness machinery (``RelationChainRunner``: marshal, hints,
    ``build_witness`` -- its prover never runs), the layout constraints and the verification key."""

    def __init__(self, rel_name: str, device: str = "cuda", l: int = 16384, engine: str | None = "auto"):
        self.rel = get_relation(rel_name)
        self.device = device
        self.l = l
        self.engine = ("cuda" if str(device).startswith("cuda") else None) if engine == "auto" else engine
        t0 = time.perf_counter()
        self.runner = RelationChainRunner(self.rel, device, 128.0)
        self.sys = self.runner.sys
        self.sys_id = protocol_mod.system_id(self.sys)
        self.steps = self.rel.steps
        lay0 = layout_for(self.sys, l, [1], self.steps)
        self.cons = constraints(self.sys, lay0, device)
        self.key = Key(self.rel.name, self.sys_id, lay0.m, lay0.n_i, lay0.n_k, lay0.K, list(lay0.c_rows), dict(lay0.virt),
                       {s: tuple(t.cpu().numpy() for t in self.cons.coo[s]) for s in "abc"})
        self.key_bytes = self.key.to_bytes()
        self.key_sha256 = self.key.digest()
        self.compile_seconds = time.perf_counter() - t0
        self.word_bytes = np.dtype(self.rel.word_dtype).itemsize
        self.y_bytes = np.dtype(self.rel.y_dtype).itemsize

    # --- statement --------------------------------------------------------------------------------------------------
    def batch(self, vus: list, coins_kind: str = "fiat-shamir", zk: bool = False, S: int | None = None) -> Batch:
        l, steps = self.l, self.steps
        per = l // steps
        groups = [vus[i:i + per] for i in range(0, len(vus), per)]
        n_vus = [len(g) for g in groups]
        lay = layout_for(self.sys, l, n_vus, steps)
        if S is not None and S != lay.S:
            raise ValueError(f"{len(vus)} VUs make S = {lay.S}, asked {S}")
        a, b, y = [], [], []
        for g in groups:
            bd = self.runner.marshal(list(g), min_l=l)
            a.append(np.asarray(bd["a"], dtype=np.int64))
            b.append(np.asarray(bd["b"], dtype=np.int64))
            y.append(bd["ypub"].cpu().numpy().astype(np.int64))
        stmt = Statement(self.rel.name, self.sys_id, self.key_sha256, l, steps, self.rel.k, n_vus + [0] * (lay.S - len(groups)),
                         self.word_bytes, self.y_bytes, a, b, y)
        dims, lp = dims_for(lay.n_vars, self.rel.name)
        splits = [list(split((1 << k) << r)) for k, r in zip(dims.k, dims.rates)]
        params = {"relation": self.rel.name, "l": l, "n": dims.n, "kprime": dims.kprime, "queries": dims.queries,
                  "rate_log2": dims.rates, "splits": splits, "order": "row-major", "coins": coins_kind, "zk": zk,
                  "n_claims": 1 + lay.n_links}
        return Batch(groups, lay, stmt, dims, lp, params)

    # --- witness ----------------------------------------------------------------------------------------------------
    def _subs(self, bt: Batch, mutate=None):
        """The S sub-batches one at a time (``build_z`` iterates it): W is materialised per sub-batch and dropped."""
        prover = self

        class _Seq:
            def __len__(self):
                return bt.lay.S

            def __iter__(self):
                for s in range(bt.lay.S):
                    yield prover._sub(bt, s, mutate if s == 0 else None)
        return _Seq()

    def _sub(self, bt: Batch, s: int, mutate=None) -> SubBatch:
        rel, dev, l = self.rel, self.device, self.l
        if s < len(bt.groups):
            bd = self.runner.marshal(list(bt.groups[s]), min_l=l)
        else:
            bd = sc_mod._marshal_pad(rel, self.runner, l, dev)
        hints = rel.hints(rel.params, self.sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], dev)
        W = build_witness(self.sys, bd["pub"], hints, dev)
        if mutate is not None:
            mutate(W)
        n_vus = len(bt.groups[s]) if s < len(bt.groups) else 0
        return SubBatch(W, bd["pub"], bd["ypub_d"].to(dev), n_vus)

    # --- prove ------------------------------------------------------------------------------------------------------
    def prove(self, bt: Batch, coins, mutate=None, cheat: str | None = None, ypub_override=None) -> Result:
        """``mutate(W)``: a witness mutation on sub-batch 0 (negatives); ``cheat``: ``sumcheck.prove``'s dishonest modes;
        ``ypub_override``: the prover claims these chain-end words for sub-batch 0 (the statement says so too)."""
        dev = torch.device(self.device)
        tm = Timer(dev)
        lay = bt.lay
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        stmt = bt.stmt
        t_all = time.perf_counter()
        tm.start("witness")
        subs = self._subs(bt, mutate)
        z = build_z(self.sys, lay, subs, dev)
        if ypub_override is not None:
            z[lay.virt["y16"], :lay.l] = torch.as_tensor(ypub_override, device=dev).to(torch.int32) % P
        tm.stop()
        tm.start("transcript")
        coins.absorb(b"lgto/params", params_bytes(bt.params))
        coins.absorb(b"lgto/stmt", stmt.digest())
        coins.absorb(b"lgto/vk", self.key_sha256)
        virt = z[lay.m:].clone()
        z[lay.m:] = 0
        tm.stop()
        com = pcs_commit(z.view(-1), bt.dims, tm)
        coins.absorb(b"root", com.root1)
        tm.start("zero_check")
        z[lay.m:] = virt
        scp = sc_mod.prove(lay, self.cons, z, coins, cheat=cheat, engine=self.engine)
        z[lay.m:] = 0
        del virt
        tm.stop()
        points, values = pcs_points(lay, scp.claims)
        pcs = pcs_open(com, points, values, coins, tm)
        tm.start("serialize_proof")
        proof = Proof(bt.params, scp.to_bytes(), pcs)
        tm.stop()
        total = time.perf_counter() - t_all
        timings = dict(tm.buckets)
        timings["total"] = total
        timings["sumcheck_detail"] = {k: v for k, v in scp.timings.items() if k != "sumcheck1_rounds"}
        peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0
        del z, com
        return Result(proof, timings, coins.rounds, peak)

    # --- verify -----------------------------------------------------------------------------------------------------
    def verify(self, stmt: Statement, proof: Proof, coins, device: str | None = None) -> tuple[bool, str, dict]:
        """The Python verifier (shares the claim algebra with the prover): statement -> public rows -> sumcheck ->
        PCS.  Returns ``(accepted, reason, info)``."""
        dev = torch.device(device or self.device)
        t0 = time.perf_counter()
        info: dict = {}
        try:
            prm = proof.params
            if stmt.relation != self.rel.name or prm.get("relation") != self.rel.name:
                raise Reject("relation mismatch")
            if stmt.sys_id != self.sys_id:
                raise Reject("statement names another system")
            if stmt.vk_sha256 != self.key_sha256:
                raise Reject("statement names another verification key")
            if stmt.l != self.l or prm.get("l") != self.l or stmt.steps != self.steps or stmt.K != self.rel.k:
                raise Reject("statement shape")
            real = [nv for nv in stmt.n_vus if nv > 0]
            if stmt.n_vus[:len(real)] != real or any(nv * self.steps > self.l for nv in real) or not real:
                raise Reject("statement sub-batch list")
            lay = layout_for(self.sys, self.l, real, self.steps)
            if lay.S != stmt.S:
                raise Reject("statement S does not match its sub-batches")
            dims = Dims(prm["n"], list(prm["kprime"]), list(prm["queries"]), list(prm["rate_log2"]))
            if dims.n != lay.n_vars or prm.get("order") != "row-major" or prm.get("n_claims") != 1 + lay.n_links:
                raise Reject("proof parameters do not match the layout")
            if dims.k[-1] < 0 or min(dims.kprime) < 1:
                raise Reject("bad round structure")
            snd = self.soundness(dims)
            if snd["total_log2"] > -128.0 + 1e-9:
                raise Reject(f"parameters give 2^{snd['total_log2']:.2f} > 2^-128")
            # public rows from the statement's words (the verifier's decode: rel.public_vectors raises off the domain)
            subs = []
            for s in range(lay.S):
                if s < len(real):
                    a_d = torch.as_tensor(stmt.a[s], device=dev)
                    b_d = torch.as_tensor(stmt.b[s], device=dev)
                    try:
                        pub = self.rel.public_vectors(self.rel.params, None, a_d, b_d, None, device=dev)
                    except ValueError as e:
                        raise Reject(f"statement sub-batch {s}: {e}") from None
                    ypub = torch.as_tensor(stmt.y[s], device=dev)
                    if bool((ypub < 0).any()) or bool((ypub >= 1 << self.rel.y_bits).any()):
                        raise Reject(f"statement sub-batch {s}: claimed word out of range")
                    subs.append(SubBatch(None, pub, ypub, real[s]))
                else:
                    bd = sc_mod._marshal_pad(self.rel, self.runner, self.l, dev)
                    subs.append(SubBatch(None, bd["pub"], bd["ypub_d"], 0))
            pub_rows = public_rows(self.sys, lay, subs, dev)
            info["t_public"] = time.perf_counter() - t0
            coins.absorb(b"lgto/params", proof.absorbed_params())
            coins.absorb(b"lgto/stmt", stmt.digest())
            coins.absorb(b"lgto/vk", self.key_sha256)
            coins.absorb(b"root", proof.pcs.roots[0])
            try:
                scp = sc_mod.SumcheckProof.from_bytes(proof.sc_bytes, device=dev)
            except (ValueError, IndexError) as e:
                raise Reject(f"sumcheck bytes: {e}") from None
            t1 = time.perf_counter()
            ok, why, claims = sc_mod.verify(lay, self.cons, pub_rows, scp, coins, device=dev)
            info["t_sumcheck"] = time.perf_counter() - t1
            if not ok:
                raise Reject(why)
            points, values = pcs_points(lay, claims)
            t2 = time.perf_counter()
            pcs_verify(proof.pcs, dims, prm["splits"], points, values, coins)
            info["t_pcs"] = time.perf_counter() - t2
            info["soundness_log2"] = snd["total_log2"]
        except Reject as e:
            info["seconds"] = time.perf_counter() - t0
            return False, str(e), info
        info["seconds"] = time.perf_counter() - t0
        info["coins"] = coins.rounds
        return True, "accept", info

    # --- soundness --------------------------------------------------------------------------------------------------
    def soundness(self, dims: Dims) -> dict:
        """The union bound over the PCS levels (paper (17) with ligerito-design's sampling base) and the three
        sumchecks + batching coins over F_{p^6} (``params.LigeritoParams.soundness`` with the relation shape)."""
        lp = params_mod.LigeritoParams(tall1=1 << dims.k[0], wide_log2=tuple(dims.kprime), rate_log2=tuple(dims.rates),
                                       queries=tuple(dims.queries), zk=False,
                                       relation=params_mod.RELATIONS.get(self.rel.name))
        return lp.soundness({"PCS.claim batching beta J/|F|": math.log2(1 + len(self.key.c_rows)) - lp.fields.ext_log2})
