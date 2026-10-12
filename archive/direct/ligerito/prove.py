"""End-to-end Ligerito prover / verifier for one batch of a registered relation (lane ligerito-relation).

    witness (Ligero's generator, relchain's runner: marshal -> hints -> build_witness; nothing of Ligero's proof runs)
    -> ``layout.build_z`` -> commit ``w`` (``z`` with the virtual rows zeroed, row-major, ``proof.py``)
    -> ``sumcheck.prove`` (zero-check + rows + shift) -> 1 + n_links evaluation claims on ``w``
    + the zero claims ``w~ = 0`` on the aligned blocks covering the virtual rows (``zero_blocks``)
    -> ONE Ligerito evaluation proof for all of them (``pcs_open``: the claims batched by a coin ``beta``).

The transcript order is in ``proof.py``.  ``pcs_commit`` / ``pcs_open`` / ``pcs_verify`` are the commit / multi-claim-open
split of ``proto/pcs.prove`` / ``verify``, here until lane ligerito-pcs-fast (owner of ``pcs.py``) exposes
``commit(f, dims)`` / ``open(state, claims, coins)`` (handoff 20260923T1850Z): they call proto's encoder, Merkle tree,
contractions and claim algebra unchanged; the only protocol change is J eq terms with coefficients ``beta_j`` in place of
proto's single eq term with coefficient 1.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from archive.direct.ligero import protocol as protocol_mod
from archive.direct.ligero.compile import P
from archive.direct.ligero.relations import relation as get_relation
from archive.direct.ligero.relchain import RelationChainRunner
from archive.direct.ligero.witness import build_witness

from . import params as params_mod
from . import sumcheck as sc_mod
from . import zk as zk_mod
from .layout import ZK_B, Layout, SubBatch, build_z, constraints, end_values, fill_zk, layout_for, public_rows
from .proof import FS_QUERIES_LOG2, Key, PcsProof, Proof, Statement, params_bytes, soundness_claim, zero_blocks, zero_points
from .proto import claims as cl
from .proto import device as pdev
from .proto import ext, merkle_multi, stream
from .proto._ligero import commit_gpu as _commit_gpu
from .proto._ligero import merkle as ligero_merkle
from .proto.dims import Dims
from .proto.kernels import col_contract, row_contract
from .proto.linalg import _powers_rows, contract_columns_ext, contract_rows_ext, eq_planes, matmul_mod_p_np, powers_matrix
from .proto.ntt_gpu import Encoder, split
from .proto.pcs import (Timer, _commit, _commit_fast, _ext_bytes, _HostTree, _open_fast, _open_recompute, _open_rows,
                        _row_claims, _stream_group)
from .transcript import FiatShamirCoins, LiveCoins, LocalCoins, _distinct_indices, _ext_from_stream

D = 6
#: parameter set B (ligerito-design FINAL) at the pow2 shape: level-1 rate 1/2, later levels 1/4, 2^-128 union bound
SET_B_RATES = (1, 2, 2, 2, 2)
#: the zero-check's device working set per committed cell beyond z (ligerito-sumcheck-2: 14.6 GB peak at N = 2^30 with z's
#: 4 B/cell)
ZC_BYTES_PER_CELL = 10


def _stream_for_zero_check(f: torch.Tensor, dims: Dims, dev) -> bool | None:
    """True = stream the round-1 commit (no resident codeword; the opening recomputes the |S_1| rows, same bytes) when the
    codeword + its tree would not leave the zero-check's working set free on the card; None = pcs-fast's own rule
    (``LIGERITO_STREAM``, which only budgets the PCS rounds).  Dropping a materialised codeword afterwards costs more:
    4090, fp8-ada 4096 VUs: streamed 0.56 s, drop + empty_cache 0.79 s, drop alone 0.86 s (the allocator's retry)."""
    if dev.type != "cuda" or os.environ.get("LIGERITO_STREAM"):
        return None
    n1 = 1 << (dims.k[0] + dims.rates[0])
    need = torch.cuda.memory_allocated(dev) + (4 << dims.kprime[0]) * n1 + 64 * n1 + ZC_BYTES_PER_CELL * f.numel()
    return True if need > torch.cuda.get_device_properties(dev).total_memory - (5 << 29) else None

__all__ = ["FS_QUERIES_LOG2", "Committed", "Prover", "Reject", "SessionReplayCoins", "StreamCoins", "coins_kind", "dims_for",
           "make_coins", "pcs_commit", "pcs_open", "pcs_verify", "soundness_claim", "statement_for_stream"]


class Reject(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Parameters and coins
# ---------------------------------------------------------------------------------------------------------------------

def dims_for(n: int, rel_name: str | None = None, L: int | None = None,
             target_log2: float = -128.0) -> tuple[Dims, params_mod.LigeritoParams]:
    """Parameter set B for ``N = 2^n`` cells: L = 5 at batch scale (n >= 26), fewer levels on the gate's small batches;
    |S_i| allocated by ``params.allocate_queries`` for the ``target_log2`` union bound (2^-(128 + log2 n) per proof of an
    n-proof job, red-team F5)."""
    shape = params_mod.RELATIONS.get(rel_name) if rel_name else None
    for L_ in ([L] if L is not None else range(5, 1, -1)):
        try:
            p = params_mod.optimize(1 << n, L_, rate_log2=SET_B_RATES[:L_], relation=shape, zk=False, min_wide_log2=3,
                                    target_log2=target_log2)
        except ValueError:
            continue
        return Dims(n, list(p.wide_log2), list(p.queries), list(p.rate_log2)), p
    raise ValueError(f"no parameter-set-B dims for N = 2^{n} (the queries do not fit the codewords)")


class _FP4Relation:
    """The ``relations.Relation`` fields this backend reads, for fp4-nvf4 (its Ligero runner is ``fp4/chain.FP4ChainRunner``,
    not ``relations.py``): 24 chained NVFP4 units per VU, 68 byte words per operand, the claim an FP32 word bound as
    components (``sys.chain["y_end"]``)."""

    name = "fp4-nvf4"
    y_bits, word_dtype, y_dtype = 32, "<u1", "<u4"
    off_domain = (0xFF,)
    instances_dataset, instances_tier, frozen_tier = "synthetic", "fp4-nvf4 synthetic (fp4/chain.instances_fp4)", None

    def __init__(self):
        from verity.ml.circuits.targets.target import NVFP4_SM120

        from archive.direct.ligero.fp4 import chain as fc
        from archive.direct.ligero.fp4 import relation as fr
        self.params, self.target, self.k, self.steps = fr.NVFP4, NVFP4_SM120, fr.WORDS, fr.STEPS
        self.instance_seed = fc.INSTANCE_SEED
        self.relation_name, self.model_name = fc.RELATION_NAME, fc.MODEL_NAME
        self.instances_recipe = f"instances_fp4(seed {fc.INSTANCE_SEED}): VU i from family {list(fc.VU_FAMILIES)}[i % 14]"
        self.witness_generation = "fp4/witness.hints_fp4 (numpy) + ligero build_witness; accumulators from BLACKWELL_SM120_NVF4.step"
        self.claim_note = "the claim per VU is its final FP32 accumulator word, bound as sign / exponent / fraction (3 end rows)"

    def public_vectors(self, p, c, a, b, y, device="cpu"):
        from archive.direct.ligero.fp4.witness import public_vectors_fp4
        host = lambda x: None if x is None else (x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x))  # noqa: E731
        out = public_vectors_fp4(host(a), host(b), host(c), host(y))
        return {k: torch.as_tensor(v, dtype=torch.int64, device=device) for k, v in out.items()}

    def hints(self, p, sys, c_d, a_d, b_d, y_d, device):
        from archive.direct.ligero.fp4.witness import hints_fp4
        h = hints_fp4(sys, c_d.cpu().numpy(), a_d.cpu().numpy(), b_d.cpu().numpy())
        return torch.as_tensor(h, dtype=torch.int64, device=device)


class _FP4Runner:
    """``RelationChainRunner``'s ``sys`` / ``p`` / ``marshal`` / ``_pad_unit`` over ``FP4ChainRunner`` (its prover never runs)."""

    def __init__(self, rel: _FP4Relation, device: str):
        from archive.direct.ligero.fp4.chain import FP4ChainRunner
        self.rel, self.device = rel, device
        self.inner = FP4ChainRunner(device, -128.0, 2, hints="numpy")
        self.sys, self.p = self.inner.sys, self.inner.p
        self._cache: dict = {}

    def _pad_unit(self):
        from archive.direct.ligero.fp4.chain import _pad_unit
        return _pad_unit()

    def marshal(self, vus: list, min_l: int = 256) -> dict:
        key = (len(vus), id(vus[0]), id(vus[-1]), id(vus[len(vus) // 2]), min_l)
        if key in self._cache:
            return self._cache[key]
        bd = self.inner.marshal(vus, min_l)
        dev = self.device
        to_dev = lambda x: torch.as_tensor(np.asarray(x, dtype=np.int64)).to(dev)  # noqa: E731
        out = {"lay": bd["lay"], "a": bd["a"], "b": bd["b"], "c_in": bd["c_in"], "ypub": bd["y32"],
               "a_d": to_dev(bd["a"]), "b_d": to_dev(bd["b"]), "c_d": to_dev(bd["c_in"]),
               "y_d": torch.zeros(bd["lay"].l, dtype=torch.int64, device=dev), "ypub_d": bd["y32"].to(dev)}
        out["pub"] = self.rel.public_vectors(self.p, None, out["a_d"], out["b_d"], None, device=dev)
        self._cache[key] = out
        return out


def make_coins(kind: str, seed: int = 0, slots=None):
    if kind == "fiat-shamir":
        return FiatShamirCoins()
    if kind == "local":
        return LocalCoins(seed)
    if kind == "live":
        return LiveCoins(slots)
    raise ValueError(f"coins {kind!r}")


def coins_kind(coins) -> str:
    return {FiatShamirCoins: "fiat-shamir", LocalCoins: "local"}.get(type(coins), "live" if isinstance(coins, LiveCoins) else type(coins).__name__)


class StreamCoins(LiveCoins):
    """``transcript.LiveCoins`` fed by lane live-2's many-round challenge stream (``ligero/live.ChallengeStream`` or
    ``LocalChallenges``): verifier coin slot k = ``expand_challenge(H(stmt, c_0..c_{R-1}), k, label, r_k, 32)`` -- the
    verifier's k-th committed coin, opened after it received MSG k = BLAKE2b of the prover's messages absorbed since slot
    k-1.  ``slots`` (the 32-byte slot values) replay the run in ``LiveCoins(slots)`` (the verifier side, the dump's
    ``.coins`` file); ``binding()`` is what re-derives them from the stream's commitments and openings."""

    def __init__(self, stream, statement: bytes):
        super().__init__(iter(()))
        self.stream = stream
        self.statement = statement
        self.slots_out: list[bytes] = []
        self.labels: list[bytes] = []
        self._h = hashlib.blake2b(digest_size=32)
        stream.begin(statement)

    def absorb(self, label: bytes, data: bytes) -> None:
        self.absorbed += 1
        self._h.update(len(label).to_bytes(4, "little") + label + len(data).to_bytes(8, "little"))
        self._h.update(data)

    def _slot(self, label: bytes) -> None:
        msg = self._h.digest()
        self._h = hashlib.blake2b(digest_size=32)
        coin = self.stream.challenge(label, msg, 32)
        self.slots_out.append(coin)
        self.labels.append(label)
        self._coin, self._sub = coin, 0
        self.rounds += 1

    def challenge(self, label: bytes, n: int) -> np.ndarray:
        self._slot(label)
        return _ext_from_stream(self._stream(8 * n * D), n)

    def indices(self, label: bytes, domain: int, count: int) -> list[int]:
        self._slot(label)
        return _distinct_indices(self._stream, domain, count)

    def binding(self) -> dict:
        return {"statement": self.statement.hex(), "commitments": [c.hex() for c in self.stream.commitments()],
                "openings": [[r.hex(), s.hex()] for r, s in self.stream.openings()], "labels": [x.decode() for x in self.labels],
                "derivation": "slot_k = SHAKE-256(b'ligero-chal|v1|' | blake2b(b'ligero-chal-ctx|' | u32 len(stmt) | stmt | c_0..c_{R-1}) "
                              "| u32 k | u32 len(label) | label | r_k)[:32]; coin_commit(r_k, s_k) == c_k (ligero/live.py)"}


class SessionReplayCoins(LiveCoins):
    """The verifier's replay of a stream proof (``LiveCoins`` over the dump's slots) that also recomputes MSG k -- the
    BLAKE2b of the proof's absorbs since slot k-1, as ``StreamCoins`` sent it -- and the label of every slot, so the proof
    can be matched against the issuing verifier's session record (``msg_sha256`` = sha256(MSG k) per round)."""

    def __init__(self, slots):
        super().__init__(slots)
        self.msgs: list[bytes] = []
        self.labels: list[bytes] = []
        self._h = hashlib.blake2b(digest_size=32)

    def absorb(self, label: bytes, data: bytes) -> None:
        self.absorbed += 1
        self._h.update(len(label).to_bytes(4, "little") + label + len(data).to_bytes(8, "little"))
        self._h.update(data)

    def _slot(self, label: bytes) -> None:
        self.msgs.append(self._h.digest())
        self.labels.append(label)
        self._h = hashlib.blake2b(digest_size=32)
        self._next()

    def challenge(self, label: bytes, n: int) -> np.ndarray:
        self._slot(label)
        return _ext_from_stream(self._stream(8 * n * D), n)

    def indices(self, label: bytes, domain: int, count: int) -> list[int]:
        self._slot(label)
        return _distinct_indices(self._stream, domain, count)


def statement_for_stream(bt, key_sha256: bytes) -> bytes:
    """STMT i of the challenge stream: the batch's public claim (params, statement file, key), fixed before any coin."""
    return b"lgto-stmt|v1|" + hashlib.sha256(params_bytes(bt.params)).digest() + (bt.stmt_digest or bt.stmt.digest()) + key_sha256


# ---------------------------------------------------------------------------------------------------------------------
# PCS: commit / multi-claim open / verify (proto/pcs.py split; see the module docstring)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Committed:
    dims: Dims
    M1: torch.Tensor                    # (C1, R1) int32 view of f
    U1: torch.Tensor | None             # (C1, n_1) codeword (None: streamed commit, opened rows recomputed from M1)
    tree1: object
    enc1: Encoder
    root1: bytes
    fast: bool = False                  # pcs-fast's device path (commit_gpu tree, device contractions / sumcheck)


def _gpu(device) -> bool:
    return torch.device(device).type == "cuda" and ligero_merkle._gpu_merkle is not None


def _fast(f: torch.Tensor) -> bool:
    return f.device.type == "cuda" and pdev.available(f) and ligero_merkle._gpu_merkle is not None


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


def pcs_commit(f: torch.Tensor, dims: Dims, tm: Timer, fast: bool | None = None) -> Committed:
    """Round 1 of proto's ``prove``: encode the columns of ``M1 = f.view(C1, R1)`` and Merkle-commit the codeword rows
    (pcs-fast: streamed by column groups when the codeword does not fit the card, ``LIGERITO_STREAM``)."""
    ks, kps, rates = dims.k, dims.kprime, dims.rates
    R1, C1 = 1 << ks[0], 1 << kps[0]
    assert f.numel() == 1 << dims.n and f.dtype == torch.int32
    fast = _fast(f) if fast is None else fast
    M1 = f.view(C1, R1)
    enc1 = Encoder(R1, f.device, rates[0])
    if fast:
        pdev.share_torch_allocator()
        group = _stream_group(dims, f.device)
        if group:
            tm.start("commit_r1_stream")
            root1, tree1 = stream.commit_streamed(M1, enc1, group, _commit_gpu)
            tm.stop()
            return Committed(dims, M1, None, tree1, enc1, root1, True)
    tm.start("commit_r1_alloc")
    U1 = torch.empty((C1, enc1.n), dtype=torch.int32, device=f.device)
    tm.stop()
    tm.start("commit_r1_encode")
    U1 = enc1.encode(M1, out=U1)
    tm.stop()
    tm.start("commit_r1_merkle")
    root1, tree1 = _commit_fast(U1) if fast else _tree(U1)
    tm.stop()
    return Committed(dims, M1, U1, tree1, enc1, root1, fast)


def pcs_open(com: Committed, points: np.ndarray, values: np.ndarray, coins, tm: Timer, n_c: int = 0) -> PcsProof:
    """The evaluation proof of ``w~(points[j]) = values[j]`` for all ``j`` (points in PCS variable order, (J, n, 6));
    ``n_c``: the number of column variables (the low PCS variables), for the Boolean-row shortcut of round 1."""
    if com.fast:
        return _pcs_open_fast(com, points, values, coins, tm, n_c)
    return pcs_open_reference(com, points, values, coins, tm)


def _xpow_ext(u: np.ndarray) -> np.ndarray:
    """(6, C, 6) plane contractions ``u[a, c] = <M_a[c, :], eq>`` (M = sum_a x^a M_a) -> (C, 6): sum_a x^a u[a, c]."""
    out = ext.zero((u.shape[1],))
    for a in range(D):
        xa = ext.zero()
        xa[a] = 1
        out = ext.add(out, ext.mul(u[a], xa))
    return out


def _r1_eq_contract(M1: torch.Tensor, z: np.ndarray, k1: int, n_c: int) -> np.ndarray:
    """(C1, 6) ``u[c] = sum_x M1[c, x] eq(z[:k1], x)``.  When the point's coordinates ``n_c .. k1-1`` are Boolean (the
    chain-row claims ``w(c_x, rho)``: the low row bits of ``c_x``) the eq table is zero outside one block of ``2^n_c``
    words per column, so only that block is read (2^{k1 - n_c} x fewer words)."""
    C1, R1 = M1.shape
    zl = np.asarray(z[:k1], dtype=np.int64) % P
    tail = zl[n_c:]
    if 0 < n_c < k1 and not tail[:, 1:].any() and np.isin(tail[:, 0], (0, 1)).all():
        b = int(sum(int(tail[t, 0]) << t for t in range(tail.shape[0])))
        sub = M1.view(C1, R1 >> n_c, 1 << n_c)[:, b, :]
        u = pdev.contract_planes(sub, pdev.eq_table(zl[:n_c], M1.device))
    else:
        u = pdev.contract_planes(M1, pdev.eq_table(zl, M1.device))
    return (u % P).cpu().numpy()


def _dedupe_eq(coefs: list, zs: list) -> tuple[list, list]:
    """Merge eq terms with the same point (their coefficients add; the claim algebra is unchanged)."""
    oc, oz = [], []
    for c, z in zip(coefs, zs):
        for j, z2 in enumerate(oz):
            if z2.shape == z.shape and np.array_equal(z2, z):
                oc[j] = ext.add(oc[j], c)
                break
        else:
            oc.append(c)
            oz.append(z)
    return oc, oz


def _pcs_open_fast(com: Committed, points: np.ndarray, values: np.ndarray, coins, tm: Timer, n_c: int = 0) -> PcsProof:
    """``pcs_open_reference``'s protocol on pcs-fast's device pieces (``proto/pcs.prove``'s loop): J eq terms instead of
    one -- round 1's column sumcheck on the host (C_1 = 2^{k'_1} values per term), rounds >= 2 on ``DeviceSumcheck`` with
    the extra eq terms appended as table rows.  Same transcript and proof bytes as the reference."""
    dims, device = com.dims, com.M1.device
    n, ks, kps, qs, L, rates = dims.n, dims.k, dims.kprime, dims.queries, dims.rounds, dims.rates
    J = points.shape[0]
    assert points.shape == (J, n, D) and values.shape == (J, D)
    tr = coins
    tr.absorb(b"z", _ext_bytes(points))
    beta = tr.challenge(b"beta", J)
    roots, sumcheck, rows_out, sibs = [com.root1], [], [], []
    Us, trees, encs = [com.U1], [com.tree1], [com.enc1]
    M1 = com.M1
    # ---------------- round 1: J eq terms, host sumcheck
    tm.start("sumcheck_r1_contract")
    terms = [cl.Term(beta[j] % P, "eq", z=points[j] % P, nvars=n) for j in range(J)]
    us = [_r1_eq_contract(M1, points[j], ks[0], n_c) for j in range(J)]
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
    running = sc.claim()
    eqs = [t.restrict(ks[0], r1) for t in terms]
    coef_eq, z_eq = _dedupe_eq([t.coef for t in eqs], [t.z for t in eqs])
    etas_g = np.zeros(0, dtype=np.int64)
    coef_g = ext.zero((0,))
    tm.stop()
    tm.start("sumcheck_r1_fold")
    y = row_contract(M1, rbar)                                           # (6, R1) int32 planes
    tm.stop()
    rbars = [rbar]
    # ---------------- rounds 2..L
    for i in range(1, L):
        R, C = 1 << ks[i], 1 << kps[i]
        Mi = y.view(D, C, R)
        Mflat = Mi.reshape(D * C, R)
        tm.start("recursion_encode")
        enc = Encoder(R, device, rates[i])
        U = enc.encode(Mflat)
        tm.stop()
        tm.start("recursion_merkle")
        root, tree = _commit_fast(U)
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
        depth = n_prev.bit_length() - 1
        if Us[prev] is None:
            tm.stop()
            tm.start("commit_r1_recompute")
            op = _open_recompute(M1, encs[prev], trees[prev], S, depth)
            tm.stop()
            tm.start("serialize_openings")
        else:
            op = _open_fast(Us[prev], trees[prev], S, depth)
        rows_out.append(op.rows)
        sibs.append(op.siblings)
        tr.absorb(b"rows", np.ascontiguousarray(op.rows.astype(np.uint32)).tobytes())
        tr.absorb(b"sib", op.siblings)
        Us[prev] = trees[prev] = None
        tm.stop()
        tm.start("sumcheck_host")
        etas = encs[prev].etas(S)
        rhs = _row_claims(op.rows, rbars[prev], base=(prev == 0))
        alpha = tr.challenge(b"alpha", 1 + len(S))
        coef_eq = [ext.mul(c, alpha[0]) for c in coef_eq]
        coef_g = ext.mul(coef_g, alpha[0]) if len(etas_g) else coef_g
        coef_g = np.concatenate((coef_g, alpha[1:]), axis=0)
        etas_g = np.concatenate((etas_g, np.asarray(etas, dtype=np.int64)))
        running = ext.add(ext.mul(alpha[0], running), ext.sum_ext(ext.mul(alpha[1:], rhs), axis=0))
        colg = cl.geom_colparts(etas_g, ks[i], kps[i])[:, :, 0]
        eq_terms = [cl.Term(ext.one(), "eq", z=z, nvars=len(z)) for z in z_eq]
        coles = [t.colpart(ks[i], kps[i]) for t in eq_terms]
        coef = np.concatenate((coef_eq[0][None], coef_g), axis=0)
        tm.stop()
        tm.start("recursion_contract")
        ug = pdev.contract_geom(Mflat, etas_g)
        ues = [pdev.contract_planes(Mflat, pdev.eq_table(z[:ks[i]], device)) for z in z_eq]
        dsc = pdev.DeviceSumcheck(ug, ues[0], coef, colg, coles[0], C, device)
        if len(z_eq) > 1:
            A = [ext.mul(coef_eq[j], _xpow_ext((ues[j] % P).view(D, C, D).cpu().numpy())) for j in range(1, len(z_eq))]
            dt = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a) % P).astype(np.int32), device=device)  # noqa: E731
            dsc.A = torch.cat((dsc.A, dt(A)), 0).contiguous()
            dsc.B = torch.cat((dsc.B, dt(coles[1:])), 0).contiguous()
            dsc.T += len(z_eq) - 1
        del ug, ues
        tm.stop()
        tm.start("sumcheck_device")
        msgs, rs = [], []
        for j in range(kps[i]):
            g = dsc.message()
            if j == 0 and not np.array_equal(ext.add(g[0], g[1]), running):
                raise AssertionError(f"PCS round {i + 1}: prover-side claim mismatch")
            tr.absorb(b"g", _ext_bytes(g))
            r = tr.challenge(b"r", 1)[0]
            dsc.fold(r)
            msgs.append(g)
            rs.append(r)
        sumcheck.append(msgs)
        running = cl.interpolate_deg2(g[0], g[1], g[2], rs[-1])
        ri = np.stack(rs[::-1], axis=0)
        rbar = cl.rbar(ri)
        tm.stop()
        tm.start("sumcheck_host")
        coef_eq = [ext.mul(c, t.colpart_at(ks[i], ri)) for c, t in zip(coef_eq, eq_terms)]
        coef_eq, z_eq = _dedupe_eq(coef_eq, [z[:ks[i]] for z in z_eq])
        coef_g = ext.mul(coef_g, cl.geom_colpart_at(etas_g, ks[i], ri))
        rbars.append(rbar)
        del dsc
        tm.stop()
        tm.start("recursion_fold")
        y = pdev.fold_ext(Mi, rbar)
        tm.stop()
    # ---------------- final vector, open U_L
    tm.start("serialize_final")
    final_y = y.to(torch.int64).t().contiguous().cpu().numpy() % P
    tr.absorb(b"y", _ext_bytes(final_y))
    n_last = encs[-1].n
    S = tr.indices(b"S", n_last, qs[-1])
    depth = n_last.bit_length() - 1
    op = _open_fast(Us[-1], trees[-1], S, depth) if Us[-1] is not None else _open_recompute(M1, encs[-1], trees[-1], S, depth)
    rows_out.append(op.rows)
    sibs.append(op.siblings)
    tm.stop()
    return PcsProof(roots, sumcheck, rows_out, sibs, final_y)


def pcs_open_reference(com: Committed, points: np.ndarray, values: np.ndarray, coins, tm: Timer) -> PcsProof:
    """The multi-claim open on proto's reference pieces (host column sumcheck, torch contractions)."""
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


def _canonical(x, what: str) -> np.ndarray:
    """A proof word must be canonical in [0, p): a lifted x + p would be absorbed as other bytes than Rust's reader
    accepts (red-team F11)."""
    x = np.asarray(x)
    if x.dtype.kind not in "iu" or (x < 0).any() or (x >= P).any():
        raise Reject(f"{what} word out of range")
    return x.astype(np.int64)


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
        _canonical(g, "PCS sumcheck message")
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
            _canonical(g, "PCS sumcheck message")
            tr.absorb(b"g", _ext_bytes(g))
            rs.append(tr.challenge(b"r", 1)[0])
        ok, running = cl.verify_column_rounds(running, pf.sumcheck[i], rs)
        if not ok:
            raise Reject(f"PCS sumcheck round {i + 1} failed")
        ri = np.stack(rs[::-1], axis=0)
        rbars.append(cl.rbar(ri))
        terms = cl.restrict_terms(terms, ks[i], ri)
    y = _canonical(pf.final_y, "final vector")
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

def pcs_points(lay: Layout, claims: list, zl=None, blocks=()) -> tuple[np.ndarray, np.ndarray]:
    """The sumcheck's claims ``(p_i || p_c)`` + the zero claims on ``blocks`` (``proof.zero_blocks``: without them the
    prover commits non-zero virtual rows and the verifier's ``w~ + sum eq . pub~`` absorbs any change of the public rows,
    red-team-ligerito-2 V1) -> PCS order: ``(p_c || p_i)`` for the row-major commitment, or ``zk.permute_point`` for the
    ZK index map ``zk.z_to_f`` (``zl``: the ``RowMaskLayout``); numpy."""
    zp = [c.point.cpu().numpy() for c in claims]
    zp += zero_points(lay.n_i, zp[0], list(blocks))
    if zl is None:
        pts = [np.concatenate((p[lay.n_i:], p[:lay.n_i])) for p in zp]
    else:
        pts = [zk_mod.permute_point(p, zl) for p in zp]
    vals = [c.value.cpu().numpy() for c in claims] + [np.zeros(6, dtype=np.int64)] * len(blocks)
    return np.stack(pts).astype(np.int64) % P, np.stack(vals).astype(np.int64) % P


def n_eval_claims(lay: Layout) -> int:
    """The sumcheck's evaluation claims on ``w``: LGSC0003 ``w~(r_i, r_c)`` + one per link; LGSC0004 ``w~(r_i, r_c)`` and
    ``w~(rho_i, rho_c)`` (its three mask claims are sparse PCS terms, not points)."""
    return 2 if lay.zk is not None else 1 + lay.n_links


def split_claims(claims: list) -> tuple[list, list]:
    """LGSC0004's claims -> (evaluation claims, sparse mask claims); LGSC0003 has no sparse ones."""
    return ([c for c in claims if isinstance(c, sc_mod.EvalClaim)], [c for c in claims if isinstance(c, sc_mod.SparseClaim)])


def pcs_sparse(claims: list, zl) -> list:
    """LGSC0004's sparse mask claims (mask(zc), mask(cmb), mask(rb), in that order) as proto ``open``'s sparse terms: the
    relation cells ``(rows, cols)`` through the ZK index map (``zk.f_index``)."""
    return [(np.asarray(zk_mod.f_index(c.rows.cpu().numpy().astype(np.int64), c.cols.cpu().numpy().astype(np.int64), zl)),
             c.weights.cpu().numpy().astype(np.int64) % P, c.value.cpu().numpy().astype(np.int64) % P) for c in claims]


def zk_layout(lay: Layout, dims: Dims):
    """The PCS's view of LGSC0004's ZK rows (lane ligerito-zk's ``RowMaskLayout``: the index map interleaves ``k_L - b``
    column bits below the row bits): ``lay.zk``'s uniform rows, product triple and first Libra row.  The Libra
    coefficients are LGSC0004's (its sparse claims on ``lay.zk.g_rows``), so no PCS-side sumcheck masks (``sumchecks=()``):
    two mask systems on the same g cells would correlate their messages (red-team-ligerito-3)."""
    zr = lay.zk
    return zk_mod.RowMaskLayout(lay.n_i, lay.n_c, dims.k[-1], tuple(zr.uniform), (tuple(zr.product),), zr.g_rows[0],
                                sumchecks=(), b=ZK_B)


@dataclass
class Batch:
    groups: list                        # per real sub-batch: the VU tuples
    lay: Layout
    stmt: Statement
    dims: Dims
    lparams: params_mod.LigeritoParams
    params: dict
    stmt_digest: bytes = b""            # sha256(LGTS0001 bytes): the statement is fixed before the prover's clock starts
    key_bytes: bytes = b""              # the layout's LGVK0001 key (``Prover.key_for``)
    key_sha256: bytes = b""


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
        self.rel = _FP4Relation() if rel_name == "fp4-nvf4" else get_relation(rel_name)
        self.device = device
        self.l = l
        self.engine = ("cuda" if str(device).startswith("cuda") else None) if engine == "auto" else engine
        self.pcs_fast: bool | None = None           # None: pcs-fast's device path whenever it is available
        #: "pcs-fast": lane ligerito-pcs-fast's ``proto/pcs.commit`` / ``open`` (same transcript); "own": prove.py's adapter
        self.pcs_impl = os.environ.get("LIGERITO_RELATION_PCS", "pcs-fast")
        self._witness_lock = threading.Lock()     # windowed batches: Ligero's marshal cache is not thread-safe
        t0 = time.perf_counter()
        self.runner = _FP4Runner(self.rel, device) if rel_name == "fp4-nvf4" else RelationChainRunner(self.rel, device, 128.0)
        self.sys = self.runner.sys
        self.sys_id = protocol_mod.system_id(self.sys)
        self.steps = self.rel.steps
        lay0 = layout_for(self.sys, l, [1], self.steps)
        self.cons = constraints(self.sys, lay0, device)
        self.key = Key(self.rel.name, self.sys_id, lay0.m, lay0.n_i, lay0.n_k, lay0.K, list(lay0.c_rows), dict(lay0.virt),
                       {s: tuple(t.cpu().numpy() for t in self.cons.coo[s]) for s in "abc"})
        self.key_bytes = self.key.to_bytes()
        self.key_sha256 = self.key.digest()
        self._zk_keys: dict = {}
        #: LGSC0004's round schedule: a ``sumcheck._schedule_arg`` preset or string; None = ``zk_default_schedule`` (13 coins)
        self.zk_schedule: str | None = None
        self.compile_seconds = time.perf_counter() - t0
        self.word_bytes = np.dtype(self.rel.word_dtype).itemsize
        self.y_bytes = np.dtype(self.rel.y_dtype).itemsize

    def key_for(self, lay: Layout) -> tuple:
        """``(constraints, Key, key bytes, sha256)`` of a layout.  LGSC0003: one key per relation.  ZK (LGSC0004): the
        committed next / mask / product rows sit below the Libra rows, whose count is ``ceil(g_cells / C)``, so the
        constraint rows and the key follow the column count C: one key per (relation, C), cached."""
        if lay.zk is None:
            return self.cons, self.key, self.key_bytes, self.key_sha256
        hit = self._zk_keys.get(lay.C)
        if hit is None:
            cons = constraints(self.sys, lay, self.device)
            key = Key(self.rel.name, self.sys_id, lay.m, lay.n_i, lay.n_k, lay.K, list(lay.c_rows), dict(lay.virt),
                      {s: tuple(t.cpu().numpy() for t in cons.coo[s]) for s in "abc"})
            hit = self._zk_keys[lay.C] = (cons, key, key.to_bytes(), key.digest())
        return hit

    def sumcheck_schedule(self, lay: Layout):
        """The sumcheck's round schedule: None on LGSC0003 (its default); on LGSC0004 ``zk_schedule`` (default
        ``zk_default_schedule``), refused unless ``vf <= sumcheck.zk_vf_max(lay) = n_c - 4``: the product triple blinds
        each of the 2^vf final-table entries only with >= 12 base dimensions of i1 per entry (ligerito-sumcheck-3 §1 step 2;
        red-team-ligerito-3 R3-6).  ``sumcheck.prove`` refuses it too (796d8a11); this fails before any work."""
        if lay.zk is None:
            return None
        sch = sc_mod.zk_schedule(lay, sc_mod._schedule_arg(self.zk_schedule or "zk", lay))
        if sch.vf > sc_mod.zk_vf_max(lay):
            raise ValueError(f"LGSC0004 schedule vf = {sch.vf} > n_c - 4 = {sc_mod.zk_vf_max(lay)}: the final "
                             f"tables would not be blinded")
        return sch

    def stmt_layout(self, stmt: Statement, zk: bool) -> Layout:
        return layout_for(self.sys, self.l, [nv for nv in stmt.n_vus if nv > 0], self.steps, zk=zk)

    # --- statement --------------------------------------------------------------------------------------------------
    def batch(self, vus: list, coins_kind: str = "fiat-shamir", zk: bool = False, S: int | None = None,
              target_log2: float = -128.0) -> Batch:
        l, steps = self.l, self.steps
        per = l // steps
        groups = [vus[i:i + per] for i in range(0, len(vus), per)]
        n_vus = [len(g) for g in groups]
        lay = layout_for(self.sys, l, n_vus, steps, zk=zk)
        _, _, key_bytes, key_sha = self.key_for(lay)
        if S is not None and S != lay.S:
            raise ValueError(f"{len(vus)} VUs make S = {lay.S}, asked {S}")
        a, b, y = [], [], []
        for g in groups:
            bd = self.runner.marshal(list(g), min_l=l)
            a.append(np.asarray(bd["a"], dtype=np.int64))
            b.append(np.asarray(bd["b"], dtype=np.int64))
            y.append(bd["ypub"].cpu().numpy().astype(np.int64))
        stmt = Statement(self.rel.name, self.sys_id, key_sha, l, steps, self.rel.k, n_vus + [0] * (lay.S - len(groups)),
                         self.word_bytes, self.y_bytes, a, b, y)
        dims, lp = dims_for(lay.n_vars, self.rel.name, target_log2=target_log2)
        t_pad = zk_mod.T_PAD if zk else 0
        kprime, rates, target = list(dims.kprime), list(dims.rates), target_log2
        while True:
            if zk:
                # ZK column padding lowers every level's rate to (2^{k_i} + t_pad) / n_i: |S_i| re-derived for the padded shape
                from .proto import pcs as ppcs
                dims = ppcs.dims_for(lay.n_vars, kprime, rates, target_log2=target, t_pad=t_pad)
            if self.soundness(dims, t_pad, lay)["total_log2"] <= target_log2 + 1e-9:
                break
            target -= 0.25                  # the relation's own terms on top of the PCS union
            if not zk:
                dims, lp = dims_for(lay.n_vars, self.rel.name, target_log2=target)
        splits = [list(split((1 << k) << r)) for k, r in zip(dims.k, dims.rates)]
        params = {"relation": self.rel.name, "l": l, "n": dims.n, "kprime": dims.kprime, "queries": dims.queries,
                  "rate_log2": dims.rates, "splits": splits, "order": "zk-interleaved" if zk else "row-major", "coins": coins_kind, "zk": zk,
                  "n_claims": n_eval_claims(lay) + len(zero_blocks(lay.virt)), "zero_blocks": [list(x) for x in zero_blocks(lay.virt)],
                  **({"t_pad": t_pad} if zk else {})}
        return Batch(groups, lay, stmt, dims, lp, params, stmt.digest(), key_bytes, key_sha)

    # --- witness ----------------------------------------------------------------------------------------------------
    def _pad(self) -> dict:
        """The pad sub-batch's marshalled words (``l`` zero-word units, device-resident), once per prover like relchain's
        marshal cache."""
        if getattr(self, "_pad_bd", None) is None:
            self._pad_bd = sc_mod._marshal_pad(self.rel, self.runner, self.l, self.device)
        return self._pad_bd

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
            bd = self._pad()
        hints = rel.hints(rel.params, self.sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], dev)
        W = build_witness(self.sys, bd["pub"], hints, dev)
        if mutate is not None:
            mutate(W)
        n_vus = len(bt.groups[s]) if s < len(bt.groups) else 0
        return SubBatch(W, bd["pub"], bd["ypub_d"].to(dev), n_vus)

    def _stmt_subs(self, stmt: Statement, lay: Layout, real: list, dev) -> list:
        """The statement's sub-batches as the verifier decodes them (``rel.public_vectors`` raises off the domain)."""
        subs = []
        for s in range(lay.S):
            if s < len(real):
                # pad units (columns past the sub-batch's chains) feed no output: the canonical words there are 0
                pc = real[s] * self.steps
                if np.any(np.asarray(stmt.a[s])[pc:] != 0) or np.any(np.asarray(stmt.b[s])[pc:] != 0):
                    raise Reject(f"statement sub-batch {s}: operand word in a pad unit (non-canonical)")
                a_d = torch.as_tensor(stmt.a[s], device=dev)
                b_d = torch.as_tensor(stmt.b[s], device=dev)
                try:
                    pub = self.rel.public_vectors(self.rel.params, None, a_d, b_d, None, device=dev)
                except ValueError as e:
                    raise Reject(f"statement sub-batch {s}: {e}") from None
                ypub = torch.as_tensor(stmt.y[s], device=dev)
                if bool((ypub < 0).any()) or bool((ypub >= 1 << self.rel.y_bits).any()):
                    raise Reject(f"statement sub-batch {s}: claimed word out of range")
                # the end constraint is masked off the chain ends: a word there binds nothing, so the canonical statement
                # has 0 (ligero-statement/v4) and any other value would make the statement bytes malleable
                _, _, end = lay.chain_layout(s).masks(dev)
                if bool((ypub[end.to(torch.bool).logical_not()] != 0).any()):
                    raise Reject(f"statement sub-batch {s}: claimed word off a chain end (non-canonical)")
                subs.append(SubBatch(None, pub, ypub, real[s]))
            else:
                bd = self._pad() if dev == torch.device(self.device) else sc_mod._marshal_pad(self.rel, self.runner, self.l, dev)
                subs.append(SubBatch(None, bd["pub"], bd["ypub_d"], 0))
        return subs

    def _forge(self, z, lay: Layout, stmt: Statement, forge_rows=None) -> dict:
        """The committed virtual rows of a forged proof (negatives): the verifier rebuilds ``z = w + public rows`` (+ the
        ``next:*`` claims) from ``stmt``, so a prover running the sumchecks on its own ``z`` commits ``z - public rows`` on
        the public virtual rows (``next:*``: 0), plus ``forge_rows`` (row -> (C,) int64, added).  Returns {row: (C,) int64
        mod p} for the rows where the committed ``w`` is non-zero."""
        dev = z.device
        real = [nv for nv in stmt.n_vus if nv > 0]
        lay_c = layout_for(self.sys, self.l, real, self.steps, zk=lay.zk is not None)   # the statement's own masks (n_vus)
        claimed = public_rows(self.sys, lay_c, self._stmt_subs(stmt, lay_c, real, dev), dev)
        out = {lay.virt[k]: (z[lay.virt[k]].to(torch.int64) - v) % P for k, v in claimed.items()}
        for r, d in (forge_rows or {}).items():
            out[r] = (out.get(r, torch.zeros(lay.C, dtype=torch.int64, device=dev)) + d.to(dev)) % P
        return {r: d for r, d in out.items() if bool(d.any())}

    def _forged_sumcheck(self, lay: Layout, cons, z, coins, forge: dict, cheat):
        """The honest zero-check on the true ``z`` with ``values[0]`` shifted to ``w'~(r_i, r_c)`` for the committed
        ``w' = w + forge`` (row -> (C,) int64), the shifted value absorbed in place of the honest one: the red-team V1
        forgery (a negative; the challenges before ``values`` do not depend on it, so a copied coin state finds the point)."""
        s1 = sc_mod.prove(lay, cons, z, copy.deepcopy(coins), cheat=cheat, engine=self.engine,
                          schedule=self.sumcheck_schedule(lay))
        pt = s1.claims[0].point
        rows = sorted(forge)
        fold = sc_mod.matvec_base_ext(torch.stack([forge[r] for r in rows]), sc_mod.eq_table(pt[lay.n_i:]))
        adj = sc_mod.mle_eval_ext(fold, sc_mod.eq_table(pt[:lay.n_i])[rows])
        vals = s1.values.clone()
        vals[0] = (vals[0] + adj) % P

        class _Swap:
            def __getattr__(self, k):
                return getattr(coins, k)

            def absorb(self, label, data):
                coins.absorb(label, sc_mod._ext_bytes(vals) if label == b"values" else data)
        scp = sc_mod.prove(lay, cons, z, _Swap(), cheat=cheat, engine=self.engine,
                           schedule=self.sumcheck_schedule(lay))
        scp.values = vals
        scp.claims[0] = dataclasses.replace(scp.claims[0], value=vals[0])
        return scp

    # --- prove ------------------------------------------------------------------------------------------------------
    def prove(self, bt: Batch, coins, mutate=None, cheat: str | None = None, ypub_override=None, forge_stmt=None,
              forge_z=None, forge_rows=None, edit_zk=None) -> Result:
        """``mutate(W)``: a witness mutation on sub-batch 0 (negatives); ``cheat``: ``sumcheck.prove``'s dishonest modes;
        ``ypub_override``: the prover claims these chain-end words for sub-batch 0 (the statement says so too).
        Red-team V1 forgeries (the zero claims must reject them): the sumchecks run honestly on the prover's ``z`` (the
        true one, or ``forge_z(z, lay)``'s in-place edit of it, virtual rows included) and the commitment carries whatever
        makes the verifier's reconstruction from ``forge_stmt`` (default: the batch's statement) equal that ``z``, plus
        ``forge_rows`` (row -> (C,) added to the committed ``w``).  ``edit_zk(z, lay)``: an in-place edit of the committed
        ZK rows after the masks are sampled (negatives: committed and proved as edited)."""
        dev = torch.device(self.device)
        tm = Timer(dev)
        lay = bt.lay
        cons = self.key_for(lay)[0]
        zk_on = lay.zk is not None
        if zk_on != bool(bt.params.get("zk")):
            raise ValueError("the batch's layout and its zk parameter disagree")
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        stmt = bt.stmt
        t_all = time.perf_counter()
        tm.start("witness")
        with self._witness_lock:
            subs = self._subs(bt, mutate)
            z = build_z(self.sys, lay, subs, dev)
        if ypub_override is not None:
            for name, v in end_values(self.sys, torch.as_tensor(ypub_override, device=dev, dtype=torch.int64)).items():
                z[lay.virt[name], :lay.l] = (v % P).to(torch.int32)
        forge = None
        if forge_z is not None:
            forge_z(z, lay)
        if forge_stmt is not None or forge_z is not None or forge_rows:
            forge = self._forge(z, lay, forge_stmt if forge_stmt is not None else stmt, forge_rows)
        tm.stop()
        if zk_on:
            tm.start("zk_masks")
            fill_zk(lay, z)                 # a fresh os.urandom key per proof (mask_sampler)
            if edit_zk is not None:
                edit_zk(z, lay)
            tm.stop()
        tm.start("transcript")
        coins.absorb(b"lgto/params", params_bytes(bt.params))
        coins.absorb(b"lgto/stmt", forge_stmt.digest() if forge_stmt is not None else (bt.stmt_digest or stmt.digest()))
        coins.absorb(b"lgto/vk", bt.key_sha256 or self.key_for(lay)[3])
        vr = slice(lay.m, lay.m + len(lay.virt)) if zk_on else slice(lay.m, None)
        virt = z[vr].clone()

        def committed_rows():
            z[vr] = 0
            for r, d in (forge or {}).items():
                z[r] = d.to(z.dtype)
        committed_rows()
        tm.stop()
        zl = None
        if zk_on:
            tm.start("zk_masks")
            zl = zk_layout(lay, bt.dims)
            f = zk_mod.z_to_f(z, zl).contiguous()
            tm.stop()
        else:
            f = z.view(-1)
        owner = self.pcs_impl == "pcs-fast" and self.pcs_fast is not False and _fast(z)
        t_pad = int(bt.params.get("t_pad", 0))
        if t_pad and not owner:
            raise ValueError("ZK column padding needs ligerito-pcs-fast's device PCS (pcs.commit / pcs.open)")
        if owner:
            from .proto import pcs as ppcs
            com = ppcs.commit(f, bt.dims, tm, stream=_stream_for_zero_check(f, bt.dims, dev), t_pad=t_pad)
            coins.absorb(b"root", com.root)
        else:
            com = pcs_commit(f, bt.dims, tm, fast=self.pcs_fast)
            coins.absorb(b"root", com.root1)
        del f
        # ZK on a card that streams its round-1 codeword: f = z_to_f(z) is a second copy of z (4 GiB at 2^30 cells) that the
        # zero-check never reads; it is rebuilt from the committed z for the opening (4090, fp8-ada 4096 VUs: OOM otherwise).
        # LIGERITO_ZK_REBUILD_F=1 forces it (gate sizes never stream)
        rebuild_f = zk_on and owner and (com.U1 is None or os.environ.get("LIGERITO_ZK_REBUILD_F") == "1")
        if rebuild_f:
            m1_shape, com.M1 = com.M1.shape, None
        tm.start("zero_check")
        z[vr] = virt
        if forge:
            scp = self._forged_sumcheck(lay, cons, z, coins, forge, cheat)
        else:
            scp = sc_mod.prove(lay, cons, z, coins, cheat=cheat, engine=self.engine,
                               schedule=self.sumcheck_schedule(lay))
        committed_rows()                    # the row-major commitment's M1 is a view of z: the openings read it
        del virt
        tm.stop()
        if rebuild_f:
            tm.start("zk_masks")
            com.M1 = zk_mod.z_to_f(z, zl).contiguous().view(m1_shape)
            z = None
            tm.stop()
        evals, sparse = split_claims(scp.claims)
        points, values = pcs_points(lay, evals, zl, zero_blocks(lay.virt))
        if owner:
            pp = ppcs.open(com, points, None, coins, tm, sparse=pcs_sparse(sparse, zl) if sparse else None)
            pcs = PcsProof(list(pp.roots), [list(m_) for m_ in pp.sumcheck], [o.rows for o in pp.opens],
                           [o.siblings for o in pp.opens], pp.final_y, list(pp.ybar))
        else:
            if sparse:
                raise ValueError("LGSC0004's sparse mask claims need ligerito-pcs-fast's device PCS (pcs.open(sparse=))")
            pcs = pcs_open(com, points, values, coins, tm, n_c=0 if zk_on else lay.C.bit_length() - 1)
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
            if stmt.l != self.l or prm.get("l") != self.l or stmt.steps != self.steps or stmt.K != self.rel.k \
                    or stmt.word_bytes != self.word_bytes or stmt.y_bytes != self.y_bytes:
                raise Reject("statement shape")
            real = [nv for nv in stmt.n_vus if nv > 0]
            if stmt.n_vus[:len(real)] != real or any(nv * self.steps > self.l for nv in real) or not real:
                raise Reject("statement sub-batch list")
            zk_on = bool(prm.get("zk"))
            lay = layout_for(self.sys, self.l, real, self.steps, zk=zk_on)
            if lay.S != stmt.S:
                raise Reject("statement S does not match its sub-batches")
            cons, _, _, key_sha = self.key_for(lay)
            if stmt.vk_sha256 != key_sha:
                raise Reject("statement names another verification key")
            dims = Dims(prm["n"], list(prm["kprime"]), list(prm["queries"]), list(prm["rate_log2"]))
            blocks = zero_blocks(lay.virt)
            if dims.n != lay.n_vars or prm.get("order") != ("zk-interleaved" if zk_on else "row-major") \
                    or prm.get("n_claims") != n_eval_claims(lay) + len(blocks) \
                    or [tuple(x) for x in prm.get("zero_blocks", [])] != blocks:
                raise Reject("proof parameters do not match the layout")
            if dims.k[-1] < 0 or min(dims.kprime) < 1:
                raise Reject("bad round structure")
            t_pad = int(prm.get("t_pad", 0))
            if zk_on != bool(t_pad) or (t_pad and t_pad != zk_mod.T_PAD):
                raise Reject("ZK padding does not match the parameters")
            snd = self.soundness(dims, t_pad, lay)
            if snd["total_log2"] > -128.0 + 1e-9:
                raise Reject(f"parameters give an interactive union of 2^{snd['total_log2']:.2f} > 2^-128")
            pub_rows = public_rows(self.sys, lay, self._stmt_subs(stmt, lay, real, dev), dev)
            info["t_public"] = time.perf_counter() - t0
            coins.absorb(b"lgto/params", proof.absorbed_params())
            coins.absorb(b"lgto/stmt", stmt.digest())
            coins.absorb(b"lgto/vk", key_sha)
            coins.absorb(b"root", proof.pcs.roots[0])
            try:
                scp = sc_mod.SumcheckProof.from_bytes(proof.sc_bytes, device=dev)
            except (ValueError, IndexError) as e:
                raise Reject(f"sumcheck bytes: {e}") from None
            t1 = time.perf_counter()
            ok, why, claims = sc_mod.verify(lay, cons, pub_rows, scp, coins, device=dev)
            info["t_sumcheck"] = time.perf_counter() - t1
            if not ok:
                raise Reject(why)
            # the verifier's reading of the prover's privacy (not soundness), as ligerito-verify's zk_mode
            info["zk_mode"] = sc_mod.zk_mode(lay, scp.schedule) if zk_on else "none"
            zl = zk_layout(lay, dims) if zk_on else None
            evals, sparse = split_claims(claims)
            points, values = pcs_points(lay, evals, zl, blocks)
            if sparse and not t_pad:
                raise Reject("sparse mask claims without the ZK PCS")
            t2 = time.perf_counter()
            if t_pad:
                from .proto import pcs as ppcs
                pp = ppcs.Proof(dims.n, list(dims.kprime), list(dims.queries), list(dims.rates), [tuple(x) for x in prm["splits"]],
                                values, list(proof.pcs.roots), [[np.asarray(g, dtype=np.int64) for g in m_] for m_ in proof.pcs.sumcheck],
                                [ppcs.RoundOpen(np.asarray(r_, dtype=np.int64), s_) for r_, s_ in zip(proof.pcs.rows, proof.pcs.siblings)],
                                np.asarray(proof.pcs.final_y, dtype=np.int64), order=ppcs.COMMIT_FIRST,
                                ybar=[np.asarray(y_, dtype=np.int64) for y_ in proof.pcs.ybar])
                ok_p, why_p = ppcs.verify_open(pp, points, values, coins, t_pad=t_pad,
                                               sparse=pcs_sparse(sparse, zl) if sparse else None)
                if not ok_p:
                    raise Reject(f"PCS: {why_p}")
            else:
                pcs_verify(proof.pcs, dims, prm["splits"], points, values, coins)
            info["t_pcs"] = time.perf_counter() - t2
            info["soundness"] = soundness_claim(snd, coins_kind(coins))
            info["union"] = snd
        except Reject as e:
            info["seconds"] = time.perf_counter() - t0
            return False, str(e), info
        info["seconds"] = time.perf_counter() - t0
        info["coins"] = coins.rounds
        return True, "accept", info

    # --- soundness --------------------------------------------------------------------------------------------------
    def soundness(self, dims: Dims, t_pad: int = 0, lay: Layout | None = None) -> dict:
        """The interactive union bound over the PCS levels (paper (17) with ligerito-design's sampling base) and the three
        sumchecks + batching coins over F_{p^6} (``params.LigeritoParams.soundness`` with the relation shape); what a run
        may claim from it depends on its coins: ``soundness_claim``.  ``lay``: the batch's layout (default: the relation's
        LGSC0003 key).  A ZK layout batches LGSC0004's 2 points + 3 sparse terms; its sumcheck terms (184/|F|) are counted
        as LGSC0003's (153/|F|): both ~2^-178."""
        lp = params_mod.LigeritoParams(tall1=1 << dims.k[0], wide_log2=tuple(dims.kprime), rate_log2=tuple(dims.rates),
                                       queries=tuple(dims.queries), zk=False, t_pad=t_pad,
                                       relation=params_mod.RELATIONS.get(self.rel.name))
        zk = lay is not None and lay.zk is not None
        virt, n_i = (lay.virt, lay.n_i) if lay is not None else (self.key.virt, self.key.n_i)
        zb = zero_blocks(virt)                      # the virtual rows are functions of the relation only
        J = (2 + 3 if zk else 1 + len(self.key.c_rows)) + len(zb)
        extra = {"PCS.claim batching beta J/|F|": math.log2(J) - lp.fields.ext_log2}
        if zb:
            extra["zero claims on the virtual rows, sum (b + n_c)/|F|"] = \
                math.log2(sum(b + dims.n - n_i for b, _ in zb)) - lp.fields.ext_log2
        return lp.soundness(extra)
