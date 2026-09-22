"""Ligero commitment over BabyBear (``backends/gkr/src/ligero.rs`` in torch; SHA-256 Merkle via ``hash_gpu``).

The committed vector ``x`` (length ``l``) is laid out as ``rows`` rows of ``k`` base elements; row ``i`` is the message
``X_i`` with ``X_i[m] = codeword_i[m · n/k]`` (the ``k``-th roots are a subgroup of the ``n``-th roots): the row is the
low-degree extension of its values at the ``k``-th roots evaluated at all ``n``-th roots (INTT_k, zero-pad, NTT_n).
The ``n`` columns are Merkle-committed with SHA-256 (leaf = the column's ``rows`` u32 LE, node = SHA-256(left||right)).

Opening ``⟨a, X⟩ = b`` for a public extension vector ``a`` (``rows × k``, zero padded):
  * ``r <- tr``; prover sends ``w = Σ_i r^i X_i`` (``k``) and the coefficients of ``q(Z) = Σ_i a_i(Z) U_i(Z)`` (``2k-1``),
    ``a_i`` / ``U_i`` the degree ``< k`` polynomials through row ``i`` of ``a`` / ``X`` on the ``k``-th roots;
  * verifier: ``Σ_{ζ∈H} q(ζ) = k (c_0 + c_k)`` must equal ``b``; ``t`` distinct random columns ``j``: Merkle path,
    ``Σ_i r^i C[i,j] = Enc(w)[j]`` (proximity), ``q(η_j) = Σ_i a_i(η_j) C[i,j]`` (linear test).
Parameters: ``k = 4096, n = 16384, t = 192`` as the CPU crate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .field import DEG, P, Ext, e_add, e_mul, e_scale, ext_mul, ext_scale, inv_mod, py_exts, root_of_unity, t_ext, t_exts
from .gkr import Reader, VerifyError
from .transcript import Transcript


@dataclass(frozen=True)
class LigeroParams:
    k: int = 4096
    n: int = 16384
    t: int = 192

    def rows_for(self, l: int) -> int:
        return max(1, (l + self.k - 1) // self.k)

    @property
    def depth(self) -> int:
        return self.n.bit_length() - 1


DEFAULT = LigeroParams()
ROW_CHUNK = 512

# ---------------------------------------------------------------------------------------------------------------------
# NTT
# ---------------------------------------------------------------------------------------------------------------------

_TW: dict[tuple, object] = {}


def _bitrev(n: int, device) -> torch.Tensor:
    key = ("rev", n, str(device))
    t = _TW.get(key)
    if t is None:
        bits = n.bit_length() - 1
        idx = torch.arange(n)
        rev = torch.zeros(n, dtype=torch.int64)
        for b in range(bits):
            rev |= ((idx >> b) & 1) << (bits - 1 - b)
        t = rev.to(device)
        _TW[key] = t
    return t


def _twiddles(n: int, inverse: bool, device) -> list[torch.Tensor]:
    key = ("tw", n, inverse, str(device))
    t = _TW.get(key)
    if t is None:
        w = root_of_unity(n)
        if inverse:
            w = inv_mod(w)
        t = []
        length = 2
        while length <= n:
            wl = pow(w, n // length, P)
            tw = [1]
            for _ in range(length // 2 - 1):
                tw.append(tw[-1] * wl % P)
            t.append(torch.tensor(tw, dtype=torch.int64, device=device))
            length *= 2
        _TW[key] = t
    return t


def ntt(v: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Row-wise NTT of ``(rows, n)`` int64, natural order in and out (``ntt.rs`` semantics: forward evaluates
    ``Σ_m c_m ω^{mj}``)."""
    rows, n = v.shape
    dev = v.device
    a = v.to(torch.int64)[:, _bitrev(n, dev)].contiguous()
    length = 2
    for tw in _twiddles(n, inverse, dev):
        half = length // 2
        a = a.view(rows, n // length, length)
        lo = a[:, :, :half]
        hi = (a[:, :, half:] * tw) % P
        a = torch.cat([(lo + hi) % P, (lo - hi) % P], dim=2)
        length *= 2
    a = a.reshape(rows, n)
    if inverse:
        a = (a * inv_mod(n)) % P
    return a


def encode_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    """``(rows, k)`` values at the ``k``-th roots -> ``(rows, n)`` values at the ``n``-th roots."""
    rows, k = x.shape
    coeffs = ntt(x, inverse=True)
    padded = torch.zeros((rows, n), dtype=torch.int64, device=x.device)
    padded[:, :k] = coeffs
    return ntt(padded)


def encode_ext(u: torch.Tensor, n: int) -> torch.Tensor:
    """``(..., k, 6)`` -> ``(..., n, 6)`` coordinate-wise."""
    shape = u.shape[:-2]
    k = u.shape[-2]
    flat = u.reshape(-1, k, DEG).transpose(1, 2).reshape(-1, k)
    enc = encode_rows(flat, n).reshape(-1, DEG, n).transpose(1, 2)
    return enc.reshape(*shape, n, DEG).contiguous()


# ---------------------------------------------------------------------------------------------------------------------
# Merkle (hash_gpu on CUDA, hashlib on CPU)
# ---------------------------------------------------------------------------------------------------------------------

class Merkle:
    def __init__(self, code: torch.Tensor) -> None:
        rows, n = code.shape
        self.rows, self.n = rows, n
        if code.is_cuda:
            import cupy as cp

            from hash_gpu.merkle import commit as gpu_commit, digest_bytes

            m = cp.asarray(code.to(torch.int32).contiguous()).view(cp.uint32)
            self.tree = gpu_commit(m, "sha256")
            self.root = digest_bytes(self.tree.root, "sha256")
            self._gpu = True
        else:
            a = code.cpu().numpy().astype("<u4")
            level = [hashlib.sha256(np.ascontiguousarray(a[:, j]).tobytes()).digest() for j in range(n)]
            self.levels = [level]
            while len(level) > 1:
                level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
                self.levels.append(level)
            self.root = level[0]
            self._gpu = False

    def path(self, j: int) -> list[bytes]:
        if self._gpu:
            from hash_gpu.merkle import digest_bytes

            lv = [l.get() for l in self.tree.levels[:-1]] if not hasattr(self, "_host") else self._host
            self._host = lv
            out, idx = [], j
            for level in lv:
                out.append(np.asarray(level[idx ^ 1]).astype(">u4").tobytes())
                idx >>= 1
            return out
        out, idx = [], j
        for lvl in self.levels[:-1]:
            out.append(lvl[idx ^ 1])
            idx >>= 1
        return out


def merkle_verify(root: bytes, j: int, leaf: bytes, path: list[bytes]) -> bool:
    h = hashlib.sha256(leaf).digest()
    idx = j
    for sib in path:
        h = hashlib.sha256((sib + h) if idx & 1 else (h + sib)).digest()
        idx >>= 1
    return h == root


# ---------------------------------------------------------------------------------------------------------------------
# Commitment
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Committed:
    params: LigeroParams
    length: int
    rows: int
    x_rows: torch.Tensor        # (rows, k) int64
    code: torch.Tensor          # (rows, n) int32
    merkle: Merkle

    @property
    def root(self) -> bytes:
        return self.merkle.root


def commit(x: torch.Tensor, params: LigeroParams = DEFAULT) -> Committed:
    l = x.shape[0]
    rows = params.rows_for(l)
    xr = torch.zeros((rows * params.k,), dtype=torch.int64, device=x.device)
    xr[:l] = x.to(torch.int64)
    xr = xr.view(rows, params.k)
    code = torch.empty((rows, params.n), dtype=torch.int32, device=x.device)
    for lo in range(0, rows, ROW_CHUNK * 8):
        code[lo:lo + ROW_CHUNK * 8] = encode_rows(xr[lo:lo + ROW_CHUNK * 8], params.n).to(torch.int32)
    return Committed(params, l, rows, xr, code, Merkle(code))


# ---------------------------------------------------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Opening:
    w: list[Ext]                # k
    qc: list[Ext]               # 2k - 1
    columns: list[tuple[list[int], list[bytes]]]   # per opened column: values (rows) + path

    def nbytes(self) -> int:
        b = (len(self.w) + len(self.qc)) * 4 * DEG
        for col, path in self.columns:
            b += 4 * len(col) + 32 * len(path)
        return b


def open_set(tr: Transcript, n: int, t: int) -> list[int]:
    s: list[int] = []
    while len(s) < t:
        j = tr.challenge()[0] % n
        if j not in s:
            s.append(j)
    return s


def row_coeffs(tr: Transcript, rows: int, device) -> torch.Tensor:
    r = tr.challenge()
    v = [(1,) + (0,) * (DEG - 1)]
    for _ in range(1, rows):
        v.append(e_mul(v[-1], r))
    return t_exts(v, device)


def _pad_a(a: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    full = torch.zeros((rows * k, DEG), dtype=torch.int64, device=a.device)
    full[: a.shape[0]] = a.to(torch.int64)
    return full.view(rows, k, DEG)


def prove_open(cm: Committed, a: torch.Tensor, tr: Transcript) -> Opening:
    """``a`` ``(l, 6)`` int64 (the linear functional); ``w`` and ``qc`` travel in the Opening."""
    p = cm.params
    dev = a.device
    rows, k, n = cm.rows, p.k, p.n
    r = row_coeffs(tr, rows, dev)                                     # (rows, 6)
    ar = _pad_a(a, rows, k)                                           # (rows, k, 6)
    code = cm.code
    w = torch.zeros((k, DEG), dtype=torch.int64, device=dev)
    qd = torch.zeros((n, DEG), dtype=torch.int64, device=dev)
    for lo in range(0, rows, ROW_CHUNK):
        hi = min(rows, lo + ROW_CHUNK)
        xr = cm.x_rows[lo:hi]                                         # (c, k)
        w = (w + (r[lo:hi].unsqueeze(1) * xr.unsqueeze(-1) % P).sum(dim=0)) % P
        ae = encode_ext(ar[lo:hi], n)                                 # (c, n, 6)
        cw = code[lo:hi].to(torch.int64)                              # (c, n)
        qd = (qd + (ae * cw.unsqueeze(-1) % P).sum(dim=0)) % P
    q = ntt(qd.T.contiguous(), inverse=True).T                        # (n, 6) coefficients
    qc = q[: 2 * k - 1]
    w_py, qc_py = py_exts(w), py_exts(qc)
    tr.absorb_exts(w_py)
    tr.absorb_exts(qc_py)
    js = open_set(tr, n, p.t)
    code_cols = code[:, torch.tensor(js, dtype=torch.int64, device=dev)].cpu()
    cols = []
    for oi, j in enumerate(js):
        cols.append(([int(v) & 0xFFFFFFFF for v in code_cols[:, oi].tolist()], cm.merkle.path(j)))
    return Opening(w_py, qc_py, cols)


def verify_open(root: bytes, params: LigeroParams, length: int, a: torch.Tensor, b: Ext, op: Opening, tr: Transcript, device) -> None:
    p = params
    rows, k, n = p.rows_for(length), p.k, p.n
    r = row_coeffs(tr, rows, device)
    if len(op.w) != k or len(op.qc) != 2 * k - 1 or len(op.columns) != p.t:
        raise VerifyError("ligero: malformed opening")
    tr.absorb_exts(op.w)
    tr.absorb_exts(op.qc)
    if e_scale(e_add(op.qc[0], op.qc[k]), k) != b:
        raise VerifyError("ligero: linear functional value mismatch")
    js = open_set(tr, n, p.t)
    w_enc = py_exts(encode_ext(t_exts(op.w, device), n))
    ar = _pad_a(a, rows, k)
    jt = torch.tensor(js, dtype=torch.int64, device=device)
    # a_i(η_j) at the opened columns only (encode every row of a; keep t positions)
    a_at = torch.empty((rows, p.t, DEG), dtype=torch.int64, device=device)
    for lo in range(0, rows, ROW_CHUNK):
        hi = min(rows, lo + ROW_CHUNK)
        a_at[lo:hi] = encode_ext(ar[lo:hi], n)[:, jt]
    cols = torch.tensor([c for c, _ in op.columns], dtype=torch.int64, device=device).T   # (rows, t)
    if bool((cols >= P).any()):
        raise VerifyError("ligero: non-canonical column value")
    lhs = py_exts((r.unsqueeze(1) * cols.unsqueeze(-1) % P).sum(dim=0) % P)               # (t, 6)
    rhs = py_exts((a_at * cols.unsqueeze(-1) % P).sum(dim=0) % P)
    omega = root_of_unity(n)
    for oi, j in enumerate(js):
        col, path = op.columns[oi]
        if len(col) != rows or len(path) != p.depth:
            raise VerifyError("ligero: opening shape")
        if not merkle_verify(root, j, np.array(col, dtype="<u4").tobytes(), path):
            raise VerifyError(f"ligero: merkle path {j}")
        if lhs[oi] != w_enc[j]:
            raise VerifyError(f"ligero: proximity test at column {j}")
        eta = pow(omega, j, P)
        qv = (0,) * DEG
        for c in reversed(op.qc):
            qv = e_add(e_scale(qv, eta), c)
        if qv != rhs[oi]:
            raise VerifyError(f"ligero: linear test at column {j}")
