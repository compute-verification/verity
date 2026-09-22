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
import os
from dataclasses import dataclass

import numpy as np
import torch

from .field import (
    DEG,
    Ext,
    Limbs8,
    P,
    e_add,
    e_scale,
    geometric,
    inv_mod,
    mm_mod_int8,
    py_exts,
    root_of_unity,
    t_exts,
)
from .gkr import VerifyError
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
            from hash_gpu.merkle import commit as gpu_commit
            from hash_gpu.merkle import digest_bytes

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


_ENC: dict = {}


def simt_encoder(k: int, n: int):
    """The b-encode lane's SIMT Reed-Solomon encoder (``backends/direct/encode/encode_ntt_simt.py``: one CTA per row,
    INTT_k + three coset NTT_k in shared memory, 32-bit Montgomery; BabyBear, ``n = 4k``, the same ``ω = 31^((p-1)/n)``
    as ``ntt``) or ``None`` when it is not importable / ``VERITY_GPU_TORCH_NTT`` is set.  Its codeword is coset-major
    (column ``c k + j`` = evaluation at ``ω^(4j + c)``); ``commit`` permutes it to the natural order ``ntt`` produces."""
    if os.environ.get("VERITY_GPU_TORCH_NTT") or n != 4 * k:
        return None
    key = (k, n)
    if key not in _ENC:
        try:
            try:
                from backends.direct.encode import babybear as bb
                from backends.direct.encode.encode_ntt_simt import RSEncoderSIMT
            except ImportError:
                import sys
                from pathlib import Path

                sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
                from backends.direct.encode import babybear as bb
                from backends.direct.encode.encode_ntt_simt import RSEncoderSIMT
            if bb.P != P or bb.root_of_unity(n) != root_of_unity(n):
                raise ImportError("encoder field / root mismatch")
            _ENC[key] = (RSEncoderSIMT(k, n), torch.from_numpy(bb.coset_to_natural(n, k).astype("int64")))
        except Exception:  # noqa: BLE001 -- no cupy / NVRTC: the torch NTT path
            _ENC[key] = None
    return _ENC[key]


def encode_rows_device(xr: torch.Tensor, n: int) -> torch.Tensor:
    """``(rows, k)`` int64 -> ``(rows, n)`` int32 codewords in natural order: the SIMT encoder on CUDA (bit-exact with
    ``encode_rows``), else the torch NTT."""
    rows, k = xr.shape
    enc = simt_encoder(k, n) if xr.is_cuda else None
    if enc is None:
        code = torch.empty((rows, n), dtype=torch.int32, device=xr.device)
        for lo in range(0, rows, ROW_CHUNK * 8):
            code[lo:lo + ROW_CHUNK * 8] = encode_rows(xr[lo:lo + ROW_CHUNK * 8], n).to(torch.int32)
        return code
    import cupy as cp

    kernel, perm = enc
    out = cp.empty((rows, n), dtype=cp.int32)
    kernel(cp.asarray(xr.to(torch.int32).contiguous()).view(cp.uint32), out.view(cp.uint32))
    code_coset = torch.as_tensor(out, device=xr.device)
    return code_coset[:, perm.to(xr.device)].contiguous()


def commit(x: torch.Tensor, params: LigeroParams = DEFAULT) -> Committed:
    l = x.shape[0]
    rows = params.rows_for(l)
    xr = torch.zeros((rows * params.k,), dtype=torch.int64, device=x.device)
    xr[:l] = x.to(torch.int64)
    xr = xr.view(rows, params.k)
    code = encode_rows_device(xr, params.n)
    return Committed(params, l, rows, xr, code, Merkle(code))


# ---------------------------------------------------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Opening:
    w: list[Ext]                # k
    qc: list[Ext]               # 2k - 1
    columns: np.ndarray         # (t, rows) '<u4': the opened columns' values
    paths: list[list[bytes]]    # per opened column its Merkle path (depth siblings)

    def nbytes(self) -> int:
        b = (len(self.w) + len(self.qc)) * 4 * DEG
        b += 4 * self.columns.size
        for path in self.paths:
            b += 32 * len(path)
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
    return geometric((1,) + (0,) * (DEG - 1), r, rows, device)


def _pad_a(a: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    full = torch.zeros((rows * k, DEG), dtype=torch.int64, device=a.device)
    full[: a.shape[0]] = a.to(torch.int64)
    return full.view(rows, k, DEG)


def _a_transposed(a: torch.Tensor, rows: int, k: int, int8: bool):
    """``aᵀ`` as the ``(k·6, rows)`` GEMM operand ``af[(i, c), r] = a[r k + i, c]`` (zero past ``len(a)``), written once
    in int32 straight into ``Limbs8``'s padded layout on the INT8 path (no int64 ``(rows, k, 6)`` copies: the functional
    is 26 GB at the full VU batch and the transposed copies were the opening's memory peak)."""
    l = a.shape[0]
    if not int8:
        return _pad_a(a, rows, k).permute(1, 2, 0).reshape(k * DEG, rows)
    M = k * DEG
    Mp, Kp = Limbs8.padded_shape(M, rows, "a")
    af = torch.zeros((Mp, Kp), dtype=torch.int32, device=a.device)
    full_rows = l // k
    if full_rows:
        af[:M, :full_rows] = a[: full_rows * k].view(full_rows, M).t()
    rem = l - full_rows * k
    if rem:
        af[: rem * DEG, full_rows] = a[full_rows * k:].reshape(-1)
    return Limbs8(af, "a", shape=(M, rows))


def _intt_matrix(k: int, device) -> torch.Tensor:
    """``M[j, m] = k^-1 ω_k^{-jm}``: ``coeffs = values @ M`` (the same map as ``ntt(·, inverse=True)``)."""
    key = ("intt_mat", k, str(device))
    t = _TW.get(key)
    if t is None:
        w_inv = inv_mod(root_of_unity(k))
        pw = [1]
        for _ in range(k - 1):
            pw.append(pw[-1] * w_inv % P)
        pw_t = torch.tensor(pw, dtype=torch.int64)
        idx = (torch.arange(k).unsqueeze(1) * torch.arange(k).unsqueeze(0)) % k
        t = ((pw_t[idx] * inv_mod(k)) % P).to(device)
        _TW[key] = t
    return t


def _mm_mod_rows(A: torch.Tensor, B: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """``(A @ B) mod p`` with the contraction axis (``A`` columns / ``B`` rows) processed in chunks."""
    from .field import mm_mod

    n = A.shape[1]
    out = None
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        part = mm_mod(A[:, lo:hi], B[lo:hi])
        out = part if out is None else (out + part) % P
    return out


def use_int8(dev) -> bool:
    """INT8 tensor-core limb matmuls (``field.mm_mod_int8``) on CUDA unless ``VERITY_GPU_FP64_MM`` is set (the fp64
    16-bit-limb path of v2e stays the reference)."""
    return dev.type == "cuda" and not os.environ.get("VERITY_GPU_FP64_MM")


def _mm(A, B) -> torch.Tensor:
    """Modular matmul on the device's fast path; ``A``/``B`` may be pre-split ``Limbs8`` (INT8 path only)."""
    a_t = A if isinstance(A, torch.Tensor) else A.limbs[0]
    if use_int8(a_t.device) and (isinstance(A, Limbs8) or isinstance(B, Limbs8) or a_t.is_cuda):
        return mm_mod_int8(A, B)
    return _mm_mod_rows(A, B)


def open_w_qc(cm: Committed, a: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w = Σ_i r_i X_i`` ``(k, 6)`` and the ``2k-1`` coefficients of ``q(Z) = Σ_i a_i(Z) U_i(Z)`` ``(2k-1, 6)``,
    all as modular matmuls (tensor cores): with ``M`` the INTT matrix, ``q``'s coefficient matrix is
    ``G = Mᵀ (aᵀ X) M`` (``G[u, c, v]`` = coefficient ``u`` of ``a``-side times coefficient ``v`` of ``X``-side,
    summed over rows) and ``qc[m] = Σ_{u+v=m} G[u, ·, v]``.  Identical to encode-multiply-INTT over the ``n``-domain."""
    p = cm.params
    dev = a.device
    rows, k = cm.rows, p.k
    int8 = use_int8(dev)
    xr = Limbs8(cm.x_rows, "b") if int8 else cm.x_rows                # (rows, k), the B operand twice
    w = _mm(r.T.contiguous(), xr).T.contiguous()                      # (k, 6)
    af = _a_transposed(a, rows, k, int8)                              # aᵀ: (k·6, rows)
    H = _mm(af, xr).reshape(k, DEG, k)                                # [u_val, c, i_val]
    del af, xr
    M = _intt_matrix(k, dev)
    Mt = M.T.contiguous()
    if int8:
        M, Mt = Limbs8(M, "b"), Limbs8(Mt, "a")
    G1 = _mm(H.permute(1, 0, 2).reshape(DEG * k, k), M)               # (H M): (6·k, k) -> [c, u_val, v]
    G = _mm(Mt, G1.reshape(DEG, k, k).permute(1, 0, 2).reshape(k, DEG * k))   # Mᵀ (H M): [u, (c, v)]
    G = G.reshape(k, DEG, k)
    idx = (torch.arange(k, device=dev).unsqueeze(1) + torch.arange(k, device=dev).unsqueeze(0)).reshape(-1)
    qc = torch.zeros((2 * k - 1, DEG), dtype=torch.int64, device=dev)
    qc.index_add_(0, idx, G.permute(0, 2, 1).reshape(k * k, DEG))
    return w, qc % P


def open_w_qc_reference(cm: Committed, a: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The v1 path: encode ``a`` rows on the ``n``-domain, multiply by the codewords, INTT (NTT butterflies in torch)."""
    p = cm.params
    dev = a.device
    rows, k, n = cm.rows, p.k, p.n
    ar = _pad_a(a, rows, k)
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
    return w, q[: 2 * k - 1]


def prove_open(cm: Committed, a: torch.Tensor, tr: Transcript) -> Opening:
    """``a`` ``(l, 6)`` int64 (the linear functional); ``w`` and ``qc`` travel in the Opening."""
    p = cm.params
    dev = a.device
    rows, n = cm.rows, p.n
    r = row_coeffs(tr, rows, dev)                                     # (rows, 6)
    code = cm.code
    w, qc = open_w_qc(cm, a, r)
    w_py, qc_py = py_exts(w), py_exts(qc)
    tr.absorb_exts(w_py)
    tr.absorb_exts(qc_py)
    js = open_set(tr, n, p.t)
    code_cols = np.ascontiguousarray(code[:, torch.tensor(js, dtype=torch.int64, device=dev)].T.contiguous().cpu().numpy().view("<u4"))
    return Opening(w_py, qc_py, code_cols, [cm.merkle.path(j) for j in js])


def verify_open(root: bytes, params: LigeroParams, length: int, a: torch.Tensor, b: Ext, op: Opening, tr: Transcript, device) -> None:
    p = params
    rows, k, n = p.rows_for(length), p.k, p.n
    r = row_coeffs(tr, rows, device)
    if len(op.w) != k or len(op.qc) != 2 * k - 1 or op.columns.shape != (p.t, rows) or len(op.paths) != p.t:
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
    cols = torch.from_numpy(op.columns.astype("int64")).to(device).T                      # (rows, t)
    if bool((cols >= P).any()):
        raise VerifyError("ligero: non-canonical column value")
    lhs = py_exts((r.unsqueeze(1) * cols.unsqueeze(-1) % P).sum(dim=0) % P)               # (t, 6)
    rhs = py_exts((a_at * cols.unsqueeze(-1) % P).sum(dim=0) % P)
    omega = root_of_unity(n)
    for oi, j in enumerate(js):
        path = op.paths[oi]
        if len(path) != p.depth:
            raise VerifyError("ligero: opening shape")
        if not merkle_verify(root, j, np.ascontiguousarray(op.columns[oi]).astype("<u4").tobytes(), path):
            raise VerifyError(f"ligero: merkle path {j}")
        if lhs[oi] != w_enc[j]:
            raise VerifyError(f"ligero: proximity test at column {j}")
        eta = pow(omega, j, P)
        qv = (0,) * DEG
        for c in reversed(op.qc):
            qv = e_add(e_scale(qv, eta), c)
        if qv != rhs[oi]:
            raise VerifyError(f"ligero: linear test at column {j}")
