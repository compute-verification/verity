"""Ligero commitment over BabyBear (``backends/gkr/src/ligero.rs`` in torch; SHA-512 Merkle via ``sha512_cuda``).

The committed vector ``x`` (length ``l``) is laid out as ``rows`` rows of ``k`` base elements; row ``i`` is the message
``X_i`` with ``X_i[m] = codeword_i[m · n/k]`` (the ``k``-th roots are a subgroup of the ``n``-th roots): the row is the
low-degree extension of its values at the ``k``-th roots evaluated at all ``n``-th roots (INTT_k, zero-pad, NTT_n).
The ``n`` columns are Merkle-committed with SHA-512 (leaf = the column's ``rows`` u32 LE, node = SHA-512(left||right),
64-byte digests; proof.bin version 2).  SHA-256's generic collision term is 2^-128 by itself at q = 2^64 hash
evaluations, which left the protocol total at 2^-127.7 (README "Soundness").

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
    exts_to_bytes,
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
# Merkle (sha512_cuda on CUDA, hashlib on CPU)
# ---------------------------------------------------------------------------------------------------------------------

MERKLE_HASH = "SHA-512"
DIGEST_BYTES = 64


class Merkle:
    def __init__(self, code: torch.Tensor) -> None:
        rows, n = code.shape
        self.rows, self.n = rows, n
        if code.is_cuda:
            import cupy as cp

            from . import sha512_cuda

            m = cp.asarray(code.to(torch.int32).contiguous()).view(cp.uint32)
            self.tree_levels = sha512_cuda.commit(m)
            self.root = sha512_cuda.digest_bytes(self.tree_levels[-1][0])
            self._gpu = True
        else:
            a = code.cpu().numpy().astype("<u4")
            level = [hashlib.sha512(np.ascontiguousarray(a[:, j]).tobytes()).digest() for j in range(n)]
            self.levels = [level]
            while len(level) > 1:
                level = [hashlib.sha512(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
                self.levels.append(level)
            self.root = level[0]
            self._gpu = False

    def path(self, j: int) -> list[bytes]:
        if self._gpu:
            if not hasattr(self, "_host"):              # every level big-endian once, not per node (t · depth nodes)
                self._host = [np.asarray(l.get(), dtype=np.uint64).astype(">u8") for l in self.tree_levels[:-1]]
            out, idx = [], j
            for level in self._host:
                out.append(level[idx ^ 1].tobytes())
                idx >>= 1
            return out
        out, idx = [], j
        for lvl in self.levels[:-1]:
            out.append(lvl[idx ^ 1])
            idx >>= 1
        return out


def merkle_verify(root: bytes, j: int, leaf: bytes, path: list[bytes]) -> bool:
    h = hashlib.sha512(leaf).digest()
    idx = j
    for sib in path:
        if len(sib) != DIGEST_BYTES:
            return False
        h = hashlib.sha512((sib + h) if idx & 1 else (h + sib)).digest()
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
            # radix-4 stages with two CTAs per SM: 5.1 -> 3.6 ms for the opening's 70k-row encode on the 5090, same words
            enc = RSEncoderSIMT(k, n, radix4=k >= 4 * 512, min_blocks_per_sm=2)
            _ENC[key] = (enc, torch.from_numpy(bb.coset_to_natural(n, k).astype("int64")))
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
    raw: tuple[bytes, bytes] | None = None    # the prover's serialized (w, qc), as absorbed

    def nbytes(self) -> int:
        b = (len(self.w) + len(self.qc)) * 4 * DEG
        b += 4 * self.columns.size
        for path in self.paths:
            b += DIGEST_BYTES * len(path)
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


def _write_a_transposed(a: torch.Tensor, rows: int, k: int, af: torch.Tensor) -> None:
    """``af[(i, c), r] = a[r k + i, c]`` (zero past ``len(a)``) into the top-left ``(k·6, rows)`` of the zeroed ``af``."""
    l = a.shape[0]
    M = k * DEG
    full_rows = l // k
    if full_rows:
        af[:M, :full_rows] = a[: full_rows * k].view(full_rows, M).t()
    rem = l - full_rows * k
    if rem:
        af[: rem * DEG, full_rows] = a[full_rows * k:].reshape(-1)


def _a_transposed(a: torch.Tensor, rows: int, k: int, int8: bool):
    """``aᵀ`` as the ``(k·6, rows)`` GEMM operand, written once in int32 straight into ``Limbs8``'s padded layout on
    the INT8 path (no int64 ``(rows, k, 6)`` copies: those were the opening's memory peak)."""
    if not int8:
        return _pad_a(a, rows, k).permute(1, 2, 0).reshape(k * DEG, rows)
    M = k * DEG
    Mp, Kp = Limbs8.padded_shape(M, rows, "a")
    af = torch.zeros((Mp, Kp), dtype=torch.int32, device=a.device)
    _write_a_transposed(a, rows, k, af)
    return Limbs8(af, "a", shape=(M, rows))


def _omega_powers(n: int, device) -> torch.Tensor:
    """``ω_n^e`` for ``e < n`` as int64."""
    key = ("omega_pows", n, str(device))
    t = _TW.get(key)
    if t is None:
        w = root_of_unity(n)
        pw = [1]
        for _ in range(n - 1):
            pw.append(pw[-1] * w % P)
        t = _TW[key] = torch.tensor(pw, dtype=torch.int64, device=device)
    return t


def eval_rows_at(a: torch.Tensor, rows: int, k: int, js: list[int], n: int, device) -> torch.Tensor:
    """``a_i(η_j)`` for every row ``i`` and opened column ``j`` as ``(rows, t, 6)``: the rows are values on the
    ``k``-domain, so ``a_i(η_j) = Σ_m (INTT a_i)_m η_j^m`` = one modular GEMM ``(M V)ᵀ aᵀ`` with ``M`` the INTT matrix
    and ``V[m, j] = ω_n^(j m)`` -- the same evaluation as encoding every row on the ``n``-domain and keeping ``t``
    positions (``encode_ext(·)[:, js]``), without the 43k-row NTT."""
    t = len(js)
    jt = torch.tensor(js, dtype=torch.int64, device=device)
    exps = (jt[None, :] * torch.arange(k, dtype=torch.int64, device=device)[:, None]) % n     # (k, t)
    V = _omega_powers(n, device)[exps]                                                        # (k, t)
    MV = _mm_mod_rows(_intt_matrix(k, device), V)                                             # (k, t)
    aT = torch.zeros((k * DEG, rows), dtype=torch.int32, device=device)
    _write_a_transposed(a, rows, k, aT)
    aT = aT.view(k, DEG * rows)
    if not use_int8(aT.device):
        aT = aT.to(torch.int64)
    out = _mm(MV.T.contiguous(), aT)                                                          # (t, 6·rows)
    return out.view(t, DEG, rows).permute(2, 0, 1)                                            # (rows, t, 6)


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
    a = a.to(torch.int64)
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


def opening_path(dev, params: LigeroParams = DEFAULT) -> str:
    """How ``prove_open`` computes ``w`` / ``qc`` on ``dev`` (for result records)."""
    from . import kernels

    mm = "INT8 tensor-core limb GEMM" if use_int8(dev) else "fp64-limb GEMM"
    if (dev.type == "cuda" and simt_encoder(params.k, params.n) is not None and kernels.AVAILABLE
            and not os.environ.get("VERITY_GPU_QC_GEMM")):
        return f"q by evaluation (SIMT encode of a's rows, split-row reduction, INTT_n); w by {mm}"
    return f"w and q by {mm}s"


def open_row_chunk(rows: int, k: int, dev) -> int:
    """Rows per block of ``open_w_qc_eval``'s encode + reduction (the result is the same for any block size: the
    blocks' reductions are summed mod p).  All rows at once on an 80 GB part (the recorded A100 / H100 path); on a
    smaller one (RTX 4090, 24 GB) blocks of ``VERITY_GPU_OPEN_ROWS`` (default 4096) rows, whose ``at`` + ``ev``
    (``6 * 4k * 4`` bytes per row, 1.6 GB per 4096 rows) fit beside the committed code and the functional."""
    env = int(os.environ.get("VERITY_GPU_OPEN_ROWS", "0"))
    if env > 0:
        return env
    if dev.type == "cuda" and torch.cuda.get_device_properties(dev).total_memory < 40 * 2**30:
        return 4096
    return rows


def open_w_qc_eval(cm: Committed, a: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``open_w_qc`` by evaluation: ``deg q <= 2k - 2 < n``, so ``q`` is fixed by ``q(ω^m) = Σ_i a_i(ω^m) U_i(ω^m)``
    over the ``n``-domain, where ``U_i(ω^m)`` is the committed codeword and ``a_i(ω^m)`` is row ``i`` of ``a``
    itself on coset 0 and the SIMT encoder's cosets 1..3 (every row and coordinate encoded once, as the commitment
    encodes ``X``); one split-row reduction (``kernels.row_code_dot``) and one ``INTT_n`` give the coefficients.  The
    ~2.7T-multiply-add ``aᵀ X`` GEMM of ``open_w_qc`` becomes ``6 rows`` encodings + ``6 rows n`` multiply-adds.  Falls
    back to ``open_w_qc`` without the SIMT encoder / Triton; ``VERITY_GPU_QC_GEMM`` forces the GEMM route."""
    from . import kernels

    p = cm.params
    dev = a.device
    rows, k, n = cm.rows, p.k, p.n
    enc = simt_encoder(k, n) if a.is_cuda else None
    if enc is None or not kernels.AVAILABLE or os.environ.get("VERITY_GPU_QC_GEMM"):
        return open_w_qc(cm, a, r)
    import cupy as cp

    w = kernels.eq_rows_dot(r, cm.x_rows)                             # (k, 6), one read of the rows
    l = a.shape[0]
    full = l // k
    q_eval = None
    ch = open_row_chunk(rows, k, dev)
    for lo in range(0, rows, ch):
        hi = min(rows, lo + ch)
        at = torch.zeros((hi - lo, DEG, k), dtype=torch.int32, device=dev)   # at[i - lo, d, j] = a[i k + j, d]
        f_hi = min(full, hi)
        if f_hi > lo:
            at[: f_hi - lo] = a[lo * k: f_hi * k].reshape(f_hi - lo, k, DEG).permute(0, 2, 1)
        if l - full * k and lo <= full < hi:
            at[full - lo, :, : l - full * k] = a[full * k:].T
        at = at.view((hi - lo) * DEG, k)
        ev = torch.empty(((hi - lo) * DEG, 3 * k), dtype=torch.int32, device=dev)   # torch's pool, not cupy's
        enc[0](cp.asarray(at).view(cp.uint32), cp.asarray(ev).view(cp.uint32), write_systematic=False)
        part = kernels.row_code_dot(at, ev, cm.code[lo:hi], split=64)
        q_eval = part if q_eval is None else (q_eval + part) % P
        del at, ev
    q = ntt(q_eval, inverse=True)                                     # (6, n) coefficients
    if bool(q[:, 2 * k - 1:].any()):
        raise RuntimeError("open_w_qc_eval: q has coefficients past 2k - 2 (encoder / reduction mismatch)")
    return w, q[:, : 2 * k - 1].T.contiguous()


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


def prove_open(cm: Committed, a: torch.Tensor, tr: Transcript, timers: dict | None = None) -> Opening:
    """``a`` ``(l, 6)`` int64 (the linear functional); ``w`` and ``qc`` travel in the Opening.  ``timers["wq"]``: the
    seconds up to ``w`` / ``qc`` on the host (the rest is the column opening)."""
    import time

    t0 = time.perf_counter()
    p = cm.params
    dev = a.device
    rows, n = cm.rows, p.n
    r = row_coeffs(tr, rows, dev)                                     # (rows, 6)
    code = cm.code
    w, qc = open_w_qc_eval(cm, a, r)
    h = torch.cat([w, qc]).to(torch.int64).cpu().numpy()
    w_np, qc_np = h[: w.shape[0]], h[w.shape[0]:]
    w_py, qc_py = list(map(tuple, w_np.tolist())), list(map(tuple, qc_np.tolist()))
    if timers is not None:
        timers["wq"] = time.perf_counter() - t0
    raw = (exts_to_bytes(w_np), exts_to_bytes(qc_np))
    tr.absorb(raw[0])
    tr.absorb(raw[1])
    js = open_set(tr, n, p.t)
    sel = code[:, torch.tensor(js, dtype=torch.int64, device=dev)].T.contiguous()
    host = torch.empty(sel.shape, dtype=sel.dtype, pin_memory=dev.type == "cuda")
    host.copy_(sel)
    code_cols = np.ascontiguousarray(host.numpy().view("<u4"))
    return Opening(w_py, qc_py, code_cols, [cm.merkle.path(j) for j in js], raw)


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
    jt = torch.tensor(js, dtype=torch.int64, device=device)
    # a_i(η_j) at the opened columns only (one GEMM; = encode every row of a and keep t positions)
    a_at = eval_rows_at(a, rows, k, js, n, device)                                         # (rows, t, 6)
    cols = torch.from_numpy(op.columns.astype("int64")).to(device).T                      # (rows, t)
    if bool((cols >= P).any()):
        raise VerifyError("ligero: non-canonical column value")
    lhs = py_exts((r.unsqueeze(1) * cols.unsqueeze(-1) % P).sum(dim=0) % P)               # (t, 6)
    rhs = py_exts((a_at * cols.unsqueeze(-1) % P).sum(dim=0) % P)
    del a_at
    # q(η_j) = Σ_m qc[m] η_j^m for the t columns as one modular matmul (was a Python Horner loop: 1.6M ext ops)
    m_idx = torch.arange(2 * k - 1, dtype=torch.int64, device=device)
    E = _omega_powers(n, device)[(jt[:, None] * m_idx[None, :]) % n]                       # (t, 2k-1) = η_j^m
    qv_all = py_exts(_mm_mod_rows(E, t_exts(op.qc, device)))                              # (t, 6)
    for oi, j in enumerate(js):
        path = op.paths[oi]
        if len(path) != p.depth:
            raise VerifyError("ligero: opening shape")
        if not merkle_verify(root, j, np.ascontiguousarray(op.columns[oi]).astype("<u4").tobytes(), path):
            raise VerifyError(f"ligero: merkle path {j}")
        if lhs[oi] != w_enc[j]:
            raise VerifyError(f"ligero: proximity test at column {j}")
        if qv_all[oi] != rhs[oi]:
            raise VerifyError(f"ligero: linear test at column {j}")
