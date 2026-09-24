"""Candidate A end to end on the device (``backends/gkr/src/main.rs`` ported to torch over BabyBear^6).

Instance = segments (data-parallel circuits over ``units`` rows of committed columns) + optional VU chain
(PROTOCOL.md §12).  Proof = Ligero root, the extension messages, the Ligero opening.  Order of the protocol
(identical to the CPU crate, only the field and the transcript hash-domain differ):

  commit X = [segment columns row-major..., table multiplicities...] (challenge-independent)
  transcript preamble (circuit hashes, unit counts, root, chain public words); table challenges (z, β);
  per segment: the output-layer point ``z0`` and copy point ``γ``
  per table: LogUp fractional-sum GKR                          -> leaf claim
  per segment: GKR layers from the assertions down to the columns  -> two input claims
  ρ_1..ρ_m, ρ_geo: every residual claim as one linear functional ⟨a, X⟩ = b (input claims, LogUp leaves, chain)
  Ligero opening of ⟨a, X⟩

``mode`` ``pcs`` is the proof; ``clear`` is the ARITHMETIC_DIAGNOSTIC verifier reading the columns (no commitment).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import torch

from . import field as _field
from . import gkr, ligero, logup, logup_packed
from .circuit import (
    Circuit,
    Layer,
    Query,
    Table,
    eval_wires,
    log2ceil,
    masked_wires,
    pad_rows,
)
from .field import (
    DEG,
    ONE,
    ZERO,
    Ext,
    Limbs,
    P,
    W,
    e_add,
    e_mul,
    e_scale,
    e_sub,
    eq_table,
    ext_mul,
    ext_scale,
    mm_mod,
    py_ext,
    t_ext,
    t_exts,
)
from .gkr import LayerClaim, Reader, VerifyError
from .logup import LeafClaim
from .transcript import Transcript

DOMAIN = b"verity-gkr/candidate-a/v1/babybear6"


@dataclass
class Segment:
    name: str
    circ: Circuit
    layers: list[Layer]
    cols: torch.Tensor          # (units, ncols) int64 canonical, on the device
    hash: bytes

    @property
    def units(self) -> int:
        return self.cols.shape[0]

    @property
    def ncols(self) -> int:
        return self.circ.ncols


@dataclass
class Link:
    """One linked quantity of the VU chain (``main.rs::Link``, PROTOCOL §12.2 / §14.3).  With ``col_c`` set:
    ``X[(v,0),c] = init``, ``X[(v,i),c] = X[(v,i-1),y]``, ``X_epi[v,epi] = X[(v,steps-1),y]``.  With ``col_c`` None the
    quantity is not an input of the next unit: only the epilogue link and, if ``zero_nonlast``, ``X[(v,i),y] = 0`` for
    every ``i < steps-1`` (v2's saturation flags)."""
    col_c: int | None
    col_y: int
    epi: int
    init: int = 0
    zero_nonlast: bool = False

    def constraints_per_vu(self, steps: int) -> int:
        if self.col_c is not None:
            return steps + 1
        return 1 + (steps - 1 if self.zero_nonlast else 0)


@dataclass
class Chain:
    """The VU chain.  v1: the word chain on limbed words, ``col_c[j]`` ↔ ``col_y[j]`` ↔ ``epi_y32[j]`` per limb ``j``
    (PROTOCOL §12 with the FP32 word as two 16-bit limbs, rule LIMB16); ``links`` is None and the functional is
    built limb-major within a step (``add_chain``, unchanged).  v2: ``links`` is the ``chain.txt`` list (state ×4,
    ``ovf`` ×2 with ``zero_nonlast``, ``sgn`` ×2) and the functional follows ``main.rs::add_chain`` link by link."""
    steps: int
    col_c: list[int]
    col_y: list[int]
    epi_y32: list[int]
    epi_y16: int
    public_y16: list[int]
    links: list[Link] | None = None
    spec_hash: bytes | None = None      # v2: sha256 of chain.txt, absorbed in the preamble (the links are public data)
    # v2 with a `public` line in chain.txt: the epilogue columns bound to the public words, `public_y16` then holds
    # len(epi_pub) words per VU, VU-major (fp4-nvf4: the FP32 word's sign, exponent field and fraction)
    epi_pub: list[int] | None = None

    @property
    def limbs(self) -> int:
        return len(self.col_c)

    @property
    def pub_cols(self) -> list[int]:
        return self.epi_pub if self.epi_pub is not None else [self.epi_y16]

    def constraints_per_vu(self) -> int:
        if self.links is None:
            return (self.steps + 1) * self.limbs + 1
        return len(self.pub_cols) + sum(lk.constraints_per_vu(self.steps) for lk in self.links)


def union_tables(segs: list[Segment]) -> list[Table]:
    """Every table any segment queries: segment 0's tables in their order, then the later segments' new ones.
    (``main.rs`` uses segment 0's list only; the v2 epilogue's ``R16``/``R15`` are not in the unit's list.)"""
    out: list[Table] = []
    names: set[str] = set()
    for s in segs:
        for t in s.circ.tables:
            if t.name not in names:
                names.add(t.name)
                out.append(t)
    return out


@dataclass
class Instance:
    segs: list[Segment]
    chain: Chain | None = None
    _tables: list[Table] | None = None

    @property
    def tables(self) -> list[Table]:
        if self._tables is None:
            self._tables = union_tables(self.segs)
        return self._tables

    @property
    def device(self):
        return self.segs[0].cols.device

    def vus(self) -> int:
        return self.segs[0].units // self.chain.steps if self.chain else 0


@dataclass
class Proof:
    root: bytes
    msgs: list[Ext]
    opening: ligero.Opening | None

    def nbytes(self) -> int:
        return (len(self.root) if any(self.root) else 0) + 24 * len(self.msgs) + (self.opening.nbytes() if self.opening else 0)

    # ``proof.bin`` (README "Proof file"): every count is written so the file is self-delimiting, but a verifier
    # takes every size from the circuit/config and rejects a file whose counts disagree (statement binding).
    # Version 2: SHA-512 Merkle, the u32 after the version is the digest size (64); version 1 was SHA-256 (32).
    MAGIC = b"VGKRBB6\0"
    VERSION = 2

    def to_bytes(self) -> bytes:
        import struct

        if len(self.root) != ligero.DIGEST_BYTES:
            raise ValueError("root is not a SHA-512 digest")
        out = bytearray(self.MAGIC)
        out += struct.pack("<II", self.VERSION, ligero.DIGEST_BYTES)
        out += self.root
        out += struct.pack("<Q", len(self.msgs))
        out += _field.exts_to_bytes(self.msgs)
        op = self.opening
        if op is None:
            out += struct.pack("<Q", 0)
            return bytes(out)
        out += struct.pack("<Q", 1)
        out += struct.pack("<Q", len(op.w))
        out += _field.exts_to_bytes(op.w)
        out += struct.pack("<Q", len(op.qc))
        out += _field.exts_to_bytes(op.qc)
        # a-gpu2: ``columns`` is an ``(t, rows)`` ``<u4`` array and the Merkle paths live in ``paths``
        import numpy as np

        columns = np.asarray(op.columns, dtype="<u4")
        t = columns.shape[0] if columns.ndim == 2 else 0
        rows = columns.shape[1] if columns.ndim == 2 else 0
        depth = len(op.paths[0]) if t else 0
        if len(op.paths) != t:
            raise ValueError("ragged opening")
        out += struct.pack("<QQQ", t, rows, depth)
        for oi in range(t):
            path = op.paths[oi]
            if len(path) != depth:
                raise ValueError("ragged opening")
            out += np.ascontiguousarray(columns[oi]).tobytes()
            out += b"".join(path)
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> Proof:
        import struct

        if data[:8] != cls.MAGIC:
            raise ValueError("not a proof.bin")
        ver, dlen = struct.unpack_from("<II", data, 8)
        if ver != cls.VERSION or dlen != ligero.DIGEST_BYTES:
            raise ValueError(f"proof.bin version {ver} / digest {dlen} B (this verifier takes version {cls.VERSION}, SHA-512)")
        D = ligero.DIGEST_BYTES
        pos = 16
        root = data[pos:pos + D]
        pos += D
        (n,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        msgs = [_field.from_bytes(data[pos + 24 * i:pos + 24 * i + 24]) for i in range(n)]
        pos += 24 * n
        (has,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        if not has:
            if pos != len(data):
                raise ValueError("trailing bytes")
            return cls(root, msgs, None)
        (k,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        w = [_field.from_bytes(data[pos + 24 * i:pos + 24 * i + 24]) for i in range(k)]
        pos += 24 * k
        (nq,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        qc = [_field.from_bytes(data[pos + 24 * i:pos + 24 * i + 24]) for i in range(nq)]
        pos += 24 * nq
        t, rows, depth = struct.unpack_from("<QQQ", data, pos)
        pos += 24
        import numpy as np

        columns = np.zeros((t, rows), dtype="<u4")
        paths = []
        for oi in range(t):
            columns[oi] = np.frombuffer(data, dtype="<u4", count=rows, offset=pos)
            pos += 4 * rows
            paths.append([data[pos + D * i:pos + D * i + D] for i in range(depth)])
            pos += D * depth
        if pos != len(data):
            raise ValueError("trailing bytes")
        return cls(root, msgs, ligero.Opening(w, qc, columns, paths))


@dataclass
class Stats:
    t_mults: float = 0.0            # LogUp multiplicities (before the commitment: they are committed)
    t_commit: float = 0.0           # x assembly + RS encode + Merkle
    t_lookup: float = 0.0           # transcript start + every table's LogUp GKR
    t_witness_wires: float = 0.0
    t_arith: float = 0.0            # the circuit GKR layers (wire evaluation excluded)
    t_open: float = 0.0             # = t_open_acc + t_open_wq + t_open_cols
    t_open_acc: float = 0.0         # the linear functional a (input / lookup / chain claims)
    t_open_wq: float = 0.0          # Ligero w = Σ r_i X_i and q = Σ a_i U_i (the proximity / linear tests)
    t_open_cols: float = 0.0        # the opened columns + their Merkle paths
    proof_elems: int = 0
    slots: int = 0
    depth_circuit: int = 0
    depth_lookup_max: int = 0
    lookup_dims: list = field(default_factory=list)
    committed_elements: int = 0
    ligero_rows: int = 0
    linkage_constraints: int = 0
    peak_mem: int = 0

    def sequential_depth(self, pcs: bool) -> int:
        return 1 + max(self.depth_circuit, self.depth_lookup_max) + (2 if pcs else 0)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def use_packed_logup(dev) -> bool:
    """The packed LogUp kernels (``logup_packed``) run on CUDA unless ``VERITY_GPU_TORCH_ONLY`` is set; the torch
    path (``logup.prove_table``) produces the identical transcript."""
    import os

    return (dev.type == "cuda" and logup_packed.AVAILABLE and not os.environ.get("VERITY_GPU_TORCH_ONLY")
            and not os.environ.get("VERITY_GPU_NO_PACKED"))


# ---------------------------------------------------------------------------------------------------------------------
# Queries / transcript / layout
# ---------------------------------------------------------------------------------------------------------------------

def table_queries(circ: Circuit, name: str) -> list[Query]:
    return [q for q in circ.queries if q.table == name]


def seg_query_values(inst: Instance, name: str) -> list[torch.Tensor]:
    """Per segment the ``(units · q_t, cols)`` matrix of query tuples in leaf order (unit-major)."""
    out = []
    for s in inst.segs:
        qs = table_queries(s.circ, name)
        if not qs:
            out.append(None)
            continue
        if _field._fast_gate_eval is not None and s.cols.is_cuda:
            w = len(qs[0].cols)
            ptr, idx, coef = _query_csr(s.circ, name, qs, s.cols.device)
            vals = _field._fast_gate_eval(s.cols, ptr, idx, coef, len(qs) * w)        # (units, q_t w)
            out.append(vals.reshape(-1, w))
            continue
        vals = torch.stack([logup.query_values(q, s.cols) for q in qs], dim=1)       # (units, q_t, w)
        out.append(vals.reshape(-1, vals.shape[-1]))
    return out


_QUERY_CSR: dict = {}


def _query_csr(circ: Circuit, name: str, qs: list[Query], device):
    """The query columns of one table as ``gate_eval`` linear forms (gate ``q w + j`` = column ``j`` of query ``q``; the
    constant on the one column): CSR ``ptr``, ``idx`` int32, ``coef R mod p`` int64 (cached per circuit, table, device)."""
    key = (circ.hash, name, str(device))
    if key not in _QUERY_CSR:
        if len({len(q.cols) for q in qs}) != 1:
            raise ValueError(f"table {name}: queries of different widths")
        ptr, idx, coef = [0], [], []
        for q in qs:
            for lin in q.cols:
                acc: dict[int, int] = {}
                for i, c in lin.terms:
                    acc[i] = (acc.get(i, 0) + c) % P
                if lin.konst % P:
                    acc[circ.one_col] = (acc.get(circ.one_col, 0) + lin.konst) % P
                for i, c in sorted(acc.items()):
                    if c:
                        idx.append(i)
                        coef.append(c * (1 << 32) % P)
                ptr.append(len(idx))
        _QUERY_CSR[key] = (torch.tensor(ptr, dtype=torch.int32, device=device), torch.tensor(idx, dtype=torch.int32, device=device),
                           torch.tensor(coef, dtype=torch.int64, device=device))
    return _QUERY_CSR[key]


def hypercube_dim(inst: Instance, name: str, table: Table) -> int:
    n_wit = sum(s.units * len(table_queries(s.circ, name)) for s in inst.segs)
    return log2ceil(n_wit + table.nrows)


def start_transcript(inst: Instance, pcs: bool, root: bytes):
    tr = Transcript(DOMAIN)
    for s in inst.segs:
        tr.absorb(s.hash)
        tr.absorb(s.units.to_bytes(8, "little"))
    if pcs:
        tr.absorb(root)
    else:
        h = hashlib.sha256()
        for s in inst.segs:
            h.update(s.cols.to(torch.int32).cpu().numpy().astype("<u4").tobytes())
        tr.absorb(h.digest())
    if inst.chain:
        tr.absorb(inst.chain.steps.to_bytes(8, "little"))
        tr.absorb(b"".join(int(y).to_bytes(4, "little") for y in inst.chain.public_y16))
        if inst.chain.spec_hash is not None:
            tr.absorb(inst.chain.spec_hash)
    chs = []
    for t in inst.tables:
        z = tr.challenge()
        beta = tr.challenge() if t.cols > 1 else ONE
        chs.append((z, beta))
    pts = []
    for s in inst.segs:
        z0 = tr.challenges(s.layers[0].s_out)
        gamma = tr.challenges(log2ceil(s.units))
        pts.append((z0, gamma))
    return tr, chs, pts


def x_layout(inst: Instance):
    seg_offs, l = [], 0
    for s in inst.segs:
        seg_offs.append(l)
        l += s.units * s.ncols
    offs = []
    for t in inst.tables:
        offs.append(l)
        l += t.nrows
    return seg_offs, offs, l


def n_functionals(inst: Instance) -> int:
    return 2 * len(inst.segs) + len(inst.tables) + (1 if inst.chain else 0)


# ---------------------------------------------------------------------------------------------------------------------
# The batched functional ⟨a, X⟩ = b
# ---------------------------------------------------------------------------------------------------------------------

class Acc:
    def __init__(self, l: int, device) -> None:
        self.a = torch.zeros((l, DEG), dtype=torch.int64, device=device)
        self.b: Ext = ZERO
        self.device = device

    def seg_view(self, off: int, units: int, ncols: int) -> torch.Tensor:
        return self.a[off: off + units * ncols].view(units, ncols, DEG)


def add_input_claim(acc: Acc, rho: Ext, seg_off: int, seg: Segment, x_pt: list[Ext], c_pt: list[Ext], value: Ext) -> None:
    dev = acc.device
    ec = eq_table(c_pt, dev)[: seg.units]                            # (units, 6)
    ex = eq_table(x_pt, dev)[: seg.ncols]                            # (ncols, 6)
    e = ext_mul(ec, t_ext(rho, dev))                                  # (units, 6)
    v = acc.seg_view(seg_off, seg.units, seg.ncols)
    if _field._fast_rank1_add is not None and dev.type == "cuda":
        _field._fast_rank1_add(v, e, ex)                              # one read + one write of the segment
    else:
        step = max(1, (1 << 21) // seg.ncols)
        for lo in range(0, seg.units, step):
            hi = min(seg.units, lo + step)
            v[lo:hi] = (v[lo:hi] + ext_mul(e[lo:hi].unsqueeze(1), ex.unsqueeze(0))) % P
    acc.b = e_add(acc.b, e_mul(rho, value))


def add_input_claims(acc: Acc, rhos: tuple[Ext, Ext], seg_off: int, seg: Segment, x_pts: tuple[list[Ext], list[Ext]],
                     c_pt: list[Ext], values: tuple[Ext, Ext]) -> None:
    """Both of a segment's input claims (same copy point ``c_pt``) as one rank-1 pass:
    ``eq(c)·(ρ_l eq(l, x) + ρ_r eq(r, x))``, the same ``a`` as two ``add_input_claim`` calls."""
    dev = acc.device
    if _field._fast_rank1_add is None or dev.type != "cuda":
        for rho, x_pt, value in zip(rhos, x_pts, values):
            add_input_claim(acc, rho, seg_off, seg, x_pt, c_pt, value)
        return
    ec = eq_table(c_pt, dev)[: seg.units]
    ex = [ext_mul(eq_table(x_pt, dev)[: seg.ncols], t_ext(rho, dev)) for rho, x_pt in zip(rhos, x_pts)]
    _field._fast_rank1_add(acc.seg_view(seg_off, seg.units, seg.ncols), ec, (ex[0] + ex[1]) % P)
    for rho, value in zip(rhos, values):
        acc.b = e_add(acc.b, e_mul(rho, value))


def add_lookup_claim(acc: Acc, rho: Ext, inst: Instance, seg_offs: list[int], table_off: int, table: Table, ch, n: int, c: LeafClaim) -> None:
    dev = acc.device
    z, beta = ch
    eqt = eq_table(c.rho, dev, out_dtype=torch.int32)                 # (2^n, 6), written once in int32
    bs = logup.beta_powers(beta, table.cols, dev)                     # (cols, 6)
    bs_py = [py_ext(b) for b in bs]
    lam_t = t_ext(c.lam, dev)
    rho_t = t_ext(rho, dev)
    rl_t = t_ext(e_mul(rho, c.lam), dev)
    konst = torch.zeros((DEG,), dtype=torch.int64, device=dev)
    base = 0
    for si, s in enumerate(inst.segs):
        qs = table_queries(s.circ, table.name)
        q_t = len(qs)
        if q_t == 0:
            continue
        v = acc.seg_view(seg_offs[si], s.units, s.ncols)
        # per query the constant part of  z - Σ_k β^k lin_k  and the per-column coefficients -Σ_k β^k coef_{k,idx}
        csts = []
        terms: dict[tuple[int, int], Ext] = {}                        # (j, idx) -> -Σ_k β^k coef_{k, idx}
        for j, qu in enumerate(qs):
            cst = z
            for kk, lin in enumerate(qu.cols):
                cst = e_sub(cst, e_scale(bs_py[kk], lin.konst))
                for idx, coef in lin.terms:
                    terms[(j, idx)] = e_sub(terms.get((j, idx), ZERO), e_scale(bs_py[kk], coef))
            csts.append(cst)
        cst_t = t_exts(csts, dev)                                      # (q_t, 6)
        e_all = eqt[base: base + s.units * q_t].view(s.units, q_t, DEG)
        if _field._fast_scatter is not None and dev.type == "cuda":
            # konst += Σ_u Σ_j e[u, j] (1 + λ cst_j)   (p = 1 for every query leaf)
            S = e_all.sum(dim=0, dtype=torch.int64) % P                # (q_t, 6)
            K = ext_mul(cst_t, lam_t)
            K[:, 0] = (K[:, 0] + 1) % P
            konst = (konst + ext_mul(S, K).sum(dim=0)) % P
            # v[u, idx] += Σ_j e[u, j] · (ρ λ C[j, idx])   as a CSR scatter over the (query, column) terms
            keys = sorted(terms, key=lambda k: (k[1], k[0]))
            counts = torch.zeros((s.ncols + 1,), dtype=torch.int64)
            for (_j, idx) in keys:
                counts[idx + 1] += 1
            ptr = torch.cumsum(counts, dim=0).to(torch.int32).to(dev)
            tj = torch.tensor([j for j, _ in keys], dtype=torch.int32, device=dev)
            tc = ext_mul(t_exts([terms[k] for k in keys], dev), rl_t)  # (T, 6)
            _field._fast_scatter(e_all, v, ptr, tj, tc)
        else:
            C = torch.zeros((q_t, s.ncols, DEG), dtype=torch.int64)   # C[j, idx] = -Σ_k β^k coef_{k, idx}
            for (j, idx), val in terms.items():
                C[j, idx] = torch.tensor(val, dtype=torch.int64)
            # Cmat[(j, i), (idx, d)]: coefficient of rle[j]_i in coordinate d of rle[j] · C[j, idx]
            Cmat = torch.zeros((q_t, DEG, s.ncols, DEG), dtype=torch.int64)
            for i in range(DEG):
                for jj in range(DEG):
                    d = i + jj
                    if d < DEG:
                        Cmat[:, i, :, d] = (Cmat[:, i, :, d] + C[:, :, jj]) % P
                    else:
                        Cmat[:, i, :, d - DEG] = (Cmat[:, i, :, d - DEG] + W * C[:, :, jj]) % P
            Cmat = Limbs(Cmat.reshape(q_t * DEG, s.ncols * DEG).to(dev))
            step = max(1, (1 << 23) // q_t)
            for lo in range(0, s.units, step):
                hi = min(s.units, lo + step)
                e = e_all[lo:hi].to(torch.int64)
                konst = (konst + e.sum(dim=(0, 1))) % P                # p = 1
                le = ext_mul(e, lam_t)                                 # λ e        (chunk, q_t, 6)
                konst = (konst + ext_mul(le, cst_t.unsqueeze(0)).sum(dim=(0, 1))) % P
                del le
                rle = ext_mul(e, rl_t)                                 # ρ λ e
                del e
                v[lo:hi] = (v[lo:hi] + mm_mod(rle.reshape(hi - lo, q_t * DEG), Cmat).view(hi - lo, s.ncols, DEG)) % P
        base += s.units * q_t
    rows = table.nrows
    e_tab = eqt[base: base + rows]
    acc.a[table_off: table_off + rows] = (acc.a[table_off: table_off + rows] - ext_mul(e_tab, rho_t)) % P
    trows = logup.table_rows(table, dev)
    comb = logup.combine(trows, bs)                                    # (rows, 6)
    zt = t_ext(z, dev)
    konst = (konst + ext_mul(ext_mul(e_tab, lam_t), (zt - comb) % P).sum(dim=0)) % P
    pad = eqt[base + rows:]
    if pad.shape[0]:
        konst = (konst + ext_mul(pad, lam_t).sum(dim=0)) % P
    acc.b = e_add(acc.b, e_mul(rho, e_sub(c.value, py_ext(konst))))


def add_chain(acc: Acc, rho: Ext, geo: Ext, inst: Instance, seg_offs: list[int]) -> int:
    """Chain linkage with geometric coefficients ρ·geo^k in the CPU crate's order: for each VU ``v``: steps
    constraints ``c_{v,i} - y_{v,i-1} = 0``, then ``y32_v - y_{v,last} = 0``, then ``y16_v = public``."""
    from .field import geometric

    ch = inst.chain
    if ch.links is not None:
        return add_chain_links(acc, rho, geo, inst, seg_offs)
    dev = acc.device
    u_seg, e_seg = inst.segs[0], inst.segs[1]
    vus = u_seg.units // ch.steps
    L = ch.limbs
    per_vu = (ch.steps + 1) * L + 1
    pw = geometric(rho, geo, vus * per_vu, dev).view(vus, per_vu, DEG)   # pw[v, k] = ρ geo^(v·per_vu + k)
    vu = acc.seg_view(seg_offs[0], u_seg.units, u_seg.ncols).view(vus, ch.steps, u_seg.ncols, DEG)
    ve = acc.seg_view(seg_offs[1], e_seg.units, e_seg.ncols).view(vus, e_seg.ncols, DEG)
    for j in range(L):
        step_pw = pw[:, j: ch.steps * L: L]                                # (vus, steps, 6): constraint (i, limb j)
        vu[:, :, ch.col_c[j]] = (vu[:, :, ch.col_c[j]] + step_pw) % P
        vu[:, :-1, ch.col_y[j]] = (vu[:, :-1, ch.col_y[j]] - step_pw[:, 1:]) % P
        k = ch.steps * L + j
        ve[:, ch.epi_y32[j]] = (ve[:, ch.epi_y32[j]] + pw[:, k]) % P
        vu[:, -1, ch.col_y[j]] = (vu[:, -1, ch.col_y[j]] - pw[:, k]) % P
    last = pw[:, per_vu - 1]
    ve[:, ch.epi_y16] = (ve[:, ch.epi_y16] + last) % P
    y16 = torch.tensor([int(y) % P for y in ch.public_y16], dtype=torch.int64, device=dev)
    acc.b = e_add(acc.b, py_ext(ext_scale(last, y16).sum(dim=0) % P))
    return vus * per_vu


def add_chain_links(acc: Acc, rho: Ext, geo: Ext, inst: Instance, seg_offs: list[int]) -> int:
    """``main.rs::add_chain`` for a general link list (v2, PROTOCOL §14.3): per VU, link by link, the coefficient
    ``ρ·geo^k`` runs over [steps incoming constraints (or steps-1 ``zero_nonlast`` constraints), the epilogue link],
    then the public word -- a strided read of the linked columns (12 for v2), one O(U) pass per column."""
    from .field import geometric

    ch = inst.chain
    dev = acc.device
    u_seg, e_seg = inst.segs[0], inst.segs[1]
    steps = ch.steps
    vus = u_seg.units // steps
    per_vu = ch.constraints_per_vu()
    pw = geometric(rho, geo, vus * per_vu, dev).view(vus, per_vu, DEG)     # pw[v, k] = ρ geo^(v·per_vu + k)
    vu = acc.seg_view(seg_offs[0], u_seg.units, u_seg.ncols).view(vus, steps, u_seg.ncols, DEG)
    ve = acc.seg_view(seg_offs[1], e_seg.units, e_seg.ncols).view(vus, e_seg.ncols, DEG)
    b_acc = torch.zeros((DEG,), dtype=torch.int64, device=dev)
    o = 0
    for lk in ch.links:
        if lk.col_c is not None:
            step_pw = pw[:, o: o + steps]                                       # (vus, steps, 6): constraint i
            vu[:, :, lk.col_c] = (vu[:, :, lk.col_c] + step_pw) % P
            vu[:, :-1, lk.col_y] = (vu[:, :-1, lk.col_y] - step_pw[:, 1:]) % P
            if lk.init % P:
                b_acc = (b_acc + ext_scale(pw[:, o], torch.full((vus,), lk.init % P, dtype=torch.int64, device=dev)).sum(dim=0)) % P
            o += steps
        elif lk.zero_nonlast:
            vu[:, :-1, lk.col_y] = (vu[:, :-1, lk.col_y] + pw[:, o: o + steps - 1]) % P
            o += steps - 1
        ve[:, lk.epi] = (ve[:, lk.epi] + pw[:, o]) % P
        vu[:, -1, lk.col_y] = (vu[:, -1, lk.col_y] - pw[:, o]) % P
        o += 1
    npub = len(ch.pub_cols)
    ys = torch.tensor([int(y) % P for y in ch.public_y16], dtype=torch.int64, device=dev).view(vus, npub)
    for j, col in enumerate(ch.pub_cols):
        last = pw[:, o + j]
        ve[:, col] = (ve[:, col] + last) % P
        b_acc = (b_acc + ext_scale(last, ys[:, j]).sum(dim=0)) % P
    acc.b = e_add(acc.b, py_ext(b_acc))
    assert o + npub == per_vu
    return vus * per_vu


def check_chain_clear(inst: Instance) -> None:
    ch = inst.chain
    if ch.links is not None:
        return check_chain_links_clear(inst)
    u_seg, e_seg = inst.segs[0], inst.segs[1]
    vus = u_seg.units // ch.steps
    uc = u_seg.cols.view(vus, ch.steps, u_seg.ncols)
    ec = e_seg.cols
    for j in range(ch.limbs):
        if bool((uc[:, 0, ch.col_c[j]] != 0).any()):
            raise VerifyError("chain: a VU's first incoming word is not zero")
        if bool((uc[:, 1:, ch.col_c[j]] != uc[:, :-1, ch.col_y[j]]).any()):
            raise VerifyError("chain: incoming word differs from the previous outgoing word")
        if bool((ec[:, ch.epi_y32[j]] != uc[:, -1, ch.col_y[j]]).any()):
            raise VerifyError("chain: epilogue input differs from the last word")
    y16 = torch.tensor([int(y) % P for y in ch.public_y16], dtype=torch.int64, device=ec.device)
    if bool((ec[:, ch.epi_y16] != y16).any()):
        raise VerifyError("chain: output word differs from the public word")


def check_chain_links_clear(inst: Instance) -> None:
    """``main.rs::check_chain_clear`` for a general link list."""
    ch = inst.chain
    u_seg, e_seg = inst.segs[0], inst.segs[1]
    vus = u_seg.units // ch.steps
    uc = u_seg.cols.view(vus, ch.steps, u_seg.ncols)
    ec = e_seg.cols
    for lk in ch.links:
        if lk.col_c is not None:
            if bool((uc[:, 0, lk.col_c] != lk.init % P).any()):
                raise VerifyError(f"chain: a VU's first incoming value differs from init (column {lk.col_c})")
            if bool((uc[:, 1:, lk.col_c] != uc[:, :-1, lk.col_y]).any()):
                raise VerifyError(f"chain: incoming value differs from the previous outgoing value (column {lk.col_c})")
        elif lk.zero_nonlast and bool((uc[:, :-1, lk.col_y] != 0).any()):
            raise VerifyError(f"chain: column {lk.col_y} must be 0 before the last unit")
        if bool((ec[:, lk.epi] != uc[:, -1, lk.col_y]).any()):
            raise VerifyError(f"chain: epilogue input (column {lk.epi}) differs from the last unit's value")
    ys = torch.tensor([int(y) % P for y in ch.public_y16], dtype=torch.int64, device=ec.device).view(vus, len(ch.pub_cols))
    if bool((ec[:, ch.pub_cols] != ys).any()):
        raise VerifyError("chain: output word differs from the public word")


# ---------------------------------------------------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------------------------------------------------

def prove_segment(seg: Segment, z0: list[Ext], gamma: list[Ext], tr: Transcript, out: list[Ext], st: Stats):
    dev = seg.cols.device
    t1 = time.perf_counter()
    wires = eval_wires(seg.circ, seg.cols)                           # (units, nwires) int32
    _sync(dev)
    st.t_witness_wires += time.perf_counter() - t1
    b = log2ceil(seg.units)
    copies = 1 << b
    depth_of = torch.tensor(seg.circ.depth_of(), dtype=torch.int32, device=dev)
    claim = LayerClaim(eq_table(z0, dev), list(gamma), ZERO)
    nl = len(seg.layers)
    finals = None
    depth = 0
    for i, layer in enumerate(seg.layers):
        last = i == nl - 1
        k_in = seg.circ.max_depth - i
        if last:
            rows = pad_rows(seg.cols, copies, 1 << layer.s_in)
        else:
            rows = pad_rows(masked_wires(wires, depth_of, k_in), copies, 1 << layer.s_in)
        nxt, fin = gkr.prove_layer(layer, rows, b, claim, tr, out, last)
        del rows
        depth += b + 2 * layer.s_in + (0 if last else 1)
        if fin is not None:
            finals = (fin, nxt.copy_point)
        claim = nxt
    st.depth_circuit = max(st.depth_circuit, depth)
    return finals


def prove(inst: Instance, pcs: bool = True, params: ligero.LigeroParams = ligero.DEFAULT) -> tuple[Proof, Stats]:
    dev = inst.device
    st = Stats()
    out: list[Ext] = []
    seg_offs, offs, l = x_layout(inst)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    tm = time.perf_counter()
    committed = None
    mults = []
    for t in inst.tables:
        vals = [v for v in seg_query_values(inst, t.name) if v is not None]
        mults.append(logup.multiplicities(t, logup.table_rows(t, dev), vals))
    _sync(dev)
    st.t_mults = time.perf_counter() - tm
    tc = time.perf_counter()
    if pcs:
        x = torch.empty((l,), dtype=torch.int64, device=dev)
        for s, off in zip(inst.segs, seg_offs):
            x[off: off + s.units * s.ncols] = s.cols.reshape(-1)
        for t, off, m in zip(inst.tables, offs, mults):
            x[off: off + t.nrows] = m % P
        committed = ligero.commit(x, params)
        del x
    root = committed.root if committed else bytes(ligero.DIGEST_BYTES)
    _sync(dev)
    st.t_commit = time.perf_counter() - tc
    t0 = time.perf_counter()
    tr, chs, pts = start_transcript(inst, pcs, root)
    lookup_claims = []
    packed = use_packed_logup(dev)
    graphs = packed and logup_packed.use_graphs(dev)
    for ti, t in enumerate(inst.tables):
        n = hypercube_dim(inst, t.name, t)
        vals = [v for v in seg_query_values(inst, t.name) if v is not None]
        trows = logup.table_rows(t, dev)
        n_wit = sum(v.shape[0] for v in vals)
        if packed and t.cols == 1 and n_wit % 2 == 0 and t.nrows % 2 == 0:
            w = torch.cat([v[:, 0] for v in vals]).to(torch.int32)
            del vals
            tab = trows[:, 0].to(torch.int32)
            # the graphed tree takes the multiplicities from a histogram of the queries: identity tables only
            if graphs and bool(torch.equal(tab, torch.arange(tab.numel(), dtype=torch.int32, device=dev))):
                lc = logup_packed.prove_range_table_graphed(w, tab, chs[ti][0], n, tr, out)
            else:
                lc = logup_packed.prove_range_table(w, tab, mults[ti], chs[ti][0], n, tr, out)
            del w
        else:
            p_leaf, q_leaf = logup.build_leaves(t, trows, vals, mults[ti], chs[ti][0], chs[ti][1], n)
            del vals
            if graphs:
                lc = logup_packed.prove_ext_table_graphed(p_leaf, q_leaf, n, tr, out)
            elif packed:
                lc = logup_packed.prove_ext_table(p_leaf, q_leaf, n, tr, out)
            else:
                lc = logup.prove_table(p_leaf, q_leaf, n, tr, out)
            del p_leaf, q_leaf
        st.lookup_dims.append((t.name, n, sum(len(table_queries(s.circ, t.name)) for s in inst.segs)))
        lookup_claims.append((n, lc))
        st.depth_lookup_max = max(st.depth_lookup_max, 1 + sum(k + 2 for k in range(n)))
    _sync(dev)
    st.t_lookup = time.perf_counter() - t0
    t2 = time.perf_counter()
    finals = [prove_segment(s, pts[si][0], pts[si][1], tr, out, st) for si, s in enumerate(inst.segs)]
    _sync(dev)
    st.t_arith = time.perf_counter() - t2 - st.t_witness_wires
    opening = None
    if pcs:
        t3 = time.perf_counter()
        rhos = tr.challenges(n_functionals(inst))
        geo = tr.challenge() if inst.chain else ZERO
        acc = Acc(l, dev)
        ri = 0
        for si, s in enumerate(inst.segs):
            (l_star, vl, r_star, vr), c_star = finals[si]
            add_input_claims(acc, (rhos[ri], rhos[ri + 1]), seg_offs[si], s, (l_star, r_star), c_star, (vl, vr))
            ri += 2
        for ti, t in enumerate(inst.tables):
            n, lc = lookup_claims[ti]
            add_lookup_claim(acc, rhos[ri], inst, seg_offs, offs[ti], t, chs[ti], n, lc)
            ri += 1
        if inst.chain:
            st.linkage_constraints = add_chain(acc, rhos[ri], geo, inst, seg_offs)
        _sync(dev)
        st.t_open_acc = time.perf_counter() - t3
        timers: dict = {}
        opening = ligero.prove_open(committed, acc.a, tr, timers)
        _sync(dev)
        st.t_open = time.perf_counter() - t3
        st.t_open_wq = timers["wq"]
        st.t_open_cols = st.t_open - st.t_open_acc - st.t_open_wq
        st.ligero_rows = committed.rows
    st.proof_elems = len(out)
    st.slots = tr.slots
    st.committed_elements = l
    if dev.type == "cuda":
        st.peak_mem = torch.cuda.max_memory_allocated(dev)
    return Proof(root, out, opening), st


# ---------------------------------------------------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify(inst: Instance, proof: Proof, pcs: bool = True, params: ligero.LigeroParams = ligero.DEFAULT) -> None:
    """Raises ``VerifyError`` on rejection.  In ``pcs`` mode the segments' columns are never read (only their
    shapes); in ``clear`` mode the verifier evaluates the columns itself."""
    dev = inst.device
    tr, chs, pts = start_transcript(inst, pcs, proof.root)
    rd = Reader(proof.msgs)
    seg_offs, offs, l = x_layout(inst)
    lookup_claims = []
    for ti, t in enumerate(inst.tables):
        n = hypercube_dim(inst, t.name, t)
        lc = logup.verify_table(n, tr, rd)
        if pcs:
            lookup_claims.append((n, lc))
        else:
            vals = [v for v in seg_query_values(inst, t.name) if v is not None]
            trows = logup.table_rows(t, dev)
            m = logup.multiplicities(t, trows, vals)
            p_leaf, q_leaf = logup.build_leaves(t, trows, vals, m, chs[ti][0], chs[ti][1], n)
            e = eq_table(lc.rho, dev)
            pv = ext_scale(e, p_leaf.to(torch.int64)).sum(dim=0) % P
            qv = ext_mul(e, q_leaf.to(torch.int64)).sum(dim=0) % P
            got = e_add(py_ext(pv), e_mul(lc.lam, py_ext(qv)))
            if got != lc.value:
                raise VerifyError(f"lookup {t.name}: leaf claim mismatch")
    finals = []
    for si, s in enumerate(inst.segs):
        b = log2ceil(s.units)
        claim = LayerClaim(eq_table(pts[si][0], dev), list(pts[si][1]), ZERO)
        nl = len(s.layers)
        s_c = s.layers[nl - 1].s_in
        for i, layer in enumerate(s.layers):
            last = i == nl - 1
            nxt, fin = gkr.verify_layer(layer, b, claim, tr, rd, last, dev)
            if fin is not None:
                l_star, vl, r_star, vr = fin
                c_star = nxt.copy_point
                if pcs:
                    finals.append((fin, c_star))
                else:
                    if gkr.eval_columns(s.cols, s_c, l_star, c_star) != vl:
                        raise VerifyError(f"{s.name}: input layer: left evaluation mismatch")
                    if gkr.eval_columns(s.cols, s_c, r_star, c_star) != vr:
                        raise VerifyError(f"{s.name}: input layer: right evaluation mismatch")
            claim = nxt
    if not pcs:
        if inst.chain:
            check_chain_clear(inst)
    else:
        rhos = tr.challenges(n_functionals(inst))
        geo = tr.challenge() if inst.chain else ZERO
        acc = Acc(l, dev)
        ri = 0
        for si, s in enumerate(inst.segs):
            (l_star, vl, r_star, vr), c_star = finals[si]
            add_input_claim(acc, rhos[ri], seg_offs[si], s, l_star, c_star, vl)
            add_input_claim(acc, rhos[ri + 1], seg_offs[si], s, r_star, c_star, vr)
            ri += 2
        for ti, t in enumerate(inst.tables):
            n, lc = lookup_claims[ti]
            add_lookup_claim(acc, rhos[ri], inst, seg_offs, offs[ti], t, chs[ti], n, lc)
            ri += 1
        if inst.chain:
            add_chain(acc, rhos[ri], geo, inst, seg_offs)
        if proof.opening is None:
            raise VerifyError("missing opening")
        ligero.verify_open(proof.root, params, l, acc.a, acc.b, proof.opening, tr, dev)
    if rd.pos != len(proof.msgs):
        raise VerifyError("trailing proof data")
