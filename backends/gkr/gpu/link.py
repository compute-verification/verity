"""The bit link's prime side in sigma form (PROTOCOL.md §17; the red team's S1, NON_ZK_PROOF only).

The unit commits the bits of every operand word it consumes (``tools/link_stub.extend_unit``: booleanity and
recomposition per bit).  After the GKR layers, with the Ligero root and the binary proof's root ``root_b`` absorbed:

  r in GF(2^256)^m        one challenge point (m = log2ceil of the linked positions), a coin slot of its own
  y = z^(r)               the binary side's claim about the hashed message bits z (here a STAND-IN: the verifier's own
                          copy of the operand words, until the binary proof is wired)
  sigma_t, t < 256        prover messages: sigma_t = sum over the linked cells q of bit_t(eq(r, pos(q))) b_q, integers;
                          the verifier requires 0 <= sigma_t <= n_cells and sigma_t = bit_t(y) mod 2
  rho                     one BabyBear^6 challenge; the 256 dense constraints sum_q C(t, q) b_q = sigma_t enter the
                          batched Ligero functional as a[q] += sum_t rho^t bit_t(eq(r, pos(q))), b += sum_t rho^t sigma_t

Both sides lie in [0, p - 1] (n_cells <= p - 1), so each constraint holds over the integers, and the parities give
b^(r) = y.  GF(2^256) is GF(2)[x] / (x^256 + x^10 + x^5 + x^2 + 1), polynomial basis, bit t = the coefficient of x^t,
elements little-endian in 32 bytes.

The map pos (Lambda_F) is derived here from the unit circuit and the checker-v2 layout, never read from a statement:
unit u = v * steps + s consumes words k s .. k s + k - 1 of x row v (columns 5 .. 5 + k) and of W column v (the next
k); link column col0 + j * bits + i is bit i of operand column 5 + j, and it goes to
pos = ((port * vus + v) * K + k s + j mod k) * bits + i, port = j // k.  ``derive`` checks the names, the booleanity
and recomposition constraints of every link column, and that pos is a bijection onto [0, 2 vus K bits).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .circuit import Circuit, log2ceil
from .field import DEG, P, Ext, e_add, e_mul, e_scale
from .gkr import Reader, VerifyError
from .transcript import Transcript

TAG = b"verity-gkr/link/sigma-gf2_256/v1"
RED = (1 << 10) | (1 << 5) | (1 << 2) | 1
MASK = (1 << 256) - 1
PLANES = 256
BLOCK = 128


def gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 256:
            a = (a & MASK) ^ RED
    return r


@dataclass(frozen=True)
class Layout:
    vus: int
    steps: int
    K: int
    k: int
    bits: int
    col0: int
    ncols: int

    @property
    def per_unit(self) -> int:
        return 2 * self.k * self.bits

    @property
    def n_cells(self) -> int:
        return self.vus * self.steps * self.per_unit

    @property
    def n_pos(self) -> int:
        return 2 * self.vus * self.K * self.bits

    @property
    def m(self) -> int:
        return log2ceil(self.n_pos)

    def pos(self, q: np.ndarray) -> np.ndarray:
        u, jb = np.divmod(q, self.per_unit)
        v, s = np.divmod(u, self.steps)
        j, i = np.divmod(jb, self.bits)
        port, jj = np.divmod(j, self.k)
        return ((port * self.vus + v) * self.K + self.k * s + jj) * self.bits + i


def derive(circ: Circuit, vus: int, steps: int, K: int, bits: int) -> Layout:
    """Lambda_F of a checker-v2 unit circuit extended by ``extend_unit``; raises ``VerifyError`` on any mismatch."""
    names = circ.col_names
    if "y.s" not in names:
        raise VerifyError("link: not a checker-v2 unit (no y.s column)")
    k = (names.index("y.s") - 5) // 2
    if k <= 0 or steps * k != K or bits not in (8, 16):
        raise VerifyError(f"link: layout k = {k}, steps = {steps}, K = {K}, bits = {bits}")
    nb = 2 * k * bits
    col0 = circ.ncols - nb
    want = [f"link.b{5 + j}.{i}" for j in range(2 * k) for i in range(bits)]
    if names[col0:] != want or any(n.startswith("link.") for n in names[:col0]):
        raise VerifyError("link: the link columns are not the last 2 k bits columns, in operand-column order")
    colwire = {w[1]: wi for wi, w in enumerate(circ.wires) if w[0] == "in"}
    sq: dict[int, list[int]] = {}
    for wi, w in enumerate(circ.wires):
        if w[0] == "prod" and len(w[2].terms) == 1 and tuple(w[2].terms[0]) == (w[2].terms[0][0], 1) and w[2].konst == 0:
            x = w[2].terms[0][0]
            if [tuple(t) for t in w[3].terms] == [(x, 1)] and w[3].konst == P - 1:
                sq.setdefault(x, []).append(wi)
    zero = {(tuple(sorted(tuple(t) for t in l.terms)), l.konst) for _n, l in circ.asserts}
    for j in range(2 * k):
        rec = [(colwire[5 + j], 1)]
        for i in range(bits):
            bw = colwire[col0 + j * bits + i]
            if not any((((pw, 1),), 0) in zero for pw in sq.get(bw, [])):
                raise VerifyError(f"link: no booleanity constraint on link.b{5 + j}.{i}")
            rec.append((bw, (P - (1 << i)) % P))
        if (tuple(sorted(rec)), 0) not in zero:
            raise VerifyError(f"link: no recomposition constraint for operand column {5 + j}")
    lay = Layout(vus, steps, K, k, bits, col0, circ.ncols)
    if lay.n_cells != lay.n_pos or lay.n_cells > P - 1:
        raise VerifyError(f"link: {lay.n_cells} cells for {lay.n_pos} positions (p - 1 = {P - 1})")
    seen = np.zeros(lay.n_pos, dtype=bool)
    step = 1 << 24
    for lo in range(0, lay.n_cells, step):
        ps = lay.pos(np.arange(lo, min(lay.n_cells, lo + step), dtype=np.int64))
        if ps.min() < 0 or ps.max() >= lay.n_pos:
            raise VerifyError("link: Lambda_F leaves the linked positions")
        seen[ps] = True
    # n_cells == n_pos cells landing on n_cells distinct positions: injective and onto
    if int(np.count_nonzero(seen)) != lay.n_cells:
        raise VerifyError("link: Lambda_F is not injective")
    return lay


@dataclass
class Link:
    """What an instance links: the layout, the binary proof's root, and (STAND-IN for the binary proof) the operand
    words z is made of, ``x`` rows and ``w`` columns ``(vus, K)``."""
    layout: Layout
    root_b: bytes
    x: np.ndarray
    w: np.ndarray


@dataclass
class Claim:
    T: torch.Tensor
    rho: Ext
    sigma: list[int]


# ---- device kernels ---------------------------------------------------------------------------------------------

_K = None
triton = tl = None


def _kernels():
    # the kernels resolve `tl` through the module globals
    global _K, triton, tl
    if _K is not None:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def eq_level(T, TB, TAB, h, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = i < h
        l0 = tl.zeros([BLOCK], dtype=tl.int64)
        l1 = tl.zeros([BLOCK], dtype=tl.int64)
        l2 = tl.zeros([BLOCK], dtype=tl.int64)
        l3 = tl.zeros([BLOCK], dtype=tl.int64)
        for kk in tl.static_range(32):
            b = tl.load(TB + i * 32 + kk, mask=m, other=0).to(tl.int64)
            q = TAB + (kk * 256 + b) * 4
            l0 ^= tl.load(q, mask=m, other=0)
            l1 ^= tl.load(q + 1, mask=m, other=0)
            l2 ^= tl.load(q + 2, mask=m, other=0)
            l3 ^= tl.load(q + 3, mask=m, other=0)
        o = T + (i + h) * 4
        tl.store(o, l0, mask=m)
        tl.store(o + 1, l1, mask=m)
        tl.store(o + 2, l2, mask=m)
        tl.store(o + 3, l3, mask=m)
        o = T + i * 4
        tl.store(o, tl.load(o, mask=m, other=0) ^ l0, mask=m)
        tl.store(o + 1, tl.load(o + 1, mask=m, other=0) ^ l1, mask=m)
        tl.store(o + 2, tl.load(o + 2, mask=m, other=0) ^ l2, mask=m)
        tl.store(o + 3, tl.load(o + 3, mask=m, other=0) ^ l3, mask=m)

    @triton.jit
    def cells(TW, Z, B, SP, A, R8, n, rs, per_unit, steps, bits, k, vus, K, col0, ncols, off, p,
              MODE: tl.constexpr, HAS_B: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        x = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = x < n
        if MODE == 0:
            # x is a position; the committed bit it links is cell (u, jb) = pos^-1(x), read from the unit matrix
            pos = x
            i = x % bits
            wd = x // bits
            word = wd % K
            vv = wd // K
            v = vv % vus
            port = vv // vus
            u = v * steps + word // k
            jb = (port * k + word % k) * bits + i
        else:
            # x is a cell
            u = x // per_unit
            jb = x % per_unit
            j = jb // bits
            pos = ((j // k * vus + u // steps) * K + k * (u % steps) + j % k) * bits + jb % bits
        t = tl.arange(0, 256)
        words = tl.load(TW + pos[:, None] * 8 + tl.arange(0, 8)[None, :], mask=m[:, None], other=0)
        e = tl.reshape((words[:, :, None] >> tl.arange(0, 32)[None, None, :]) & 1, (BLOCK, 256)).to(tl.int8)
        r16 = tl.arange(0, 16)
        if MODE == 0:
            # rows 0 / 1 of [z; b; 0] @ e: the block's plane sums of z and of the committed bits, on the tensor cores
            # (bits are 0 / 1)
            zq = tl.load(Z + x, mask=m, other=0).to(tl.int8)
            zero = tl.zeros([16, BLOCK], dtype=tl.int8)
            if HAS_B:
                bq = tl.load(B + u * rs + col0 + jb, mask=m, other=0).to(tl.int8)
                bm = tl.where(r16[:, None] == 0, zq[None, :], tl.where(r16[:, None] == 1, bq[None, :], zero))
            else:
                bm = tl.where(r16[:, None] == 0, zq[None, :], zero)
            s2 = tl.dot(bm, e, out_dtype=tl.int32)
            tl.store(SP + pid * 512 + t, tl.sum(tl.where(r16[:, None] == 0, s2, 0), 0))
            tl.store(SP + pid * 512 + 256 + t, tl.sum(tl.where(r16[:, None] == 1, s2, 0), 0))
        else:
            # coordinate d of sum_t rho^t e_t from the 7-bit limbs of rho^t: one [256, 16] int8 matrix per limb
            cd = tl.zeros([BLOCK, 16], dtype=tl.int64)
            for qq in tl.static_range(5):
                r8 = tl.load(R8 + qq * 4096 + t[:, None] * 16 + r16[None, :])
                cd += tl.dot(e, r8, out_dtype=tl.int32).to(tl.int64) << (7 * qq)
            ptr = A + (off + u * ncols + col0 + jb)[:, None] * 6 + r16[None, :]
            mk = m[:, None] & (r16[None, :] < 6)
            old = tl.load(ptr, mask=mk, other=0).to(tl.int64)
            tl.store(ptr, ((old + cd) % p).to(A.dtype.element_ty), mask=mk)

    _K = (triton, eq_level, cells)
    return _K


def _split(v: int) -> list[int]:
    return [(x - (1 << 64)) if x >> 63 else x for x in ((v >> (64 * i)) & ((1 << 64) - 1) for i in range(4))]


def eq_table(r: list[int], device) -> torch.Tensor:
    """``(2^m, 4)`` int64: eq(r, i) for every i < 2^m (bit j of i against r_j)."""
    triton, eq_level, _ = _kernels()
    m = len(r)
    sel = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(bool)
    tabs = np.zeros((m, 32, 256, 4), dtype=np.int64)
    for j in range(m):
        basis, v = np.zeros((256, 4), dtype=np.int64), r[j]
        for s in range(256):
            basis[s] = _split(v)
            v = ((v << 1) & MASK) ^ (RED if v >> 255 else 0)
        basis = basis.reshape(32, 8, 4)
        for q in range(8):
            tabs[j] ^= np.where(sel[None, :, q, None], basis[:, None, q, :], 0)
    tabs = torch.from_numpy(tabs).to(device)
    T = torch.zeros((1 << m, 4), dtype=torch.int64, device=device)
    T[0, 0] = 1
    tb = T.view(torch.uint8).view(1 << m, 32)
    for j in range(m):
        h = 1 << j
        eq_level[(triton.cdiv(h, 1024),)](T, tb, tabs[j], h, BLOCK=1024)
    return T


def eq_value(T: torch.Tensor, i: int) -> int:
    return sum((int(x) & ((1 << 64) - 1)) << (64 * j) for j, x in enumerate(T[i].tolist()))


def plane_sums(T: torch.Tensor, lay: Layout, z: torch.Tensor, cols: torch.Tensor | None) -> tuple[list[int], list[int] | None]:
    """``(sum_p bit_t(eq(p)) z_p, sum_q bit_t(eq(pos(q))) b_q)`` for t < 256 in one pass over the positions: ``z`` in
    position order, the committed bits ``b`` read from the unit matrix ``cols`` (link columns ``col0 ..``) at pos^-1."""
    triton, _, cells = _kernels()
    n = lay.n_pos
    nb = triton.cdiv(n, BLOCK)
    sp = torch.empty((nb, 2, PLANES), dtype=torch.int32, device=z.device)
    tw = T.view(torch.int32).reshape(-1)
    if cols is not None:
        assert cols.stride(1) == 1 and cols.shape[1] == lay.ncols
    cb = z if cols is None else cols
    cells[(nb,)](tw, z, cb, sp, sp, sp, n, cb.stride(0) if cols is not None else 0, lay.per_unit, lay.steps, lay.bits, lay.k,
                 lay.vus, lay.K, lay.col0, lay.ncols, 0, P, MODE=0, HAS_B=cols is not None, BLOCK=BLOCK)
    s = sp.sum(0, dtype=torch.int64).tolist()
    return [int(v) for v in s[0]], ([int(v) for v in s[1]] if cols is not None else None)


def z_bits(link: Link, device) -> torch.Tensor:
    """STAND-IN for the binary proof: the hashed message bits in position order (x rows, then W columns, words
    little-endian)."""
    bits = link.layout.bits
    sh = torch.arange(bits, device=device, dtype=torch.int32)
    out = []
    for arr in (link.x, link.w):
        a = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32)).to(device)
        out.append(((a[:, :, None] >> sh) & 1).reshape(-1).to(torch.int32))
    return torch.cat(out)


def parity(s: list[int]) -> int:
    return sum((v & 1) << t for t, v in enumerate(s))


def points(tr: Transcript, link: Link) -> list[int]:
    tr.absorb(TAG + link.root_b)
    return [int.from_bytes(tr.challenge_bytes(), "little") for _ in range(link.layout.m)]


def _msg(v: int) -> Ext:
    return (v % P,) + (0,) * (DEG - 1)


def _cols(inst) -> torch.Tensor:
    c = inst.segs[0].cols
    return c if c.stride(1) == 1 else c.contiguous()


def prove(inst, tr: Transcript, out: list[Ext]) -> Claim:
    link = inst.link
    lay = link.layout
    dev = inst.device
    T = eq_table(points(tr, link), dev)
    sy, sigma = plane_sums(T, lay, z_bits(link, dev), _cols(inst))
    tr.absorb(b"link/y" + parity(sy).to_bytes(32, "little"))
    msgs = [_msg(s) for s in sigma]
    out.extend(msgs)
    tr.absorb_exts(msgs)
    return Claim(T, tr.challenge(), sigma)


def verify(inst, tr: Transcript, rd: Reader) -> Claim:
    link = inst.link
    lay = link.layout
    dev = inst.device
    T = eq_table(points(tr, link), dev)
    y = parity(plane_sums(T, lay, z_bits(link, dev), None)[0])
    tr.absorb(b"link/y" + y.to_bytes(32, "little"))
    msgs = rd.take(PLANES)
    sigma = []
    for t, s in enumerate(msgs):
        if any(s[1:]) or not 0 <= s[0] <= lay.n_cells:
            raise VerifyError(f"link: sigma_{t} is not a canonical integer in [0, {lay.n_cells}]")
        if (s[0] & 1) != (y >> t) & 1:
            raise VerifyError(f"link: sigma_{t} parity disagrees with the binary side's y")
        sigma.append(s[0])
    tr.absorb_exts(msgs)
    return Claim(T, tr.challenge(), sigma)


def check_clear(inst, claim: Claim) -> None:
    lay = inst.link.layout
    if plane_sums(claim.T, lay, z_bits(inst.link, inst.device), _cols(inst))[1] != claim.sigma:
        raise VerifyError("link: the committed bits' plane sums differ from sigma")


def add(acc, claim: Claim, lay: Layout, seg_off: int) -> None:
    """a[q] += sum_t rho^t bit_t(eq(pos(q))) over the link cells of segment 0; b += sum_t rho^t sigma_t."""
    triton, _, cells = _kernels()
    dev = acc.a.device
    pw, g = [], (1,) + (0,) * (DEG - 1)
    for _ in range(PLANES):
        pw.append(g)
        g = e_mul(g, claim.rho)
    rh = torch.tensor(pw, dtype=torch.int64)
    r8 = torch.zeros((5, PLANES, 16), dtype=torch.int64)
    for q in range(5):
        r8[q, :, :DEG] = (rh >> (7 * q)) & 127
    r8 = r8.to(torch.int8).to(dev)
    tw = claim.T.view(torch.int32).reshape(-1)
    n = lay.n_cells
    cells[(triton.cdiv(n, BLOCK),)](tw, r8, r8, r8, acc.a, r8, n, 0, lay.per_unit, lay.steps, lay.bits, lay.k, lay.vus, lay.K, lay.col0,
                                    lay.ncols, seg_off, P, MODE=1, HAS_B=False, BLOCK=BLOCK)
    for t in range(PLANES):
        acc.b = e_add(acc.b, e_scale(pw[t], claim.sigma[t] % P))


def standin_root_b(commitment_text: bytes) -> bytes:
    """The STAND-IN binary root until the binary proof is wired: SHA-256 of a tag and the statement's commitment."""
    return hashlib.sha256(TAG + b"/root-b-standin/" + commitment_text).digest()
