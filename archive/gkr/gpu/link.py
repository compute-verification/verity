"""The bit link's prime side in sigma form (PROTOCOL.md §17; the red team's S1, NON_ZK_PROOF only), against the live
Flock session of route (a) (``verity/ml/flock/live``, ``flock-link cell-prove``).

The unit commits the bits of every operand word it consumes (``tools/link_stub.extend_unit``: booleanity and
recomposition per bit).  After the GKR layers:

  root_F                  SHA-256(ROOT_F_TAG || state || ctr) of the transcript: this proof's commitment (the Ligero root
                          and every GKR message), handed to the session's ``Commit`` (``Link.exchange``)
  r_0, r_1 in GF(2^128)^m the session's two link points (the verifier's OS coins, drawn after root_F and root_B) and
  y_0, y_1                the binary side's claims y_k = z^(r_k), both from the session record (``Context``), absorbed as
                          TAG || Σ || SHA-256(root_B) || points || y
  sigma_t, t < 256        prover messages: sigma_t = sum over the linked cells q of bit_t(eq(r, pos(q))) b_q, integers,
                          planes t < 128 of point 0 and t - 128 of point 1; the verifier requires 0 <= sigma_t <= n_cells
                          and sigma_t = bit_t(y_0 || y_1) mod 2
  rho                     one BabyBear^6 challenge; the 256 dense constraints sum_q C(t, q) b_q = sigma_t enter the
                          batched Ligero functional as a[q] += sum_t rho^t bit_t(eq(r, pos(q))), b += sum_t rho^t sigma_t

Both sides lie in [0, p - 1] (n_cells <= p - 1), so each constraint holds over the integers, and the parities give
b^(r_k) = y_k.  GF(2^128) is Flock's, GF(2)[x] / (x^128 + x^7 + x^2 + x + 1), bit t = the coefficient of x^t.  A pair
(point 0, point 1) is one 256-bit value (point 0 in the low half), four int64 limbs on the device, so eq(r, i) is the pair
of both points' eq values and its 256 bits are the 256 planes.

The map pos (Lambda_F) is derived here from the unit circuit and the checker-v2 layout, never read from a statement:
unit u = v * steps + s consumes words k s .. k s + k - 1 of x row v (columns 5 .. 5 + k) and of W column v (the next
k); link column col0 + j * bits + i is bit i of operand column 5 + j, and it goes to
pos = ((port * vus + v) * K + k s + j mod k) * bits + i, port = j // k.  ``derive`` checks the names, the booleanity
and recomposition constraints of every link column, and that pos is a bijection onto [0, 2 vus K bits).  That is
Flock's linked position order (flock-link: bit i of word k of operand row l at (l K + k) 16 + i).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .circuit import Circuit, log2ceil
from .field import DEG, P, Ext, e_add, e_mul, e_scale
from .gkr import Reader, VerifyError
from .transcript import ReplayTranscript, Transcript

TAG = b"verity-gkr/link/sigma-2xgf2_128/v2"
ROOT_F_TAG = b"verity-gkr/link/root-f/v2"
LINK_TXT = "link session-2xgf2_128/v2"
RED = (1 << 7) | (1 << 2) | (1 << 1) | 1
MASK = (1 << 128) - 1
PLANES = 256
BLOCK = 128


def gmul128(a: int, b: int) -> int:
    """Flock's GF(2^128) product (the reference)."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 128:
            a = (a & MASK) ^ RED
    return r


def pair(a: int, b: int) -> int:
    return a | (b << 128)


@dataclass(frozen=True)
class Context:
    """The session's link context, from the Flock verifier's record (``link.txt``)."""
    sigma: bytes
    root_f_sha256: bytes
    root_b_sha256: bytes
    points: tuple[tuple[int, ...], tuple[int, ...]]
    y: tuple[int, int]

    def text(self) -> str:
        f = lambda v: f"{v:032x}"                                   # noqa: E731  the record's {hi:016x}{lo:016x}
        return (f"{LINK_TXT}\nsigma {self.sigma.hex()}\nroot_f_sha256 {self.root_f_sha256.hex()}\n"
                f"root_b_sha256 {self.root_b_sha256.hex()}\n"
                + "".join(f"point {' '.join(f(c) for c in p)}\n" for p in self.points) + f"y {f(self.y[0])} {f(self.y[1])}\n")

    def absorbed(self) -> bytes:
        le = lambda v: v.to_bytes(16, "little")                     # noqa: E731  lo then hi, Flock's coin bytes
        return TAG + self.sigma + self.root_b_sha256 + b"".join(le(c) for p in self.points for c in p) + le(self.y[0]) + le(self.y[1])


def parse_context(text: str) -> Context:
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    if not lines or " ".join(lines[0]) != LINK_TXT:
        raise VerifyError(f"link.txt: not a {LINK_TXT} link")
    kv = {t[0]: t[1:] for t in lines[1:] if t[0] != "point"}
    pts = tuple(tuple(int(c, 16) for c in t[1:]) for t in lines[1:] if t[0] == "point")
    if len(pts) != 2 or len(kv.get("y", ())) != 2:
        raise VerifyError("link.txt: two points and two y values expected")
    return Context(bytes.fromhex(kv["sigma"][0]), bytes.fromhex(kv["root_f_sha256"][0]), bytes.fromhex(kv["root_b_sha256"][0]),
                   pts, (int(kv["y"][0], 16), int(kv["y"][1], 16)))


def root_f(tr: Transcript) -> bytes:
    return hashlib.sha256(ROOT_F_TAG + tr.state + tr.ctr.to_bytes(8, "little")).digest()


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
    """What an instance links: the layout, and either the session's ``context`` (the verifier) or ``exchange``, which
    takes root_F to the live session and returns its ``Context`` (the prover).  The prover's operand words ``x`` rows
    and ``w`` columns ``(vus, K)`` give its own y, which must equal the session's."""
    layout: Layout
    context: Context | None = None
    exchange: object = None
    x: np.ndarray | None = None
    w: np.ndarray | None = None


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
    def eq_level(T, TW, TAB, h, lim, BLOCK: tl.constexpr):
        # T[i + h] = T[i] r_j, T[i] = T[i] (1 + r_j): multiplication by r_j as 64 nibble lookups (TAB: 64 x 16 x 32 B)
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = (i < h) & (i < lim)
        l0 = tl.zeros([BLOCK], dtype=tl.int64)
        l1 = tl.zeros([BLOCK], dtype=tl.int64)
        l2 = tl.zeros([BLOCK], dtype=tl.int64)
        l3 = tl.zeros([BLOCK], dtype=tl.int64)
        for w in tl.static_range(8):
            x = tl.load(TW + i * 8 + w, mask=m, other=0)
            for nb in tl.static_range(8):
                q = TAB + ((w * 8 + nb) * 16 + ((x >> (4 * nb)) & 15).to(tl.int64)) * 4
                l0 ^= tl.load(q, mask=m, other=0)
                l1 ^= tl.load(q + 1, mask=m, other=0)
                l2 ^= tl.load(q + 2, mask=m, other=0)
                l3 ^= tl.load(q + 3, mask=m, other=0)
        mo = m & (i + h < lim)
        o = T + (i + h) * 4
        tl.store(o, l0, mask=mo)
        tl.store(o + 1, l1, mask=mo)
        tl.store(o + 2, l2, mask=mo)
        tl.store(o + 3, l3, mask=mo)
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


def eq_table(r: list[int], device, n: int | None = None) -> torch.Tensor:
    """``(n, 4)`` int64: the pair (eq(r_0, i), eq(r_1, i)) for every i < n <= 2^m (bit j of i against coordinate j);
    ``r[j]`` is the pair of both points' coordinate j."""
    triton, eq_level, _ = _kernels()
    m = len(r)
    sel = ((np.arange(16)[:, None] >> np.arange(4)[None, :]) & 1).astype(bool)
    tabs = np.zeros((m, 64, 16, 4), dtype=np.int64)
    for j in range(m):
        # multiplication by r[j] is GF(2)-linear and block diagonal: the images of the 256 basis vectors
        basis = np.zeros((256, 4), dtype=np.int64)
        for half in range(2):
            v = (r[j] >> (128 * half)) & MASK
            for s in range(128):
                basis[128 * half + s] = _split(v << (128 * half))
                v = ((v << 1) & MASK) ^ (RED if v >> 127 else 0)
        basis = basis.reshape(64, 4, 4)
        for c in range(4):
            tabs[j] ^= np.where(sel[None, :, c, None], basis[:, None, c, :], 0)
    tabs = torch.from_numpy(tabs).to(device)
    n = 1 << m if n is None else n
    assert 0 < n <= 1 << m
    T = torch.zeros((n, 4), dtype=torch.int64, device=device)
    T[0, 0] = 1
    T[0, 2] = 1
    tw = T.view(torch.int32)
    for j in range(m):
        h = 1 << j
        eq_level[(triton.cdiv(min(h, n), 512),)](T, tw, tabs[j], h, n, BLOCK=512)
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
    """The prover's operand bits in position order (x rows, then W columns, words little-endian): its own y."""
    bits = link.layout.bits
    sh = torch.arange(bits, device=device, dtype=torch.int32)
    out = []
    for arr in (link.x, link.w):
        a = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32)).to(device)
        out.append(((a[:, :, None] >> sh) & 1).reshape(-1).to(torch.int32))
    return torch.cat(out)


def parity(s: list[int]) -> int:
    return sum((v & 1) << t for t, v in enumerate(s))


def points(ctx: Context) -> list[int]:
    """The coordinate pairs of the session's two points."""
    return [pair(a, b) for a, b in zip(*ctx.points)]


def _enter(tr: Transcript, ctx: Context, lay: Layout) -> None:
    if len(ctx.points[0]) != lay.m or len(ctx.points[1]) != lay.m:
        raise VerifyError(f"link: the session's points are not two of {lay.m} coordinates")
    tr.absorb(ctx.absorbed())


def _msg(v: int) -> Ext:
    return (v % P,) + (0,) * (DEG - 1)


def _cols(inst) -> torch.Tensor:
    c = inst.segs[0].cols
    return c if c.stride(1) == 1 else c.contiguous()


def prove(inst, tr: Transcript, out: list[Ext], spare: int = 0) -> Claim:
    """``spare``: the challenges that follow rho with no absorb between (``prover.open_coins``), one live round with it."""
    link = inst.link
    lay = link.layout
    dev = inst.device
    ctx = link.exchange(root_f(tr)) if link.exchange is not None else link.context
    link.context = ctx
    _enter(tr, ctx, lay)
    T = eq_table(points(ctx), dev, lay.n_pos)
    sy, sigma = plane_sums(T, lay, z_bits(link, dev), _cols(inst))
    if parity(sy) != pair(*ctx.y):
        raise VerifyError("link: the session's y is not these operands' (the binary side opened other bits)")
    msgs = [_msg(s) for s in sigma]
    out.extend(msgs)
    tr.absorb_exts(msgs)
    tr.reserve(1 + spare)
    return Claim(T, tr.challenge(), sigma)


def verify(inst, tr: Transcript, rd: Reader, spare: int = 0) -> Claim:
    link = inst.link
    lay = link.layout
    dev = inst.device
    ctx = link.context
    if isinstance(tr, ReplayTranscript):
        tr.at_root_f()
    if hashlib.sha256(root_f(tr)).digest() != ctx.root_f_sha256:
        raise VerifyError("link: root_F in the session is not this proof's commitment (Ligero root and GKR messages)")
    _enter(tr, ctx, lay)
    T = eq_table(points(ctx), dev, lay.n_pos)
    y = pair(*ctx.y)
    msgs = rd.take(PLANES)
    sigma = []
    for t, s in enumerate(msgs):
        if any(s[1:]) or not 0 <= s[0] <= lay.n_cells:
            raise VerifyError(f"link: sigma_{t} is not a canonical integer in [0, {lay.n_cells}]")
        if (s[0] & 1) != (y >> t) & 1:
            raise VerifyError(f"link: sigma_{t} parity disagrees with the binary side's y")
        sigma.append(s[0])
    tr.absorb_exts(msgs)
    tr.reserve(1 + spare)
    return Claim(T, tr.challenge(), sigma)


def check_clear(inst, claim: Claim) -> None:
    lay = inst.link.layout
    z = torch.zeros(lay.n_pos, dtype=torch.int32, device=inst.device)
    if plane_sums(claim.T, lay, z, _cols(inst))[1] != claim.sigma:
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
