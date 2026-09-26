"""PR #83's circuit statement (PROTOCOL.md §16.1, §16.2, §16.5), split into the public frame B and the private circuit C.

The prototype's split (Project store `docs/circuit-privacy.md` I.9, staged):
- **C, private**: the unit slot type U (its row count and every entry of A and B outside its input and constant rows), the
  units' leaf maps (`leaves_in`, `leaves_out`) and the padding instances' outputs (U on zero rows).
- **B, public**: the block's shape and schedule, the protocol slots (the keyed-BLAKE3 row compressions, the mask), their
  glue, the unit grid (count and slot size), U's port positions, the regions' layout and the data trees' shape.

Everything here is native. `Statement.full_fold` is the Lean verifier's fold (`Flock.Stmt.fold`), kept to test that the
public fold plus V[B]'s private terms add up to it.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from . import gf

WORD = 128
PT_LOCAL = 24
COMP_LOG = 14
Z_CONST = 11706
CV_BASE, OUT_LO, M_BASE, T_LO = 0, 256, 512, 1024
CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16


# ---------------------------------------------------------------------------------------------------------------------
# sparse matrices
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Sparse:
    """Rows of column lists, stored CSR; `fold_t(e)` is Xᵀe over `width` columns."""
    indptr: np.ndarray
    indices: np.ndarray
    width: int

    @classmethod
    def of_rows(cls, rows: list[list[int]], width: int) -> "Sparse":
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(r) for r in rows])
        idx = np.fromiter((c for r in rows for c in r), dtype=np.int64, count=int(indptr[-1]))
        return cls(indptr, idx, width)

    @property
    def nrows(self) -> int:
        return len(self.indptr) - 1

    @cached_property
    def _csc(self):
        rows = np.repeat(np.arange(self.nrows, dtype=np.int64), np.diff(self.indptr))
        order = np.argsort(self.indices, kind="stable")
        cols = self.indices[order]
        starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]]) if len(cols) else np.zeros(0, np.int64)
        return rows[order], cols[starts], starts

    def fold_t(self, e: np.ndarray) -> np.ndarray:
        out = np.zeros((self.width, 2), dtype=np.uint64)
        rows, ucols, starts = self._csc
        if len(rows):
            v = e[np.minimum(rows, len(e) - 1)] * (rows < len(e))[:, None].astype(np.uint64)
            out[ucols] = np.bitwise_xor.reduceat(v, starts, axis=0)
        return out


def load_comp(path: str | Path) -> tuple[dict, Sparse, Sparse]:
    """`flock-r1cs-rows/v1`: a JSON header line, then A's and B's rows as (le32 count, le32 columns...)."""
    path = Path(path)
    cache = path.with_suffix(".npz")
    b = path.read_bytes() if not cache.exists() else None
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        head = json.loads(bytes(z["head"]).decode())
        return head, Sparse(z["ap"], z["ai"], head["cols"]), Sparse(z["bp"], z["bi"], head["cols"])
    nl = b.index(b"\n")
    head = json.loads(b[:nl])
    words = np.frombuffer(b[nl + 1:nl + 1 + ((len(b) - nl - 1) // 4) * 4], dtype="<u4")
    pos = 0
    mats = []
    for _ in range(2):
        n = head["rows"]
        indptr = np.zeros(n + 1, dtype=np.int64)
        starts = np.zeros(n, dtype=np.int64)
        for i in range(n):
            k = int(words[pos])
            starts[i] = pos + 1
            indptr[i + 1] = indptr[i] + k
            pos += 1 + k
        idx = np.concatenate([words[s:s + (indptr[i + 1] - indptr[i])] for i, s in enumerate(starts)]).astype(np.int64)
        mats.append(Sparse(indptr, idx, head["cols"]))
    np.savez(cache, head=np.frombuffer(json.dumps(head).encode(), dtype=np.uint8), ap=mats[0].indptr, ai=mats[0].indices,
             bp=mats[1].indptr, bi=mats[1].indices)
    return head, mats[0], mats[1]


# ---------------------------------------------------------------------------------------------------------------------
# the circuit file
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Group:
    col: int
    words: int
    bits: list[int]


@dataclass
class Net:
    name: str
    useful: int
    const_pos: int
    in_groups: list[Group]
    out_groups: list[Group]
    a: list[list[int]]
    b: list[list[int]]

    @property
    def in_words(self) -> int:
        g = self.in_groups[-1]
        return g.col + g.words

    @property
    def unit_log(self) -> int:
        return max(7, (self.useful - 1).bit_length() if self.useful > 1 else 0)


def parse_net(lines: list[str]) -> Net:
    h = lines[0].split()
    if h[0] != "flock-ir-unit/v2":
        raise ValueError("not a flock-ir-unit/v2 expanded circuit")
    n = lambda i: int(h[i])
    useful, const_pos = n(2), n(3)
    at = 4

    def groups():
        nonlocal at
        cnt = n(at)
        at += 1
        out = []
        for _ in range(cnt):
            col, words, np_ = n(at), n(at + 1), n(at + 2)
            out.append(Group(col, words, [n(at + 3 + i) for i in range(np_)]))
            at += 3 + np_
        return out

    ig, og = groups(), groups()
    a, b = [], []
    for i in range(useful):
        v = list(map(int, lines[1 + i].split()))
        na = v[0]
        a.append(v[1:1 + na])
        b.append(v[2 + na:2 + na + v[1 + na]])
    return Net(h[1], useful, const_pos, ig, og, a, b)


@dataclass
class Range:
    kind: str          # comp | unit | mask
    slot_log: int
    pos0: int
    count: int
    role: str | None   # runs | parents | None
    port: int | None
    slots: list | None

    def start(self) -> int:
        return self.pos0 << self.slot_log

    def slot(self, s: int) -> int:
        return (self.pos0 + s) << self.slot_log


@dataclass
class Port:
    name: str
    words: int
    chunks: int
    nb: int


@dataclass
class Circuit:
    text: str
    meta: dict
    nets: dict[str, Net]
    ranges: list[Range]
    ports: list[Port]

    @property
    def unit(self) -> Net:
        return self.nets["unit"]

    @property
    def unit_range(self) -> Range:
        return next(r for r in self.ranges if r.kind == "unit")


def parse_circuit(text: str) -> Circuit:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    meta = json.loads(lines[1][5:])
    for k in ("tail", "cut", "cut_words", "public_ports", "public", "native", "outputs_by_tail"):
        if k in meta:
            raise ValueError(f"META: {k}: the verifier evaluates nothing")
    if meta.get("lookups") or meta.get("wires"):
        raise ValueError("this prototype's bounds admit no lookup slots and no wires")
    nets, at = {}, 2
    while at < len(lines):
        h = lines[at].split()
        cnt = int(h[2])
        nets[h[1]] = parse_net(lines[at + 1:at + 1 + cnt])
        at += 1 + cnt
    ranges = [Range(r["type"], r["slot_log"], r["pos0"], r["count"], r.get("role"), r.get("port"),
                    [tuple(x) if x is not None else None for x in r["slots"]] if "slots" in r else None)
              for r in meta["ranges"]]
    ports = [Port(*p) for p in meta["ports"]]
    return Circuit(text, meta, nets, ranges, ports)


@dataclass
class Public:
    header: dict
    n: int
    digests: list[list[bytes]]    # [instance][port]
    outputs: list[list[int]]      # [instance][output word]


def load_public(path: str | Path, circ: Circuit) -> Public:
    b = Path(path).read_bytes()
    nl = b.index(b"\n")
    header = json.loads(b[:nl])
    n = header["instances"]
    np_ = len(circ.ports)
    nout = sum(w for _, w in circ.meta["out_ports"])
    body = b[nl + 1:]
    if header["rows"] or len(body) != 32 * n * np_ + 2 * n * nout:
        raise ValueError("public file: not the verifier's (rows present, or wrong length)")
    dig = [[body[32 * (i * np_ + p):32 * (i * np_ + p) + 32] for p in range(np_)] for i in range(n)]
    o = body[32 * n * np_:]
    outs = [[struct.unpack_from("<H", o, 2 * (i * nout + w))[0] for w in range(nout)] for i in range(n)]
    return Public(header, n, dig, outs)


# ---------------------------------------------------------------------------------------------------------------------
# the private circuit C
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class PrivateCircuit:
    """C: what the prototype hides. Entries are (row, col) of U's A and B outside the implied rows (the identity input
    rows of the leaf bits and the constant row, which B fixes)."""
    useful: int
    a: list[tuple[int, int]]
    b: list[tuple[int, int]]
    leaves_in: list[list[int]]
    leaves_out: list[list[int]]
    dummy_outputs: list[int]

    @property
    def const_pos(self) -> int:
        return self.useful - 1


@dataclass(frozen=True)
class Bounds:
    """B: the public bounds and frame. Two circuits with equal Bounds give V[B]s with equal shapes."""
    k_log: int
    nbl: int
    n: int
    g: int
    upv: int
    unit_log: int
    unit_pos0: int
    leaf_rows: int           # U's leaf-bit input rows are 0 .. leaf_rows - 1
    in_rows: int             # rows below this are U's input rows (identity for leaf bits, else empty)
    out_col: int             # U's returned-output word column
    leaves_per_unit_in: int
    leaves_per_unit_out: int
    n_in_words: int
    n_out_words: int
    qa_max: int
    qb_max: int
    hash_len: int = 32

    @property
    def m(self) -> int:
        return self.k_log + self.nbl

    @property
    def m_pts(self) -> int:
        return PT_LOCAL + self.nbl

    @property
    def units(self) -> int:
        return self.g * self.upv


def private_of(circ: Circuit) -> PrivateCircuit:
    u = circ.unit
    ig = u.in_groups[0]
    leaf_rows = sum(ig.bits)
    in_rows = u.in_words * WORD
    a, b = [], []
    for r in range(u.useful):
        implied = (r < leaf_rows) or r == u.const_pos
        for mat, rows in ((a, u.a), (b, u.b)):
            if implied:
                if rows[r] != [r]:
                    raise ValueError(f"unit row {r} is an implied row but not the identity")
            elif r < in_rows:
                if rows[r]:
                    raise ValueError(f"unit row {r}: an input row past the leaf bits that is not empty")
            else:
                mat.extend((r, c) for c in rows[r])
    return PrivateCircuit(u.useful, a, b, [list(x) for x in circ.meta["leaves_in"]], [list(x) for x in circ.meta["leaves_out"]],
                          list(circ.meta["dummy"]["outputs"]))


# ---------------------------------------------------------------------------------------------------------------------
# the statement
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Region:
    name: str
    kind: str
    free: list[int]
    mask: int
    fixed: int
    port: int | None = None
    range_index: int | None = None
    const_bytes: bytes | None = None


def _bits(lo: int, n: int) -> list[int]:
    return list(range(lo, lo + n))


def _log2(x: int) -> int:
    return x.bit_length() - 1


def blake3_compress(cv, m, ctr, bl, flags):
    IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
    P = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
    M32 = 0xFFFFFFFF
    rot = lambda x, n: ((x >> n) | (x << (32 - n))) & M32
    v = list(cv) + IV[:4] + [ctr & M32, ctr >> 32, bl, flags]
    m = list(m)

    def g(a, b, c, d, x, y):
        v[a] = (v[a] + v[b] + x) & M32; v[d] = rot(v[d] ^ v[a], 16)
        v[c] = (v[c] + v[d]) & M32; v[b] = rot(v[b] ^ v[c], 12)
        v[a] = (v[a] + v[b] + y) & M32; v[d] = rot(v[d] ^ v[a], 8)
        v[c] = (v[c] + v[d]) & M32; v[b] = rot(v[b] ^ v[c], 7)

    for rnd in range(7):
        g(0, 4, 8, 12, m[0], m[1]); g(1, 5, 9, 13, m[2], m[3]); g(2, 6, 10, 14, m[4], m[5]); g(3, 7, 11, 15, m[6], m[7])
        g(0, 5, 10, 15, m[8], m[9]); g(1, 6, 11, 12, m[10], m[11]); g(2, 7, 8, 13, m[12], m[13]); g(3, 4, 9, 14, m[14], m[15])
        if rnd < 6:
            m = [m[P[i]] for i in range(16)]
    return [v[i] ^ v[i + 8] for i in range(8)] + [v[i + 8] ^ cv[i] for i in range(8)]


class Statement:
    def __init__(self, circ: Circuit, pub: Public, comp: tuple[dict, Sparse, Sparse] | None):
        self.circ, self.pub = circ, pub
        self.comp = comp
        meta = circ.meta
        self.k_log = meta["k_log"]
        self.g = meta["vus_per_block"]
        self.upv = meta["units_per_vu"]
        self.n = pub.n
        self.blocks_real = -(-self.n // self.g)
        self.nbl = max(3, (self.blocks_real - 1).bit_length())
        self.m = self.k_log + self.nbl
        self.m_pts = PT_LOCAL + self.nbl
        self.key = bytes.fromhex(meta["row_key"])
        self.dummy_digests = [bytes.fromhex(d) for d in meta["dummy"]["digests"]]
        self.C = private_of(circ)
        u = circ.unit
        ur = circ.unit_range
        first = next(r for r in circ.ranges if r.kind != "mask")
        if first.kind != "comp":
            raise ValueError("the pin must be a compression slot's constant (B)")
        self.pin = first.start() + Z_CONST
        self.B = Bounds(self.k_log, self.nbl, self.n, self.g, self.upv, ur.slot_log, ur.pos0, sum(u.in_groups[0].bits),
                        u.in_words * WORD, u.out_groups[0].col, len(u.in_groups[0].bits), len(u.out_groups[0].bits),
                        sum(p.words for p in circ.ports), sum(w for _, w in meta["out_ports"]), 1 << 16, 1 << 17)
        if len(u.in_groups) != 1 or len(u.out_groups) != 1 or u.out_groups[0].words != 1:
            raise ValueError("the bounds admit one leaf input group and one returned-output word per unit")
        if any(b != 16 for b in u.in_groups[0].bits + u.out_groups[0].bits):
            raise ValueError("leaves are 16 bits")
        self.regions = self._regions()

    # --- layout (Flock.Blake3Row) ---------------------------------------------------------------------------------------

    def comp_role(self, r: Range, s: int):
        if r.role == "runs":
            nb = self.circ.ports[r.port].nb
            x = r.slots[s // nb]
            return ("run", x[0], r.port, x[1], s % nb) if x is not None else ("dummy_run", s % nb)
        if r.role == "parents":
            ch = self.circ.ports[r.port].chunks
            return ("dummy_parent",) if s % ch == ch - 1 else ("parent", s // ch, r.port, s % ch)
        return ("dummy_parent",)

    def params(self, role) -> tuple[int, int, int]:
        if role[0] == "run":
            _, _, p, ch, j = role
            port = self.circ.ports[p]
            fin = j + 1 == port.nb
            f = KEYED_HASH | (CHUNK_START if j == 0 else 0) | ((CHUNK_END | (ROOT if port.chunks == 1 else 0)) if fin else 0)
            return ch, 64, f
        if role[0] == "dummy_run":
            return 0, 64, KEYED_HASH | (CHUNK_START if role[1] == 0 else 0)
        if role[0] == "parent":
            return 0, 64, PARENT | KEYED_HASH | (ROOT if role[3] == 0 else 0)
        return 0, 64, PARENT | KEYED_HASH

    def chunk_end(self, g: int, p: int, ch: int) -> int:
        nb = self.circ.ports[p].nb
        for r in self.circ.ranges:
            if r.role == "runs" and r.port == p and (g, ch) in r.slots:
                q = r.slots.index((g, ch))
                return r.slot(q * nb + nb - 1)
        raise ValueError(f"no run of VU {g} chunk {ch} of port {p}")

    def message_col(self, g: int, p: int, byte: int) -> int:
        nb = self.circ.ports[p].nb
        ch, rest = divmod(byte, 1024)
        j, off = divmod(rest, 64)
        return self.chunk_end(g, p, ch) - (nb - 1 - j) * (1 << COMP_LOG) + M_BASE + 8 * off

    def leaf_src(self, g: int, leaf: int, t: int) -> int:
        """The message bit that hashes bit t of input leaf `leaf` of VU g."""
        starts = np.cumsum([0] + [p.words for p in self.circ.ports])
        p = int(np.searchsorted(starts, leaf, side="right") - 1)
        return self.message_col(g, p, 2 * (leaf - int(starts[p]))) + t

    def delta_public(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Δ's protocol part: the compression slots' constants and chaining (the units' constant and leaf wiring are C's)."""
        da, db = [], []
        for r in self.circ.ranges:
            if r.kind != "comp":
                continue
            for s in range(r.count):
                base = r.slot(s)
                k = base + Z_CONST
                if k != self.pin:
                    da += [(k, k), (k, self.pin)]
                    db += [(k, k), (k, self.pin)]
                role = self.comp_role(r, s)
                if role[0] in ("run", "dummy_run") and role[-1] > 0:
                    for t in range(256):
                        i = base + CV_BASE + t
                        da += [(i, i), (i, base - (1 << COMP_LOG) + OUT_LO + t)]
                elif role[0] == "parent":
                    raise ValueError("this prototype's bounds admit single-chunk ports only")
        return da, db

    def delta_full(self, C: PrivateCircuit) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Δ for circuit C, as `Flock.Circuit.delta` builds it (the order does not matter to the fold)."""
        da, db = self.delta_public()
        ur = self.circ.unit_range
        for k in range(ur.count):
            base = ur.slot(k)
            i = base + C.const_pos
            da += [(i, i), (i, self.pin)]
            db += [(i, i), (i, self.pin)]
            g, u = divmod(k, self.upv)
            for j, leaf in enumerate(C.leaves_in[u]):
                for t in range(16):
                    i = base + 16 * j + t
                    src = self.leaf_src(g, leaf, t)
                    da += [(i, i), (i, src)]
                    db += [(i, i), (i, src)]
        return da, db

    def inst_of(self, b: int, g: int) -> int | None:
        i = b * self.g + g
        return i if i < self.n else None

    def _regions(self) -> list[Region]:
        kl = self.k_log
        out = []

        def mk(name, kind, free, fixed, **kw):
            mask = ((1 << kl) - 1) & ~sum(1 << f for f in free)
            if fixed & ~mask or len(free) > PT_LOCAL:
                raise ValueError(f"{name}: not an aligned region")
            return Region(name, kind, free, mask, fixed, **kw)

        for ri, r in enumerate(self.circ.ranges):
            if r.role != "runs":
                continue
            port = self.circ.ports[r.port]
            lr, lnb = _log2(r.count), _log2(port.nb)
            pb = b"".join(struct.pack("<QII", *self.params(self.comp_role(r, s))) for s in range(r.count))
            out.append(mk(f"params({ri})", "const", _bits(0, 7) + _bits(COMP_LOG, lr), r.start() + T_LO, const_bytes=pb))
            out.append(mk(f"cv_in({ri})", "const", _bits(0, 8) + _bits(COMP_LOG + lnb, lr - lnb), r.start() + CV_BASE,
                          const_bytes=self.key * len(r.slots)))
            if port.chunks != 1:
                raise ValueError("single-chunk ports only")
            out.append(mk(f"digest({r.port},{ri})", "digest", _bits(0, 8) + _bits(COMP_LOG + lnb, lr - lnb),
                          r.start() + ((port.nb - 1) << COMP_LOG) + OUT_LO, port=r.port, range_index=ri))
        u = self.circ.unit
        ur = self.circ.unit_range
        g0 = u.out_groups[0]
        out.append(mk("out", "out", _bits(0, 7 + _log2(g0.words)) + _bits(ur.slot_log, _log2(ur.count)), ur.start() + g0.col * WORD))
        return out

    def dummy_run_cv(self, p: int) -> bytes:
        cv = list(struct.unpack("<8I", self.key))
        for j in range(self.circ.ports[p].nb):
            ctr, bl, f = self.params(("dummy_run", j))
            cv = blake3_compress(cv, [0] * 16, ctr, bl, f)[:8]
        return struct.pack("<8I", *cv)

    def region_bytes(self, reg: Region, b: int, C: PrivateCircuit, pub: Public) -> bytes:
        """A region's known bytes in block b, for circuit C and data `pub` (the Lean verifier's `Region.bytes`)."""
        if reg.kind == "const":
            return reg.const_bytes
        if reg.kind == "digest":
            r = self.circ.ranges[reg.range_index]
            out = b""
            for x in r.slots:
                if x is None:
                    out += self.dummy_run_cv(reg.port)
                else:
                    i = self.inst_of(b, x[0])
                    out += pub.digests[i][reg.port] if i is not None else self.dummy_digests[reg.port]
            return out
        ur = self.circ.unit_range
        out = b""
        for k in range(ur.count):
            g, u = divmod(k, self.upv)
            w = 0
            for j, o in enumerate(C.leaves_out[u]):
                i = self.inst_of(b, g)
                v = pub.outputs[i][o] if i is not None else C.dummy_outputs[o]
                w |= v << (16 * j)
            out += gf.to_bytes(w)
        return out

    def region_point(self, reg: Region, r: list[int]) -> list[int]:
        kl = self.k_log
        p = [1 if (j < kl and (reg.mask >> j) & 1 and (reg.fixed >> j) & 1) else 0 for j in range(self.m)]
        for t, f in enumerate(reg.free):
            p[f] = r[t]
        for i in range(self.nbl):
            p[kl + i] = r[PT_LOCAL + i]
        return p

    def region_value(self, reg: Region, r: list[int], C: PrivateCircuit, pub: Public) -> int:
        nf = len(reg.free)
        wt = gf.eq_table(r[:nf])
        eb = gf.eq_table(r[PT_LOCAL:PT_LOCAL + self.nbl])
        acc = 0
        for b in range(1 << self.nbl):
            by = self.region_bytes(reg, b, C, pub)
            if len(by) * 8 != 1 << nf:
                raise ValueError(f"region {reg.name}: bytes are not its free bits")
            s = 0
            for i, x in enumerate(by):
                for t in range(8):
                    if (x >> t) & 1:
                        s ^= wt[8 * i + t]
            acc ^= gf.mul(eb[b], s)
        return acc

    # --- the lincheck fold ----------------------------------------------------------------------------------------------

    def _tensor(self, lw: list[int], rho: list[int]) -> np.ndarray:
        """e[s + 64 u] = lw[s] · eq(rho, u)."""
        return gf.vmul(gf.to_arr(gf.eq_table(rho))[:, None, :], gf.to_arr(lw)[None, :, :]).reshape(-1, 2)

    def _fold(self, alpha: int, lw: list[int], rho: list[int], unit: tuple[Sparse, Sparse] | None,
              delta: tuple[list, list]) -> np.ndarray:
        kl = self.k_log
        out = np.zeros((1 << kl, 2), dtype=np.uint64)
        al = gf.to_arr([alpha])[0]
        a1 = gf.to_arr([alpha ^ 1])[0]
        bases = {}
        for r in self.circ.ranges:
            sl = r.slot_log
            if r.kind == "unit" and unit is None:
                continue
            if (r.kind, sl) not in bases:
                e_lo = self._tensor(lw, rho[:sl - 6])
                if r.kind == "comp":
                    _, ca, cb = self.comp
                    bases[(r.kind, sl)] = gf.vmul(al, ca.fold_t(e_lo)) ^ cb.fold_t(e_lo)
                elif r.kind == "unit":
                    bases[(r.kind, sl)] = gf.vmul(al, unit[0].fold_t(e_lo)) ^ unit[1].fold_t(e_lo)
                else:
                    bases[(r.kind, sl)] = gf.vmul(a1, e_lo)
            base = bases[(r.kind, sl)]
            e_hi = gf.to_arr(gf.eq_table(rho[sl - 6:]))
            for k in range(r.count):
                q = r.pos0 + k
                out[q << sl:(q + 1) << sl] = gf.vmul(e_hi[q], base)
        e = self._tensor(lw, rho)
        da, db = delta
        if da:
            ia, ca_ = np.array(da).T
            np.bitwise_xor.at(out, ca_, gf.vmul(al, e[ia]))
        if db:
            ib, cb_ = np.array(db).T
            np.bitwise_xor.at(out, cb_, e[ib])
        return out

    def unit_matrices(self, C: PrivateCircuit) -> tuple[Sparse, Sparse]:
        """U's full A and B (implied rows restored), 2^unit_log square."""
        k = 1 << self.circ.unit_range.slot_log
        ra = [[] for _ in range(k)]
        rb = [[] for _ in range(k)]
        for r in list(range(self.B.leaf_rows)) + [C.const_pos]:
            ra[r].append(r)
            rb[r].append(r)
        for r, c in C.a:
            ra[r].append(c)
        for r, c in C.b:
            rb[r].append(c)
        return Sparse.of_rows(ra, k), Sparse.of_rows(rb, k)

    def full_fold(self, C: PrivateCircuit, alpha: int, lw: list[int], rho: list[int]) -> np.ndarray:
        """αA_0ᵀe + B_0ᵀe for circuit C (`Flock.Stmt.fold`)."""
        return self._fold(alpha, lw, rho, self.unit_matrices(C), self.delta_full(C))

    def public_fold(self, alpha: int, lw: list[int], rho: list[int]) -> np.ndarray:
        """The same fold over B's entries only: the compression and mask slots and Δ's protocol part."""
        return self._fold(alpha, lw, rho, None, self.delta_public())
