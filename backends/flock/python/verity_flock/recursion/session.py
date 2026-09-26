"""A recorded circuit-statement session (PROTOCOL.md §5, §6, §8) as the inner protocol of the private track.

`decode_proof` reads a proof file (§8). `replay` runs the §9–§13 transcript order over a sink: `Render` produces each
round's framed bytes from a decoded proof (the bytes the prover committed before that round's coins), and `Layout`
produces the same rounds as segments, constant bytes or named message fields, which is all V[B] needs to parse a
round it only holds as a witness. The two share one replay, so a round's bytes and its layout cannot disagree.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import gf

Q = [218, 106, 71, 53, 43, 36, 32]


# ---------------------------------------------------------------------------------------------------------------------
# the fast100 schedule (§15)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    rate: int
    q: int
    cols: int
    d: int
    strata: tuple[int, ...]
    cap: int
    lanes: int


@dataclass(frozen=True)
class Schedule:
    m: int
    n: int
    k0: int
    levels: tuple[Level, ...]
    ext: int


def _decompose(q: int, d: int) -> tuple[int, ...]:
    rem, out = q, []
    while rem:
        c = min(rem.bit_length() - 1, d)
        out.append(c)
        rem -= 2**c
    return tuple(out)


def fast100(m: int) -> Schedule:
    if not 22 <= m <= 35:
        raise ValueError(f"m = {m}: no fast100 schedule")
    nlev = 3 if m <= 24 else 4 if m <= 27 else 5 if m <= 30 else 6 if m <= 33 else 7
    k0 = 4 if m == 28 else 5 if m == 29 else 6
    n = m - 7
    ls, cols = [], n - k0
    for lv in range(nlev):
        rate, q = (4, 53) if (m == 22 and lv == 2) else (lv + 1, Q[lv])
        d = cols + rate
        st = _decompose(q, d)
        ls.append(Level(rate, q, cols, d, st, st[0], k0 if lv == 0 else 4))
        cols -= 3
    return Schedule(m, n, k0, tuple(ls), ls[-1].cols)


# ---------------------------------------------------------------------------------------------------------------------
# proof container (§8)
# ---------------------------------------------------------------------------------------------------------------------

class _R:
    def __init__(self, b: bytes, h: int):
        self.b, self.i, self.h = b, 0, h

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise ValueError("proof: truncated")
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def f(self) -> int:
        return gf.from_bytes(self.take(16))

    def vec(self, fn, cap: int = 1 << 26) -> list:
        n = self.u64()
        if n > cap:
            raise ValueError(f"proof: vector length {n}")
        return [fn() for _ in range(n)]

    def hsh(self) -> bytes:
        return self.take(self.h)


@dataclass
class Proof:
    cap: list[bytes]
    params: tuple
    zc: dict
    lc: dict
    rs: list[tuple[list[int], int]]
    bn: list[int]
    lig: dict


def decode_proof(b: bytes, hash_len: int = 32) -> Proof:
    r = _R(b, hash_len)
    cap = r.vec(r.hsh)
    m, lir, k0 = r.u64(), r.u64(), r.u64()
    prof = r.u32()
    tag = r.u8()
    if tag not in (0, 1):
        raise ValueError("proof: option tag")
    lanes = r.u64() if tag else None
    mh = r.u32()
    pair = lambda: (r.f(), r.f())
    zc = dict(ab=r.vec(r.f), c=r.vec(r.f), rounds=r.vec(pair), fa=r.f(), fb=r.f(), fc=r.f(), nonces=r.vec(r.u64))
    lc = dict(rounds=r.vec(pair), zp=r.vec(r.f), me=r.vec(pair), nonces=r.vec(r.u64))
    rs = r.vec(lambda: (r.vec(r.f), r.u64()))
    bn = r.vec(r.u64)
    level = lambda: (r.vec(lambda: r.vec(r.f)), r.vec(r.hsh))
    lig = dict(icap=r.vec(r.hsh), iproof=level(), rcaps=r.vec(lambda: r.vec(r.hsh)), rproofs=r.vec(level))
    lig["yr"] = r.vec(r.f)
    lig["frows"] = r.vec(lambda: r.vec(r.f))
    lig["fsibs"] = r.vec(r.hsh)
    lig["sc"] = r.vec(pair)
    lig["msgs"] = r.vec(lambda: ((r.f(), r.f()), (r.f(), r.f())))
    lig["qn"] = r.vec(r.u64)
    lig["ood"] = r.vec(r.f)
    lig["fold"] = r.vec(r.u64)
    lig["cbn"] = r.vec(r.u64)
    lig["consn"] = r.vec(r.u64)
    return Proof(cap, (m, lir, k0, prof, lanes, mh), zc, lc, rs, bn, lig)


# ---------------------------------------------------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------------------------------------------------

def hdr(op: int, kind: int, n: int) -> bytes:
    return bytes([op, kind]) + b"\0" * 6 + struct.pack("<Q", n)


def pad16(b: bytes) -> bytes:
    return b + b"\0" * ((-len(b)) % 16)


class Render:
    """Framed bytes per round, from a decoded proof; coins come from the recorded rounds."""

    def __init__(self, domain: bytes, coins: list[list[int]] | None = None):
        self.rounds: list[bytes] = []
        self.pending = hdr(1, 0, len(domain)) + pad16(domain)
        self.coins = coins
        self.ns: list[int] = []
        self.names: list = []

    def label(self, b: bytes):
        self.pending += hdr(2, 0, len(b)) + pad16(b)

    def obs(self, key, v: int):
        self.pending += hdr(3, 1, 1) + gf.to_bytes(v)

    def obs_slice(self, key, vs: list[int]):
        self.pending += hdr(3, 2, len(vs)) + b"".join(gf.to_bytes(v) for v in vs)

    def obs_bytes(self, key, b: bytes):
        self.pending += hdr(5, 0, len(b)) + pad16(b)

    def squeeze(self, kind: int, n: int, name=None) -> list[int]:
        self.pending += hdr(4, kind, n)
        if n == 0:
            return []
        k = len(self.rounds)
        self.rounds.append(self.pending)
        self.ns.append(n)
        self.names.append(name)
        self.pending = b""
        if self.coins is None or k >= len(self.coins):
            return [0] * n
        return self.coins[k]


@dataclass
class Seg:
    """A run of bytes in a round: constant `const`, or the message field `key` of `size` bytes (elements of 16 bytes
    for field values)."""
    offset: int
    size: int
    const: bytes | None = None
    key: tuple | None = None


class Layout:
    """The same rounds as `Render`, with every message field named instead of rendered."""

    def __init__(self, domain: bytes):
        self.rounds: list[list[Seg]] = []
        self.ns: list[int] = []
        self.names: list = []
        self.cur: list[Seg] = []
        self.at = 0
        self._const(hdr(1, 0, len(domain)) + pad16(domain))

    def _const(self, b: bytes):
        if self.cur and self.cur[-1].const is not None:
            last = self.cur[-1]
            last.const += b
            last.size += len(b)
        else:
            self.cur.append(Seg(self.at, len(b), const=bytes(b)))
        self.at += len(b)

    def _val(self, key, size: int):
        self.cur.append(Seg(self.at, size, key=key))
        self.at += size

    def label(self, b: bytes):
        self._const(hdr(2, 0, len(b)) + pad16(b))

    def obs(self, key, v):
        self._const(hdr(3, 1, 1))
        self._val(key, 16)

    def obs_slice(self, key, vs):
        self._const(hdr(3, 2, len(vs)))
        self._val(key, 16 * len(vs))

    def obs_bytes(self, key, b):
        n = len(b)
        self._const(hdr(5, 0, n))
        self._val(key, n)
        if (-n) % 16:
            self._const(b"\0" * ((-n) % 16))

    def squeeze(self, kind: int, n: int, name=None):
        self._const(hdr(4, kind, n))
        if n == 0:
            return [0] * 0
        self.rounds.append(self.cur)
        self.ns.append(n)
        self.names.append(name)
        self.cur, self.at = [], 0
        return [0] * n

    def round_bytes(self, k: int) -> int:
        return sum(s.size for s in self.rounds[k])


@dataclass(frozen=True)
class Shape:
    """What fixes a rep's transcript: the statement's m and k_log, the claim count, the schedule and the hash width."""
    m: int
    k_log: int
    claims: int
    hash_len: int

    @property
    def sched(self) -> Schedule:
        return fast100(self.m)


class _Stub:
    """A proof-shaped object whose values are all zero: enough for `Layout`, which never reads values."""

    def __init__(self, sh: Shape):
        s, h = sh.sched, sh.hash_len
        R = len(s.levels) - 1
        self.cap = [bytes(h)] * 2 ** s.levels[0].cap
        self.zc = dict(ab=[0] * 64, c=[0] * 64, rounds=[(0, 0)] * (sh.m - 6), fa=0, fb=0)
        self.lc = dict(rounds=[(0, 0)] * (sh.k_log - 6), zp=[0] * 64)
        self.rs = [([0] * 128, 0)] * sh.claims
        nmsg = s.k0 + 8 + 7 * (R - 1)
        self.lig = dict(icap=self.cap, rcaps=[[bytes(h)] * 2 ** s.levels[i + 1].cap for i in range(R)],
                        msgs=[((0, 0), (0, 0))] * nmsg, ood=[0] * (1 + 2 * R), qn=[0] * (R + 1), consn=[0] * (R + 1),
                        cbn=[0] * (3 * R + 2), yr=[0] * 2 ** (s.ext + 1))


def replay(sink, sh: Shape, digest: bytes, p, T_of=None):
    """§9.1–§13.3 in order over `sink`. `T_of(rs, rpps, gammas)` gives the batched target the prover absorbs; `Layout`
    names it as the field ("T",)."""
    m, s = sh.m, sh.sched
    k0 = s.k0
    zc, lc, lig = p.zc, p.lc, p.lig
    sink.label(b"flock-r1cs-v0")
    sink.obs_bytes(("digest",), digest)
    sink.obs_bytes(("cap",), b"".join(p.cap))
    sink.label(b"flock-zerocheck-v0")
    sink.squeeze(2, 6, ("zc_rskip",))
    sink.squeeze(2, m - 13, ("zc_rout",))
    sink.obs_slice(("zc_ab",), zc["ab"])
    sink.obs_slice(("zc_c",), zc["c"])
    sink.squeeze(1, 1, ("zc_z",))
    for i, (g1, ginf) in enumerate(zc["rounds"]):
        sink.obs(("zc_g1", i), g1)
        sink.obs(("zc_ginf", i), ginf)
        sink.squeeze(1, 1, ("zc_rho", i))
    sink.obs(("zc_fa",), zc["fa"])
    sink.obs(("zc_fb",), zc["fb"])
    sink.label(b"flock-lincheck-v0")
    sink.squeeze(1, 1, ("lc_alpha",))
    sink.squeeze(1, 1, ("lc_beta",))
    for j, (q1, qinf) in enumerate(lc["rounds"]):
        sink.obs(("lc_q1", j), q1)
        sink.obs(("lc_qinf", j), qinf)
        sink.squeeze(1, 1, ("lc_t", j))
    sink.obs_slice(("lc_zp",), lc["zp"])
    sink.squeeze(1, 1, ("lc_rp",))
    sink.label(b"flock-pcs-open-batch-v0")
    rpps = []
    for k, (s_k, _nonce) in enumerate(p.rs):
        sink.label(b"flock-ring-switch-v0")
        sink.obs_slice(("rs", k), s_k)
        rpps.append(sink.squeeze(2, 7, ("rs_rpp", k)))
    gam = sink.squeeze(2, len(p.rs), ("rs_gamma",))
    T = T_of([x[0] for x in p.rs], rpps, gam) if T_of else 0
    n = m - 7
    it = {k: iter(range(len(lig[k]))) for k in ("msgs", "ood", "qn", "cbn", "consn")}
    oods = iter(range(len(lig["ood"])))

    def nxt(k):
        return next(it[k])

    def msg():
        i = nxt("msgs")
        (a0, a1), (b0, b1) = lig["msgs"][i]
        sink.obs_slice(("lig_msg", i, 0), [a0, a1])
        sink.obs_slice(("lig_msg", i, 1), [b0, b1])

    def pow_(lst):
        i = nxt(lst)
        sink.obs_bytes(("pow", lst, i), struct.pack("<Q", lig[lst][i]))

    def ood(dim, with_msg):
        i = next(oods)
        sink.squeeze(2, dim, ("ood_z", i))
        assert nxt("ood") == i
        sink.obs(("lig_ood", i), lig["ood"][i])
        if with_msg:
            msg()
        pow_("cbn")
        sink.squeeze(1, 1, ("ood_beta", i))

    def queries(level):
        pow_("qn")
        sink.squeeze(2, s.levels[level].q, ("q_coins", level))
        pow_("consn")
        sink.squeeze(2, math.ceil(math.log2(s.levels[level].q)), ("q_alpha", level))

    sink.label(b"flock-ligerito-basis-f256-split-v0")
    sink.obs(("T",), T)
    sink.obs_bytes(("icap",), b"".join(lig["icap"]))
    ood(n, False)
    msg()
    for c in range(k0):
        sink.squeeze(2, 2, ("chal", 0, c))
        msg()
    split = n - k0 + 1
    R = len(s.levels) - 1
    sink.obs_bytes(("rcap", 0), b"".join(lig["rcaps"][0]))
    ood(split, True)
    ood(split, True)
    queries(0)
    msg()
    pow_("cbn")
    sink.squeeze(1, 1, ("cons_beta", 0))
    for i in range(R):
        for c in range(4):
            sink.squeeze(2, 2, ("chal", i + 1, c))
            msg()
        ext = split - 4
        if i + 1 < R:
            split = ext + 1
            sink.obs_bytes(("rcap", i + 1), b"".join(lig["rcaps"][i + 1]))
            ood(split, True)
            ood(split, True)
            queries(i + 1)
            msg()
            pow_("cbn")
            sink.squeeze(1, 1, ("cons_beta", i + 1))
        else:
            for t, w in enumerate(lig["yr"]):
                sink.obs(("lig_yr", t), w)
            queries(R)
            pow_("cbn")
            sink.squeeze(1, 1, ("cons_beta", R))
    return sink


def layout(sh: Shape, domain: bytes) -> Layout:
    return replay(Layout(domain), sh, bytes(32), _Stub(sh))


def batched_target(rs: list[list[int]], rpps: list[list[int]], gammas: list[int]) -> int:
    """§12.3: T = Σ_k γ_k Σ_u eq(r''_k, u) t_k[u], t_k the bit transpose of s_k."""
    T = 0
    for s_k, rpp, g in zip(rs, rpps, gammas):
        eq = gf.eq_table(rpp)
        beta = 0
        for u in range(128):
            t = 0
            for v in range(128):
                if (s_k[v] >> u) & 1:
                    t |= 1 << v
            beta ^= gf.mul(eq[u], t)
        T ^= gf.mul(g, beta)
    return T


# ---------------------------------------------------------------------------------------------------------------------
# the session record (§5)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Round:
    k: int
    n: int
    sha: bytes
    coins: list[int]
    g: int | None
    msg: bytes | None


@dataclass
class Stream:
    name: str
    label: str
    root_sha256: bytes
    rounds: list[Round]


@dataclass
class Session:
    raw: dict
    hello: str
    digest: bytes
    root_b: bytes
    points: list[list[int]]
    points_g: int | None
    streams: dict[str, Stream]
    proof_sha: dict[str, bytes]
    proofs: dict[int, bytes] = field(default_factory=dict)


def load_session(d: str | Path, table: str = "circuit") -> Session:
    d = Path(d)
    r = json.loads((d / "session.json").read_text())
    binding = r["config"]["tables"][0]["binding"]
    digest = bytes(int(x) for x in binding.split("[", 1)[1].split("]")[0].split(","))
    link = r["link"]
    streams = {}
    for s in r["streams"]:
        rounds = [Round(x["k"], x["n"], bytes.fromhex(x["msg_sha256"]), [gf.from_hex(c) for c in x["coins"]], x.get("g"),
                        bytes.fromhex(x["msg"]) if x.get("msg") is not None else None) for x in s["rounds"]]
        streams[s["stream"]] = Stream(s["stream"], s["label"], bytes.fromhex(s["root_sha256"]), rounds)
    sess = Session(r, r["hello"], digest, bytes.fromhex(link["root_b"][table]),
                   [[gf.from_hex(c) for c in pt] for pt in link["points"]], link.get("points_g"), streams,
                   {x["stream"]: bytes.fromhex(x["sha256"]) for x in r["proof_sha256"]})
    for rep in (0, 1):
        f = d / f"{table}.rep{rep}.bin"
        if f.exists():
            sess.proofs[rep] = f.read_bytes()
    return sess


def render_rounds(sh: Shape, sess: Session, rep: int, proof: Proof, table: str = "circuit") -> list[bytes]:
    st = sess.streams[f"{table}/rep{rep}"]
    rd = Render(st.label.encode(), [x.coins for x in st.rounds])
    replay(rd, sh, sess.digest, proof, batched_target)
    return rd.rounds


def rounds_match(rounds: list[bytes], st: Stream) -> list[int]:
    """Indices of rounds whose framed bytes do not answer the record (retained bytes, else the digest)."""
    bad = [k for k, (b, x) in enumerate(zip(rounds, st.rounds))
           if (x.msg is not None and b != x.msg) or (x.msg is None and hashlib.sha256(b).digest() != x.sha)]
    if len(rounds) != len(st.rounds):
        bad.append(-1)
    return bad


def named_coins(lay: Layout, st: Stream) -> dict:
    """The recorded coins of a rep's rounds, by the name the replay gives each squeeze."""
    if len(lay.names) != len(st.rounds):
        raise ValueError("the record's rounds are not the schedule's")
    return {name: x.coins for name, x in zip(lay.names, st.rounds)}
