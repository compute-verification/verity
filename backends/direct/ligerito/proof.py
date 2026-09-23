"""Files of the Ligerito backend (lane ligerito-relation): the batch proof ``LGTO0001``, its statement ``LGTS0001`` and the
verification key ``LGVK0001``.  numpy only (importable where torch is not), all integers little-endian, field elements
``u32`` in ``[0, p)`` (BabyBear), extension elements 6 of them (``F_p[x] / (x^6 - 31)``, coordinate i = coefficient of x^i).

~~~text
LGTS0001  statement   magic | str relation | 32 B sys_id (protocol.system_id) | 32 B vk_sha256 | u32 l | u32 steps | u32 K
                      | u32 S | S x u32 n_vus (0 = a pad sub-batch: l pad units, y = 0) | u8 word_bytes | u8 y_bytes
                      | for every s with n_vus[s] > 0: l x K a | l x K b | l x y   (the relation's word widths; exactly the
                      words of ligero-statement/v4 per sub-batch, units in column order, pad units zero)
LGVK0001  key         magic | str relation | 32 B sys_id | u32 m | u32 n_i | u32 n_k | u32 K | u32 n_links | n_links x u32 c_row
                      | u32 n_virt | n_virt x (str name, u32 row) | 3 x (u32 nnz | nnz x (u32 k, u32 row, u32 coef))  [A, B, C]
                      (layout.layout_for + layout.constraints of the system: independent of l and n_vus; the Rust verifier
                      may read it instead of re-deriving it from system.bin -- its digest is in the statement)
LGTO0001  proof       magic | u32 len | params JSON (absorbed verbatim) | u32 len | framing JSON (not absorbed)
                      | u32 len | sumcheck (sumcheck.py LGSC0003) | L x 32 B root_i
                      | for i < L: k'_i x (3 x 6 u32) PCS column-sumcheck messages (g(0), g(1), g(2))
                      | for i < L: |S_i| x W_i u32 opened codeword rows (W_1 = 2^{k'_1} base words; W_i = 6 * 2^{k'_i}
                        plane-major for i >= 2) | sib_len[i] bytes BLAKE3 multi-path (proto merkle_multi.plan order)
                      | 2^{k_L} x 6 u32 final vector y_L
    str               u32 len | utf-8 bytes
    params            {"coins","kprime","l","n","n_claims","order","queries","rate_log2","relation","splits","zero_blocks","zk"
                       [,"t_pad"]}   (compact JSON, sorted keys; zero_blocks = [[b, pfx], ...], n_claims = 1 + n_links + |zero_blocks|)
    framing           {"sib_len": [...], "final_len": int[, "t_pad": int]}
    (zk only)         | L x (t_pad x 6 u32) ybar_i  (ligerito-pcs-fast's ZK column padding: absorbed as b"ybar" right after
                        root_{i+1}, the last after y; row checks subtract sum_t eta_s^{2^{k_i} + t} ybar_i[t])
~~~

The committed polynomial is ``w`` = the layout's ``z`` with the virtual rows ``[m, 2^n_i)`` zeroed, in ROW-MAJOR order
(``order = "row-major"``): flat index ``x = c + C * i`` (``c = s * l + j`` the column, ``i`` the row), so PCS variable ``t <
n_c`` is column bit ``t`` and ``t >= n_c`` is row bit ``t - n_c``.  The sumcheck's claim points ``(p_i || p_c)`` enter the PCS
as ``(p_c || p_i)``.

``zk = true`` (``order = "zk-interleaved"``): only the virtual rows ``[m, m + n_virt)`` are zeroed (the free rows above hold
ligerito-zk's mask rows, ``prove.zk_layout``), and the index map is ligerito-zk's ``f_index``: with ``a = k_L - 3``,
``x = (c mod 2^a) + 2^a i + 2^{a + n_i} (c >> a)``; claim points enter the PCS as ``zk.permute_point``:
``(p_c[:a] || p_i || p_c[a:])``.

Claims (``J = 1 + n_links + |zero_blocks|``, in this order; points in the sumcheck's order ``(p_i || p_c)`` before the rotation
above): ``w~(r_i, r_c) = values[0]``; ``w~(bits(c_x), rho_c) = values[1 + x]`` for each chain row ``c_x``; then per zero block
``(b, pfx)``: ``w~(r_i[:b] || bits(pfx, n_i - b) || r_c) = 0`` (the first claim's ``r_i`` / ``r_c``; ``bits`` LSB first).  The
blocks are the aligned cover (greedy from the low end, largest aligned block that fits) of every run of consecutive virtual rows
(``prove.zero_blocks``; ligerito-verify ``relation::virtual_zero_claims``): the verifier adds the public rows to ``w`` itself,
which is sound only if ``w`` is zero there.

Live coins (``coins = "live"``): verifier coin slot k (32 bytes) = lane live-2's ``expand_challenge(ctx, k, label_k, r_k, 32)``;
``prove.StreamCoins`` records the slots (the dump's ``.coins``) and the binding (statement, commitments, openings, labels).

Transcript (one Coins object, ``transcript.py``; ``FiatShamirCoins()`` default label ``ligerito-proto/v1``):
    absorb("lgto/params", params JSON bytes) | absorb("lgto/stmt", sha256(statement)) | absorb("lgto/vk", sha256(key))
    | absorb("root", root_1) | the sumcheck (sumcheck.py: challenge "zc/tau", absorbs of its messages, ..., absorb "values")
    | absorb("z", the J claim points in PCS order, (J, n, 6) u32) | beta <- challenge("beta", J)
    | proto/pcs.py from its first column round on, with the running claim sum_j beta_j v_j and J eq terms (coef beta_j):
      k'_1 x [absorb("g", g) -> challenge("r", 1)]; for i >= 2: absorb("root", root_i) | indices("S", n_{i-1}, |S_{i-1}|)
      | absorb("rows", u32 rows) | absorb("sib", siblings) | challenge("alpha", 1 + |S|) | k'_i x [g -> r];
      absorb("y", y_L) | indices("S", n_L, |S_L|).
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

import numpy as np

P = 2013265921
D = 6
PROOF_MAGIC, STMT_MAGIC, VK_MAGIC = b"LGTO0001", b"LGTS0001", b"LGVK0001"
HASH = "blake3"

__all__ = ["FS_QUERIES_LOG2", "Key", "PcsProof", "Proof", "Statement", "aligned_cover", "read_key", "read_proof", "read_statement",
           "sha256", "soundness_claim", "zero_blocks", "zero_points"]


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def aligned_cover(lo: int, hi: int) -> list[tuple[int, int]]:
    """Aligned blocks ``{i : i >> b = pfx}`` exactly covering ``[lo, hi)``, as ``(b, pfx)``, greedy from ``lo`` (the largest
    aligned block that fits); ligerito-verify ``relation::aligned_cover``."""
    out, i = [], lo
    while i < hi:
        b = 0
        while i % (2 << b) == 0 and i + (2 << b) <= hi:
            b += 1
        out.append((b, i >> b))
        i += 1 << b
    return out


def zero_blocks(virt: dict) -> list[tuple[int, int]]:
    """The zero claims' blocks ``(b, pfx)`` for the virtual rows ``virt`` (name -> row; the key's or ``Layout.virt``): the
    aligned cover of every run of consecutive rows, runs in row order (fp8-ada: 3577 | 3578-79 | 3580-83 | 3584-3711 |
    3712-75 | 3776), ZK or not; ligerito-verify ``relation::virtual_zero_claims``."""
    rows = sorted(set(virt.values()))
    out, k = [], 0
    while k < len(rows):
        lo = rows[k]
        while k + 1 < len(rows) and rows[k + 1] == rows[k] + 1:
            k += 1
        out += aligned_cover(lo, rows[k] + 1)
        k += 1
    return out


def zero_points(n_i: int, point: np.ndarray, blocks: list[tuple[int, int]]) -> list[np.ndarray]:
    """Per block ``(b, pfx)``: ``(r_i[:b] || bits(pfx) || r_c)`` in the sumcheck's order (``n_i`` row bits low -> high, then
    the columns), ``(r_i || r_c)`` = the first claim's point ((n, 6)): ``r_i`` from the combined sumcheck, ``r_c`` from the
    zero-check, both drawn after root_1.  The columns must not come from the combined sumcheck's ``rho``: row bit t and
    column bit t would share a coordinate and two entries with those bits swapped cancel identically."""
    out = []
    for b, pfx in blocks:
        p = np.array(point, dtype=np.int64, copy=True)
        p[b:n_i] = 0
        p[b:n_i, 0] = [(pfx >> t) & 1 for t in range(n_i - b)]
        out.append(p)
    return out


#: hash-query budget of the stated Fiat-Shamir bound
FS_QUERIES_LOG2 = 64


def soundness_claim(snd: dict, coins: str, verifier_record: bool = False) -> dict:
    """What a run's soundness is (red-team F12).  ``snd`` (``Prover.soundness``) is the interactive union: it holds only
    when every coin is drawn by the verifier after the prover's preceding message.  ``claimed_log2``:
      live + ``verifier_record`` (the issuing verifier's own session accepted: the coin log is authenticated) -> the union;
      fiat-shamir -> log2 Q + the largest term (each term bounds the round-by-round error of the rounds it covers; every
        coin including the query sets is FS, so the query terms dominate), Q = 2^FS_QUERIES_LOG2 hash queries;
      local, or live slots replayed from a prover-side record -> None (the prover knows / chose the coins)."""
    union = snd["total_log2"]
    out = {"coins": coins, "interactive_union_log2": union}
    if coins == "live" and verifier_record:
        return {**out, "claimed_log2": union,
                "basis": "interactive: coins drawn by the live verifier after each prover message (its session accepted)"}
    if coins == "fiat-shamir":
        worst = max(snd["terms_log2"], key=snd["terms_log2"].get)
        return {**out, "claimed_log2": FS_QUERIES_LOG2 + snd["terms_log2"][worst], "queries_log2": FS_QUERIES_LOG2,
                "worst_term": worst, "basis": f"Fiat-Shamir: Q x the largest round-by-round term, Q = 2^{FS_QUERIES_LOG2} hash queries"}
    why = {"local": "local-seed coins are known to the prover before it commits (diagnostic)",
           "live": "live coin slots from a prover-side record, not authenticated by the issuing verifier"}.get(coins, f"coins {coins!r}")
    return {**out, "claimed_log2": None, "basis": f"none: {why}"}


class _W:
    def __init__(self):
        self.parts: list[bytes] = []

    def u8(self, v: int):
        self.parts.append(struct.pack("<B", v))

    def u32(self, v: int):
        self.parts.append(struct.pack("<I", v))

    def raw(self, b: bytes):
        self.parts.append(bytes(b))

    def str(self, s: str):
        b = s.encode()
        self.u32(len(b))
        self.raw(b)

    def blob(self, b: bytes):
        self.u32(len(b))
        self.raw(b)

    def arr(self, a: np.ndarray, dtype: str):
        self.raw(np.ascontiguousarray(np.asarray(a).astype(dtype)).tobytes())

    def bytes(self) -> bytes:
        return b"".join(self.parts)


class _R:
    def __init__(self, b: bytes):
        self.b, self.pos = b, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.b):
            raise ValueError("truncated")
        out = self.b[self.pos:self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return struct.unpack("<B", self.take(1))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def str(self) -> str:
        return self.take(self.u32()).decode()

    def blob(self) -> bytes:
        return self.take(self.u32())

    def arr(self, dtype: str, shape) -> np.ndarray:
        n = int(np.prod(shape)) if len(shape) else 1
        size = np.dtype(dtype).itemsize
        return np.frombuffer(self.take(n * size), dtype=dtype).astype(np.int64).reshape(shape)

    def end(self):
        if self.pos != len(self.b):
            raise ValueError(f"{len(self.b) - self.pos} trailing bytes")


def _udtype(nbytes: int) -> str:
    return {1: "<u1", 2: "<u2", 4: "<u4"}[nbytes]


# ---------------------------------------------------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Statement:
    relation: str
    sys_id: bytes
    vk_sha256: bytes
    l: int
    steps: int
    K: int
    n_vus: list[int]                    # per sub-batch (S entries; 0 = pad)
    word_bytes: int
    y_bytes: int
    a: list[np.ndarray] = field(default_factory=list)      # per real sub-batch (l, K) int64
    b: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)      # per real sub-batch (l,) int64

    @property
    def S(self) -> int:
        return len(self.n_vus)

    @property
    def total_vus(self) -> int:
        return sum(self.n_vus)

    def to_bytes(self) -> bytes:
        w = _W()
        w.raw(STMT_MAGIC)
        w.str(self.relation)
        w.raw(self.sys_id)
        w.raw(self.vk_sha256)
        for v in (self.l, self.steps, self.K, self.S):
            w.u32(v)
        for v in self.n_vus:
            w.u32(v)
        w.u8(self.word_bytes)
        w.u8(self.y_bytes)
        real = [s for s, nv in enumerate(self.n_vus) if nv > 0]
        if not (len(self.a) == len(self.b) == len(self.y) == len(real)):
            raise ValueError("statement: one (a, b, y) per real sub-batch")
        wd, yd = _udtype(self.word_bytes), _udtype(self.y_bytes)
        for a, b, y in zip(self.a, self.b, self.y):
            if a.shape != (self.l, self.K) or b.shape != (self.l, self.K) or y.shape != (self.l,):
                raise ValueError("statement: word array shape")
            w.arr(a, wd)
            w.arr(b, wd)
            w.arr(y, yd)
        return w.bytes()

    def digest(self) -> bytes:
        return sha256(self.to_bytes())


def read_statement(buf: bytes) -> Statement:
    r = _R(buf)
    if r.take(8) != STMT_MAGIC:
        raise ValueError("not an LGTS0001 statement")
    rel = r.str()
    sys_id, vk = r.take(32), r.take(32)
    l, steps, K, S = r.u32(), r.u32(), r.u32(), r.u32()
    n_vus = [r.u32() for _ in range(S)]
    wb, yb = r.u8(), r.u8()
    st = Statement(rel, sys_id, vk, l, steps, K, n_vus, wb, yb)
    for nv in n_vus:
        if nv > 0:
            st.a.append(r.arr(_udtype(wb), (l, K)))
            st.b.append(r.arr(_udtype(wb), (l, K)))
            st.y.append(r.arr(_udtype(yb), (l,)))
    r.end()
    return st


# ---------------------------------------------------------------------------------------------------------------------
# Verification key
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Key:
    relation: str
    sys_id: bytes
    m: int
    n_i: int
    n_k: int
    K: int
    c_rows: list[int]
    virt: dict                          # name -> row
    coo: dict                           # side "a"/"b"/"c" -> (k, row, coef) int64 numpy arrays

    def to_bytes(self) -> bytes:
        w = _W()
        w.raw(VK_MAGIC)
        w.str(self.relation)
        w.raw(self.sys_id)
        for v in (self.m, self.n_i, self.n_k, self.K, len(self.c_rows)):
            w.u32(v)
        for c in self.c_rows:
            w.u32(c)
        w.u32(len(self.virt))
        for name, row in self.virt.items():
            w.str(name)
            w.u32(row)
        for side in "abc":
            k, row, coef = (np.asarray(t, dtype=np.int64) for t in self.coo[side])
            w.u32(len(k))
            w.arr(np.stack((k, row, coef % P), axis=1), "<u4")
        return w.bytes()

    def digest(self) -> bytes:
        return sha256(self.to_bytes())


def read_key(buf: bytes) -> Key:
    r = _R(buf)
    if r.take(8) != VK_MAGIC:
        raise ValueError("not an LGVK0001 key")
    rel = r.str()
    sys_id = r.take(32)
    m, n_i, n_k, K, nl = (r.u32() for _ in range(5))
    c_rows = [r.u32() for _ in range(nl)]
    virt = {}
    for _ in range(r.u32()):
        name = r.str()
        virt[name] = r.u32()
    coo = {}
    for side in "abc":
        nnz = r.u32()
        t = r.arr("<u4", (nnz, 3))
        coo[side] = (t[:, 0].copy(), t[:, 1].copy(), t[:, 2].copy())
    r.end()
    return Key(rel, sys_id, m, n_i, n_k, K, c_rows, virt, coo)


# ---------------------------------------------------------------------------------------------------------------------
# Proof
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class PcsProof:
    """The multi-claim Ligerito evaluation proof (proto/pcs.py's ``Proof`` without ``v``: the claim values are the
    sumcheck's ``values``)."""
    roots: list[bytes]
    sumcheck: list[list[np.ndarray]]    # per committed round: k'_i messages (3, 6)
    rows: list[np.ndarray]              # per committed round: (|S_i|, W_i) int64
    siblings: list[bytes]
    final_y: np.ndarray                 # (2^{k_L}, 6)
    ybar: list = field(default_factory=list)    # ZK column padding: per committed round (t_pad, 6); [] without


def params_bytes(params: dict) -> bytes:
    return json.dumps(params, separators=(",", ":"), sort_keys=True).encode()


@dataclass
class Proof:
    params: dict
    sc_bytes: bytes                     # LGSC0002
    pcs: PcsProof
    params_raw: bytes = b""             # the absorbed bytes as read (== params_bytes(params) when written here)

    def absorbed_params(self) -> bytes:
        return self.params_raw or params_bytes(self.params)

    def to_bytes(self) -> bytes:
        w = _W()
        w.raw(PROOF_MAGIC)
        w.blob(self.absorbed_params())
        fr = {"sib_len": [len(s) for s in self.pcs.siblings], "final_len": int(self.pcs.final_y.shape[0])}
        if self.pcs.ybar:
            fr["t_pad"] = int(self.pcs.ybar[0].shape[0])
        w.blob(json.dumps(fr, separators=(",", ":")).encode())
        w.blob(self.sc_bytes)
        for r_ in self.pcs.roots:
            if len(r_) != 32:
                raise ValueError("root length")
            w.raw(r_)
        for msgs in self.pcs.sumcheck:
            for g in msgs:
                w.arr(np.asarray(g, dtype=np.int64) % P, "<u4")
        for rows, sib in zip(self.pcs.rows, self.pcs.siblings):
            w.arr(rows, "<u4")
            w.raw(sib)
        w.arr(np.asarray(self.pcs.final_y, dtype=np.int64) % P, "<u4")
        for yb in self.pcs.ybar:
            w.arr(np.asarray(yb, dtype=np.int64) % P, "<u4")
        return w.bytes()

    def n_bytes(self) -> dict:
        """Byte accounting of the file by section."""
        rows = sum(int(r_.size) * 4 for r_ in self.pcs.rows)
        sib = sum(len(s) for s in self.pcs.siblings)
        pcs_sc = sum(len(m) for m in self.pcs.sumcheck) * 3 * D * 4
        final = int(self.pcs.final_y.size) * 4
        ybar = sum(int(np.asarray(yb).size) * 4 for yb in self.pcs.ybar)
        total = len(self.to_bytes())
        return {"total": total, "sumcheck": len(self.sc_bytes), "pcs_rows": rows, "pcs_merkle": sib, "pcs_sumcheck": pcs_sc,
                "pcs_final": final, "pcs_ybar": ybar, "roots": 32 * len(self.pcs.roots),
                "header": total - rows - sib - pcs_sc - final - ybar - 32 * len(self.pcs.roots) - len(self.sc_bytes)}


def read_proof(buf: bytes) -> Proof:
    r = _R(buf)
    if r.take(8) != PROOF_MAGIC:
        raise ValueError("not an LGTO0001 proof")
    praw = r.blob()
    params = json.loads(praw.decode())
    framing = json.loads(r.blob().decode())
    sc = r.blob()
    kps, qs = params["kprime"], params["queries"]
    L = len(kps)
    if not (len(qs) == len(params["rate_log2"]) == len(params["splits"]) == len(framing["sib_len"]) == L):
        raise ValueError("inconsistent round counts")
    roots = [r.take(32) for _ in range(L)]
    msgs = [[r.arr("<u4", (3, D)) for _ in range(kp)] for kp in kps]
    rows, sibs = [], []
    for i in range(L):
        W = (1 << kps[i]) * (1 if i == 0 else D)
        rows.append(r.arr("<u4", (qs[i], W)))
        sibs.append(r.take(framing["sib_len"][i]))
    final_len = framing["final_len"]
    if final_len != 1 << (params["n"] - sum(kps)):
        raise ValueError("final vector length")
    fy = r.arr("<u4", (final_len, D))
    t_pad = int(framing.get("t_pad", 0))
    if t_pad != int(params.get("t_pad", 0)):
        raise ValueError("framing t_pad differs from the parameters'")
    ybar = [r.arr("<u4", (t_pad, D)) for _ in range(L)] if t_pad else []
    r.end()
    return Proof(params, sc, PcsProof(roots, msgs, rows, sibs, fy, ybar), praw)
