"""``leaf/ajtai``: a Ring-SIS (Ajtai / SWIFFT-style) row digest over BabyBear, linear in the committed operand bits --
the parametric family ``ajtai-n<N>`` of the leaf registry (``fp8-ada+ajtai-n64``, ``bf16-hopper+ajtai-n128``).

The Ligero chain applies ONE constraint system at every column, so a digest carried column to column must be a public
linear state transition ``acc_out = M acc_in + B s_j`` (``s_j``: the column's operand bits).  With ``M`` = multiplication
by X in ``R = F_p[X] / (X^n + 1)`` (a negacyclic shift) and ``B = [b_0 .. b_{m-1}]`` seeded ring elements, one per bit
position of a column, the VU's digest is

    h = sum_j X^(steps-1-j) B s_j = sum_i b_i(X) g_i(X),     g_i(X) = sum_j X^(steps-1-j) s_{j,i}   (deg < steps),

i.e. Ring-SIS with m = 256 public ring elements and short (binary, degree < steps) unknowns: a collision is a nonzero
ternary solution of ``sum_i b_i gamma_i = 0`` in R_p (Lyubashevsky-Micciancio-Peikert-Rosen 2008, SWIFFT).  A digest
carried as ``acc + A_blk s_j`` WITHOUT the shift would be ``A_blk sum_j s_j`` -- column-swap collisions -- which is why
the ring structure is not optional (lane note §1).  Injectivity needs ``n >= steps`` (with wrap-around ``X^(n+k) = -X^k``
makes ``s_a - s_b`` a coefficient and (1, 1) collides with (0, 0)); :meth:`AjtaiLeaf.gadget` refuses a relation with
more columns per VU than the ring degree.

**Binding only, NOT hiding** (red-team-leaf F5): the digest is LINEAR in the row's bits and unsalted (``r0``), so it
leaks row equality, sparse byte differences (``h(x) - h(x')`` is ``B`` applied to the difference: recoverable in
microseconds), and determines a low-entropy row outright.  The ZK rows hide the states inside the proof, not the digest
the statement publishes.  Use it where the committed rows are public or already known to the verifier, not as a hiding
commitment.  Roles: the x and W trees are separate SHA-256 commitment domains (the tree binding and the leaf framing
carry the tree name), so a cross-role collision opens nothing in the other tree; the digests are separated anyway
(conformance rule; red-team NIT) by a key per role: x rows use ``b_0 .. b_255``, W columns ``b_256 .. b_511`` (the next
256 elements of the same XOF family).  Within one tree the instance is exactly the one costed above (256 ring elements);
a cross-role collision ``B_x s = B_W s'`` is a Ring-SIS instance over the 512 elements, and harmless besides.  (A public
per-role constant would not do: the chain applies every Linear row at the pad columns too, whose witness is 0.)

Parameters (lane ajtai-design FINAL 17:10Z; red-team-leaf FINAL 18:00Z): ``n`` = ring degree = digest elements, a power
of two -- 64 for the 48-step relations (fp8-ada, fp4), 128 for the 96-step bf16-hopper (n = 96 is reducible and refused);
best known attack Wagner k-tree 2^289 (n = 64) / 2^571 (n = 128), lattice floor 2^283 / 2^557 classical.  ``beta = 1``
(bits), ``B = [b_0 .. b_{255} | b_256 .. b_511]`` (x key | W key), ``b_i`` = the SHAKE-256(b"verity/ajtai-babybear/B/v1"
|| LE32(i)) stream read as LE u32 words masked to 31 bits, values >= p rejected (acceptance 15/16; ajtai-design's
``derive_ring_element``), prefix-consistent in n.  ``leaf_bytes`` = n x LE u32.  ``params_sha256`` hashes the
descriptor AND the SHA-256 of the derived ``B`` (all 512 elements; F6: the Rust verifier re-derives ``B`` from the tag,
pins the same digest and checks every ``+ajtai`` system's key rows -- a system with another key is refused even under
``--allow-any-system``).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ..compile import P, Expr, Linear
from . import registry

__all__ = ["AjtaiLeaf", "B_DIGEST_SHA256", "B_TAG", "DEFAULT_N", "M_COL", "ROLE_W", "SUPPORTED_N", "chain_key", "derive_B", "derive_matrix",
           "derive_ring_element", "digest_rows", "key_offset", "matrix_sha256", "resolve", "shift_x"]

DEFAULT_N = 64
#: the ring degrees the family admits (a power of two; the Rust verifier knows exactly these)
SUPPORTED_N = (64, 128)
#: operand bits per chain column: k words x word_bits = 256 for every registered relation (32 x 8, 16 x 16)
M_COL = 256
#: the W role (``poseidon2_babybear.ROLE_W``): its rows use the second key block ``b_256 .. b_511``; x (1) and the
#: role-free chain of the tests (0) use the first
ROLE_W = 2
B_TAG = b"verity/ajtai-babybear/B/v1"
VERSION = "v1"
#: the SHA-256 of ``derive_matrix(n)`` (n x 512: the x key, then the W key) as row-major LE u32, n = 64 / 128 --
#: what ``params_sha256`` absorbs and the Rust verifier's ``leaf.rs`` re-derives (regression: a change in the derivation
#: changes every relation string's digest)
B_DIGEST_SHA256 = {
    64: "1f3df70fd5de615f8bf3ce77fc5515057e41d2561019036399b3b4b76a68ebca",
    128: "37eed28c0803e05c9a90335e0c011b7ba6c19717d4eb63434addaf1c8fcf92c3",
}


def derive_ring_element(i: int, n: int) -> np.ndarray:
    """``b_i`` (n coefficients): the SHAKE-256 XOF of ``B_TAG || LE32(i)`` read as LE u32 words ``u``, ``v = u & 0x7fffffff``,
    accepted iff ``v < p`` (acceptance 15/16, no modular bias); coefficient k = the k-th accepted value.  Byte-identical to
    lane ajtai-design's ``ajtai_params.derive_ring_element`` (their regression vector is in the tests).  Prefix-consistent
    in n (an XOF): ``ajtai-n64`` and ``ajtai-n128`` are one family."""
    h = hashlib.shake_256(B_TAG + int(i).to_bytes(4, "little"))
    stream_len = 4 * (int(n * 1.2) + 64)
    while True:
        words = np.frombuffer(h.digest(stream_len), dtype="<u4").astype(np.int64) & 0x7FFFFFFF
        words = words[words < P]
        if len(words) >= n:
            return np.ascontiguousarray(words[:n])
        stream_len *= 2


def derive_B(m: int, n: int) -> list[np.ndarray]:
    """The m ring elements of the leaf, ``[b_0 .. b_{m-1}]``, one per operand bit position of a column."""
    return [derive_ring_element(i, n) for i in range(m)]


_MATRICES: dict[tuple[int, int], np.ndarray] = {}


def derive_matrix(n: int, m: int = 2 * M_COL) -> np.ndarray:
    """``B`` as the (n, m) int64 matrix the gadget and the native side apply: ``B[x, i]`` = coefficient x of ``b_i``
    (``B[:, o:o + 256] @ s`` = ``sum_i s_i b_{o+i}`` in R_p, ``o`` = :func:`key_offset` of the role); the one derivation
    for every call site (memoised: 512 XOF streams)."""
    key = (int(n), int(m))
    if key not in _MATRICES:
        _MATRICES[key] = np.stack(derive_B(m, n), axis=1)
    return _MATRICES[key]


def matrix_sha256(B: np.ndarray) -> str:
    """SHA-256 of ``B`` as row-major LE u32 (what the Rust verifier re-derives and compares)."""
    return hashlib.sha256(np.ascontiguousarray(B, dtype=np.int64).astype("<u4").tobytes()).hexdigest()


def params_descriptor(n: int, m_col: int, salt_bits: int, b_sha256: str) -> bytes:
    """What ``params_sha256`` hashes: everything the verifier needs to rebuild the scheme, plus the digest of the key it
    rebuilt (the verifier regenerates B from the tag; it never takes B from a statement or a system file)."""
    return (f"ajtai-babybear/{VERSION}|p={P}|n={n}|m_col={m_col}|beta=1|salt_bits={salt_bits}|ring=F_p[X]/(X^n+1)|M=X"
            f"|B={B_TAG.decode()}|b_i=SHAKE-256(tag||LE32(i)) LE u32 &0x7fffffff reject>=p|keys=x:b_0..b_{m_col - 1},W:b_{m_col}..b_{2 * m_col - 1}|leaf=LE u32 x n"
            f"|B_sha256={b_sha256}").encode()


def shift_x(v):
    """Multiplication by X in F_p[X]/(X^n + 1) along the last axis: out[0] = -v[n-1], out[x] = v[x-1]."""
    if isinstance(v, np.ndarray):
        out = np.roll(v, 1, axis=-1)
        out[..., 0] = (-out[..., 0]) % P
        return out % P
    import torch
    out = torch.roll(v, 1, dims=-1)
    out[..., 0] = (-out[..., 0]) % P
    return out % P


def _bits_of(words, word_bits: int):
    """(..., k) words -> (..., k * word_bits) bits, word-major, little-endian within a word (compose's bit-row order)."""
    if isinstance(words, np.ndarray):
        sh = np.arange(word_bits, dtype=np.int64)
        b = (words[..., None] >> sh) & 1
        return b.reshape(*words.shape[:-1], words.shape[-1] * word_bits)
    import torch
    sh = torch.arange(word_bits, dtype=torch.int64, device=words.device)
    b = (words[..., None] >> sh) & 1
    return b.reshape(*words.shape[:-1], words.shape[-1] * word_bits)


def key_offset(role: int, m_col: int = M_COL) -> int:
    """The first column of ``role``'s key block in ``B``: ``m_col`` for W, 0 for x (and the role-free 0)."""
    return m_col if role == ROLE_W else 0


def _apply_B(bits, B, role: int = 0):
    """(..., m) bits (0/1) -> (..., n) = bits @ B[:, o:o + m]^T mod p (``o`` = the role's key block), exactly: float64
    products of a 0/1 by a < 2^31 value, at most m = 256 terms (< 2^39 < 2^53), reduced once.  Non-binary ``bits`` (the
    negatives) stay exact below 2^22."""
    m = bits.shape[-1]
    o = key_offset(role, m)
    if isinstance(bits, np.ndarray):
        acc = bits.astype(np.float64) @ B[:, o:o + m].astype(np.float64).T
        return np.rint(acc).astype(np.int64) % P
    import torch
    Bt = torch.as_tensor(B[:, o:o + m], dtype=torch.float64, device=bits.device).T
    return torch.round(bits.to(torch.float64) @ Bt).to(torch.int64) % P


def digest_rows(rows, word_bits: int, B: np.ndarray, m_col: int = M_COL, role: int = 0) -> tuple:
    """``rows`` (R, n_words) -> (acc_in (R, steps, n), acc_out (R, steps, n)): the carried state entering every column
    and leaving it; the digest is ``acc_out[:, -1]``.  numpy or torch, on the rows' device.  ``role`` selects the key
    block (W: the second; x and the role-free 0 of the tests: the first)."""
    k = m_col // word_bits
    if rows.shape[-1] % k:
        raise ValueError(f"a row of {rows.shape[-1]} words is not a whole number of {k}-word columns")
    steps = rows.shape[-1] // k
    R = rows.shape[0]
    n = B.shape[0]
    if R > 1024:
        # the (R, steps, 256) bit tensor of a committed set (4096 BF16 rows) is 0.8 GB int64 plus its float64 copy: chunk
        parts = [digest_rows(rows[r0:r0 + 1024], word_bits, B, m_col, role) for r0 in range(0, R, 1024)]
        cat = np.concatenate if isinstance(rows, np.ndarray) else __import__("torch").cat
        return cat([p[0] for p in parts]), cat([p[1] for p in parts])
    bits = _bits_of(rows.reshape(R, steps, k), word_bits)                   # (R, steps, m)
    contrib = _apply_B(bits, B, role)                                        # (R, steps, n)
    if isinstance(rows, np.ndarray):
        acc_in = np.zeros((R, steps, n), dtype=np.int64)
        acc_out = np.zeros((R, steps, n), dtype=np.int64)
        acc = np.zeros((R, n), dtype=np.int64)
    else:
        import torch
        acc_in = torch.zeros((R, steps, n), dtype=torch.int64, device=rows.device)
        acc_out = torch.zeros((R, steps, n), dtype=torch.int64, device=rows.device)
        acc = torch.zeros((R, n), dtype=torch.int64, device=rows.device)
    for j in range(steps):
        acc_in[:, j] = acc
        acc = (shift_x(acc) + contrib[:, j]) % P
        acc_out[:, j] = acc
    return acc_in, acc_out


def chain_key(sys, n: int) -> tuple[list[int], list[int]]:
    """Red-team-leaf-2 G1, the Python twin of the Rust ``leaf.rs::chain_key``: follow the public digest lanes
    ``hash.d[x]`` of a compiled hashed system to the key rows the chain ACTUALLY uses and check every coefficient against
    ``B`` re-derived from :data:`B_TAG` (never the scheme object's key).  Per lane: the one linear row ``W[pin] = pub``,
    the one quadratic on that row (``is_end * (out - d) = 0``), the one linear row on ``out``, which must be
    ``out - (X acc_in)[x mod n] - sum_i B[x mod n][o + i] bit_i`` over the chain row the neighbour lane's ``out`` is linked
    into; the bit rows one per position, the same in all n key rows of a role, increasing, boolean, disjoint across roles
    and from the chain / pin rows.  Returns the (x, W) bit rows; raises ``ValueError`` naming the first violation."""
    B = derive_matrix(n)
    if matrix_sha256(B) != B_DIGEST_SHA256.get(n):
        raise ValueError(f"ajtai-n{n}: the re-derived key does not match the pinned digest")
    nd = 2 * n
    single = lambda e: next(iter(e.lin)) if (e.c % P == 0 and len(e.lin) == 1 and next(iter(e.lin.values())) % P == 1) else None  # noqa: E731
    lin_of = lambda e: {r: v % P for r, v in e.lin.items() if v % P}  # noqa: E731
    end_row, d_row = None, [None] * nd
    pin_row = {name: idx for idx, name in sys.pins}
    carried: dict[str, list] = {}
    for c in sys.linear:
        if c.pub is not None:
            lin = {r: v % P for r, v in c.lin.items() if v % P}
            carried.setdefault(c.pub, []).append(next(iter(lin)) if (c.c % P == 0 and len(lin) == 1 and next(iter(lin.values())) == 1
                                                                      and next(iter(lin)) == pin_row.get(c.pub)) else None)

    def carried_row(name: str) -> int:
        rows = carried.get(name, [])
        if name not in pin_row or len(rows) != 1 or rows[0] is None:
            raise ValueError(f"{name} is not pinned through exactly one constraint W[pin row] = pub")
        return rows[0]

    end_row = carried_row("hash.is_end")
    d_row = [carried_row(f"hash.d[{x}]") for x in range(nd)]
    extra = [nm for _, nm in sys.pins if nm.startswith("hash.d[") and nm not in {f"hash.d[{x}]" for x in range(nd)}]
    if extra:
        raise ValueError(f"pin {extra[0]}: not one of the {nd} digest lanes")
    lane_of = {r: x for x, r in enumerate(d_row)}
    if len(lane_of) != nd or end_row in lane_of:
        raise ValueError("two public lanes share a pin row")
    out: list = [None] * nd
    for q in sys.quadratic:
        a, b, cc = lin_of(q.a), lin_of(q.b), lin_of(q.c)
        lanes = [lane_of[r] for e in (a, b, cc) for r in e if r in lane_of]
        if not lanes:
            continue
        if len(lanes) != 1:
            raise ValueError("a quadratic constraint touches two digest lanes")
        x = lanes[0]
        o = [r for r, v in b.items() if r != d_row[x] and v == 1]
        tie = (single(q.a) == end_row and q.c.c % P == 0 and not cc and q.b.c % P == 0 and len(b) == 2 and b.get(d_row[x]) == P - 1
               and len(o) == 1)
        if not tie or out[x] is not None:
            raise ValueError(f"hash.d[{x}]: not tied by exactly one is_end * (out - d) = 0")
        out[x] = o[0]
    if any(o is None for o in out):
        raise ValueError(f"hash.d[{out.index(None)}] is tied to no chain row")
    out_set = set(out)
    if len(out_set) != nd or end_row in out_set or out_set & set(lane_of):
        raise ValueError("two digest lanes are tied to one row, or to a pin row")
    acc_of: dict[int, int] = {}
    for c, y in zip(sys.chain["c"], sys.chain["y"]):
        o = single(y)
        if o in out_set:
            if o in acc_of:
                raise ValueError(f"chain row {o} is linked twice")
            acc_of[o] = c
    if any(o not in acc_of for o in out):
        raise ValueError("a chain row tied to a digest lane is never linked")
    acc = [acc_of[o] for o in out]
    acc_set = set(acc)
    if len(acc_set) != nd or acc_set & (out_set | set(lane_of) | {end_row}):
        raise ValueError("the carried rows of two lanes coincide, or a carried row is a digest / pin row")
    defs: dict[int, list[int]] = {}
    for qi, c in enumerate(sys.linear):
        for r in c.lin:
            if r in out_set and c.lin[r] % P:
                defs.setdefault(r, []).append(qi)
    boolean = set()
    for q in sys.quadratic:
        r = single(q.a)
        if r is not None and single(q.b) == r and single(q.c) == r:
            boolean.add(r)
    roles, all_bits = [], set()
    for k in range(2):
        bit_row: list = [None] * M_COL
        for xx in range(n):
            x = k * n + xx
            o = out[x]
            prev, prev_coef = (acc[k * n + n - 1], 1) if xx == 0 else (acc[x - 1], P - 1)
            qs = defs.get(o, [])
            if len(qs) != 1:
                raise ValueError(f"hash.d[{x}]: its chain row is not defined by exactly one linear constraint")
            c = sys.linear[qs[0]]
            if c.pub is not None or c.c % P:
                raise ValueError(f"hash.d[{x}]: the key row has a constant or a public right-hand side")
            blk = B[xx, k * M_COL:(k + 1) * M_COL]
            pos = {(P - int(v)) % P: i for i, v in enumerate(blk) if int(v)}
            if len(pos) != int((blk != 0).sum()):
                raise ValueError(f"row {xx} of the derived B repeats a coefficient")
            seen_o = seen_prev = False
            nbits = 0
            for r, v in lin_of(c).items():
                if r == o and v == 1:
                    seen_o = True
                elif r == prev and v == prev_coef:
                    seen_prev = True
                else:
                    i = pos.get(v)
                    if i is None:
                        raise ValueError(f"hash.d[{x}]: row {r} has coefficient {v}, not an entry of the derived key row")
                    if bit_row[i] is None:
                        bit_row[i] = r
                    elif bit_row[i] != r:
                        raise ValueError(f"key role {k}: bit position {i} is row {bit_row[i]} in one key row and {r} in another")
                    nbits += 1
            if not (seen_o and seen_prev and nbits == len(pos)):
                raise ValueError(f"hash.d[{x}]: the key row is not out - (X acc_in)[{xx}] - B[{xx}] bits over the linked neighbour")
        if any(r is None for r in bit_row):
            raise ValueError(f"key role {k}: bit position {bit_row.index(None)} is never keyed")
        if any(bit_row[i] >= bit_row[i + 1] for i in range(M_COL - 1)):
            raise ValueError(f"key role {k}: the bit rows are not in bit-position order")
        chain_rows = set(sys.chain["c"])
        for r in bit_row:
            if r not in boolean:
                raise ValueError(f"key role {k}: bit row {r} is not boolean-constrained")
            if r in all_bits or r in out_set or r in acc_set or r in lane_of or r == end_row or r in chain_rows:
                raise ValueError(f"key role {k}: bit row {r} is shared with the other role or is a chain / pin row")
            all_bits.add(r)
        roles.append(bit_row)
    return roles[0], roles[1]


def _single_row(e: Expr, what: str) -> int:
    if e.c != 0 or len(e.lin) != 1 or next(iter(e.lin.values())) != 1:
        raise ValueError(f"ajtai: {what} must be a single committed row, got {e!r}")
    return next(iter(e.lin))


def _bit_rows_of_word(w: Expr, word_bits: int) -> list[int]:
    """The committed bit rows behind an affine word expression ``sum_j 2^j b_j`` (what ``hashchain.compose`` builds from
    the boolean rows it committed): beta = 1 means the scheme's inputs ARE those bits; nothing is re-decomposed."""
    if w.c != 0 or len(w.lin) != word_bits or sorted(int(v) for v in w.lin.values()) != [2**j for j in range(word_bits)]:
        raise ValueError(f"ajtai: an operand word must be the affine form sum_j 2^j b_j of {word_bits} committed bit rows, got {w!r}")
    by_weight = {int(v): r for r, v in w.lin.items()}
    return [by_weight[2**j] for j in range(word_bits)]


class AjtaiLeaf:
    """The scheme object (:class:`leaf.base.LeafScheme`): registry ``name`` "ajtai", relation suffix ``tag`` "ajtai-n<N>".
    ``m_col`` = operand bits per chain column (k words x word_bits; 256 for every registered relation), fixed at
    construction so ``B`` is one matrix per (n, m_col)."""

    #: the registry family prefix; an instance's ``name`` is its member name ``ajtai-n<N>`` (= the relation suffix)
    family = "ajtai"
    # red-team-leaf FINAL: the Table 1 assumption line
    assumption = ("Ring-SIS collision resistance over BabyBear[X]/(X^n+1), beta = 1 bit inputs; best known attack Wagner k-tree "
                  "2^289 (n = 64) / 2^571 (n = 128); lattice floor 2^283 / 2^557 classical")
    privacy_note = ("linear, unsalted (r0): binding-only, leaks equality and linear relations between rows -- a row digest is "
                    "not a hiding commitment")

    def __init__(self, n: int = DEFAULT_N, m_col: int = M_COL, salt_bits: int = 0):
        if n <= 0 or m_col <= 0:
            raise ValueError("n and m_col must be positive")
        if n & (n - 1):
            raise ValueError(f"the ring degree n = {n} must be a power of two (X^n + 1 irreducible; ajtai-design §8)")
        if salt_bits:
            raise NotImplementedError("the salted (hiding) variant is not built: v1 is the unsalted binding-only leaf (r0)")
        self.n, self.m_col, self.salt_bits = int(n), int(m_col), int(salt_bits)
        self.digest_elems = self.n
        self.carry_elems = self.n
        self.name = self.tag = f"ajtai-n{self.n}"
        # `r0`: unsalted -- binding only, NOT hiding (the digest is linear in the row's bits; ajtai-design §3 / §8)
        self.hash_name = f"ajtai-babybear-n{self.n}-b1-r{self.salt_bits}/{VERSION}"
        self.schema = f"ajtai-babybear-n{self.n}-b1-r{self.salt_bits}/row/{VERSION}"
        self._B = None
        b_sha = B_DIGEST_SHA256.get(self.n) if self.m_col == M_COL else None
        if b_sha is None:
            b_sha = matrix_sha256(self.B)
        elif __debug__ and self.n in SUPPORTED_N and matrix_sha256(self.B) != b_sha:
            raise AssertionError(f"ajtai-n{self.n}: the derived B does not match B_DIGEST_SHA256 (derivation changed?)")
        self.params_sha256 = hashlib.sha256(params_descriptor(self.n, self.m_col, self.salt_bits, b_sha)).hexdigest()

    def __repr__(self):
        return f"AjtaiLeaf(n={self.n}, m_col={self.m_col})"

    @property
    def B(self) -> np.ndarray:
        """(n, 2 m_col): the x key block, then the W key block."""
        if self._B is None:
            self._B = derive_matrix(self.n, 2 * self.m_col)
        return self._B

    # --- committer (native) ---------------------------------------------------------------------------------------------
    def native(self, rows, *, word_bits: int, role: int = 0):
        """(R, n) digests of ``rows`` (R, n_words) -- numpy in, numpy out; torch in, torch out (same device).  ``role``
        (x: 1, W: 2) selects the key block (module docstring)."""
        if isinstance(rows, np.ndarray):
            rows = np.ascontiguousarray(rows, dtype=np.int64)
            if rows.ndim != 2 or ((rows < 0) | (rows >= 2**word_bits)).any():
                raise ValueError(f"ajtai: rows must be (R, n_words) words of {word_bits} bits")
            return digest_rows(rows, word_bits, self.B, self.m_col, role)[1][:, -1]     # chunked over rows inside
        _, acc_out = digest_rows(rows, word_bits, self.B, self.m_col, role)
        return acc_out[:, -1]

    def step(self, acc, bits, role: int):
        """One column of the state transition on arbitrary (not necessarily binary) bit values: ``X acc + B_role bits`` mod
        p.  ``acc`` (..., n), ``bits`` (..., m_col) int64 numpy or torch; exact for bit values < 2^22."""
        return (shift_x(acc) + _apply_B(bits, self.B, role)) % P

    def leaf_bytes(self, digest_row) -> bytes:
        """The n elements as little-endian u32 (4n bytes; the design note's framing)."""
        d = np.asarray([int(x) for x in digest_row] if not isinstance(digest_row, np.ndarray) else digest_row, dtype=np.int64).ravel()
        if d.shape != (self.n,) or ((d < 0) | (d >= P)).any():
            raise ValueError(f"an Ajtai row digest is {self.n} BabyBear elements")
        return d.astype("<u4").tobytes()

    # --- gadget -----------------------------------------------------------------------------------------------------------
    @staticmethod
    def _out_name(role: int, x: int) -> str:
        return f"ajtai.r{role}.out[{x}]"

    def gadget(self, ctx, words: Sequence[Expr], *, word_bits: int, role: int, row_words: int, carry_in: Sequence[Expr],
               first: bool = False, last: bool = True) -> tuple[list[Expr], list[Expr]]:
        """One chain column: ``carry_in`` = the n linked ``acc_in`` rows (the chain-start constraint makes them 0 at a VU's
        first column); the column's ``m_col`` operand bits are the committed boolean rows behind ``words`` (word-major,
        LSB first: the order of ``derive_B``).  Appends n hint rows ``acc_out`` (values from :meth:`chain_witness`) and n
        Linear constraints ``acc_out[x] - (X acc_in)[x] - sum_i B[x, o + i] bit_i = 0`` (``o`` = the role's key block);
        returns (digest exprs = acc_out, carry_out = acc_out).  Refuses a relation whose VU has more columns than the ring degree (injectivity)."""
        if len(carry_in) != self.n:
            raise ValueError(f"the Ajtai gadget carries {self.n} linked rows, got {len(carry_in)}")
        k = len(words)
        if k * word_bits != self.m_col:
            raise ValueError(f"ajtai: {k} words x {word_bits} bits per column, the leaf hashes {self.m_col}")
        steps = row_words // k
        if steps * k != row_words or steps > self.n:
            raise ValueError(f"ajtai-n{self.n}: a VU of {steps} columns needs a ring degree >= {steps} (n >= steps: "
                             f"injectivity of the Ring-SIS accumulation); use ajtai-n{1 << (steps - 1).bit_length()}")
        bit_rows = [r for w in words for r in _bit_rows_of_word(w, word_bits)]
        cin = [_single_row(a, "the carried state") for a in carry_in]
        B = self.B
        off = key_offset(role, self.m_col)
        out = []
        for x in range(self.n):
            o = ctx.hint(self._out_name(role, x))
            lin = {next(iter(o.lin)): 1}
            if x == 0:
                lin[cin[self.n - 1]] = 1                     # -(-acc_in[n-1])
            else:
                lin[cin[x - 1]] = P - 1
            for i, r in enumerate(bit_rows):
                v = int(B[x, off + i])
                if v:
                    lin[r] = (lin.get(r, 0) + P - v) % P
            # a field identity by construction (the acc rows have no integer interval): appended without the audit
            ctx.sys.linear.append(Linear(f"hash.{self._out_name(role, x)}", {kk: v for kk, v in lin.items() if v}, 0, None))
            out.append(o)
        return out, out

    def witness(self):
        return None

    def check_system(self, sys) -> None:
        """The verifier's key check on a compiled hashed system (:func:`chain_key`; raises ``ValueError``)."""
        chain_key(sys, self.n)

    # --- chain witness --------------------------------------------------------------------------------------------------
    def chain_witness(self, words, *, word_bits: int, role: int, row_words: int, n_vus: int, steps: int, l: int):
        """``words`` (l, k) int64 torch (pad columns 0; VU v at columns [v steps, (v + 1) steps)) -> (carry_in (n, l), digests
        (n_vus, n), {acc_out hint name: (l,)}) on the words' device.  Vectorised over the VUs: one (n_vus, n) shift-add per
        column."""
        import torch
        k = words.shape[1]
        dev = words.device
        cols = words[:n_vus * steps].reshape(n_vus, steps * k)                       # VU rows
        acc_in, acc_out = digest_rows(cols, word_bits, self.B, self.m_col, role)      # (n_vus, steps, n) each
        carry = torch.zeros((self.n, l), dtype=torch.int64, device=dev)
        carry[:, :n_vus * steps] = acc_in.reshape(n_vus * steps, self.n).T
        outs = torch.zeros((self.n, l), dtype=torch.int64, device=dev)
        outs[:, :n_vus * steps] = acc_out.reshape(n_vus * steps, self.n).T
        extras = {self._out_name(role, x): outs[x] for x in range(self.n)}
        return carry, acc_out[:, -1].contiguous(), extras

    def census(self) -> dict:
        return {"ajtai_n": self.n, "ajtai_m_col": self.m_col, "ajtai_linear_rows_per_unit": 2 * self.n,
                "ajtai_terms_per_linear_row": 2 + self.m_col, "ajtai_key_blocks": 2}


def resolve(name: str) -> AjtaiLeaf | None:
    """The registry family resolver: ``"ajtai"`` -> n = 64, ``"ajtai-n<N>"`` -> n = N when N is a power of two in [8, 512],
    else ``None`` (``registry.get`` then reports an unknown scheme).  The Rust verifier knows :data:`SUPPORTED_N` only."""
    if name == "ajtai":
        return AjtaiLeaf(DEFAULT_N)
    if not name.startswith("ajtai-n"):
        return None
    try:
        n = int(name[len("ajtai-n"):])
    except ValueError:
        return None
    if n < 8 or n > 512 or n & (n - 1):
        return None
    return AjtaiLeaf(n)


registry.register_family("ajtai", resolve)
