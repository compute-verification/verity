"""A read at a private index of a registered table: the circuit opens the entry's row against the table's registered
``frame-v3-sha512`` root, with the index's bits choosing the side at every level (Daniel, 1 Oct: a table a program brings,
``gelu_tanh_bf16`` and ``tanh_rn`` included, is a value, committed and always hidden; the measured MUFU tables stay gates).

**The registered tree** is a registered value's (``verity_sampled_proofs.one_stage.registered``, verity PR #730), byte for
byte. A table of 2^n entries is registered as the u16 words of its entries, an entry's low word first, in rows of
:data:`ROW_WORDS`: ``gelu_tanh_bf16`` one word per entry (1,024 rows), ``tanh_rn`` two (each u32 entry, 2^22 rows). Row ``r``'s
inner digest is ``x = SHA-512(sha512_row_prefix(ROLE_X, 16, 64) || row)``, its commit string ``b || c =
hm96.SHA512.commit_string(default_key, x, y)`` under the row's own 192-byte salt ``y``, and its leaf ``D.leaf(r, r,
"hm96-sha512/row/v1", hm96.SHA512.tree_leaf(default_key, b || c))``; the nodes are ``D.node(l, i, left, right)``
(``verity.commitments.merkle``), under the domain :func:`domain` ``D``. :class:`Tree` is the prover's side.

**The read** (:func:`gather`): the public inputs are the table's root and its domain id (64 bytes each: constants, or wires a
statement makes public), the private ones the index, the row, its salt and the path (a sibling per level). The circuit hashes
the row to ``b || c`` and its tree leaf (six compressions), that to the frame leaf at the position ``r = index >>
log2(entries per row)`` (two), climbs a node per level (a conditional swap by the position's bit, then two compressions after
a first block that only the domain and the level fix: constant for a constant domain, else computed once per table and level
in a circuit), and picks the entry from the row by the index's low bits (a mux tree). :func:`read` asserts the root; a
statement whose padding instances are all-zero takes :func:`gather`'s root as a public output instead.

**Frame-v3's integers have a variable length** (``merkle._uint``: big-endian, minimal, at least one byte). The leaf's rank and
position, and a node's index ``r >> (l + 1)``, take 1 byte below 2^8, 2 below 2^16 and 3 below 2^24, and every later byte of
the message moves with that length. So at a private position each hash has one message layout per possible length; the
circuit builds them all and multiplexes them byte by byte, by whether the integer reaches 2^8 and 2^16 (an OR of its high
bits). Only bytes that differ between layouts cost an AND, about 2,000 per extra layout, against 2 x 81k for the compressions.
A message never changes its block count (the leaf is 227 + 2L bytes, two blocks; a node 274 + L, three). Over a ``c2`` node
(SHA-512 of the two digests, #726), frame-v3 costs the frame leaf's two compressions once per read, a node's second block
being message where c2's is constant padding, and the multiplexers on the levels whose index can reach 2^8.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from verity.commitments import hm96
from verity.commitments.identity import identity_digest
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, _message, _uint, merkle_depth, tree_levels
from verity.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, row_bytes, row_leaf, sha512_row_digest, sha512_row_prefix

from . import sha512_circuit as S
from .gf2 import Circuit, L, ZERO, const_bits, or_reduce

H5 = hm96.SHA512
KEY = H5.default_key
SCHEMA = SCHEMA_HM96_SHA512_ROW
#: #730's registered values: the domain tag, the owner, the row width (u16 words) and its word size
REGISTERED_DOMAIN = "verity/one-stage/registered-domain/v0"
OWNER = -2
ROW_WORDS, WORD_BITS = 64, 16
ROW_PREFIX = sha512_row_prefix(ROLE_X, WORD_BITS, ROW_WORDS)
_LEAF, _NODE = b"leaf", b"node"
_FILLER = b"verity_flock/hidden-gather/sparse-filler/v0\0"


@dataclass(frozen=True)
class Table:
    """A registered table's shape: 2^index_bits entries of ``entry_words`` u16 words, of which a read takes the low
    ``value_bits``, in rows of :data:`ROW_WORDS` words."""

    name: str
    index_bits: int
    entry_words: int
    value_bits: int

    @property
    def per_row(self) -> int:
        return ROW_WORDS // self.entry_words

    @property
    def entry_log(self) -> int:
        return self.per_row.bit_length() - 1

    @property
    def depth(self) -> int:
        return self.index_bits - self.entry_log

    @property
    def rows(self) -> int:
        return 1 << self.depth

    def __post_init__(self) -> None:
        if ROW_WORDS % self.entry_words or self.per_row & (self.per_row - 1) or not 0 < self.value_bits <= 16 * self.entry_words:
            raise ValueError(f"{self.name}: entries of {self.entry_words} words do not pack rows of {ROW_WORDS}")


#: the tables a program brings (``tail_pieces._DERIVED``'s gelu and RN32 tanh), registered values; never a MUFU table
TABLES = {t.name: t for t in (Table("gelu_tanh_bf16", 16, 1, 16), Table("tanh_rn", 27, 2, 31))}


# --- the registered tree (#730's), natively ------------------------------------------------------------------------------

def domain(program: str, value: str, leaves: int, words: int = ROW_WORDS) -> CommitmentDomain:
    """#730's ``registered.domain``: ``CommitmentDomain(B, -2, RangeIndexedDomain(0, leaves), hash="sha512")``, ``B`` the bytes of
    ``identity_digest(REGISTERED_DOMAIN, {program, value, schema, words, leaves})``; ``program`` is the Program's SHA-512 hex."""
    binding = identity_digest(REGISTERED_DOMAIN, {"program": program, "value": value, "schema": SCHEMA, "words": int(words),
                                                  "leaves": int(leaves)})
    return CommitmentDomain(bytes.fromhex(binding), OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


def table_rows(table: Table, entries: Sequence[int]) -> list[tuple[int, ...]]:
    """The registered rows of a table's entries (2^index_bits words of at most 16 * entry_words bits): each entry's u16 words,
    its low word first, :data:`ROW_WORDS` to a row."""
    e = np.asarray(entries, dtype=np.uint64)
    if len(e) != 1 << table.index_bits or (e >> np.uint64(16 * table.entry_words)).any():
        raise ValueError(f"{table.name}: {1 << table.index_bits} entries of {16 * table.entry_words} bits")
    words = np.stack([(e >> np.uint64(16 * k)) & np.uint64(0xFFFF) for k in range(table.entry_words)], axis=1).reshape(-1, ROW_WORDS)
    return [tuple(int(w) for w in r) for r in words]


def commit_string(row: Sequence[int], salt: bytes) -> bytes:
    """A row's ``b || c`` under its salt (#730's: ``hm96-sha512/v1`` over ``sha512/row/v1``)."""
    return H5.commit_string(KEY, sha512_row_digest(row, WORD_BITS, ROLE_X), salt)


def frame_leaf(dom: CommitmentDomain, position: int, bc: bytes) -> bytes:
    """The frame-v3-sha512 leaf of the row at ``position`` (#730's ``registered.leaf``)."""
    return row_leaf(dom, position, H5.tree_leaf(KEY, bc), SCHEMA)


def verify(dom: CommitmentDomain, root: bytes, position: int, bc: bytes, path: Sequence[bytes]) -> bool:
    """``bc``, as the row at ``position``, opens ``root`` under ``dom`` through ``path`` (siblings, leaf level first): #730's
    ``verify_read``, the check the circuit makes."""
    depth = merkle_depth(dom.count)
    if not 0 <= position < dom.count or len(bc) != H5.proven_bytes or len(path) != depth or any(len(s) != 64 for s in path):
        return False
    digest, i = frame_leaf(dom, position, bc), position
    for level, sib in enumerate(path):
        digest = dom.node(level, i >> 1, *((digest, sib) if i % 2 == 0 else (sib, digest)))
        i >>= 1
    return digest == root


@dataclass(frozen=True)
class Tree:
    """The prover's side of a registered table: its rows and salts (private), and the frame-v3-sha512 levels over them, leaf
    level first, as ``{index: digest}`` per level.

    :meth:`commit` is #730's ``Registered.commit`` (every row). :meth:`sparse` commits only some rows: every subtree without one
    is a pseudo-random digest of its level and index (``filler``), so its root commits to those rows and to nothing else; a
    statement over it costs what a read of the real table costs, for a table whose every row a test cannot hash (2^22 for
    ``tanh_rn``)."""

    table: Table
    domain: CommitmentDomain
    rows: Mapping[int, tuple[int, ...]]
    salts: Mapping[int, bytes]
    levels: tuple[Mapping[int, bytes], ...]
    filler: bytes | None = None

    @classmethod
    def commit(cls, table: Table, dom: CommitmentDomain, rows: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None) -> "Tree":
        """Every row of the table; ``salts`` one per row, default fresh from the operating system (never reused)."""
        rows = [tuple(int(w) for w in r) for r in rows]
        if len(rows) != table.rows or dom.count != table.rows or any(len(r) != ROW_WORDS for r in rows):
            raise ValueError(f"{table.name}: {table.rows} rows of {ROW_WORDS} words under a domain of as many")
        salts = [bytes(s) for s in salts] if salts is not None else [H5.fresh_salt() for _ in rows]
        if len(salts) != len(rows) or len(set(salts)) != len(salts):
            raise ValueError("one salt per row, never reused")
        leaves = [frame_leaf(dom, i, commit_string(r, s)) for i, (r, s) in enumerate(zip(rows, salts))]
        levels = tuple(dict(enumerate(level)) for level in tree_levels(dom, leaves))
        return cls(table, dom, dict(enumerate(rows)), dict(enumerate(salts)), levels)

    @classmethod
    def sparse(cls, table: Table, dom: CommitmentDomain, rows: Mapping[int, Sequence[int]], seed: bytes,
               salts: Mapping[int, bytes] | None = None) -> "Tree":
        """Only ``rows`` (``{position: row}``) committed; ``seed`` names the filler."""
        rows = {int(p): tuple(int(w) for w in r) for p, r in rows.items()}
        if not rows or dom.count != table.rows or any(not 0 <= p < table.rows or len(r) != ROW_WORDS for p, r in rows.items()):
            raise ValueError(f"{table.name}: rows at positions below {table.rows}, of {ROW_WORDS} words")
        salts = {p: bytes(salts[p]) for p in rows} if salts is not None else {p: H5.fresh_salt() for p in rows}
        tree = cls(table, dom, rows, salts, (), bytes(seed))
        level = {p: frame_leaf(dom, p, commit_string(r, salts[p])) for p, r in sorted(rows.items())}
        levels = [level]
        for lv in range(table.depth):
            up = {}
            for i in sorted({p >> 1 for p in level}):
                up[i] = dom.node(lv, i, level.get(2 * i) or tree._filler(lv, 2 * i), level.get(2 * i + 1) or tree._filler(lv, 2 * i + 1))
            levels.append(up)
            level = up
        object.__setattr__(tree, "levels", tuple(levels))
        return tree

    def _filler(self, level: int, index: int) -> bytes:
        return hashlib.sha512(_FILLER + self.filler + level.to_bytes(1, "big") + index.to_bytes(8, "big")).digest()

    def node(self, level: int, index: int) -> bytes:
        got = self.levels[level].get(index)
        if got is None:
            if self.filler is None:
                raise KeyError((level, index))
            return self._filler(level, index)
        return got

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, position: int) -> list[bytes]:
        return [self.node(lv, (position >> lv) ^ 1) for lv in range(self.table.depth)]

    def bc(self, position: int) -> bytes:
        return commit_string(self.rows[position], self.salts[position])

    def entry(self, index: int) -> int:
        """The table's entry at ``index``, from its row: the value a read returns."""
        t = self.table
        row, k = self.rows[index >> t.entry_log], index & (t.per_row - 1)
        words = row[k * t.entry_words:(k + 1) * t.entry_words]
        return sum(w << (16 * j) for j, w in enumerate(words)) & ((1 << t.value_bits) - 1)


# --- circuit pieces ----------------------------------------------------------------------------------------------------

def _const(data: bytes) -> list[L]:
    """Constant bits in byte-string order (bit ``t & 7`` of byte ``t >> 3``, hm96's order)."""
    return [b for byte in data for b in const_bits(byte, 8)]


def _block(bits: Sequence[L]) -> list[list[L]]:
    """1,024 byte-string-order bits as SHA-512's 16 message words (LSB-first bit lists; a word's bytes big-endian)."""
    w: list[list] = [[None] * 64 for _ in range(16)]
    for t, b in enumerate(bits):
        p = S.word_bit(t)
        w[p >> 6][p & 63] = b
    return w


def _digest(words: Sequence[Sequence[L]]) -> list[L]:
    """A chaining value's 8 words as its 512 digest bits in byte-string order."""
    flat = [b for w in words for b in w]
    return [flat[S.word_bit(t)] for t in range(512)]


def _cv(words: Sequence[int]) -> list[list[L]]:
    return [const_bits(v, 64) for v in words]


def compress(C: Circuit, cv: Sequence[Sequence[L]], block: Sequence[L], share: bool = False) -> list[list[L]]:
    """One SHA-512 compression of a 1,024-bit block (byte-string order). ``share``: an identical (cv, block) compressed before
    in this circuit is reused (a node's first block, which only the domain and the level fix)."""
    if not share:
        return S.compress_gadget(C, [list(w) for w in cv], _block(block))
    cache = C.__dict__.setdefault("_hidden_gather_cv", {})
    key = (tuple(b.key() for w in cv for b in w), tuple(b.key() for b in block))
    if key not in cache:
        cache[key] = S.compress_gadget(C, [list(w) for w in cv], _block(block))
    return cache[key]


def _sha512(C: Circuit, cv: Sequence[Sequence[L]], msg: Sequence[L], share_first: bool = False) -> list[list[L]]:
    assert len(msg) % 1024 == 0
    for k in range(0, len(msg), 1024):
        cv = compress(C, cv, msg[k:k + 1024], share=share_first and k == 0)
    return cv


def _be(bits: Sequence[L], length: int) -> list[L]:
    """The low ``8 length`` bits of the integer whose bits (LSB first) are ``bits``, as ``length`` big-endian bytes (byte-string
    order bits): the layout of that length, which the multiplexer takes only when the bits above are zero."""
    padded = (list(bits) + [ZERO] * (8 * length))[:8 * length]
    return [b for k in range(length) for b in padded[8 * (length - 1 - k):8 * (length - k)]]


def _framed(C: Circuit, tag: bytes, parts: Sequence, ubits: Sequence[L], share_first: bool = False) -> list[L]:
    """``SHA-512(merkle._message(tag, *parts))``'s 512 bits, ``ubits`` (LSB first) the one private integer that the parts marked
    ``None`` write with ``merkle._uint``'s length; every other part is constant bytes or bits in byte-string order."""
    lengths = range(1, max(1, -(-len(ubits) // 8)) + 1)
    layouts = []
    for n in lengths:
        sizes = [n if p is None else len(p) if isinstance(p, bytes) else len(p) // 8 for p in parts]
        skeleton = _message(tag, *(bytes(s) for s in sizes))
        bits = _const(skeleton + S.pad_tail(len(skeleton)))
        at = len(_message(tag))
        for p, s in zip(parts, sizes):
            at += 8
            if p is not None and isinstance(p, bytes):
                bits[8 * at:8 * (at + s)] = _const(p)
            else:
                bits[8 * at:8 * (at + s)] = _be(ubits, n) if p is None else list(p)
            at += s
        assert at == len(skeleton)
        layouts.append(bits)
    if len({len(b) for b in layouts}) != 1:
        raise ValueError("a message changes its block count with the integer's length")
    msg = layouts[0]
    for n, alt in zip(lengths[1:], layouts[1:]):
        s = or_reduce(C, ubits[8 * (n - 1):])
        msg = [x ^ C.AND(s, y ^ x) for x, y in zip(msg, alt)]
    return _digest(_sha512(C, _cv(S.IV), msg, share_first))


def frame_leaf_gadget(C: Circuit, dom_id: Sequence[L], pos: Sequence[L], value: Sequence[L]) -> list[L]:
    """``D.leaf(r, r, SCHEMA, value)`` at the private position ``r`` (bits LSB first), ``dom_id`` the domain id's 512 bits."""
    return _framed(C, _LEAF, [list(dom_id), None, None, SCHEMA.encode(), list(value)], pos)


def node_gadget(C: Circuit, dom_id: Sequence[L], level: int, index: Sequence[L], left: Sequence[L], right: Sequence[L]) -> list[L]:
    """``D.node(level, i, left, right)`` at the private index ``i`` (bits LSB first)."""
    return _framed(C, _NODE, [list(dom_id), _uint(level), None, list(left), list(right)], index, share_first=True)


def hm96_leaf(C: Circuit, row: Sequence[L], salt: Sequence[L]) -> list[L]:
    """``hm96.SHA512.tree_leaf(key, b || c)`` of a one-block row (1,024 bits) and its 192-byte salt (1,536 bits), both in
    byte-string order: six compressions (#726's ``leaf_hm96``)."""
    x = _sha512(C, _cv(S.midstate(ROW_PREFIX)), list(row) + _const(S.pad_tail(len(ROW_PREFIX) + 2 * ROW_WORDS)))
    c = _sha512(C, _cv(S.midstate(H5.salt_prefix)), list(salt) + _const(S.pad_tail(len(H5.salt_prefix) + H5.salt_bytes)))
    xs, rows = _digest(x), H5.rows_of(KEY)
    b = []
    for i in range(512):
        acc, r = xs[i], rows[i]
        while r:
            j = (r & -r).bit_length() - 1
            acc = acc ^ salt[j]
            r &= r - 1
        b.append(acc)
    bc = b + _digest(c)
    return _digest(_sha512(C, _cv(S.midstate(H5.leaf_prefix(KEY))), bc + _const(S.pad_tail(128 + H5.proven_bytes))))


def swap(C: Circuit, s: L, x: Sequence[L], y: Sequence[L]) -> tuple[list[L], list[L]]:
    """``(x, y)`` if s is 0, else ``(y, x)``: one AND per bit."""
    d = [C.AND(s, a ^ b) for a, b in zip(x, y)]
    return [a ^ e for a, e in zip(x, d)], [b ^ e for b, e in zip(y, d)]


def mux_tree(C: Circuit, idx: Sequence[L], entries: Sequence[Sequence[L]]) -> list[L]:
    """``entries[idx]`` (idx LSB first): one index bit per level, ``w (2^n - 1)`` ANDs."""
    level = [list(e) for e in entries]
    assert len(level) == 1 << len(idx)
    for b in idx:
        level = [[e0 ^ C.AND(b, e0 ^ e1) for e0, e1 in zip(level[2 * i], level[2 * i + 1])] for i in range(len(level) // 2)]
    return level[0]


def gather(C: Circuit, table: Table, idx: Sequence[L], row: Sequence[L], salt: Sequence[L], path: Sequence[Sequence[L]],
           dom_id: Sequence[L]) -> tuple[list[L], list[L]]:
    """The entry at the private ``idx`` (``table.index_bits``, LSB first) of the row ``row`` (1,024 bits, its u16 words LE)
    under ``salt`` (1,536 bits) and ``path`` (``table.depth`` siblings of 512 bits, leaf level first), all byte-string order:
    its ``value_bits`` bits, and the root the opening climbs to (512 bits) for the caller to bind."""
    if len(idx) != table.index_bits or len(row) != 16 * ROW_WORDS or len(salt) != 8 * H5.salt_bytes or len(dom_id) != 512:
        raise ValueError(f"{table.name}: a {table.index_bits}-bit index, a row of {ROW_WORDS} words, a salt and a domain id")
    if len(path) != table.depth or any(len(s) != 512 for s in path):
        raise ValueError(f"{table.name}: {table.depth} siblings of 64 bytes")
    pos = list(idx[table.entry_log:])
    cur = frame_leaf_gadget(C, dom_id, pos, hm96_leaf(C, row, salt))
    for lv in range(table.depth):
        left, right = swap(C, pos[lv], cur, path[lv])
        cur = node_gadget(C, dom_id, lv, pos[lv + 1:], left, right)
    w = 16 * table.entry_words
    value = mux_tree(C, list(idx[:table.entry_log]), [row[w * k:w * k + table.value_bits] for k in range(table.per_row)])
    return value, cur


# --- a circuit's registered tables -------------------------------------------------------------------------------------

@dataclass
class Binding:
    """A registered table in one circuit: its public inputs (root and domain id, 512 bits each in byte-string order) and its
    reads (each read's index forms, input prefix and value forms)."""

    table: Table
    root: list
    domain: list
    reads: list = field(default_factory=list)


def register(C: Circuit, name: str, root: bytes | None = None, domain_id: bytes | None = None) -> Binding:
    """Make ``name`` a registered table of ``C``: every ``tail_pieces.table_read`` of it in ``C`` is then a :func:`read`. The
    root and domain id are constants when given, else the public inputs ``registered/<name>/root`` and ``.../domain``."""
    if name not in TABLES:
        raise ValueError(f"{name!r} is no registered table ({sorted(TABLES)}); a measured MUFU table is read by a lookup")
    reg = C.__dict__.setdefault("registered", {})
    if name in reg:
        raise ValueError(f"{name!r} is registered in this circuit already")

    def bits(v, part):
        if v is None:
            return C.input(f"registered/{name}/{part}", 512)
        if len(v) != 64:
            raise ValueError(f"a {part} is 64 bytes")
        return _const(bytes(v))

    reg[name] = b = Binding(TABLES[name], bits(root, "root"), bits(domain_id, "domain"))
    return b


def read(C: Circuit, name: str, idx: Sequence[L]) -> list[L]:
    """The registered table ``name``'s entry at the private ``idx``: :func:`gather` over fresh private inputs
    ``registered/<name>/<k>/row``, ``.../salt`` and ``.../path<l>`` (read ``k`` of the table in ``C``), the root asserted."""
    b = C.registered[name]
    pre = f"registered/{name}/{len(b.reads)}/"
    row, salt = C.input(pre + "row", 16 * ROW_WORDS), C.input(pre + "salt", 8 * H5.salt_bytes)
    path = [C.input(f"{pre}path{lv}", 512) for lv in range(b.table.depth)]
    value, root = gather(C, b.table, list(idx), row, salt, path, b.domain)
    for t, (x, y) in enumerate(zip(root, b.root)):
        C.assert_zero(f"{pre}root[{t}]", x ^ y)
    b.reads.append((list(idx), pre, value))
    return value


def slice_bytes(per_lane: Sequence[bytes]) -> list[int]:
    """Lane-major byte strings (equal lengths) as bit-sliced ints in byte-string order: bit ``lane`` of ``out[t]`` is bit
    ``t & 7`` of byte ``t >> 3`` of ``per_lane[lane]``."""
    a = np.unpackbits(np.frombuffer(b"".join(per_lane), dtype=np.uint8).reshape(len(per_lane), -1), axis=1, bitorder="little")
    return [int.from_bytes(np.packbits(a[:, t], bitorder="little").tobytes(), "little") for t in range(a.shape[1])]


def _lanes(values: Sequence[int], T: int) -> np.ndarray:
    """Bit-sliced ints (LSB first) as T lane-major integers."""
    out = [0] * T
    for b, v in enumerate(values):
        for t in range(T):
            out[t] |= ((v >> t) & 1) << b
    return out


def _evaluate_forms(C: Circuit, inputs: Mapping[str, Sequence[int]], T: int, forms: Sequence[L]) -> list[int]:
    """``forms``' T-bit values from ``inputs``, evaluating only the committed bits they reach (a missing input is zero): a
    read's index, before its witness exists."""
    need = 1 + max((v for f in forms for v in f.s), default=-1)
    mask = (1 << T) - 1
    val = [0] * need
    for name, bits in C.inputs.items():
        vs = inputs.get(name)
        for k, w in enumerate(bits):
            (v,) = tuple(w.s)
            if v < need and vs is not None:
                val[v] = vs[k] & mask

    def ev(f):
        x = mask if f.c else 0
        for v in f.s:
            x ^= val[v]
        return x

    looked: dict = {}
    for v in range(need):
        k = C.kind[v]
        if k == "and":
            a, b = C.data[v]
            val[v] = ev(a) & ev(b)
        elif k == "lookup":
            g, j = C.data[v]
            if g not in looked:
                looked[g] = C.lookup_values(g, [ev(f) for f in C.lookups[g][1]], T)
            val[v] = looked[g][j]
        elif k != "in":
            raise ValueError(f"a {k} bit")
    return [ev(f) for f in forms]


def witness(C: Circuit, inputs: Mapping[str, Sequence[int]], T: int, trees: Mapping[str, Tree]) -> dict[str, list[int]]:
    """The registered inputs' values for :meth:`Circuit.evaluate` over T lanes, the prover's: each registered table's root and
    domain id when they are inputs, and every read's row, salt and path at the index its lane reads (evaluated from
    ``inputs``)."""
    out: dict[str, list[int]] = {}
    ones = (1 << T) - 1
    reads = []
    for name, b in getattr(C, "registered", {}).items():
        tree = trees[name]
        for part, data in (("root", tree.root), ("domain", tree.domain.domain_id)):
            if f"registered/{name}/{part}" in C.inputs:
                out[f"registered/{name}/{part}"] = [ones * x for x in slice_bytes([data])]
        reads += [(next(iter(C.inputs[pre + "row"][0].s)), b, tree, idx, pre) for idx, pre, _ in b.reads]
    # in creation order: a read's index may depend on an earlier read's entry
    for _, b, tree, idx, pre in sorted(reads, key=lambda r: r[0]):
        lanes = _lanes(_evaluate_forms(C, {**inputs, **out}, T, idx), T)
        pos = [i >> b.table.entry_log for i in lanes]
        out[pre + "row"] = slice_bytes([row_bytes(tree.rows[p], WORD_BITS) for p in pos])
        out[pre + "salt"] = slice_bytes([tree.salts[p] for p in pos])
        paths = [tree.path(p) for p in pos]
        for lv in range(b.table.depth):
            out[f"{pre}path{lv}"] = slice_bytes([pa[lv] for pa in paths])
    return out


def evaluate(C: Circuit, inputs: Mapping[str, Sequence[int]], T: int, trees: Mapping[str, Tree]):
    """:meth:`Circuit.evaluate` with the registered inputs filled in by :func:`witness`."""
    return C.evaluate({**inputs, **witness(C, inputs, T, trees)}, T)


__all__ = ["Binding", "H5", "KEY", "OWNER", "REGISTERED_DOMAIN", "ROW_WORDS", "SCHEMA", "TABLES", "Table", "Tree", "WORD_BITS",
           "commit_string", "compress", "domain", "evaluate", "frame_leaf", "frame_leaf_gadget", "gather", "hm96_leaf", "mux_tree",
           "node_gadget", "read", "register", "slice_bytes", "swap", "table_rows", "verify", "witness"]
