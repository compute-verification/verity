"""W, a deployment's weights, committed once by weight row, and the tensor table that says where each tensor's rows are:
the one commitment PoUW and PoUS both read (`verity/tensor-table/v1`).

**W.** The deployment's tensors come in a committed order, `t = 0 … T−1`. Tensor `t` is `rows_t` rows of `row_bits_t` bits,
each row its elements in W's stored dtype (`elem_bits_t` bits each: 16 for BF16), little-endian, low bits first: the
tensor's memory (`bit_row((elements, elem_bits))`). W's rows are the tensors' rows in that order, `N_W = Σ rows_t` of
them: W's row `i` is row `r` of tensor `t` when `i = first_t + r`, `0 ≤ r < rows_t`. Each is committed as a registered row
is (`registered.py`): `x = sha512/row/v2` over its `row_bits_t` bits (the digest's prefix carries its length in bits), a
192-byte salt drawn once, `b || c = hm96.SHA512.commit_string(default_key, x, y)` and the `hm96-sha512/row/v2` leaf at
position `i` (`registered.leaf`), under W's domain (`weights_domain`): `CommitmentDomain(B, -2, RangeIndexedDomain(0,
N_W), hash="sha512")`, `B` the bytes of `identity_digest("verity/weights/domain/v1", {"deployment", "schema", "leaves":
N_W, "bytes": |W|, "table": the table's root})`. W's byte stream, PoUS's, is the rows' bytes in W's order: byte
`offset_t + r · row_bits_t / 8 + j` is byte `j` of row `r` of tensor `t` (`byte_rows`).

**The table.** Entry `t` is `(first_t, rows_t, row_bits_t, elem_bits_t, offset_t, name_t)`: `first_t` W's row of the
tensor's row 0, `offset_t` its first byte in W's stream, and `name_t` the SHA-512 of the tensor's name (UTF-8), which
fixes the committed order. The table has a public number of entries, `cap ≥ T`: entries `T … cap−1` are padding, `(N_W,
0, 0, 0, |W|, 64 zero bytes)`, so the table hides `T` and, when `T > 1`, how `N_W` and `|W|` split among the tensors; with
one tensor, `N_W` and `|W|` are its shape. An entry is 104 bytes (five u64 and
the name), 52 u16 words, little-endian (`entry_words`), and the table is committed as `cap` registered rows of 52 16-bit
leaves (`hm96-sha512/row/v2`, 832 bits each) under its own domain (`table_domain`: `identity_digest(
"verity/tensor-table/domain/v1", {"deployment", "schema", "words": 52, "word_bits": 16, "leaves": cap})`). Its root, `N_W`,
`|W|` and `cap` are public; the entries, the rows and the salts are the prover's.

**Well-formedness**, the statement PoUS's setup and PoUW's epoch statement prove about the hidden table (`check` is its
reference): with every entry opened against the table's root,
- W1: entry 0 starts W: `first_0 = offset_0 = 0`;
- W2: each entry follows the one before: `first_{t+1} = first_t + rows_t` and `offset_{t+1} = offset_t + rows_t ·
  row_bits_t / 8`;
- W3: each entry is a tensor (`rows ≥ 1`, `row_bits` a positive multiple of 8 within the statement's public bounds,
  `elem_bits ≥ 1` dividing it) or padding (`rows = row_bits = elem_bits = 0`, a zero name), and no tensor follows padding;
- W4: the last entry ends W: `first + rows = N_W` and `offset + rows · row_bits / 8 = |W|`.
Then the tensors' rows tile W's rows and W's bytes exactly once, in order, and a read of W's row `i` through an entry `t`
with `first_t ≤ i < first_t + rows_t` is the row it names: its `row_bits_t` are in its digest's prefix, so a leaf committed
at another length opens no read of this one. PoUW reads one weight row at a hidden position this way; PoUS's payload
`[lo, hi)` falls in the rows `byte_rows` gives, at most two when every row is at least `hi − lo − 1` bytes (a lower bound
the statement fixes, W3's `min_row_bits`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from verity.primitives.commitments import hm96
from verity.primitives.commitments.identity import identity_digest
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.merkle import CommitmentDomain, tree_levels
from verity.primitives.commitments.rowleaf import sha512_row_v2_digest

from . import registered as G

FORMAT = "verity/tensor-table/v1"
TABLE_DOMAIN = "verity/tensor-table/domain/v1"
WEIGHTS_DOMAIN = "verity/weights/domain/v1"
SCHEMA = G.SCHEMA
NAME_BYTES = 64
#: an entry's words: five u64 (first, rows, row_bits, elem_bits, offset), four u16 words each, then the name's 32
ENTRY_WORDS = 5 * 4 + NAME_BYTES // 2
ENTRY_BITS = 16 * ENTRY_WORDS
_U64 = 1 << 64
_INST = hm96.SHA512


@dataclass(frozen=True)
class Tensor:
    """One tensor of W, in the committed order: `rows` rows of `row_bits` bits, elements of `elem_bits` bits."""

    name: str
    rows: int
    row_bits: int
    elem_bits: int = 16


@dataclass(frozen=True)
class Entry:
    first: int
    rows: int
    row_bits: int
    elem_bits: int
    offset: int
    name: bytes

    @property
    def padding(self) -> bool:
        return self.rows == 0 and self.row_bits == 0 and self.elem_bits == 0 and self.name == bytes(NAME_BYTES)

    @property
    def row_bytes(self) -> int:
        return self.row_bits // 8


def name_digest(name: str) -> bytes:
    return hashlib.sha512(name.encode()).digest()


def table(tensors: Sequence[Tensor], cap: int) -> list[Entry]:
    """The table of `tensors` in their order, padded to `cap` entries."""
    if len(tensors) > cap:
        raise ValueError(f"{len(tensors)} tensors do not fit a table of {cap} entries")
    out, first, offset = [], 0, 0
    for t in tensors:
        if t.rows < 1 or t.row_bits < 8 or t.row_bits % 8 or t.elem_bits < 1 or t.row_bits % t.elem_bits:
            raise ValueError(f"tensor {t.name!r}: {t.rows} rows of {t.row_bits} bits in {t.elem_bits}-bit elements is no "
                             f"tensor of whole-byte rows")
        out.append(Entry(first, t.rows, t.row_bits, t.elem_bits, offset, name_digest(t.name)))
        first, offset = first + t.rows, offset + t.rows * t.row_bits // 8
    return out + [Entry(first, 0, 0, 0, offset, bytes(NAME_BYTES))] * (cap - len(tensors))


def entry_words(e: Entry) -> list[int]:
    """The entry's row: `first, rows, row_bits, elem_bits, offset` as u64, then the name, in u16 words, little-endian."""
    nums = (e.first, e.rows, e.row_bits, e.elem_bits, e.offset)
    if any(not 0 <= v < _U64 for v in nums) or len(e.name) != NAME_BYTES:
        raise ValueError("an entry is five u64 and a 64-byte name")
    raw = b"".join(v.to_bytes(8, "little") for v in nums) + e.name
    return [int.from_bytes(raw[k:k + 2], "little") for k in range(0, len(raw), 2)]


def decode_entry(words: Sequence[int]) -> Entry:
    if len(words) != ENTRY_WORDS or any(not 0 <= w < 1 << 16 for w in words):
        raise ValueError(f"an entry is {ENTRY_WORDS} u16 words")
    raw = b"".join(int(w).to_bytes(2, "little") for w in words)
    nums = [int.from_bytes(raw[8 * k:8 * k + 8], "little") for k in range(5)]
    return Entry(*nums, name=raw[40:])


def check(entries: Sequence[Entry], leaves: int, total_bytes: int, cap: int, min_row_bits: int = 8,
          max_row_bits: int = _U64 - 1) -> None:
    """W1-W4 on the opened table against the public `leaves` (N_W), `total_bytes` (|W|) and `cap`; raises ValueError
    naming the first failure. `min_row_bits` is at least 8: below it a zero-width tensor would pass W3."""
    if min_row_bits < 8:
        raise ValueError(f"W3's lower bound is {min_row_bits} bits, and a row is at least one byte")
    if len(entries) != cap or cap < 1:
        raise ValueError(f"the table is {len(entries)} entries, and its public size is {cap}")
    if entries[0].first != 0 or entries[0].offset != 0:
        raise ValueError("W1: entry 0 does not start W at row 0, byte 0")
    seen_padding = False
    for t, e in enumerate(entries):
        if e.padding:
            seen_padding = True
        elif seen_padding:
            raise ValueError(f"W3: entry {t} is a tensor after padding")
        elif not (e.rows >= 1 and min_row_bits <= e.row_bits <= max_row_bits and e.row_bits % 8 == 0 and e.elem_bits >= 1
                  and e.row_bits % e.elem_bits == 0):
            raise ValueError(f"W3: entry {t} is neither a tensor of whole-byte rows of {min_row_bits} to {max_row_bits} "
                             f"bits nor padding")
        if t + 1 < len(entries):
            nxt = entries[t + 1]
            if nxt.first != e.first + e.rows or nxt.offset != e.offset + e.rows * e.row_bytes:
                raise ValueError(f"W2: entry {t + 1} does not start where entry {t} ends")
    last = entries[-1]
    if last.first + last.rows != leaves or last.offset + last.rows * last.row_bytes != total_bytes:
        raise ValueError(f"W4: the table ends at row {last.first + last.rows}, byte {last.offset + last.rows * last.row_bytes}, "
                         f"and W is {leaves} rows, {total_bytes} bytes")


def locate(entries: Sequence[Entry], i: int) -> tuple[int, int]:
    """W's row `i` as `(t, r)`: row `r` of tensor `t`."""
    for t, e in enumerate(entries):
        if e.first <= i < e.first + e.rows:
            return t, i - e.first
    raise ValueError(f"row {i} is not a row of W")


def byte_rows(entries: Sequence[Entry], lo: int, hi: int) -> list[tuple[int, int, int]]:
    """The rows W's bytes `[lo, hi)` fall in, in order: `(i, a, b)`, bytes `[a, b)` of W's row `i`."""
    out = []
    for e in entries:
        end = e.offset + e.rows * e.row_bytes
        if e.padding or end <= lo or e.offset >= hi:
            continue
        for r in range((max(lo, e.offset) - e.offset) // e.row_bytes, (min(hi, end) - e.offset - 1) // e.row_bytes + 1):
            start = e.offset + r * e.row_bytes
            out.append((e.first + r, max(lo, start) - start, min(hi, start + e.row_bytes) - start))
    if sum(b - a for _i, a, b in out) != hi - lo:
        raise ValueError(f"bytes [{lo}, {hi}) are not bytes of W")
    return out


def table_domain(deployment: str, cap: int) -> CommitmentDomain:
    binding = identity_digest(TABLE_DOMAIN, {"deployment": deployment, "schema": SCHEMA, "words": ENTRY_WORDS, "word_bits": 16,
                                             "leaves": int(cap)})
    return CommitmentDomain(bytes.fromhex(binding), G.OWNER, RangeIndexedDomain(0, int(cap)), hash="sha512")


def weights_domain(deployment: str, leaves: int, total_bytes: int, table_root: bytes) -> CommitmentDomain:
    binding = identity_digest(WEIGHTS_DOMAIN, {"deployment": deployment, "schema": SCHEMA, "leaves": int(leaves),
                                               "bytes": int(total_bytes), "table": table_root.hex()})
    return CommitmentDomain(bytes.fromhex(binding), G.OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


@dataclass(frozen=True)
class Committed:
    """The prover's side of one tree of registered rows: the domain, every row's salt and `b || c`, the levels."""

    domain: CommitmentDomain
    salts: tuple[bytes, ...]
    bcs: tuple[bytes, ...]
    levels: tuple[tuple[bytes, ...], ...]

    @classmethod
    def commit(cls, dom: CommitmentDomain, digests: Sequence[bytes], salts: Sequence[bytes] | None = None) -> "Committed":
        salts = tuple(bytes(s) for s in salts) if salts is not None else tuple(_INST.fresh_salt() for _ in digests)
        if len(salts) != len(digests) or len(set(salts)) != len(salts):
            raise ValueError("one salt per row, never reused")
        bcs = tuple(_INST.commit_string(_INST.default_key, x, s) for x, s in zip(digests, salts))
        return cls(dom, salts, bcs, tree_levels(dom, [G.leaf(dom, i, bc) for i, bc in enumerate(bcs)]))

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, position: int) -> bytes:
        out, i = b"", position
        for level in self.levels[:-1]:
            out += level[i ^ 1]
            i >>= 1
        return out


def commit_table(deployment: str, entries: Sequence[Entry], salts: Sequence[bytes] | None = None) -> Committed:
    dom = table_domain(deployment, len(entries))
    return Committed.commit(dom, [G.row_digest(entry_words(e)) for e in entries], salts)


def commit_weights(deployment: str, entries: Sequence[Entry], rows: Sequence[bytes], table_root: bytes,
                   salts: Sequence[bytes] | None = None) -> Committed:
    """W's tree over `rows` (each W's row `i` as its bytes, `bit_row`'s), each at its entry's length in bits."""
    leaves = sum(e.rows for e in entries)
    total = sum(e.rows * e.row_bytes for e in entries)
    if len(rows) != leaves:
        raise ValueError(f"the table names {leaves} rows of W, and there are {len(rows)}")
    digests = []
    for i, row in enumerate(rows):
        e = entries[locate(entries, i)[0]]
        if len(row) != e.row_bytes:
            raise ValueError(f"row {i} is {len(row)} bytes, and its tensor's rows are {e.row_bytes}")
        digests.append(sha512_row_v2_digest(bytes(row), e.row_bits))
    return Committed.commit(weights_domain(deployment, leaves, total, table_root), digests, salts)


def verify_entry(dom: CommitmentDomain, root: bytes, t: int, bc: bytes, path: bytes) -> bool:
    """`bc` as the table's entry `t` opens `root` (the check each of the statement's hidden reads makes in gates)."""
    return G.verify_read(dom, root, t, bc, path)


def manifest(deployment: str, cap: int, leaves: int, total_bytes: int, table_root: bytes, weights_root: bytes) -> dict[str, Any]:
    """What the deployment's registration names: the two roots and the public sizes."""
    return {"format": FORMAT, "deployment": deployment, "schema": SCHEMA, "cap": int(cap), "leaves": int(leaves),
            "bytes": int(total_bytes), "table": table_root.hex(), "weights": weights_root.hex()}


__all__ = ["ENTRY_BITS", "ENTRY_WORDS", "FORMAT", "NAME_BYTES", "SCHEMA", "TABLE_DOMAIN", "WEIGHTS_DOMAIN", "Committed",
           "Entry", "Tensor", "byte_rows", "check", "commit_table", "commit_weights", "decode_entry", "entry_words",
           "locate", "manifest", "name_digest", "table", "table_domain", "verify_entry", "weights_domain"]
