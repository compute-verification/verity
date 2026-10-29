"""ShaLB's numbers: SHA-512's registers written a compression, the bounds below it, and what each leaves of T1: stdlib.

    python3 benchmarks/pouw/shalb/shalb.py [--counts COUNTS_JSON] [--minimal MINIMAL_JSON] [--json OUT]
        [--l1-leaves 1,4,128,...] [--l1-value BYTES] [--l1-row-bytes 2048,...] [--l1-row-leaves N]
        [--words-bytes 16384,65536] [--words-leaves N] [--lead-w1 W1] [--a-w1 W1] [--a-leaves N]
        [--card-credit REGISTERS] [--honest-extra-per-compression REGISTERS]

- **The structural count.** One compression in the minimal kernel's dataflow (sha512_min.cuh), each 64-bit word tagged
  with a level and constants propagated: Σ/σ 8 registers (6 SHF, 2 LOP3), Ch and Maj 2 (a LOP3 a half), an add of m
  terms 2·⌈(m − 1)/2⌉ (IADD3 / IADD3.X; constants fold into one immediate), each round's T1 either materialized or
  folded into e′ and a′, whichever writes fewer, and e₈₀ and a₈₀ folded into the feed-forward. A word is a compile-time
  constant, per call (shared by a call's leaves, or a level's nodes, and computed once a call), per item (a leaf's or
  node's, data-independent: the domain id's, rank's, position's or index's) or data; a value written is at its
  inputs' highest level, and terms at or below per call fold into one before an add. COUNTS_JSON (shalb_sass.sh's
  counts.json) gives the compiled counts beside it. A chain's blocks in chain form (`chain`) write 4 fewer a pair.
- **hm96's leaf** (verity/ml/commitments/hm96/PROTOCOL.md), compression by compression and form by form.
- **L1** (`l1_call`): one call's hashing when hm96's inner digest is a frame-v3-sha512 subtree root over the call's
  values, in the call's own range-u64 domain (frame_v3/PROTOCOL.md §1 and §6), with the shared work counted once a call
  (each leaf's prefix through the domain id, each level's node midstate, the domain id), the per-item work split into
  data and data-independent, and what an exact repeat of a leaf's value shares: its blocks' schedules, ASICBoost's reuse
  across chaining values. A leaf's value may instead be a row's digest: a row-seg row (`rowseg_row`: 1 KiB segments,
  each from its own prefix block, with or without (salt_w, f, j, row) bound into it) or a row/v2 row (`rowv2_row`, the
  prefix block hoisted). ShaWrites at the lead is priced at the hiding module's layout: A as 1 KiB leaves of the call's domain,
  the words as row/v2 chunks, each digest a leaf of the forward's words domain; a repeated or constant value's shared
  schedule is priced on A only, since the words depend on the noise.
- **Honest against DistinctOn** (`distinct_writes`): the minimal kernel's 3,534 registers, simulated, on a random grid:
  pairwise distinct, none constant, none an input half plus a constant, so they witness ShaWrites' R for it.
- **The bound ladder** (rungs B0–B3): what each rank's premise proves about any program's registers written a compression, and the
  check its fan-in rungs read: every output register of the compression depends on every input register (a witness
  pair each, from a stdlib SHA-512 held to hashlib).
- **The bitsliced count**, the alternative representation every per-function argument has to cover.
- **T1**: the hashing W1 a useful MAC at SSN's (`ssn_1l`, α 1.044, every k32) and NCP's volumes, the δ T1 allows, the
  term each rung of the ladder leaves, the frontier (bytes a MAC each rung allows), and the pooled figures for a week, a
  month and a year: one H200 (serving at `MFU_SERVE` of its tensor peak), the reference datacenter, the world.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

ROUNDS, SCHEDULE = 80, 64
W1_PER_REG = 32
BLOCK = 128
AND_PER_ADD64 = 63
AND_PER_MAC = 130
H200_MACS_PER_S = 9.89e14
MFU_SERVE = 0.4
H200_SERVED_MACS_PER_S = H200_MACS_PER_S * MFU_SERVE
DC_GPUS = 100_000
WINDOW_S = 600
WORLD_MACS_PER_YEAR = 4.68e30
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
VOLUMES = {"SSN ssn_1l (α 1.044, every k32)": 1.044 * 4 / 32, "NCP k32": 0.375}


ZERO, CONST, CALL, ITEM, DATA = range(-1, 4)
LEVELS = {CALL: "per call", ITEM: "per leaf or node, data-independent", DATA: "data"}


class Count:
    """Registers written, by kind and by the level of the value written: ZERO and CONST (compile time, free; a zero term
    drops out of an add), CALL (the same for every leaf of a call or every node of a level: computed once a call), ITEM
    (one leaf's or node's, from the domain id, rank, position or index alone) or DATA. Schedule ops also record the block
    words they read (`sched`)."""

    def __init__(self):
        self.by = {(k, lv): 0 for k in ("SHF", "LOP3", "adds") for lv in (CALL, ITEM, DATA)}
        self.sched: list[tuple[int, int, int]] = []

    def charge(self, kind: str, level: int, n: int, mask: int | None) -> None:
        if level > CONST and n:
            self.by[kind, level] += n
            if mask is not None:
                self.sched.append((n, level, mask))

    def sigma(self, level: int, mask: int | None = None) -> int:
        self.charge("SHF", level, 6, mask)
        self.charge("LOP3", level, 2, mask)
        return level

    def bitwise(self, *levels: int) -> int:
        self.charge("LOP3", max(levels), 2, None)
        return max(levels)

    def add(self, terms: list[int], mask: int | None = None) -> int:
        """A 64-bit add: terms at CONST and CALL fold first into one term (constants into an immediate, a per-call sum
        once a call), then that term and the rest in IADD3 / IADD3.X pairs, 2·⌈(m − 1)/2⌉ registers for m terms."""
        terms = [t for t in terms if t != ZERO]
        if not terms:
            return ZERO
        top = max(terms)
        low = [t for t in terms if t <= CALL]
        if top <= CALL:
            m = terms.count(CALL) + (1 if CONST in terms and CALL in terms else 0)
        else:
            if CALL in low:
                self.add(low, mask)
            m = len(terms) - len(low) + (1 if low else 0)
        self.charge("adds", top, 0 if m <= 1 else 2 * math.ceil((m - 1) / 2), mask)
        return top

    def at(self, *levels: int) -> int:
        return sum(v for (_, lv), v in self.by.items() if lv in levels)


def _compress(c: Count, cv: int, words: list[int]) -> None:
    w = list(words)
    dep = [1 << i for i in range(16)]
    for t in range(16, ROUNDS):
        s1, s0 = c.sigma(w[t - 2], dep[t - 2]), c.sigma(w[t - 15], dep[t - 15])
        dep.append(dep[t - 2] | dep[t - 7] | dep[t - 15] | dep[t - 16])
        w.append(c.add([s1, w[t - 7], s0, w[t - 16]], dep[t]))
    st = [cv] * 8
    for t in range(ROUNDS):
        a, b, cc, d, e, f, g, h = st
        s1, ch = c.sigma(e), c.bitwise(e, f, g)
        s0, mj = c.sigma(a), c.bitwise(a, b, cc)
        t1 = [h, w[t], s1, ch, CONST]
        last = t == ROUNDS - 1
        e_x, a_x = [d] + ([cv] if last else []), [s0, mj] + ([cv] if last else [])
        options = []
        for fold in (False, True):
            x = Count()
            head = t1 if fold else [x.add(t1)]
            x.add(head + e_x)
            x.add(head + a_x)
            options.append((x.at(ITEM, DATA), x.at(CALL), x))
        x = min(options, key=lambda o: o[:2])[2]
        for k, v in x.by.items():
            c.by[k] += v
        st = [max(t1 + a_x[:2]), a, b, cc, max(t1 + [d]), e, f, g]
    for i in (1, 2, 3, 5, 6, 7):
        c.add([st[i], cv])


def compression_levels(cv: int, words: list[int]) -> Count:
    """One compression whose chaining value and 16 message words are at the given levels."""
    c = Count()
    _compress(c, cv, words)
    return c


def compression(cv_var: bool, words_var: list[bool]) -> dict:
    """Registers written by one compression whose chaining value is variable (or constant) and whose 16 message words
    are variable or constant as `words_var` says."""
    c = compression_levels(DATA if cv_var else CONST, [DATA if v else CONST for v in words_var])
    kinds = {k: sum(v for (kk, _), v in c.by.items() if kk == k) for k in ("SHF", "LOP3", "adds")}
    return {**kinds, "registers": sum(kinds.values())}


CHAIN_PAIR_SAVING = 4


def chain(n: int) -> int:
    """Registers a chain of n compressions from a constant chaining value saves in chain form (sha512_min.cuh's CHAIN):
    a pair of consecutive blocks leaves the first's H3 and H7 + K_0 as two terms each, read by the second only in adds
    with a free IADD3 slot, so 4 fewer registers a pair. The constant chaining value takes K_0 into H7 as an immediate
    and a trailing unpaired block takes it out in its feed-forward; an even chain longer than 2 has no such block and
    pays an add (2 registers) to take it out."""
    return CHAIN_PAIR_SAVING * (n // 2) - (2 if n % 2 == 0 and n > 2 else 0)


FRAME = b"veritor/protocol/merkle/frame/v3\0"


def frame_message(tag: bytes, parts: list[tuple[bytes | int, int]]) -> list[list[int]]:
    """The word levels of each SHA-512 block of a frame-v3 message (the frame, the tag after its u32be length, then each
    part after its u64be length; `verity.ml.commitments.merkle`), padding included. `parts` are (bytes, CONST) or
    (length, level); a word of constant bytes all zero is ZERO."""
    msg = [(CONST, x) for x in FRAME + len(tag).to_bytes(4, "big") + tag]
    for part, level in parts:
        n = len(part) if level == CONST else part
        msg += [(CONST, x) for x in n.to_bytes(8, "big")]
        msg += [(CONST, x) for x in part] if level == CONST else [(level, None)] * n
    return _blocks(msg)


def _blocks(msg: list[tuple[int, int | None]]) -> list[list[int]]:
    """A message's bytes, each (level, byte) with the byte given only for a constant, as SHA-512 blocks of word levels,
    padding included; a word of constant bytes all zero is ZERO."""
    bits = 8 * len(msg)
    msg += [(CONST, x) for x in b"\x80" + bytes(-(len(msg) + 17) % BLOCK) + bits.to_bytes(16, "big")]

    def word(ws: list[tuple[int, int | None]]) -> int:
        top = max(lv for lv, _ in ws)
        return ZERO if top == CONST and not any(x for _, x in ws) else top

    return [[word(msg[i:i + 8]) for i in range(b, b + BLOCK, 8)] for b in range(0, len(msg), BLOCK)]


def message_cost(blocks: list[list[int]], cv: int = CONST) -> dict:
    """One message's registers by level, its per-item blocks in chain form, and `repeat`: what a message with the same
    DATA words but other ITEM words (another rank or position, so another chaining value) shares with it, the schedule
    ops that read no ITEM word (ASICBoost's reuse of a block's schedule across chaining values)."""
    r = _message_cost(tuple(map(tuple, blocks)), cv)
    return {**r, "by_level": dict(r["by_level"])}


@functools.lru_cache(maxsize=None)
def _message_cost(blocks: tuple[tuple[int, ...], ...], cv: int) -> dict:
    by = {CALL: 0, ITEM: 0, DATA: 0}
    repeat, item = 0, []
    for words in blocks:
        c = compression_levels(cv, words)
        for lv in by:
            by[lv] += c.at(lv)
        d = sum(1 << i for i, w in enumerate(words) if w == ITEM)
        repeat += sum(n for n, lv, m in c.sched if lv == DATA and not m & d)
        cv = max(cv, *words)
        if cv >= ITEM:
            item.append(cv)
    for p in range(len(item) // 2):
        by[item[2 * p]] -= CHAIN_PAIR_SAVING
    if len(item) % 2 == 0 and len(item) > 2:
        by[item[-1]] += 2
    return {"compressions": len(blocks), "per_item": len(item), "by_level": by, "repeat": repeat}


def _ulen(x: int) -> int:
    return max(1, (x.bit_length() + 7) // 8)


ROW_SEG_TAG, ROW_SEG_SEGMENT_TAG, ROW_SEG_BYTES = b"verity/sha512-row-seg/v1\0", b"verity/sha512-row-seg/v1/segment\0", 1024


def rowseg_segment(j: int, s: int, m: int, bound: bool = True, value: int = DATA) -> list[tuple[int, int | None]]:
    """Segment j of s, m bytes, as (level, byte) pairs: S(j, s, m) and, when `bound`, salt_w, f and j (the call's 48
    bytes) and the row (8 bytes, per item), zero padded to a block; then the segment (DATA, or zeros for `value`
    CONST). sha512_rowseg.cuh computes a full one."""
    msg = [(CONST, x) for x in ROW_SEG_SEGMENT_TAG + j.to_bytes(8, "big") + s.to_bytes(8, "big")
           + (8 * m).to_bytes(8, "big")]
    msg += [(CALL, None)] * 48 + [(ITEM, None)] * 8 if bound else []
    msg += [(CONST, 0)] * (BLOCK - len(msg))
    return msg + ([(DATA, None)] * m if value == DATA else [(CONST, 0)] * m)


def rowseg_row(row_bytes: int, bound: bool = True, value: int = DATA) -> dict:
    """One `sha512/row-seg/v1` row (`verity.ml.commitments.rowleaf`): each 1 KiB segment hashed from its own 128-byte
    prefix block S(j, s, m), then the outer hash over Q(n_bits) and the segments' digests. `bound`: (salt_w, f, j, row)
    follow S(j, s, m) in the prefix block (prefix binding; the 32-byte salt and the call per call, the row per
    item), else the prefix block is a constant midstate. `value`: DATA, or CONST for a row the prover makes all zero.
    message_cost's fields summed over the row's messages; `repeat` counts only the segments' schedules (a repeated row
    under binding has other segment digests), and `digest` is the row digest's level."""
    s = -(-row_bytes // ROW_SEG_BYTES)
    by, comp, per_item, repeat, seg_level = {CALL: 0, ITEM: 0, DATA: 0}, 0, 0, 0, CONST
    for j in range(s):
        m = min(ROW_SEG_BYTES, row_bytes - ROW_SEG_BYTES * j)
        msg = rowseg_segment(j, s, m, bound, value)
        c = message_cost(_blocks(msg))
        for lv in by:
            by[lv] += c["by_level"][lv]
        comp, per_item, repeat = comp + c["compressions"], per_item + c["per_item"], repeat + c["repeat"]
        seg_level = max([seg_level] + [lv for lv, _ in msg])
    outer = [(CONST, x) for x in (ROW_SEG_TAG + (8 * row_bytes).to_bytes(8, "big") + s.to_bytes(8, "big")
                                  + ROW_SEG_BYTES.to_bytes(4, "big")).ljust(BLOCK, b"\0")]
    c = message_cost(_blocks(outer + [(seg_level, None)] * (64 * s)))
    for lv in by:
        by[lv] += c["by_level"][lv]
    if not bound:
        repeat = by[ITEM] + by[DATA]
    return {"compressions": comp + c["compressions"], "per_item": per_item + c["per_item"], "by_level": by,
            "repeat": repeat, "digest": seg_level}


ROW_V2_TAG = b"verity/sha512-row/v2\0"


def rowv2_row(row_bytes: int, value: int = DATA) -> dict:
    """One `sha512/row/v2` row, SHA-512(P(n_bits) ‖ row) (`verity.ml.commitments.rowleaf`). P(n_bits) is one constant
    block, so the row's first block starts from a constant midstate (the prefix block hoisted) and is not an item's
    compression. Nothing binds the row, so an exact repeat shares its whole digest (`repeat`)."""
    head = [(CONST, x) for x in (ROW_V2_TAG + (8 * row_bytes).to_bytes(8, "big")).ljust(BLOCK, b"\0")]
    c = message_cost(_blocks(head + ([(DATA, None)] * row_bytes if value == DATA else [(CONST, 0)] * row_bytes)))
    return {**c, "repeat": c["by_level"][ITEM] + c["by_level"][DATA], "digest": value}


def l1_call(leaves: int, value_bytes: int = 1024, schema: str = "u8", row_bytes: int = 0, bound: bool = True,
            row_schema: str = "row-seg") -> dict:
    """One call's hashing under L1 (`hm96-sha512/subtree/v1`): `leaves` frame-v3-sha512 leaves of `value_bytes` each in
    the call's own range-u64 domain (positions 0..leaves − 1), their tree, the domain id and the call's hm96 instance
    (c, its leaf and the salt PRF, 3 compressions). With `row_bytes`, each leaf's value is instead a row's digest, and
    the rows' hashing is the leaves': a `sha512/row-seg/v1` row (`rowseg_row`, `bound` as there) under the schema
    `hm96-sha512/row-seg/v1`, or with `row_schema` "row/v2" a `sha512/row/v2` row (`rowv2_row`, the words' chunks)
    under `sha512/row/v2/l1-w`. Shared work is counted once: the leaves' prefix through the domain id (a midstate a
    call), each level's node midstate, the domain id. Per call, by level; and what a value the prover makes constant
    saves (its blocks' schedule is then a constant: `constant_value_share`, of the leaf and of the call) and what
    exact repeats of one value share (`repeat_share`, of the call)."""
    depth = (leaves - 1).bit_length()
    rl = _ulen(leaves - 1)
    if row_bytes:
        schema = "sha512/row/v2/l1-w" if row_schema == "row/v2" else "hm96-sha512/row-seg/v1"
        value_bytes = 64

    def leaf_cost(value: int) -> dict:
        if row_bytes:
            row = rowv2_row(row_bytes, value) if row_schema == "row/v2" else rowseg_row(row_bytes, bound, value)
            value = row["digest"]
        part = (bytes(value_bytes), CONST) if value == CONST else (value_bytes, value)
        c = message_cost(frame_message(b"leaf", [(64, CALL), (rl, ITEM), (rl, ITEM), (schema.encode(), CONST),
                                                 part]))
        if row_bytes:
            c = {"compressions": c["compressions"] + row["compressions"], "per_item": c["per_item"] + row["per_item"],
                 "by_level": {lv: c["by_level"][lv] + row["by_level"][lv] for lv in c["by_level"]},
                 "repeat": c["repeat"] + row["repeat"]}
        return c

    leaf = leaf_cost(DATA)
    constant = leaf_cost(CONST)
    what = (f"row/v2 row ({row_bytes} B, prefix hoisted) and its leaf" if row_schema == "row/v2" else
            f"row-seg row ({row_bytes} B, {'bound' if bound else 'constant'} prefix) and its leaf") if row_bytes \
        else f"leaf ({value_bytes} B value)"
    rows = [{"what": what, "n": leaves, **leaf}]
    for lvl in range(1, depth + 1):
        width, real = 1 << depth - lvl, -(-leaves // (1 << lvl))
        node = message_cost(frame_message(b"node", [(64, CALL), (1, CALL), (_ulen(width - 1), ITEM), (64, DATA),
                                                    (64, DATA)]))
        pad_node = {**node, "by_level": {CALL: node["by_level"][CALL], ITEM: node["by_level"][ITEM]
                                         + node["by_level"][DATA], DATA: 0}, "repeat": 0}
        rows.append({"what": f"node, level {lvl}", "n": real, **node})
        if width > real:
            rows.append({"what": f"node over pads only, level {lvl}", "n": width - real, **pad_node})
    if (1 << depth) > leaves:
        pad = message_cost(frame_message(b"pad", [(64, CALL), (rl, ITEM)]))
        rows.append({"what": "pad", "n": (1 << depth) - leaves, **pad})
    rows.append({"what": "domain id (range-u64)", "n": 1,
                 **message_cost(frame_message(b"domain/range-u64", [(64, CALL)] + [(8, CALL)] * 3))})
    inst = {
        "salt PRF (3 compressions; (a)'s derivation)": ([(CONST, [ITEM] * 16)] + [(ITEM, [ITEM] * 16)] * 2, ITEM),
        "c = H(salt prefix ‖ y)": ([(CONST, [ITEM] * 16), (ITEM, [ITEM] * 8 + [CONST] + [ZERO] * 6 + [CONST])], ITEM),
        "instance leaf = H(leaf_prefix(key) ‖ b ‖ c)": ([(CONST, [DATA] * 16), (DATA, [CONST] + [ZERO] * 14 + [CONST])],
                                                        DATA),
    }
    for what, (blocks, lv) in inst.items():
        regs = sum(compression_levels(cv, w).at(ITEM, DATA) for cv, w in blocks) - chain(len(blocks))
        rows.append({"what": what, "n": 1, "compressions": len(blocks), "per_item": len(blocks),
                     "by_level": {CALL: 0, ITEM: regs if lv == ITEM else 0, DATA: regs if lv == DATA else 0},
                     "repeat": 0})
    shared = {"leaf": leaf["by_level"][CALL], "node": 0, "domain id": 0}
    comp = 0
    for r in rows:
        r["registers"] = sum(r["by_level"][lv] for lv in (ITEM, DATA))
        comp += r["n"] * r["per_item"]
        for key in ("node", "domain id"):
            if r["what"].startswith(key) and "pads only" not in r["what"]:
                shared[key] += r["by_level"][CALL]
                comp += r["compressions"] - r["per_item"]
    items = sum(r["n"] * r["registers"] for r in rows)
    data = sum(r["n"] * r["by_level"][DATA] for r in rows)
    total = items + sum(shared.values())
    leaf_regs = sum(leaf["by_level"][lv] for lv in (ITEM, DATA))
    constant_drop = leaf_regs - sum(constant["by_level"][lv] for lv in (ITEM, DATA))
    return {"leaves": leaves, "value_bytes": value_bytes, "row_bytes": row_bytes, "bound": bound, "rows": rows,
            "shared_once_a_call": shared,
            "registers": total, "compressions": comp, "per_compression": total / comp,
            "data": data, "data_independent": total - data, "chained_3532": 3532 * comp,
            "leaf_repeat_share": leaf["repeat"] / leaf_regs,
            "leaf_constant_value_share": constant_drop / leaf_regs,
            "constant_value_share": leaves * constant_drop / total,
            "repeat_share": (leaves - 1) * leaf["repeat"] / total}


def and_gates() -> int:
    """AND gates of one compression as a Boolean circuit (Ch and Maj one AND a bit, an n-bit add n − 1)."""
    return ROUNDS * (7 * AND_PER_ADD64 + 2 * 64) + SCHEDULE * 3 * AND_PER_ADD64 + 8 * AND_PER_ADD64


def bitsliced() -> dict:
    """LOP3s a compression, bitsliced 32 instances to a register: rotations free, a three-way XOR or a Ch/Maj one LOP3 a
    bit, a two-term add two a bit but its top bit's carry, transposes left out."""
    add = 2 * 64 - 1
    per_round = 4 * 64 + 7 * add
    per_word = 2 * 64 + 3 * add
    total = ROUNDS * per_round + SCHEDULE * per_word + 8 * add
    return {"LOP3 per 32 instances": total, "per instance": total / 32}


def hm96_leaf(row_bytes: int) -> list[dict]:
    """hm96-sha512's compressions for one leaf of `row_bytes` value bytes behind a block-aligned prefix, by form."""
    var16 = [True] * 16
    blocks = math.ceil((row_bytes + 17) / BLOCK)
    tail_words = (row_bytes % BLOCK) // 8
    last = [i < tail_words for i in range(16)] if row_bytes % BLOCK <= BLOCK - 17 else var16
    rows = [
        {"what": "x: first value block (constant midstate of the prefix)", "n": 1, **compression(False, var16)},
        {"what": "x: middle value blocks", "n": max(blocks - 2, 0), **compression(True, var16)},
        {"what": "x: last block (value tail and padding)", "n": 1 if blocks > 1 else 0,
         **compression(True, last)},
        {"what": "c: salt bytes 0–127 (constant midstate of the salt prefix)", "n": 1, **compression(False, var16)},
        {"what": "c: salt bytes 128–191 and padding", "n": 1,
         **compression(True, [i < 8 for i in range(16)])},
        {"what": "leaf: b ‖ c (constant midstate of leaf_prefix(key))", "n": 1, **compression(False, var16)},
        {"what": "leaf: padding block", "n": 1, **compression(True, [False] * 16)},
        {"what": "chain form: x's, c's and the leaf's chains in pairs of blocks", "n": 1,
         "registers": -(chain(blocks) + chain(2) + chain(2))},
    ]
    return rows


M64 = (1 << 64) - 1


def _primes(n: int) -> list[int]:
    ps: list[int] = []
    c = 2
    while len(ps) < n:
        if all(c % p for p in ps):
            ps.append(c)
        c += 1
    return ps


def _icbrt(n: int) -> int:
    x = 1 << -(-n.bit_length() // 3)
    while (y := (2 * x + n // (x * x)) // 3) < x:
        x = y
    while x ** 3 > n:
        x -= 1
    return x


SHA_K = [_icbrt(p << 192) & M64 for p in _primes(80)]
SHA_IV = [math.isqrt(p << 128) & M64 for p in _primes(8)]


def _rotr(x: int, n: int) -> int:
    return (x >> n | x << (64 - n)) & M64


def sha512_compress(cv: list[int], block: list[int]) -> list[int]:
    """SHA-512's compression (FIPS 180-4), its constants computed from the primes."""
    w = list(block)
    for t in range(16, ROUNDS):
        s0 = _rotr(w[t - 15], 1) ^ _rotr(w[t - 15], 8) ^ w[t - 15] >> 7
        s1 = _rotr(w[t - 2], 19) ^ _rotr(w[t - 2], 61) ^ w[t - 2] >> 6
        w.append((s1 + w[t - 7] + s0 + w[t - 16]) & M64)
    a, b, c, d, e, f, g, h = cv
    for t in range(ROUNDS):
        t1 = (h + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + (e & f ^ ~e & g) + SHA_K[t] + w[t]) & M64
        t2 = ((_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + (a & b ^ a & c ^ b & c)) & M64
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M64, c, b, a, (t1 + t2) & M64
    return [(x + y) & M64 for x, y in zip(cv, (a, b, c, d, e, f, g, h))]


def sha512(msg: bytes) -> bytes:
    data = msg + b"\x80" + bytes(-(len(msg) + 17) % BLOCK) + (8 * len(msg)).to_bytes(16, "big")
    cv = list(SHA_IV)
    for i in range(0, len(data), BLOCK):
        cv = sha512_compress(cv, [int.from_bytes(data[j:j + 8], "big") for j in range(i, i + BLOCK, 8)])
    return b"".join(x.to_bytes(8, "big") for x in cv)


def dependence(seed: int = 1, max_trials: int = 8) -> dict:
    """Checks that each of the compression's 16 output registers depends on each of its 48 input registers (8 chaining
    words and 16 block words, lo and hi): for every pair, an input differing in that register alone whose output
    differs in that output register. A witness proves the dependence; `sha512` is first held to hashlib."""
    import hashlib
    import random
    for n in (0, 3, 111, 112, 127, 128, 200, 1000):
        msg = bytes((i * 151 + 7) % 256 for i in range(n))
        assert sha512(msg) == hashlib.sha512(msg).digest(), n
    rng = random.Random(seed)
    def halves(ws: list[int]) -> list[int]:
        return [x >> 32 * s & 0xFFFFFFFF for x in ws for s in (0, 1)]

    need = {(o, i) for o in range(16) for i in range(48)}
    trials = 0
    while need and trials < max_trials:
        trials += 1
        words = [rng.getrandbits(64) for _ in range(24)]
        base = halves(sha512_compress(words[:8], words[8:]))
        for i in range(48):
            flipped = list(words)
            flipped[i // 2] ^= 1 << (32 * (i % 2) + rng.randrange(32))
            out = halves(sha512_compress(flipped[:8], flipped[8:]))
            need -= {(o, i) for o in range(16) if out[o] != base[o]}
    return {"pairs": 16 * 48, "witnessed": 16 * 48 - len(need), "trials": trials, "missing": sorted(need)}


def honest_writes(cv: list[int], block: list[int]) -> tuple[list[int], list[int]]:
    """The 32-bit registers sha512_min.cuh's compression (a variable chaining value and block) writes, in its dataflow:
    each 64-bit rotation's two SHF windows, each three-way XOR's or Ch's or Maj's two LOP3 halves, and each add's partial
    sums, three terms an IADD3 pair (K_t in the round's first). Returns the registers and the output words."""
    regs: list[int] = []

    def put(x: int) -> int:
        regs.extend((x & 0xFFFFFFFF, x >> 32))
        return x

    def xor3(x: int, rots: tuple[int, int, int], shift: int = 0) -> int:
        parts = [put(_rotr(x, n)) if n & 31 else _rotr(x, n) for n in rots]
        if shift:
            parts[2] = put(x >> shift)
        return put(parts[0] ^ parts[1] ^ parts[2])

    def add(*terms: int) -> int:
        acc = terms[0]
        for i in range(1, len(terms), 2):
            acc = put(sum(terms[i:i + 2], acc) & M64)
        return acc

    w = list(block)
    for t in range(16, ROUNDS):
        w.append(add(xor3(w[t - 2], (19, 61, 0), 6), w[t - 7], xor3(w[t - 15], (1, 8, 0), 7), w[t - 16]))
    a, b, c, d, e, f, g, h = cv
    for t in range(ROUNDS):
        u = add(h, w[t], SHA_K[t])
        s1, ch = xor3(e, (14, 18, 41)), put(e & f ^ ~e & g & M64)
        t1 = add(u, s1, ch)
        s0, mj = xor3(a, (28, 34, 39)), put(a & b ^ a & c ^ b & c)
        if t == ROUNDS - 1:
            out = [add(t1, s0, mj, cv[0]), add(cv[1], a), add(cv[2], b), add(cv[3], c), add(t1, d, cv[4]),
                   add(cv[5], e), add(cv[6], f), add(cv[7], g)]
        else:
            h, g, f, e, d, c, b, a = g, f, e, add(d, t1), c, b, a, add(t1, s0, mj)
    return regs, out


def distinct_writes(samples: int = 6, seed: int = 2) -> dict:
    """Checks honest's registers against ShaWrites' DistinctOn on a random grid of `samples` inputs: each register is a
    function on the grid (its values there), and none may equal another, a constant, an input half, or an input half
    plus a constant (a carrier). The outputs are held to `sha512_compress`."""
    import random
    rng = random.Random(seed)
    runs = []
    for _ in range(samples):
        words = [rng.getrandbits(64) for _ in range(24)]
        regs, out = honest_writes(words[:8], words[8:])
        assert out == sha512_compress(words[:8], words[8:])
        runs.append((words, regs))
    fns = list(zip(*(r for _, r in runs)))
    inputs = list(zip(*([x >> 32 * s & 0xFFFFFFFF for x in ws for s in (0, 1)] for ws, _ in runs)))
    carriers = sum(len({(v - x) & 0xFFFFFFFF for v, x in zip(f, i)}) == 1 for f in fns for i in inputs)
    return {"registers": len(fns), "samples": samples, "distinct": len(set(fns)),
            "constant": sum(len(set(f)) == 1 for f in fns), "carriers_of_inputs": carriers}


def ladder(honest: int) -> list[dict]:
    """Each rung: the premise, its rank, and the registers any program writes a compression under it."""
    inputs, outputs = 2 * (16 + 8), 16
    def fan_in(d: int) -> int:
        return math.ceil((inputs - 1) / (d - 1)) + outputs - 1

    materialized = 2 * (2 * ROUNDS + SCHEDULE) + outputs - 4
    dataflow = ("a dataflow refinement of MH100w (rank 3): each value written is a function of the values its "
                "instruction reads, and a predicate write is priced as a register write (under today's W1 it is free, "
                "and free predicate chains fold any number of registers into one value read, so every B1 falls to B0). "
                "Every output register depends on all 48 input registers (checked, rank 2), so the first output "
                "written has a cone of ⌈47/(d − 1)⌉ writes and the other 15 outputs lie outside it")
    rungs = [
        ("B0", "1",
         "H32 and DistinctWrites (MH100w): the 16 output registers are distinct, priced writes. MH100w attains it: "
         "an int instruction writes any function, so 16 of them write the digest, and nothing stated in MH100w gives "
         "more", outputs),
        ("B1-isa", "3", f"{dataflow}; sm_90's whole ISA, d = 32 (REDUX, VOTE and MATCH read a value from every lane)",
         fan_in(32)),
        ("B1-5", "3", f"{dataflow}; instructions reading at most 3 registers and 2 predicates, d = 5 (the integer "
         "lane without warp-wide instructions or tensor cores)", fan_in(5)),
        ("B1-3", "3", f"{dataflow}; at most 3 values read, d = 3", fan_in(3)),
        ("B2", "4a",
         "materialization: every program holds each a_t, e_t (t = 1..80), each W_t (t = 16..79) and each output word "
         "in registers (a_80 and e_80 folded into H0 and H4). A claim about every algorithm, a weak form of B3",
         materialized),
        ("B3", "4a (per function checked)",
         "every Σ/σ half at its minimal 4, every Ch/Maj half 1, every add at its fan-in minimum, and no two of them "
         "share work: the honest count", honest),
    ]
    return [{"rung": r, "rank": k, "premise": p, "registers": n, "share_of_honest": n / honest,
             "residual": 1 - n / honest} for r, k, p, n in rungs]


BEST_RANK3 = "B0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="")
    ap.add_argument("--minimal", default="")
    ap.add_argument("--row-kib", type=int, default=256)
    ap.add_argument("--l1-leaves", default="1,4,128,512,1024,16384")
    ap.add_argument("--l1-value", type=int, default=1024)
    ap.add_argument("--l1-row-bytes", default="2048,8192,65536,1048576", help="row-seg row sizes")
    ap.add_argument("--l1-row-leaves", type=int, default=512)
    ap.add_argument("--words-bytes", default="16384,65536",
                    help="the words' sha512/row/v2 chunk sizes; the first is the lead's layout below")
    ap.add_argument("--words-leaves", type=int, default=1 << 20, help="the words' chunks a forward")
    ap.add_argument("--lead-w1", type=float, default=233.0, help="the lead's hashing, W1 a useful MAC")
    ap.add_argument("--a-w1", type=float, default=0.321,
                    help="A's pre-noise hashing, W1 a useful MAC: the values chosen requests can repeat or zero (the "
                         "words depend on the noise)")
    ap.add_argument("--a-leaves", type=int, default=512, help="A's 1 KiB leaves a call")
    ap.add_argument("--card-credit", type=float, default=3449.3,
                    help="a card's credit a compression, priced against R beside 3,532")
    ap.add_argument("--honest-extra-per-compression", type=float, default=0.0,
                    help="registers a compression an honest build writes beyond R (a measured SASS count less R): "
                         "priced as honest-only work")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out: dict = {}
    forms = {"variable chaining value and block": compression(True, [True] * 16),
             "constant chaining value, variable block": compression(False, [True] * 16),
             "variable chaining value, constant block (padding)": compression(True, [False] * 16)}
    out["structural"] = forms
    honest = forms["variable chaining value and block"]["registers"]
    out["chain_pair"] = 2 * honest - CHAIN_PAIR_SAVING
    print("## Structural count (derived)")
    for k, v in forms.items():
        print(f"- {k}: {v['registers']} registers ({v['SHF']} SHF, {v['LOP3']} LOP3, {v['adds']} adds)")
    print(f"- a pair of a chain's blocks in chain form: {out['chain_pair']} registers, {out['chain_pair'] / 2:g} a "
          "compression")
    if args.counts:
        counts = json.loads(Path(args.counts).read_text())
        out["compiled"] = {k: v["compression"]["registers_written"] for k, v in counts.items()}
        print("## Compiled (SASS, sm_90a)")
        for k, v in out["compiled"].items():
            print(f"- {k}: {v:.0f} registers written a compression")
    ands = and_gates()
    bs = bitsliced()
    out["and_gates"], out["bitsliced"] = ands, bs
    print(f"## AND gates a compression: {ands}; bitsliced: {bs['per instance']:.0f} LOP3 an instance "
          f"({bs['per instance'] / honest - 1:+.1%} against {honest})")

    leaf = hm96_leaf(args.row_kib * 1024)
    out["hm96_leaf"] = leaf
    tot = sum(r["n"] * r["registers"] for r in leaf)
    bulk = args.row_kib * 1024 / BLOCK * honest
    print(f"## hm96-sha512 leaf, {args.row_kib} KiB rows: {tot:,} registers ({tot / bulk - 1:+.3%} against "
          f"{args.row_kib} KiB at {honest} a block); the salt PRF (3 compressions) and M·y are the hiding module's")
    for r in leaf:
        print(f"- {r['what']}: {r['n']} × {r['registers']}")

    out["l1"] = []
    print(f"## L1 (hm96-sha512/subtree/v1): one call's hashing, {args.l1_value} B leaves, shared work once a call")
    for n in (int(x) for x in args.l1_leaves.split(",")):
        r = l1_call(n, args.l1_value)
        out["l1"].append(r)
        print(f"- {n} leaves: {r['compressions']:,} compressions, {r['registers']:,} registers, "
              f"{r['per_compression']:.1f} a compression ({r['per_compression'] / 3532 - 1:+.3%} against 3,532 "
              f"chained); data-independent {r['data_independent'] / r['registers']:.2%}; shared once a call "
              f"{r['shared_once_a_call']}; an exact repeat of a leaf's value shares "
              f"{r['leaf_repeat_share']:.2%} of its registers")
    for row in out["l1"][-1]["rows"]:
        print(f"  - {row['what']}: {row['n']} × {row['registers']} ({row['compressions']} compressions, "
              f"{row['per_item']} per item; {', '.join(f'{LEVELS[k]} {v}' for k, v in row['by_level'].items())})")
    out["l1_rows"] = []
    print(f"## L1 with sha512/row-seg/v1 rows (a frame-v3 leaf over each row's row-seg digest), "
          f"{args.l1_row_leaves} rows a call")
    seg = message_cost(_blocks(rowseg_segment(0, 1024, ROW_SEG_BYTES)))
    seg_item = seg["by_level"][ITEM] + seg["by_level"][DATA]
    out["rowseg_segment"] = {**seg, "per_item_registers": seg_item}
    print(f"- one bound 1 KiB segment: {seg_item:,} registers an item over {seg['compressions']} compressions "
          f"({seg_item / seg['compressions']:.1f} a compression), {seg['by_level'][CALL]} once a call "
          f"(sha512_rowseg.cuh's segment_from and seg_pre)")
    for k in (int(x) for x in args.l1_row_bytes.split(",")):
        for bound in (True, False):
            r = l1_call(args.l1_row_leaves, row_bytes=k, bound=bound)
            out["l1_rows"].append(r)
            print(f"- {k:,} B rows, {'bound' if bound else 'constant'} segment prefixes: {r['per_compression']:.1f} a "
                  f"compression ({r['per_compression'] / 3532 - 1:+.3%} against 3,532 chained), "
                  f"{r['registers'] / (k * args.l1_row_leaves):.3f} a row byte; data-independent "
                  f"{r['data_independent'] / r['registers']:.2%}; a constant row {r['constant_value_share']:.2%} of "
                  f"the call; an exact repeat of a row shares {r['leaf_repeat_share']:.2%} of its registers")

    dw = distinct_writes()
    out["distinct_writes"] = dw
    print(f"## Honest's registers against ShaWrites' DistinctOn (checked on {dw['samples']} random inputs): "
          f"{dw['registers']} written, {dw['distinct']} distinct, {dw['constant']} constant, "
          f"{dw['carriers_of_inputs']} an input half plus a constant")

    dep = dependence()
    out["dependence"] = dep
    print(f"## Dependence (checked): {dep['witnessed']} of {dep['pairs']} (output, input) register pairs witnessed in "
          f"{dep['trials']} trial(s); sha512 matches hashlib")
    lad = ladder(honest)
    out["ladder"] = lad
    rung = {r["rung"]: r for r in lad}
    print("## The ladder: registers any program writes a compression")
    for r in lad:
        print(f"- {r['rung']} (rank {r['rank']}): {r['registers']} ({r['share_of_honest']:.2%} of {honest}; "
              f"residual {r['residual']:.4f}): {r['premise']}")

    w1_byte = honest * W1_PER_REG / BLOCK
    out["w1_per_byte"] = w1_byte
    print(f"## T1 (W1 floor {W1_PER_REG} a register; {w1_byte:.1f} W1 a hashed byte)")
    t1 = []
    for vol, bpm in VOLUMES.items():
        hw = bpm * w1_byte
        delta = 0.01 / hw
        row = {"volume": vol, "bytes_per_mac": bpm, "hash_w1_per_mac": hw, "delta": delta,
               "delta_registers": delta * honest, "rungs": {}}
        for r in lad:
            term = r["residual"] * hw
            row["rungs"][r["rung"]] = {"t1_term_share_of_stock": term,
                                       "bytes_per_mac_allowed": 0.01 / (r["residual"] * w1_byte) if r["residual"] else
                                       math.inf}
        t1.append(row)
        print(f"- {vol}: {bpm:.4f} B/MAC, hashing {hw:.1f} W1 a MAC ({hw:.0f}× stock); δ = {delta:.2e} "
              f"({delta * honest:.2f} registers a compression)")
        for k, v in row["rungs"].items():
            print(f"  - {k}: T1 term {v['t1_term_share_of_stock']:.3g}× stock; allows "
                  f"{v['bytes_per_mac_allowed']:.3g} B/MAC")
    out["t1"] = t1

    print("## Frontier: residual reached → bytes a MAC T1 allows")
    frontier = []
    for r in [rung[n]["residual"] for n in ("B0", "B1-isa", "B1-5", "B2")] + [0.1, 0.01, 1e-3, 1e-4,
                                                                             t1[0]["delta"], t1[1]["delta"]]:
        bpm = 0.01 / (r * w1_byte)
        frontier.append({"residual": r, "bytes_per_mac": bpm})
        print(f"- residual {r:.4g}: {bpm:.3g} B/MAC")
    out["frontier"] = frontier

    print(f"## Per monitoring period: the unproved hashing (rung {BEST_RANK3}'s residual, the best at rank ≤ 3 under "
          "today's W1) against T1's 1% of stock")
    pooled = []
    for vol_row in t1:
        hw, res = vol_row["hash_w1_per_mac"], rung[BEST_RANK3]["residual"]
        for scope, rate in (("one H200", H200_SERVED_MACS_PER_S),
                            ("datacenter (100,000 H200)", H200_SERVED_MACS_PER_S * DC_GPUS)):
            for p, s in {"window (600 s)": WINDOW_S, **PERIODS}.items():
                macs = rate * s
                comp = macs * vol_row["bytes_per_mac"] / BLOCK
                pooled.append({"volume": vol_row["volume"], "scope": scope, "period": p, "stock_macs": macs,
                               "compressions": comp, "unproved_w1": res * hw * macs, "t1_budget_w1": 0.01 * macs,
                               "unproved_and": res * comp * ands, "stock_and": macs * AND_PER_MAC})
        macs = WORLD_MACS_PER_YEAR
        comp = macs * vol_row["bytes_per_mac"] / BLOCK
        pooled.append({"volume": vol_row["volume"], "scope": "world", "period": "year", "stock_macs": macs,
                       "compressions": comp, "unproved_w1": res * hw * macs, "t1_budget_w1": 0.01 * macs,
                       "unproved_and": res * comp * ands, "stock_and": macs * AND_PER_MAC})
    for r in pooled:
        print(f"- {r['volume']}, {r['scope']}, {r['period']}: {r['compressions']:.3g} compressions; unproved "
              f"{r['unproved_w1']:.3g} W1 against T1's {r['t1_budget_w1']:.3g} ({r['unproved_and']:.3g} AND gates "
              f"against stock's {r['stock_and']:.3g})")
    out["pooled"] = pooled

    out["l1_words"] = []
    print(f"## L1 words (the hiding module's layout): sha512/row/v2 chunks, each digest a frame-v3 leaf of D_(f,W), prefix "
          f"block hoisted, {args.words_leaves:,} chunks a forward")
    v2 = rowv2_row(1024)
    v2_item = v2["by_level"][ITEM] + v2["by_level"][DATA]
    out["rowv2_chunk"] = {**v2, "per_item_registers": v2_item}
    print(f"- one 1 KiB chunk after its hoisted prefix: {v2_item:,} registers over {v2['per_item']} compressions "
          f"({v2_item / v2['per_item']:.1f} a compression; sha512_rowv2.cuh's chunk)")
    for k in (int(x) for x in args.words_bytes.split(",")):
        r = l1_call(args.words_leaves, row_bytes=k, row_schema="row/v2")
        out["l1_words"].append(r)
        print(f"- {k:,} B chunks: {r['per_compression']:.2f} a compression ({r['per_compression'] / 3532 - 1:+.3%} "
              f"against 3,532 chained), {r['registers'] / (k * args.words_leaves):.3f} a word byte, "
              f"{r['compressions'] / args.words_leaves:.3f} compressions a chunk; data-independent "
              f"{r['data_independent'] / r['registers']:.3%}")
        for row in r["rows"][:2]:
            print(f"  - {row['what']}: {row['registers']:,} registers, {row['per_item']} compressions an item")

    a, w = l1_call(args.a_leaves, 1024), out["l1_words"][0]
    a_frac = args.a_w1 / args.lead_w1
    per_comp = 1 / (a_frac / a["per_compression"] + (1 - a_frac) / w["per_compression"])
    di = a_frac * a["data_independent"] / a["registers"] + (1 - a_frac) * w["data_independent"] / w["registers"]
    delta = 0.01 / args.lead_w1
    over = 3532 / per_comp - 1
    card = args.card_credit / per_comp - 1
    out["sha_writes_l1"] = {"lead_w1_per_mac": args.lead_w1, "a_w1_per_mac": args.a_w1, "a_leaves": args.a_leaves,
                            "a_per_compression": a["per_compression"], "words_bytes": w["row_bytes"],
                            "words_per_compression": w["per_compression"], "per_compression": per_comp,
                            "delta": delta, "delta_registers": delta * per_comp, "over_credit_at_3532": over,
                            "t1_over_credit_at_3532": over * args.lead_w1, "card_credit": args.card_credit,
                            "card_credit_error": card, "data_independent": di, "periods": []}
    print(f"## ShaWrites under L1 at the lead's volume ({args.lead_w1:g} W1 of hashing a useful MAC: A {args.a_w1:g} "
          f"as {args.a_leaves} 1 KiB leaves of D_(f,j) a call, the rest the words as {w['row_bytes']:,} B row/v2 "
          f"chunks)")
    print(f"- R a compression: A {a['per_compression']:.2f}, the words {w['per_compression']:.2f}, the lead "
          f"{per_comp:.2f}; δ = 0.01 / {args.lead_w1:g} = {delta:.3g}, {delta * per_comp:.3f} registers a compression")
    print(f"- crediting 3,532 a compression over-credits {over:.3%} of the hashing: a T1 term of "
          f"{over * args.lead_w1:.3g}× stock against T1's 0.01×")
    print(f"- crediting {args.card_credit:g} a compression {'over' if card > 0 else 'under'}-credits {abs(card):.3%} of "
          f"the hashing: {abs(card) * args.lead_w1:.3g}× stock "
          f"{'to a cheating prover' if card > 0 else 'of honest work uncredited'}")
    print(f"- data-independent {di:.3%} of the hashing: every part of it reads salt_w (D's id, the PRF), so it varies "
          f"on the coin slice and R keeps it")
    a_share = max(a["repeat_share"], a["constant_value_share"])
    mix = 2 * 1024 // 8 * args.a_leaves / a["registers"]
    print(f"- A's value blocks: a repeated leaf value shares {a['leaf_repeat_share']:.2%} of its leaf "
          f"({a['repeat_share']:.2%} of the call if every leaf repeats one), a constant value frees "
          f"{a['leaf_constant_value_share']:.2%} of its leaf ({a['constant_value_share']:.2%} of the call). Their "
          f"schedules are constant on the coin slice, so R drops them: crediting A at honest's count is "
          f"{a_share * args.a_w1:.3g}× stock to a prover who chooses A; a keyed word mixed into each value word "
          f"(2 LOP3 a u64 word) costs honest {mix:.2%} of A, {mix * args.a_w1:.3g}× stock")
    keyed = compression_levels(CALL, [DATA] * 16)
    print(f"- a compression from a per-window midstate (a keyed first block): {keyed.at(ITEM, DATA)} registers a "
          f"digest and {keyed.at(CALL)} once a window (from the IV: {compression(False, [True] * 16)['registers']})")
    honest_only = args.honest_extra_per_compression / per_comp
    if args.honest_extra_per_compression:
        print(f"- an honest build {args.honest_extra_per_compression:g} registers a compression above R: honest-only "
              f"{honest_only:.3g} of the hashing, {honest_only * args.lead_w1:.3g}× stock")
    out["sha_writes_l1"].update({"a_repeat_share": a_share, "t1_repeated_a_rows": a_share * args.a_w1,
                                 "a_keyed_mix_share": mix, "t1_a_keyed_mix": mix * args.a_w1,
                                 "keyed_per_digest": keyed.at(ITEM, DATA), "keyed_once_a_window": keyed.at(CALL),
                                 "honest_only_share": honest_only})
    scopes = [(scope, rate * s, p) for scope, rate in (("one H200", H200_SERVED_MACS_PER_S),
                                                       ("datacenter (100,000 H200)", H200_SERVED_MACS_PER_S * DC_GPUS))
              for p, s in PERIODS.items()]
    scopes += [("world", WORLD_MACS_PER_YEAR * s / PERIODS["year"], p) for p, s in PERIODS.items()]
    for scope, macs, p in scopes:
        regs = macs * args.lead_w1 / W1_PER_REG
        comp = regs / per_comp
        row = {"scope": scope, "period": p, "stock_macs": macs, "compressions": comp, "hash_and": comp * ands,
               "stock_and": macs * AND_PER_MAC, "R": (1 - delta) * regs, "delta_registers": delta * regs,
               "t1_allowance_w1": 0.01 * macs, "over_credit_w1": over * args.lead_w1 * macs,
               "over_credit_and": over * comp * ands, "card_credit_error_w1": card * args.lead_w1 * macs,
               "repeated_a_rows_w1": a_share * args.a_w1 * macs,
               "repeated_a_rows_and": a_share * args.a_w1 / args.lead_w1 * comp * ands,
               "honest_only_w1": honest_only * args.lead_w1 * macs}
        out["sha_writes_l1"]["periods"].append(row)
        print(f"- {scope}, {p}: {comp:.3g} compressions ({row['hash_and']:.3g} AND gates; stock {row['stock_and']:.3g}"
              f"); R = {row['R']:.4g} registers, δ·R = {row['delta_registers']:.3g}; T1 allows "
              f"{row['t1_allowance_w1']:.3g} W1; over-credit at 3,532 {row['over_credit_w1']:.3g} W1 "
              f"({row['over_credit_and']:.3g} AND gates); at {args.card_credit:g} {row['card_credit_error_w1']:+.3g} "
              f"W1; repeated A rows {row['repeated_a_rows_w1']:.3g} W1 ({row['repeated_a_rows_and']:.3g} AND gates)"
              + (f"; honest-only {row['honest_only_w1']:.3g} W1" if args.honest_extra_per_compression else ""))

    if args.minimal:
        out["minimal"] = json.loads(Path(args.minimal).read_text())["summary"]
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
