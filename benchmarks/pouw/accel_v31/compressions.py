"""SHA-512 compressions a no-fold call hashes, by component, as v31's Fill hashes them (`v31.cu`, from commit_gpu.cuh's
midstates), from `dither.Shape`'s layout: per call and per T MAC of h200_step's models at each leaf row size, and with
`--check` the same counts taken from dither.py's reference hashing on small calls.

A call's components (compressions on the device; a leaf's row is R bytes):
  t_S    a filler's T_S rows (a served call's O implies its T_S): R / 128 + 1 a leaf, from P's midstate
  t_s    the T_s rows for s < S
  O      O's rows (a served call's)
  salt   a leaf's salt: prf_salt's y (3) and hm96's c = H(salt_prefix ‖ y) from its midstate (2)
  hide   a leaf's hm96 tree leaf from leaf_prefix's midstate (2) and its frame-v3 leaf (2)
  node   the tree's nodes over the leaves padded to a power of two, 2 each from the level's midstate
  pad    the padding leaves, 2 each
  bits   d_A and d_B: a block per 512 inner indices a row
  noise  V and U: L′ / 16 blocks a row, when the call commits a T_s (v31's Fill skips them when W = 0)
  perm   π_τ's words: K / 16 blocks, likewise
  apad   a filler's pad A: m ceil(K / 64) blocks (fill-bench's `ovl_pad`), none when A is given
  seeds  the call's seeds, two blocks a stream (on the host)
  A      A's operand root (`dither.operand_tree`): its m rows of K bytes as leaves, each salted and hidden, and the tree;
         v31 doesn't hash it on the device

A row is a multiple of 16 bytes dividing a tile-segment's 32 KiB in dither.py (`Shape`) and in v31's Fill (`bad_shape`).
A longer row is priced here as the same bytes cut into fewer rows, which crosses tile-segments: a format change.

    python3 compressions.py [--rows 8192,16384,32768,65536,262144] [--L 4096] [--lp 512] [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h200_step import MODELS

from verity.ml.protocols.pouw.circuit import dither
from verity.ml.commitments import merkle, scheme
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

M = 16384
NAMES = {"8b": {(6144, 4096): "qkv", (4096, 4096): "o", (28672, 4096): "up", (4096, 14336): "down"},
         "mqa16k52": {(16384, 16384): "qo", (128, 16384): "kv", (65536, 16384): "up", (16384, 65536): "down",
                      (256, 16384): "head"}}
#: a call's kinds: (O committed, A's source): a filler with its pad, a filler on a given A (no A hashed), a served call
KINDS = {"filler": (False, "pad"), "filler-a": (False, "given"), "served": (True, "root")}
ROWS, DATA = ("t_S", "t_s", "O"), ("t_S", "t_s", "O")
LEAF = ("salt", "hide", "node", "pad")
COINS = ("bits", "noise", "perm", "apad", "seeds")
PARTS = (*ROWS, *LEAF, *COINS, "A")
#: compressions on the device a hash (commit_gpu.cuh): salt_prf, salt_digest, leaf_from_digest's h, frame_first and
#: frame_second, node_ni, pad_nb, and a coin block (`prg`)
SALT_Y, SALT_C, LEAF_H, FRAME, NODE, PAD, PRG = 3, 2, 2, 2, 2, 2, 1


def blocks(n: int) -> int:
    """SHA-512's compressions of an n-byte message: its bytes, 0x80 and the 16-byte length, in 128-byte blocks."""
    return (n + 17 + 127) // 128


def row_blocks(row: int) -> int:
    """A row's compressions past P's midstate (v31's `row_blocks`)."""
    return blocks(row)


def legal(row: int) -> bool:
    return row > 0 and row % 16 == 0 and dither.TILE_WORD_BYTES % row == 0


def tree(leaves: int) -> tuple[int, int]:
    """(nodes, padding leaves) of a tree over `leaves` leaves padded to a power of two."""
    width = 1 << (leaves - 1).bit_length()
    return width - 1, width - leaves


def operand(m: int, k: int) -> int:
    """A's operand root: m rows of K bytes, each a hiding leaf, and its tree."""
    nodes, pads = tree(m)
    return m * (row_blocks(k) + SALT_Y + SALT_C + LEAF_H + FRAME) + NODE * nodes + PAD * pads


def components(m: int, n: int, k: int, lp: int = 512, L: int = 4096, row: int = 16384, kind: str = "filler") -> dict:
    """A call's compressions by component (`PARTS`) and its leaves, at any row (`legal` or priced as a format change)."""
    o, a = KINDS[kind]
    sh = dither.Shape(m, n, k, lp, L, o=o)
    seg = sh.tiles * dither.TILE_WORD_BYTES
    by = {"t_S": 0 if o else seg, "t_s": (sh.S - 1) * seg, "O": sh.tiles * dither.TILE_O_BYTES if o else 0}
    nl = {x: -(-b // row) for x, b in by.items()}
    leaves = sum(nl.values())
    nodes, pads = tree(leaves)
    noised = sh.W > 0
    c = {x: v * row_blocks(row) for x, v in nl.items()}
    c |= {"salt": (SALT_Y + SALT_C) * leaves, "hide": (LEAF_H + FRAME) * leaves, "node": NODE * nodes, "pad": PAD * pads,
          "bits": PRG * (m + n) * -(-k // 512), "noise": PRG * (m + n) * lp // 16 if noised else 0,
          "perm": PRG * k // 16 if noised else 0, "apad": PRG * m * -(-k // 64) if a == "pad" else 0,
          "seeds": 2 * (len(dither.STREAMS) + (not o)), "A": operand(m, k) if a == "root" else 0}
    return {"parts": c, "leaves": leaves, "nodes": nodes, "pads": pads, "bytes": sum(by.values()), "S": sh.S, "W": sh.W,
            "macs": m * n * k}


def step(model: str, **kw) -> dict:
    """`components` of a model's step at M = 16,384 (h200_step's calls), per call shape and summed."""
    calls = {NAMES[model][nk]: (cnt, components(M, *nk, **kw)) for nk, cnt in MODELS[model].items()}
    tot = Counter()
    for cnt, c in calls.values():
        tot.update({x: cnt * v for x, v in c["parts"].items()})
    keys = ("leaves", "nodes", "pads", "bytes", "macs")
    return {"calls": calls, "parts": dict(tot), **{x: sum(cnt * c[x] for cnt, c in calls.values()) for x in keys}}


def per_tmac(parts: dict, macs: int) -> dict:
    """Each component's compressions per T MAC, in millions."""
    return {x: v / macs * 1e6 for x, v in parts.items()}


def total(parts: dict, a: bool = False) -> int:
    return sum(v for x, v in parts.items() if a or x != "A")


# -- the reference's count --------------------------------------------------------------------------------------------------

_REAL = hashlib.sha512


class _Count:
    """A counting SHA-512: every digest adds its compressions to the category on top of `cat`."""

    counts: Counter = Counter()
    cat: list[str] = ["other"]

    def __init__(self, data: bytes = b""):
        self.h, self.n = _REAL(data), len(data)

    def update(self, data: bytes) -> None:
        self.h.update(data)
        self.n += len(data)

    def digest(self) -> bytes:
        _Count.counts[_Count.cat[-1]] += blocks(self.n)
        return self.h.digest()


def _tagged(cat: str, f):
    def g(*a, **kw):
        _Count.cat.append(cat)
        try:
            return f(*a, **kw)
        finally:
            _Count.cat.pop()
    return g


class _Hashlib:
    sha512 = _Count


def counted() -> dict:
    """Patch dither.py's hashing to count compressions by category; returns what to restore. A coin block is counted once
    however often the reference recomputes it (the device computes it once)."""
    hm = type(dither.INST)
    saved = {"sha": scheme._SHA["sha512"], "hashlib": dither.hashlib, "row": dither.sha512_row_v2_digest,
             "prf": dither.prf_salt, "prg": dither.prg, "seeds": dither.seeds, "fill_seeds": dither.fill_seeds,
             "salt_c": hm.salt_digest, "leaf_h": hm.tree_leaf, "frame": merkle.CommitmentDomain.leaf,
             "node": merkle.CommitmentDomain.node, "pad": merkle.CommitmentDomain.pad, "domain": dither.Shape.domain,
             "operand_domain": dither.operand_domain}
    scheme._SHA["sha512"], dither.hashlib = _Count, _Hashlib
    dither.Shape.domain = _tagged("domain", saved["domain"])
    dither.operand_domain = _tagged("domain", saved["operand_domain"])
    dither.sha512_row_v2_digest = _tagged("row", saved["row"])
    dither.prf_salt = _tagged("salt_y", saved["prf"])
    dither.seeds, dither.fill_seeds = _tagged("seeds", saved["seeds"]), _tagged("seeds", saved["fill_seeds"])
    hm.salt_digest, hm.tree_leaf = _tagged("salt_c", saved["salt_c"]), _tagged("leaf_h", saved["leaf_h"])
    merkle.CommitmentDomain.leaf = _tagged("frame", saved["frame"])
    merkle.CommitmentDomain.node = _tagged("node", saved["node"])
    merkle.CommitmentDomain.pad = _tagged("pad", saved["pad"])
    memo: dict = {}

    def prg(seed: bytes, tag: str, idx: int) -> bytes:
        if (seed, tag, idx) not in memo:
            memo[seed, tag, idx] = _tagged("coin:" + tag, saved["prg"])(seed, tag, idx)
        return memo[seed, tag, idx]
    dither.prg = prg
    return saved


def restore(saved: dict) -> None:
    hm = type(dither.INST)
    scheme._SHA["sha512"], dither.hashlib = saved["sha"], saved["hashlib"]
    dither.sha512_row_v2_digest, dither.prf_salt, dither.prg = saved["row"], saved["prf"], saved["prg"]
    dither.seeds, dither.fill_seeds = saved["seeds"], saved["fill_seeds"]
    hm.salt_digest, hm.tree_leaf = saved["salt_c"], saved["leaf_h"]
    merkle.CommitmentDomain.leaf, merkle.CommitmentDomain.node = saved["frame"], saved["node"]
    merkle.CommitmentDomain.pad = saved["pad"]
    dither.Shape.domain, dither.operand_domain = saved["domain"], saved["operand_domain"]


def midstates(row: int) -> dict[str, int]:
    """The reference's compressions a hash beyond the device's: the constant first blocks the device starts past (P, the
    salt prefix, H(key) and the leaf prefix, a node's level block)."""
    key = dither.INST.check_key(dither.KEY)
    return {"row": len(sha512_row_v2_prefix(8 * row)) // 128, "salt_c": len(dither.INST.salt_prefix) // 128,
            "leaf_h": blocks(len(key)) + len(dither.INST.leaf_prefix(key)) // 128, "node": 1}


def reference(m: int, n: int, k: int, lp: int, L: int, row: int, kind: str) -> tuple[Counter, Counter]:
    """(the reference's compressions by category, the model's of the same hashes) for one call through dither.py: its seeds,
    the coins the device draws, its values (any bytes: the layout fixes the hashing) under the value root, and A's
    operand root for a served call. A domain's id, a constant of its shape, isn't counted."""
    o, a = KINDS[kind]
    sh = dither.Shape(m, n, k, lp, L, row, o=o)
    c = components(m, n, k, lp, L, row, kind)
    p, mid = c["parts"], midstates(row)
    saved = counted()
    _Count.counts = Counter()
    try:
        sd = dither.seeds(bytes(64), bytes(64), 0) if o else dither.fill_seeds(bytes(64), bytes(64))
        dither.dithers(sh, sd, "A", range(m))
        dither.dithers(sh, sd, "B", range(n))
        if sh.W:
            dither.noise_a(sh, sd, range(m))
            dither.noise_b(sh, sd, range(n))
        if a == "pad":
            dither.pad_rows(sd, k, range(m))
        dither.commit(sh, bytes(range(256)) * (sh.leaves * row // 256), dither.salts(bytes(64), 0))
        got = Counter(_Count.counts)
        _Count.counts = Counter()
        if a == "root":
            dither.operand_tree("A", [[(i + j) % 251 - 125 for j in range(k)] for i in range(m)], dither.salts(bytes(64), 1))
            got["A"] = sum(v for x, v in _Count.counts.items() if x != "domain")
        got.pop("domain", None)
    finally:
        restore(saved)
    leaves, nodes, pads = c["leaves"], c["nodes"], c["pads"]
    want = Counter({"seeds": p["seeds"], "coin:dA": m * -(-k // 512), "coin:dB": n * -(-k // 512),
                    "row": sum(p[x] for x in ROWS) + mid["row"] * leaves, "salt_y": SALT_Y * leaves,
                    "salt_c": (SALT_C + mid["salt_c"]) * leaves, "leaf_h": (LEAF_H + mid["leaf_h"]) * leaves,
                    "frame": FRAME * leaves, "node": (NODE + mid["node"]) * nodes, "pad": PAD * pads})
    if sh.W:
        want.update({"coin:V": m * lp // 16, "coin:U": n * lp // 16, "coin:P": k // 16})
    if a == "pad":
        want["coin:pad"] = p["apad"]
    if a == "root":
        an, ap = tree(m)
        want["A"] = p["A"] + m * (mid["row"] + mid["salt_c"] + mid["leaf_h"]) + mid["node"] * an
    return +got, +want


CHECKS = ((128, 256, 768, 128, 256, 2048, "filler"), (256, 128, 1024, 128, 512, 4096, "served"),
          (128, 128, 512, 128, 512, 1024, "served"), (256, 256, 1024, 128, 512, 32768, "filler-a"),
          (384, 256, 1024, 128, 512, 32768, "served"), (128, 384, 640, 128, 256, 8192, "filler"))


def check() -> bool:
    ok = True
    for case in CHECKS:
        got, want = reference(*case)
        same = got == want
        ok &= same
        m, n, k, lp, L, row, kind = case
        diff = {x: (got[x], want[x]) for x in sorted(set(got) | set(want)) if got[x] != want[x]}
        print(f"{kind} {m}x{n}x{k} L' {lp} L {L} row {row}: {'equal' if same else 'DIFFER ' + repr(diff)}; "
              f"{sum(got.values())} reference compressions")
    print(f"compressions check: {'ACCEPT' if ok else 'REFUSE'}: {sum(1 for _ in CHECKS)} calls' reference hashing "
          f"{'equals' if ok else 'differs from'} the model plus the device's midstates")
    return ok


# -- the tables -------------------------------------------------------------------------------------------------------------

def _row_name(row: int) -> str:
    return f"{row // 1024} KiB" + ("" if legal(row) else " (format change)")


def breakdown(model: str, kind: str, row: int, L: int, lp: int) -> str:
    st = step(model, lp=lp, L=L, row=row, kind=kind)
    cols = [x for x in PARTS if st["parts"].get(x)]
    out = [f"\n{model} {kind}, row {_row_name(row)}, L {L}, L' {lp}: M compressions per T MAC\n",
           "| call | count | S | W | leaves | " + " | ".join(cols) + " | total | total with A |",
           "|---|---|---|---|---|" + "---|" * (len(cols) + 2)]
    for name, (cnt, c) in st["calls"].items():
        t = per_tmac(c["parts"], c["macs"])
        out.append(f"| {name} | {cnt} | {c['S']} | {c['W']} | {c['leaves']} | " + " | ".join(f"{t[x]:.3f}" for x in cols)
                   + f" | {total(t):.3f} | {total(t, True):.3f} |")
    t = per_tmac(st["parts"], st["macs"])
    out.append(f"| step (MAC-weighted) | | | | {st['leaves']} | " + " | ".join(f"{t[x]:.3f}" for x in cols)
               + f" | {total(t):.3f} | {total(t, True):.3f} |")
    data = st["bytes"] / 128 / st["macs"] * 1e6
    out.append(f"\nsummary: {model} {kind} row {row} L {L} L' {lp}: {total(t):.3f} M compressions per T MAC "
               f"({total(st['parts'])} a step of {st['macs']} MACs; {total(t, True):.3f} with A's operand root); the "
               f"committed words' bytes alone {data:.3f}; {st['leaves']} leaves, {st['nodes']} nodes, {st['pads']} pads")
    return "\n".join(out)


def rows_table(model: str, rows: list[int], L: int, lp: int) -> str:
    out = [f"\n{model}, L {L}, L' {lp}: M compressions per T MAC by row\n",
           "| row | legal | kind | words | leaf overhead (row pad, salt, hide, node, pad) | coins | total | total with A |",
           "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for row in rows:
            st = step(model, lp=lp, L=L, row=row, kind=kind)
            t, data = per_tmac(st["parts"], st["macs"]), st["bytes"] / 128 / st["macs"] * 1e6
            words = sum(t[x] for x in ROWS)
            over = words - data + sum(t[x] for x in LEAF)
            out.append(f"| {row} | {'yes' if legal(row) else 'no'} | {kind} | {data:.3f} | {over:.3f} | "
                       f"{sum(t[x] for x in COINS):.3f} | {total(t):.3f} | {total(t, True):.3f} |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", default="8192,16384,32768,65536,262144")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--lp", type=int, default=512)
    ap.add_argument("--check", action="store_true", help="count dither.py's reference hashing on small calls")
    a = ap.parse_args(argv[1:])
    if a.check:
        return 0 if check() else 1
    rows = [int(x) for x in a.rows.split(",")]
    for model in MODELS:
        for kind in KINDS:
            print(breakdown(model, kind, 8192 if kind != "served" else 16384, a.L, a.lp))
        print(rows_table(model, rows, a.L, a.lp))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
