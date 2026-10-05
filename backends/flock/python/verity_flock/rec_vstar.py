"""The reference for V*'s hashing (hiding model architectures, lane rec-v0): every inner Ligerito opening's ``hm96-sha512/v1``
leaf climbs to the public cap node its coin-derived position names, and every message row opens its commitment.

V* (the outer statement) checks an inner session of `verity_flock.rec_live` without seeing the inner statement. Its hashing
part, which this module states over plain Python:

* **Openings** (PROTOCOL.md §13.3, §13.5, §14). Level ``l`` of a rep's Ligerito opening is ``Q_l`` queries into the tree
  whose cap the prover published in the clear (the coin server's record: level 0's in the rep's binding round and in
  ``Commit``, level ``l >= 1``'s in the round that absorbed it). Query ``j``'s position is ``pos_j = (s_j << (d - c_j)) |
  (coin_j.lo mod 2^(d - c_j))`` over the level's query round's coins, its leaf ``SHA-512(leaf_prefix || b || c)`` for
  ``x = SHA-512(row)`` (the row's elements, 16 bytes each, lo LE then hi LE) under the opening's 192-byte salt
  (``hm96.SHA512``, ``default_key``), and the ``d - c`` siblings climb it (``node(acc, sib)`` at an even index, else
  ``node(sib, acc)``) to ``cap[pos >> (d - c)]``. The rows, salts and siblings are the inner proof's (V*'s witness); the
  positions, depths and cap nodes are public.
* **Message rows**: each round's free message rows open the ``b || c`` the coin server recorded for them before it drew the
  round's coins (`verity_flock.rec_live.commit_rows`).

`check` runs both over a finished inner session (``public.json``, ``private.json`` and the proofs); `queries` lists the
openings in the order the RecOpen gadgets take them, each with its public part (level, position, depth, cap node).
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from verity.commitments import hm96

from . import rec_live as RL

HASH = 64
SALT = 192
#: PcsParams.merkle_hash of an hm96-sha512/v1 tree (PROTOCOL.md §14)
HM96_KIND = 3


class _R:
    def __init__(self, b: bytes):
        self.b, self.i = b, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise ValueError("a truncated proof")
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def f128(self) -> int:
        lo, hi = struct.unpack("<QQ", self.take(16))
        return (hi << 64) | lo

    def vec(self, f, cap: int = 1 << 26) -> list:
        n = self.u64()
        if n > cap:
            raise ValueError(f"a list of {n}")
        return [f() for _ in range(n)]

    def h(self) -> bytes:
        return self.take(HASH)

    def salt(self) -> bytes:
        return self.take(SALT)


@dataclass
class Level:
    """One opened Ligerito level: its tree's cap (the proof's copy), and per query the row, the salt and the siblings."""

    cap: list[bytes]
    rows: list[list[int]]
    salts: list[bytes]
    sibs: list[list[bytes]]

    @property
    def c(self) -> int:
        return (len(self.cap) - 1).bit_length()

    @property
    def depth(self) -> int:
        """``d``: the tree's depth (the cap's ``c`` plus each query's siblings)."""
        return self.c + (len(self.sibs[0]) if self.sibs else 0)


def decode(b: bytes) -> dict:
    """The parts of an hm96-sha512/v1 inner proof (PROTOCOL.md §8.2 with each level's ``opened_salts``) V*'s hashing reads:
    ``m``, ``k0`` and the opened levels in query order (level 0 against ``initial_cap``, then each ``recursive_proofs`` entry
    against the cap before it, then ``final_proof`` against the last ``recursive_caps`` entry)."""
    r = _R(b)
    cap = r.vec(r.h)
    m, _rate, k0 = r.u64(), r.u64(), r.u64()
    r.u32()
    if r.u8():
        r.u64()
    if r.u32() != HM96_KIND:
        raise ValueError("not an hm96-sha512/v1 proof")
    pair = lambda: (r.f128(), r.f128())
    r.vec(r.f128), r.vec(r.f128), r.vec(pair), r.f128(), r.f128(), r.f128(), r.vec(r.u64)
    r.vec(pair), r.vec(r.f128), r.vec(pair), r.vec(r.u64)
    r.vec(lambda: (r.vec(r.f128), r.u64())), r.vec(r.u64)

    def level(cap_):
        rows = r.vec(lambda: r.vec(r.f128))
        flat = r.vec(r.h)
        salts = r.vec(r.salt)
        q = len(rows)
        if q == 0 or len(flat) % q or len(salts) != q:
            raise ValueError("an opening whose rows, siblings and salts disagree")
        w = len(flat) // q
        return Level(cap_, rows, salts, [flat[j * w:(j + 1) * w] for j in range(q)])

    icap = r.vec(r.h)
    if icap != cap:
        raise ValueError("initial_cap is not the commitment's cap")
    levels = [level(icap)]
    rcaps = r.vec(lambda: r.vec(r.h))
    nrec = r.u64()
    for i in range(nrec):
        levels.append(level(rcaps[i]))
    yr = r.vec(r.f128)
    levels.append(level(rcaps[-1]))
    return {"m": m, "k0": k0, "levels": levels, "yr": yr, "rcaps": rcaps}


def strata(Q: int, d: int) -> list[int]:
    """§13.5: ``Q`` as a sum of powers of two, greedily, each ``c_i = min(floor(log2 rem), d)``."""
    out, rem = [], Q
    while rem:
        c = min(rem.bit_length() - 1, d)
        out.append(c)
        rem -= 1 << c
    return out


def positions(Q: int, d: int, coins: list[int]) -> list[int]:
    """§13.5: query ``j``'s position from its coin."""
    if len(coins) != Q:
        raise ValueError(f"{len(coins)} coins for {Q} queries")
    out = []
    for c in strata(Q, d):
        for s in range(1 << c):
            lo = coins[len(out)] & RL.MASK64
            out.append((s << (d - c)) | (lo & ((1 << (d - c)) - 1)))
    return out


def row_bytes(row: list[int]) -> bytes:
    return b"".join(RL.f128_bytes(v) for v in row)


def leaf(row: list[int], salt: bytes) -> bytes:
    """``hm96-sha512/v1``: ``SHA-512(leaf_prefix || b || c)`` for ``x = SHA-512(row)`` and ``salt``."""
    inst = hm96.SHA512
    return inst.tree_leaf(inst.default_key, inst.commit_string(inst.default_key, hashlib.sha512(row_bytes(row)).digest(), salt))


def climb(acc: bytes, pos: int, sibs: list[bytes]) -> bytes:
    for s in sibs:
        acc = hashlib.sha512(acc + s if pos % 2 == 0 else s + acc).digest()
        pos //= 2
    return acc


@dataclass
class Query:
    rep: int
    level: int
    j: int
    pos: int
    depth: int
    c: int
    cap_node: bytes
    row: list[int]
    salt: bytes
    sibs: list[bytes]


def _stream_rounds(public: dict, stream: str) -> list[dict]:
    return [r for r in public["rounds"] if r["stream"] == stream]


def query_rounds(public: dict, stream: str, qs: list[int]) -> list[dict]:
    """The rounds whose coins are the query coins of a rep's levels in order: the rounds with no caps whose coin count is
    the next level's ``Q`` (the last level's also absorbs ``yr``; the verifier's own schedule places them, v0 finds them by
    that shape)."""
    out, li = [], 0
    for r in _stream_rounds(public, stream):
        if li < len(qs) and not r["caps"] and r["n"] == qs[li]:
            out.append(r)
            li += 1
    if li != len(qs):
        raise ValueError(f"{stream}: query rounds for {li} of {len(qs)} levels")
    return out


def public_caps(public: dict, stream: str) -> list[list[bytes]]:
    """Each Ligerito level's cap as the coin server received it: level 0's from the binding round, then each later cap in
    the order the rounds absorbed them (one cap a round). Ligerito absorbs level 0's cap again when it starts; that copy
    must equal the binding round's and is not a level of its own."""
    caps = []
    for r in _stream_rounds(public, stream):
        for c in r["caps"]:
            b = bytes.fromhex(c)
            caps.append([b[i:i + HASH] for i in range(0, len(b), HASH)])
    if len(caps) > 1 and len(caps[1]) == len(caps[0]):
        if caps[1] != caps[0]:
            raise ValueError(f"{stream}: Ligerito's initial cap is not the binding round's")
        del caps[1]
    return caps


def queries(public: dict, proofs: dict[int, bytes], table: str = "circuit") -> list[Query]:
    """Every opening of every rep, in rep, level and query order, with its public part from ``public`` and its witness part
    from the proof."""
    out = []
    for rep, b in sorted(proofs.items()):
        p = decode(b)
        stream = f"{table}/rep{rep}"
        levels = p["levels"]
        caps = public_caps(public, stream)
        if len(caps) != len(levels):
            raise ValueError(f"{stream}: {len(caps)} public caps for {len(levels)} opened levels")
        if public["link"]["roots"].get(table) != b"".join(caps[0]).hex():
            raise ValueError(f"{stream}: the binding round's cap is not the committed root_B")
        rounds = query_rounds(public, stream, [len(lv.rows) for lv in levels])
        for li, (lv, cap, rd) in enumerate(zip(levels, caps, rounds)):
            d, c = lv.depth, (len(cap) - 1).bit_length()
            pos = positions(len(lv.rows), d, [int(x, 16) for x in rd["coins"]])
            for j, ps in enumerate(pos):
                out.append(Query(rep, li, j, ps, d, c, cap[ps >> (d - c)], lv.rows[j], lv.salts[j], lv.sibs[j]))
    return out


def opens(q: Query) -> bool:
    return len(q.sibs) == q.depth - q.c and climb(leaf(q.row, q.salt), q.pos, q.sibs) == q.cap_node


def check(public: dict, private: dict, proofs: dict[int, bytes], table: str = "circuit") -> dict:
    """V*'s hashing over a finished inner session: every opening (`opens`) and every message row's commitment."""
    qs = queries(public, proofs, table)
    bad_q = [(q.rep, q.level, q.j) for q in qs if not opens(q)]
    by_round = {(r["stream"], r["k"]): r for r in public["rounds"]}
    bad_m, rows = [], 0
    for pr in private["rounds"]:
        want = by_round[(pr["stream"], pr["k"])]["commits"]
        got = RL.commit_rows(pr["rows"], [bytes.fromhex(s) for s in pr["salts"]])
        rows += len(got)
        if [g.hex() for g in got] != want:
            bad_m.append((pr["stream"], pr["k"]))
    per_level: dict[str, int] = {}
    for q in qs:
        per_level[f"rep{q.rep}/L{q.level}"] = per_level.get(f"rep{q.rep}/L{q.level}", 0) + 1
    return {"queries": len(qs), "per_level": per_level, "openings_failed": bad_q, "message_rows": rows, "message_rows_failed": bad_m,
            "accepted": not bad_q and not bad_m and rows == public["message_rows"]}


def load(d: Path, table: str = "circuit") -> tuple[dict, dict, dict[int, bytes]]:
    """``public.json``, ``private.json`` and the proofs of an inner session directory (`verity_flock.rec_live.Proxy.write`)."""
    d = Path(d)
    proofs = {r: (d / "session" / f"{table}.rep{r}.bin").read_bytes() for r in (0, 1) if (d / "session" / f"{table}.rep{r}.bin").exists()}
    return json.loads((d / "public.json").read_text()), json.loads((d / "private.json").read_text()), proofs


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("inner", help="an inner session directory (public.json, private.json, session/)")
    a = ap.parse_args(argv)
    res = check(*load(Path(a.inner)))
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0 if res["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
