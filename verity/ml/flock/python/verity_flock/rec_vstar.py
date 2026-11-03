"""The reference for V*'s hashing (hiding model architectures, lane rec-v0): every inner Ligerito opening's leaf climbs to
the node of its tree's top that its coin-derived position names, every top opens the salted commitment the coin server
received, and every message row opens its commitment.

V* (the outer statement) checks an inner session of `verity_flock.rec_live` without seeing the inner statement or a tree's
top. Its hashing part, which this module states over plain Python:

* **Tops** (`tops`). Each tree's top (its Ligerito cap, ``2^c`` nodes) is the prover's (``private.json``); the coin server
  received each node's salted commitment (`verity_flock.rec_live.commit_top`): level 0's at ``Commit`` (``root_B``, the one
  tree both reps share), level ``l >= 1``'s in the round that absorbed it. A top whose nodes and salts do not open those
  commitments is refused (``FIREWALL-REFUSED: tree top``). The stacked table R has no root at ``Commit``: its level 0 is
  its binding round's top, which must be the top registered at staging (`registered`: the ``stacked`` entry of the
  circuit registration, ``verity/registration/v2``, `rec_live.REGISTRATION`, whose digest the statement's META binds).
* **Openings** (PROTOCOL.md §13.3, §13.5). Level ``l`` of a rep's Ligerito opening is ``Q_l`` queries into that tree. Query
  ``j``'s position is ``pos_j = (s_j << (d - c_j)) | (coin_j.lo mod 2^(d - c_j))`` over the level's query round's coins,
  its leaf ``SHA-512(row)`` (the row's elements, 16 bytes each, lo LE then hi LE: the statement's
  ``flock-leaf/sha512-unsalted``, `verity_flock.circuit.LEAF_SCHEME_PLAIN`), and the ``d - c`` siblings climb it
  (``node(acc, sib)`` at an even index, else ``node(sib, acc)``) to ``top[pos >> (d - c)]``. The rows and siblings are the
  inner proof's and the top's nodes the prover's (V*'s witness); the positions and depths are public. A proof with
  ``hm96-sha512/v1`` leaves (``SHA-512(leaf_prefix || b || c)`` for ``x = SHA-512(row)`` under the opening's 192-byte
  salt: rec-step2's sessions and the fixture) decodes and climbs too; the RecOpen gadgets open plain leaves alone.
* **Message rows**: each round's free message rows open the ``b || c`` the coin server recorded for them before it flipped the
  round's coins (`verity_flock.rec_live.commit_rows`).

`check` runs all three over a finished inner session (``public.json``, ``private.json`` and the proofs); `queries` lists the
openings in the order the RecOpen gadgets take them, each with its public part (level, position, depth) and the top's node
it climbs to.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from verity.ml.commitments import hm96

from . import rec_live as RL

HASH = 64
SALT = 192
#: PcsParams.merkle_hash (bincode's variant index of flock-hash's HashKind) of a tree with plain SHA-512 leaves
PLAIN_KIND = 2
#: and of an hm96-sha512/v1 tree (PROTOCOL.md §14)
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
    """One opened Ligerito level: its tree's cap (the proof's copy), and per query the row, the salt (None for a plain
    leaf) and the siblings."""

    cap: list[bytes]
    rows: list[list[int]]
    salts: list[bytes | None]
    sibs: list[list[bytes]]

    @property
    def c(self) -> int:
        return (len(self.cap) - 1).bit_length()

    @property
    def depth(self) -> int:
        """``d``: the tree's depth (the cap's ``c`` plus each query's siblings)."""
        return self.c + (len(self.sibs[0]) if self.sibs else 0)


def decode(b: bytes) -> dict:
    """The parts of an inner proof of the SHA-512 build (PROTOCOL.md §8.2 with each level's ``opened_salts``: 192 bytes a
    query for ``hm96-sha512/v1`` leaves, empty for plain ones) V*'s hashing reads: ``m``, ``k0``, ``plain`` and the opened
    levels in query order (level 0 against ``initial_cap``, then each ``recursive_proofs`` entry against the cap before it,
    then ``final_proof`` against the last ``recursive_caps`` entry)."""
    r = _R(b)
    cap = r.vec(r.h)
    m, _rate, k0 = r.u64(), r.u64(), r.u64()
    r.u32()
    if r.u8():
        r.u64()
    kind = r.u32()
    if kind not in (PLAIN_KIND, HM96_KIND):
        raise ValueError(f"merkle_hash {kind}: neither plain SHA-512 nor hm96-sha512/v1 leaves")
    plain = kind == PLAIN_KIND
    pair = lambda: (r.f128(), r.f128())
    r.vec(r.f128), r.vec(r.f128), r.vec(pair), r.f128(), r.f128(), r.f128(), r.vec(r.u64)
    r.vec(pair), r.vec(r.f128), r.vec(pair), r.vec(r.u64)
    r.vec(lambda: (r.vec(r.f128), r.u64())), r.vec(r.u64)

    def level(cap_):
        rows = r.vec(lambda: r.vec(r.f128))
        flat = r.vec(r.h)
        salts = r.vec(r.salt)
        q = len(rows)
        if q == 0 or len(flat) % q or len(salts) != (0 if plain else q):
            raise ValueError("an opening whose rows, siblings and salts disagree")
        w = len(flat) // q
        return Level(cap_, rows, [None] * q if plain else salts, [flat[j * w:(j + 1) * w] for j in range(q)])

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
    return {"m": m, "k0": k0, "plain": plain, "levels": levels, "yr": yr, "rcaps": rcaps}


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


def leaf(row: list[int], salt: bytes | None = None) -> bytes:
    """A plain leaf, ``SHA-512(row)``; with ``salt``, ``hm96-sha512/v1``'s: ``SHA-512(leaf_prefix || b || c)`` for
    ``x = SHA-512(row)``."""
    x = hashlib.sha512(row_bytes(row)).digest()
    if salt is None:
        return x
    inst = hm96.SHA512
    return inst.tree_leaf(inst.default_key, inst.commit_string(inst.default_key, x, salt))


def climb(acc: bytes, pos: int, sibs: list[bytes]) -> bytes:
    for s in sibs:
        acc = hashlib.sha512(acc + s if pos % 2 == 0 else s + acc).digest()
        pos //= 2
    return acc


@dataclass
class Query:
    """One opening: its public part (rep, level, query, position, depth, the top's ``c``) and its witness (the top's node it
    climbs to, the row, the salt of an hm96 leaf or None, the siblings)."""

    rep: int
    level: int
    j: int
    pos: int
    depth: int
    c: int
    cap_node: bytes
    row: list[int]
    salt: bytes | None
    sibs: list[bytes]


def _stream_rounds(public: dict, stream: str) -> list[dict]:
    return [r for r in public["rounds"] if r["stream"] == stream]


def query_rounds(public: dict, stream: str, qs: list[int]) -> list[dict]:
    """The rounds whose coins are the query coins of a rep's levels in order: the rounds that absorb no top (salted or a
    level-0 copy) whose coin count is the next level's ``Q`` (the last level's also absorbs ``yr``; the verifier's own
    schedule places them, v0 finds them by that shape)."""
    out, li = [], 0
    for r in _stream_rounds(public, stream):
        if li < len(qs) and not r["tops"] and not r["level0"] and r["n"] == qs[li]:
            out.append(r)
            li += 1
    if li != len(qs):
        raise ValueError(f"{stream}: query rounds for {li} of {len(qs)} levels")
    return out


@dataclass(frozen=True)
class Top:
    """A tree's top as the ZK server committed it: its nodes and their salts (the prover's, ``private.json``) and each node's
    ``b || c`` (the coin server's record, public)."""

    nodes: tuple[bytes, ...]
    salts: tuple[bytes, ...]
    bcs: tuple[bytes, ...]


def opened_top(opening: dict, bcs: list[str], what: str) -> Top:
    """The prover's ``opening`` of a top (its nodes and salts), when they open ``bcs``, the ``b || c`` a node the coin server
    received; refused by name otherwise."""
    nodes = [bytes.fromhex(x) for x in opening["nodes"]]
    salts = [bytes.fromhex(x) for x in opening["salts"]]
    if (not nodes or len(nodes) != len(bcs) or len(salts) != len(nodes) or any(len(n) != HASH for n in nodes)
            or [b.hex() for b in RL.commit_top(b"".join(nodes), salts)] != bcs):
        raise ValueError(f"FIREWALL-REFUSED: tree top: {what} does not open its commitment")
    return Top(tuple(nodes), tuple(salts), tuple(bytes.fromhex(b) for b in bcs))


def tops(public: dict, private: dict, stream: str, table: str = "circuit") -> list[Top]:
    """Each Ligerito level's top of a rep's stream, opened (`opened_top`) against the coin server's record: level 0's the
    table's ``root_B`` from ``Commit`` (for a table with none there, the stacked table R, its binding round's salted top),
    then each later salted top in the order the stream's rounds absorbed them. A round's copies of a level-0 top
    (``level0``: the binding round's, Ligerito's first absorb) are not levels of their own."""
    if public.get("tops") != RL.TOPS or private.get("tops") != RL.TOPS:
        raise ValueError(RL.refused("tree top: the record's tops are not salted (rec_live.TOPS)"))
    out = []
    if table in public["link"]["roots"]:
        out.append(opened_top(private["roots"][table], public["link"]["roots"][table], f"{table}'s root_B"))
    prv = {(r["stream"], r["k"]): r for r in private["rounds"]}
    for r in _stream_rounds(public, stream):
        got = prv.get((stream, r["k"]), {}).get("tops", [])
        if len(got) != len(r["tops"]):
            raise ValueError(f"FIREWALL-REFUSED: tree top: {stream} round {r['k']} commits {len(r['tops'])} tops, the prover "
                             f"opens {len(got)}")
        out += [opened_top(o, bcs, f"{stream} round {r['k']}'s top {i}") for i, (o, bcs) in enumerate(zip(got, r["tops"]))]
    return out


def registered(public: dict, registration: dict | None, bound: str | None, table: str = RL.STACKED) -> None:
    """A table with no root at ``Commit`` (R) is bound by the circuit registration alone (`rec_live.register_top`, at
    staging): the record must be the one whose digest the statement's META binds (``bound``, one-stage
    ``registration.digest``), and its registered entry named ``table`` the entry its stream's binding round's salted top makes
    (`rec_live.top_entry` under the record's ``program``: root, leaves, words and domain); refused by name otherwise."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R

    stream = f"{table}/rep0"
    first = next((r for r in _stream_rounds(public, stream) if r["k"] == 0), None)
    if bound is None or not isinstance(registration, dict) or R.digest(registration) != bound:
        raise ValueError(RL.refused(f"registration: the record is not the circuit registration META binds ({bound})"))
    entry = RL.registered_entry(registration, table)
    if entry is None:
        raise ValueError(RL.refused(f"registration: {table} has no root at Commit and no registered top"))
    if first is None or not first["tops"] or RL.top_entry(registration["program"], table,
                                                          [bytes.fromhex(b) for b in first["tops"][0]]) != entry:
        raise ValueError(RL.refused(f"registration: {stream}'s binding round's top is not {table}'s registered top"))


def reps(public: dict) -> list[int]:
    """The reps the session's ``hello`` declares; each has a stream and a proof, or V* refuses the session."""
    return list(range(public["hello"]["reps"]))


def queries(public: dict, private: dict, proofs: dict[int, bytes], table: str = "circuit") -> list[Query]:
    """Every opening of every rep, in rep, level and query order, with its public part from ``public``, the top's node it
    climbs to from ``private`` (`tops`) and the rest of its witness from the proof. Refused by name unless every declared rep
    (`reps`; the stacked table R, `rec_live.STACKED`, its one) has its stream and its proof, and nothing else does."""
    want = [0] if table == RL.STACKED else reps(public)
    if not want or sorted(proofs) != want or any(f"{table}/rep{r}" not in public["streams"] for r in want):
        raise ValueError(f"FIREWALL-REFUSED: proofs: the session declares {table}'s reps {want}; it has streams "
                         f"{sorted(public['streams'])} and proofs of reps {sorted(proofs)}")
    out = []
    for rep, b in sorted(proofs.items()):
        p = decode(b)
        stream = f"{table}/rep{rep}"
        levels = p["levels"]
        ts = tops(public, private, stream, table)
        if len(ts) != len(levels):
            raise ValueError(f"{stream}: {len(ts)} committed tops for {len(levels)} opened levels")
        if any(len(t.nodes) != len(lv.cap) for t, lv in zip(ts, levels)):
            raise ValueError(f"{stream}: a committed top is not the width of the proof's cap")
        rounds = query_rounds(public, stream, [len(lv.rows) for lv in levels])
        for li, (lv, t, rd) in enumerate(zip(levels, ts, rounds)):
            top = t.nodes
            d, c = lv.depth, (len(top) - 1).bit_length()
            pos = positions(len(lv.rows), d, [int(x, 16) for x in rd["coins"]])
            for j, ps in enumerate(pos):
                out.append(Query(rep, li, j, ps, d, c, top[ps >> (d - c)], lv.rows[j], lv.salts[j], lv.sibs[j]))
    return out


def opens(q: Query) -> bool:
    return len(q.sibs) == q.depth - q.c and climb(leaf(q.row, q.salt), q.pos, q.sibs) == q.cap_node


def check(public: dict, private: dict, proofs: dict[int, bytes], table: str = "circuit") -> dict:
    """V*'s hashing over a finished inner session: every top (`tops`, refused by name when one does not open), every
    opening (`opens`) and every message row's commitment."""
    qs = queries(public, private, proofs, table)
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
    return {"queries": len(qs), "per_level": per_level, "plain_leaves": all(q.salt is None for q in qs), "openings_failed": bad_q,
            "message_rows": rows, "message_rows_failed": bad_m,
            "accepted": bool(qs) and not bad_q and not bad_m and rows == public["message_rows"]}


def load(d: Path, table: str = "circuit") -> tuple[dict, dict, dict[int, bytes]]:
    """``public.json``, ``private.json`` and the proofs of an inner session directory (`verity_flock.rec_live.Proxy.write`):
    every ``<table>.rep<r>.bin`` there; `queries` refuses a missing or an undeclared rep."""
    d = Path(d)
    pre = f"{table}.rep"
    proofs = {int(p.name[len(pre):-4]): p.read_bytes() for p in (d / "session").glob(f"{pre}*.bin") if p.name[len(pre):-4].isdigit()}
    return json.loads((d / "public.json").read_text()), json.loads((d / "private.json").read_text()), proofs


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("inner", help="an inner session directory (public.json, private.json, session/)")
    ap.add_argument("--circuit", help="the inner statement's circuit.txt, whose META binds the circuit registration (a "
                    "session with the stacked table R is refused without it)")
    a = ap.parse_args(argv)
    try:
        public, private, proofs = load(Path(a.inner))
        res = check(public, private, proofs)
        if f"{RL.STACKED}/rep0" in public["streams"]:
            reg = Path(a.inner) / RL.REGISTRATION
            meta = json.loads(Path(a.circuit).read_text().split("\n", 2)[1][5:]) if a.circuit else {}
            registered(public, json.loads(reg.read_text()) if reg.exists() else None, meta.get("registration"))
            res["stacked"] = check(public, private, load(Path(a.inner), RL.STACKED)[2], RL.STACKED)
            res["accepted"] = res["accepted"] and res["stacked"]["accepted"]
    except ValueError as e:
        if not str(e).startswith("FIREWALL-REFUSED"):
            raise
        res = {"accepted": False, "refused": str(e)}
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0 if res["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
