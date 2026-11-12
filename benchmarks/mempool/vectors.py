"""MemCheck_v1's pool vectors (`verity/ml/circuits/tests/fixtures/memcheck_pool.json`): the RAM pool's log (`pool_ram`) on
the encodings' traces, cut to a window of whole operations, its four memories under one record layout (`LAYOUT`), with the
window's touched list and the fingerprints under fixed coins that pool_ram's own field arithmetic gives.

    python3 vectors.py [--traces DIR] [--positions N] [--out PATH]

A window starts from the memory as it stands, each value read as written at time 0: its first access of an address reads
time 0, its list's initial records are those values and its final records the last written. Each operation keeps all its
positions (`pool_ram.SHAPE`), an unused one zero but for its now, as the circuit's are. The vectors are the RAM pool's own
log, held to vLLM's pool (the replay's `BlockPool`) after every call by `pool_ram.run`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import difftest as DT
import pool_ram as PR

#: the record layout (tag, address, value, time widths): M's addresses and K's values are 128-bit keys, B's value is its six
#: fields (4·NW + REFW + RW = 81 bits), and times are 28 bits
LAYOUT = {"G": 2, "A": 128, "V": 128, "T": PR.GLM.TW}
#: (trace, where its window starts): its first operation, a little before its first eviction, or before its first free
WINDOWS = (("async-prefix-cache.jsonl", "start"), ("async-crush.json.gz", "eviction"), ("sync-crush.json.gz", "free"))
OUT = Path(__file__).resolve().parents[2] / "verity" / "ml" / "circuits" / "tests" / "fixtures" / "memcheck_pool.json"


def address(a: tuple) -> tuple[int, int]:
    """(tag, address) of a pool address: RB's (r, k) is r | k << RW."""
    tag = PR.TAGS[a[0]]
    return (tag, a[1] | a[2] << PR.GLM.RW) if a[0] == "RB" else (tag, a[1])


def value(a: tuple, v) -> int:
    """A pool value as one integer: B's fields (prev, next, cprev, cnext, ref, owner) LSB first, M's (first, last)."""
    S = PR.GLM
    if a[0] == "B":
        x, s = 0, 0
        for f, w in zip(v, (S.NW,) * 4 + (S.REFW, S.RW)):
            assert 0 <= f < 1 << w, (a, v)
            x |= f << s
            s += w
        return x
    if a[0] == "M":
        return v[0] | v[1] << S.NW
    return v


def window(ram: PR.Ram, start: int, positions: int) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """The accesses of whole operations from operation ``start``, at most ``positions`` of them, as rows
    (en, tag, a, v, t, w, now); the touched list's entries (tag, a, v0, v, t), sorted; and the operations' range."""
    rows, cur, first = [], {}, {}
    end = start
    for op in ram.ops[start:]:
        k = PR.SHAPE[op.kind]
        if len(rows) + k > positions:
            break
        for pos in range(k):
            now = op.base + pos
            if pos not in op.at:
                rows.append([0, 0, 0, 0, 0, 0, now])
                continue
            a, v, t, w, at = ram.log[op.at[pos]]
            assert at == now
            if a in cur:
                assert cur[a] == (v, t)
            else:
                first[a], t = v, 0
            cur[a] = (w, now)
            rows.append([1, *address(a), value(a, v), t, value(a, w), now])
        end += 1
    entries = sorted([*address(a), value(a, first[a]), value(a, w), t] for a, (w, t) in cur.items())
    return rows, entries, [start, end]


def element(rec: tuple[int, int, int, int], gamma: int) -> int:
    """Σ γ^i c_i over the record's chunks under `LAYOUT`, by powers of γ (the circuit's is Horner's)."""
    G, A, V = LAYOUT["G"], LAYOUT["A"], LAYOUT["V"]
    tag, a, v, t = rec
    x = tag | a << G | v << (G + A) | t << (G + A + V)
    e, g = 0, 1
    while x:
        e ^= PR.gf_mul(g, x & ((1 << 128) - 1))
        g = PR.gf_mul(g, gamma)
        x >>= 128
    return e


def fingerprints(rows: list, entries: list, alpha: int, gamma: int) -> tuple[int, int]:
    reads = [(r[1], r[2], r[3], r[4]) for r in rows if r[0]] + [(e[0], e[1], e[3], e[4]) for e in entries]
    writes = [(r[1], r[2], r[5], r[6]) for r in rows if r[0]] + [(e[0], e[1], e[2], 0) for e in entries]
    assert sorted(reads) == sorted(writes), "the window's multisets differ"
    out = []
    for side in (reads, writes):
        p = 1
        for rec in side:
            p = PR.gf_mul(p, alpha ^ element(rec, gamma))
        out.append(p)
    return out[0], out[1]


def first(ram: PR.Ram, where: str) -> int:
    """The index of the first operation that evicts a cached block (an alloc that reads M), or of the first free."""
    if where == "eviction":
        return next(o for o, op in enumerate(ram.ops) if op.kind == "alloc" and 4 in op.at)
    return next(o for o, op in enumerate(ram.ops) if op.kind == "free")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", type=Path, default=DT.FIXTURES)
    ap.add_argument("--positions", type=int, default=800)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    out = {"layout": LAYOUT, "windows": []}
    for i, (name, where) in enumerate(WINDOWS):
        eng, reqs, admit = DT.load(a.traces / name)
        ram = PR.run(eng, reqs, admit=admit).pool.ram
        start = 0 if where == "start" else max(first(ram, where) - 24, 0)
        rows, entries, ops = window(ram, start, a.positions)
        rng = random.Random(i)
        alpha, gamma = rng.getrandbits(128), rng.getrandbits(128)
        R, W = fingerprints(rows, entries, alpha, gamma)
        assert R == W
        kinds = sorted({ram.ops[o].kind for o in range(*ops)})
        evictions = sum(ram.ops[o].kind == "alloc" and 4 in ram.ops[o].at for o in range(*ops))
        out["windows"].append({"trace": name, "ops": ops, "kinds": kinds, "evictions": evictions, "alpha": alpha,
                               "gamma": gamma, "R": R, "accesses": rows, "entries": entries})
        print(f"{name}: operations {ops}, {sum(r[0] for r in rows)} accesses of {len(rows)} positions, "
              f"{len(entries)} touched, kinds {kinds}, {evictions} evictions")
    a.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
