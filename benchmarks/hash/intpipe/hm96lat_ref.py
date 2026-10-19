"""hm96lat.cu's k-ary trees against a CPU reference: a FORMAT CHANGE (hypothetical; whether a frame tree may be k-ary is
proofs' decision).

    python3 hm96lat_ref.py check DIR    one `hm96lat verify` k-ary dump (DIR/meta.json, DIR/levels.bin): one JSON line,
                                        exit 1 on any difference

The leaves and pads are hm96_ref's (Verity's hm96-sha512/row-seg/v1 in frame-v3-sha512); only the nodes change: a node of
level d (its children's depth) and index i over children c_0 .. c_(k-1) is SHA-512(frame "node" did uint(d) uint(i) c_0 ..
c_(k-1)), the binary frame node with k children; the width is k^depth, padded as the binary tree pads. Up to 2048 leaves
the check recomputes the whole tree; above, sampled leaves and pads, and sampled nodes of every level from the dump's own
children, the root included. Run from the source tree's root with PYTHONPATH=.
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hm96_ref as R


def node_k(f: R.Format, level: int, index: int, children: list[bytes]) -> bytes:
    return f.p.h(R.message(b"node", f.did, R.uint(level), R.uint(index), *children))


def check(d: str) -> dict:
    with open(os.path.join(d, "meta.json")) as fh:
        meta = json.load(fh)
    n, k = meta["leaves"], meta["k"]
    R.set_geometry(meta["leaf_bytes"], meta["seg_bytes"])
    f = R.Format(R.Prim("sha512"), n)
    width, depth = 1, 0
    while width < n:
        width, depth = width * k, depth + 1
    sizes = [width // k ** l for l in range(depth + 1)]
    off = [sum(sizes[:l]) for l in range(depth + 2)]
    with open(os.path.join(d, "levels.bin"), "rb") as fh:
        raw = fh.read()
    if len(raw) != 64 * off[-1]:
        return {"ok": False, "error": f"size {len(raw)} for {off[-1]} nodes"}
    dig = lambda l, i: f.p.digest_of(raw[64 * (off[l] + i):64 * (off[l] + i + 1)])
    bad: list[str] = []
    rng = random.Random(n)
    if n <= 2048:
        lv = [f.leaf(pos, R.fill_rows(pos))[1] for pos in range(n)] + [f.pad(r) for r in range(n, width)]
        levels = [lv]
        for l in range(depth):
            lv = [node_k(f, l, i, lv[k * i:k * (i + 1)]) for i in range(len(lv) // k)]
            levels.append(lv)
        for l in range(depth + 1):
            bad += [f"level {l} node {i}" for i, want in enumerate(levels[l]) if dig(l, i) != want]
        checked = off[-1]
    else:
        picks = sorted({0, 1, 255, 256, n // 2, n - 2, n - 1, *(q for q in (65535, 65536, n - 3) if q < n),
                        *rng.sample(range(n), 24)})
        bad += [f"leaf {p}" for p in picks if dig(0, p) != f.leaf(p, R.fill_rows(p))[1]]
        pads = sorted({n, width - 1, *rng.sample(range(n, width), min(16, width - n))}) if width > n else []
        bad += [f"pad {r}" for r in pads if dig(0, r) != f.pad(r)]
        checked = len(picks) + len(pads)
        for l in range(1, depth + 1):
            m, c = sizes[l], -(-n // k ** l)
            for i in sorted({0, m - 1, c - 1, *(q for q in (c,) if q < m), *rng.sample(range(m), min(m, 8))}):
                checked += 1
                if dig(l, i) != node_k(f, l - 1, i, [dig(l - 1, k * i + j) for j in range(k)]):
                    bad.append(f"level {l} node {i}")
    return {"ok": not bad, "k": k, "leaves": n, "depth": depth, "full": n <= 2048, "nodes_checked": checked,
            "bad_count": len(bad), "bad": bad[:8], "root": dig(depth, 0).hex(), "format": "hypothetical k-ary frame tree"}


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "check":
        r = check(argv[2])
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
