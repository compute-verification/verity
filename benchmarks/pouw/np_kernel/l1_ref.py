"""L1 + (a)'s seed path from the tree's Python, for l1_check.cpp to agree with (l1.h's header states the construction).

    python3 benchmarks/pouw/np_kernel/l1_ref.py SEED ROWS K CALL

One JSON line with the fields l1_check prints; frame-v3-sha512 from verity.ml.commitments.merkle, hm96-sha512/v1 from
verity.ml.commitments.hm96, the call seed and noise blocks from pouw's seeds.
"""

from __future__ import annotations

import hashlib
import json
import sys

from verity.core.protocols.pouw.circuit import seeds
from verity.ml.commitments import hm96
from verity.ml.commitments.indexed import RangeU64Domain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels

M64 = (1 << 64) - 1
SEG = 1024


def mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def u64(v: int) -> bytes:
    return v.to_bytes(8, "big")


def sha(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()


def main() -> None:
    seed, rows, k, call = (int(a) for a in sys.argv[1:5])
    forward, calls = 3, 64
    salt, ks, root_b = (sha(t + u64(seed)) for t in (b"np-kernel/salt", b"np-kernel/ks", b"np-kernel/root-b"))
    a = bytes(mix64(seed * 1000003 + i) & 0xFF for i in range(rows * k))
    n = len(a) // SEG
    dom = CommitmentDomain(sha(salt + u64(forward) + u64(call)), -1, RangeU64Domain(0, n), hash="sha512")
    leaves = [dom.leaf(r, r, "pouw/a-seg/v1", a[SEG * r:SEG * (r + 1)]) for r in range(n)]
    x = tree_levels(dom, leaves)[-1][0]
    kw = sha(ks + salt)
    y = b"".join(sha(kw + u64(forward) + u64(call) + u64(r)) for r in range(3))
    inst = hm96.SHA512
    key = inst.default_key
    bc = inst.commit_string(key, x, y)
    h = inst.tree_leaf(key, bc)
    root_a = h
    s = seeds.call_seed(salt, forward * calls + call, root_a, root_b)
    print(json.dumps({"leaves": n, "dom": dom.domain_id.hex(), "leaf0": leaves[0].hex(), "leaf_last": leaves[-1].hex(),
                      "x": x.hex(), "y": y.hex(), "bc": bc.hex(), "h": h.hex(), "root_a": root_a.hex(), "seed": s.hex(),
                      "noise0": seeds.noise_block(s, 0).hex(), "noise1": seeds.noise_block(s, 1).hex()}))


if __name__ == "__main__":
    main()
