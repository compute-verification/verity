"""Gate: `ncp_amx`'s snapshots equal `schemes.ncp.checked_route_u` bit for bit, both forms, at W = 0 and at random W.

Builds X, Y with the scheme's own `stack` (test instance `NCP(k_min=64, n_min=16)`), writes X + β (u8), Y's columns (s8)
and c₀ = −β·Σ Y (int32), runs `ncp_amx gate DIR KS` for KS in 16, 32, 64, 128 and 0 (final only), and compares snapshot s
with checkpoint t = KS/16·(s+1) − 1 (the last is A·Bᵀ).  m = 16 runs the 1 x 4 form, m = 32 and 64 the 2 x 2.  Prints one
JSON line per case and exits nonzero on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.core.protocols.pouw.schemes import ncp as N

M32 = (1 << 32) - 1


def ctx_of(label: str, shape) -> Context:
    h = lambda s: hashlib.sha256((label + s).encode()).digest()
    return Context(salt=h("salt"), matmul=Matmul(index=7, shape=Shape(*shape), weight=2), root_a=h("ra"), root_b=h("rb"))


def case(binary: str, label: str, m: int, k: int, n: int, w_zero: bool) -> list[dict]:
    rng = random.Random(label)
    a = [[rng.randint(-64, 63) for _ in range(k)] for _ in range(m)]
    b = [[0] * k for _ in range(n)] if w_zero else [[rng.randint(-64, 63) for _ in range(k)] for _ in range(n)]
    scheme = N.NCP(k_min=64, n_min=16)
    ctx = ctx_of(label, (m, k, n))
    scheme.check(ctx.matmul)
    pi, sigma = scheme.noise(ctx)
    ops = N.stack(ctx, pi, sigma, a, b, Tile(tuple(range(m)), tuple(range(n))))
    chk = N.checked_route_u(ops)
    K = 3 * k
    out = []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        p.joinpath("shape.bin").write_bytes(struct.pack("<3i", m, K, n))
        p.joinpath("x.bin").write_bytes(bytes(v + N.BETA for row in ops.x for v in row))
        p.joinpath("y.bin").write_bytes(bytes(v & 0xFF for col in ops.y for v in col))
        c0 = [(-N.BETA * sum(col)) & M32 for col in ops.y]
        p.joinpath("c0.bin").write_bytes(struct.pack(f"<{n}I", *c0))
        for ks in (16, 32, 64, 128, 0):
            subprocess.run([binary, "gate", d, str(ks)], check=True, capture_output=True)
            raw = p.joinpath(f"out_{ks}.bin").read_bytes()
            got = struct.unpack(f"<{len(raw) // 4}I", raw)
            nsnap = K // ks if ks else 1
            bad = len(got) != nsnap * m * n
            for s in range(nsnap if not bad else 0):
                t = (ks // 16) * (s + 1) - 1 if ks else K // 16 - 1
                for i in range(m):
                    for j in range(n):
                        bad += got[(s * m + i) * n + j] != chk[i][j][t]
            out.append({"gate": "ncp_amx", "case": label, "W": "zero" if w_zero else "random", "m": m, "k": k, "n": n,
                        "ks": ks, "form": "1x4" if m % 32 else "2x2", "snapshots": nsnap, "words": nsnap * m * n,
                        "mismatches": int(bad)})
    return out


if __name__ == "__main__":
    binary = sys.argv[1]
    rows = []
    for m in (16, 32, 64):
        rows += case(binary, f"gate-w0-m{m}", m, 256, 128, True) + case(binary, f"gate-wr-m{m}", m, 256, 128, False)
    for r in rows:
        print(json.dumps(r))
    sys.exit(0 if all(r["mismatches"] == 0 for r in rows) else 1)
