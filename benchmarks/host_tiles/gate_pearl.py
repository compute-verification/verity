"""Gate: `pearl_c_cpu`'s k-slice accumulators equal `pearl_c_u.chain(..., trace, device=SM120)` bit for bit.

Every atom's local word (k_s = 32) and every promotion's total (k_s = 128) of every output, from the vector kernel and from
its scalar exact path (`gate DIR scalar`, the C statement of `pearl_c_u.grouped`), on:
  formed     the registered scheme's operands, `PearlC(k_min=32, forming="v1", device=SM120, hashing="h3")` over the
             test vectors' rows (`pearl_c_vectors.rows_f32`), A′ and B̃ as `encode` forms them;
  gauss      the same scheme over Gaussian FP32 rows (what a weight and an activation look like);
  uniform    uniformly random finite E4M3 codes (the slow path's worst case);
  extreme    codes drawn from the format's corners (±0, the smallest subnormal, ±448, mid-range), rows and their negations
             (cancellation), and mostly-zero rows;
with an odd m (the one-row tail).  `python gate_pearl.py BINARY [POOL]` prints one JSON line per case and exits nonzero on
any mismatch; POOL, if given, receives the gauss case's formed codes (A′ then B̃) for `bench ... file:POOL`.
"""

from __future__ import annotations

import json
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verity/core/protocols/pouw/tests"))

from verity.core.protocols.pouw.protocol import Tile  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c_u as U  # noqa: E402
from verity.core.protocols.pouw.schemes.pearl_c_device import SM120  # noqa: E402

import pearl_c_vectors as V  # noqa: E402


def formed(label: str, m: int, k: int, n: int, gauss: bool) -> tuple[list[list[int]], list[list[int]]]:
    scheme = C.PearlC(k_min=32, tile=(m, n), forming="v1", device=SM120, hashing="h3")
    ctx = V.ctx_of("host-tiles/" + label, (m, k, n), 7, 2, wide=True)
    scheme.check(ctx.matmul)
    tile = Tile(tuple(range(m)), tuple(range(n)))
    if gauss:
        rng = random.Random(label)
        f32 = lambda: [struct.unpack("<I", struct.pack("<f", rng.gauss(0, 1)))[0] for _ in range(k)]  # noqa: E731
        a, b = [C.f32_row(f32()) for _ in range(m)], [C.f32_row(f32()) for _ in range(n)]
    else:
        a, b = V.rows_f32("host-tiles/" + label, "A", m, k), V.rows_f32("host-tiles/" + label, "B", n, k)
    ops = scheme.encode(ctx, scheme.noise(ctx), a, b, tile)
    return [list(r) for r in ops.a.codes], [list(r) for r in ops.b.codes]


def finite(rng: random.Random) -> int:
    while True:
        c = rng.randrange(256)
        if c & 0x7F != 0x7F:
            return c


def uniform(label: str, m: int, k: int, n: int):
    rng = random.Random(label)
    return [[finite(rng) for _ in range(k)] for _ in range(m)], [[finite(rng) for _ in range(k)] for _ in range(n)]


CORNERS = [0x00, 0x80, 0x01, 0x81, 0x07, 0x08, 0x7E, 0xFE, 0x40, 0xC0, 0x77, 0xF7, 0x38, 0xB8, 0x30, 0x48, 0x5F]


def extreme(label: str, m: int, k: int, n: int):
    rng = random.Random(label)
    corner = lambda: rng.choice(CORNERS) if rng.random() < 0.7 else finite(rng)  # noqa: E731
    a = [[corner() for _ in range(k)] for _ in range(m)]
    b = [[corner() for _ in range(k)] for _ in range(n)]
    for j in range(0, n, 4):                       # cancellation: B̃_j = −A′_i (and its sign-flipped halves)
        b[j] = [c ^ 0x80 for c in a[j % m]]
    for j in range(1, n, 8):                       # mostly zero
        b[j] = [c if rng.random() < 0.05 else 0 for c in b[j]]
    for j in range(2, n, 8):                       # one huge product per atom over small ones
        b[j] = [0x7E if l % 32 == 5 else (0x01 if rng.random() < 0.5 else 0x81) for l in range(k)]
    a[m - 1] = [c if rng.random() < 0.1 else 0 for c in a[m - 1]]
    return a, b


def case(binary: str, label: str, a: list[list[int]], b: list[list[int]]) -> list[dict]:
    m, k, n = len(a), len(a[0]), len(b)
    atoms_ref = [[0] * (m * n) for _ in range(k // 32)]
    totals_ref = [[0] * (m * n) for _ in range(k // 128)]
    for i in range(m):
        for j in range(n):
            tr: dict = {}
            U.chain(a[i], b[j], trace=tr, device=SM120)
            for t, w in enumerate(tr["atoms"]):
                atoms_ref[t][i * n + j] = w
            for t, w in enumerate(tr["totals"]):
                totals_ref[t][i * n + j] = w
    out = []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        p.joinpath("shape.bin").write_bytes(struct.pack("<3i", m, k, n))
        p.joinpath("a.bin").write_bytes(bytes(c for row in a for c in row))
        p.joinpath("b.bin").write_bytes(bytes(c for row in b for c in row))
        for mode in ("vector", "scalar"):
            r = subprocess.run([binary, "gate", d] + (["scalar"] if mode == "scalar" else []), check=True,
                               capture_output=True, text=True)
            info = json.loads(r.stdout)
            raw_a, raw_t = p.joinpath("atoms.bin").read_bytes(), p.joinpath("totals.bin").read_bytes()
            got_a = struct.unpack(f"<{len(raw_a) // 4}I", raw_a)
            got_t = struct.unpack(f"<{len(raw_t) // 4}I", raw_t)
            want_a = [w for row in atoms_ref for w in row]
            want_t = [w for row in totals_ref for w in row]
            bad = (len(got_a) != len(want_a)) + (len(got_t) != len(want_t))
            bad += sum(x != y for x, y in zip(got_a, want_a)) + sum(x != y for x, y in zip(got_t, want_t))
            out.append({"gate": "pearl_c_cpu", "case": label, "mode": mode, "m": m, "k": k, "n": n,
                        "atom_words": len(want_a), "total_words": len(want_t), "slow_lanes": info["slow_lanes"],
                        "distinct_atoms": len(set(want_a)), "mismatches": bad})
    return out


if __name__ == "__main__":
    binary = sys.argv[1]
    rows = []
    fa, fb = formed("formed", 16, 512, 64, False)
    rows += case(binary, "formed", fa, fb)
    ga, gb = formed("gauss", 16, 512, 64, True)
    rows += case(binary, "gauss", ga, gb)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_bytes(bytes(c for row in ga + gb for c in row))
    rows += case(binary, "uniform", *uniform("uniform", 5, 256, 64))
    rows += case(binary, "extreme", *extreme("extreme", 7, 512, 64))
    for r in rows:
        print(json.dumps(r))
    sys.exit(0 if all(r["mismatches"] == 0 for r in rows) else 1)
