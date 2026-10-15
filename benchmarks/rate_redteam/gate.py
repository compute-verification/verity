"""The co-residency probe's key file and its gate against `codec.py` (lane ra-rate-redteam).

    gate.py key   --k K --out KEYFILE                       k, c_0 … c_{k-1}, tau and tstep, decimal
    gate.py check --what W --k K --key KEYFILE --got FILE   every codec chain of `co gate` recomputed with codec.rho

A chain starts at canon((i · 0x9E3779B97F4A7C15 mod 2^64) & P) with tweak t = tau · i, steps through encode (W = 1:
x ↦ ρ(x + c_j + t) for j = 0 … k − 1) or decode (W = 0: x ↦ ρ⁻¹(x) − c_j − t for j = k − 1 … 0), and advances t by
tstep = tau · STRIDE after each step, as `co.cu`'s kernel does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "word_codec"))
import codec  # noqa: E402

SEED = bytes(range(32))
STRIDE = 1 << 20
PHI = 0x9E3779B97F4A7C15


def chain(what: int, k: codec.Key, i: int, steps: int, tstep: int) -> int:
    y = (i * PHI % (1 << 64)) & codec.P
    y %= codec.P
    t = k.tau * i % codec.P
    for _ in range(steps):
        if what:
            for c in k.c:
                y = codec.rho((y + c + t) % codec.P)
        else:
            for c in reversed(k.c):
                y = (codec.rho_inv(y) - c - t) % codec.P
        t = (t + tstep) % codec.P
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["key", "check"])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--what", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--key")
    ap.add_argument("--got")
    a = ap.parse_args()
    k = codec.key(SEED, a.k)
    tstep = k.tau * STRIDE % codec.P
    if a.mode == "key":
        Path(a.out).write_text(f"{a.k}\n{' '.join(map(str, k.c))}\n{k.tau}\n{tstep}\n")
        return 0
    lines = Path(a.got).read_text().split("\n")
    steps = int(next(x for x in lines if x.startswith("steps")).split()[1])
    rows = [tuple(map(int, x.split())) for x in lines if x and not x.startswith("steps")]
    bad = [(i, y) for i, y in rows if chain(a.what, k, i, steps, tstep) != y]
    print(f"check what={a.what} k={a.k}: {len(rows)} codec chains, {len(bad)} differ from codec.py")
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
