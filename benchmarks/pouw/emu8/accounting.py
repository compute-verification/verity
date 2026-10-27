"""EMU8's security accounting: which of the honest emulation's FP8 limb products a cheater still needs, and its routes.

  python accounting.py [--json OUT]

Dither commits every k32 prefix word of Ã B̃ with Ã = A + R (R fresh bits) and B̃ = B + S (S[t, j] = U[j, t mod 32]).
The prover picks A and B, so a cheater takes A = B = 0 (or any one-limb A, B = 0): Ã = R, B̃ = S, and each word is the
cross term alone. `needed` counts, for each scheme (k3: the products hh', mm', ll'; k4: the four pairings) and each range
of U, the limb products that are not identically zero over every (R, U) and not a fixed multiple of another product
(a cheater computes such a product once): the products a cheater who keeps the scheme's structure still runs. `fp4_route`
is the cheaper route on FP4 tensor cores: U written as two E2M1 terms under per-block scales, adaptively per block, with a
third term only where a block's values aren't covered.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emu8  # noqa: E402

U_RANGES = {"±1..±64": [v for v in range(-64, 65) if v], "±1..±16": [v for v in range(-16, 17) if v],
            "one limb [-8, 7]": [v for v in range(-8, 8) if v]}
SCHEMES = {"k3": [(0, 0), (1, 1), (2, 2)], "k4": [(0, 0), (0, 1), (1, 0), (1, 1)]}  # (A plane, B plane) a product


def needed(scheme: str, urange: str, a_values=(0, 1)) -> dict:
    """The scheme's limb products a cheater still needs with Ã in `a_values` (A = 0: Ã = R) and B̃ = S, U in `urange`."""
    a = np.array(a_values)
    u = np.array(U_RANGES[urange])
    A, U = np.meshgrid(a, u, indexing="ij")
    pa, pb = emu8.planes(A, scheme, 0), emu8.planes(U, scheme, 1)
    prods = [(pa[x] * pb[y]).ravel() for x, y in SCHEMES[scheme]]
    kept: list[np.ndarray] = []
    for p in prods:
        if not p.any():
            continue
        if any(np.linalg.matrix_rank(np.stack([p, q]).astype(float)) < 2 for q in kept):
            continue
        kept.append(p)
    e4m3_exact = all(int(v) in emu8.E4M3_CODE for v in u)
    return {"scheme": scheme, "u": urange, "honest": len(prods), "needed": len(kept),
            "share": len(kept) / len(prods), "S_is_e4m3": e4m3_exact}


E2M1 = sorted({s * v for v in (0, 0.5, 1, 1.5, 2, 3, 4, 6) for s in (1, -1)})


def _ue4m3() -> list[float]:
    return sorted({emu8.e4m3_value(c) for c in range(1, 127) if emu8.e4m3_value(c)})


def fp4_route(umax: int = 64, block: int = 16, blocks: int = 2000, seed: int = 1) -> dict:
    """NVFP4 (UE4M3 scales, 16-element blocks): the share of a row's blocks whose U values two E2M1 terms cover under one
    pair of block scales chosen for that block, and the FP4 products a MAC that gives (two, and a third where a 32-step
    segment has an uncovered block). Global (one scale pair for every block) coverage for comparison."""
    scales = _ue4m3()
    targets = [v for v in range(-umax, umax + 1) if v]
    covers = []
    for s1, s2 in itertools.combinations_with_replacement(scales, 2):
        vals = {a * s1 + b * s2 for a in E2M1 for b in E2M1}
        covers.append(frozenset(t for t in targets if t in vals))
    best_global = max(len(c) for c in covers)
    rng = np.random.default_rng(seed)
    covered = 0
    for _ in range(blocks):
        vals = set(int(x) for x in rng.integers(1, umax + 1, block) * rng.choice([-1, 1], block))
        covered += any(vals <= c for c in covers)
    p = covered / blocks
    third = 1 - p ** (32 // block)
    return {"u": f"±1..±{umax}", "block": block, "global_best_covered": best_global, "targets": len(targets),
            "block_covered": p, "fp4_products_a_mac": 2 + third}


def epsilon(t_honest_over_tN: float, adversary_units: float) -> float:
    """Neekon's ε: honest time over the adversary's, both in units of t_N = N / r8 (one FP8 product a MAC)."""
    return t_honest_over_tN / adversary_units


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    rows = [needed(s, n) for s in SCHEMES for n in U_RANGES]
    rows += [dict(needed(s, "±1..±64", a_values=tuple(range(-8, 8))), A="one-limb Ã in [-8, 7]") for s in SCHEMES]
    fp4 = fp4_route()
    out = {"needed": rows, "fp4_route": fp4}
    for r in rows:
        print(f"{r['scheme']} U {r['u']:>16} {r.get('A', 'Ã = R'):>22}: needs {r['needed']} of {r['honest']} "
              f"(share {r['share']:.3f}){' [S E4M3-exact: one product suffices]' if r['S_is_e4m3'] else ''}")
    print(f"FP4 route, U {fp4['u']}: 2 E2M1 terms cover {fp4['global_best_covered']}/{fp4['targets']} values under one scale "
          f"pair; per-block scales cover {fp4['block_covered']:.3f} of 16-element blocks -> {fp4['fp4_products_a_mac']:.2f} "
          f"FP4 products a MAC")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
