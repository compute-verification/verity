"""Cross-check data for the direct AIR against WP2.

    python crosscheck.py gen  N SEED units.json      random finite units: (c, a, b, y = tc_dot) and the WP2 hints
    python crosscheck.py cmp  units.json hints.json  compare the Rust hint columns (direct-unit crosscheck) with
                                                     WP2's ``gen_step`` hints, flattened in the AIR's order, and
                                                     run ``check_step`` on the Rust hints re-shaped into WP2's layout

Run from the repo root with ``uv run --package verity-numerical python archive/direct/tools/crosscheck.py ...``.
"""

from __future__ import annotations

import json
import random
import sys

from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot
from verity_numerical.checker import REAL, check_step, gen_step

P = REAL
K = 16


def finite16(rng: random.Random) -> int:
    while True:
        w = rng.getrandbits(16)
        if (w >> 7) & 0xFF != 0xFF:
            return w


def finite32(rng: random.Random) -> int:
    while True:
        w = rng.getrandbits(32)
        if (w >> 23) & 0xFF != 0xFF:
            return w


def structured(rng: random.Random) -> tuple[int, list[int], list[int]]:
    """Bias towards the interesting regimes: zeros, subnormals, cancellations, large shifts, saturation."""
    kind = rng.randrange(6)
    if kind == 0:
        return (
            finite32(rng),
            [finite16(rng) for _ in range(K)],
            [finite16(rng) for _ in range(K)],
        )
    if kind == 1:  # many zero operands
        return (
            finite32(rng),
            [finite16(rng) if rng.random() < 0.3 else 0 for _ in range(K)],
            [finite16(rng) for _ in range(K)],
        )
    if kind == 2:  # subnormal / tiny state, normal products
        c = rng.getrandbits(23) | (rng.getrandbits(1) << 31)
        return c, [finite16(rng) for _ in range(K)], [finite16(rng) for _ in range(K)]
    if kind == 3:  # cancellation: b_i = -a_i pattern with a shared partner
        a = [finite16(rng) for _ in range(K)]
        b = [finite16(rng) for _ in range(K)]
        for i in range(0, K, 2):
            a[i + 1] = a[i] ^ 0x8000
            b[i + 1] = b[i]
        return 0, a, b
    if kind == 4:  # huge exponents: saturation likely
        e = lambda: (0, rng.randrange(240, 255), rng.getrandbits(7))
        mk = lambda s, x, f: (s << 15) | (x << 7) | f
        return (
            finite32(rng),
            [mk(rng.getrandbits(1), *e()[1:]) for _ in range(K)],
            [mk(rng.getrandbits(1), *e()[1:]) for _ in range(K)],
        )
    # large state, small products: everything shifted out
    c = (
        (rng.getrandbits(1) << 31)
        | (rng.randrange(200, 255) << 23)
        | rng.getrandbits(23)
    )
    mk = lambda: (
        (rng.getrandbits(1) << 15) | (rng.randrange(1, 60) << 7) | rng.getrandbits(7)
    )
    return c, [mk() for _ in range(K)], [mk() for _ in range(K)]


def flatten(
    c: int, a: list[int], b: list[int], y: int, h: dict
) -> list[tuple[str, int]]:
    """WP2's hints in the AIR's hint order (program.rs), including the public words."""
    out: list[tuple[str, int]] = [("x.c", c)]
    out += [(f"x.a[{i}]", a[i]) for i in range(K)]
    out += [(f"x.b[{i}]", b[i]) for i in range(K)]
    out.append(("y", y))
    out += [(f"acc.{k}", h["acc"][k]) for k in ("s", "t", "f", "nz", "z")]
    for i in range(K):
        out += [(f"a[{i}].{k}", h["a"][i][k]) for k in ("s", "t", "f", "nz")]
    for i in range(K):
        out += [(f"b[{i}].{k}", h["b"][i][k]) for k in ("s", "t", "f", "nz")]
    for i in range(K):
        out.append((f"prod[{i}].z", h["prod"][i]["z"]))
    for g, hg in enumerate(h["groups"]):
        t = f"g{g}"
        out.append((f"{t}.max.E", hg["max"]["E"]))
        out += [(f"{t}.max.sel[{i}]", s) for i, s in enumerate(hg["max"]["sel"])]
        out += [(f"{t}.align_state.{k}", hg["acc_align"][k]) for k in ("q", "r")]
        for i, hq in enumerate(hg["prod_align"]):
            out += [(f"{t}.align_prod[{i}].{k}", hq[k]) for k in ("q", "r")]
        out += [(f"{t}.sum.{k}", hg["sum"][k]) for k in ("sgn", "z")]
        out.append((f"{t}.lead.l", hg["lead"]["l"]))
        out += [(f"{t}.norm.{k}", hg["norm"][k]) for k in ("clamp", "M", "r")]
        out += [(f"{t}.out.{k}", hg["out"][k]) for k in ("z", "ovf")]
    out += [(f"pack.{k}", h["pack"][k]) for k in ("hidden", "low")]
    return out


def unflatten(flat: list[tuple[str, int]]) -> dict:
    d = dict(flat)
    h = {
        "acc": {k: d[f"acc.{k}"] for k in ("s", "t", "f", "nz", "z")},
        "a": [{k: d[f"a[{i}].{k}"] for k in ("s", "t", "f", "nz")} for i in range(K)],
        "b": [{k: d[f"b[{i}].{k}"] for k in ("s", "t", "f", "nz")} for i in range(K)],
        "prod": [{"z": d[f"prod[{i}].z"]} for i in range(K)],
        "groups": [],
        "pack": {k: d[f"pack.{k}"] for k in ("hidden", "low")},
    }
    for g, size in enumerate(P.groups):
        t = f"g{g}"
        h["groups"].append(
            {
                "max": {
                    "E": d[f"{t}.max.E"],
                    "sel": [d[f"{t}.max.sel[{i}]"] for i in range(size + 2)],
                },
                "acc_align": {k: d[f"{t}.align_state.{k}"] for k in ("q", "r")},
                "prod_align": [
                    {k: d[f"{t}.align_prod[{i}].{k}"] for k in ("q", "r")}
                    for i in range(size)
                ],
                "sum": {k: d[f"{t}.sum.{k}"] for k in ("sgn", "z")},
                "lead": {"l": d[f"{t}.lead.l"]},
                "norm": {k: d[f"{t}.norm.{k}"] for k in ("clamp", "M", "r")},
                "out": {k: d[f"{t}.out.{k}"] for k in ("z", "ovf")},
            }
        )
    return h


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "gen":
        n, seed, path = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        rng = random.Random(seed)
        units = []
        for _ in range(n):
            c, a, b = structured(rng)
            y = tc_dot(AMPERE_BF16_M16N8K16, c, a, b)
            h = gen_step(P, c, a, b)
            check_step(P, c, a, b, y, h)
            units.append({"c": c, "a": a, "b": b, "y": y})
        with open(path, "w") as f:
            json.dump(units, f)
        print(
            f"wrote {n} units (y = tc_dot, WP2 check_step accepts gen_step) to {path}"
        )
    elif cmd == "cmp":
        with open(sys.argv[2]) as f:
            units = json.load(f)
        with open(sys.argv[3]) as f:
            hints = json.load(f)
        assert len(units) == len(hints), (len(units), len(hints))
        n_equal = n_checked = 0
        for u, hr in zip(units, hints):
            c, a, b, y = u["c"], u["a"], u["b"], u["y"]
            want = flatten(c, a, b, y, gen_step(P, c, a, b))
            got = [(k, int(v)) for k, v in hr]
            if got != want:
                diffs = [(w, g) for w, g in zip(want, got) if w != g][:5]
                raise SystemExit(
                    f"hint mismatch on unit c={c:#x}: {diffs} (lengths {len(want)} vs {len(got)})"
                )
            n_equal += 1
            check_step(
                P, c, a, b, y, unflatten(got)
            )  # WP2 accepts the Rust hints in WP2's own layout
            n_checked += 1
        print(
            f"{n_equal} units: Rust hint columns == WP2 gen_step hints; WP2 check_step accepts the Rust hints on {n_checked}"
        )
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
