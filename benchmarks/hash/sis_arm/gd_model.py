"""GD-MODEL's J with each SIS opening in place of `row_ands`: the proving side of the swap on the shared model.

    git fetch origin cursor/compute-research-gd-model-6eb4 && git --work-tree=DIR checkout FETCH_HEAD -- golden/
    python3 gd_model.py DIR OUT.json

GD-MODEL (`golden/model.py`) prices a window's proving from its draws, each unit's ANDs plus `row_ands` of every row it
opens. This replaces `row_ands(b)` by an SIS row's opening at n = 224 (`ands.py`, `ands.json`'s per-byte counts, the
same ANDs-only model as 58,120 a compression) and leaves the rest of the model as it is. The serve side is not swapped:
`j_effect.py` puts SIS's serve change within 0.43 of J at T 8,192, against prove terms of 10^4 to 10^5 here at one step a
window.

A row's opening, b bytes, n = 224 (digest 4n = 896 B, D = A x over Z 7n = 1,568 B, r = 4n + 33 = 929 words):
- `sha512`: 116,240 + 58,120 ceil((b + 17) / 128), `optimize.row_ands`.
- `anf-8`, `pp-naf` (C-Flock as it is): h b + 58,120 ceil((896 + 17) / 128) + 116,240.
- `proj-const` (two-phase): 166.3 b + n 54 61 / 3 + 4 61 + 58,120 ceil((1,568 + 17) / 128) + 116,240.
- `proj-lhl` (two-phase, d = A x + B r hides): 166.3 (b + 929) + n 22 61 / 3 + 4 61.
"""
from __future__ import annotations

import json
import math
import sys

N = 224
SHA, HM96 = 58_120, 116_240
H = {"anf-8": 6722.0, "pp-naf": 16034.1}
PROJ = 166.3


def openings() -> dict:
    digest = SHA * math.ceil((4 * N + 17) / 128) + HM96
    return {
        "sha512": lambda b: HM96 + SHA * math.ceil((b + 17) / 128),
        "anf-8": lambda b: H["anf-8"] * b + digest,
        "pp-naf": lambda b: H["pp-naf"] * b + digest,
        "proj-const": lambda b: PROJ * b + N * 54 * 61 / 3 + 4 * 61 + SHA * math.ceil((7 * N + 17) / 128) + HM96,
        "proj-lhl": lambda b: PROJ * (b + 4 * N + 33) + N * 22 * 61 / 3 + 4 * 61,
    }


def break_even_bytes(f, g) -> int | None:
    """The smallest row size (bytes, up to 1 MiB) from which f's opening is cheaper than g's."""
    for b in range(1, 1 << 20):
        if f(b) < g(b) and all(f(c) < g(c) for c in (b * 2, b * 4, b * 16)):
            return b
    return None


def main() -> None:
    sys.path.insert(0, sys.argv[1])
    from golden import model as M
    ops = openings()
    out = {"n": N, "break_even_row_bytes": {k: break_even_bytes(f, ops["sha512"]) for k, f in ops.items() if k != "sha512"},
           "rows": []}
    shipped = M.row_ands
    try:
        for preset in ("demo", "throughput"):
            for steps in (1, 1000, 100_000):
                for row_bytes in (4096, 16384):
                    for name, f in ops.items():
                        M.row_ands = f
                        r = M.J(M.design(preset, point={"steps_per_window": steps}, commit={"row_bytes": row_bytes}))
                        out["rows"].append({"preset": preset, "steps_per_window": steps, "commit_row_bytes": row_bytes,
                                            "opening": name, "J": round(r["J"], 1), "prove_term": round(r["prove_term"], 1),
                                            "serve_term": round(r["serve_term"], 2)})
    finally:
        M.row_ands = shipped
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print("break-even row bytes against SHA-512:", out["break_even_row_bytes"])
    for r in out["rows"]:
        print(f"{r['preset']:10} steps {r['steps_per_window']:>6} rows {r['commit_row_bytes']:5} {r['opening']:10} "
              f"J {r['J']:>12,.1f}  prove {r['prove_term']:>12,.1f}  serve {r['serve_term']:6.2f}")


if __name__ == "__main__":
    main()
