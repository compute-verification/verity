"""The partial-root attack's floor by operation count, from `partial_count.py`'s counts, against one root.

A tuned reduction (Lehmer-style) runs every elementary step on the basis's leading word and applies the accumulated
transform to the full basis in batches. Its cost is then
- leading part: swaps × c_swap + size steps × c_size word multiplies, c_swap 5-8 (the GSO pair and the
  coefficients below it, one quotient) and c_size 3-7 (≤ 3 coefficients and a transform row of 4); steps from fplll's
  swap count (with the textbook's ratio of size steps to swaps) to the textbook's;
- full part: the textbook's limb multiplies of its size steps, over a batching gain g of 1-4.
On the host a word is 64 bits; on the GPU 32-bit IMAD, with a 64-bit leading-part multiply at 4 of them. One root is
b squarings and b/5 multiplies of products only (`root_limb_muls_32`), as is the reduction's count.

The same model predicts k = 1's 2D Gauss reduction (k1_gauss_steps × 3-5 word multiplies, plus 2 entries of ¾·b bits
a step over g): `partial_floor.py` measured that floor at half of GMP's gcdext, 0.025 of a root at 512 bits.

`python3 partial_model.py COUNTS.jsonl` prints one JSON line a field: each floor's range over one root, and the
per-bit claim 2x/b.
"""

from __future__ import annotations

import json
import sys


def main(path: str) -> None:
    for line in open(path):
        r = json.loads(line)
        b, x = r["bits"], r["x"]
        claim = 2 * x / b
        ratio = r["textbook_size_steps"] / r["textbook_swaps"]
        steps = {"lo": (r["fplll_swaps"], r["fplll_swaps"] * ratio), "hi": (r["textbook_swaps"], r["textbook_size_steps"])}
        lead = {"lo": steps["lo"][0] * 5 + steps["lo"][1] * 3, "hi": steps["hi"][0] * 8 + steps["hi"][1] * 7}
        full32 = {"lo": r["size_limb_muls_32"] / 4, "hi": r["size_limb_muls_32"]}
        root32 = r["root_limb_muls_32"]
        host = {k: (lead[k] + full32[k] / 2) / (root32 / 4) for k in ("lo", "hi")}
        gpu = {k: (4 * lead[k] + full32[k]) / root32 for k in ("lo", "hi")}
        g = r["k1_gauss_steps"]
        k1 = {k: (g * c + g * 2 * (0.75 * b / 64) / gg) / (root32 / 4) for k, c, gg in (("lo", 3, 4), ("hi", 5, 1))}
        print(json.dumps({"field": r["field"], "bits": b, "x": x, "claim_over_root": round(claim, 4),
                          "host_floor_over_root": [round(host["lo"], 3), round(host["hi"], 3)],
                          "gpu_floor_over_root": [round(gpu["lo"], 3), round(gpu["hi"], 3)],
                          "margin_host": [round(host["lo"] / claim, 2), round(host["hi"] / claim, 2)],
                          "margin_gpu": [round(gpu["lo"] / claim, 2), round(gpu["hi"] / claim, 2)],
                          "k1_2d_model_over_root": [round(k1["lo"], 4), round(k1["hi"], 4)]}))


if __name__ == "__main__":
    main(sys.argv[1])
