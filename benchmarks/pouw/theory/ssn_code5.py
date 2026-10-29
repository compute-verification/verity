"""ssn_1l column codes past 28,559 columns: what a 5th step-1 lane adds to alpha, forming, coins and the served noise."""
import json
import math
import sys

import numpy as np

B = 13
H32 = 32


def shifted(n, lanes):
    c = np.arange(1, n + 1, dtype=np.int64)
    centre = (B ** lanes - 1) // 2
    c = np.where(c < centre, c, c + 1)
    assert c.max() < B ** lanes
    return np.stack([(c // B ** e) % B - 6 for e in range(lanes)])


def balanced(n, lanes):
    j = np.arange(n, dtype=np.int64)
    z = np.where(j % 2 == 0, j // 2 + 1, -(j // 2 + 1))
    assert np.abs(z).max() <= (B ** lanes - 1) // 2
    out, r = [], z.copy()
    for _ in range(lanes):
        d = ((r + 6) % B) - 6
        out.append(d)
        r = (r - d) // B
    assert (r == 0).all()
    return np.stack(out)


def steps(k, lanes):
    return 1 + math.ceil((k - (32 - lanes)) / 31)


def words(s, lanes):
    return 1 + math.ceil(max(0, s + lanes - 1 - 32) / 32)


def forming_writes(s, lanes):
    w = words(s, lanes)
    extra = 2 * (lanes - 4)
    return s + (s - w) + extra


def row(name, k, n, lanes, code):
    c = code(n, lanes)
    assert len({tuple(x) for x in c.T}) == n and (np.abs(c).sum(0) > 0).all()
    s = steps(k, lanes)
    w = words(s, lanes)
    var = (c.astype(float) ** 2).sum(0) / 4 + (s - 1) / 4
    return {"shape": name, "k": k, "n": n, "lanes": lanes, "code": code.__name__, "S": s,
            "alpha": H32 * s / k, "W": w, "coin_bits_used": s + lanes - 1, "coin_bits_expanded": 32 * w,
            "forming_writes_row": forming_writes(s, lanes),
            "forming_share": H32 * forming_writes(s, lanes) / (n * k),
            "forming_c1_forced_share": H32 * (s + (lanes - 4)) / (n * k),
            "coins_share_uncredited": 2101 * 4 * w / (n * k),
            "std_units_mean": float(np.sqrt(var).mean()), "std_units_max": float(np.sqrt(var).max()),
            "row_common_std": math.sqrt((s - 1) / 4)}


def main():
    shapes = [("llama qkv", 2048, 3072), ("llama down", 8192, 2048), ("llama lm_head", 2048, 128256),
              ("v4-pro wq_b, K assumed 1,536", 1536, 65536), ("v4-pro wq_b, K assumed 7,168", 7168, 65536)]
    rows = []
    for name, k, n in shapes:
        if n <= B ** 4 - 2:
            rows.append(row(name, k, n, 4, shifted))
        rows.append(row(name, k, n, 5, shifted))
        rows.append(row(name, k, n, 5, balanced))
    checks = [("5-lane codes are injective and nonzero up to 13^5 - 2 / (13^5 - 1) columns",
               len({tuple(x) for x in balanced(371290, 5).T}) == 371290)]
    for r in rows:
        print(f"{r['shape']:30s} K {r['k']:5d} n {r['n']:6d} {r['lanes']}L {r['code']:8s} S {r['S']:3d} "
              f"alpha {r['alpha']:.4f} W {r['W']} forming {100 * r['forming_share']:.5f}% "
              f"coins(uncr) {100 * r['coins_share_uncredited']:.5f}% std {r['std_units_mean']:.2f} (max {r['std_units_max']:.2f})")
    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)
    if len(sys.argv) > 1:
        json.dump({"rows": rows, "checks": checks}, open(sys.argv[1], "w"), indent=1)
    sys.exit(0 if all(ok for _, ok in checks) else 1)


main()
