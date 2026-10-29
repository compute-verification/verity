"""Candidate 2's arithmetic: committing a word every L inner steps under the sm_90 semantic premises
(`Security/Definitions/Pouw/Assumptions/Dimension/H100Sem.lean`), at the floor table `h100Floor`.

1. **The work bound a committed word**: `MH100w`'s 32 (`anyFunctionCeiling`), the counting lemma's 64 − 1024/L
   (`BamFloor`, from `lcommitFloorArith`) and `NoSharingGain`'s L, each as a share of the honest L.
2. **Strassen and Winograd at NCP's operand ranges**: D4's schedule model (`Verity.Accel.NaiveStrassen`: the H200's
   atoms at the measured `k`, digit splits, Strassen, Winograd and the trivial split, additions free) evaluated by a
   cell table that bounds the least cost from below, as D4's kernel tables do: a cell's entry is at most the least cost
   at its lower corner, and the cost is monotone in the operand ranges. Operands: a block-1 or block-2 entry is int7
   data (127 values) plus a 64-value noise entry, so 190 values; a block-3 entry 253. The salt-free data product is
   free, so the decompositions `A·F + E·Y`, `X·F + E·B` and `A·F + E·B + E·F` are evaluated too. A schedule's depth is
   at most log2(L/32) when every leaf keeps k32; `--depth` evaluates deeper, ignoring that.
3. **Strassen–Winograd without operand ranges**, merges priced at 32: (7/8)^d of the tensor work plus the merges.
4. **Per-MAC prices** of the other routes at the floor table and at the H200's measured prices (`--measured`).
5. **Hashing and T2 at each L**, from k32 figures given by `--t2` (a JSON file): hashed bytes and hash time scale by 32/L.
6. **Per monitoring period**: the world (`benchmarks/census/adversary.json`) and the reference datacenter
   (`benchmarks/census/deployments.json`), in AND gates, at `--and-per-flop` and `--utilization`.

    uv run python benchmarks/pouw/theory/lcommit_sem.py [--t2 T2.json] [--measured MEASURED.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LS = [64, 128, 256, 512, 1024, 2048]
PERIODS = {"window": None, "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}

# D4's atoms on the H200 at the measured k (`h200Measured`): (weight in eighths of an s8 MAC, cap, coupling).
ATOMS = {
    "s8": (8, 256, None), "e4m3": (8, 33, (32, 14)), "e5m2": (8, 17, (32, 14)), "f16": (16, 4097, (16, 24)),
    "bf16": (16, 513, (16, 24)), "tf32": (32, 4097, (8, 24)), "b1": (1, 2, None),
}
STRASSEN = [(2, 2), (2, 1), (1, 2), (1, 2), (2, 1), (2, 2), (2, 2)]
WINOGRAD = [(1, 1), (1, 1), (4, 1), (1, 4), (2, 2), (3, 3), (2, 2)]
BIG = float("inf")


def holds(atom, a, b):
    w, cap, cp = atom
    return a <= cap and b <= cap and (cp is None or cp[0] * (a // 2) * (b // 2) <= 2 ** cp[1])


def grid(top):
    cells = set(range(1, 521))
    x = 520.0
    while x < top:
        x *= 1.02
        cells.add(int(x))
    for c in (2, 17, 33, 256, 513, 4097):
        cells.update((c, c + 1))
    return sorted(cells)


class Tables:
    """Lower bounds on D4's least cost at depth budget l, at every pair of grid cells: `lo` over every schedule, `has`
    over those that use Strassen or Winograd somewhere."""

    def __init__(self, depth):
        self.g = grid(260 * 4 ** depth + 2)
        n = len(self.g)
        self.n = n
        g = np.array(self.g)
        atom = np.full((n, n), BIG)
        for at in ATOMS.values():
            for i, a in enumerate(self.g):
                if a > at[1]:
                    break
                ok = np.array([holds(at, a, b) for b in self.g])
                atom[i, ok] = np.minimum(atom[i, ok], at[0])
        atom[0, :] = 0
        atom[:, 0] = 0
        self.atom = atom
        self.rd = lambda v: bisect.bisect_right(self.g, v) - 1
        self.kid = {nu: np.array([self.rd((a - 1) * nu + 1) for a in self.g]) for nu in (1, 2, 3, 4)}
        # split cells: for row i, every cell e < i with d in [g_e, min(g_(e+1) - 1, g_i - 1)], rest ceil(g_i / d_max)
        self.splits = []
        for i, a in enumerate(self.g):
            sp = []
            for e in range(1, i):
                if self.g[e] < 2:
                    continue
                dmax = min(self.g[e + 1] - 1, a - 1)
                if self.g[e] > dmax:
                    continue
                sp.append((e, self.rd(-(-a // dmax))))
            self.splits.append(sp)
        self.lo, self.has = [], []
        for l in range(depth + 1):
            self._level(l)

    def _scheme(self, prev, prof):
        out = np.zeros((self.n, self.n))
        for nu, nv in prof:
            out += prev[np.ix_(self.kid[nu], self.kid[nv])]
        return out

    def _level(self, l):
        lo = self.atom * 8 ** l
        has = np.full((self.n, self.n), BIG)
        if l > 0:
            plo, phas = self.lo[l - 1], self.has[l - 1]
            fast = np.minimum(self._scheme(plo, STRASSEN), self._scheme(plo, WINOGRAD))
            lo = np.minimum(np.minimum(lo, 8 * plo), fast)
            has = np.minimum(fast, 7 * plo + phas)
        for _ in range(4):
            changed = False
            for i in range(self.n):
                sp = self.splits[i]
                if not sp:
                    continue
                e = np.array([s[0] for s in sp])
                r = np.array([s[1] for s in sp])
                cand = (lo[e] + lo[r]).min(axis=0)
                hcand = np.minimum((has[e] + lo[r]).min(axis=0), (lo[e] + has[r]).min(axis=0))
                nl, nh = np.minimum(lo[i], cand), np.minimum(has[i], hcand)
                if (nl < lo[i]).any() or (nh < has[i]).any():
                    changed = True
                lo[i], has[i] = nl, nh
            lo, has = np.minimum(lo, lo.T), np.minimum(has, has.T)
            if not changed:
                break
        self.lo.append(lo)
        self.has.append(has)

    def cost(self, table, l, a, b):
        return float(table[l][self.rd(a), self.rd(b)])


def strassen_margins(depth):
    """Least D4 cost over naive's 8^(l+1), at NCP's operand ranges, for each depth and decomposition."""
    T = Tables(depth)
    options = {
        "X·Y, blocks 1-2 (190 × 190)": [(190, 190)],
        "X·Y, block 3 (253 × 253)": [(253, 253)],
        "A·F + E·Y (127 × 64, 64 × 190)": [(127, 64), (64, 190)],
        "X·F + E·B (190 × 64, 64 × 127)": [(190, 64), (64, 127)],
        "A·F + E·B + E·F": [(127, 64), (64, 127), (64, 64)],
        "full int8 (256 × 256), D4's case": [(256, 256)],
    }
    rows = []
    for l in range(depth + 1):
        naive = 8 ** (l + 1)
        row = {"depth": l, "naive": naive}
        for name, parts in options.items():
            lo = sum(T.cost(T.lo, l, a, b) for a, b in parts)
            fast = min(T.cost(T.has, l, a, b) + sum(T.cost(T.lo, l, a2, b2) for a2, b2 in parts if (a2, b2) != (a, b))
                       for a, b in parts) if l > 0 else BIG
            row[name] = {"least": lo / naive, "least_with_fast": fast / naive}
        rows.append(row)
    return rows, list(options)


def no_range_strassen(L, d, m):
    """Strassen–Winograd at depth d without operand ranges, per word: (7/8)^d L + 32 m sum_(s<d) (7/4)^s."""
    return (7 / 8) ** d * L + 32 * m * sum((7 / 4) ** s for s in range(d))


def per_mac(measured):
    """Units a useful int8 MAC on each route, at the floor table and at measured prices (per register written)."""
    fl = {"wgmma_s8": 32, "mma_s8_k32": 32, "mma_s8_k16": 32, "fp8": 32, "f16": 32, "s4": 32, "b1": 32,
          "idp4a": 32, "imad": 32, "ffma": 32, "dfma": 64, "lds": 32, "sparse": 32}
    routes = [
        ("wgmma s8 k32 (N ≥ 64)", "wgmma_s8", 32, 1),
        ("mma.sync s8 m16n8k32", "mma_s8_k32", 32, 1),
        ("mma.sync s8 m16n8k16", "mma_s8_k16", 16, 1),
        ("FP8 k32, operands split in 2 digits a side (±16 exact)", "fp8", 32, 4),
        ("f16 / bf16 k16 (exact on these ranges)", "f16", 16, 1),
        ("int4 k64, nibble split (4 products a MAC)", "s4", 64, 4),
        ("b1 k256, bit planes (8 × 8 planes a MAC)", "b1", 256, 64),
        ("sparse 2:4 wgmma, dense operand split in 2 halves", "sparse", 32, 1),
        ("IDP4A (4 products)", "idp4a", 4, 1),
        ("IMAD, 2 products packed (SWAR)", "imad", 2, 1),
        ("FFMA (1 product)", "ffma", 1, 1),
        ("DFMA, 3 products packed in 53 bits", "dfma", 3, 1),
        ("LDS product table (1 product a load)", "lds", 1, 1),
        ("Winograd inner product on IDP4A (L/2 products, sums overflow s8: x2)", "idp4a", 4, 0.5 * 2),
    ]
    out = []
    for name, cls, macs, factor in routes:
        f = fl[cls] / macs * factor
        mprice = measured.get(cls)
        out.append({"route": name, "floor": f, "measured": None if mprice is None else mprice / macs * factor})
    return out


def t2_rows(t2):
    rows = []
    for case in t2["cases"]:
        for L in [32] + LS:
            s = 32 / L
            r = {"case": case["name"], "L": L, "hashed_GB": case["hashed_GB"] * s, "hash_ms": case["hash_ms"] * s}
            for mode, step in case["stock_ms"].items():
                r[f"t_H/t_S {mode}"] = case["overhead_no_hash"][mode] + case["hash_ms"] * s / step
            rows.append(r)
    lin = t2["llama_linear_macs_per_token"]
    per_tok = [{"L": L, "MB_per_token": 0.375 * 32 / L * lin / 1e6, "B_per_MAC": 0.375 * 32 / L} for L in [32] + LS]
    rate = t2["hash_GB_per_s"] * 1e9
    peak = t2["tensor_mac_per_s"]
    v4 = [{"L": L, "hash_over_tensor_peak": 0.375 * 32 / L * peak / rate} for L in [32] + LS]
    return rows, per_tok, v4


def periods(and_per_flop, util):
    adv = json.loads((ROOT / "benchmarks/census/adversary.json").read_text())[0]
    dep = json.loads((ROOT / "benchmarks/census/deployments.json").read_text())[0]
    hw = 1.979e15
    world_rate = adv["total_flop"] / adv["monitoring_period_s"] * and_per_flop
    dc_rate = dep["machines"] * dep["gpus_per_machine"] * hw * util * and_per_flop
    out = {}
    for p, sec in PERIODS.items():
        sec = dep["window_s"] if sec is None else sec
        out[p] = {"world": world_rate * sec, "datacenter": dc_rate * sec}
    return out, dep["window_s"]


def shares(L):
    return {"MH100w": 32 / L, "semantic floor": (64 - 1024 / L) / L, "NoSharingGain": 1.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--t2", type=Path)
    ap.add_argument("--measured", type=Path)
    ap.add_argument("--and-per-flop", type=float, default=250.0)
    ap.add_argument("--utilization", type=float, default=0.4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = {}
    res["word_bound"] = [{"L": L, "honest": L, "MH100w": 32, "semantic_floor": 64 - 1024 / L,
                          **{f"share {k}": v for k, v in shares(L).items()}} for L in LS]
    rows, names = strassen_margins(a.depth)
    res["strassen_at_ncp_ranges"] = rows
    res["no_range_strassen"] = [{"L": L, **{f"d{d} m{m}": no_range_strassen(L, d, m) / L for d in (1, 2, 3)
                                            for m in (0.5, 1)}} for L in LS]
    measured = json.loads(a.measured.read_text()) if a.measured else {}
    res["per_mac"] = per_mac(measured)
    if a.t2:
        t2 = json.loads(a.t2.read_text())
        res["t2"], res["llama_bytes"], res["v4_like"] = t2_rows(t2)
    per, window = periods(a.and_per_flop, a.utilization)
    res["periods"] = per
    res["uncredited"] = [{"L": L, "premises": k, "share": 1 - v,
                          **{f"{scope} {p}": (1 - v) * per[p][scope] for p in PERIODS for scope in ("world", "datacenter")}}
                         for L in LS for k, v in shares(L).items()]

    lines = ["# lcommit_sem", "", "## Work bound a committed word (units at h100Floor; share of the honest L)", "",
             "| L | honest | MH100w | semantic floor | share MH100w | share floor | share NoSharingGain |", "|---|---|---|---|---|---|---|"]
    for r in res["word_bound"]:
        lines.append(f"| {r['L']} | {r['L']} | 32 | {r['semantic_floor']:.2f} | {r['share MH100w']:.3f} | "
                     f"{r['share semantic floor']:.3f} | 1 |")
    lines += ["", "## D4's model at NCP's operand ranges: least cost / naive (all schedules; with Strassen or Winograd)", "",
              "| depth | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for r in rows:
        lines.append(f"| {r['depth']} | " + " | ".join(
            f"{r[n]['least']:.4f}; {r[n]['least_with_fast']:.4f}" for n in names) + " |")
    lines += ["", "## Strassen–Winograd without ranges, merges at 32: per-word cost / L (m merges a word a level)", "",
              "| L | " + " | ".join(f"d{d} m{m}" for d in (1, 2, 3) for m in (0.5, 1)) + " |", "|---" * 7 + "|"]
    for r in res["no_range_strassen"]:
        lines.append(f"| {r['L']} | " + " | ".join(f"{r[f'd{d} m{m}']:.3f}" for d in (1, 2, 3) for m in (0.5, 1)) + " |")
    lines += ["", "## Units a useful int8 MAC by route", "", "| route | floor | measured |", "|---|---|---|"]
    for r in res["per_mac"]:
        ms = "" if r["measured"] is None else f"{r['measured']:.2f}"
        lines.append(f"| {r['route']} | {r['floor']:.2f} | {ms} |")
    if a.t2:
        lines += ["", "## Hashing and T2 at each L", "", "| case | L | hashed GB | hash ms | t_H/t_S sync | t_H/t_S async |",
                  "|---|---|---|---|---|---|"]
        for r in res["t2"]:
            lines.append(f"| {r['case']} | {r['L']} | {r['hashed_GB']:.1f} | {r['hash_ms']:.2f} | "
                         f"{r['t_H/t_S sync']:.2f} | {r['t_H/t_S async']:.2f} |")
        lines += ["", "| L | MB hashed a token (Llama-3.2-1B linears) | B a useful MAC | V4-like: hash time / tensor-peak time |",
                  "|---|---|---|---|"]
        for p, v in zip(res["llama_bytes"], res["v4_like"]):
            lines.append(f"| {p['L']} | {p['MB_per_token']:.1f} | {p['B_per_MAC']:.4f} | {v['hash_over_tensor_peak']:.1f} |")
    lines += ["", f"## Per monitoring period (AND gates; window {window} s; scaled, nothing in the totals is proved)", "",
              "| scope | window | week | month | year |", "|---|---|---|---|---|"]
    for scope in ("world", "datacenter"):
        lines.append(f"| {scope}, all compute | " + " | ".join(f"{per[p][scope]:.3g}" for p in PERIODS) + " |")
    lines += ["", "| L | premises | uncredited share | world: window / week / month / year | datacenter: window / week / month / year |",
              "|---|---|---|---|---|"]
    for r in res["uncredited"]:
        w = " / ".join(f"{r[f'world {p}']:.2g}" for p in PERIODS)
        d = " / ".join(f"{r[f'datacenter {p}']:.2g}" for p in PERIODS)
        lines.append(f"| {r['L']} | {r['premises']} | {r['share']:.3f} | {w} | {d} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "lcommit_sem.json").write_text(json.dumps(res, indent=1, default=float))
        (a.out / "lcommit_sem.md").write_text(text)


if __name__ == "__main__":
    main()
