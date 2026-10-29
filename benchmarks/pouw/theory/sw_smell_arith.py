#!/usr/bin/env python3
"""The PoUW sweep's gap terms, re-derived for its smell check: arithmetic on published figures, no measurement.

Every figure is W1 priced work against stock's mnK (CORE.md §2a), at H32's floor of 32 units a 32-bit register:
- the final word, 32/K of stock;
- forming NCP's noised operands, 3p(1/n + 1/m) at p = 8 units an entry (one packed add a register);
- deriving the noise, (XOF byte & 63) − 31 (`PouwNoiseByte`): one LOP3 a packed register of E₁ and of F₁, 8/n + 8/m;
- committing the formed operands X and Y (the sweep's candidate 1): 3/n + 3/m bytes a stock MAC beside the k32 words'
  0.375 B;
- committing a word every L instead of every 32 (candidate 2): 0.375·32/L B a stock MAC;
- ShaSpeed's tolerance: T1's 1% over the hashing's priced work a stock MAC, at HASH-38's bar (1,775 units a byte) and
  GD-HASH's h_x (2,123.4);
- T2 scaled from CORE-NCP's measured split (hashing 98.4% of t_H at T 8,192; 155× sync, 186× async);
- a MoE's routed experts, which see about T·k/E rows a call, beside the dense reading m = T. E = 256 and k = 8 are
  DeepSeek-V3's public config, taken for the sweep's "V4-like" shape (not in the tree); the expert widths are SW-GPU's
  v4gu (4,096 × 7,168) and v4dn (7,168 × 2,048). T 64 is the demo's T 512 step (64 real tokens). Below 64 rows a call
  wgmma misses H32's 32 a register (`Shape100`), so there the floor is loose as well.
- per monitoring period, each share times all compute: the world's 9.36e30 FLOP a year
  (`benchmarks/census/adversary.json` world-inference-2027) and a datacenter of 100,000 H200s
  (`deployments.json` dc-12500-machines-600s) at 1,979 TFLOP/s and 0.4 utilization; as if the design covered it all.

  python3 benchmarks/pouw/theory/sw_smell_arith.py [--json OUT]
"""
from __future__ import annotations

import json
import sys

LLAMA_APART = (("q", 2048, 2048), ("k", 512, 2048), ("v", 512, 2048), ("o", 2048, 2048), ("gate", 8192, 2048),
               ("up", 8192, 2048), ("down", 2048, 8192))
LLAMA_FUSED = (("qkv", 3072, 2048), ("o", 2048, 2048), ("gate_up", 16384, 2048), ("down", 2048, 8192))
V4_EXPERT = (("v4gu", 4096, 7168), ("v4dn", 7168, 2048))
MS = (64, 512, 8192)
P_FORM = 8.0
H_BAR, H_X = 1775.0, 2123.4
WORD_BYTES = 0.375
HASH_SHARE, T2_SYNC, T2_ASYNC = 0.984, 155.0, 186.0


def terms(linears, m: int) -> dict[str, float]:
    stock = sum(m * n * k for _, n, k in linears)
    return {
        "final_word": sum(32 * m * n for _, n, _ in linears) / stock,
        "forming": sum(3 * P_FORM * k * (m + n) for _, n, k in linears) / stock,
        "forming_x": sum(3 * P_FORM * k * m for _, n, k in linears) / stock,
        "forming_y": sum(3 * P_FORM * k * n for _, n, k in linears) / stock,
        "noise_derivation": sum(8 * k * (m + n) for _, n, k in linears) / stock,
        "xy_commit_bytes_per_mac": sum(3 * k * (m + n) for _, n, k in linears) / stock,
    }


def delta(bytes_per_mac: float, h: float) -> float:
    return 0.01 / (bytes_per_mac * h)


def t2_at(L: int, t2: float) -> float:
    return t2 * ((1 - HASH_SHARE) + HASH_SHARE * 32 / L)


def main() -> int:
    out: dict = {"llama": {}, "candidate2": {}, "v4_expert": {}, "checks": []}
    for name, lin in (("apart", LLAMA_APART), ("fused", LLAMA_FUSED)):
        for m in MS:
            t = terms(lin, m)
            out["llama"][f"{name}_m{m}"] = t
            print(f"llama {name:5s} m {m:5d}: final {t['final_word']:.3%}  forming {t['forming']:.3%} "
                  f"(X {t['forming_x']:.2%}, Y {t['forming_y']:.2%})  noise derivation {t['noise_derivation']:.3%}  "
                  f"X,Y commit +{t['xy_commit_bytes_per_mac']:.5f} B/MAC")
    for L in (32, 64, 128, 256):
        b = WORD_BYTES * 32 / L
        row = {"bytes_per_mac": b, "delta_bar": delta(b, H_BAR), "delta_hx": delta(b, H_X),
               "t2_sync": t2_at(L, T2_SYNC), "t2_async": t2_at(L, T2_ASYNC)}
        out["candidate2"][L] = row
        print(f"words every L {L:3d}: {b:.4f} B/MAC  ShaSpeed tolerance {row['delta_hx']:.5%}–{row['delta_bar']:.5%}  "
              f"T2 at T 8,192 ≈ {row['t2_sync']:.0f}× sync / {row['t2_async']:.0f}× async (scaled, estimate)")
    b1 = WORD_BYTES + out["llama"]["fused_m64"]["xy_commit_bytes_per_mac"]
    out["candidate1_m64_delta"] = {"bytes_per_mac": b1, "delta_bar": delta(b1, H_BAR), "delta_hx": delta(b1, H_X)}
    print(f"candidate 1 at m 64: {b1:.4f} B/MAC, ShaSpeed tolerance {delta(b1, H_X):.5%}–{delta(b1, H_BAR):.5%}")
    for T in (64, 512, 8192):
        for E, k in ((256, 8),):
            m_e = max(1, T * k // E)
            for label, m in (("dense reading m = T", T), (f"routed m = T·k/E = {m_e}", m_e)):
                t = terms(V4_EXPERT, m)
                out["v4_expert"][f"T{T}_{'dense' if m == T else 'routed'}"] = {"m": m, **t}
                print(f"V4-like experts, T {T}, {label}: final {t['final_word']:.3%}  forming {t['forming']:.3%}  "
                      f"noise derivation {t['noise_derivation']:.3%}  total after the final-word fix "
                      f"{t['forming'] + t['noise_derivation']:.3%}")
    world_year = 9.36e30
    dc_rate = 100_000 * 1.979e15 * 0.4
    periods = {"window 600 s": 600, "week": 7 * 86400, "month (30 d)": 30 * 86400, "year": 365 * 86400}
    shares = {
        "T1's bar, 1% of stock": 0.01,
        "noise derivation at T 8,192, fused": out["llama"]["fused_m8192"]["noise_derivation"],
        "noise derivation at T 512's 64 rows, fused": out["llama"]["fused_m64"]["noise_derivation"],
        "V4-like experts after the final-word fix, T 8,192, dense reading": (
            out["v4_expert"]["T8192_dense"]["forming"] + out["v4_expert"]["T8192_dense"]["noise_derivation"]),
        "V4-like experts after the final-word fix, T 8,192, routed rows": (
            out["v4_expert"]["T8192_routed"]["forming"] + out["v4_expert"]["T8192_routed"]["noise_derivation"]),
    }
    out["per_period"] = {}
    for what, s in shares.items():
        row = {p: {"world": s * world_year * sec / (365 * 86400), "datacenter": s * dc_rate * sec}
               for p, sec in periods.items()}
        out["per_period"][what] = {"share": s, **row}
        print(f"{what} ({s:.3%} of covered compute), FLOP: world " +
              " / ".join(f"{row[p]['world']:.1e}" for p in periods) + "; datacenter " +
              " / ".join(f"{row[p]['datacenter']:.1e}" for p in periods))
    checks = (
        ("final word over the block linears is CORE-TIME's 1.239%", abs(out["llama"]["apart_m8192"]["final_word"] - 0.01239) < 5e-5),
        ("forming apart at T 8,192 is CORE-TIME's 1.10%", abs(out["llama"]["apart_m8192"]["forming"] - 0.01101) < 5e-5),
        ("forming apart at 64 rows is CORE-TIME's 38.3%", abs(out["llama"]["apart_m64"]["forming"] - 0.383) < 5e-4),
        ("forming fused at T 8,192 is CORE-SMELL's 0.86%", abs(out["llama"]["fused_m8192"]["forming"] - 0.0086) < 5e-5),
        ("ShaSpeed tolerance at k32 words is C27's 0.0013–0.0015%",
         abs(delta(WORD_BYTES, H_X) - 1.26e-5) < 1e-7 and abs(delta(WORD_BYTES, H_BAR) - 1.50e-5) < 1e-7),
    )
    ok = True
    for what, passed in checks:
        out["checks"].append({"check": what, "pass": passed})
        print(("PASS " if passed else "FAIL ") + what)
        ok &= passed
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
