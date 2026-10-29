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
- Y's signed bytes on route U (X + β u8, Y s8): B + F crosses zero for small |B|, so the carry out of a byte depends on
  the noise and no single packed add of a bitwise-derived noise word gives Y's s8 bytes; each Y register takes a second
  write (a LOP3 xor after a biased add), 24/m more. Route U′ (X s8, Y + γ u8, init −γ·Σ_k A_ik a row) moves that write to
  X, 24/n, and adds the row sums, 8/n; route UU (both u8) adds the row sums and a per-entry init, 32/K.
- NCP at k32 words with the pre-noise hashing of A closed by atomicity rather than by the final-word lemma: final word,
  forming and derivation.
- per monitoring period, each share times all compute: the world's 9.36e30 FLOP a year
  (`benchmarks/census/adversary.json` world-inference-2027) and a datacenter of 100,000 H200s
  (`deployments.json` dc-12500-machines-600s) at 1,979 TFLOP/s and 0.4 utilization; as if the design covered it all.
  In AND gates at 2,079/16 an int8 MAC (CORE-TIME's figure, relayed), 2 FLOP a MAC.
- the T1 allowance a year under each period basis the sweep's notes use: the world as the census's served FLOP (stock
  work) or divided by a design's tensor width α; the datacenter at the int8 peak, at `mfu_serve` 0.4
  (`benchmarks/rho/data/reference.toml`'s fleet card) or at the peak over α.
- the bits a token a challenger would stream if it issued one side's noise as a signed coin stream (the network term's
  condition in `Pouw.Guarantees.DitherInstance`): NCP's E₁ at 6 bits an X entry (E₂ and E₃ are E₁'s moves), SSN-A's
  4 bits (b1) or 16 (b4) a row a step of 28 entries; per GPU at the design's served rate, stock's 8,192 tokens in
  37.77 ms (CORE-NCP) over its T2 (estimates). SSN-A's step-1 column code, base 13 in four lanes less the zero vector,
  holds n ≤ 13⁴ − 2.
- k32's hashed bytes, α/8 B a useful MAC (H32's floor over α·K tensor work a dot product), and the terms that scale
  with them: SW-HASH's hiding work (unforced, and with the salt derivation forced; W1 floor / alone) and its T2's
  bytes part, both stated for NCP (α = 3), scaled to SSN-A's and ssn_1l's α over Llama's fused linears. The non-hash
  part of SSN's T2 is (α − 1) of stock's GEMM (12.49 ms of 37.77) plus its coins (0.60 ms) and A's root (0.096).

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


def routes(linears, m: int) -> dict[str, float]:
    stock = sum(m * n * k for _, n, k in linears)
    mk = sum(m * k for _, n, k in linears)
    return {
        "route_u_y_fix": sum(3 * P_FORM * k * n for _, n, k in linears) / stock,
        "route_u_prime": (3 * P_FORM + 8) * mk / stock,
        "route_uu": (8 * mk + sum(32 * m * n for _, n, _ in linears)) / stock,
    }


def single_add_lanes() -> list[int]:
    """The int7 weights B for which no byte constant c ≡ B − 31 (mod 256) keeps c + r's carry fixed over r in 0..63."""
    bad = []
    for b in range(-63, 64):
        cs = [c for c in range(768) if (c - (b - 31)) % 256 == 0]
        if not any(len({(c + r) // 256 for r in range(64)}) == 1 for c in cs):
            bad.append(b)
    return bad


def period_bases(world_flop_year: float, h200_mac_s: float, gpus: int, and_per_mac: float) -> dict[str, float]:
    """T1's allowance (1% of covered stock) a year in AND gates, under each basis a sweep note uses."""
    year = 365 * 86400
    world, dc_peak = world_flop_year / 2, gpus * h200_mac_s * year
    return {f: 0.01 * v * and_per_mac for f, v in {
        "world, census": world, "world, census / 3 (NCP's α)": world / 3, "world, census / 1.153 (SSN-A's α)": world / 1.153,
        "datacenter, peak": dc_peak, "datacenter, mfu_serve 0.4": 0.4 * dc_peak,
        "datacenter, peak / 3 (NCP's α)": dc_peak / 3, "datacenter, peak / 1.153 (SSN-A's α)": dc_peak / 1.153}.items()}


def stream_bits_per_token(linears, layers: int) -> dict[str, float]:
    steps = sum(-(-k // 28) for _, _, k in linears)
    return {"ncp_e1": layers * 6 * sum(k for _, _, k in linears), "ssn_a_b1": layers * 4 * steps,
            "ssn_a_b4": layers * 16 * steps}


SW_HASH_K32 = {"16k": (157.3, 20.8, 41.6, 13.0, 26.0), "256k": (140.9, 1.46, 2.93, 1.00, 2.01),
               "1m": (None, 0.42, 0.85, 0.27, 0.54)}
STOCK_MS, GEMM_MS, NCP_EXTRA_MS, A_ROOT, SSN_COIN_MS = 37.77, 12.49, 56.7, 0.096, 0.60


def tensor_width(linears, steps) -> float:
    macs = sum(n * k for _, n, k in linears)
    return sum(n * 32 * steps(k) for _, n, k in linears) / macs


def by_alpha(alpha: float) -> dict:
    f, out = alpha / 3, {"alpha": alpha, "bytes_per_mac": alpha / 8}
    for rows, (t2, hu, hua, hf, hfa) in SW_HASH_K32.items():
        out[rows] = {"hiding_unforced": [hu * f, hua * f], "hiding_salt_forced": [hf * f, hfa * f]}
        if t2:
            bytes_part = t2 - 1 - NCP_EXTRA_MS / STOCK_MS - A_ROOT
            out[rows]["t2_sync"] = 1 + ((alpha - 1) * GEMM_MS + SSN_COIN_MS) / STOCK_MS + A_ROOT + bytes_part * f
    return out


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
    out["routes"], out["ncp_k32_atomic"] = {}, {}
    for m in MS:
        r = routes(LLAMA_FUSED, m)
        t = out["llama"][f"fused_m{m}"]
        out["routes"][m] = r
        out["ncp_k32_atomic"][m] = t["final_word"] + t["forming"] + t["noise_derivation"]
        print(f"fused m {m:5d}: route U's Y fix +{r['route_u_y_fix']:.3%}  route U′ +{r['route_u_prime']:.3%}  "
              f"route UU +{r['route_uu']:.3%}  ncp-k32 with A's pre-noise hashing closed by atomicity "
              f"{out['ncp_k32_atomic'][m]:.3%}")
    bad = single_add_lanes()
    out["single_add_bad_b"] = bad
    print(f"int7 B with no single-add s8 lane for B + F₁: {len(bad)} of 127, B in [{min(bad)}, {max(bad)}]")
    world_year = 9.36e30
    dc_rate = 100_000 * 1.979e15 * 0.4
    and_per_mac = 2079 / 16
    periods = {"window 600 s": 600, "week": 7 * 86400, "month (30 d)": 30 * 86400, "year": 365 * 86400}
    shares = {
        "T1's bar, 1% of stock": 0.01,
        "noise derivation at T 8,192, fused": out["llama"]["fused_m8192"]["noise_derivation"],
        "noise derivation at T 512's 64 rows, fused": out["llama"]["fused_m64"]["noise_derivation"],
        "V4-like experts after the final-word fix, T 8,192, dense reading": (
            out["v4_expert"]["T8192_dense"]["forming"] + out["v4_expert"]["T8192_dense"]["noise_derivation"]),
        "V4-like experts after the final-word fix, T 8,192, routed rows": (
            out["v4_expert"]["T8192_routed"]["forming"] + out["v4_expert"]["T8192_routed"]["noise_derivation"]),
        "route U's Y fix at T 512's 64 rows, fused": out["routes"][64]["route_u_y_fix"],
        "route U's Y fix at T 8,192, fused": out["routes"][8192]["route_u_y_fix"],
        "route U′'s extra writes, any m, fused": out["routes"][64]["route_u_prime"],
        "A's pre-noise hashing, closed by the final-word lemma in SW-PRICE's card": 14336 / 60817408 * H_BAR,
    }
    out["per_period"] = {}
    for what, s in shares.items():
        row = {p: {"world": s * world_year * sec / (365 * 86400), "datacenter": s * dc_rate * sec}
               for p, sec in periods.items()}
        out["per_period"][what] = {"share": s, **row}
        print(f"{what} ({s:.3%} of covered compute), AND gates: world " +
              " / ".join(f"{row[p]['world'] / 2 * and_per_mac:.1e}" for p in periods) + "; datacenter " +
              " / ".join(f"{row[p]['datacenter'] / 2 * and_per_mac:.1e}" for p in periods))
    bases = period_bases(world_year, 1.979e15 / 2, 100_000, and_per_mac)
    out["period_bases_t1_allowance_year"] = bases
    for what, v in bases.items():
        print(f"T1 allowance a year, {what}: {v:.3g} AND gates")
    stock_tok_s = 8192 / 37.77e-3
    bits = stream_bits_per_token(LLAMA_FUSED, 16)
    rate = {"ncp_e1": stock_tok_s / T2_SYNC, "ssn_a_b1": stock_tok_s / 59.6, "ssn_a_b4": stock_tok_s / 59.6}
    out["stream"] = {k: {"bits_per_token": b, "gbit_s_per_gpu": b * rate[k] / 1e9,
                         "gbit_s_per_gpu_at_stock_rate": b * stock_tok_s / 1e9} for k, b in bits.items()}
    for k, v in out["stream"].items():
        print(f"stream one side, {k}: {v['bits_per_token']:,.0f} bits a token (Llama fused, 16 layers); "
              f"{v['gbit_s_per_gpu']:.2f} Gbit/s a GPU at the design's served rate, "
              f"{v['gbit_s_per_gpu_at_stock_rate']:.1f} at stock's (estimates)")
    out["ssn_a_max_n"] = 13 ** 4 - 2
    print(f"SSN-A's step-1 code holds n ≤ {out['ssn_a_max_n']:,}; Llama's head has n 128,256, DeepSeek-V3's 129,280")
    widths = {"ssn_a": tensor_width(LLAMA_FUSED, lambda k: -(-k // 28)),
              "ssn_1l": tensor_width(LLAMA_FUSED, lambda k: 1 + -(-(k - 28) // 31))}
    out["by_alpha"] = {name: by_alpha(a) for name, a in widths.items()}
    for name, v in out["by_alpha"].items():
        print(f"{name}: α {v['alpha']:.4f}, {v['bytes_per_mac']:.4f} B a useful MAC; T2 sync at T 8,192 "
              f"{v['16k']['t2_sync']:.1f}× (16 KiB) / {v['256k']['t2_sync']:.1f}× (256 KiB); hiding × stock, unforced "
              + ", ".join(f"{r} {v[r]['hiding_unforced'][0]:.3g}–{v[r]['hiding_unforced'][1]:.3g}" for r in SW_HASH_K32)
              + "; salt forced " + ", ".join(f"{r} {v[r]['hiding_salt_forced'][0]:.3g}–{v[r]['hiding_salt_forced'][1]:.3g}"
                                             for r in SW_HASH_K32) + " (estimates)")
    checks = (
        ("final word over the block linears is CORE-TIME's 1.239%", abs(out["llama"]["apart_m8192"]["final_word"] - 0.01239) < 5e-5),
        ("forming apart at T 8,192 is CORE-TIME's 1.10%", abs(out["llama"]["apart_m8192"]["forming"] - 0.01101) < 5e-5),
        ("forming apart at 64 rows is CORE-TIME's 38.3%", abs(out["llama"]["apart_m64"]["forming"] - 0.383) < 5e-4),
        ("forming fused at T 8,192 is CORE-SMELL's 0.86%", abs(out["llama"]["fused_m8192"]["forming"] - 0.0086) < 5e-5),
        ("ShaSpeed tolerance at k32 words is C27's 0.0013–0.0015%",
         abs(delta(WORD_BYTES, H_X) - 1.26e-5) < 1e-7 and abs(delta(WORD_BYTES, H_BAR) - 1.50e-5) < 1e-7),
        ("ncp-k32 at T 8,192 is SW-PRICE's 44.221% less its 41.837% pre-noise term",
         abs(out["ncp_k32_atomic"][8192] - (0.44221 - 0.41837)) < 5e-5),
        ("route U's Y fix equals the Y side's forming term (one more write a Y register)",
         abs(out["routes"][64]["route_u_y_fix"] - out["llama"]["fused_m64"]["forming_y"]) < 1e-12),
        ("the bases reproduce the notes' year allowances: SW-PREMISES 2.03e30 / 1.35e27, SW-NOISE 5.27e30 / 3.52e27",
         abs(bases["world, census / 3 (NCP's α)"] / 2.03e30 - 1) < 0.01
         and abs(bases["datacenter, peak / 3 (NCP's α)"] / 1.35e27 - 1) < 0.01
         and abs(bases["world, census / 1.153 (SSN-A's α)"] / 5.27e30 - 1) < 0.01
         and abs(bases["datacenter, peak / 1.153 (SSN-A's α)"] / 3.52e27 - 1) < 0.01),
        ("SSN-A's code: the zero vector (6,6,6,6) is c = 14,280, skipped", sum(6 * 13 ** e for e in range(4)) == 14280),
        ("tensor widths are SW-NOISE's α over Llama: SSN-A 1.153, ssn_1l 1.044",
         abs(widths["ssn_a"] - 1.153) < 5e-4 and abs(widths["ssn_1l"] - 1.044) < 5e-4),
        ("α/8 B at α = 3 is SW-HASH's 0.375 B and NCP's T2 scaling returns its 157.3× / 140.9×",
         abs(by_alpha(3)["bytes_per_mac"] - WORD_BYTES) < 1e-12 and abs(by_alpha(3)["16k"]["t2_sync"] - 157.3) < 2.5
         and abs(by_alpha(3)["256k"]["t2_sync"] - 140.9) < 2.5),
        ("scaled to SSN-A's α, T2 sync is within 5% of SW-PRICE's ssn-k32-w rows at T 8,192 (62.68× / 55.28×)",
         abs(out["by_alpha"]["ssn_a"]["16k"]["t2_sync"] / 62.68 - 1) < 0.05
         and abs(out["by_alpha"]["ssn_a"]["256k"]["t2_sync"] / 55.28 - 1) < 0.05),
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
