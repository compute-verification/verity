"""Certified share of int8 tensor capacity for Llama-3.1-8B serving, from dcap.cu's measurements (CAP, cmp-13).

Reads one run's dcap.jsonl and prints the tables of note:20261016T0215Z-draft-capacity-share: τ for the honest kernels,
nominal ÷ capacity over the decode grid, and the serving mixes with and without certified filler. Writes them as JSON.

  python3 share.py RUN_OUT_DIR [--json OUT.json]

Definitions:
- capacity C: the int8 IMMA peak, 1024 MACs/SM/clk × 188 SMs × f; f is the read SM clock unless a column says 2430.
- nominal N: the int8 MACs the served matmuls execute (linears, lm_head, QKᵀ and P·V against the context).
- certified ÷ nominal: the target, 0.99 (note:20261015T2356Z-draft-rs-99-budget).
- τ = honest priced ÷ honest real = A × N ÷ C_2430, where A = p_MAC ÷ (tensor-peak time a MAC) is the frozen vector's
  MAC price over the time a MAC takes at tensor peak at the model's 2430 MHz. V99 (rs-price-99.py): p_MAC = 0.2424 u,
  tensor-peak time R*/256 = 1.1722 u, so A = 0.2068. The vector has no DRAM row, so a memory-bound step is priced at its
  MACs only.
- forced share = 0.99 τ: the share of the window's capacity, in the priced model's own units, that a cheater must spend
  on the declared work. certified ÷ capacity = 0.99 N ÷ C counts MACs instead, and binds only an honest node.

Constants not measured by dcap.cu, each from a note of record:
- non-GEMM kernels: decode 1.09 ms a step (vLLM 0.28 FP8, full decode graphs, b32 / 1,024), prefill 38.37 ms per 8,192
  tokens (note:20261015T0417Z-draft-saturation-budget). A served decode step adds max(1.09 ms, 4.684 µs × tokens).
- prefill attention: FlashAttention-2 (BF16), 73.11 ms for causal 8,192, η = 0.30 of the int8 peak; an int8 kernel at
  FA2's efficiency of its own peak would be 0.60 (assumed, ETA_FA_INT8).
- certified filler beside decode on a green-context split: 0.70 × its SM share of the tensor peak; 60 filler SMs give
  0.222 at ≤ 4.7% decode slowdown, 92 give 0.34 at ≤ 10.6% (note:20261015T0713Z-draft-filler-issue-slack, measured at
  batch ≤ 32). The chained int8 GEMM (note:20261015T1622Z-draft-chained-tensor-work) is 0.81 alone, 0.74 committing every
  4 steps.
- KV pool: 80 GB of int8 KV (64 KiB a token), 1.22 M tokens.
"""

import argparse
import json
import math
import pathlib

SMS, PER_SM, F_MAX = 188, 1024, 2430e6
A_V99 = 0.2424 / (300.08 / 256)
A_LADDER = [  # (what changes, A), rs-price-99.py's LP re-solved (note §1)
    ("V99 as frozen", A_V99),
    ("u5 alphabet only", 0.3478),
    ("and no W′ head credits", 0.5108),
    ("and ρ = 0 (no FP4 two-window split)", 0.5648),
    ("product routes only, A = 256/R*", 256 / 300.08),
    ("no CUDA-core product route", 1.0),
]
H, LAYERS, NQ, NKV, HD, INTER, VOCAB = 4096, 32, 32, 8, 128, 14336, 128256
LIN = {"qkv": (H, (NQ + 2 * NKV) * HD), "o": (H, H), "gate_up": (H, 2 * INTER), "down": (INTER, H)}
LIN_TOK = LAYERS * sum(k * n for k, n in LIN.values())
LM_TOK = H * VOCAB
ATT_TOK_L = LAYERS * NQ * HD * 2  # MACs a token a context position (QKᵀ and P·V)
KV_TOK = 2 * NKV * HD * LAYERS  # int8 bytes
POOL_TOK = int(80e9 // KV_TOK)
CERT = 0.99
DEC_OTHER_MS, PRE_OTHER_MS_TOK = 1.09, 38.37 / 8192
FA2_MS_8192, ETA_FA_INT8 = 73.11, 0.60
FILLER = {"60 SMs": (0.222, 1.047, 128), "92 SMs": (0.34, 1.106, 96)}  # share at read clock, decode slowdown, decode SMs
MIXES = [("chat", 1024, 256), ("RAG", 8192, 256), ("code / agents", 24576, 512), ("reasoning", 1024, 8192)]


def causal(p):
    return LAYERS * NQ * HD * 2 * p * (p + 1) / 2


def interp(xs, ys, x):
    """Log-log interpolation; linear in x past the last point."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1] * x / xs[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (math.log(x) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
            return math.exp(math.log(ys[i]) + t * (math.log(ys[i + 1]) - math.log(ys[i])))
    raise ValueError(x)


class Model:
    def __init__(self, rows):
        bw = [r for r in rows if r["mode"] == "bw"]
        self.bw = max(r["gb_per_s"] for r in bw) * 1e9
        lin = sorted((r for r in rows if r["mode"] == "lin"), key=lambda r: r["M"])
        self.mhz = sorted(r["mhz"] for r in rows if r["mode"] != "gate")[len(rows) // 2]
        self.cap = PER_SM * SMS * self.mhz * 1e6
        self.cap_max = PER_SM * SMS * F_MAX
        lm = [(r["M"], next(a["ms_alone"] for a in r["algos"] if a["name"] == "lm_head")) for r in lin if r["M"] <= 256]
        self.lm_m, self.lm_ms = [m for m, _ in lm], [t for _, t in lm]
        self.lin_m = [r["M"] for r in lin]
        self.lin_body = [r["ms"] - (self.lm(r["M"]) if r["M"] <= 256 else 0.0) for r in lin]
        self.lin_rows = lin
        self.att = {(r["M"], r["L"]): r for r in rows if r["mode"] == "att"}
        self.step = {(r["M"], r["L"]): r for r in rows if r["mode"] == "step"}
        pts = [(b * l * KV_TOK, r["ms"]) for (b, l), r in self.att.items()]
        n = len(pts)
        mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
        slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        self.att_bw, self.att_c0 = 1e3 / slope, my - slope * mx
        self.delta = sum(r["ms"] - self.lin(b) - self.att[(b, l)]["ms"] for (b, l), r in self.step.items()) / len(self.step)
        self.eta_fa2 = causal(8192) / (FA2_MS_8192 * 1e-3 * self.cap)

    def lm(self, b):
        return interp(self.lm_m, self.lm_ms, b)

    def lin(self, m, heads=None):
        """32 layers of linears at m rows plus lm_head for `heads` rows (m when m ≤ 256)."""
        heads = (m if m <= 256 else 0) if heads is None else heads
        return interp(self.lin_m, self.lin_body, m) + (self.lm(heads) if heads else 0.0)

    def att_ms(self, b, l):
        return self.att_c0 + b * l * KV_TOK / self.att_bw * 1e3

    def step_ms(self, b, l, measured=True):
        r = self.step.get((b, l))
        core = r["ms"] if (measured and r) else self.lin(b) + self.att_ms(b, l) + self.delta
        return core + max(DEC_OTHER_MS, PRE_OTHER_MS_TOK * b)

    def prefill_ms(self, p, eta_fa):
        """One prompt of p tokens, its linears in a prefill batch of max(p, 8192) tokens (k prompts packed)."""
        k = max(1, 8192 // p)
        return (self.lin(k * p, heads=k) / k + causal(p) / (eta_fa * self.cap) * 1e3 + PRE_OTHER_MS_TOK * p)

    def row(self, n, ms, fill=0.0):
        """n nominal MACs (fill of them filler) in ms: the table's ratios."""
        nc, nc_max = n / (self.cap * ms * 1e-3), n / (self.cap_max * ms * 1e-3)
        tau = A_V99 * nc_max
        return {"nominal/capacity": nc, "nominal/capacity_2430": nc_max, "certified/nominal": CERT, "tau": tau,
                "certified/capacity": CERT * nc, "forced_share": CERT * tau, "filler_share": fill}


def decode_n(b, l):
    return b * (LIN_TOK + LM_TOK + ATT_TOK_L * l)


def with_filler(m, b, ms, lin_ms, name):
    share, slow, dsms = FILLER[name]
    lin_macs = b * (LIN_TOK + LM_TOK)
    floor = lin_macs / (0.88 * m.cap * dsms / SMS) * 1e3
    t = ms * slow + max(0.0, floor - lin_ms * slow)
    return t, share * m.cap * t * 1e-3


def tau_table(m):
    out = []
    r8k = next(r for r in m.lin_rows if r["M"] == 8192)
    for a in r8k["algos"]:
        if a["name"] in LIN:
            k, n = LIN[a["name"]]
            macs = 8192 * k * n
            eta = macs / (a["ms_alone"] * 1e-3 * m.cap)
            out.append({"kernel": f"{a['name']} m = 8192 (cuBLASLt int8, one layer)", "eta": eta,
                        "tau": A_V99 * macs / (a["ms_alone"] * 1e-3 * m.cap_max)})
    eta = r8k["macs"] / (r8k["ms"] * 1e-3 * m.cap)
    out.append({"kernel": "32 layers of linears, m = 8192 (measured)", "eta": eta, "tau": A_V99 * eta * m.mhz * 1e6 / F_MAX})
    # T1's unfused attention at s = 8192, η at its read clock (note:20261015T1814Z-draft-rs-masked-gemm-cost)
    for name, eta_t1 in [("QKᵀ s = 8192, materialized (T1)", 0.121), ("P·V s = 8192 (T1)", 0.443),
                         ("FA2 BF16 causal 8192 (vLLM)", m.eta_fa2), ("int8 FA at FA2's efficiency (assumed)", ETA_FA_INT8)]:
        out.append({"kernel": name, "eta": eta_t1, "tau": A_V99 * eta_t1 * m.mhz * 1e6 / F_MAX})
    for name, eta_fa in [("prefill step 8192, int8 linears + FA2 + non-GEMM", m.eta_fa2),
                         ("prefill step 8192, int8 linears + int8 FA + non-GEMM", ETA_FA_INT8)]:
        n = 8192 * LIN_TOK + LM_TOK + causal(8192)
        ms = m.prefill_ms(8192, eta_fa)
        r = m.row(n, ms)
        out.append({"kernel": name, "eta": r["nominal/capacity"], "tau": r["tau"], "ms": ms})
    return out


def grid(m):
    out = []
    pairs = sorted(set([(b, l) for b in (1, 8, 32, 64, 128, 256) for l in (2048, 8192, 32768)] + list(m.step)))
    for b, l in pairs:
        ms = m.step_ms(b, l)
        n = decode_n(b, l)
        r = {"B": b, "L": l, "fits": b * l <= POOL_TOK, "measured": (b, l) in m.step, "ms": ms, **m.row(n, ms)}
        for name in FILLER:
            t, f = with_filler(m, b, ms, m.lin(b), name)
            r[f"filler {name}"] = {"ms": t, **m.row(n + f, t, f / (m.cap * t * 1e-3))}
        out.append(r)
    return out


def mixes(m):
    out = []
    for name, p, d in MIXES:
        bmax = min(256, POOL_TOK // (p + d))
        for b in sorted({min(32, bmax), bmax}):
            lbar = p + d / 2
            n = p * LIN_TOK + LM_TOK + causal(p) + d * (LIN_TOK + LM_TOK + ATT_TOK_L * lbar)
            for eta_fa, fa in [(m.eta_fa2, "FA2"), (ETA_FA_INT8, "int8 FA")]:
                t_pre = m.prefill_ms(p, eta_fa)
                step = m.step_ms(b, int(lbar), measured=False)
                t_dec = d * step / b
                base = {"mix": name, "P": p, "D": d, "B": b, "attention": fa}
                r = {**base, "schedule": "separate", "prefill_time_share": t_pre / (t_pre + t_dec),
                     **m.row(n, t_pre + t_dec)}
                for fname in FILLER:
                    t, f = with_filler(m, b, step, m.lin(b), fname)
                    td = d * t / b
                    fm = f * d / b
                    r[f"filler {fname}"] = m.row(n + fm, t_pre + td, fm / (m.cap * (t_pre + td) * 1e-3))
                out.append(r)
                c = b * p / d
                mrows = b + c
                if mrows > 8192:
                    continue
                st = (m.lin(int(mrows), heads=b) + m.att_ms(b, lbar) + m.delta
                      + (b / d) * causal(p) / (eta_fa * m.cap) * 1e3 + max(DEC_OTHER_MS, PRE_OTHER_MS_TOK * mrows))
                out.append({**base, "schedule": f"chunked, M = {mrows:.0f}", "prefill_time_share": None,
                            **m.row(n, st * d / b)})
    return out


def fmt(x):
    return "–" if x is None else f"{x:.3f}" if x < 10 else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()
    rows = [json.loads(line) for line in (args.out / "dcap.jsonl").read_text().splitlines() if line.strip()]
    m = Model(rows)
    res = {"read_mhz": m.mhz, "dram_read_gb_s": m.bw / 1e9, "att_fit": {"gb_s": m.att_bw / 1e9, "c0_ms": m.att_c0},
           "step_delta_ms": m.delta, "eta_fa2": m.eta_fa2, "A_v99": A_V99, "pool_tokens": POOL_TOK,
           "model_check": [{"B": b, "L": l, "measured": r["ms"], "model": m.lin(b) + m.att_ms(b, l) + m.delta}
                           for (b, l), r in sorted(m.step.items())],
           "tau": tau_table(m), "grid": grid(m), "mixes": mixes(m),
           "a_ladder": [{"what": w, "A": a} for w, a in A_LADDER]}
    print(f"read clock {m.mhz:.0f} MHz, C = {m.cap / 1e12:.1f} TMAC/s (C_2430 = {m.cap_max / 1e12:.1f}); "
          f"DRAM read {m.bw / 1e9:.0f} GB/s; attention fit {m.att_bw / 1e9:.0f} GB/s + {m.att_c0:.3f} ms; "
          f"step δ {m.delta:.3f} ms; FA2 η {m.eta_fa2:.3f}; A(V99) {A_V99:.4f}; pool {POOL_TOK} tokens")
    worst = max(abs(c["model"] / c["measured"] - 1) for c in res["model_check"])
    print(f"step model vs measured: worst {worst:.1%}")
    print("\n| kernel | η (read clock) | τ (V99) |\n|---|---|---|")
    for r in res["tau"]:
        print(f"| {r['kernel']} | {fmt(r['eta'])} | {fmt(r['tau'])} |")
    print("\n| B | L | fits | step ms | N/C | τ | cert/C | forced | +60 SMs N/C | τ | +92 SMs N/C | τ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["grid"]:
        f6, f9 = r["filler 60 SMs"], r["filler 92 SMs"]
        print(f"| {r['B']} | {r['L']} | {'yes' if r['fits'] else 'no'}{'' if r['measured'] else ' (model)'} | "
              f"{fmt(r['ms'])} | {fmt(r['nominal/capacity'])} | {fmt(r['tau'])} | {fmt(r['certified/capacity'])} | "
              f"{fmt(r['forced_share'])} | {fmt(f6['nominal/capacity'])} | {fmt(f6['tau'])} | "
              f"{fmt(f9['nominal/capacity'])} | {fmt(f9['tau'])} |")
    print("\n| mix | P / D | B | attention | schedule | prefill time | N/C | τ | cert/C | forced | +60 SMs N/C | τ | "
          "+92 SMs N/C | τ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["mixes"]:
        f6, f9 = r.get("filler 60 SMs"), r.get("filler 92 SMs")
        g = lambda f, k: fmt(f[k]) if f else "–"  # noqa: E731
        print(f"| {r['mix']} | {r['P']} / {r['D']} | {r['B']} | {r['attention']} | {r['schedule']} | "
              f"{fmt(r['prefill_time_share'])} | {fmt(r['nominal/capacity'])} | {fmt(r['tau'])} | "
              f"{fmt(r['certified/capacity'])} | {fmt(r['forced_share'])} | {g(f6, 'nominal/capacity')} | "
              f"{g(f6, 'tau')} | {g(f9, 'nominal/capacity')} | {g(f9, 'tau')} |")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
