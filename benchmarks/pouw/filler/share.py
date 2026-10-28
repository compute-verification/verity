"""Certified share of int8 tensor capacity for Llama-3.1-8B serving, from dcap.cu's measurements.

Reads one run's dcap.jsonl and prints its tables: τ for the honest kernels,
nominal ÷ capacity over the decode grid, and the serving mixes with and without certified filler. Writes them as JSON.

  python3 share.py RUN_OUT_DIR [--fill FILL_RUN_OUT_DIR] [--dfill DFILL_RUN_OUT_DIR ...] [--fstep FLIN_RUN_OUT_DIR ...]
                   [--json OUT.json]

With --dfill, FILL's DRAM-free filler (dfill.sh) joins the fillers, read at each step's own (batch, context) point
(`PointFiller`), and the adaptive split takes, at each measured point, the most filler SMs whose decode slowdown is ≤ 5%.
With --fstep, FILL's fused filler (flin.sh: filler warps inside the decode linears and attention blocks) joins them, on
the whole GPU and beside the dfill pump on a split's other SMs, each the measured configuration with the largest filler
share at ≤ 5% slowdown (`StepFiller`, which credits the filler warps over the core step only).

With --fill, the filler is the one fill.sh measured (T2's chained int8 GEMM at m = k = 8192 on a green-context split, beside
dcap's decode step): its share of the GPU's peak and the decode slowdown at L = 2048, by batch, the share divided by
COMMIT = 1.09 for committing X every 4 steps. Without it, the filler is W11's register-resident one (below).

Definitions:
- capacity C: the int8 IMMA peak, 1024 MACs/SM/clk × 188 SMs × f; f is the read SM clock unless a column says 2430.
- nominal N: the int8 MACs the served matmuls execute (linears, lm_head, QKᵀ and P·V against the context).
- certified ÷ nominal: the target, 0.99.
- τ = honest priced ÷ honest real = A × N ÷ C_2430, where A = p_MAC ÷ (tensor-peak time a MAC) is the frozen vector's
  MAC price over the time a MAC takes at tensor peak at the model's 2430 MHz. V99 (rs-price-99.py): p_MAC = 0.2424 u,
  tensor-peak time R*/256 = 1.1722 u, so A = 0.2068. The vector has no DRAM row, so a memory-bound step is priced at its
  MACs only.
- forced share = 0.99 τ: the share of the window's capacity, in the priced model's own units, that a cheater must spend
  on the declared work. certified ÷ capacity = 0.99 N ÷ C counts MACs instead, and binds only an honest node.

Constants not measured by dcap.cu, each from a note of record:
- non-GEMM kernels: decode 1.09 ms a step (vLLM 0.28 FP8, full decode graphs, b32 / 1,024), prefill 38.37 ms per 8,192
  tokens. A served decode step adds max(1.09 ms, 4.684 µs × tokens).
- prefill attention: FlashAttention-2 (BF16), 73.11 ms for causal 8,192, η = 0.30 of the int8 peak; an int8 kernel at
  FA2's efficiency of its own peak would be 0.60 (assumed, ETA_FA_INT8).
- without --fill, certified filler beside decode on a green-context split: 0.70 × its SM share of the tensor peak; 60 filler SMs give
  0.222 at ≤ 4.7% decode slowdown, 92 give 0.34 at ≤ 10.6% (measured at
  batch ≤ 32). The chained int8 GEMM is 0.81 alone, 0.74 committing every
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
    ("u5 design only", 0.3478),
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
COMMIT = 1.09  # T2: committing X every 4 steps on a high-priority stream costs 1.09x the bare chain step at 8192²
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
                "certified/capacity": CERT * nc, "forced_share": CERT * tau, "hidden": hidden(tau, A_V99),
                "filler_share": fill}


def hidden(tau, a):
    """The share of tensor peak (at 2430 MHz) that undeclared MMAs may take while the guarantee holds.

    Priced work in a window is at most the window (Σθ = 1, no row busier than real time). Undeclared MMAs at a share u
    of peak are priced A·u of it, so the declared floor 0.99τ fits beside them while 0.99τ + A·u <= 1."""
    return min(1.0, (1 - CERT * tau) / a)


def decode_n(b, l):
    return b * (LIN_TOK + LM_TOK + ATT_TOK_L * l)


class Filler:
    """A certified filler beside decode: (batch, share of the GPU's peak at the read clock, decode slowdown) points,
    interpolated in log batch. `dsms`: the decode partition's SMs, for the linears' compute floor a batch ≤ 32 run never
    reached (W11's); None when the points were measured at every batch used."""

    def __init__(self, pts, dsms=None, step=False):
        pts = sorted(pts)
        self.lb = [math.log(b) for b, _, _ in pts]
        self.f = [f for _, f, _ in pts]
        self.s = [s for _, _, s in pts]
        self.dsms = dsms
        self.step = step  # between points, the next larger batch's point (an adaptive split's choice there)

    def at(self, b, l=None):
        x = math.log(b)
        if self.step:
            i = next((j for j, lb in enumerate(self.lb) if lb >= x - 1e-12), len(self.lb) - 1)
            return self.f[i], self.s[i]
        if x <= self.lb[0] or len(self.lb) == 1:
            return self.f[0], self.s[0]
        if x >= self.lb[-1]:
            return self.f[-1], self.s[-1]
        i = next(j for j in range(len(self.lb) - 1) if self.lb[j] <= x <= self.lb[j + 1])
        t = (x - self.lb[i]) / (self.lb[i + 1] - self.lb[i])
        return self.f[i] + t * (self.f[i + 1] - self.f[i]), self.s[i] + t * (self.s[i + 1] - self.s[i])


class PointFiller:
    """A filler measured at (batch, context) points: {(B, L): (share, slowdown)}. A step at (b, l) reads the measured
    point with the smallest B ≥ b and, among those, the longest L ≤ l: a larger batch and a shorter context leave decode
    more linear time on the partition, so a step is never credited a slowdown smaller than one measured nearer it."""

    dsms = None

    def __init__(self, pts):
        self.pts = dict(pts)

    def at(self, b, l=None):
        l = 2048 if l is None else l
        bs = sorted({pb for pb, _ in self.pts})
        pb = next((x for x in bs if x >= b), bs[-1])
        ls = sorted(pl for xb, pl in self.pts if xb == pb)
        pl = max((x for x in ls if x <= l), default=ls[0])
        return self.pts[(pb, pl)]


class StepFiller(PointFiller):
    """fstep's points {(B, L): (fused share, pump share, core slowdown)}: the fused filler (filler warps inside flin's
    linears and the attention blocks) runs only while the measured core step does, so it is credited over the core step's
    time alone; the pump on a split's other SMs runs through the non-GEMM time too, which a split charges max(1, slowdown)
    (those kernels then have fewer SMs) and which is unchanged without one."""

    def apply(self, m, b, l, ms):
        fused, pump, slow = self.at(b, l)
        other = max(DEC_OTHER_MS, PRE_OTHER_MS_TOK * b)
        tc = (ms - other) * slow
        to = other * (max(1.0, slow) if pump else 1.0)
        return tc + to, ((fused + pump) * tc + pump * to) * m.cap * 1e-3


FILLER = {  # W11: 0.70 x the SM share, worst slowdown at batch ≤ 32
    "60 SMs": Filler([(1, 0.222, 1.047)], dsms=128),
    "92 SMs": Filler([(1, 0.34, 1.106)], dsms=96),
}


def measured_fillers(rows, shape="big"):
    out = {}
    for sms in sorted({r["fill_sms"] for r in rows if r["shape"] == shape}):
        pts = [(r["B"], r["window"]["fill_share"] / COMMIT, r["slowdown"]) for r in rows
               if r["shape"] == shape and r["fill_sms"] == sms and r["L"] == 2048]
        out[f"{sms} SMs"] = Filler(pts)
    return out


def dfill_name(r):
    return f"{r['shape']} {r['fill_sms']} SMs"


def dfill_points(rows, tuned):
    """{(name, B, L): (share, slowdown)}, each averaged over the runs that measured it. `tuned`: a split's arm with
    cuBLASLt re-tuned on the decode partition."""
    acc = {}
    for r in rows:
        if tuned and "tuned" not in r:
            continue
        v = (r["tuned"]["fill_share"], r["tuned"]["slowdown"]) if tuned else (r["window"]["fill_share"], r["slowdown"])
        acc.setdefault((dfill_name(r), r["B"], r["L"]), []).append(v)
    return {k: (sum(s for s, _ in v) / len(v), sum(s for _, s in v) / len(v)) for k, v in acc.items()}


def dfill_fillers(rows):
    """FILL's DRAM-free filler (dfill.sh): one filler per (shape, SMs) over its measured (B, L) points, on a split also
    the arm with cuBLASLt re-tuned on the decode partition (" tuned"), and the adaptive split. Its whole output is an
    8-byte receipt a tile of NC·S MMAs, so it has no commitment factor (COMMIT applies to T2's X only)."""
    out = {}
    for tuned in (False, True):
        pts = dfill_points(rows, tuned)
        for name in sorted({n for n, _, _ in pts}):
            out[name + (" tuned" if tuned else "")] = PointFiller({(b, l): v for (n, b, l), v in pts.items() if n == name})
        pick = adaptive_pick(rows, tuned)
        if pick:
            out[f"dfree adaptive ≤ {LIMIT - 1:.0%}{' tuned' if tuned else ''}"] = PointFiller(
                {bl: (share, slow) for bl, (_, share, slow) in pick.items()})
    return out


LIMIT = 1.05  # the decode slowdown the adaptive split may cost


def adaptive_pick(rows, tuned):
    """Per measured (B, L), the split whose filler share is largest among those whose decode slowdown is ≤ LIMIT:
    {(B, L): (filler SMs, share, slowdown)}, (0, 0.0, 1.0) where no split met it (decode keeps the whole GPU).
    `tuned`: the arms with cuBLASLt re-tuned on the decode partition."""
    split = [r for r in rows if r["place"] == "split" and not r["shape"].endswith(("-lin", "-att"))]
    sms = {dfill_name(r): r["fill_sms"] for r in split}
    best = {}
    for (name, b, l), (share, slow) in dfill_points(split, tuned).items():
        cur = best.setdefault((b, l), (0, 0.0, 1.0))
        if slow <= LIMIT and share > cur[1]:
            best[(b, l)] = (sms[name], share, slow)
    return best


def fstep_key(r):
    return (r["B"], r["L"], r["FW"], r["NC"], r.get("AFW", 0), r.get("fill_sms", 0), r.get("lt_mask", 0), r["pdl"],
            r.get("fatt_pdl", 0), r["thr"], r.get("ANC", 4), r.get("APS", 0), r.get("ALZ", 0))


def afw_name(afw, anc, aps=0, alz=0):
    return (f"{afw}" if anc == 4 else f"{afw} NC {anc}") + (" persistent" if aps else "") + (f" lazy {alz}" if alz else "")


def fstep_points(rows):
    """{config: (fused share, pump share, slowdown)}, each averaged over the runs that measured it (fstep_key)."""
    acc = {}
    for r in rows:
        acc.setdefault(fstep_key(r), []).append((r["fill_share"], r.get("fill_share_pump", 0.0), r["slowdown"]))
    return {k: tuple(sum(x[i] for x in v) / len(v) for i in range(3)) for k, v in acc.items()}


def fstep_pick(rows, split):
    """Per measured (B, L), the fstep configuration with the largest filler share (fused, plus the pump's when `split`)
    among those whose core-step slowdown is ≤ LIMIT: {(B, L): (config, fused share, pump share, slowdown)}; whole-GPU
    configurations only unless `split`."""
    best = {}
    for k, (fused, pump, slow) in fstep_points(rows).items():
        if (k[5] and not split) or slow > LIMIT:
            continue
        cur = best.get(k[:2])
        if cur is None or fused + pump > cur[1] + cur[2]:
            best[k[:2]] = (k, fused, pump, slow)
    return best


def fstep_fillers(rows):
    """FILL's fused filler (flin.sh's fstep): the adaptive choice at ≤ LIMIT on the whole GPU, and with the dfill pump on
    a split's other SMs. Receipts are 8 bytes a tile of NC·S MMAs, as dfill's, so no commitment factor."""
    out = {}
    for split, name in ((False, "fused"), (True, "fused + pump")):
        pick = fstep_pick(rows, split)
        if pick:
            out[f"{name} ≤ {LIMIT - 1:.0%}"] = StepFiller({bl: v[1:] for bl, v in pick.items()})
    return out


HEADLINE = [("decode", 1, 2048), ("decode", 32, 8192), ("decode", 128, 2048), ("decode", 256, 2048),
            ("chat", 32, None), ("chat", 256, None), ("RAG", 144, None), ("code / agents", 48, None),
            ("reasoning", 132, None)]


def headline(res, names):
    """Rows of note §0's table: N/C, cert/C and the time factor, without filler and with each named filler."""
    out = []
    for what, b, l in HEADLINE:
        if what == "decode":
            r = next(g for g in res["grid"] if (g["B"], g["L"]) == (b, l))
            base, label = r["ms"], f"decode B = {b}, L = {l // 1024}K"
            cols = {n: (r[f"filler {n}"], r[f"filler {n}"]["ms"] / base) for n in names}
        else:
            r = next(x for x in res["mixes"] if x["mix"] == what and x["B"] == b and x["schedule"] == "separate"
                     and x["attention"] == "FA2")
            label = f"{what}, B = {b}"
            cols = {n: (r[f"filler {n}"], r[f"filler {n}"].get("time", float("nan"))) for n in names}
        out.append({"row": label, "N/C": r["nominal/capacity"], "cert/C": r["certified/capacity"],
                    **{n: {"N/C": c["nominal/capacity"], "cert/C": c["certified/capacity"], "time": t}
                       for n, (c, t) in cols.items()}})
    return out


def with_filler(m, b, ms, lin_ms, name, l=None):
    fl = FILLER[name]
    if isinstance(fl, StepFiller):
        return fl.apply(m, b, l, ms)
    share, slow = fl.at(b, l)
    t = ms * slow
    if fl.dsms:
        floor = b * (LIN_TOK + LM_TOK) / (0.88 * m.cap * fl.dsms / SMS) * 1e3
        t += max(0.0, floor - lin_ms * slow)
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
    # T1's unfused attention at s = 8192, η at its read clock
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
            t, f = with_filler(m, b, ms, m.lin(b), name, l)
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
                    t, f = with_filler(m, b, step, m.lin(b), fname, lbar)
                    td = d * t / b
                    fm = f * d / b
                    r[f"filler {fname}"] = {**m.row(n + fm, t_pre + td, fm / (m.cap * (t_pre + td) * 1e-3)),
                                            "time": (t_pre + td) / (t_pre + t_dec)}
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


def fmt_in(r, key):
    return fmt(r[key]) if r else "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--fill", type=pathlib.Path)
    ap.add_argument("--dfill", type=pathlib.Path, action="append", default=[],
                    help="a dfill.sh run's out dir (repeatable): FILL's DRAM-free filler")
    ap.add_argument("--fstep", type=pathlib.Path, action="append", default=[],
                    help="a flin.sh run's out dir (repeatable): FILL's fused filler, alone and beside the pump")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()
    rows = [json.loads(line) for line in (args.out / "dcap.jsonl").read_text().splitlines() if line.strip()]
    m = Model(rows)
    fill_rows = []
    if args.fill:
        fill_rows = [json.loads(line) for line in (args.fill / "fill.jsonl").read_text().splitlines() if line.strip()]
        FILLER.clear()
        FILLER.update(measured_fillers(fill_rows))
    dfill_rows = [json.loads(line) for d in args.dfill for line in (d / "dfill.jsonl").read_text().splitlines()
                  if line.strip()]
    dfillers = dfill_fillers(dfill_rows)
    FILLER.update(dfillers)
    fstep_rows = [json.loads(line) for d in args.fstep for line in (d / "fstep.jsonl").read_text().splitlines()
                  if line.strip()]
    FILLER.update(fstep_fillers(fstep_rows))
    res = {"read_mhz": m.mhz, "dram_read_gb_s": m.bw / 1e9, "att_fit": {"gb_s": m.att_bw / 1e9, "c0_ms": m.att_c0},
           "step_delta_ms": m.delta, "eta_fa2": m.eta_fa2, "A_v99": A_V99, "pool_tokens": POOL_TOK,
           "model_check": [{"B": b, "L": l, "measured": r["ms"], "model": m.lin(b) + m.att_ms(b, l) + m.delta}
                           for (b, l), r in sorted(m.step.items())],
           "tau": tau_table(m), "grid": grid(m), "mixes": mixes(m),
           "a_ladder": [{"what": w, "A": a} for w, a in A_LADDER],
           "filler": "measured (--fill)" if args.fill else "W11 (assumed)", "fill": fill_rows, "dfill": dfill_rows,
           "fstep": fstep_rows}
    res["headline"] = headline(res, list(FILLER))
    res["adaptive"] = {("tuned" if t else "as tuned for the whole GPU"):
                       {f"{b}:{l}": v for (b, l), v in adaptive_pick(dfill_rows, t).items()} for t in (False, True)}
    res["fstep_pick"] = {("fused + pump" if s else "fused"): {f"{b}:{l}": v for (b, l), v in fstep_pick(fstep_rows, s).items()}
                         for s in (False, True)}
    print(f"read clock {m.mhz:.0f} MHz, C = {m.cap / 1e12:.1f} TMAC/s (C_2430 = {m.cap_max / 1e12:.1f}); "
          f"DRAM read {m.bw / 1e9:.0f} GB/s; attention fit {m.att_bw / 1e9:.0f} GB/s + {m.att_c0:.3f} ms; "
          f"step δ {m.delta:.3f} ms; FA2 η {m.eta_fa2:.3f}; A(V99) {A_V99:.4f}; pool {POOL_TOK} tokens")
    worst = max(abs(c["model"] / c["measured"] - 1) for c in res["model_check"])
    print(f"step model vs measured: worst {worst:.1%}")
    if fill_rows:
        print("\n| chain | filler SMs | B | L | alone share | share beside decode | certified (÷ COMMIT) | "
              "decode slowdown | split only | filler GB/s |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(fill_rows, key=lambda r: (r["shape"], r["fill_sms"], r["B"], r["L"])):
            a = r["arms"]
            split = a["split"]["ms"] / ((a["full"]["ms"] + a["full2"]["ms"]) / 2)
            print(f"| {r['shape']} {r['m']}×{r['n']}×{r['k']} | {r['fill_sms']} | {r['B']} | {r['L']} | "
                  f"{fmt(r['alone']['share'])} | {fmt(r['window']['fill_share'])} | "
                  f"{fmt(r['window']['fill_share'] / COMMIT)} | {fmt(r['slowdown'])} | {fmt(split)} | "
                  f"{r['window']['fill_gb_per_s']:.0f} |")
    if dfill_rows:
        print("\n| DRAM-free filler | B | L | alone share | share beside decode | decode slowdown | split only | "
              "tuned: share, slowdown, split only | receipts checked (bad) |\n|---|---|---|---|---|---|---|---|---|")
        for r in sorted(dfill_rows, key=lambda r: (dfill_name(r), r["B"], r["L"])):
            a = r["arms"]
            split = a["split"]["ms"] / ((a["full"]["ms"] + a["full2"]["ms"]) / 2) if "split" in a else None
            tu = r.get("tuned")
            tus = f"{fmt(tu['fill_share'])}, {fmt(tu['slowdown'])}, {fmt(tu['split_only'])}" if tu else "–"
            print(f"| {dfill_name(r)} | {r['B']} | {r['L']} | {fmt(r['alone']['share'])} | "
                  f"{fmt(r['window']['fill_share'])} | {fmt(r['slowdown'])} | {fmt(split)} | {tus} | "
                  f"{r['rec_checked']} ({r['rec_bad']}) |")
        for tuned in (False, True):
            pick = adaptive_pick(dfill_rows, tuned)
            print(f"\nadaptive split ≤ {LIMIT - 1:.0%}{', cuBLASLt tuned on the partition' if tuned else ''}: "
                  + "; ".join(f"B {b} L {l}: {sms} SMs, share {share:.3f}, slowdown {slow:.3f}"
                              for (b, l), (sms, share, slow) in sorted(pick.items())))
    if fstep_rows:
        print("\n| fused step | B | L | FW | AFW | pump SMs | cuBLASLt mask | slowdown | split only | filler warps "
              "(linears + attention) | pump | total | receipts checked (bad) | outputs bad |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(fstep_rows, key=fstep_key):
            parts = [r.get(k, {}) for k in ("lin", "att", "pump")]
            nchk = sum(p.get("rec_checked", 0) for p in parts) or r.get("rec_checked", 0)
            nbad = sum(p.get("rec_bad", 0) for p in parts) + r.get("rec_bad", 0)
            print(f"| NC {r['NC']}{'' if r['pdl'] else ', no PDL'} | {r['B']} | {r['L']} | {r['FW']} | "
                  f"{afw_name(r.get('AFW', 0), r.get('ANC', 4), r.get('APS', 0), r.get('ALZ', 0))} | "
                  f"{r.get('fill_sms', 0)} | {r.get('lt_mask', 0)} | {fmt(r['slowdown'])} | {fmt(r['fused0_vs_base'])} | "
                  f"{fmt(r['fill_share'])} ({fmt(r.get('fill_share_lin', r['fill_share']))} + "
                  f"{fmt(r.get('fill_share_att', 0.0))}) | "
                  f"{fmt(r.get('fill_share_pump', 0.0))} | {fmt(r.get('fill_share_total', r['fill_share']))} | "
                  f"{nchk} ({nbad}) | {r['out_mismatches']} |")
        for split in (False, True):
            print(f"\nfstep pick ≤ {LIMIT - 1:.0%}{' with the pump' if split else ', whole GPU'}: "
                  + "; ".join(f"B {b} L {l}: FW {k[2]} AFW {afw_name(k[4], k[10], k[11], k[12])} pump {k[5]} LT {k[6]}, filler {fu:.3f} + {pu:.3f}, "
                              f"slowdown {s:.3f}" for (b, l), (k, fu, pu, s) in sorted(fstep_pick(fstep_rows, split).items())))
    names = [n for n in FILLER if not n.startswith("dfree-") or n == "dfree-split-w8-nc4-t8 92 SMs tuned"]
    print("\nHeadline: N/C, cert/C (= 0.99 N/C) and time with each filler (decode rows: the step's slowdown; mixes: the "
          "mix's time over its time without filler)\n")
    print("| phase or mix | N/C | cert/C | " + " | ".join(f"{n}: N/C, cert/C, time" for n in names) + " |")
    print("|---|---|---|" + "---|" * len(names))
    for h in res["headline"]:
        print(f"| {h['row']} | {fmt(h['N/C'])} | {fmt(h['cert/C'])} | "
              + " | ".join(f"{fmt(h[n]['N/C'])}, {fmt(h[n]['cert/C'])}, {fmt(h[n]['time'])}" for n in names) + " |")
    print(f"\nfiller: {res['filler']}")
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
        print(f"| {r['mix']} | {r['P']} / {r['D']} | {r['B']} | {r['attention']} | {r['schedule']} | "
              f"{fmt(r['prefill_time_share'])} | {fmt(r['nominal/capacity'])} | {fmt(r['tau'])} | "
              f"{fmt(r['certified/capacity'])} | {fmt(r['forced_share'])} | {fmt_in(f6, 'nominal/capacity')} | "
              f"{fmt_in(f6, 'tau')} | {fmt_in(f9, 'nominal/capacity')} | {fmt_in(f9, 'tau')} |")
    picks = [("linears m = 8192", res["tau"][4]["tau"]), ("prefill step 8192 (FA2)", res["tau"][9]["tau"])]
    for b, l in [(32, 8192), (256, 2048)]:
        r = next(g for g in res["grid"] if (g["B"], g["L"]) == (b, l))
        picks += [(f"decode {b} / {l}", r["tau"]), (f"decode {b} / {l} + 92-SM filler", r["filler 92 SMs"]["tau"])]
    for name, b in [("chat", 256), ("reasoning", 132)]:
        r = next(x for x in res["mixes"] if x["mix"] == name and x["B"] == b and x["schedule"] == "separate"
                 and x["attention"] == "FA2")
        picks += [(f"{name} B = {b}", r["tau"]), (f"{name} B = {b} + 92-SM filler", r["filler 92 SMs"]["tau"])]
    res["tau_ladder"] = [{"row": n, **{w: {"tau": t * a / A_V99, "hidden": hidden(t * a / A_V99, a)}
                                       for w, a in A_LADDER}} for n, t in picks]
    print("\nτ (hidden tensor share the guarantee allows) under each vector")
    print("\n| row | " + " | ".join(f"{w} (A = {a:.3f})" for w, a in A_LADDER) + " |")
    print("|---|" + "---|" * len(A_LADDER))
    for n, t in picks:
        print(f"| {n} | " + " | ".join(f"{fmt(t * a / A_V99)} ({fmt(hidden(t * a / A_V99, a))})"
                                       for _, a in A_LADDER) + " |")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
