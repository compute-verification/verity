"""The filler scorer's suite and arithmetic (locked; owner compute-accounting; note:20261016T0435Z-report-filler-spec).
Stdlib only: it runs on the node after the lease and in the pouw suite's CPU test. `score.py` measures; this module turns
the driver's rows (`dscore.cu`) and the gates' outputs into SCORE, the per-entry minimum, the gates and the secondary
metrics.

The suite (CAP's mixes, Llama-3.1-8B on one RTX PRO 6000; `SUITE`): decode at B 1/32/128/256 over 2K–32K contexts (the
longest dcap holds, B·L ≤ 786K), chat 1K/256 at B 32 and 256, RAG 8K/256 at B 144, code/agents 24K/512 at B 48,
reasoning 1K/8K at B 132, and an 8,192-token prefill step. A mix's decode is measured at its mean context P + D/2,
rounded down to 512 and capped at 786K/B (RAG reads 5K, code 16K, chat 1K). `ROBUST` points are reported, not scored.

Each entry is one unit of serving-alone wall time. Per entry, with the arm:
- served decode step: dcap's step + the non-GEMM kernels, max(1.09 ms, 4.684 µs × B) (CAP's convention), those stretched
  by the step's own slowdown;
- prefill step: dcap's 32 layers of linears at M = 8192 (measured) + FA2 attention (73.11 ms causal 8,192) + non-GEMM
  (38.37 ms), the unmeasured two stretched by the linears' slowdown; a mix's prefill of P tokens is that step's time
  scaled by P/8192 (linears and non-GEMM) and causal(P)/causal(8192) (attention);
- coverage = (c_serving × serving MACs + c_filler × filler MACs) ÷ (C × wall time), C the int8 IMMA peak (1024 MACs
  a SM a clock × 188) at the serving-alone SM clock, so a filler that throttles the clock gains nothing.

SCORE is the time-weighted mean of the entries' coverage (weights: each entry's wall time with the arm, one unit alone),
i.e. Σ certified ÷ Σ capacity × time. The c's are the PROVED certified fractions (`CERTIFIED`); the filler's is the
spec's provisional 0.99 until FILL-SEC proves one, and every result with filler work carries that flag.
"""

from __future__ import annotations

import csv
import io
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decode_capacity"))
import share  # noqa: E402  CAP's constants: MACs a token, non-GEMM times, FA2, the int8 peak

SCORER = "filler-scorer/v1"
LIMIT = 1.05  # serving slowdown, every entry
DRAM_LIMIT = 0.01  # of the HBM bandwidth the same run's validated counter measures
CONTROL_BYTES = 1 << 30  # bw_read streams 1 GiB a launch: the DRAM counter's positive control
CONTROL_RANGE = (0.97, 1.10)  # dram__bytes.sum ÷ 1 GiB that validates the counter (FILL read 1.074–1.078 GB)
TOO_GOOD = 1.02  # a filler faster beside serving than alone on the same SMs goes to the assessor
SMS, PER_SM = share.SMS, share.PER_SM
MAX_BL = 786432

CERTIFIED = {
    "serving": {"fraction": 0.225, "status": "proved",
                "source": "certifiedFraction120θ_half, (φ/4)·N at φ = 0.9 (note:20261015T2040Z-report-relative-security-decision)"},
    "filler": {"fraction": 0.99, "status": "provisional",
               "source": "the spec's placeholder until FILL-SEC proves the filler's floor (note:20261016T0418Z-draft-filler-floor)"},
}
#: VF's honest MAC price over a MAC's tensor-peak time: τ = 0.2447 at 1.18 u (note:20261015T1814Z-draft-rs-machine-model §12)
A_VF = 0.2447 * 1.18 / (300.08 / 256)

DECODE = [(1, 2048), (1, 8192), (1, 32768), (32, 2048), (32, 8192), (32, 24576), (128, 2048), (128, 6144), (256, 2048),
          (256, 3072)]
MIXES = [("chat", 1024, 256, 32), ("chat", 1024, 256, 256), ("RAG", 8192, 256, 144), ("code / agents", 24576, 512, 48),
         ("reasoning", 1024, 8192, 132)]
PREFILL_M = 8192
ROBUST = [(8, 16384), (64, 4096), (192, 2048)]


def mix_context(p: int, d: int, b: int) -> int:
    return min((p + d // 2) // 512 * 512, MAX_BL // b // 512 * 512)


def suite() -> list[dict]:
    """The scored entries, each naming the device points it reads."""
    out = [{"entry": f"decode B {b} L {l // 1024}K", "kind": "decode", "B": b, "L": l, "points": [("decode", b, l)]}
           for b, l in DECODE]
    for name, p, d, b in MIXES:
        l = mix_context(p, d, b)
        out.append({"entry": f"{name} {p // 1024}K/{d} B {b}", "kind": "mix", "P": p, "D": d, "B": b, "L": l,
                    "points": [("prefill", PREFILL_M, 0), ("decode", b, l)]})
    out.append({"entry": f"prefill {PREFILL_M}", "kind": "prefill", "B": PREFILL_M, "L": 0,
                "points": [("prefill", PREFILL_M, 0)]})
    return out


def points(robust: bool = True) -> list[tuple[str, int, int]]:
    """Every device point the suite reads, in the order the run measures them; then the robustness points."""
    seen = []
    for e in suite():
        for pt in e["points"]:
            if pt not in seen:
                seen.append(pt)
    if robust:
        seen += [("decode", b, l) for b, l in ROBUST if ("decode", b, l) not in seen]
    return seen


# ---- one device point ------------------------------------------------------------------------------------------------------


def _med(xs):
    return statistics.median(xs) if xs else float("nan")


def point_stats(row: dict) -> dict:
    """A `dscore` row's numbers: the paired slowdown (median over rounds of the arm block's median rep over the alone
    block's), both sides' median time, clock and power, the filler's rate over its windows and its per-block spread."""
    al = [r for b in row["blocks"] for r in b["alone"]["reps"]]
    ar = [r for b in row["blocks"] for r in b["arm"]["reps"]]
    pairs = [_med([r[0] for r in b["arm"]["reps"]]) / _med([r[0] for r in b["alone"]["reps"]]) for b in row["blocks"]]
    fill = row["place"] != "none"
    rate, cv = 0.0, None
    if fill:
        win = sum(b["arm"]["window_ms"] for b in row["blocks"])
        chunks = sum(b["arm"]["chunks"] for b in row["blocks"])
        rate = chunks * row["chunk_macs"] / (win * 1e-3) if win > 0 else 0.0
        per = [b["arm"]["chunks"] * row["chunk_macs"] / (b["arm"]["window_ms"] * 1e-3) for b in row["blocks"]
               if b["arm"]["window_ms"] > 0]
        cv = statistics.pstdev(per) / statistics.mean(per) if len(per) > 1 and statistics.mean(per) > 0 else None
    alone_rate = row["alone_fill"]["mac_per_s"] if fill else 0.0
    return {
        "slowdown": _med(pairs), "slowdown_rounds": pairs, "alone_ms": _med([r[0] for r in al]),
        "arm_ms": _med([r[0] for r in ar]), "alone_mhz": _med([r[1] for r in al]), "arm_mhz": _med([r[1] for r in ar]),
        "alone_watt": _med([r[2] for r in al]), "arm_watt": _med([r[2] for r in ar]),
        "clock_reads": sum(1 for r in al + ar if r[1] > 0), "power_reads": sum(1 for r in al + ar if r[2] > 0),
        "reps": len(al) + len(ar), "fill_rate": rate, "fill_rate_alone": alone_rate, "fill_rate_cv": cv,
        "beside_over_alone": rate / alone_rate if alone_rate else None,
        "rec_checked": row["rec_checked"], "rec_bad": row["rec_bad"], "place": row["place"],
        "fill_sms": row["fill_sms"], "setup_ms": row["setup_ms"], "spread": row["spread"],
        "kv_wrapped": row["kv_wrapped"],
    }


def cap(mhz: float) -> float:
    return PER_SM * SMS * mhz * 1e6


def other_ms(b: int) -> float:
    return max(share.DEC_OTHER_MS, share.PRE_OTHER_MS_TOK * b)


def decode_entry(st: dict, b: int, l: int) -> dict:
    t0 = st["alone_ms"] + other_ms(b)
    t1 = st["arm_ms"] + other_ms(b) * st["slowdown"]
    n = share.decode_n(b, l)
    return {"t0_ms": t0, "t1_ms": t1, "serving_macs": n, "filler_macs": st["fill_rate"] * t1 * 1e-3,
            "mhz": st["alone_mhz"], "slowdowns": {"decode": st["slowdown"]}}


def prefill_parts(st: dict) -> tuple[float, float]:
    """(alone, with the arm) ms of the 8,192-token prefill step."""
    rest = share.FA2_MS_8192 + share.PRE_OTHER_MS_TOK * PREFILL_M
    return st["alone_ms"] + rest, st["arm_ms"] + rest * st["slowdown"]


def prefill_entry(st: dict) -> dict:
    t0, t1 = prefill_parts(st)
    n = PREFILL_M * share.LIN_TOK + share.causal(PREFILL_M)
    return {"t0_ms": t0, "t1_ms": t1, "serving_macs": n, "filler_macs": st["fill_rate"] * t1 * 1e-3,
            "mhz": st["alone_mhz"], "slowdowns": {"prefill": st["slowdown"]}}


def mix_entry(pre: dict, dec: dict, p: int, d: int, b: int, l: int) -> dict:
    """One request of the mix: its prompt's share of prefill steps, and D/B decode steps."""
    def tp(lin, s):
        return (lin * p / PREFILL_M + share.PRE_OTHER_MS_TOK * p * s
                + share.FA2_MS_8192 * share.causal(p) / share.causal(PREFILL_M) * s)

    tp0, tp1 = tp(pre["alone_ms"], 1.0), tp(pre["arm_ms"], pre["slowdown"])
    td0 = d / b * (dec["alone_ms"] + other_ms(b))
    td1 = d / b * (dec["arm_ms"] + other_ms(b) * dec["slowdown"])
    n = p * share.LIN_TOK + share.LM_TOK + share.causal(p) + d * (share.LIN_TOK + share.LM_TOK + share.ATT_TOK_L * l)
    f = (pre["fill_rate"] * tp1 + dec["fill_rate"] * td1) * 1e-3
    mhz = (tp0 * pre["alone_mhz"] + td0 * dec["alone_mhz"]) / (tp0 + td0)
    return {"t0_ms": tp0 + td0, "t1_ms": tp1 + td1, "serving_macs": n, "filler_macs": f, "mhz": mhz,
            "prefill_time_share": tp0 / (tp0 + td0),
            "slowdowns": {"prefill": pre["slowdown"], "decode": dec["slowdown"]}}


def coverage(e: dict, certified: dict = CERTIFIED) -> dict:
    c = cap(e["mhz"]) * e["t1_ms"] * 1e-3
    cs, cf = certified["serving"]["fraction"], certified["filler"]["fraction"]
    cert = cs * e["serving_macs"] + cf * e["filler_macs"]
    return {**e, "capacity_macs": c, "certified_macs": cert, "coverage": cert / c,
            "nominal_coverage": (e["serving_macs"] + e["filler_macs"]) / c,
            "coverage_at_target": share.CERT * (e["serving_macs"] + e["filler_macs"]) / c,
            "filler_share": e["filler_macs"] / c, "time": e["t1_ms"] / e["t0_ms"],
            "tau_vf": A_VF * (e["serving_macs"] + e["filler_macs"]) / (PER_SM * SMS * share.F_MAX * e["t1_ms"] * 1e-3)}


def score(stats: dict, certified: dict = CERTIFIED) -> dict:
    """SCORE, the per-entry minimum and every entry, from {(phase, B, L): point_stats}. An entry whose points weren't
    measured is reported missing, and a run with a missing entry has no SCORE."""
    entries, missing = [], []
    for e in suite():
        if any(pt not in stats for pt in e["points"]):
            missing.append(e["entry"])
            continue
        if e["kind"] == "decode":
            r = decode_entry(stats[("decode", e["B"], e["L"])], e["B"], e["L"])
        elif e["kind"] == "prefill":
            r = prefill_entry(stats[("prefill", PREFILL_M, 0)])
        else:
            r = mix_entry(stats[("prefill", PREFILL_M, 0)], stats[("decode", e["B"], e["L"])], e["P"], e["D"], e["B"],
                          e["L"])
        entries.append({"entry": e["entry"], "kind": e["kind"], **coverage(r, certified)})
    out = {"entries": entries, "missing": missing, "certified": certified,
           "flags": [f"filler certified fraction {certified['filler']['fraction']} is {certified['filler']['status']}"]
           if certified["filler"]["status"] != "proved" and any(e["filler_share"] > 0 for e in entries) else []}
    if entries and not missing:
        w = [e["time"] for e in entries]
        out["SCORE"] = sum(e["coverage"] * x for e, x in zip(entries, w)) / sum(w)
        out["score_nominal"] = sum(e["nominal_coverage"] * x for e, x in zip(entries, w)) / sum(w)
        out["score_at_target"] = sum(e["coverage_at_target"] * x for e, x in zip(entries, w)) / sum(w)
        lo = min(entries, key=lambda e: e["coverage"])
        out["min"] = {"entry": lo["entry"], "coverage": lo["coverage"]}
        mixes = [e for e in entries if e["kind"] == "mix"]
        if mixes:
            lm = min(mixes, key=lambda e: e["coverage"])
            out["min_mix"] = {"entry": lm["entry"], "coverage": lm["coverage"]}
    return out


# ---- the gates -------------------------------------------------------------------------------------------------------------


def _ncu_rows(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    return list(csv.DictReader(io.StringIO("\n".join(lines)))) if lines else []


_UNITS = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9,
          "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1, "ns": 1e-9, "us": 1e-6, "ms": 1e-3}


def ncu_metric(text: str, metric: str, kernel: str) -> list[float]:
    """Each launch's value of `metric` (in bytes or seconds) for kernels whose name contains `kernel`."""
    out = []
    for r in _ncu_rows(text):
        if r.get("Metric Name") == metric and kernel in r.get("Kernel Name", ""):
            v = float(r["Metric Value"].replace(",", ""))
            out.append(v * _UNITS.get(r.get("Metric Unit", "byte"), 1))
    return out


def dram_gate(filler_csv: str | None, control_csv: str | None) -> dict:
    """The filler's DRAM bytes a second (ncu, its own launches) against DRAM_LIMIT of the HBM bandwidth the positive
    control measures with the same counter in the same run; the counter is valid only if the control reads its 1 GiB."""
    if filler_csv is None:
        return {"ok": True, "why": "no filler"}
    cb = ncu_metric(control_csv or "", "dram__bytes.sum", "bw_read")
    ct = ncu_metric(control_csv or "", "gpu__time_duration.sum", "bw_read")
    fb = ncu_metric(filler_csv, "dram__bytes.sum", "dfill_k")
    ft = ncu_metric(filler_csv, "gpu__time_duration.sum", "dfill_k")
    if not (cb and ct and fb and ft):
        return {"ok": False, "why": "an ncu metric is missing (control bytes/time, filler bytes/time: "
                                    f"{len(cb)}/{len(ct)}, {len(fb)}/{len(ft)})"}
    ratio = _med(cb) / CONTROL_BYTES
    rate = max(b / t for b, t in zip(fb, ft))
    valid = CONTROL_RANGE[0] <= ratio <= CONTROL_RANGE[1] and _med(ct) > 0
    if not valid:
        return {"ok": False, "control_bytes_ratio": ratio, "counter_valid": False, "filler_bytes_per_s": rate,
                "why": f"the counter's control read {ratio:.3f} of 1 GiB"}
    hbm = _med(cb) / _med(ct)
    ok = rate <= DRAM_LIMIT * hbm
    return {"ok": ok, "control_bytes_ratio": ratio, "counter_valid": valid, "hbm_bytes_per_s": hbm,
            "filler_bytes_per_s": rate, "filler_bytes_per_launch": max(fb), "share_of_hbm": rate / hbm,
            "why": None if ok else f"filler DRAM {rate / hbm:.2%} of HBM > {DRAM_LIMIT:.0%}"}


RANGE_LOG2 = 32  # dscore.cu's: round R's tiles are [R·2^32, (R + 1)·2^32)
_M64 = (1 << 64) - 1


def smix(x: int) -> int:
    """dcap.cu's smix (splitmix64's finaliser)."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def dseed(c0: int, c1: int, tile: int) -> int:
    """dcap.cu's dseed: a tile's seed from the round's coins and its id."""
    return smix(smix(c0 ^ smix(tile)) ^ c1)


def replay_gate(rows: list[dict]) -> dict:
    """Anti-replay over every filler round of the run (each point's calibration, slot 0, and each arm block, slot 1 + r):
    the round's tile range is the scorer's, [R·2^32, (R + 1)·2^32) with R = point·(rounds + 1) + slot, so no id is the
    prover's choice; its tiles start at the range's start and stay inside it; no two rounds' tiles overlap; no two rounds
    share coins; the first tile's seed is dseed(the round's coins, its id) as the device printed it; and no sampled seed
    (each round's first, last and checked tiles) repeats across rounds."""
    bad, spans = [], []
    fill_rows = [r for r in rows if r.get("place") != "none"]
    pts = [r.get("point") for r in fill_rows]
    if len(set(pts)) != len(pts) or None in pts:
        bad.append(f"point indices missing or repeated: {pts}")
    for r in fill_rows:
        rds = [r["alone_fill"].get("round")] + [b["arm"].get("round") for b in r["blocks"]]
        for s, rd in enumerate(rds):
            where = f"{r['kind']} {r['B']}/{r['L']} slot {s}"
            if rd is None or r.get("point") is None:
                bad.append(f"{where}: no round record")
                continue
            want = r["point"] * (r["rounds"] + 1) + s
            lo, hi = want << RANGE_LOG2, (want + 1) << RANGE_LOG2
            t0, t1 = rd["tiles"]
            if rd["id"] != want or rd["range"] != [lo, hi]:
                bad.append(f"{where}: range {rd['range']} is not the scorer's [{lo}, {hi})")
            if not lo == t0 < t1 <= hi:
                bad.append(f"{where}: tiles [{t0}, {t1}) not at the start of [{lo}, {hi})")
            if any(not t0 <= t < t1 for t in rd["checked"]):
                bad.append(f"{where}: a checked tile outside the round's tiles")
            c0, c1 = (int(x, 16) for x in rd["coins"])
            if f"{dseed(c0, c1, t0):016x}" != rd["seed0"]:
                bad.append(f"{where}: seed {rd['seed0']} is not dseed(the round's coins, {t0})")
            spans.append((t0, t1, (c0, c1), {dseed(c0, c1, t) for t in (t0, t1 - 1, *rd["checked"])}, where))
    spans.sort()
    overlaps = [f"{a[4]} and {b[4]}" for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    coins = [s[2] for s in spans]
    seeds = [x for s in spans for x in s[3]]
    coin_rep, seed_rep = len(coins) - len(set(coins)), len(seeds) - len(set(seeds))
    why = bad + [f"tiles overlap: {o}" for o in overlaps] + ([f"{coin_rep} rounds reuse coins"] if coin_rep else []) + \
        ([f"{seed_rep} sampled seeds repeat"] if seed_rep else []) + ([] if spans else ["no filler rounds"])
    return {"ok": not why, "rounds": len(spans), "tiles": sum(s[1] - s[0] for s in spans), "seeds_sampled": len(seeds),
            "overlaps": overlaps, "coin_repeats": coin_rep, "seed_repeats": seed_rep,
            "why": "; ".join(why[:5]) or None}


def gates(stats: dict, sc: dict, *, has_filler: bool, sgate: dict | None, dgate: list[dict], dram: dict,
          clocks: dict, honest: dict, replay: dict) -> dict:
    out = {}
    worst = []
    for e in sc["entries"]:
        for what, s in e["slowdowns"].items():
            worst.append((s, e["entry"], what))
    s, entry, what = max(worst) if worst else (float("nan"), None, None)
    out["slowdown"] = {"ok": bool(worst) and s <= LIMIT and not sc["missing"], "worst": s, "at": entry, "phase": what,
                       "limit": LIMIT}
    if has_filler:
        timed = [(k, v) for k, v in stats.items() if v["place"] != "none"]
        bad = sum(v["rec_bad"] for _, v in timed)
        checked = sum(v["rec_checked"] for _, v in timed)
        unchecked = [f"{k[0]} {k[1]}/{k[2]}" for k, v in timed if not v["rec_checked"]]
        g_ok = bool(sgate) and sgate.get("ok") is True and bool(dgate) and all(r.get("ok") for r in dgate)
        out["receipts"] = {"ok": g_ok and bad == 0 and not unchecked, "sgate": sgate,
                           "dgate": [{k: r.get(k) for k in ("NC", "S", "mismatches", "control_equal", "ok")} for r in dgate],
                           "timed_checked": checked, "timed_bad": bad, "unchecked": unchecked}
    else:
        out["receipts"] = {"ok": True, "why": "no filler"}
    out["replay"] = replay
    out["honest"] = honest
    out["dram"] = dram
    out["clocks"] = clocks
    out["ok"] = all(g["ok"] for g in out.values())
    return out


def honesty(arm, stats: dict, phases_points) -> dict:
    """The arm's placement is a pure function of the point (asked twice, the same answer), and no filler ran faster
    beside serving than alone on the same SMs (TOO_GOOD; such a result goes to the assessor)."""
    import arms as A  # noqa: PLC0415

    unstable = [f"{ph} {b}/{l}" for ph, b, l in phases_points if A.placed(arm, ph, b, l) != A.placed(arm, ph, b, l)]
    fast = [f"{k[0]} {k[1]}/{k[2]}: {v['beside_over_alone']:.3f}" for k, v in stats.items()
            if v.get("beside_over_alone") and v["beside_over_alone"] > TOO_GOOD]
    return {"ok": not unstable and not fast, "unstable_placement": unstable, "too_good": fast}


def clock_gate(stats: dict, smi: list[dict]) -> dict:
    """Clocks read and never set: every rep has an SM clock and a board power reading, the nvidia-smi log has rows, and
    no application clock differs from its default (the scorer never sets one)."""
    reps = sum(v["reps"] for v in stats.values())
    reads = sum(v["clock_reads"] for v in stats.values())
    power = sum(v["power_reads"] for v in stats.values())
    set_ = [r for r in smi if r.get("app_mhz") not in (None, r.get("default_app_mhz"))]
    ok = bool(smi) and reps > 0 and reads == reps and power == reps and not set_
    return {"ok": ok, "reps": reps, "clock_reads": reads, "power_reads": power, "smi_rows": len(smi),
            "app_clock_set": len(set_), "mhz": [min(r["mhz"] for r in smi), max(r["mhz"] for r in smi)] if smi else None,
            "watt": [min(r["watt"] for r in smi), max(r["watt"] for r in smi)] if smi else None,
            "power_limit_w": smi[0].get("limit_w") if smi else None}


def parse_smi(text: str) -> list[dict]:
    """nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,
    clocks.default_applications.graphics,clocks_event_reasons.active --format=csv,noheader,nounits rows."""
    out = []
    for ln in text.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 7:
            continue
        try:
            out.append({"mhz": float(f[0]), "max_mhz": float(f[1]), "watt": float(f[2]), "limit_w": float(f[3]),
                        "app_mhz": None if f[4] in ("[N/A]", "N/A") else float(f[4]),
                        "default_app_mhz": None if f[5] in ("[N/A]", "N/A") else float(f[5]), "reasons": f[6]})
        except ValueError:
            continue
    return out


def secondary(stats: dict, sc: dict) -> dict:
    """Reported, not scored: τ under VF per entry, the filler's per-block rate spread (work per round), power and clock
    beside serving alone, what changing the placement costs (the split's setup), and the robustness points."""
    timed = {k: v for k, v in stats.items() if v["place"] != "none"}
    rob = []
    for b, l in ROBUST:
        st = stats.get(("decode", b, l))
        if st:
            r = coverage(decode_entry(st, b, l))
            rob.append({"B": b, "L": l, "place": st["place"], "fill_sms": st["fill_sms"], "slowdown": st["slowdown"],
                        "coverage": r["coverage"], "ok": st["slowdown"] <= LIMIT})
    return {
        "tau_vf": {e["entry"]: e["tau_vf"] for e in sc["entries"]},
        "work_cv": {f"{k[0]} {k[1]}/{k[2]}": v["fill_rate_cv"] for k, v in timed.items()},
        "work_cv_max": max((v["fill_rate_cv"] for v in timed.values() if v["fill_rate_cv"] is not None), default=None),
        "power": {f"{k[0]} {k[1]}/{k[2]}": [v["alone_watt"], v["arm_watt"]] for k, v in stats.items()},
        "clock": {f"{k[0]} {k[1]}/{k[2]}": [v["alone_mhz"], v["arm_mhz"]] for k, v in stats.items()},
        "placement_setup_ms": {f"{k[0]} {k[1]}/{k[2]}": v["setup_ms"] for k, v in timed.items()},
        "robust": rob,
        "aa_slowdown": {f"{k[0]} {k[1]}/{k[2]}": v["slowdown"] for k, v in stats.items() if v["place"] == "none"},
    }


def isfinite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
