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
i.e. Σ certified ÷ Σ capacity × time. The c's are the PROVED certified fractions. Under filler-scorer/v4 (the lock) the
filler is read at the entry of `FILLER_ENTRIES` that `filler_entry_v4` decides from its declared noise and the program
facts of its build (`program_facts`; v3's `filler_entry` read `kernel_facts`): Theorem K's 0.9, Theorem E's 0.225, or a
proved floor of 0 (a chained filler's, or any filler no
proved setting covers). SCORE under CH (`UNDER_CH`: every filler at 0.99, the withdrawn assumption's reading) is a
secondary readout, comparable with filler-scorer/v2's SCORE, which read every filler at the spec's provisional 0.99
(`CERTIFIED`) and flagged it.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share  # noqa: E402  CAP's constants: MACs a token, non-GEMM times, FA2, the int8 peak

V2, V3, V4, V5 = "filler-scorer/v2", "filler-scorer/v3", "filler-scorer/v4", "filler-scorer/v5"
#: the active lock. V3 adds the arm's record checked before its lease (`arms.record_problems`), each point's slowdown
#: margin (`margins`), and the filler read at its proved entry (`FILLER_ENTRIES`, `filler_entry`) with SCORE under CH
#: beside it. V4 reads the filler wherever it's called from, filler warps inside serving kernels included
#: (`program_facts`, `filler_entry_v4`, `credit_fixed`, `paths_gate`, `whole_of_peak`). V5 (active, coordinator 12:10Z:
#: its fold check is a soundness correction, decision 43) reads the receipt's fold (`fold_reading`), the served products
#: at the MODELLED edited design's reading (`score_v5`), and the TP entries (`tp_score`); a v5 correction (coordinator
#: 13:09Z, RS-E-REF r7) reads the XOR-family folds at Theorem K's proved half (`FOLDS`, `SELECT_LEMMA`).
#: `score.py report OUT --as v2|v3|v4` reads a run as that version does
SCORER = V5
LIMIT = 1.05  # serving slowdown, every entry
DRAM_LIMIT = 0.01  # of the HBM bandwidth the same run's validated counter measures
CONTROL_BYTES = 1 << 30  # bw_read streams 1 GiB a launch: the DRAM counter's positive control
CONTROL_RANGE = (0.97, 1.10)  # dram__bytes.sum ÷ 1 GiB that validates the counter (FILL read 1.074–1.078 GB)
TOO_GOOD = 1.02  # a filler faster beside serving than alone on the same SMs by more than this goes to the assessor
TOO_GOOD_FLOOR = TOO_GOOD - 1  # beside ÷ the standalone blocks' median above 1 + max(this, their trimmed spread)
CHEAT = 16  # the honest gate's positive control credits S steps a tile and runs S - S/16 beside serving
SMS, PER_SM = share.SMS, share.PER_SM
MAX_BL = 786432

CERTIFIED = {
    "serving": {"fraction": 0.225, "status": "proved",
                "source": "certifiedFraction120θ_half, (φ/4)·N at φ = 0.9 (note:20261015T2040Z-report-relative-security-decision)"},
    "filler": {"fraction": 0.99, "status": "provisional",
               "source": "filler-scorer/v2's reading of every filler, the spec's placeholder (CH's 0.99, withdrawn in decision "
                         "27; note:20261016T0418Z-draft-filler-floor); v3 reads it at FILLER_ENTRIES"},
}
#: the certified fractions a filler can be read at (filler-scorer/v3), in the order `filler_entry` tries them: the
#: first that applies and is active is SCORE's, and an inactive one that applies ahead of it is reported with the SCORE
#: it would give. ``noise`` is the declarations an entry covers and ``kinds`` its MMA kinds, each with the most products
#: a word may sum; "chained" and "unproven" are the proved floor, 0, for what no theorem's setting covers
#: (decisions of 16 Oct 08:35Z: the primary SCORE is the proved reading).
FILLER_ENTRIES = {
    "theorem-k": {
        "fraction": 0.9, "status": "proved", "active": True, "noise": ("bits01", "bitsU", "dither"), "kinds": {"int8": 32},
        "source": "Theorem K, c = 1 at φ = 0.9: every cell-pure run with integer affine operands and s32 words read "
                  "unscaled, s32 wrap included, on uniform {0,1} bits, A ∈ {0,1} with B = U ∈ {±1..±64} (pfill-bU, "
                  "note:20261016T0710Z-draft-filler-floor-r2 §5) or the dither ⌊x/s⌋ + b; FP kinds and scaled reads by "
                  "Lemma W at VF2, at L = 32 (note:20261016T0525Z-draft-fullfloor-s-r6 §2, §4). Refereed: RS-E-REF r4 "
                  "passed K (note:20261015T1931Z-draft-theorem-e-referee Part IV). Open: the luck share of non-affine "
                  "bytes reused across products, ½ where (L) takes ¾ (IV.0 1)",
        "unchecked": ("cell-purity: a condition on the runs Theorem K bounds (its model), not on the honest kernel",)},
    "theorem-e": {
        "fraction": 0.225, "status": "proved", "active": True, "noise": ("pm1",), "kinds": {"int8": 32, "e2m1": 64},
        "source": "Theorem E, c/m = ¼ (c = ½, m_max = 2) at φ = 0.9, heads priced at 8 int8 slots: fresh i.i.d. ±1 "
                  "operands for every MMA, every MMA's sum committed; 0.225 an int8 slot for the s8 tile (L = 32) and the "
                  "e2m1 tile (L = 64, m = 2) alike (note:20261015T1845Z-draft-rs-d1e-edited-floor)",
        "unchecked": ("an e2m1 tile's block scales' values (the OMMA's UE4M3 operands), held to the host reference by "
                      "the receipts gate",)},
    "chained": {
        "fraction": 0.0, "status": "proved", "active": True,
        "source": "a chained filler's proved floor: operands or accumulators that are earlier MMAs' outputs are heads "
                  "(note:20261016T0418Z-draft-filler-floor)"},
    "unproven": {
        "fraction": 0.0, "status": "proved", "active": True,
        "source": "the trivial floor: no proved setting covers this filler (its declaration or its build's SASS)"},
}
#: SCORE under CH, a secondary readout beside v3's SCORE: every filler at CH's 0.99, the reading of filler-scorer/v2's
#: SCORE, so earlier results stay comparable. CH is withdrawn (decision 27), so this is never SCORE
UNDER_CH = {"fraction": 0.99, "status": "under CH (withdrawn, decision 27)",
            "source": "CH: a straight-line program that gets a filler tile's receipt right more often than chance pays at "
                      "least (1 − ε) of its int8 MMAs, 0.99 at 2,759 checks a window (note:20261016T0418Z-draft-filler-floor)"}
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


def fill_parts(blocks: list[dict], chunk_macs: float) -> tuple[float, float]:
    """The filler's pooled rate over `blocks` (arm or standalone block records) in its two parts: the split filler's chunks
    seen done inside a window, less one, over the span from the first to the last (rows from before span_ms: completions
    over the window, whose phase biases a ~20-chunk window by up to a chunk), and any fixed work a block's filler counted
    on the device (`fixed_macs` over `fixed_ms`: an arm-side phase filler's tiles a step, on the row's `fixed_sms`)."""
    span = all("span_ms" in b for b in blocks)
    work = [(b["chunks"] - 1, b["span_ms"]) if span else (b["chunks"], b["window_ms"]) for b in blocks]
    ok = [(n, ms) for n, ms in work if n > 0 and ms > 0]
    tot = sum(ms for _, ms in ok)
    split = sum(n for n, _ in ok) * chunk_macs / (tot * 1e-3) if tot > 0 else 0.0
    fx = [(b["fixed_macs"], b["fixed_ms"]) for b in blocks if b.get("fixed_ms")]
    fixed = sum(m for m, _ in fx) / (sum(ms for _, ms in fx) * 1e-3) if fx else 0.0
    return split, fixed


def fill_rate(blocks: list[dict], chunk_macs: float) -> tuple[float, list[float]]:
    """The filler's rate over `blocks`, pooled (`fill_parts`' sum), and each block's own."""
    rate = sum(fill_parts(blocks, chunk_macs))
    span = all("span_ms" in b for b in blocks)
    work = [(b["chunks"] - 1, b["span_ms"]) if span else (b["chunks"], b["window_ms"]) for b in blocks]
    per = []
    for b, (n, ms) in zip(blocks, work):
        x = n * chunk_macs / (ms * 1e-3) if n > 0 and ms > 0 else 0.0
        if b.get("fixed_ms"):
            x += b["fixed_macs"] / (b["fixed_ms"] * 1e-3)
        if x > 0:
            per.append(x)
    return rate, per


def solo_spread(rates: list[float]) -> float:
    """The standalone blocks' spread: their range over their median, without the single highest and lowest of four or
    more (one block whose host-side completion stamps a scheduling hiccup delayed reads up to ~15% off on a ~20-chunk
    window; the rest of a point's blocks agree within ~0.3%)."""
    xs = sorted(rates)
    if len(xs) >= 4:
        xs = xs[1:-1]
    return (xs[-1] - xs[0]) / _med(rates)


def of_peak(row: dict, reps: list) -> float:
    """The filler's arm-block rate over its SMs' int8 tensor peak at the blocks' clock, the larger of its parts': the split
    filler's on `fill_sms`, the fixed work's on `fixed_sms` (a row that counts fixed work without naming its SMs is
    bounded on `fill_sms`)."""
    split, fixed = fill_parts([b["arm"] for b in row["blocks"]], row["chunk_macs"])
    per = PER_SM * _med([r[1] for r in reps]) * 1e6
    return max(split / (per * row["fill_sms"]), fixed / (per * (row.get("fixed_sms") or row["fill_sms"])))


def point_stats(row: dict) -> dict:
    """A `dscore` row's numbers: the paired slowdown (median over rounds of the arm block's median rep over the alone
    block's), both sides' median time, clock and power, the filler's rate over its windows and its per-block spread, and
    its standalone reference: the median of the interleaved standalone blocks' rates of the split filler's chunks, with
    the too-good limit set from their spread (rows without them: the one calibration, against the whole rate, and
    TOO_GOOD). An arm-side filler's fixed work is not in that comparison: alone it runs at its full rate where beside
    serving it gets a share, which would hide an inflated split; its tiles are a count its driver checks."""
    al = [r for b in row["blocks"] for r in b["alone"]["reps"]]
    ar = [r for b in row["blocks"] for r in b["arm"]["reps"]]
    pairs = [_med([r[0] for r in b["arm"]["reps"]]) / _med([r[0] for r in b["alone"]["reps"]]) for b in row["blocks"]]
    fill = row["place"] != "none"
    rate, split, cv, solo = 0.0, 0.0, None, []
    if fill:
        rate, per = fill_rate([b["arm"] for b in row["blocks"]], row["chunk_macs"])
        split = fill_parts([b["arm"] for b in row["blocks"]], row["chunk_macs"])[0]
        cv = statistics.pstdev(per) / statistics.mean(per) if len(per) > 1 and statistics.mean(per) > 0 else None
        solo = [x for b in row["blocks"] if "solo" in b if (x := fill_parts([b["solo"]], row["chunk_macs"])[0]) > 0]
    cal = row["alone_fill"]["mac_per_s"] if fill else 0.0
    if len(solo) >= 2:
        ref, beside = _med(solo), split
        spread = solo_spread(solo)
        limit = 1 + max(TOO_GOOD_FLOOR, spread)
    else:
        ref, beside, spread, limit = cal, rate, None, TOO_GOOD
    pw = row.get("power") or {}
    return {
        "slowdown": _med(pairs), "slowdown_rounds": pairs, "alone_ms": _med([r[0] for r in al]),
        "arm_ms": _med([r[0] for r in ar]), "alone_mhz": _med([r[1] for r in al]), "arm_mhz": _med([r[1] for r in ar]),
        "alone_watt": _med([r[2] for r in al]), "arm_watt": _med([r[2] for r in ar]),
        "clock_reads": sum(1 for r in al + ar if r[1] > 0), "power_reads": sum(1 for r in al + ar if r[2] > 0),
        "reps": len(al) + len(ar), "fill_rate": rate, "fill_rate_alone": ref, "fill_rate_cv": cv,
        "solo_rates": solo, "solo_spread": spread, "too_good_limit": limit,
        "beside_over_alone": beside / ref if fill and ref else None,
        "beside_over_calibration": rate / cal if fill and cal else None,
        "power_settled": {k: pw[k]["watt"] for k in ("alone", "arm", "filler_alone") if k in pw},
        "of_peak": of_peak(row, ar) if fill and row["fill_sms"] else None,
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


def mix_parts(pre: dict, dec: dict, p: int, d: int, b: int, l: int) -> dict:
    """One request of the mix in its two phases: its prompt's share of prefill steps and D/B decode steps, alone and
    with the arm (ms), each phase's served MACs and filler MACs, and the prompt's GEMM time alone (linears and FA2)."""
    att = share.FA2_MS_8192 * share.causal(p) / share.causal(PREFILL_M)

    def tp(lin, s):
        return lin * p / PREFILL_M + share.PRE_OTHER_MS_TOK * p * s + att * s

    tp1, td1 = tp(pre["arm_ms"], pre["slowdown"]), d / b * (dec["arm_ms"] + other_ms(b) * dec["slowdown"])
    return {"tp0": tp(pre["alone_ms"], 1.0), "tp1": tp1, "td0": d / b * (dec["alone_ms"] + other_ms(b)), "td1": td1,
            "n_pre": p * share.LIN_TOK + share.LM_TOK + share.causal(p),
            "n_dec": d * (share.LIN_TOK + share.LM_TOK + share.ATT_TOK_L * l),
            "f_pre": pre["fill_rate"] * tp1 * 1e-3, "f_dec": dec["fill_rate"] * td1 * 1e-3,
            "gemm_pre_ms": pre["alone_ms"] * p / PREFILL_M + att}


def mix_entry(pre: dict, dec: dict, p: int, d: int, b: int, l: int) -> dict:
    """One request of the mix: its prompt's share of prefill steps, and D/B decode steps."""
    m = mix_parts(pre, dec, p, d, b, l)
    tp0, tp1, td0, td1 = m["tp0"], m["tp1"], m["td0"], m["td1"]
    n, f = m["n_pre"] + m["n_dec"], (pre["fill_rate"] * tp1 + dec["fill_rate"] * td1) * 1e-3
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


# ---- the filler's certified fraction (filler-scorer/v3) ---------------------------------------------------------------------

#: the dfill ABI's kernels, the functions whose SASS `kernel_facts` reads as the filler: the split filler dfill_k and the
#: phase filler dfill_p (dcap.cu's, for the leader's placement), read together whether or not the arm launches dfill_p
FILLER_KERNELS = ("dfill_k", "dfill_p")
#: SASS opcodes that compute in floating point or convert to or from it
FP_OPS = frozenset("FADD FADD2 FADD32I FMUL FMUL2 FMUL32I FFMA FFMA2 FFMA32I FMNMX FMNMX3 FSET FSETP FSEL FRND FCHK "
                   "FSWZADD MUFU F2F F2I F2IP F2FP I2F I2FP HADD2 HADD2_32I HMUL2 HMUL2_32I HFMA2 HFMA2_32I HMNMX2 "
                   "HSET2 HSETP2 DADD DMUL DFMA DSETP DMNMX DSET".split())
#: the MMA kinds an entry can cover, by mnemonic: dense int8 IMMA (S8/U8 operands, s32 accumulators, no sparsity .SP,
#: saturation .SAT or other kind) and the block-scaled E2M1 OMMA (F32 words, UE4M3 scales a 16-element block)
MMA_KINDS = {"int8": re.compile(r"IMMA\.(?P<shape>\d+)\.[SU]8\.[SU]8"),
             "e2m1": re.compile(r"OMMA\.SF\.(?P<shape>\d+)\.F32\.E2M1\.E2M1\.UE4M3\.4X")}
#: a shape's k (m16n8k32 is 16832): the products one MMA sums into each word
MMA_K = {"8816": 16, "16816": 16, "16832": 32, "16864": 64}
_SASS = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?(?P<op>[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\s*(?P<args>[^;]*);")
_CONST_HFMA2 = re.compile(r",\s*-?RZ\s*,\s*-?RZ\s*,")  # HFMA2.MMA Rd, -RZ, RZ, imm, imm: the compiler's constant load


def kernel_facts(sass: str) -> dict:
    """What the SASS of the binary the scorer built (``cuobjdump -sass``) says about its filler, the functions named
    `FILLER_KERNELS`: their MMA mnemonics, their FP and conversion opcodes, the MMAs whose accumulator input (operand C,
    the fourth) isn't RZ, and every other function with an MMA.
    - ``kind``: the one `MMA_KINDS` kind of every MMA there, or None (none, mixed, or a kind no entry covers: FP8, FP6,
      sparse, saturating, other scales).
    - ``fp_free``: no FP arithmetic or conversion (`FP_OPS`), so no word is scaled or converted through FP before the
      receipt reads it.
    - ``L``: each word's products, the one shape's k when every accumulator input is RZ (one MMA a word), else None.
    - ``accumulating_mma``: MMAs that add into an earlier word, a chain's accumulate.
    - ``mma_outside``: functions besides the filler's with an MMA (serving's GEMMs are cuBLASLt's, outside the binary)."""
    fn, mine, functions, mma, fp, acc, outside = None, False, [], {}, {}, 0, set()
    for ln in sass.splitlines():
        if "Function : " in ln:
            fn = ln.split("Function : ", 1)[1].strip()
            mine = any(k in fn for k in FILLER_KERNELS)
            if mine:
                functions.append(fn)
            continue
        m = _SASS.match(ln)
        if m is None or fn is None:
            continue
        op, args = m["op"], m["args"]
        base = op.split(".")[0]
        if base.endswith("MMA"):
            if not mine:
                outside.add(fn)
                continue
            mma[op] = mma.get(op, 0) + 1
            a = [x.strip() for x in args.split(",")]
            acc += len(a) < 4 or a[3] != "RZ"
        elif mine and base in FP_OPS and not (base == "HFMA2" and _CONST_HFMA2.search(args)):
            fp[op] = fp.get(op, 0) + 1
    kind, L = _kind_l(mma, acc)
    return {"functions": functions, "mma": dict(sorted(mma.items())), "fp_ops": dict(sorted(fp.items())),
            "accumulating_mma": acc, "mma_outside": sorted(outside), "kind": kind, "fp_free": not fp, "L": L}


def _kind_l(mma: dict, acc: int) -> tuple[str | None, int | None]:
    """The one `MMA_KINDS` kind of the MMA mnemonics `mma`, or None; and each word's products, the one shape's k when no
    MMA accumulates (`acc` 0), else None."""
    found = {next(((k, m["shape"]) for k, rx in MMA_KINDS.items() if (m := rx.fullmatch(op))), (None, None))
             for op in mma}
    kinds, shapes = {k for k, _ in found}, {s for _, s in found}
    kind = next(iter(kinds)) if len(kinds) == 1 else None
    return kind, MMA_K.get(next(iter(shapes))) if kind and len(shapes) == 1 and not acc else None


#: what `filler_entry` does not check when a theorem's entry applies, stated with the decision (each entry's own
#: ``unchecked`` adds to it); a floor of 0 needs nothing
UNCHECKED = (
    "the operand bytes' values and freshness: the alphabet, and that no MMA's operands are earlier words, are the arm's "
    "declaration (filler.noise), auditable in the variant's sources, whose host reference the receipts gate holds the "
    "device to",
    "that no word is scaled by a shift or merged with another word before the receipt: the receipt's own rotations and "
    "folds are the same integer ops")


def _theorem_checks(e: dict, noise: str | None, facts: dict | None, fixed_work: bool) -> tuple[dict, list[str]]:
    f = facts or {}
    kind, L, kinds = f.get("kind"), f.get("L"), e["kinds"]
    checks = {"noise": noise in e["noise"], "kernel facts read": facts is not None, "kind": kind in kinds,
              "fp_free": bool(f.get("fp_free")), "L": L is not None and kind in kinds and L <= kinds[kind],
              "all_read": facts is not None and not f.get("mma_outside") and not fixed_work}
    why = {"noise": f"declared noise {noise!r}, not one of {e['noise']}",
           "kernel facts read": "the build recorded no kernel facts (a build from before v3)",
           "kind": f"the filler's MMAs are {sorted(f.get('mma') or {}) or 'none'}, not all one of {sorted(kinds)}",
           "fp_free": f"FP or conversion ops in the filler's kernel: {f.get('fp_ops')}",
           "L": (f"{f['accumulating_mma']} MMAs accumulate into an earlier word, so a word may sum more than one MMA"
                 if f.get("accumulating_mma") else "no one shape for every word" if L is None
                 else f"each word sums {L} products, past {kinds[kind]}" if kind in kinds
                 else f"each word sums {L} products of a kind the entry doesn't cover"),
           "all_read": f"MMA outside the filler's kernels {FILLER_KERNELS} ({f.get('mma_outside')}) or an arm-side "
                       "driver's fixed work"}
    return checks, [why[c] for c, ok in checks.items() if not ok]


def filler_entry(noise: str | None, facts: dict | None, *, fixed_work: bool = False,
                 entries: dict = FILLER_ENTRIES, check=None) -> dict:
    """Which certified-fraction entry the filler is read at (filler-scorer/v3), from its declared operand noise and the
    kernel facts of the scorer's own build (`kernel_facts`), never the declaration alone. A theorem's entry (Theorem K's,
    then Theorem E's) applies when every check holds:
    - ``noise``: the arm declares noise the theorem covers: K's uniform {0,1} bits on both sides, bits A with U ∈
      {±1..±64} B (bitsU) or the dither ⌊x/s⌋ + b; E's fresh ±1. ±1's coefficients are even, where K is vacuous;
    - ``kind``: every MMA in the filler's kernel is one kind the theorem covers (K: dense int8 IMMA with s32 words; E:
      that or the block-scaled E2M1 OMMA);
    - ``fp_free``: no FP arithmetic or conversion there (words read unscaled, not through FP);
    - ``L``: one MMA a word (every accumulator input RZ), its k at most the kind's in the entry: K's L = 32, the L Lemma
      W prices the routes outside K at (they break even above 42.7); E's L = 32 int8 or 64 e2m1, every MMA's sum
      committed. A design that accumulates and commits every MMA's prefix sum has L = k too, but its SASS can't show
      that, so neither entry applies to it;
    - ``all_read``: no other function of the build has an MMA, and no row counts work from an arm-side driver's kernel,
      so every credited MAC is the read kernel's.
    Otherwise the proved floor, 0: "chained" when the arm declares a chain or any MMA accumulates into an earlier word,
    else "unproven". ``failed`` says why each theorem ahead of the one that applies doesn't; `UNCHECKED` and the entry's
    own ``unchecked`` are what the checks can't see. ``active`` is the entry SCORE reads: the first that applies and is
    active. ``check(entry)`` replaces these checks (filler-scorer/v4's, `filler_entry_v4`)."""
    f, checks, failed, order = facts or {}, {}, {}, []
    for name, e in entries.items():
        if "noise" in e:
            checks[name], fails = check(e) if check else _theorem_checks(e, noise, facts, fixed_work)
            if fails:
                failed[name] = fails
            else:
                order.append(name)
        elif name == "unproven" or noise == "chained" or f.get("accumulating_mma"):
            order.append(name)
    applies = order[0]
    unchecked = list(UNCHECKED) + list(entries[applies].get("unchecked", ())) if "noise" in entries[applies] else []
    return {"applies": applies, "active": next(x for x in order if entries[x]["active"]), "noise": noise,
            "fraction": entries[applies]["fraction"], "checks": checks,
            "failed": {k: v for k, v in failed.items() if list(entries).index(k) < list(entries).index(applies)},
            "unchecked": unchecked,
            "kernel": {x: f.get(x) for x in ("mma", "fp_ops", "L", "accumulating_mma", "mma_outside")}
            if facts is not None else None}


def certified(entry: str | dict, entries: dict = FILLER_ENTRIES) -> dict:
    """`CERTIFIED` with the filler read at `entry`: a name in `entries`, or an entry such as `UNDER_CH`."""
    e = entries[entry] if isinstance(entry, str) else entry
    return {"serving": CERTIFIED["serving"], "filler": {k: e[k] for k in ("fraction", "status", "source")}}


# ---- filler-scorer/v4: the filler read wherever it's called from -----------------------------------------------------

#: opcodes (their base) that can write global memory, where receipts are: stores (global, generic, surface), reductions,
#: atomics, and the TMA and bulk copies out of shared memory. Shared memory's (STS, ATOMS) and the stack's (STL) can't
STORE_OPS = frozenset("ST STG RED REDG ATOM ATOMG SUST SURED SUATOM UTMASTG UTMAREDG UBLKCP UBLKRED".split())
#: what a dfill_k device function that writes global memory may run (coordinator 11:05Z: a store-only __noinline__
#: dfill_k function, `dfill_k_put`, may take the receipt and step-count pointers if its body holds stores and address
#: arithmetic only): plain stores, address arithmetic and moves, constant-bank loads (parameters, the memory descriptor)
#: and its return. Reductions and atomics combine values, so they aren't stores here
PUT_OPS = frozenset("ST STG IADD IADD3 IADD32I LEA IMAD SHF MOV MOV32I UMOV UIADD3 ULEA LDC LDCU ULDC RET NOP".split())
#: branches whose target the listing doesn't name: a caller with one has no control-flow graph to bound
INDIRECT_OPS = frozenset("BRX JMX JMP".split())
TERMINAL_OPS = frozenset("EXIT RET BPT KILL".split())
_INSN = re.compile(r"^\s*/\*(?P<addr>[0-9a-f]{4,})\*/\s+(?P<guard>@!?U?P[T0-9]+\s+)?"
                   r"(?P<op>[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\s*(?P<args>[^;]*);")
_FUNC = re.compile(r"^\s*\.type\s+(?P<name>[^\s,]+)\s*,\s*@function")
_ENTRY = re.compile(r'^\s*\.other\s+(?P<name>[^\s,]+)\s*,\s*@"STO_CUDA_ENTRY')
_LABEL = re.compile(r"^(?P<name>[^\s:/][^\s:]*):\s*$")
_REF = re.compile(r"`\((?P<name>[^)\s]+)\)")


def parse_nvdisasm(text: str) -> list[dict]:
    """The functions of an ``nvdisasm -c`` listing, in order, each {name, entry, insns: [(address, guarded, op, args)],
    labels: {label: index of the instruction it marks}}. A kernel is an entry; nvcc compiles a __noinline__ device
    function into each kernel that calls it as its own function ``$<kernel>$<callee>`` (cuobjdump -sass prints it inside
    the kernel, unnamed), and every CALL and branch names its target."""
    lines = text.splitlines()
    declared = {m["name"] for ln in lines if (m := _FUNC.match(ln))}
    entries = {m["name"] for ln in lines if (m := _ENTRY.match(ln))}
    fns, cur = [], None
    for ln in lines:
        if ln.startswith("//--") or ln.lstrip().startswith(".section"):
            cur = None
        elif m := _LABEL.match(ln):
            if m["name"] in declared:
                cur = {"name": m["name"], "entry": m["name"] in entries, "insns": [], "labels": {}}
                fns.append(cur)
            elif cur is not None:
                cur["labels"][m["name"]] = len(cur["insns"])
        elif cur is not None and (m := _INSN.match(ln)):
            cur["insns"].append((m["addr"], bool(m["guard"]), m["op"], m["args"].strip()))
    return fns


def _step(fn: dict, i: int) -> tuple[list[int], str | None, bool]:
    """Instruction i's successors in fn, the symbol it calls (None if it calls none or an unnamed target), and whether
    its control transfer resolved. A guarded or predicated branch, EXIT or RET also falls through."""
    _, guard, op, args = fn["insns"][i]
    base, nxt = op.split(".")[0], [i + 1] if i + 1 < len(fn["insns"]) else []
    ref = _REF.search(args)
    if base == "BRA":
        t = fn["labels"].get(ref["name"]) if ref else None
        if t is None:
            return nxt, None, False
        cond = guard or bool(args.replace(ref.group(0), "").strip(" ,"))
        return [t] + nxt if cond else [t], None, True
    if base in INDIRECT_OPS:
        return nxt, None, False
    if base == "CALL":
        return nxt, ref["name"] if ref else None, True
    if base in TERMINAL_OPS:
        return nxt if guard else [], None, True
    return nxt, None, True


def _reach(starts, edges: dict) -> set:
    seen, todo = set(), list(starts)
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(edges.get(i, ()))
    return seen


def _region(fn: dict, into: set) -> tuple[set, list[int], bool]:
    """The instructions of fn a warp that calls into `into` may run besides those CALLs: every one on a path from fn's
    entry to such a CALL, and every one reachable after it returns; the CALLs; whether fn's branches all resolved."""
    n, fwd, rev, calls, ok = len(fn["insns"]), {}, {}, [], True
    for i in range(n):
        s, callee, resolved = _step(fn, i)
        ok &= resolved
        fwd[i] = s
        for j in s:
            rev.setdefault(j, []).append(i)
        if callee in into:
            calls.append(i)
    before = _reach(calls, rev) & _reach([0], fwd) if n else set()
    after = _reach([i + 1 for i in calls if i + 1 < n], fwd)
    return (before | after) - set(calls), calls, ok


def _mma_read(insns) -> tuple[dict, dict, int]:
    """(MMA mnemonics, FP and conversion opcodes, MMAs whose accumulator input isn't RZ) over `insns`, as
    `kernel_facts` counts them."""
    mma, fp, acc = {}, {}, 0
    for _, _, op, args in insns:
        base = op.split(".")[0]
        if base.endswith("MMA"):
            mma[op] = mma.get(op, 0) + 1
            a = [x.strip() for x in args.split(",")]
            acc += len(a) < 4 or a[3] != "RZ"
        elif base in FP_OPS and not (base == "HFMA2" and _CONST_HFMA2.search(args)):
            fp[op] = fp.get(op, 0) + 1
    return mma, fp, acc


def _bad(fn: dict, idx, where: str) -> list[str]:
    """What refuses an arm among fn's instructions `idx`, all outside the filler: an MMA, a store that can write global
    memory, a call to an unnamed target."""
    out = []
    for i in sorted(idx):
        addr, _, op, args = fn["insns"][i]
        base = op.split(".")[0]
        if base.endswith("MMA"):
            out.append(f"{where}: {op} at {addr}")
        elif base in STORE_OPS:
            out.append(f"{where}: {op} at {addr} (a store that can write a receipt)")
        elif base == "CALL" and not _REF.search(args):
            out.append(f"{where}: {op} {args} at {addr} (a call through a register)")
    return out


def _width(op: str) -> int:
    return 4 if ".128" in op else 2 if ".64" in op or ".WIDE" in op else 1


def _regs(operand: str, n: int) -> set[str]:
    m = re.match(r"-?(?P<u>U?)R(?P<i>\d+)\b", operand.strip())
    return {f"{m['u']}R{int(m['i']) + j}" for j in range(n)} if m else set()


def _put_bad(fn: dict, where: str) -> list[str]:
    """Why fn, a dfill_k device function with a store, isn't store-only: an instruction reachable from its entry (not
    nvcc's self-branch after the last RET) with an opcode outside `PUT_OPS`, or a store whose value an instruction of
    the body computes before it (a store-only function stores its arguments as they came; a register-to-register move,
    which the compiler places arguments with, copies one)."""
    seen, todo = set(), [0] if fn["insns"] else []
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo += _step(fn, i)[0]
    out, computed = [], set()
    for i in sorted(seen):
        addr, _, op, args = fn["insns"][i]
        base, ops = op.split(".")[0], [a.strip() for a in re.split(r",(?![^\[]*\])", args)] if args else []
        if base not in PUT_OPS:
            out.append(f"{where}: {op} at {addr} (a dfill_k function that writes global memory holds only stores and "
                       "address arithmetic)")
        elif base in ("ST", "STG"):
            if ops and _regs(ops[-1], _width(op)) & computed:
                out.append(f"{where}: {op} at {addr} stores a value the function computes, not its argument")
        elif ops and base != "RET":
            dest, src = _regs(ops[0], _width(op)), ops[-1]
            copy = (base in ("MOV", "UMOV") or (base == "IMAD" and "MOV" in op.split(".") and ops[1:3] == ["RZ", "RZ"])) \
                and bool(_RREG.match(src)) and not src.startswith(("-", "~", "|"))
            computed = (computed - dest) | (dest if not copy or _regs(src, _width(op)) & computed else set())
    return out


def program_facts(listings: dict[str, str], names: tuple = FILLER_KERNELS) -> dict:
    """filler-scorer/v4's reading of the scorer's own build: `listings` maps each binary that ran scored rows (dscore,
    and each driver an arm brings, `score.build`) to its cubins' ``nvdisasm -c`` text.
    - ``filler``: the functions whose symbol has a `FILLER_KERNELS` name (dfill_k, dfill_p), whether a kernel or a
      __noinline__ device function's clone in a serving kernel (``$<kernel>$dfill_k...``), wherever it's called from.
      The theorem entries' checks read their bodies and what they call (``helpers``) alone, as `kernel_facts` reads a
      kernel: ``mma``, ``fp_ops``, ``accumulating_mma``, ``kind``, ``fp_free``, ``L``. The functions the filler warps'
      paths call outside the filler (``path_helpers``, e.g. the u64 remainder a ring index takes, whose reciprocal is FP)
      aren't read there: they're held to no MMA and no store.
    - ``callers``: each other function that calls into the filler, directly or through a function that does, with its
      filler warps' region (`_region`): the instructions on a path from its entry to such a CALL or after one returns.
    - ``violations``, which refuse the arm (`paths_gate`): in a region, or in any function a region or the filler calls
      that isn't the filler, an MMA, a store that can write global memory (`STORE_OPS`: a receipt's), or a call through
      a register; a caller with a branch whose target isn't named; and a dfill_k device function (not a kernel) that
      writes global memory, itself or through what it calls, unless it is store-only (``stores``, `_put_bad`: stores,
      address arithmetic and moves alone, storing its arguments unmodified; such a function, `dfill_k_put`, may take
      the receipt and step-count pointers, coordinator 11:05Z), so an MMA function's inputs stay coins, tile and step.
      A guarded instruction counts as run: a warp-role split by predication instead of a branch is refused.
    - ``serving_mma``: every other function's MMAs (a serving kernel's consumer warps, kernels with no filler), outside
      the filler warps' paths, credited at the serving reading.
    - ``fixed_from``: each binary's functions that fixed work can come from, its callers and its dfill_p phase kernels
      (a dfill_k kernel is the split filler, counted in chunks): none, and its rows' fixed work isn't credited.
    `names` are the filler's names (filler-scorer/v5's TP reading passes `TP_FILLER`)."""
    fns = {(b, f["name"]): f for b, text in listings.items() for f in parse_nvdisasm(text)}
    calls = {k: {(k[0], m["name"]) for _, _, op, args in f["insns"] if op.startswith("CALL") and (m := _REF.search(args))}
             for k, f in fns.items()}
    filler = {k for k in fns if any(x in k[1] for x in names)}
    reach = set(filler)
    while grow := {k for k in fns if k not in reach and calls[k] & reach}:
        reach |= grow

    def below(starts):
        out, todo = set(), list(starts)
        while todo:
            k = todo.pop()
            if k not in out and k not in reach and k in fns:
                out.add(k)
                todo.extend(calls[k])
        return out

    own = below(c for k in filler for c in calls[k])
    violations, callers, regions, todo, puts = [], {}, {}, [], []
    for k in sorted(filler):
        if not fns[k]["entry"] and any(op.split(".")[0] in STORE_OPS for g in [k, *below(calls[k])]
                                       for _, _, op, _ in fns[g]["insns"]):
            puts.append(f"{k[0]}: {k[1]}")
            violations += _put_bad(fns[k], f"{k[1]} ({k[0]}), a dfill_k function that writes global memory")
    for k in sorted(reach - filler):
        f = fns[k]
        reg, at, ok = _region(f, {n for b, n in reach if b == k[0]})
        regions[k] = reg
        if not ok:
            violations.append(f"{k[1]} ({k[0]}): a branch whose target isn't named, so its filler warps' paths can't be "
                              "bounded")
        violations += _bad(f, reg, f"{k[1]} ({k[0]}), on its filler warps' path")
        todo += [(k[0], m["name"]) for i in reg if f["insns"][i][2].startswith("CALL")
                 and (m := _REF.search(f["insns"][i][3]))]
        callers[f"{k[0]}: {k[1]}"] = {"calls": sorted({f["insns"][i][3] for i in at}), "region": len(reg)}
    helpers = own | below(todo)
    for k in sorted(helpers):
        violations += _bad(fns[k], range(len(fns[k]["insns"])), f"{k[1]} ({k[0]}), called on the filler warps' path")
    mma, fp, acc = _mma_read(i for k in filler | own for i in fns[k]["insns"])
    kind, L = _kind_l(mma, acc)
    serving = {}
    for k, f in fns.items():
        n = sum(op.split(".")[0].endswith("MMA") for i, (_, _, op, _) in enumerate(f["insns"])
                if i not in regions.get(k, ())) if k not in filler | helpers else 0
        if n:
            serving[f"{k[0]}: {k[1]}"] = n
    fixed_from = {b: sum(k[0] == b for k in (reach - filler) | {k for k in filler if "dfill_p" in k[1]})
                  for b in listings}
    return {"version": V4, "binaries": sorted(listings), "fixed_from": fixed_from, "filler": {
        "functions": sorted(f"{b}: {n}" for b, n in filler), "stores": puts,
        "helpers": sorted(f"{b}: {n}" for b, n in own),
        "path_helpers": sorted(f"{b}: {n}" for b, n in helpers - own),
        "mma": dict(sorted(mma.items())), "fp_ops": dict(sorted(fp.items())), "accumulating_mma": acc, "kind": kind,
        "fp_free": not fp, "L": L}, "callers": callers, "serving_mma": serving, "violations": violations}


#: what filler-scorer/v4 does not check, beside `UNCHECKED`, when a theorem's entry applies
UNCHECKED_V4 = (
    "that every receipt is written by the filler: v4 sees what code can store on the filler warps' paths, not what "
    "address another path's store hits, so in a binary that calls dfill_k (without one, its fixed work isn't "
    "credited) a serving warp, or filler warps inlined into another kernel instead of calling it (FILL r3's form), "
    "could write one; the receipts gate holds every sampled receipt to the host reference whoever wrote it",
    "that a filler warp only calls dfill_k: a path from the warp-role branch that never reaches the CALL looks like the "
    "serving warps' path, so its MMAs read as serving",
    "the filler function's inputs (only the coins, the public tile id and the step, besides its stop flag): the "
    "INTERFACE's, with the receipts gate holding its outputs to the host reference's function of (coins, tile, steps); "
    "a dfill_k function that writes memory is held to stores of its arguments (`_put_bad`), but the values the filler "
    "warps pass it are computed on their path, where integer ops aren't refused",
    "the fixed work's MACs: the driver's count over its receipts (the device's step counts, each sampled one held to "
    "the host reference by its receipt and the steps - 8 control) times the driver's MACs a step",
    "the filler warps' DRAM traffic: the scorer's dram gate measures the split filler's kernels; the fused runner's "
    "own gate (fused_score.py fused_dram) measures the serving kernels with and without their filler warps",
    "that a driver binary that ran is the scorer's compile: an arm's runner may rebuild it from the same source and "
    "flags into the scorer's build directory (fused_score.py does)")


def _theorem_checks_v4(e: dict, noise: str | None, pf: dict | None, unread) -> tuple[dict, list[str]]:
    f = (pf or {}).get("filler") or {}
    kind, L, kinds = f.get("kind"), f.get("L"), e["kinds"]
    checks = {"noise": noise in e["noise"], "program facts read": pf is not None, "kind": kind in kinds,
              "fp_free": bool(f.get("fp_free")), "L": L is not None and kind in kinds and L <= kinds[kind],
              "confined": pf is not None and not pf["violations"], "drivers read": not unread}
    why = {"noise": f"declared noise {noise!r}, not one of {e['noise']}",
           "program facts read": "the build recorded no program facts (a build from before v4)",
           "kind": f"the filler's MMAs are {sorted(f.get('mma') or {}) or 'none'}, not all one of {sorted(kinds)}",
           "fp_free": f"FP or conversion ops in the filler's code: {f.get('fp_ops')}",
           "L": (f"{f['accumulating_mma']} MMAs accumulate into an earlier word, so a word may sum more than one MMA"
                 if f.get("accumulating_mma") else "no one shape for every word" if L is None
                 else f"each word sums {L} products, past {kinds[kind]}" if kind in kinds
                 else f"each word sums {L} products of a kind the entry doesn't cover"),
           "confined": f"work outside dfill_k on the filler warps' paths: {'; '.join((pf or {}).get('violations', [])[:3])}",
           "drivers read": f"rows from drivers the scorer didn't build and read: {sorted(unread)}"}
    return checks, [why[c] for c, ok in checks.items() if not ok]


def filler_entry_v4(noise: str | None, pf: dict | None, *, unread=(), entries: dict = FILLER_ENTRIES) -> dict:
    """The entry the filler is read at under filler-scorer/v4: `filler_entry`'s order and floors, its theorem checks on
    `program_facts` (the filler's code wherever it's called from, kernels and in-kernel functions alike), with
    ``confined`` (no violation on the filler warps' paths) and ``drivers read`` (every row's binary is one the scorer
    built and read, `unread` the others) in place of v3's ``all_read``: MMAs elsewhere are serving's, and fixed work is
    credited as `credit_fixed` decides."""
    e = filler_entry(noise, pf and pf["filler"], entries=entries,
                     check=lambda x: _theorem_checks_v4(x, noise, pf, unread))
    f = (pf or {}).get("filler") or {}
    e["kernel"] = {"functions": f.get("functions"), **{x: f.get(x) for x in ("mma", "fp_ops", "L", "accumulating_mma")},
                   "callers": pf["callers"], "serving_mma": pf["serving_mma"], "violations": pf["violations"]} \
        if pf else None
    if e["unchecked"]:
        e["unchecked"] += list(UNCHECKED_V4)
    return e


def credit_fixed(rows: list[dict], pf: dict | None) -> tuple[list[dict], dict]:
    """filler-scorer/v4: the rows with each block's fixed work credited only from receipts of public filler tiles that
    pass the spot checks: its row's binary is one the scorer built and read (``driver``, else dscore), with a function
    that work can come from (``fixed_from``: a caller of dfill_k, whose filler warps v4 reads, or a dfill_p kernel; not
    filler warps inlined into serving code), the filler warps' paths there are confined (no violation), the block's
    fixed round (a ``*_round`` besides the split filler's ``round``, whose range, tag and coins the replay gate holds to
    the scorer's) has spot-checked tiles, and its row has no bad receipt. Any other block's ``fixed_macs`` becomes 0.
    Returns (rows, {credited, dropped})."""
    read = set(pf["binaries"]) if pf else set()
    out, credited, dropped = [], 0, []
    for r in rows:
        r = json.loads(json.dumps(r))
        d = r.get("driver", "dscore")
        for i, b in enumerate(r["blocks"]):
            for side in ("arm", "solo"):
                x = b.get(side) or {}
                if not x.get("fixed_ms"):
                    continue
                fr = [v for k, v in x.items() if k.endswith("_round") and k != "round" and isinstance(v, dict)]
                why = ("the build recorded no program facts" if pf is None
                       else f"its driver {d} wasn't built and read" if d not in read
                       else f"no function in {d} it can come from (no dfill_k caller, no dfill_p kernel)"
                       if not pf["fixed_from"].get(d)
                       else "the filler warps' paths aren't confined" if pf["violations"]
                       else "no round for its tiles" if not fr
                       else "none of its tiles' receipts was spot-checked" if not any(v.get("checked") for v in fr)
                       else f"{r['rec_bad']} bad receipts in its row" if r.get("rec_bad") else None)
                if why:
                    x["fixed_macs"] = 0.0
                    dropped.append(f"{r['kind']} {r['B']}/{r['L']} {side} block {i}: {why}")
                else:
                    credited += 1
        out.append(r)
    return out, {"credited": credited, "dropped": dropped}


def paths_gate(pf: dict | None) -> dict:
    """filler-scorer/v4 refuses an arm whose filler warps' paths do work outside dfill_k (`program_facts`' violations),
    so filler work can't hide in serving code; without program facts nothing past the floor is credited."""
    if pf is None:
        return {"ok": True, "why": "no program facts: the filler reads its floor"}
    v = pf["violations"]
    return {"ok": not v, "callers": len(pf["callers"]), "violations": v,
            "why": None if not v else f"{len(v)} on the filler warps' paths: " + "; ".join(v[:3])}


def whole_of_peak(row: dict) -> float | None:
    """filler-scorer/v4's too-good bound on a row whose arm blocks count fixed work, over the whole kernel: serving's
    MACs a step times the blocks' steps plus the fixed work's credited MACs, over the fixed SMs' int8 peak at each
    block's clock for its window (serving runs on those SMs beside the filler warps). Above 1 no honest arm reaches."""
    blocks = [b["arm"] for b in row["blocks"] if b["arm"].get("fixed_ms")]
    if not blocks:
        return None
    n = share.decode_n(row["B"], row["L"]) if row["kind"] == "step" else PREFILL_M * share.LIN_TOK
    sms = row.get("fixed_sms") or row["fill_sms"]
    work = sum(n * len(b["reps"]) + b["fixed_macs"] for b in blocks)
    peak = sum(PER_SM * sms * _med([r[1] for r in b["reps"]]) * 1e6 * b["fixed_ms"] * 1e-3 for b in blocks)
    return work / peak if peak > 0 else None


# ---- filler-scorer/v5: the fold ----------------------------------------------------------------------------------------------

#: the receipt folds filler-scorer/v5 reads (decision 43): ``share`` is the part of a filler's committed words its
#: receipt binds, which multiplies a theorem entry's fraction; ``nc`` the NC pfill.cuh runs the fold at, ``flag`` the
#: build flag that selects it, ``sass`` how its MMA words enter the state (`fold_facts`; ``rotations`` the state's
#: rotations into each word's XOR, 0 none), ``provisional`` why the reading isn't settled (the record flags it);
#: ``share_if`` the share it reads at if ``if`` holds, a secondary readout (`fold_reading`), never SCORE
SELECT_LEMMA = ("P2-FF r10's select-charging lemma lands: under the XOR-family folds a run may choose a complement "
                "flip from the data, so Theorem K certifies only ≥ ½ (≤ about 5/8) of the words "
                "(note:20261015T1931Z-draft-theorem-e-referee, Part VII, RS-E-REF r7)")
FOLDS = {
    "ram": {"share": 0.5, "nc": (4,), "flag": None, "sass": "d0 added (LEA.HI, IADD3), d1 an IMAD addend",
            "source": "y <- (rotl(y, 7) + d0)·a0 + d1 = rotl(y, 7)·a0 + Q reads a pair only through Q = a0·d0 + d1, "
                      "one product a pair (note:20261016T0917Z-draft-fullfloor-s-r8 §1.2; pfill_ref.q_receipt)"},
    "duo": {"share": 1.0, "nc": (2,), "flag": None, "sass": "every word into an XOR (LOP3)",
            "source": "u = y ⊕ d ⊕ k; y <- u·a + z; z <- rotl(u, r) + y: no all-state kernel at one, two or three "
                      "words, proved (Lean Pouw.SecurityProofs.Receipt.DuoWordInjective, DuoPairNoKernel, "
                      "DuoTripleNoKernel), and at four words for key_ok keys at pfill's rotations (DuoQuadNoKernel); "
                      "longer runs open, and false at four words for some keys key_ok rejects "
                      "(note:20261017T1000Z-draft-fold-r §6)",
            "provisional": "binding proved for runs of one to four words "
                           "(Pouw.SecurityProofs.Receipt.DuoQuadNoKernel); five or more words, and the served "
                           "receipt's projected form, open (FOLD-R §6)"},
    "xor-in": {"share": 0.5, "share_if": 1.0, "if": SELECT_LEMMA, "nc": (4,), "flag": "-DPFILL_XIN",
               "rotations": (7, 19),
               "sass": "d0 into an XOR (LOP3) with the state rotated by 7 and 19 (SHF.L.W), d1 an IMAD addend",
               "source": "y <- (rotl(y, 7) ⊕ d0)·a0 + d1 (-DPFILL_XIN, 5e3b88559): its all-state kernel is a group of "
                         "word bijections (note:20261016T1013Z-draft-pfill-fold §2.1) whose low-bit flips mod 2 leave "
                         "Theorem K the add words, ½ proved (RS-E-REF r7, Part VII; coordinator 13:09Z)",
               "provisional": "superseded by xor-in-13: rotation 7 puts the step boundary's bad merge ratio ±128 in Λ8 "
                              "(note:20261016T1117Z-draft-fullfloor-s-r9 §1); still read"},
    "xor-in-13": {"share": 0.5, "share_if": 1.0, "if": SELECT_LEMMA, "nc": (4,), "flag": "-DPFILL_XIN13",
                  "rotations": (13, 19),
                  "sass": "d0 into an XOR (LOP3) with the state rotated by 13 and 19 (SHF.L.W), d1 an IMAD addend",
                  "source": "y <- (rotl(y, 13) ⊕ d0)·a0 + d1; y <- (rotl(y, 19) ⊕ d2)·a1 + d3 (-DPFILL_XIN13): duo's "
                            "luck under (H-μ⁺) and key_ok_r9, no layout condition (note:20261016T1117Z-draft-fullfloor-s-r9 "
                            "§1, P2-FF's served-path pick); XOR-in's group, so Theorem K's ½ proved (RS-E-REF r7, Part VII)",
                  "provisional": "P2-FF r9's DP_M ≈ 0 over Λ8 is measured, not proved, and pairwise only"},
    "lop3-imad": {"share": 0.5, "share_if": 1.0, "if": SELECT_LEMMA, "nc": (4,), "flag": "-DPFILL_LI", "rotations": (0,),
                  "sass": "d0 into a three-input XOR (LOP3) with the unrotated state and a key, d1 an IMAD addend",
                  "source": "y <- (y ⊕ d0 ⊕ k0)·a0 + d1; y <- (y ⊕ d2 ⊕ k1)·a1 + d3 (-DPFILL_LI): duo's luck under "
                            "(H-μ⁺), key_ok_r9 and (HO), which a filler's hashed rows meet "
                            "(note:20261016T1117Z-draft-fullfloor-s-r9 §2, P2-FF's filler pick); a chain kernel with "
                            "low-bit flips, so Theorem K's ½ proved (RS-E-REF r7, Part VII)",
                  "provisional": "P2-FF r9's DP_M ≈ 0 over Λ8 is measured, not proved, and pairwise only; (HO) is the "
                                 "rows' hashing, not a check"},
}
#: the lineage tags FILL-SEC's arms declare their fold with (pfill_arm.py: receipt:ram-k32, receipt:duo-k32,
#: receipt:xin-k32; r5's -DPFILL_XIN13 and -DPFILL_LI arms as receipt:xin13-… and receipt:li-…)
RECEIPT_TAGS = {"ram": "ram", "duo": "duo", "xin": "xor-in", "xin13": "xor-in-13", "li": "lop3-imad"}
#: what `fold_reading` can't confirm from the SASS, stated with every reading
FOLD_UNCHECKED = (
    "the fold's keys and rotations (odd, screened multipliers; the rotation amounts): the receipts gate holds the "
    "device to the host reference's fold of the declared NC and build flags",
    "that an XOR's other inputs carry no other word of the tile: the trace reads each word's first reader (through "
    "moves) in listing order, not along control flow, so a word combined with another before that reader (two words "
    "XORed first, a (δ, δ) kernel) shows only where both are an MMA's outputs at the same instruction",
    "XOR-in's binding: its complement elements flip three words' low bits jointly, P2-FF's open question "
    "(note:20261016T1013Z-draft-pfill-fold §2.2); decision 43 reads it in full",
    "the r9 folds' premises (note:20261016T1117Z-draft-fullfloor-s-r9): key_ok_r9 on a0 and a1 (the receipts gate's "
    "host reference draws the keys), (HO) for LOP3+IMAD (the filler's rows are SHA-512 output), DP_M ≈ 0 over Λ8 "
    "(measured, pairwise); the SASS shows the rotations, not the keys",
    "which instantiation ran: a templated dfill_k<NC> or dfill_p<NC> is read at the arm's declared NC, the one its "
    "driver selects; a filler function no NC selects (a fused or TP filler's) is read in every clone")
_NC_ARG = re.compile(r"(?:^|\$)_Z\d+dfill_[kp]ILi(?P<nc>\d+)E")
_RREG = re.compile(r"^[-~!|]*(?P<u>U?)R(?P<i>\d+)\b")
_PRED = re.compile(r"^!?U?P(?:T|\d+)$")


def _operands(args: str) -> list[str]:
    return [a.strip() for a in re.split(r",(?![^\[]*\])", args)] if args else []


def _reads(operand: str, n: int = 1) -> list[str]:
    """The registers an operand reads, in order: a register and the n - 1 after it, or those inside its brackets."""
    if "[" in operand:
        return [f"{m['u']}R{m['i']}" for m in re.finditer(r"(?P<u>U?)R(?P<i>\d+)", operand)]
    m = _RREG.match(operand)
    return [f"{m['u']}R{int(m['i']) + j}" for j in range(n)] if m else []


def _xor_on(lut: int, pos: int) -> bool:
    """Whether a LOP3 table is an XOR or XNOR of its inputs with input `pos` (0 a, 1 b, 2 c) among them, so the result
    is a bijection of that input for any value of the others."""
    v = lut ^ 0xFF if lut & 1 else lut
    for s in range(8):
        if v == (0xF0 if s & 1 else 0) ^ (0xCC if s & 2 else 0) ^ (0xAA if s & 4 else 0):
            return bool(s >> pos & 1)
    return False


def _entry(base: str, sfx: list[str], pos: list[int], srcs: list[str]) -> str:
    """How a word enters at its first reader, from the operand positions it takes there."""
    if base == "LOP3":
        lut = int(srcs[3], 16) if len(srcs) > 3 and srcs[3].startswith("0x") else None
        return "xor" if lut is not None and all(p < 3 and _xor_on(lut, p) for p in pos) else "other:LOP3"
    if base in ("IADD3", "IADD", "IADD32I"):
        return "add"
    if base == "LEA":
        return "add" if set(pos) == {1} else "other:LEA"
    if base == "IMAD" and not {"MOV", "SHL", "HI", "X"} & set(sfx):
        if "IADD" in sfx:
            return "add"
        return "mul" if set(pos) <= {0, 1} else "addend" if set(pos) == {2} else "other:IMAD"
    if base == "IMUL":
        return "mul"
    return f"other:{base}"


def _rotation(writer: dict, regs: list[str]) -> int | None:
    """The rotation (0-31) a funnel shift of one register by itself (SHF.L.W or SHF.R.W, an immediate amount) applied
    to the value in `regs`, by the instruction that last wrote it (through moves): 0 where no such shift wrote any, None
    where two did."""
    found = set()
    for r in regs:
        op, args = writer.get(r) or ("", "")
        parts, o = op.split("."), _operands(args)
        if parts[0] == "SHF" and "W" in parts and len(o) == 4 and o[1] == o[3] and o[2].startswith("0x"):
            n = int(o[2], 16) % 32
            found.add(n if "L" in parts else (32 - n) % 32)
    return found.pop() if len(found) == 1 else 0 if not found else None


def fold_words(fn: dict, rotations: list | None = None) -> list[str]:
    """How each MMA output word of fn first enters the fold, in the order the MMAs write them: by the first instruction
    that reads it (or a copy of it, MOV or IMAD.MOV), in listing order, before its register is written again:
    - ``xor``: an operand of a LOP3 whose table is an XOR in it (duo's u = y ⊕ d ⊕ k, XOR-in's rotl(y) ⊕ d0);
    - ``add``: an addend of an add (LEA.HI's or LEA's Rb, IADD3, IMAD.IADD: ram's rotl(y) + d0);
    - ``addend``: an IMAD's addend Rc (ram's and XOR-in's ·a0 + d1);
    - ``mul``: an IMAD's or IMUL's multiplicand (a word scaled by a key, as Q = a0·d0 + d1 computes it);
    - ``combined``: read with another MMA word by one instruction, other than an IMAD that multiplies one and adds
      the other (Q's form): e.g. pfill's NC 1 fold, y ^= d0 ^ d1 ^ ..., whose LOP3 reads two words;
    - ``mma``: an operand of another MMA (a chain's); ``unread``: never read; ``other:<op>`` any other first reader.
    `rotations`, when given, gets each word's state rotation where it enters an XOR (`_rotation` of that LOP3's other
    inputs: XOR-in's rotl(y, 7) or rotl(y, 13), 0 for LOP3+IMAD's unrotated y), None for any other entry."""
    live, cls, writer, rot = {}, [], {}, {}

    def kill(regs):
        for r in regs:
            w = live.pop(r, None)
            if w is not None and cls[w] is None and w not in live.values():
                cls[w] = "unread"

    for _, _, op, args in fn["insns"]:
        base, sfx, ops = op.split(".")[0], op.split(".")[1:], _operands(args)
        if base.endswith("MMA"):
            m = re.search(r"\.(\d{4,5})\.", op)
            big = not m or m[1].startswith("16")
            for o, n in zip(ops[1:4], (4, 2, 4) if big else (1, 1, 2)):
                for r in _reads(o, n):
                    if r in live and cls[live[r]] is None:
                        cls[live[r]] = "mma"
            dest = _reads(ops[0], 4 if big else 2) if ops else []
            kill(dest)
            for r in dest:
                live[r] = len(cls)
                cls.append(None)
                writer[r] = (op, args)
            continue
        d = 1 if base in ("LOP3", "PLOP3") and ops and _PRED.match(ops[0]) else 0
        has_dest = bool(ops) and base not in STORE_OPS and base not in ("STS", "STL", "BRA", "EXIT", "RET") \
            and bool(_RREG.match(ops[d]))
        srcs = [o for o in (ops[d + 1:] if has_dest else ops) if not _PRED.match(o)]
        words = {}
        for p, o in enumerate(srcs):
            for r in _reads(o):
                if r in live:
                    words.setdefault(live[r], []).append(p)
        dest = _reads(ops[d], _width(op)) if has_dest else []
        move = (base == "MOV" or (base == "IMAD" and "MOV" in sfx and srcs[:2] == ["RZ", "RZ"])) and bool(srcs) \
            and bool(_RREG.match(srcs[-1])) and not srcs[-1].startswith(("-", "~", "|"))
        if move and len(words) == 1:
            w = next(iter(words))
            kill(dest)
            live.update({r: w for r in dest})
            writer.update({r: writer.get(_reads(srcs[-1])[0]) for r in dest})
            continue
        q_form = (len(words) == 2 and base == "IMAD" and not set(sfx) - {"WIDE", "U32"}
                  and sorted(len(p) for p in words.values()) == [1, 1]
                  and sorted(p[0] for p in words.values())[1] == 2 and min(p[0] for p in words.values()) < 2)
        for w, pos in words.items():
            if cls[w] is None:
                cls[w] = "combined" if len(words) > 1 and not q_form else _entry(base, sfx, pos, srcs)
                if cls[w] == "xor":
                    rot[w] = _rotation(writer, [r for p, o in enumerate(srcs[:3]) if p not in pos for r in _reads(o)])
        kill(dest)
        src = writer.get(_reads(srcs[-1])[0]) if move else (op, args)
        writer.update({r: src for r in dest})
    if rotations is not None:
        rotations.extend(rot.get(i) for i in range(len(cls)))
    return [c or "unread" for c in cls]


def _signature(counts: dict, rotations: dict | None = None) -> str:
    """The fold a function's word entries show (`fold_words`): duo (every word an XOR); XORs and IMAD addends, one each
    a pair, are xor-in, xor-in-13 or lop3-imad by the state's rotations into the XORs (`rotations`, a count a rotation:
    7 and 19, 13 and 19, or none, as `FOLDS` lists them, each on half the XORs), else ``xor-addend``; ram (one word of
    a pair added, or scaled as Q = a0·d0 + d1 computes it, the other an IMAD addend); or what refutes every fold v5
    reads: ``combined`` or ``unread`` words; else ``unrecognised`` (dcap's probe folds, NC 7's adds alone or NC 8's
    addends alone)."""
    ks = set(counts)
    if not counts:
        return "none"
    for bad in ("combined", "unread"):
        if bad in ks:
            return bad
    if ks == {"xor"}:
        return "duo"
    if ks == {"xor", "addend"} and counts["xor"] == counts["addend"]:
        rs = {str(k): v for k, v in (rotations or {}).items()}
        for name, f in FOLDS.items():
            want = {str(r) for r in f.get("rotations", ())}
            if want and set(rs) == want and len({rs[r] for r in want}) == 1:
                return name
        return "xor-addend"
    if ks <= {"add", "addend", "mul"} and "addend" in ks and counts["addend"] == counts.get("add", 0) + counts.get("mul", 0):
        return "ram"
    return "unrecognised"


def fold_facts(listings: dict[str, str], nc: int, names: tuple = FILLER_KERNELS) -> dict:
    """filler-scorer/v5's reading of the filler's fold in the scorer's own build: in every function `program_facts`
    reads as the filler that the run executes (a templated ``dfill_k<NC>``/``dfill_p<NC>`` only at the arm's NC, any
    other filler function in every clone), how each MMA word enters the state (`fold_words`) and the fold that shows
    (`_signature`); ``signature`` is the one every such function shows, else ``mixed`` (or the worst, ``combined`` or
    ``unread``, where any function shows it)."""
    fns, skipped, templated = {}, [], False
    for b, text in listings.items():
        for f in parse_nvdisasm(text):
            if not any(x in f["name"] for x in names):
                continue
            m = _NC_ARG.search(f["name"])
            if m and int(m["nc"]) != nc:
                skipped.append(f"{b}: {f['name']}")
                continue
            templated |= bool(m)
            rot = []
            words = fold_words(f, rot)
            if words:
                counts, rots = {}, {}
                for c, r in zip(words, rot):
                    counts[c] = counts.get(c, 0) + 1
                    if c == "xor":
                        rots[str(r)] = rots.get(str(r), 0) + 1
                fns[f"{b}: {f['name']}"] = {"words": dict(sorted(counts.items())), "rotations": dict(sorted(rots.items())),
                                            "signature": _signature(counts, rots)}
    sigs = {v["signature"] for v in fns.values()}
    sig = next((s for s in ("combined", "unread") if s in sigs), None) or \
        (next(iter(sigs)) if len(sigs) == 1 else "mixed" if sigs else "none")
    return {"version": V5, "nc": nc, "templated": templated, "signature": sig, "functions": fns,
            "not_run": len(skipped)}


def declared_fold(filler: dict | None, lineage: dict | None) -> tuple[str | None, str]:
    """The arm's declared fold and where it says so: `Filler.fold`, else its lineage's one ``receipt:<fold>-…`` tag
    (`RECEIPT_TAGS`), else None (read as ram)."""
    if (filler or {}).get("fold"):
        return filler["fold"], "Filler.fold"
    tags = [t.split(":", 1)[1].split("-")[0] for t in (lineage or {}).get("mechanism") or () if t.startswith("receipt:")]
    folds = {RECEIPT_TAGS.get(t) for t in tags}
    if len(folds) == 1 and None not in folds:
        return folds.pop(), "lineage receipt tag"
    return None, "undeclared" if not tags else f"receipt tags {tags} name no one fold v5 reads"


def fold_reading(declared: str | None, nc: int | None, flags: str | None, ff: dict | None,
                 how: str = "Filler.fold") -> dict:
    """The fold filler-scorer/v5 reads the filler at (decision 43): duo in full; XOR-in (7, 19; superseded), XOR-in
    (13, 19) or LOP3+IMAD at Theorem K's proved half, with ``share_if`` (in full) under ``if`` (`SELECT_LEMMA`) as a
    secondary readout; each only when the arm declares it and the build confirms it (`_fold_problems`). Otherwise ram,
    halved: an undeclared fold, a declared ram, or a declaration the build doesn't confirm, provided the build shows a
    fold that binds at least ram's half (any `FOLDS` entry's word entries, or XORs and addends at other rotations, at an
    NC dcap checks receipts at). No fold at all (``share`` None, so no theorem entry
    applies) where it doesn't: a word combined with another before it enters the state, one the filler never reads,
    or a probe fold (NC 3, 5-8). ``unchecked`` is what the reading can't confirm."""
    out = {"declared": declared, "declared_by": how, "nc": nc, "flags": flags,
           "signature": ff and ff["signature"], "unchecked": list(FOLD_UNCHECKED)}
    claim = declared if declared in FOLDS else "ram"
    if ff is None:
        fold, why = "ram", ["the build recorded no fold facts (a build from before v5): read as ram"]
    elif not (why := _fold_problems(claim, nc, flags, ff)):
        fold = claim
        why = [] if declared == claim else ["undeclared: read as ram"] if declared is None else \
            [f"declared {declared!r}, no fold v5 reads: read as ram"]
    elif claim != "ram" and not (ram := _fold_problems("ram", nc, flags, ff)):
        fold, why = "ram", why + ["read as ram"]
    else:
        fold = None
        why = why + (ram if claim != "ram" else []) + ["the build shows no fold v5 reads: no theorem entry applies"]
    return out | {"fold": fold, "share": FOLDS[fold]["share"] if fold else None, "why": list(dict.fromkeys(why)),
                  "source": FOLDS[fold]["source"] if fold else None,
                  "provisional": FOLDS[fold].get("provisional") if fold else None,
                  "share_if": FOLDS[fold].get("share_if") if fold else None, "if": FOLDS[fold].get("if") if fold else None}


def _fold_problems(fold: str, nc: int | None, flags: str | None, ff: dict) -> list[str]:
    """Why the build doesn't confirm `fold`: the SASS shows other entries (`fold_facts`' signature; ram's reading takes
    any fold v5 reads, or XORs and addends at rotations no fold lists, each binding at least ram's half), a templated
    filler runs at another NC (ram, the XOR-ins and LOP3+IMAD 4, duo 2), or the build's recorded NVCC_APPEND_FLAGS lack
    the fold's flag (-DPFILL_XIN, -DPFILL_XIN13, -DPFILL_LI, which the receipts gate's host reference is built with)."""
    f, out, sig = FOLDS[fold], [], ff["signature"]
    if sig in ("combined", "unread"):
        bad = [f"{k} ({v['words']})" for k, v in ff["functions"].items() if v["signature"] == sig]
        return [f"the SASS shows {'a word combined with another before the fold' if sig == 'combined' else 'an MMA word the filler never reads'}: "
                + "; ".join(bad[:2])]
    if sig != fold and not (fold == "ram" and ff["functions"]
                            and all(v["signature"] in (*FOLDS, "xor-addend") for v in ff["functions"].values())):
        words = {k: {**v["words"], **({"rotations": v["rotations"]} if v.get("rotations") else {})}
                 for k, v in list(ff["functions"].items())[:2]}
        out.append(f"the SASS shows {sig} ({words}), not {fold}'s entries ({f['sass']})")
    ncs = f["nc"] if fold != "ram" else tuple(n for g in FOLDS.values() for n in g["nc"])
    if ff["templated"] and nc not in ncs:
        out.append(f"NC {nc} runs another fold: {fold} is NC {'/'.join(map(str, sorted(set(ncs))))}")
    if f["flag"] and flags is not None and f["flag"] not in flags.split():
        out.append(f"the build's NVCC_APPEND_FLAGS {flags!r} lack {f['flag']}, so the receipts gate's host reference "
                   "is ram's")
    return out


def filler_entry_v5(noise: str | None, pf: dict | None, fold: dict, *, unread=(),
                    entries: dict = FILLER_ENTRIES) -> dict:
    """filler-scorer/v5's entry: v4's (`filler_entry_v4`), with the fold (`fold_reading`) as one more theorem check,
    ``fold``, that fails where no fold applies; a theorem entry's fraction is then its fraction times the fold's
    share (ram and the XOR-family folds ½: Theorem K 0.45, Theorem E 0.1125; duo in full). The floors stay 0.
    ``if_fold`` is the secondary readout at the fold's ``share_if`` (the XOR family's 0.9 if `SELECT_LEMMA` holds)."""
    def check(x):
        checks, why = _theorem_checks_v4(x, noise, pf, unread)
        checks["fold"] = fold["share"] is not None
        return checks, why + ([] if checks["fold"] else ["fold: " + "; ".join(fold["why"])])

    e = filler_entry(noise, pf and pf["filler"], entries=entries, check=check)
    f = (pf or {}).get("filler") or {}
    e["kernel"] = {"functions": f.get("functions"), **{x: f.get(x) for x in ("mma", "fp_ops", "L", "accumulating_mma")},
                   "callers": pf["callers"], "serving_mma": pf["serving_mma"], "violations": pf["violations"]} \
        if pf else None
    if e["unchecked"]:
        e["unchecked"] += list(UNCHECKED_V4) + list(FOLD_UNCHECKED)

    def folded(name, share):
        return entries[name]["fraction"] * (share if "noise" in entries[name] else 1.0)

    e |= {"fold": fold, "unfolded": entries[e["applies"]]["fraction"], "fraction": folded(e["applies"], fold["share"]),
          "active_fraction": folded(e["active"], fold["share"])}
    e["if_fold"] = fold.get("share_if") and {"if": fold["if"], "share": fold["share_if"],
                                             "fraction": folded(e["applies"], fold["share_if"]),
                                             "active_fraction": folded(e["active"], fold["share_if"])}
    return e


def certified_v5(e: dict, which: str = "active") -> dict:
    """`CERTIFIED` with the filler read at a v5 entry (`filler_entry_v5`), its fraction folded; a provisional fold
    (`FOLDS`) makes the filler's status provisional, which `score` flags."""
    name = e[which]
    c = certified(name)
    fold, why = e["fold"]["fold"], e["fold"].get("provisional")
    c["filler"] = {**c["filler"], "fraction": e["active_fraction" if which == "active" else "fraction"],
                   "fold": fold, "unfolded": FILLER_ENTRIES[name]["fraction"]}
    if why and "noise" in FILLER_ENTRIES[name]:
        c["filler"]["status"] = f"provisional (fold {fold}: {why})"
    return c


# ---- filler-scorer/v5: the modelled served reading ------------------------------------------------------------------------

#: the served products' reading under v5 (decisions 29 and 36): the dither X = ⌊x/s⌋ + b puts the served design under
#: Theorem K, c = 1, so a noised served product reads 0.9 (it read E½ ±1's 0.225). The harness serves plain int8, so
#: the served part is MODELLED: the plain run's served MACs at this reading, its time adjusted by the factors below
SERVED_V5 = {"fraction": 0.9, "status": "proved (modelled served component)",
             "source": "decision 29 (Theorem K, c = 1 at φ = 0.9, the dither X = ⌊x/s⌋ + b) and decision 36 (decode's "
                       "deferred edit at T = 4096, prefill's per-stage live coins)"}
#: the edited GEMM ÷ plain cuBLASLt int8: edited:e_l2q 0.640 ms at o_proj (m 8192, n = k = 4096), duo receipts at L = 32
#: (r20261016-104404-700b, note:20261016T0857Z-draft-egemm); 1.478-1.50x with the quantizer on both sides
EGEMM = 1.570
#: decode's deferred noised pass (decision 36, at T = DEFER_T tokens a sub-round): the model's unit of pass time is its
#: served MACs at DEFER_OF_PEAK of the int8 peak, the compute-bound plateau of 0.97 µs a token-layer
#: (r20261016-100833-1b7c, note:20261016T0846Z-draft-stage-cost), and the edited pass takes EGEMM of those. A share s
#: of the pass's time shows as step time, and each ms of the rest displaces u·C filler MACs: s and u are measured
#: against that unit (`defer_measured`), so the model at EGEMM = 1 gives back each measured point within its fit
DEFER_T, DEFER_OF_PEAK, DEFER_US_TOKEN_LAYER = 4096, 0.71, 0.97
#: one GPU (STAGE-COST, r20261016-100833-1b7c): the pass paced beside dcap's b32 decode step (12.14 ms), by T: the
#: step's slowdown (no forming) and the pass's µs a token-layer alone. B 32 is the one batch measured
DEFER_STEP_MS, DEFER_B = 12.14, 32
DEFER_1GPU = {1024: (0.074, 1.54), 2048: (0.056, 1.18), 4096: (0.053, 1.09), 16384: (0.048, 0.97), 32768: (0.041, 0.95)}
#: TP (HC-COMM r3 at T = 1024 and 4096, note:20261016T0948Z-draft-hc-comm-r3; r4 at 8192 and 16384, TP 2 alone, its
#: TP 4 run having failed its filler gate, note:20261016T1216Z-draft-hc-comm-r4): each arm's step over the slowest rank
#: (`_slowest`), beside HC-COMM r2's filler-only phase arm at the same point (note:20261016T0846Z-draft-hc-comm-r2), per
#: GPU; r4 ran no dalone arm (None)
DEFER_TP_RUNS = {"r3": {2: "r20261016-112405-267e@88c35d347", 4: "r20261016-112756-b2a5@cbc89fb16"},
                 "r4": {2: "r20261016-121907-6830@c286d4f81"},
                 "r2": ("r20261016-092804-019d", "r20261016-093117-ef72", "r20261016-092824-3ff3",
                        "r20261016-093123-4945")}
DEFER_TP_FIELDS = ("T", "tp", "B", "L", "base_ms", "defer_ms", "dalone_ms", "mhz", "served_macs", "defer_fill_macs",
                   "phase_slowdown", "phase_fill_macs")
DEFER_TP = [dict(zip(DEFER_TP_FIELDS, p)) for p in (
    (1024, 2, 32, 2048, 39.3010, 42.0805, 4.1787, 2081, 1.154977e+12, 7.375608e+11, 1.03130, 1.865810e+12),
    (1024, 2, 32, 8192, 50.5118, 53.2739, 4.6321, 2088, 1.283826e+12, 6.326382e+11, 1.02284, 1.877719e+12),
    (1024, 2, 64, 2048, 48.4606, 53.5460, 8.2509, 2088, 2.309954e+12, 4.050767e+11, 1.02420, 2.968836e+12),
    (1024, 2, 64, 8192, 71.3236, 77.0724, 9.1319, 2092, 2.567652e+12, 4.135474e+11, 1.01809, 3.028382e+12),
    (1024, 2, 128, 2048, 66.1285, 78.9039, 16.4402, 2078, 4.619908e+12, 2.067378e+12, 1.01812, 5.069118e+12),
    (1024, 2, 256, 2048, 98.5399, 127.7737, 32.7219, 2055, 9.239817e+12, 5.889842e+12, 1.01282, 9.252450e+12),
    (1024, 4, 32, 2048, 28.9736, 30.3966, 2.1302, 2092, 5.774886e+11, 2.311736e+12, 1.02965, 3.622696e+12),
    (1024, 4, 32, 8192, 34.6352, 36.1472, 2.3468, 2092, 6.419131e+11, 2.276712e+12, 1.02298, 3.620514e+12),
    (1024, 4, 64, 2048, 34.1937, 36.4868, 4.2013, 2092, 1.154977e+12, 2.547677e+12, 1.03446, 4.222575e+12),
    (1024, 4, 128, 2048, 47.4656, 51.5732, 8.3144, 2084, 2.309954e+12, 3.998072e+12, 1.02730, 6.952288e+12),
    (1024, 4, 128, 8192, 70.0444, 76.4922, 9.1830, 2085, 2.567652e+12, 4.104480e+12, 1.01927, 6.946705e+12),
    (1024, 4, 256, 2048, 77.5716, 90.4144, 16.5191, 2070, 4.619908e+12, 8.623566e+12, 1.01572, 1.206320e+13),
    (4096, 2, 32, 2048, 39.3144, 43.5310, 3.6181, 2084, 1.154977e+12, 1.468658e+12, 1.03130, 1.865810e+12),
    (4096, 2, 32, 8192, 50.5279, 54.8699, 4.0084, 2092, 1.283826e+12, 1.466016e+12, 1.02284, 1.877719e+12),
    (4096, 2, 64, 2048, 48.4948, 54.3338, 7.2187, 2087, 2.309954e+12, 1.922749e+12, 1.02420, 2.968836e+12),
    (4096, 2, 64, 8192, 71.3513, 77.5235, 7.9538, 2092, 2.567652e+12, 1.898832e+12, 1.01809, 3.028382e+12),
    (4096, 2, 128, 2048, 66.3771, 73.4397, 14.2900, 2070, 4.619908e+12, 2.068915e+12, 1.01812, 5.069118e+12),
    (4096, 2, 256, 2048, 99.8174, 105.9941, 28.4423, 2056, 9.239817e+12, 7.702352e+11, 1.01282, 9.252450e+12),
    (4096, 4, 32, 2048, 29.9831, 32.0820, 1.9042, 2092, 5.774886e+11, 3.122613e+12, 1.02965, 3.622696e+12),
    (4096, 4, 32, 8192, 35.5854, 37.7931, 2.0775, 2092, 6.419131e+11, 3.155913e+12, 1.02298, 3.620514e+12),
    (4096, 4, 64, 2048, 34.2244, 37.2136, 3.6234, 2092, 1.154977e+12, 3.419280e+12, 1.03446, 4.222575e+12),
    (4096, 4, 128, 2048, 47.3770, 50.5316, 7.2119, 2087, 2.309954e+12, 4.705281e+12, 1.02730, 6.952288e+12),
    (4096, 4, 128, 8192, 70.1326, 73.0792, 7.9508, 2085, 2.567652e+12, 4.504076e+12, 1.01927, 6.946705e+12),
    (4096, 4, 256, 2048, 76.8918, 80.2608, 14.2907, 2074, 4.619908e+12, 6.624876e+12, 1.01572, 1.206320e+13),
    (8192, 2, 32, 2048, 39.2983, 43.6173, None, 2092, 1.154977e+12, 1.629492e+12, 1.03130, 1.865810e+12),
    (8192, 2, 32, 8192, 50.5881, 55.0130, None, 2088, 1.283826e+12, 1.659554e+12, 1.02284, 1.877719e+12),
    (8192, 2, 64, 2048, 48.7080, 55.4561, None, 2088, 2.309954e+12, 2.435490e+12, 1.02420, 2.968836e+12),
    (8192, 2, 64, 8192, 71.7139, 79.1097, None, 2092, 2.567652e+12, 2.464857e+12, 1.01809, 3.028382e+12),
    (8192, 2, 128, 2048, 66.4863, 78.2559, None, 2078, 4.619908e+12, 4.060935e+12, 1.01812, 5.069118e+12),
    (8192, 2, 256, 2048, 100.1212, 121.4479, None, 2055, 9.239817e+12, 6.356580e+12, 1.01282, 9.252450e+12),
    (16384, 2, 32, 2048, 39.3099, 43.3963, None, 2089, 1.154977e+12, 1.596318e+12, 1.03130, 1.865810e+12),
    (16384, 2, 32, 8192, 50.5521, 54.8286, None, 2085, 1.283826e+12, 1.615910e+12, 1.02284, 1.877719e+12),
    (16384, 2, 64, 2048, 48.7568, 55.1174, None, 2088, 2.309954e+12, 2.423228e+12, 1.02420, 2.968836e+12),
    (16384, 2, 64, 8192, 71.7421, 79.2344, None, 2092, 2.567652e+12, 2.576298e+12, 1.01809, 3.028382e+12),
    (16384, 2, 128, 2048, 66.4212, 79.0456, None, 2089, 4.619908e+12, 4.353902e+12, 1.01812, 5.069118e+12),
    (16384, 2, 256, 2048, 100.1074, 123.2141, None, 2055, 9.239817e+12, 7.506000e+12, 1.01282, 9.252450e+12))]
#: HC-COMM r6's deferred pass (note:20261016T1432Z-draft-hc-comm-r6, d6d0e4e57): the duo fold's put-split filler at
#: T = 4096 and TPC_DMAXC 1 (r3's), its base, phase, defer and dalone arms in one set, over the slowest rank; its fill
#: counters are d6d0e4e57's (76c68bcf2's, HC-COMM r5's, kept each collective's last count, 22-35x low)
DEFER_TP_RUNS["r6"] = {2: "r20261016-142613-eff2@d6d0e4e57, r20261016-143248-9563@d6d0e4e57",
                       4: "r20261016-142933-e6c4@d6d0e4e57"}
DEFER_TP_R6 = [dict(zip(DEFER_TP_FIELDS, p)) for p in (
    (4096, 2, 32, 2048, 39.4189, 44.8212, 4.8656, 2091.7, 1.154977e+12, 1.110414e+12, 1.04044, 1.394642e+12),
    (4096, 2, 256, 2048, 100.2084, 110.9293, 37.7894, 2055.3, 9.239817e+12, 2.957386e+11, 1.01171, 6.681906e+12),
    (4096, 2, 32, 8192, 50.6277, 56.1886, 5.4751, 2092.0, 1.283826e+12, 1.087498e+12, 1.02957, 1.396081e+12),
    (4096, 2, 64, 2048, 49.0053, 57.2022, 9.6383, 2091.5, 2.309954e+12, 1.394104e+12, 1.03682, 2.105104e+12),
    (4096, 2, 64, 8192, 71.9169, 80.1428, 10.6793, 2092.0, 2.567652e+12, 1.363410e+12, 1.02370, 2.138440e+12),
    (4096, 2, 128, 2048, 66.7963, 77.0255, 19.0254, 2079.05, 4.619908e+12, 1.537026e+12, 1.01985, 3.577272e+12),
    (4096, 4, 32, 2048, 30.2300, 33.0232, 2.5548, 2092.0, 5.774886e+11, 2.153635e+12, 1.03945, 2.421171e+12),
    (4096, 4, 32, 8192, 35.8321, 38.4280, 2.7825, 2092.0, 6.419131e+11, 2.074844e+12, 1.03020, 2.451717e+12),
    (4096, 4, 64, 2048, 34.2544, 38.1180, 4.8688, 2092.0, 1.154977e+12, 2.427403e+12, 1.03729, 2.889602e+12),
    (4096, 4, 128, 2048, 48.2082, 52.7303, 9.6649, 2092.0, 2.309954e+12, 3.442700e+12, 1.02612, 5.018321e+12),
    (4096, 4, 128, 8192, 70.8932, 75.4798, 10.6993, 2085.0, 2.567652e+12, 3.262757e+12, 1.01883, 5.040100e+12),
    (4096, 4, 256, 2048, 79.2017, 82.0726, 19.0380, 2070.0, 4.619908e+12, 4.241883e+12, 1.02302, 8.944750e+12))]
#: the (B, L) range the TP points span; the fit is read at B and L clamped to it
DEFER_B_RANGE, DEFER_L_RANGE = (32, 256), (2048, 8192)


def defer_shown_1gpu(T: int = DEFER_T) -> float:
    """One GPU's shown share at T (STAGE-COST): the b32 step's slowdown over the pass's 32 × LAYERS token-layers at
    the model's pass rate (0.648 at T = 4096)."""
    return DEFER_1GPU[T][0] * DEFER_STEP_MS / (DEFER_B * share.LAYERS * DEFER_US_TOKEN_LAYER * 1e-3)


def defer_measured(p: dict) -> dict:
    """A TP point's (`DEFER_TP`) deferred pass against the model's unit: ``plain_ms``, its served MACs at DEFER_OF_PEAK
    of peak; ``added_ms``, the deferred arm's step beyond serving with the filler (base × r2's phase slowdown);
    ``lost_macs``, the filler MACs it gave up against the phase arm (r2's, or the point's own set); s = added ÷ plain,
    u = lost ÷ C × the hidden ms (plain − added), and d = lost ÷ C × the whole pass's ms (u(1 − s), but defined where
    s ≈ 1, where u isn't)."""
    c = cap(p["mhz"])
    plain = p["served_macs"] / (DEFER_OF_PEAK * c) * 1e3
    added = p["defer_ms"] - p["base_ms"] * p["phase_slowdown"]
    lost = p["phase_fill_macs"] - p["defer_fill_macs"]
    return {"plain_ms": plain, "added_ms": added, "lost_macs": lost, "s": added / plain,
            "u": lost / ((plain - added) * c * 1e-3), "d": lost / (plain * c * 1e-3)}


def _lstsq(rows: list[list[float]], y: list[float]) -> list[float]:
    """Least squares through the normal equations, solved by Gauss-Jordan with partial pivoting (a few columns)."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] + [sum(r[i] * v for r, v in zip(rows, y))]
         for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[p] = a[p], a[i]
        for k in range(n):
            if k != i:
                f = a[k][i] / a[i][i]
                a[k] = [x - f * z for x, z in zip(a[k], a[i])]
    return [a[i][n] / a[i][i] for i in range(n)]


def defer_fit(T: int = DEFER_T, points: list[dict] | None = None, form: str = "u") -> dict:
    """The deferred pass's s and its displacement fitted to the TP points at T (`defer_measured`): ln s and ln u (form
    "u") or ln d (form "d") each = an intercept a TP + a batch slope × ln(B/32) + a context slope × ln(L/2048), by least
    squares. Each point's residual is the fit's error, fit ÷ measured − 1 (positive overstates the added time or the
    displacement, the conservative side), the added time's error in ms and in points of the step's slowdown, and the
    filler's remaining share of C the fit gives back at the harness's plain pass (s at most 1) against the defer arm's,
    in points (``fill_err_pts``); ``rms`` is their relative rms."""
    pts = [p for p in (points if points is not None else DEFER_TP) if p["T"] == T]
    if not pts:
        raise ValueError(f"no deferred-pass TP points at T = {T}: measured at "
                         f"{sorted({p['T'] for p in (points if points is not None else DEFER_TP)})}")
    qs = ("s", form)
    tps = sorted({p["tp"] for p in pts})
    meas = [defer_measured(p) for p in pts]
    rows = [[float(p["tp"] == t) for t in tps] + [math.log(p["B"] / 32), math.log(p["L"] / 2048)] for p in pts]
    out = {"T": T, "form": form, "tps": tps, "n": len(pts)}
    for q in qs:
        coef = _lstsq(rows, [math.log(m[q]) for m in meas])
        out[q] = {"intercept": {t: math.exp(c) for t, c in zip(tps, coef)}, "batch": coef[-2], "context": coef[-1]}
    res = []
    for p, m, r in zip(pts, meas, rows):
        fit = {q: math.exp(sum(c * x for c, x in zip([math.log(out[q]["intercept"][t]) for t in tps]
                                                    + [out[q]["batch"], out[q]["context"]], r))) for q in qs}
        c, s = cap(p["mhz"]) * 1e-3, min(1.0, fit["s"])
        disp = (fit["u"] * (1 - s) if form == "u" else fit["d"]) * m["plain_ms"] * c
        t1 = p["base_ms"] * p["phase_slowdown"] + s * m["plain_ms"]
        share_fit = max(0.0, p["phase_fill_macs"] - disp) / (c * t1)
        share = p["defer_fill_macs"] / (c * p["defer_ms"])
        res.append({"tp": p["tp"], "B": p["B"], "L": p["L"],
                    **{k: v for q in qs for k, v in ((q, m[q]), (f"{q}_fit", fit[q]), (f"{q}_resid", fit[q] / m[q] - 1))},
                    "added_ms": m["added_ms"], "added_err_ms": (s - m["s"]) * m["plain_ms"],
                    "slowdown_pts": 100 * m["added_ms"] / p["base_ms"],
                    "slowdown_err_pts": 100 * (s - m["s"]) * m["plain_ms"] / p["base_ms"],
                    "fill_share": share, "fill_share_fit": share_fit, "fill_err_pts": 100 * (share_fit - share)})
    for q in qs:
        out[q]["rms"] = math.sqrt(statistics.mean(x[f"{q}_resid"] ** 2 for x in res))
    out["slowdown_err_pts_max"] = max(abs(x["slowdown_err_pts"]) for x in res)
    out["fill_err_pts_rms"] = math.sqrt(statistics.mean(x["fill_err_pts"] ** 2 for x in res))
    out["fill_err_pts_max"] = max(abs(x["fill_err_pts"]) for x in res)
    out["points"] = res
    return out


#: one GPU's displacement: r3's fit (the ram fold) in the u form, TP 2's u the floor of `defer_rates`'s one-GPU rule
DEFER_FIT = defer_fit()
#: TP's deferred pass (coordinator 14:42Z): s and d fitted jointly to HC-COMM r3 (ram) and r6 (duo) at T = 4096 in the
#: d form, displacement per ms of the whole pass, since u = d ÷ (1 − s) is ill-conditioned where s ≈ 1 (r6's TP 2 B 32)
DEFER_FIT_TP = defer_fit(points=DEFER_TP + DEFER_TP_R6, form="d")


def defer_at(fit: dict, q: str, tp: int, B: int | None = None, L: int | None = None) -> float:
    """A `defer_fit` fit's q at (TP, B, L), B and L clamped to the points' range (`DEFER_B_RANGE`, `DEFER_L_RANGE`)."""
    b = min(max(B or DEFER_B, DEFER_B_RANGE[0]), DEFER_B_RANGE[1])
    l = min(max(L or 2048, DEFER_L_RANGE[0]), DEFER_L_RANGE[1])
    if tp not in fit[q]["intercept"]:
        raise ValueError(f"the deferred pass is calibrated at TP {fit['tps']}, not TP {tp}: add HC-COMM's points "
                         f"at TP {tp} to DEFER_TP")
    return fit[q]["intercept"][tp] * (b / 32) ** fit[q]["batch"] * (l / 2048) ** fit[q]["context"]


def defer_rates(tp: int = 1, B: int | None = None, L: int | None = None, fit: dict = DEFER_FIT,
                fit_tp: dict = DEFER_FIT_TP) -> dict:
    """The deferred pass's shown share s (in [0, 1]) and displacement d (filler MACs lost a ms of the whole pass, over
    C) at (TP, B, L), T = DEFER_T, read at B and L clamped to the points' range. TP: `DEFER_FIT_TP` (r3 and r6, the d
    form). One GPU: STAGE-COST's share, measured at B 32 only and held flat in B and L (the TP points' share falls with
    B at T = 4096, so this overstates it above B 32); its displacement, which no one-GPU run measured, is u(1 − s) at
    u the larger of TP 2's r3 fit (`DEFER_FIT`, the smallest TP measured) and the MAC-for-MAC rule's DEFER_OF_PEAK (the
    hidden time was the filler's, displaced at the pass's own rate)."""
    b = min(max(B or DEFER_B, DEFER_B_RANGE[0]), DEFER_B_RANGE[1])
    l = min(max(L or 2048, DEFER_L_RANGE[0]), DEFER_L_RANGE[1])
    if tp == 1:
        s = min(1.0, defer_shown_1gpu(fit["T"]))
        u = max(DEFER_OF_PEAK, defer_at(fit, "u", min(fit["tps"]), b, l))
        return {"s": s, "u": u, "d": u * (1 - s), "B": b, "L": l}
    return {"s": min(1.0, defer_at(fit_tp, "s", tp, b, l)), "d": defer_at(fit_tp, "d", tp, b, l), "B": b, "L": l}
#: prefill's per-stage live coins over its GEMMs at 32K tokens a draw (commit then GEMM a stage: +13%,
#: r20261016-101324-cab4, note:20261016T0846Z-draft-stage-cost; STAGE-OVL and STAGE-COST r2 may lower it): a parameter
PREFILL_COINS = 0.13


def served_model(constants: dict | None = None) -> dict:
    """The modelled served component's constants, each with its source, as a run's report records them."""
    def fit(f, qs):
        return {q: {"intercept": {str(t): v for t, v in f[q]["intercept"].items()}, "batch": f[q]["batch"],
                    "context": f[q]["context"], "rms": f[q]["rms"]} for q in qs}

    c = {"fraction": SERVED_V5["fraction"], "EGEMM": EGEMM, "DEFER_OF_PEAK": DEFER_OF_PEAK, "DEFER_T": DEFER_T,
         "DEFER_SHOWN_1GPU": defer_shown_1gpu(), "DEFER_FIT": fit(DEFER_FIT, ("u",)),
         "DEFER_FIT_TP": fit(DEFER_FIT_TP, ("s", "d")), "PREFILL_COINS": PREFILL_COINS} | (constants or {})
    return {"constants": c, "sources": {
        "fraction": SERVED_V5["source"],
        "EGEMM": "edited:e_l2q 0.640 ms = 1.570x plain cuBLASLt int8 at o_proj (r20261016-104404-700b, "
                 "note:20261016T0857Z-draft-egemm)",
        "DEFER_OF_PEAK": "the model's unit of pass time: the deferred pass's plateau, 0.97 µs a token-layer at 71% of int8 "
                         "peak (r20261016-100833-1b7c, note:20261016T0846Z-draft-stage-cost)",
        "DEFER_T": "decision 36's sub-round for decode (note:20261016T0846Z-draft-stage-cost's recommendation)",
        "DEFER_SHOWN_1GPU": f"one GPU, measured: +{DEFER_1GPU[DEFER_T][0]:.1%} of the {DEFER_STEP_MS} ms b32 step for its "
                            f"{DEFER_B} x {share.LAYERS} token-layers in the model's unit (T = {DEFER_T}, "
                            "r20261016-100833-1b7c); B 32 only, held flat in B and L",
        "DEFER_FIT": f"one GPU's displacement, which no one-GPU run measured: u(1 - s), u the larger of the MAC-for-MAC "
                     f"rule's DEFER_OF_PEAK and TP {min(DEFER_FIT['tps'])}'s u fitted by TP, ln(B/32) and ln(L/2048) to "
                     f"HC-COMM r3's deferred arm (ram) beside r2's phase arm at T = {DEFER_T} "
                     f"({', '.join(DEFER_TP_RUNS['r3'].values())}; r2 {', '.join(DEFER_TP_RUNS['r2'])}; "
                     "note:20261016T0948Z-draft-hc-comm-r3)",
        "DEFER_FIT_TP": f"TP, measured: s and d (filler MACs lost a ms of the whole pass, over C) fitted by TP, ln(B/32) "
                        f"and ln(L/2048) to HC-COMM r3 (above) and r6's duo deferred arm beside its own phase arm at "
                        f"T = {DEFER_T} ({'; '.join(DEFER_TP_RUNS['r6'].values())}; "
                        "note:20261016T1432Z-draft-hc-comm-r6), read at B and L clamped to "
                        f"{DEFER_B_RANGE} x {DEFER_L_RANGE}; the filler's remaining share it gives back at the plain "
                        f"pass is within {DEFER_FIT_TP['fill_err_pts_max']:.1f} points of C "
                        f"(rms {DEFER_FIT_TP['fill_err_pts_rms']:.1f})",
        "PREFILL_COINS": "commit then GEMM a stage at 32K tokens: +13% over the GEMMs (r20261016-101324-cab4, "
                         "note:20261016T0846Z-draft-stage-cost); a parameter (STAGE-OVL, STAGE-COST r2)"},
        "model": "decode: the plain step as measured plus the deferred pass of its served MACs a GPU, EGEMM x their time "
                 "at DEFER_OF_PEAK of peak, its shown share s (one GPU DEFER_SHOWN_1GPU, TP DEFER_FIT_TP) added to the "
                 "wall time and each ms of it displacing d x C filler MACs (one GPU u(1 - s), DEFER_FIT; TP "
                 "DEFER_FIT_TP; at most the filler's); prefill: the "
                 "measured step plus (EGEMM - 1 + PREFILL_COINS) x its GEMMs' time alone (linears and FA2), the "
                 "filler's MACs as measured; every served MAC at fraction"}


def defer_pass(macs: float, mhz: float, tp: int = 1, B: int | None = None, L: int | None = None) -> dict:
    """Decode's deferred noised pass over `macs` served products a GPU at (TP, B, L): its time (EGEMM × the model's
    unit), the ms of wall time it adds (s of it) and the filler MACs it displaces (d·C a ms of the pass)."""
    r = defer_rates(tp, B, L)
    pass_ms = EGEMM * macs / (DEFER_OF_PEAK * cap(mhz)) * 1e3
    return {"pass_ms": pass_ms, "s": r["s"], "d": r["d"], "added_ms": r["s"] * pass_ms,
            "displaces": r["d"] * pass_ms * cap(mhz) * 1e-3}


def entry_v5(e: dict, stats: dict, coins: float = PREFILL_COINS) -> dict:
    """A `suite` entry under v5: its v4 arithmetic (`decode_entry`, `prefill_entry`, `mix_entry`), then the modelled
    served component's time (`served_model`): the wall time with the arm ``t1_ms`` grows by prefill's edit and coins
    and decode's deferred pass's shown part, and decode's filler MACs lose what the pass's hidden part displaces."""
    if e["kind"] == "decode":
        st = stats[("decode", e["B"], e["L"])]
        r = decode_entry(st, e["B"], e["L"])
        n_pre, f_pre, gemm, n_dec, f_dec = 0.0, 0.0, 0.0, r["serving_macs"], r["filler_macs"]
    elif e["kind"] == "prefill":
        st = stats[("prefill", PREFILL_M, 0)]
        r = prefill_entry(st)
        n_pre, f_pre, gemm, n_dec, f_dec = r["serving_macs"], r["filler_macs"], st["alone_ms"] + share.FA2_MS_8192, 0.0, 0.0
    else:
        pre, dec = stats[("prefill", PREFILL_M, 0)], stats[("decode", e["B"], e["L"])]
        r = mix_entry(pre, dec, e["P"], e["D"], e["B"], e["L"])
        m = mix_parts(pre, dec, e["P"], e["D"], e["B"], e["L"])
        n_pre, f_pre, gemm, n_dec, f_dec = m["n_pre"], m["f_pre"], m["gemm_pre_ms"], m["n_dec"], m["f_dec"]
    dp = defer_pass(n_dec, r["mhz"], 1, e.get("B"), e.get("L")) if n_dec else \
        {"pass_ms": 0.0, "added_ms": 0.0, "displaces": 0.0, "s": None, "d": None}
    pre_ms = (EGEMM - 1 + coins) * gemm
    return {**r, "t1_measured_ms": r["t1_ms"], "filler_measured_macs": r["filler_macs"],
            "t1_ms": r["t1_ms"] + pre_ms + dp["added_ms"], "filler_macs": f_pre + max(0.0, f_dec - dp["displaces"]),
            "served_model": {"prefill_added_ms": pre_ms, "defer_pass_ms": dp["pass_ms"], "defer_added_ms": dp["added_ms"],
                             "defer_s": dp["s"], "defer_d": dp["d"], "displaced_macs": min(f_dec, dp["displaces"])}}


def coverage_v5(e: dict, filler_fraction: float) -> dict:
    """An `entry_v5` entry's coverage in its two parts, each over C × its v5 wall time: the measured filler part
    (``filler_part``, its MACs at the folded fraction) and the modelled served part (``served_part_modelled``, every served MAC
    at `SERVED_V5`'s 0.9)."""
    c = cap(e["mhz"]) * e["t1_ms"] * 1e-3
    fp, sp = filler_fraction * e["filler_macs"] / c, SERVED_V5["fraction"] * e["serving_macs"] / c
    return {**e, "capacity_macs": c, "filler_part": fp, "served_part_modelled": sp, "coverage": fp + sp,
            "filler_share": e["filler_macs"] / c, "time": e["t1_ms"] / e["t0_ms"]}


def score_v5(stats: dict, filler_fraction: float, coins: float = PREFILL_COINS) -> dict:
    """filler-scorer/v5's SCORE: the time-weighted mean of the entries' coverage (`coverage_v5`, weights each entry's
    v5 wall time over its serving-alone one), shown as its measured filler part plus its modelled served part. An
    entry whose points weren't measured is missing, and a run with one has no SCORE."""
    entries, missing = [], []
    for e in suite():
        if any(pt not in stats for pt in e["points"]):
            missing.append(e["entry"])
            continue
        entries.append({"entry": e["entry"], "kind": e["kind"], **{k: e[k] for k in ("B", "L", "P", "D") if k in e},
                        **coverage_v5(entry_v5(e, stats, coins), filler_fraction)})
    out = {"entries": entries, "missing": missing, "served_model": served_model({"PREFILL_COINS": coins})}
    if entries and not missing:
        w = [e["time"] for e in entries]
        for k, name in (("coverage", "SCORE"), ("filler_part", "filler_part"), ("served_part_modelled", "served_part_modelled")):
            out[name] = sum(e[k] * x for e, x in zip(entries, w)) / sum(w)
        lo = min(entries, key=lambda e: e["coverage"])
        out["min"] = {"entry": lo["entry"], "coverage": lo["coverage"]}
    return out


# ---- filler-scorer/v5: SCORE-TP ----------------------------------------------------------------------------------------------

#: HC-COMM's TP suite (note:20261016T0846Z-draft-hc-comm-r2, round 2): Llama-3.1-70B decode under tensor parallelism on
#: GPUs of one NUMA node, at (TP, B, L); TP 2 can't hold B 128 at 8K (its KV wraps)
TP_SUITE = [(4, 32, 2048), (4, 64, 2048), (4, 128, 2048), (4, 256, 2048), (4, 32, 8192), (4, 128, 8192),
            (2, 32, 2048), (2, 64, 2048), (2, 128, 2048), (2, 256, 2048), (2, 32, 8192), (2, 64, 8192)]
#: the TP filler's function under the dfill_k interface: tpcomm's __noinline__ tp_dfill_k (898d7dbf3), kfill_k its
#: launching kernel; tpcomm.cu includes dcap.cu, whose dfill_k kernels the TP driver never launches
TP_FILLER = ("tp_dfill_k",)
TP_NCU_KERNEL = "kfill_k"  # the kernel tpcomm.sh's ncu profiles (`-k regex:kfill_k`)
TP_REPS, TP_PF_KINDS, TP_NCCL = 9, ("cut", "fixed", "up"), "NCCL_CGA_CLUSTER_SIZE=1"


def tp_rows(text_by_rank: dict[int, str]) -> dict:
    """One tpcomm.sh set's rank logs (``set*/rank<r>.jsonl``) as {(tp, B, L): {arm: [[each rank's record] per
    repeat]}}, and every rank's ``pf_gate`` and ``point`` records."""
    arms, pf, pts = {}, {}, {}
    for rank, text in sorted(text_by_rank.items()):
        seen = {}
        for ln in text.splitlines():
            if not ln.startswith("{"):
                continue
            r = json.loads(ln)
            if r.get("mode") == "pf_gate":
                pf.setdefault(rank, []).append(r)
            elif r.get("mode") == "point":
                pts[(r["tp"], r["B"], r["L"], rank)] = r
            elif r.get("mode") == "tp":
                key = (r["tp"], r["B"], r["L"])
                i = seen[(key, r["arm"])] = seen.get((key, r["arm"]), -1) + 1
                lst = arms.setdefault(key, {}).setdefault(r["arm"], [])
                while len(lst) <= i:
                    lst.append([])
                lst[i].append(r)
    return {"arms": arms, "pf_gate": pf, "points": pts}


def _slowest(reps: list[list[dict]]) -> float:
    """An arm's step time: the median over its repeats of the median over reps of the slowest rank's rep."""
    meds = []
    for ranks in reps:
        n = min(len(x["reps"]) for x in ranks)
        meds.append(statistics.median(max(x["reps"][j][0] for x in ranks) for j in range(n)))
    return statistics.median(meds)


def tp_entry(by_arm: dict, key: tuple[int, int, int], filler_fraction: float) -> dict:
    """One TP point (`key` = (TP, B, L)), per GPU over the slowest rank's step: served MACs a GPU (each rank's shard),
    the filler's MACs a GPU (the mean over ranks), C at the base arm's clock; v4's reading (serving at `CERTIFIED`'s
    0.225, plain) and v5's (the modelled served part, decode's deferred pass on each GPU's MACs, `entry_v5`'s model),
    weighted TP × the step's wall time over serving alone."""
    tp = key[0]
    base, phase = _slowest(by_arm["base"]), _slowest(by_arm["phase"])
    recs = [r for ranks in by_arm["phase"] for r in ranks]
    mhz = statistics.median(r["mhz"] for ranks in by_arm["base"] for r in ranks)
    n = statistics.median(r["served_macs"] for ranks in by_arm["base"] for r in ranks)
    f = statistics.mean(r["fill_macs_rep"] for r in recs)
    dp = defer_pass(n, mhz, tp, key[1], key[2])
    t1, fv5 = phase + dp["added_ms"], max(0.0, f - dp["displaces"])
    c4, c5 = cap(mhz) * phase * 1e-3, cap(mhz) * t1 * 1e-3
    return {"entry": f"TP {tp} B {key[1]} L {key[2] // 1024}K", "tp": tp, "B": key[1], "L": key[2], "t0_ms": base,
            "t1_measured_ms": phase, "t1_ms": t1, "slowdown": phase / base, "mhz": mhz,
            "serving_macs": n, "filler_measured_macs": f, "filler_macs": fv5,
            "filler_part": filler_fraction * fv5 / c5, "served_part_modelled": SERVED_V5["fraction"] * n / c5,
            "coverage": (filler_fraction * fv5 + SERVED_V5["fraction"] * n) / c5,
            "coverage_v4_form": (CERTIFIED["serving"]["fraction"] * n + filler_fraction * f) / c4,
            "time": tp * t1 / base, "time_v4": tp * phase / base,
            "served_model": {"defer_pass_ms": dp["pass_ms"], "defer_added_ms": dp["added_ms"],
                                                    "defer_s": dp["s"], "defer_d": dp["d"],
                                                    "displaced_macs": min(f, dp["displaces"])}}


def tp_gates(sets: list[dict], entries: dict, pf: dict | None) -> dict:
    """SCORE-TP's gates, per rank (HC-COMM's round-2 proposal): the slowdown of the slowest rank's step ≤ LIMIT at every
    entry with ≥ TP_REPS reps a record and base and phase interleaved (≥ 2 repeats each); every rank's pre-timing
    pf_gate records ok, of every kind in TP_PF_KINDS (whole tiles against dfill_ref, cut warps against kf_ref_warp);
    every rank's receipts after every phase repeat (checked, none bad, no shape error, no control equal); and every
    set recording `TP_NCCL`. ``filler_paths`` is v4's on the TP build's program facts (`program_facts` at `TP_FILLER`).
    DRAM and clocks are the run's (`tp_report`)."""
    out = {}
    worst = max(((e["slowdown"], k) for k, e in entries.items()), default=(float("nan"), None))
    short = [f"{k}: {a}" for s in sets for k, by in s["rows"]["arms"].items() for a in ("base", "phase")
             if len(by.get(a, [])) < 2 or any(len(x["reps"]) < TP_REPS for ranks in by.get(a, []) for x in ranks)]
    out["slowdown"] = {"ok": bool(entries) and worst[0] <= LIMIT and not short, "worst": worst[0], "at": worst[1],
                       "limit": LIMIT, "short": short[:4]}
    bad_pf, ranks_seen = [], 0
    for s in sets:
        for rank, recs in s["rows"]["pf_gate"].items():
            ranks_seen += 1
            kinds = {r.get("kind") for r in recs if r.get("ok")}
            if not all(r.get("ok") for r in recs) or not set(TP_PF_KINDS) <= kinds:
                bad_pf.append(f"{s['dir']} rank {rank}: kinds ok {sorted(k for k in kinds if k)}")
    out["pf_gate"] = {"ok": ranks_seen > 0 and not bad_pf, "ranks": ranks_seen, "why": "; ".join(bad_pf[:3]) or None}
    bad_rec, checked = [], 0
    for s in sets:
        for k, by in s["rows"]["arms"].items():
            for i, ranks in enumerate(by.get("phase", [])):
                for x in ranks:
                    checked += x.get("rec_checked", 0)
                    if (not x.get("rec_checked") or x.get("rec_bad") or x.get("shape_bad") or x.get("whole_bad")
                            or x.get("warps_bad") or x.get("whole_control_equal") or x.get("warps_control_equal")):
                        bad_rec.append(f"{k} repeat {i} rank {x['rank']}")
    out["receipts"] = {"ok": checked > 0 and not bad_rec, "checked": checked, "bad": bad_rec[:4]}
    nccl = [s["dir"] for s in sets if TP_NCCL not in s["settings"].split()]
    out["nccl"] = {"ok": not nccl, "missing": nccl}
    out["filler_paths"] = paths_gate(pf)
    return out


def tp_score(entries: list[dict]) -> dict:
    """SCORE-TP over the TP entries present, per GPU: the mean of their coverage weighted by TP × wall time (`tp_entry`'s
    ``time``), in its filler and served parts, beside v4's form (served plain at 0.225, weighted by v4's own wall time,
    ``time_v4``); per TP as well. Missing `TP_SUITE` entries are listed, and SCORE-TP needs them all."""
    def mean(es, k):
        w = [e["time_v4" if k == "coverage_v4_form" else "time"] for e in es]
        return sum(e[k] * x for e, x in zip(es, w)) / sum(w) if es else None

    have = {(e["tp"], e["B"], e["L"]) for e in entries}
    missing = [f"TP {t} B {b} L {l // 1024}K" for t, b, l in TP_SUITE if (t, b, l) not in have]
    out = {"missing": missing, "by_tp": {}}
    for t in sorted({e["tp"] for e in entries}):
        es = [e for e in entries if e["tp"] == t]
        out["by_tp"][t] = {k: mean(es, k) for k in ("coverage", "filler_part", "served_part_modelled", "coverage_v4_form")}
    if entries and not missing:
        out |= {"SCORE_TP": mean(entries, "coverage"), "filler_part": mean(entries, "filler_part"),
                "served_part_modelled": mean(entries, "served_part_modelled"), "v4_form": mean(entries, "coverage_v4_form")}
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


def dram_gate(filler_csv: str | None, control_csv: str | None, kernel: str = "dfill_k") -> dict:
    """The filler's DRAM bytes a second (ncu, its own launches: kernels named `kernel`) against DRAM_LIMIT of the HBM
    bandwidth the positive control measures with the same counter in the same run; the counter is valid only if the
    control reads its 1 GiB."""
    if filler_csv is None:
        return {"ok": True, "why": "no filler"}
    cb = ncu_metric(control_csv or "", "dram__bytes.sum", "bw_read")
    ct = ncu_metric(control_csv or "", "gpu__time_duration.sum", "bw_read")
    fb = ncu_metric(filler_csv, "dram__bytes.sum", kernel)
    ft = ncu_metric(filler_csv, "gpu__time_duration.sum", kernel)
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


def row_rounds(r: dict) -> list[tuple[str, int, dict | None]]:
    """Every filler round a row records, as (where, the slot it holds by position, the record or None): the calibration,
    each arm block and each standalone block, and any `*_round` an arm-side driver adds beside them (a second filler's)."""
    out = [("calibration", 0, r["alone_fill"].get("round"))]
    out += [(f"calibration {k}", None, v) for k, v in r["alone_fill"].items() if k.endswith("_round")]
    for i, b in enumerate(r["blocks"]):
        out.append((f"arm block {i}", 1 + i, b["arm"].get("round")))
        if "solo" in b:
            out.append((f"standalone block {i}", 1 + r.get("rounds", len(r["blocks"])) + i, b["solo"].get("round")))
        out += [(f"{side} block {i} {k}", None, v) for side in ("arm", "solo") for k, v in (b.get(side) or {}).items()
                if k.endswith("_round")]
    return out


def replay_gate(rows: list[dict]) -> dict:
    """Anti-replay over every filler round of the run (`row_rounds`): the round's tile range is the scorer's,
    [R·2^32, (R + 1)·2^32) with R = point·slots + slot (slots: rounds.cuh's POINT_SLOTS, one count for every row of the
    run; rows from before it, rounds + 1, the slot by position), so no id is the prover's choice; its tiles start at the range's start and stay inside
    it; no two rounds' tiles overlap; no two rounds share coins; the first tile's seed is dseed(the round's coins,
    tag | its id) as the device printed it (tag: a high bit the filler's kernel marks its ids with, else 0); and no sampled
    seed (each round's first, last and checked tiles) repeats across rounds."""
    bad, spans = [], []
    fill_rows = [r for r in rows if r.get("place") != "none"]
    pts = [r.get("point") for r in fill_rows]
    if len(set(pts)) != len(pts) or None in pts:
        bad.append(f"point indices missing or repeated: {pts}")
    counts = {r.get("slots", r.get("rounds", len(r["blocks"])) + 1) for r in fill_rows}
    if len(counts) > 1:
        bad.append(f"rows number their rounds with different slot counts {sorted(counts)}, so two points' ids can meet")
    for r in fill_rows:
        slots = r.get("slots", r.get("rounds", len(r["blocks"])) + 1)
        for name, pos, rd in row_rounds(r):
            where = f"{r['kind']} {r['B']}/{r['L']} {name}"
            if rd is None or r.get("point") is None:
                bad.append(f"{where}: no round record")
                continue
            slot = rd.get("slot", pos)
            if slot is None or not 0 <= slot < slots:
                bad.append(f"{where}: slot {slot} not in [0, {slots})")
                continue
            want = r["point"] * slots + slot
            lo, hi = want << RANGE_LOG2, (want + 1) << RANGE_LOG2
            t0, t1 = rd["tiles"]
            tag = int(rd.get("tag", "0"), 16)
            if rd["id"] != want or rd["range"] != [lo, hi]:
                bad.append(f"{where}: range {rd['range']} is not the scorer's [{lo}, {hi})")
            if not lo == t0 < t1 <= hi:
                bad.append(f"{where}: tiles [{t0}, {t1}) not at the start of [{lo}, {hi})")
            if any(not t0 <= t < t1 for t in rd["checked"]):
                bad.append(f"{where}: a checked tile outside the round's tiles")
            c0, c1 = (int(x, 16) for x in rd["coins"])
            if f"{dseed(c0, c1, tag | t0):016x}" != rd["seed0"]:
                bad.append(f"{where}: seed {rd['seed0']} is not dseed(the round's coins, {tag:#x} | {t0})")
            spans.append((tag, t0, t1, (c0, c1), {dseed(c0, c1, tag | t) for t in (t0, t1 - 1, *rd["checked"])}, where))
    spans.sort()
    overlaps = [f"{a[5]} and {b[5]}" for a, b in zip(spans, spans[1:]) if a[0] == b[0] and b[1] < a[2]]
    coins = [s[3] for s in spans]
    seeds = [x for s in spans for x in s[4]]
    coin_rep, seed_rep = len(coins) - len(set(coins)), len(seeds) - len(set(seeds))
    why = bad + [f"tiles overlap: {o}" for o in overlaps] + ([f"{coin_rep} rounds reuse coins"] if coin_rep else []) + \
        ([f"{seed_rep} sampled seeds repeat"] if seed_rep else []) + ([] if spans else ["no filler rounds"])
    return {"ok": not why, "rounds": len(spans), "tiles": sum(s[2] - s[1] for s in spans), "seeds_sampled": len(seeds),
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


def cheat_s(steps: int) -> int:
    """The positive control's steps a tile beside serving: S less S/CHEAT, a multiple of 8 (dfill_k's unroll)."""
    return steps - max(8, steps // CHEAT // 8 * 8)


def _too_good(v: dict) -> bool:
    return bool(v.get("beside_over_alone")) and v["beside_over_alone"] > v["too_good_limit"]


def honesty(arm, stats: dict, phases_points, *, control: dict | None = None, has_filler: bool = False,
            whole: dict | None = None) -> dict:
    """The arm's placement is a pure function of the point (asked twice, the same answer); no filler ran faster beside
    serving than alone on the same SMs (its split's chunks beside ÷ the median of the round-interleaved standalone
    blocks' above 1 + max(TOO_GOOD_FLOOR, their trimmed spread, `solo_spread`); such a result goes to the assessor), and
    every filler point has that reference;
    none ran above its SMs' int8 tensor peak at the clock its blocks read; and, with a filler, the positive control (a
    filler credited for S steps a tile that runs S - S/CHEAT beside serving and S alone, `control`) is caught as too good,
    so the check can see a dishonest arm at the run's own spread. filler-scorer/v4 passes `whole`, each point's
    `whole_of_peak` (the kernel's serving and filler MACs together over its SMs' peak), and refuses any above 1."""
    import arms as A  # noqa: PLC0415

    unstable = [f"{ph} {b}/{l}" for ph, b, l in phases_points if A.placed(arm, ph, b, l) != A.placed(arm, ph, b, l)]
    fast = [f"{k[0]} {k[1]}/{k[2]}: {v['beside_over_alone']:.3f} > {v['too_good_limit']:.3f}" for k, v in stats.items()
            if _too_good(v)]
    peak = [f"{k[0]} {k[1]}/{k[2]}: {v['of_peak']:.3f}" for k, v in stats.items() if (v.get("of_peak") or 0) > 1.0]
    noref = [f"{k[0]} {k[1]}/{k[2]}" for k, v in stats.items() if v["place"] != "none" and len(v["solo_rates"]) < 2]
    ctl = None
    if has_filler:
        ctl = {"caught": False, "why": "no control row"} if control is None else {
            "point": control.get("at"), "cheat_S": control.get("cheat_S"), "S": control.get("S"),
            "beside_over_alone": control["beside_over_alone"], "limit": control["too_good_limit"],
            "solo_spread": control["solo_spread"], "of_peak": control["of_peak"], "caught": _too_good(control)}
    out = {"ok": not unstable and not fast and not peak and not noref and (ctl is None or ctl["caught"]),
           "unstable_placement": unstable, "too_good": fast, "over_peak": peak, "no_reference": noref, "control": ctl}
    if whole is not None:
        out["over_peak_whole"] = [f"{k}: {v:.3f}" for k, v in whole.items() if v is not None and v > 1.0]
        out["whole_of_peak_max"] = max((v for v in whole.values() if v is not None), default=None)
        out["ok"] = out["ok"] and not out["over_peak_whole"]
    return out


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
    """Reported, not scored: τ under VF per entry, the filler's per-block rate spread (work per round), board power (the
    settled windows': serving alone, the arm, the filler alone; rows before them, the per-rep medians, which lag) and clock
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
        "power": {f"{k[0]} {k[1]}/{k[2]}": v["power_settled"] or {"alone": v["alone_watt"], "arm": v["arm_watt"]}
                  for k, v in stats.items()},
        "power_per_rep": {f"{k[0]} {k[1]}/{k[2]}": [v["alone_watt"], v["arm_watt"]] for k, v in stats.items()},
        "clock": {f"{k[0]} {k[1]}/{k[2]}": [v["alone_mhz"], v["arm_mhz"]] for k, v in stats.items()},
        "placement_setup_ms": {f"{k[0]} {k[1]}/{k[2]}": v["setup_ms"] for k, v in timed.items()},
        "robust": rob,
        "aa_slowdown": {f"{k[0]} {k[1]}/{k[2]}": v["slowdown"] for k, v in stats.items() if v["place"] == "none"},
    }


def margins(stats: dict) -> dict:
    """Each measured point's serving-slowdown margin (filler-scorer/v3): LIMIT less the point's slowdown, the number the
    slowdown gate reads, and the margins of its worst and best rounds, so a point whose rounds already reach the limit
    shows it has none (``no_margin``); ``min`` is the smallest over the scored points (robustness points are listed,
    unscored)."""
    scored = set(points(robust=False))
    pts = {}
    for k, v in stats.items():
        r = [x for x in v.get("slowdown_rounds") or [] if isfinite(x)]
        pts[f"{k[0]} {k[1]}/{k[2]}"] = {"slowdown": v["slowdown"], "margin": LIMIT - v["slowdown"],
                                        "rounds": [LIMIT - max(r), LIMIT - min(r)] if r else None, "place": v["place"],
                                        "fill_sms": v["fill_sms"], "scored": k in scored}
    lo = min(((m["margin"], name) for name, m in pts.items() if m["scored"] and isfinite(m["margin"])), default=None)
    return {"limit": LIMIT, "points": pts, "min": {"at": lo[1], "margin": lo[0]} if lo else None,
            "no_margin": [name for name, m in pts.items() if m["scored"] and m["rounds"] and m["rounds"][0] <= 0]}


def isfinite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
