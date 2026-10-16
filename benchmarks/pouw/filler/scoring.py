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
i.e. Σ certified ÷ Σ capacity × time. The c's are the PROVED certified fractions. Under filler-scorer/v3 (the lock) the
filler is read at the entry of `FILLER_ENTRIES` that `filler_entry` decides from its declared noise and its build's kernel
facts (`kernel_facts`): Theorem K's 0.9, Theorem E's 0.225, or a proved floor of 0 (a chained filler's, or any filler no
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decode_capacity"))
import share  # noqa: E402  CAP's constants: MACs a token, non-GEMM times, FA2, the int8 peak

V2, V3, V4 = "filler-scorer/v2", "filler-scorer/v3", "filler-scorer/v4"
#: the active lock. V3 adds the arm's record checked before its lease (`arms.record_problems`), each point's slowdown
#: margin (`margins`), and the filler read at its proved entry (`FILLER_ENTRIES`, `filler_entry`) with SCORE under CH
#: beside it. `score.py report OUT --as v2` reads a run as v2 did. V4 (prepared, not active: `report OUT --as v4`
#: previews it) reads the filler wherever it's called from, filler warps inside serving kernels included
#: (`program_facts`, `filler_entry_v4`, `credit_fixed`, `paths_gate`, `whole_of_peak`)
SCORER = V3
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


def program_facts(listings: dict[str, str]) -> dict:
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
      a register; and a caller with a branch whose target isn't named. A guarded instruction counts as run: a warp-role
      split by predication instead of a branch is refused.
    - ``serving_mma``: every other function's MMAs (a serving kernel's consumer warps, kernels with no filler), outside
      the filler warps' paths, credited at the serving reading."""
    fns = {(b, f["name"]): f for b, text in listings.items() for f in parse_nvdisasm(text)}
    calls = {k: {(k[0], m["name"]) for _, _, op, args in f["insns"] if op.startswith("CALL") and (m := _REF.search(args))}
             for k, f in fns.items()}
    filler = {k for k in fns if any(x in k[1] for x in FILLER_KERNELS)}
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
    violations, callers, regions, todo = [], {}, {}, []
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
    return {"version": V4, "binaries": sorted(listings), "filler": {
        "functions": sorted(f"{b}: {n}" for b, n in filler), "helpers": sorted(f"{b}: {n}" for b, n in own),
        "path_helpers": sorted(f"{b}: {n}" for b, n in helpers - own),
        "mma": dict(sorted(mma.items())), "fp_ops": dict(sorted(fp.items())), "accumulating_mma": acc, "kind": kind,
        "fp_free": not fp, "L": L}, "callers": callers, "serving_mma": serving, "violations": violations}


#: what filler-scorer/v4 does not check, beside `UNCHECKED`, when a theorem's entry applies
UNCHECKED_V4 = (
    "that every receipt is written by the filler: v4 sees what code can store on the filler warps' paths, not what "
    "address another path's store hits, so a serving warp, or a kernel whose filler warps are inlined instead of "
    "calling dfill_k (as FILL r3's are), could write one; the receipts gate holds every sampled receipt to the host "
    "reference whoever wrote it",
    "that a filler warp only calls dfill_k: a path from the warp-role branch that never reaches the CALL looks like the "
    "serving warps' path, so its MMAs read as serving",
    "the filler function's inputs (only the coins, the public tile id and the step, besides its stop flag): the "
    "INTERFACE's, with the receipts gate holding its outputs to the host reference's function of (coins, tile, steps)",
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
    pass the spot checks: its row's binary is one the scorer built and read (``driver``, else dscore), the filler
    warps' paths there are confined (no violation), the block's fixed round (a ``*_round`` besides the split filler's
    ``round``, whose range, tag and coins the replay gate holds to the scorer's) has spot-checked tiles, and its row has
    no bad receipt. Any other block's ``fixed_macs`` becomes 0. Returns (rows, {credited, dropped})."""
    read = set(pf["binaries"]) if pf else set()
    out, credited, dropped = [], 0, []
    for r in rows:
        r = json.loads(json.dumps(r))
        for i, b in enumerate(r["blocks"]):
            for side in ("arm", "solo"):
                x = b.get(side) or {}
                if not x.get("fixed_ms"):
                    continue
                fr = [v for k, v in x.items() if k.endswith("_round") and k != "round" and isinstance(v, dict)]
                why = ("the build recorded no program facts" if pf is None
                       else f"its driver {r.get('driver', 'dscore')} wasn't built and read" if r.get("driver", "dscore")
                       not in read else "the filler warps' paths aren't confined" if pf["violations"]
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
