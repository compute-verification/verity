"""Reads a DC run's probe rows (opcode_probe's rows.jsonl and gate.json) and states TENSOR's M3 / M4 quantities.

  python3 analyze.py sm90|sm100 DIR [--json OUT]
  python3 analyze.py agree DIR_A DIR_B      # a repeat: every rate both measured within 2%; exit 1 otherwise

DIR holds every rows.jsonl and gate.json of one family's run (run.sh writes probes/ and scalar/ under $OUT). A rate in
rows.jsonl is warp instructions a SM a clock; this turns it into slots (32-bit destination cells) a SM a
clock, the issue rate of every timed loop, and for each joint row the issue cycles a tensor
instruction takes from the lane beside it (a = (lane alone - lane beside) / tensor beside, F1's coefficient):

- sm90 (M3): ρ_T over every wgmma and mma.sync form; pass when none exceeds wgmma s8 m64n256k32 by more than 2%.
- sm100 (M4): ρ_T5 (tcgen05.mma i8 128x256x32), ρ_ld (best tcgen05.ld, alone and beside the saturated MMA), ρ_sync
  (best mma.sync); pass when ρ_ld ≥ ρ_T5 and ρ_sync ≤ ρ_T5 at 2%.
- both: the duo fold's words a SM a clock alone (LOP3 + IMAD + LEA.HI a word), against ρ_T.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MARGIN = 0.02
K_OF = {"s8": 32, "u8s8": 32, "e4m3": 32, "e5m2": 32, "f16": 16, "bf16": 16, "tf32": 8, "b1": 256}
#: mma.sync forms: (pattern prefix, slots a warp instruction). Every m16n8 form writes 16 x 8 = 128 32-bit slots.
SYNC_SLOTS = 128
#: tcgen05.ld 32-bit registers a thread for each probe
LD_REGS = {"ld_32x32b_x32": 32, "ld_32x32b_x16": 16, "ld_16x256b_x4": 16, "ld_16x64b_x16": 16, "ld_16x128b_x8": 16,
           "ld_32x32b_x4": 4, "ld_32x32b_x1": 1, "ld16w_32x32b_x32": 32, "ld16w_32x32b_x16": 16,
           "ld_32x32b_x64": 64, "ld_32x32b_x128": 128}
TENSOR = ("IGMMA", "QGMMA", "HGMMA", "BGMMA", "UTC", "LDTM", "IMMA", "HMMA", "BMMA")
LANE_ALONE = {"mix": "s_mix", "imad": "s_imad", "lop3": "s_lop3", "shfl": "s_shfl", "duo": "s_duo",
              "redux": "s_redux", "reduce": "s_reduce", "li": "p_imad_lop3", "imadw": "s_imadw", "lde": "s_lde",
              "atome": "s_atome", "redg": "s_redg", "atoms": "s_atoms", "lds128": "s_lds128"}
#: the first letters of a lane loop's counted opcodes (LOP3, IMAD, FFMA, SHFL, HFMA2, LD.E, LDS, ATOM, ATOMS, REDG)
LANE_OPS = ("I", "F", "L", "S", "H", "A", "R")


def load(d: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Every row by probe name under d, and the gate verdict probe.py run wrote beside each rows.jsonl."""
    rows, verdicts = {}, {}
    for f in sorted(d.rglob("rows.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[r["probe"]] = r
        g = f.parent / "gate.json"
        if g.exists():
            for v in json.loads(g.read_text())["verdicts"]:
                verdicts[v["probe"]] = v
    return rows, verdicts


def loop_rate(row: dict, loop: int, prefixes=TENSOR) -> tuple[str, float] | None:
    """The first pattern of `loop` whose opcode starts with one of `prefixes`, with its rate."""
    for key, rate in row["rates"].items():
        i, pat = key.split(":", 1)
        if int(i) == loop and pat.startswith(prefixes):
            return pat, rate
    return None


def issue_rates(row: dict, verdict: dict | None) -> list[float] | None:
    """Warp instructions issued a SM a clock in each timed loop (SASS order): its most counted pattern's rate times the
    loop's length over that count. Rates are keyed by `expect`'s loop order, which ptxas need not keep, so the pattern
    finds its rate."""
    if not verdict:
        return None
    out = []
    for lp in verdict["verdict"].get("loops", []):
        length = (int(lp["tail"], 16) - int(lp["head"], 16)) // 16 + 1
        pat, n = max(lp["counts"].items(), key=lambda pc: pc[1], default=(None, 0))
        rate = next((r for k, r in row["rates"].items() if k.split(":", 1)[1] == pat), None) if n else None
        out.append(rate * length / n if rate else float("nan"))
    return out


def slots_sm90(name: str, row: dict) -> dict | None:
    m = re.fullmatch(r"wgz?_([a-z0-9]+)_n(\d+)(_rs)?_a([01])", name)
    if m:
        form, n = m.group(1), int(m.group(2))
        pat, r = loop_rate(row, 0)
        # four warps issue one wgmma (one SASS instruction each); it writes 64 x N slots
        return {"form": f"wgmma {form} m64n{n}k{K_OF[form]}" + (" RS" if m.group(3) else ""), "opcode": pat,
                "slots": r / 4 * 64 * n, "macs": r / 4 * 64 * n * K_OF[form], "zero_data": name.startswith("wgz")}
    if name.startswith(("imma_", "sync_")):
        pat, r = loop_rate(row, 0)
        return {"form": f"mma.sync {name.split('_', 1)[1]}", "opcode": pat, "slots": r * SYNC_SLOTS}
    return None


def slots_sm100(name: str, row: dict) -> dict | None:
    m = re.fullmatch(r"tcz?_(i8|f8|f16)_m(\d+)_n(\d+)", name)
    if m:
        kind, mm, n = m.group(1), int(m.group(2)), int(m.group(3))
        pat, r = loop_rate(row, 0)
        k = 16 if kind == "f16" else 32
        return {"form": f"tcgen05.mma {kind} {mm}x{n}x{k}", "opcode": pat, "slots": r * mm * n,
                "macs": r * mm * n * k, "zero_data": name.startswith("tcz")}
    if name in LD_REGS:
        pat, r = loop_rate(row, 0)
        warps, shape = (16, name[6:]) if name.startswith("ld16w_") else (8, name[3:])
        return {"form": f"tcgen05.ld {shape}, {warps} warps", "opcode": pat, "ld_words": r * 32 * LD_REGS[name]}
    if name.startswith("imma_"):
        pat, r = loop_rate(row, 0)
        return {"form": f"mma.sync {name[5:]}", "opcode": pat, "slots": r * SYNC_SLOTS}
    return None


def lane_of(name: str) -> str | None:
    m = re.fullmatch(r"(?:wgspec|wgmix|tclane)_([a-z0-9]+)_r\d+|(?:tclaneU|wgU)_(mix|imad|lop3|duo|redux|reduce|li|lif|"
                     r"imadw|lde|atome|redg|atoms|lds128)(?:_r\d+)?", name)
    return (m.group(1) or m.group(2)) if m else None


def joint(name: str, row: dict, rows: dict, verdict: dict | None, t_alone: float | None) -> dict:
    """A tensor instruction beside a lane: both rates, both fractions of alone, and a."""
    t = loop_rate(row, 0)
    lane_loop = 0 if name.startswith("wgmix") else 1
    lane = loop_rate(row, lane_loop, LANE_OPS) if t else None
    if lane_loop == 0 and t:
        lane = next(((p, r) for k, r in row["rates"].items() for i, p in [k.split(":", 1)]
                     if int(i) == 0 and not p.startswith(TENSOR)), None)
    out = {"tensor_opcode": t and t[0], "tensor_rate": t and t[1], "lane_opcode": lane and lane[0],
           "lane_rate": lane and lane[1], "issue": issue_rates(row, verdict)}
    alone = rows.get(LANE_ALONE.get(lane_of(name) or "", ""))
    if t and lane and alone:
        pat0 = lane[0]
        la = alone["rates"].get(f"0:{pat0}")
        if la:
            out["lane_fraction"] = lane[1] / la
            out["a"] = (la - lane[1]) / t[1]
    if t and t_alone:
        out["tensor_fraction"] = t[1] / t_alone
    if "U_" in name:
        out["exact"] = True
    return out


def rate_of(row: dict, prefix: str) -> float | None:
    """The rate of the first pattern (any loop) whose opcode starts with prefix."""
    return next((r for k, r in row["rates"].items() if k.split(":", 1)[1].startswith(prefix)), None)


def upper(points: list[tuple[float, float]], x0: float) -> float:
    """The most y at x0 over the convex hull of points with free disposal (doing less of either is always possible)."""
    pts = {(0.0, 0.0)} | {q for x, y in points for q in ((x, y), (x, 0.0), (0.0, y))}
    best = max((y for x, y in pts if x >= x0), default=0.0)
    for x1, y1 in pts:
        for x2, y2 in pts:
            if x1 < x0 < x2:
                best = max(best, y1 + (y2 - y1) * (x0 - x1) / (x2 - x1))
    return best


def balanced(points: list[tuple[float, float]]) -> float:
    """The largest t with upper(points, t) ≥ t: honest at L = depth, which loads every slot it writes."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if upper(points, mid) >= mid else (lo, mid)
    return lo


def tmem_model(joints: dict, ld_alone: float, rho5: float, rho_sync: float | None) -> dict:
    """sm_100's two stages from the exact joint points (TENSOR §3, sm_100), all as fractions of ρ_T5.

    TMEM region: (tensor fraction, words loaded / ρ_T5) over every MMA-beside-load point, ld alone and the MMA alone.
    Every point is a program's rates over one period, so each is achievable; a block-timed point lies inside the region,
    an exact (OP_UNTIL) one can lie on its edge. Honest at L = 32 loads every slot it writes, so it runs at the region's
    balanced point h. At L = 32 m it loads one slot in m, so its tensor is saturated from the smallest m with
    upper(0.98) ≥ 0.98 / m. Pipe region: (tensor fraction, mma.sync slots / ρ_sync alone) over every sync point, which
    holds the shared-pipe line t + s = 1. The adversary keeps h's tcgen05 words and adds the mma.sync slots the pipe
    region leaves at h (mma.sync writes registers, so no TMEM load; conservative: no contention between mma.sync and
    the loads is counted)."""
    tm = [(1.0, 0.0), (0.0, ld_alone / rho5)] + [(j["tensor_fraction"], j["ld_words"] / rho5)
                                                for j in joints.values() if "ld_words" in j and "tensor_fraction" in j]
    h = balanced(tm)
    at_full = upper(tm, 1 - MARGIN)
    m = next((m for m in range(1, 65) if at_full >= (1 - MARGIN) / m), None)
    out = {"tmem_points": tm, "honest_fraction_L32": h, "honest_words_L32": h * rho5,
           "ld_beside_saturated": at_full * rho5, "L_saturated": m and 32 * m}
    if rho_sync:
        sp = [(1.0, 0.0), (0.0, 1.0)] + [(j["tensor_fraction"], j["sync_slots"] / rho_sync)
                                         for j in joints.values() if "sync_slots" in j and "tensor_fraction" in j]
        extra = upper(sp, h) * rho_sync
        out |= {"sync_points": sp, "adversary_words_L32": h * rho5 + extra,
                "eps_tensor_L32": extra / (h * rho5 + extra)}
    return out


def duo_words(rows: dict) -> float | None:
    r = rows.get("s_duo")
    return r["rates"].get("0:LOP3.LUT", 0) * 32 if r else None


def analyze(family: str, d: Path) -> dict:
    rows, verdicts = load(d)
    forms, joints = {}, {}
    slots_of = slots_sm90 if family == "sm90" else slots_sm100
    for name, row in rows.items():
        s = slots_of(name, row)
        if s:
            forms[name] = {**s, "sm_mhz": row.get("sm_mhz"), "tags": row.get("tags", []),
                           "issue": issue_rates(row, verdicts.get(name))}
    scalar = {n: {"rates": r["rates"], "issue": issue_rates(r, verdicts.get(n))}
              for n, r in rows.items() if n.startswith(("s_", "p_"))}
    out: dict = {"family": family, "dir": str(d), "forms": forms, "scalar": scalar, "duo_words": duo_words(rows)}
    if family == "sm90":
        honest = forms.get("wg_s8_n256_a1") or forms.get("wg_s8_n256_a0")
        t_alone = rows.get("wg_s8_n256_a1", {}).get("rates", {}).get("0:IGMMA.64x256x32.S8.S8*")
        for name, row in rows.items():
            if lane_of(name):
                joints[name] = joint(name, row, rows, verdicts.get(name), t_alone)
            elif name.startswith("wgli_") and t_alone:
                # the honest LI-CP-W loop (static-trip): its wgmma rate over the wgmma alone is the profile's tensor
                # fraction, and dc90-licpw's measured cycles a MMA = the plain GEMM's cycles over it
                joints[name] = {"tensor_fraction": rate_of(row, "IGMMA") / t_alone,
                                "issue": issue_rates(row, verdicts.get(name))}
            elif name.startswith("wgbulk_") and t_alone:
                # a TMA bulk op beside the saturated wgmma: the wgmma's rate over its alone rate (does the bulk steal the
                # wgmma's issue), and the bulk's own words a SM-clock (UBLKCP a copy, UBLKRED a reduce; 1024 words an op)
                op = "UBLKCP" if name.endswith("_cp") else "UBLKRED"
                br = rate_of(row, op)
                joints[name] = {"tensor_fraction": rate_of(row, "IGMMA") / t_alone,
                                "bulk_words_per_clk": br * 1024 if br else None,
                                "issue": issue_rates(row, verdicts.get(name))}
            elif name in ("bulkcp", "bulkred"):
                op = "UBLKCP" if name == "bulkcp" else "UBLKRED"
                br = rate_of(row, op)
                joints[name] = {"bulk_words_per_clk": br * 1024 if br else None}
            elif name.startswith("wgU_sync") and t_alone:
                joints[name] = {"tensor_fraction": rate_of(row, "IGMMA") / t_alone, "exact": True,
                                "issue": issue_rates(row, verdicts.get(name)),
                                "sync_slots": (rate_of(row, "IMMA") or 0.0) * SYNC_SLOTS}
        best = max((f for f in forms.values() if "slots" in f and not f.get("zero_data")), key=lambda f: f["slots"],
                   default=None)
        out["M3"] = {"rho_T": best and best["slots"], "rho_T_form": best and best["form"],
                     "honest_slots": honest and honest["slots"],
                     "pass": bool(best and honest and best["slots"] <= honest["slots"] * (1 + MARGIN))}
    else:
        t5 = forms.get("tc_i8_m128_n256")
        t_alone = rows.get("tc_i8_m128_n256", {}).get("rates", {}).get("0:UTCIMMA*")
        for name, row in rows.items():
            if lane_of(name) or name.startswith(("ldmma_", "syncmma_")):
                joints[name] = joint(name, row, rows, verdicts.get(name), t_alone)
            elif (li := re.fullmatch(r"tcliU_[xp](\d+)f?(?:_t\d+)?", name)) and t_alone and rate_of(row, "LDTM"):
                # words loaded and folded (one LI op each) a SM a clock; L = 64 commits one slot in two of UTCIMMA's
                cols = int(li.group(1))
                words, asked = rate_of(row, "LDTM") * 32 * cols, rate_of(row, "UTC") * 128 * 256 / 2
                joints[name] = {"tensor_fraction": rate_of(row, "UTC") / t_alone, "exact": True,
                                "issue": issue_rates(row, verdicts.get(name)), "li_words": words,
                                "l64_asked": asked, "keeps_up_l64": words / asked}
            elif name.startswith(("ldU_", "mmaU_", "syncU_", "tcliU_", "tcbulk_")) and t_alone:
                # OP_UNTIL points: both rates exact (neither includes time at the barrier)
                j = {"tensor_fraction": rate_of(row, "UTC") / t_alone, "exact": True,
                     "issue": issue_rates(row, verdicts.get(name))}
                if rate_of(row, "LDTM"):
                    j["ld_words"] = rate_of(row, "LDTM") * 32 * 32
                if rate_of(row, "IMMA"):
                    j["sync_slots"] = rate_of(row, "IMMA") * SYNC_SLOTS
                for op in ("UBLKCP", "UBLKRED"):   # a TMA bulk op moves 1024 words; its rate beside the saturated MMA
                    if rate_of(row, op):
                        j["bulk_words_per_clk"] = rate_of(row, op) * 1024
                joints[name] = j
            elif name in ("bulkcp", "bulkred"):
                op = "UBLKCP" if name == "bulkcp" else "UBLKRED"
                br = rate_of(row, op)
                joints[name] = {"bulk_words_per_clk": br * 1024 if br else None}
            elif name.startswith("tcfU_") and "tc_mxf4_m128_n256" in rows:
                # kind::mxf4 beside lanes: its fraction of mxf4 alone
                joints[name] = {"tensor_fraction": rate_of(row, "UTC") / rate_of(rows["tc_mxf4_m128_n256"], "UTC"),
                                "exact": True, "issue": issue_rates(row, verdicts.get(name))}
        rho5 = t5["slots"] if t5 else 256.0
        for name in [n for n in joints if n.startswith("ldmma_")]:
            ld = loop_rate(rows[name], 1, ("LDTM",))
            m = re.fullmatch(r"ldmma_w(\d+)(?:_r(\d+))?", name)
            if ld and m:
                j = joints[name]
                j["ld_words"] = ld[1] * 32 * 32
                # the MMA warp's iteration is 4 x 128x256 slots; the loaders' is R loads of 32 x 32 words a warp
                j["ld_asked"] = int(m.group(1)) * int(m.group(2) or 8) * 1024 * rho5 / (4 * 128 * 256)
                j["demand_limited"] = j["ld_words"] >= j["ld_asked"] * (1 - MARGIN)
                j["honest_fraction"] = min(j.get("tensor_fraction", 0), j["ld_words"] / rho5)
        for name in [n for n in joints if n.startswith("syncmma_")]:
            sy = loop_rate(rows[name], 1, ("IMMA",))
            if sy:
                joints[name]["sync_slots"] = sy[1] * SYNC_SLOTS
        rho_t5 = t5 and t5["slots"]
        ld_alone = max((f["ld_words"] for f in forms.values() if "ld_words" in f), default=None)
        # beside: the most words loaded while the tensor stays saturated; a demand-limited point is a lower bound
        ld_beside = max((j["ld_words"] for j in joints.values()
                         if "ld_words" in j and j.get("tensor_fraction", 0) >= 1 - MARGIN), default=None)
        honest = max((j["honest_fraction"] for j in joints.values() if "honest_fraction" in j), default=None)
        rho_sync = max((f["slots"] for n, f in forms.items() if n.startswith("imma_")), default=None)
        model = {}
        if rho_t5 and ld_alone:
            model = tmem_model(joints, ld_alone, rho_t5, rho_sync)
            ld_beside, honest = model["ld_beside_saturated"], model["honest_fraction_L32"]
        rho_ld = min(x for x in (ld_alone, ld_beside) if x is not None) if (ld_alone or ld_beside) else None
        out["M4"] = {"rho_T5": rho_t5, "rho_ld_alone": ld_alone, "rho_ld_beside_mma": ld_beside, "rho_ld": rho_ld,
                     "rho_sync": rho_sync, "honest_tensor_fraction": honest,
                     "pass": bool(rho_t5 and rho_ld and rho_sync and rho_ld >= rho_t5 * (1 - MARGIN)
                                  and rho_sync <= rho_t5 * (1 + MARGIN)),
                     **{k: v for k, v in model.items() if not k.endswith("_points")}}
        out["tmem_model"] = model
    out["joints"] = joints
    rho = (out.get("M3") or {}).get("rho_T") or (out.get("M4") or {}).get("rho_T5")
    if rho and out["duo_words"]:
        out["fold_headroom"] = out["duo_words"] / rho
    return out


def agree(a: Path, b: Path, margin: float = MARGIN) -> dict:
    """Every rate both runs measured (probe, loop:pattern), as b / a; agree when each is within margin of 1."""
    ra, rb = load(a)[0], load(b)[0]
    ratios = {f"{p}/{k}": rb[p]["rates"][k] / v for p in sorted(ra.keys() & rb.keys())
              for k, v in ra[p]["rates"].items() if v and rb[p]["rates"].get(k)}
    worst = max(ratios.items(), key=lambda kv: abs(kv[1] - 1), default=(None, 1.0))
    off = {k: r for k, r in ratios.items() if abs(r - 1) > margin}
    return {"a": str(a), "b": str(b), "rates": len(ratios), "worst": worst[0], "worst_ratio": worst[1],
            "outside": off, "agree": not off and bool(ratios)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["agree"]:
        out = agree(Path(argv[1]), Path(argv[2]))
        print(json.dumps(out, indent=1))
        return 0 if out["agree"] else 1
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("family", choices=("sm90", "sm100"))
    p.add_argument("dir", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    out = analyze(a.family, a.dir)
    text = json.dumps(out, indent=1, default=float)
    if a.json:
        a.json.write_text(text + "\n")
    verdict = out.get("M3") or out.get("M4")
    print(json.dumps({"family": a.family, **verdict, "duo_words": out["duo_words"],
                      "fold_headroom": out.get("fold_headroom")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
