"""Reads a DC run's probe rows (opcode_probe's rows.jsonl and gate.json) and states TENSOR's M3 / M4 quantities.

  python3 analyze.py sm90|sm100 DIR [--json OUT]

DIR holds every rows.jsonl and gate.json of one family's run (run.sh writes probes/ and scalar/ under $OUT). A rate in
rows.jsonl is warp instructions a SM a clock; this turns it into slots (32-bit destination cells, note:20261017T0321Z-
draft-eps1-tensor §1) a SM a clock, the issue rate of every timed loop, and for each joint row the issue cycles a tensor
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
           "ld_32x32b_x4": 4, "ld_32x32b_x1": 1}
TENSOR = ("IGMMA", "QGMMA", "HGMMA", "BGMMA", "UTC", "LDTM", "IMMA", "HMMA", "BMMA")
LANE_ALONE = {"mix": "s_mix", "imad": "s_imad", "lop3": "s_lop3", "shfl": "s_shfl", "duo": "s_duo"}


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
    """Warp instructions issued a SM a clock in each timed loop: a pattern's rate times loop length over its count."""
    if not verdict:
        return None
    out = []
    for i, lp in enumerate(verdict["verdict"].get("loops", [])):
        length = (int(lp["tail"], 16) - int(lp["head"], 16)) // 16 + 1
        pat, n = next(((p, c) for p, c in lp["counts"].items() if c), (None, 0))
        rate = row["rates"].get(f"{i}:{pat}")
        out.append(rate * length / n if pat and rate else float("nan"))
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
        return {"form": f"tcgen05.ld {name[3:]}", "opcode": pat, "ld_words": r * 32 * LD_REGS[name]}
    if name.startswith("imma_"):
        pat, r = loop_rate(row, 0)
        return {"form": f"mma.sync {name[5:]}", "opcode": pat, "slots": r * SYNC_SLOTS}
    return None


def lane_of(name: str) -> str | None:
    m = re.fullmatch(r"(?:wgspec|wgmix|tclane)_([a-z0-9]+)_r\d+", name)
    return m.group(1) if m else None


def joint(name: str, row: dict, rows: dict, verdict: dict | None, t_alone: float | None) -> dict:
    """A tensor instruction beside a lane: both rates, both fractions of alone, and a."""
    t = loop_rate(row, 0)
    lane_loop = 0 if name.startswith("wgmix") else 1
    lane = loop_rate(row, lane_loop, ("I", "F", "L", "S", "H")) if t else None
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
        for name in [n for n in joints if n.startswith("ldmma_")]:
            ld = loop_rate(rows[name], 1, ("LDTM",))
            if ld:
                joints[name]["ld_words"] = ld[1] * 32 * 32
        for name in [n for n in joints if n.startswith("syncmma_")]:
            sy = loop_rate(rows[name], 1, ("IMMA",))
            if sy:
                joints[name]["sync_slots"] = sy[1] * SYNC_SLOTS
        rho_t5 = t5 and t5["slots"]
        ld_alone = max((f["ld_words"] for f in forms.values() if "ld_words" in f), default=None)
        ld_beside = max((j["ld_words"] for j in joints.values() if "ld_words" in j), default=None)
        rho_sync = max((f["slots"] for n, f in forms.items() if n.startswith("imma_")), default=None)
        rho_ld = min(x for x in (ld_alone, ld_beside) if x is not None) if (ld_alone or ld_beside) else None
        out["M4"] = {"rho_T5": rho_t5, "rho_ld_alone": ld_alone, "rho_ld_beside_mma": ld_beside, "rho_ld": rho_ld,
                     "rho_sync": rho_sync,
                     "pass": bool(rho_t5 and rho_ld and rho_sync and rho_ld >= rho_t5 * (1 - MARGIN)
                                  and rho_sync <= rho_t5 * (1 + MARGIN))}
    out["joints"] = joints
    rho = (out.get("M3") or {}).get("rho_T") or (out.get("M4") or {}).get("rho_T5")
    if rho and out["duo_words"]:
        out["fold_headroom"] = out["duo_words"] / rho
    return out


def main(argv: list[str] | None = None) -> int:
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
