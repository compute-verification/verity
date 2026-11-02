"""The DC runs as census lines in machine_model's schema, a device directory a family, and the tensor theorem's tables.

  python3 census.py sm90|sm100 RUN_OUT... [--write DIR]

RUN_OUT is a DC run's `out/` (`research fetch RUN --all` puts it at ~/.research/runs/RUN/out; the run id is the name of
a directory on its path). analyze.py reads each run; a kind's line takes its fastest run (as machine_model's adopt.py
merges two lines) and names every run that measured it, and the joint rows are the upper hull of every run's points.
--write puts `census.csv` and `rows.csv` in DIR (machine_model's `data/devices/sm_90/` or `sm_100/`), keeping every
line and row this doesn't write: the uniform datapath's (UNI, and the row UP) that uniform.py wrote, the rows of
other sources (the data sheet's DS_*, TMAB, AT90), and the lines of classes no run prices (the memory and control
classes added by hand), NOROW's joint issue lines rewritten at the new hull. A row it writes whose old description
extends the new one keeps the old,
the added note with it; one whose note it can't keep is named on stderr. Else they print.

The rows (warp instructions/SM/clk, derated by the 2% noise; facets in SMSP-cycles at 1 - 2%, eps1.py's convention):
- issue: every kind holds 1 SMSP-cycle.
- M: the tensor pipe, every tensor form at its rate, mma.sync included where every point of mma.sync beside the
  honest form lies inside the shared line (else mma.sync gets its own row MS, which lets a model run both at once).
- A (LOP3, SHF, LEA.HI and IADD3), I (IMAD), SHFL: each pipe at its alone rate. FFMA has no pipe row: alone it is
  issue-bound (its loop issues ~3.93 of 4 a SM-clock), so its rate is a lower bound of the pipe's.
- sm100 TL: the TMEM read port, every tcgen05.ld shape at its alone rate; TM100_k: the facets of UTCIMMA 128x256 beside
  tcgen05.ld 32x32b.x32.
- JI90_k / JI100_k: the issue facets beside the honest tensor form (its coefficient c_M, every other kind's c_X), over
  every lane-beside-tensor point, the tensor alone and the best scalar issue; other tensor forms carry c_X, a lower
  bound on their issue cost.
- JP<row><family>_k: a lane pipe's facets beside the honest form, only where the lane loses rate beside it.

The tables: T1 a form (slots a warp instruction, depth, rate, slots/SM/clk,
fraction of ρ_T), the family's ε_tensor at L = 32 and the L where honest holds ρ_T, the joint issue rows, q* (the
binding issue facet's c_M over the words the honest instruction gives a lane, plus on sm_100 the tcgen05.ld issue that
moves them to registers: QSTAR's threshold, 2.03 on sm_120) and L_bal (32 ρ_T over the duo fold's words/SM/clk, alone
and beside the saturated tensor).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyze = _module("dc_analyze", HERE / "analyze.py")
eps1 = _module("dc_eps1", HERE.parent / "opcode_probe" / "eps1.py")
rowsmod = _module("dc_rows", HERE.parent / "opcode_probe" / "rows.py")

MARGIN = analyze.MARGIN
SMSPS = 4
DEVICE = {"sm90": "sm_90", "sm100": "sm_100"}
HONEST = {"sm90": "IGMMA_S8_N256", "sm100": "UTCIMMA_M128_N256"}
LD_HONEST = "LDTM_32X32B_X32"
#: mma.sync probes: (kind, depth). sync_e4m3_k32 is emulated on sm_90 (2 HMMA.16816 + 4 FADD): no kind of its own.
SYNC = {"imma_s8_k32": ("IMMA16832_S8", 32), "imma_s8_k16": ("IMMA16816_S8", 16), "sync_b1_k256": ("BMMA168256_B1", 256),
        "sync_f16_k16": ("HMMA16816_F16", 16), "sync_bf16_k16": ("HMMA16816_BF16", 16), "sync_tf32_k8": ("HMMA1688_TF32", 8)}
#: scalar kinds: (row, the alone probes whose fastest pattern gives its rate). The memory and multiply classes have no
#: pipe row: their lines are the issue row's and the joint rows' (each timed beside the honest form, sm90_wgmma.cu's
#: memory and imadw lanes). LD.E and IMAD.WIDE, which no class names, price UNLISTED, the measured readings' catch-all:
#: a class of IMAD.WIDE's own would carry the same lines.
SCALAR = {"ALU": ("A", {"s_lop3": "LOP3.LUT", "s_shf": "SHF.L.W.U32.HI", "s_lea": "LEA.HI"}),
          "IMAD": ("I", {"s_imad": "IMAD"}), "SHFL": ("SHFL", {"s_shfl": "SHFL.BFLY"}), "FFMA": (None, {"s_ffma": "FFMA"}),
          "UNLISTED": (None, {"s_lde": "LD.E*", "s_imadw": "IMAD.WIDE.U32"}),
          "ATOME32": (None, {"s_atome": "ATOM.E*"}), "REDG32": (None, {"s_redg": "REDG*"}),
          "ATOMS32": (None, {"s_atoms": "ATOMS*"}), "LDS128": (None, {"s_lds128": "LDS.128"}),
          "REDUX": (None, {"s_redux": "REDUX*"}),
          "STS32": (None, {"s_sts32": "STS"}), "STS128": (None, {"s_sts128": "STS.128"}),
          "STG32": (None, {"s_stg32": "STG.E"}), "STG128": (None, {"s_stg128": "STG.E.128"}),
          "ISETP": (None, {"s_isetp": "ISETP.GT.U32.XOR"})}
#: lanes beside the tensor whose pipe row a joint can bind: lane -> (scalar kind, the lane's counted pattern)
LANE_PIPE = {"imad": ("IMAD", "IMAD"), "lop3": ("ALU", "LOP3.LUT"), "shfl": ("SHFL", "SHFL.BFLY")}


def run_id(d: Path) -> str:
    for p in (d, *d.parents):
        if rowsmod.RUN_ID.fullmatch(p.name):
            return p.name
    raise SystemExit(f"{d}: no run id (rYYYYMMDD-HHMMSS-xxxx) on its path; fetch it with `research fetch RUN --all`")


def tensor_kind(family: str, name: str, form: dict) -> tuple[str, float, int] | None:
    """(kind, slots a warp instruction, depth) of a tensor probe's form; None for a zero-data twin or an emulated form."""
    if form.get("zero_data"):
        return None
    if name in SYNC:
        return SYNC[name][0], analyze.SYNC_SLOTS, SYNC[name][1]
    m = re.fullmatch(r"wg_([a-z0-9]+)_n(\d+)(_rs)?_a[01]", name)
    if m:
        ty, n = m.group(1), int(m.group(2))
        op = form["opcode"].split(".")[0]
        # four warps issue one wgmma, a warp instruction each, which writes 64 N / 4 slots
        return f"{op}_{ty.upper()}_N{n}{'_RS' if m.group(3) else ''}", 16 * n, analyze.K_OF[ty]
    m = re.fullmatch(r"tc_(i8|f8|f16)_m(\d+)_n(\d+)", name)
    if m:
        ty, mm, n = m.groups()
        op = form["opcode"].split(".")[0].rstrip("*")
        tag = {"i8": "", "f8": "_E4M3", "f16": "_F16"}[ty]
        return f"{op}{tag}_M{mm}_N{n}", int(mm) * int(n), 16 if ty == "f16" else 32
    return None


def ld_kind(name: str) -> tuple[str, float] | None:
    """(kind, words a warp instruction) of a tcgen05.ld probe."""
    if name not in analyze.LD_REGS:
        return None
    shape = name.split("_", 1)[1]
    return f"LDTM_{shape.upper()}", 32 * analyze.LD_REGS[name]


def smi_clocks(d: Path) -> dict:
    """The SM clock (MHz) over the run's nvidia-smi samples: the most (f_max at default clocks) and the median under load."""
    f = d / "env" / "smi.csv"
    if not f.exists():
        return {}
    mhz = []
    with f.open(newline="") as fh:
        for r in csv.DictReader(fh, skipinitialspace=True):
            v = r.get("clocks.current.sm [MHz]", "").split()
            if v and v[0].isdigit():
                mhz.append(int(v[0]))
    busy = [x for x in mhz if x > 500]
    return {"f_max": max(mhz, default=None), "f_median": statistics.median(busy) if busy else None,
            "f_min": min(busy, default=None)}


class Device:
    """Every run of one family, merged: a kind's fastest rate with its runs, and every joint point."""

    def __init__(self, family: str, outs: list[Path]):
        self.family = family
        self.runs = [run_id(d) for d in outs]
        self.analyses = [analyze.analyze(family, d) for d in outs]
        self.clocks = {r: smi_clocks(d) for r, d in zip(self.runs, outs)}
        self.kinds: dict[str, dict] = {}
        for run, a in zip(self.runs, self.analyses):
            for name, f in a["forms"].items():
                tk = tensor_kind(family, name, f)
                lk = ld_kind(name) if family == "sm100" else None
                if tk:
                    kind, per, depth = tk
                    self._add(kind, f["slots"] / per, run, {"tensor": True, "slots_per": per, "depth": depth,
                                                            "opcode": f["opcode"], "mhz": f.get("sm_mhz")})
                elif lk:
                    kind, per = lk
                    self._add(kind, f["ld_words"] / per, run, {"ld": True, "words_per": per, "opcode": f["opcode"],
                                                               "mhz": f.get("sm_mhz")})
            for kind, (_, probes) in SCALAR.items():
                rates = [a["scalar"][p]["rates"].get(f"0:{pat}", 0) for p, pat in probes.items() if p in a["scalar"]]
                if rates and max(rates) > 0:
                    self._add(kind, max(rates), run, {"scalar": True})
        self.joints = [(run, n, j) for run, a in zip(self.runs, self.analyses) for n, j in a["joints"].items()]

    def _add(self, kind: str, rate: float, run: str, meta: dict) -> None:
        k = self.kinds.setdefault(kind, {"rate": 0.0, "runs": set(), **meta})
        k["runs"].add(run)
        if rate > k["rate"]:
            k.update(meta, rate=rate)

    def runs_of(self, kind: str) -> str:
        return " ".join(sorted(self.kinds[kind]["runs"]))

    def tensor(self) -> dict[str, dict]:
        return {k: v for k, v in self.kinds.items() if v.get("tensor")}

    def rho_T(self) -> tuple[float, str]:
        """The most slots/SM/clk over every tensor form, and its form (the honest one when it ties within 0.01%)."""
        rho, form = max((v["rate"] * v["slots_per"], k) for k, v in self.tensor().items())
        h = self.kinds[HONEST[self.family]]
        return (rho, HONEST[self.family]) if h["rate"] * h["slots_per"] >= rho * (1 - 1e-4) else (rho, form)

    def honest_rate(self) -> float:
        return self.kinds[HONEST[self.family]]["rate"]

    def issue_max(self) -> float:
        """The most warp instructions a SM-clock any scalar probe issued (its loops' sum)."""
        return max(sum(s["issue"]) for a in self.analyses for s in a["scalar"].values()
                   if s["issue"] and not any(math.isnan(x) for x in s["issue"]))

    def issue_points(self) -> list[tuple[float, float, str]]:
        """(honest tensor, every other instruction) warp instructions a SMSP-cycle: the best scalar issue on the X axis,
        the honest form alone, and every lane-beside-tensor point (its loops' issue minus the tensor's)."""
        t_alone = self.honest_rate()
        pts = [(0.0, self.issue_max() / SMSPS, "scalar issue")]
        for run, a in zip(self.runs, self.analyses):
            for name, f in a["forms"].items():
                tk = tensor_kind(self.family, name, f)
                if tk and tk[0] == HONEST[self.family] and f["issue"]:
                    r = f["slots"] / tk[1]
                    pts.append((r / SMSPS, (sum(f["issue"]) - r) / SMSPS, f"{name}@{run}"))
        for run, name, j in self.joints:
            t = j.get("tensor_rate") or (j.get("tensor_fraction") or 0) * t_alone
            if not t or not j.get("issue") or any(math.isnan(x) for x in j["issue"]):
                continue
            pts.append((t / SMSPS, (sum(j["issue"]) - t) / SMSPS, f"{name}@{run}"))
        return pts

    def lane_points(self, lane: str) -> list[tuple[float, float, str]]:
        """(honest tensor, the lane's pattern) a SMSP-cycle over every beside point (a block-timed one is a time average a
        program achieves, so a point of the region too), its alone rate and the tensor's; empty when some point keeps
        both within the margin of alone (the pipe row holds as is)."""
        kind, pat = LANE_PIPE[lane]
        alone, t_alone = self.kinds[kind]["rate"], self.honest_rate()
        beside = [(j["tensor_rate"], j["lane_rate"], f"{name}@{run}") for run, name, j in self.joints
                  if analyze.lane_of(name) == lane and j.get("lane_opcode") == pat and j.get("tensor_rate")]
        if not beside or any(min(t / t_alone, x / alone) >= 1 - MARGIN for t, x, _ in beside):
            return []
        return [(0.0, alone / SMSPS, f"{kind} alone"), (t_alone / SMSPS, 0.0, "tensor alone")] + [
            (t / SMSPS, x / SMSPS, n) for t, x, n in beside]

    def ld_points(self) -> list[tuple[float, float, str]]:
        """(UTCIMMA 128x256, LDTM.x32) a SMSP-cycle over every exact and block-timed MMA-beside-load point."""
        t_alone, per = self.honest_rate(), 32 * 32
        pts = [(0.0, self.kinds[LD_HONEST]["rate"] / SMSPS, "ld alone"), (t_alone / SMSPS, 0.0, "tensor alone")]
        for run, name, j in self.joints:
            if "ld_words" in j and "tensor_fraction" in j:
                pts.append((j["tensor_fraction"] * t_alone / SMSPS, j["ld_words"] / per / SMSPS, f"{name}@{run}"))
        return pts

    def sync_points(self) -> list[tuple[float, float, str]]:
        """(tensor fraction, mma.sync slots over the best mma.sync's alone) for every mma.sync-beside-tensor point."""
        rho_sync = max(v["rate"] * v["slots_per"] for k, v in self.tensor().items() if k.startswith(("IMMA", "HMMA", "BMMA")))
        return merged([(j["tensor_fraction"], j["sync_slots"] / rho_sync, f"{name}@{run}")
                       for run, name, j in self.joints if "sync_slots" in j and "tensor_fraction" in j])

    def model(self) -> dict | None:
        """sm_100's TMEM model (analyze.tmem_model) over every run's joint points."""
        if self.family != "sm100":
            return None
        rho5 = self.kinds[HONEST["sm100"]]["rate"] * self.kinds[HONEST["sm100"]]["slots_per"]
        ld_alone = max(v["rate"] * v["words_per"] for v in self.kinds.values() if v.get("ld"))
        rho_sync = max(v["rate"] * v["slots_per"] for k, v in self.tensor().items() if k.startswith("IMMA"))
        joints = {f"{n}@{r}": j for r, n, j in self.joints}
        return analyze.tmem_model(joints, ld_alone, rho5, rho_sync)


def merged(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """One point a probe: its runs' mean (a hull edge between two runs of one probe would be their noise)."""
    by: dict[str, list[tuple[float, float, str]]] = {}
    for p in points:
        by.setdefault(p[2].split("@")[0], []).append(p)
    return [(statistics.fmean(p[0] for p in ps), statistics.fmean(p[1] for p in ps),
             name + ("@" + ",".join(p[2].split("@")[1] for p in ps) if "@" in ps[0][2] else ""))
            for name, ps in by.items()]


def non_axis(points: list[tuple[float, float, str]]) -> list[dict]:
    return [f for f in eps1.facets(merged(points), MARGIN) if f["MMA"] > 0 and f["X"] > 0]


def cycles_line(kind: str, row: str, cycles: float, runs: str) -> dict[str, str]:
    line = {"kind": kind, "row": row, "rate": "", "noise": "", "cycles": f"{cycles:.9g}", "u": "", "since": "",
            "until": "", "runs": runs}
    rowsmod.u_of(rowsmod.CYCLE_U, line["rate"], line["noise"], line["cycles"], line["u"])
    return line


def rate_line(kind: str, row: str, rate: float, runs: str) -> dict[str, str]:
    first = runs.split()[0]
    return {**rowsmod.census_line(kind, row, rate, first), "runs": runs}


def tight(f: dict) -> str:
    return " and ".join(t.split("@")[0] for t in f["tight"])


def census(dev: Device, carried: frozenset[str] = frozenset()) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(census.csv lines, rows.csv lines) of one device; `carried`, kinds no run prices that the old census gave the
    joint issue rows' scalar coefficient (NOROW), get it on the new rows."""
    fam, sfx = dev.family, dev.family[2:]
    honest = HONEST[fam]
    lines: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    every = " ".join(dev.runs)

    def row(name: str, status: str, runs: str, desc: str) -> None:
        rows.append({"row": name, "status": status, "runs": runs, "description": desc})

    kinds = sorted(dev.kinds)
    for k in kinds:
        lines.append(cycles_line(k, "issue", 1.0, dev.runs_of(k)))
    row("issue", "measured", every,
        f"1 warp instruction a SMSP-cycle in any mix; best scalar issue {dev.issue_max():.3f} a SM-clock. FFMA alone "
        "issues at this ceiling, so it has no pipe row")

    tensor = dev.tensor()
    sync_kinds = [k for k in tensor if k.startswith(("IMMA", "HMMA", "BMMA"))]
    sp = dev.sync_points()
    shared = bool(sp) and all(t + s <= 1 + MARGIN for t, s, _ in sp)
    for k in sorted(tensor):
        on_m = shared or k not in sync_kinds
        lines.append(rate_line(k, "M" if on_m else "MS", tensor[k]["rate"], dev.runs_of(k)))
    rho, rho_form = dev.rho_T()
    sp_desc = ", ".join(f"{n.split('@')[0]} ({t:.3f}, {s:.3f})" for t, s, n in sp)
    row("M", "measured", every,
        f"tensor pipe: rho_T {rho:.2f} slots/SM/clk ({rho_form}); "
        + (f"mma.sync shares it: every point beside the honest form lies inside t + s <= 1 ({sp_desc})" if shared
           else "wgmma and tcgen05 forms only"))
    if not shared and sync_kinds:
        row("MS", "measured", every,
            "mma.sync forms: " + (f"a point beside the honest form exceeds the shared line ({sp_desc}), so its own row"
                                  if sp else "not measured beside the honest form, so its own row (a model may run "
                                  "both at once: conservative)"))

    for k, (r, _) in SCALAR.items():
        if r and k in dev.kinds:
            lines.append(rate_line(k, r, dev.kinds[k]["rate"], dev.runs_of(k)))
    row("A", "measured", every, f"ALU pipe: LOP3, SHF, LEA.HI (and IADD3) {dev.kinds['ALU']['rate']:.4f} /SM/clk, "
        "one pipe (each 1/2 in pairs)")
    row("I", "measured", every, f"IMAD {dev.kinds['IMAD']['rate']:.4f} /SM/clk")
    row("SHFL", "measured", every, f"SHFL.BFLY {dev.kinds['SHFL']['rate']:.4f} /SM/clk")

    if fam == "sm100":
        for k in sorted(k for k, v in dev.kinds.items() if v.get("ld")):
            lines.append(rate_line(k, "TL", dev.kinds[k]["rate"], dev.runs_of(k)))
        best = max((v["rate"] * v["words_per"], k) for k, v in dev.kinds.items() if v.get("ld"))
        row("TL", "measured", every, f"TMEM read port: every tcgen05.ld shape alone; best {best[0]:.2f} words/SM/clk "
            f"({best[1]})")
        lp = dev.ld_points()
        for i, f in enumerate(non_axis(lp)):
            r = f"TM100_{i}"
            lines.append(cycles_line(honest, r, f["MMA"], every))
            lines.append(cycles_line(LD_HONEST, r, f["X"], every))
            row(r, "joint", every, f"{honest} {f['MMA']:.3f} + {LD_HONEST} {f['X']:.3f} SMSP-cycles (DC's MMA-beside-"
                f"load points, 2% noise margin; tight at {tight(f)})")

    ip = dev.issue_points()
    for i, f in enumerate(non_axis(ip)):
        r = f"JI{sfx}_{i}"
        for k in kinds:
            lines.append(cycles_line(k, r, f["MMA"] if k == honest else f["X"], every))
        for k in sorted(carried - set(kinds)):
            lines.append(cycles_line(k, r, f["X"], every))
        row(r, "joint", every, f"{honest} {f['MMA']:.3f} + every other warp instruction {f['X']:.3f} SMSP-cycles (DC's "
            f"lanes beside the tensor, 2% noise margin; tight at {tight(f)}); other tensor forms carry the scalar "
            "coefficient, a lower bound on their issue cost")

    for lane, (kind, pat) in LANE_PIPE.items():
        pts = dev.lane_points(lane)
        if len(pts) <= 2:
            continue
        pipe = SCALAR[kind][0]
        for i, f in enumerate(non_axis(pts)):
            r = f"JP{pipe}{sfx}_{i}"
            lines.append(cycles_line(honest, r, f["MMA"], every))
            lines.append(cycles_line(kind, r, f["X"], every))
            row(r, "joint", every, f"{honest} {f['MMA']:.3f} + {kind} {f['X']:.3f} SMSP-cycles (DC's {lane} lanes "
                f"beside the tensor, exact and block-timed, 2% noise margin; tight at {tight(f)})")
    return lines, rows


def rows_csv(rows: list[dict[str, str]]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=("row", "status", "runs", "description"), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def tables(dev: Device) -> str:
    fam = dev.family
    honest = HONEST[fam]
    rho, rho_form = dev.rho_T()
    out = [f"## {DEVICE[fam]}: runs {' '.join(dev.runs)}", ""]
    for r, c in dev.clocks.items():
        if c:
            out.append(f"- {r}: SM clock f_max {c['f_max']} MHz at default clocks, median {c['f_median']} and least "
                       f"{c['f_min']} MHz under load")
    out += ["", "| kind | opcode | slots a warp instr | depth | rate (warp instr/SM/clk) | SMSP-cycles held | "
            "slots/SM/clk | slots a SMSP-cycle | MACs/SM/clk | of rho_T | MHz |", "|" + "---|" * 11]
    for k, v in sorted(dev.tensor().items(), key=lambda kv: -kv[1]["rate"] * kv[1]["slots_per"]):
        s = v["rate"] * v["slots_per"]
        out.append(f"| {k} | {v['opcode']} | {v['slots_per']:.0f} | {v['depth']} | {v['rate']:.5f} | "
                   f"{SMSPS / v['rate']:.2f} | {s:.2f} | {s / SMSPS:.2f} | {s * v['depth']:.0f} | {s / rho:.4f} | "
                   f"{v['mhz']} |")
    lds = {k: v for k, v in dev.kinds.items() if v.get("ld")}
    if lds:
        out += ["", "| tcgen05.ld kind | opcode | words a warp instr | rate | words/SM/clk | of rho_T5 |", "|" + "---|" * 6]
        for k, v in sorted(lds.items(), key=lambda kv: -kv[1]["rate"] * kv[1]["words_per"]):
            w = v["rate"] * v["words_per"]
            out.append(f"| {k} | {v['opcode']} | {v['words_per']:.0f} | {v['rate']:.5f} | {w:.2f} | {w / rho:.4f} |")
    out += ["", "| scalar kind | row | rate (warp instr/SM/clk) |", "|---|---|---|"]
    for k, (r, _) in SCALAR.items():
        if k in dev.kinds:
            out.append(f"| {k} | {r or '(issue only)'} | {dev.kinds[k]['rate']:.4f} |")
    out.append(f"| (best scalar issue) | issue | {dev.issue_max():.4f} |")

    h = dev.kinds[honest]
    rho_h, d_h = h["rate"] * h["slots_per"], h["depth"]
    out += ["", f"T1: rho_T = {rho:.2f} slots/SM/clk ({rho_form}); honest {honest} rho_H = {rho_h:.2f}, depth {d_h}."]
    model = dev.model()
    if fam == "sm90":
        out.append(f"- L = 32: eps_tensor <= 1 - (rho_H/rho_T)(d_H/L) = {max(0.0, 1 - rho_h / rho * d_h / 32):.4f}; "
                   "honest holds rho_T at L = 32 (N >= 64)")
    else:
        sat = model["L_saturated"]
        out += [f"- the TMEM load beside saturated tcgen05.mma: {model['ld_beside_saturated']:.2f} words/SM/clk "
                f"(alone {max(v['rate'] * v['words_per'] for v in lds.values()):.2f}); "
                f"{'not ' if model['ld_beside_saturated'] >= rho * (1 - MARGIN) else ''}the per-word bottleneck at L = 32",
                f"- honest at L = 32 (loads every slot): {model['honest_fraction_L32']:.4f} of rho_T5 = "
                f"{model['honest_words_L32']:.2f} words/SM/clk; the tensor stays saturated from L = {sat} (T1 there: "
                f"{1 - d_h / sat:.3f})",
                f"- eps_tensor at L = 32: two-stage LP (mma.sync in honest's slack, shared-pipe line) "
                f"{model['eps_tensor_L32']:.4f}; T1 with the load alone 1 - h = {1 - model['honest_fraction_L32']:.4f}"]

    ip = dev.issue_points()
    fs = non_axis(ip)
    out += ["", "Joint issue rows (honest tensor + every other warp instruction, SMSP-cycles, at 1 - 2%):"]
    for i, f in enumerate(fs):
        out.append(f"- JI{fam[2:]}_{i}: {honest} {f['MMA']:.3f} + other {f['X']:.3f} (tight at {tight(f)})")
    for lane in LANE_PIPE:
        pts = dev.lane_points(lane)
        for f in non_axis(pts) if len(pts) > 2 else []:
            out.append(f"- {lane} pipe beside the tensor: {honest} {f['MMA']:.3f} + {LANE_PIPE[lane][0]} {f['X']:.3f} "
                       f"(tight at {tight(f)})")
    if fam == "sm100":
        for f in non_axis(dev.ld_points()):
            out.append(f"- TMEM: {honest} {f['MMA']:.3f} + {LD_HONEST} {f['X']:.3f} (tight at {tight(f)})")
    sp = dev.sync_points()
    if sp:
        out.append("- mma.sync beside the honest form (tensor fraction, mma.sync of its alone): "
                   + ", ".join(f"{n.split('@')[0]} ({t:.3f}, {s:.3f})" for t, s, n in sp))

    c_m = fs[0]["MMA"] if fs else float("nan")
    lane_words = h["slots_per"] / 32
    q = c_m / lane_words
    out += ["", f"q*: c_M {c_m:.3f} SMSP-cycles over {lane_words:.0f} words a lane = {q:.4f}"]
    if fam == "sm100":
        c_x = fs[0]["X"] if fs else 1.0
        # 32 lanes x 32 words a tcgen05.ld 32x32b.x32; tcgen05.wait::ld is a scoreboard wait in SASS, no instruction
        n_ld = h["slots_per"] / (32 * 32)
        q_ld = (c_m + n_ld * c_x) / lane_words
        out[-1] += (f"; with the {n_ld:.0f} tcgen05.ld.x32 that bring them to registers: "
                    f"{q_ld:.4f} (x128 loads: {(c_m + n_ld / 4 * c_x) / lane_words:.4f})")
    duo = max((a["duo_words"] or 0) for a in dev.analyses)
    beside = max(((j["lane_rate"] * 32, n) for _, n, j in dev.joints if analyze.lane_of(n) == "duo"
                  and (j.get("tensor_fraction") or 0) >= 1 - MARGIN), default=None)
    if duo:
        out.append(f"L_bal (duo, 3 ops a word): 32 rho_T / duo = {32 * rho / duo:.1f} alone ({duo:.2f} words/SM/clk)"
                   + (f"; beside the saturated tensor the best measured duo is {beside[0]:.2f} ({beside[1]}), so "
                      f"L_bal <= {32 * rho / beside[0]:.1f}" if beside else ""))
    return "\n".join(out) + "\n"


#: the rows census() writes; a stale one (a joint row the new hull drops) goes, every other row is kept
OWN_ROW = re.compile(r"issue|MS?|A|I|SHFL|TL|TM\d+_\d+|JI\d+_\d+|JP[A-Z]+\d+_\d+")


JI_ROW = re.compile(r"JI\d+_\d+")


def old_lines(d: Path, name: str) -> list[dict[str, str]]:
    if not (d / name).is_file():
        return []
    with open(d / name, newline="") as f:
        return list(csv.DictReader(f))


def carried(d: Path, dev: Device) -> frozenset[str]:
    """The kinds of DIR's census.csv that no run prices but that carry the joint issue rows' scalar coefficient."""
    return frozenset(r["kind"] for r in old_lines(d, "census.csv")
                     if JI_ROW.fullmatch(r["row"]) and r["kind"] != "UNI" and r["kind"] not in dev.kinds)


def kept(d: Path, name: str, written: set[str] | None = None) -> list[dict[str, str]]:
    """The lines and rows of DIR/NAME that census() doesn't write, which --write keeps: uniform.py's (UNI's census
    lines, the UP row), every row of another source (the data sheet's DS_*, TMAB) and, in census.csv, every line of a
    kind no run prices (`written` is the kinds census() writes) but its joint issue lines, which census() rewrites."""
    return [r for r in old_lines(d, name) if r.get("kind") == "UNI" or not OWN_ROW.fullmatch(r["row"])
            or (written is not None and r["kind"] not in written and not JI_ROW.fullmatch(r["row"]))]


def with_notes(d: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """`rows` with each one's old description where it extends the new one (a note added after the generated text)."""
    if not (d / "rows.csv").is_file():
        return rows
    with open(d / "rows.csv", newline="") as f:
        old = {r["row"]: r["description"] for r in csv.DictReader(f)}
    out = []
    for r in rows:
        o = old.get(r["row"], "")
        if o.startswith(r["description"]):
            r = {**r, "description": o}
        elif o and not r["description"].startswith(o):
            print(f"census.py: {r['row']}'s description is regenerated; a note after its old text, if any, is not kept",
                  file=sys.stderr)
        out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("family", choices=tuple(DEVICE))
    p.add_argument("outs", type=Path, nargs="+")
    p.add_argument("--write", type=Path, help="the device directory (machine_model's data/devices/sm_90 or sm_100)")
    a = p.parse_args(argv)
    dev = Device(a.family, a.outs)
    lines, rows = census(dev, carried(a.write, dev) if a.write else frozenset())
    print(tables(dev))
    if a.write:
        a.write.mkdir(parents=True, exist_ok=True)
        lines = lines + kept(a.write, "census.csv", set(dev.kinds))
        rows = with_notes(a.write, rows) + kept(a.write, "rows.csv")
        (a.write / "census.csv").write_text(rowsmod.census_csv(lines))
        (a.write / "rows.csv").write_text(rows_csv(rows))
        print(f"wrote {a.write / 'census.csv'} ({len(lines)} lines) and {a.write / 'rows.csv'} ({len(rows)} rows)")
    else:
        print(rowsmod.census_csv(lines) + "\n" + rows_csv(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
