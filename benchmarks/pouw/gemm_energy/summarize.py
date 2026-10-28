"""pJ a MAC of every GEMM window energy.sh recorded, by clock setting, beside the floors' arms.

    python3 summarize.py OUT [--idle-w 78] [--theta 989.4e12] [--floors FLOORS.json] [--json table.json] > table.md

OUT/<setting>/ (energy.sh's: one directory a clock setting, a locked MHz or `cap`) holds fb-<variant>.jsonl (`v31
fill-bench` lines, each with its `power` windows) and torch.jsonl (torch_energy.py's). A window of c calls of M N K MACs
over t s and E J gives:

- pJ a MAC over idle, (E - idle_w t) / (c M N K), and all-in, E / (c M N K);
- theta, c M N K / t / THETA (989.4 T MAC/s: 132 SMs, 4096 int8 MACs an SM-clock at 1830 MHz).

A row is an arm's median over its rounds, with the spread (max - min) of its pJ over idle. FLOORS is
{setting: [[name, pJ over idle], ...]}, the floor's arms in order (the resident loop first, each later one adding a step:
TMA to shared memory, L2 streaming, HBM streaming, the epilogue); each GEMM's excess over the first is split into each
later arm's step and what is left past the last (issue overhead). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

IDLE_W = 78.0
THETA = 989.4e12
#: fill-bench's power arms and torch_energy.py's, as the table names them
LABELS = {"lt": "cuBLASLt, autotuned (fill-bench lt)", "gemm": "served GEMM (fill-bench gemm)",
          "gemm_d": "served GEMM of D (fill-bench gemm_d)", "wgl": "wgmma.linear (fill-bench wgl)",
          "ovl": "served call, overlapped (fill-bench ovl)", "int_mm": "cuBLASLt, torch._int_mm"}


def label(arm: str, variant: str | None) -> str:
    if arm.startswith("wgmma:"):
        return f"s8_linear wgmma.linear {arm[6:]}"
    base = LABELS.get(arm, arm)
    return f"{base} [{variant}]" if variant and arm in ("gemm", "gemm_d", "ovl") else base


def point(w: dict, m: int, n: int, k: int, idle_w: float, theta: float) -> dict:
    """One window's numbers: mj over ms for calls of m n k MACs."""
    macs = w["calls"] * m * n * k
    t = w["ms"] / 1e3
    e = w["mj"] / 1e3
    return {"macs": macs, "s": t, "j": e, "w": e / t, "ms_call": w["ms"] / w["calls"], "theta": macs / t / theta,
            "pj_all": 1e12 * e / macs, "pj_idle": 1e12 * (e - idle_w * t) / macs}


def windows(out: Path) -> list[dict]:
    """Every window under out, as {setting, shape, m, n, k, arm, variant, round, calls, ms, mj, cmhz, nvml_mhz, reasons}."""
    rows = []
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        for f in sorted(d.glob("fb-*.jsonl")):
            variant = f.stem[3:]
            for line in f.read_text().splitlines():
                if not line.startswith("{"):
                    continue
                fb = json.loads(line)
                for w in fb.get("power", []):
                    rows.append({"setting": d.name, "shape": fb["shape"], "m": fb["m"], "n": fb["n"], "k": fb["k"],
                                 "variant": variant, **w})
        t = d / "torch.jsonl"
        for line in t.read_text().splitlines() if t.exists() else []:
            w = json.loads(line)
            if w.get("kind") == "window":
                rows.append({"setting": d.name, "variant": None, "cmhz": None, **w})
    return rows


def side(out: Path) -> dict:
    """Each setting's idle windows and counter periods (torch_energy.py's)."""
    res: dict = {}
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        t = d / "torch.jsonl"
        for line in t.read_text().splitlines() if t.exists() else []:
            w = json.loads(line)
            if w.get("kind") in ("idle", "counter"):
                res.setdefault(d.name, {}).setdefault(w["kind"], []).append(w)
    return res


def table(rows: list[dict], idle_w: float = IDLE_W, theta: float = THETA) -> list[dict]:
    """One row a (setting, shape, arm, variant): the median of its windows' numbers and its pJ-over-idle spread."""
    groups: dict = {}
    for r in rows:
        key = (r["setting"], r["shape"], r["arm"], r["variant"] if r["arm"] in ("gemm", "gemm_d", "ovl") else None)
        groups.setdefault(key, []).append(r)
    res = []
    for (setting, shp, arm, variant), ws in groups.items():
        pts = [point(w, w["m"], w["n"], w["k"], idle_w, theta) for w in ws]
        med = {f: statistics.median(p[f] for p in pts) for f in ("w", "ms_call", "theta", "pj_all", "pj_idle")}
        cm = [w["cmhz"] for w in ws if w.get("cmhz")]
        nv = [w["nvml_mhz"] for w in ws if w.get("nvml_mhz")]
        reasons = 0
        for w in ws:
            reasons |= int(w.get("reasons") or 0)
        res.append({"setting": setting, "shape": shp, "m": ws[0]["m"], "n": ws[0]["n"], "k": ws[0]["k"], "arm": arm,
                    "variant": variant, "label": label(arm, variant), "rounds": len(ws), **med,
                    "pj_spread": max(p["pj_idle"] for p in pts) - min(p["pj_idle"] for p in pts),
                    "window_s": statistics.median(p["s"] for p in pts),
                    "cmhz": statistics.median(cm) if cm else None, "nvml_mhz": statistics.median(nv) if nv else None,
                    "reasons": reasons})
    return sorted(res, key=lambda r: (r["setting"] != "cap", r["setting"], r["shape"], r["pj_idle"]))


def split(row: dict, floors: list[list]) -> dict:
    """row's excess over the first floor, by each later floor's step and the rest past the last (issue overhead)."""
    base = floors[0][1]
    steps = {name: pj - prev for (name, pj), (_, prev) in zip(floors[1:], floors[:-1])}
    return {"over": floors[0][0], "excess": row["pj_idle"] - base, "excess_pct": 100 * (row["pj_idle"] / base - 1),
            **{f"step {k}": v for k, v in steps.items()}, "issue overhead": row["pj_idle"] - floors[-1][1]}


def fmt(x, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def markdown(res: list[dict], extra: dict, floors: dict, idle_w: float, theta: float) -> str:
    out = [f"pJ a MAC over idle = (E - {idle_w:g} W t) / MACs; theta against {theta / 1e12:g} T MAC/s. Clock: clock64 "
           "over %globaltimer on CTA 0 of each wg GEMM launch; NVML's SM clock (halfway through each window) beside it.",
           ""]
    for setting in dict.fromkeys(r["setting"] for r in res):
        ex = extra.get(setting, {})
        idle = ", ".join(f"{w['w']:.1f} W" for w in ex.get("idle", []))
        per = ", ".join(f"{c['period_ms']:.0f} ms" for c in ex.get("counter", []) if c.get("period_ms"))
        out += [f"**{'locked at ' + setting + ' MHz' if setting != 'cap' else 'at the 700 W cap (clocks reset)'}**"
                + (f"; measured idle {idle}" if idle else "") + (f"; NVML energy counter period {per}" if per else ""),
                "", "| shape | arm | rounds | ms a call | theta | clock64 MHz | NVML MHz | board W | pJ/MAC all-in "
                "| pJ/MAC over idle | spread | reasons |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in res if r["setting"] == setting):
            out.append(f"| {r['shape']} | {r['label']} | {r['rounds']} | {r['ms_call']:.3f} | {r['theta']:.3f} | "
                       f"{fmt(r['cmhz'], '.0f')} | {fmt(r['nvml_mhz'], '.0f')} | {r['w']:.1f} | {r['pj_all']:.4f} | "
                       f"{r['pj_idle']:.4f} | {r['pj_spread']:.4f} | {r['reasons']:#x} |")
        fl = floors.get(setting)
        if fl:
            names = [f"step {n}" for n, _ in fl[1:]] + ["issue overhead"]
            out += ["", f"Excess over {fl[0][0]} ({fl[0][1]:.3f} pJ), split by the floor's arms "
                    + ", ".join(f"{n} {p:.3f}" for n, p in fl), "",
                    "| shape | arm | pJ/MAC over idle | excess | excess % | " + " | ".join(names) + " |",
                    "|---|---|---|---|---|" + "---|" * len(names)]
            for r in (r for r in res if r["setting"] == setting):
                s = split(r, fl)
                out.append(f"| {r['shape']} | {r['label']} | {r['pj_idle']:.4f} | {s['excess']:.4f} | "
                           f"{s['excess_pct']:.1f} | " + " | ".join(f"{s[n]:.4f}" for n in names) + " |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--idle-w", type=float, default=IDLE_W)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--floors", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = table(windows(a.out), a.idle_w, a.theta)
    if not res:
        print(f"no windows under {a.out}", file=sys.stderr)
        return 1
    floors = json.loads(a.floors.read_text()) if a.floors else {}
    extra = side(a.out)
    if a.json:
        a.json.write_text(json.dumps({"rows": res, "side": extra, "floors": floors}, indent=1) + "\n")
    print(markdown(res, extra, floors, a.idle_w, a.theta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
