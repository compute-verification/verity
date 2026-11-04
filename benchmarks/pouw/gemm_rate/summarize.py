"""MACs per SM clock of every GEMM window rate.sh recorded, against 4,096, with theta and each shape set's step.

    python3 summarize.py OUT [--theta 989.4e12] [--json table.json] > table.md

OUT holds fb-<variant>.jsonl (`v31 fill-bench` lines, each with its `windows`) and torch.jsonl (torch_rate.py's). A
window is c back-to-back calls of M N K MACs between two stamps of every SM (stamp_kernel's, stamp_k's): its SM clocks,
summed over the SMs (%clock64 differences), and its span (%globaltimer, the SMs' mean). It gives:

- of 4096: c M N K / SM clocks / 4096, the tensor pipe's share of its peak 4,096 int8 MACs an SM clock;
- the tensor-idle share, 1 - of 4096: the SM clocks of the window that issue no int8 MAC;
- theta: c M N K / span / THETA (989.4 T MAC/s: 132 SMs, 4,096 MACs an SM clock at 1,830 MHz);
- the SM clock, SM clocks / SMs / span.

A row is an arm's median over its rounds, with the spread (max - min) of its of 4096. The step is an arm's MACs a call
summed over every shape it ran, over its SM clocks a call summed the same way: the share of the four linears' SM clocks
the tensor pipe is busy. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PEAK = 4096
THETA = 989.4e12
#: fill-bench's window arms and torch_rate.py's, as the table names them
LABELS = {"lt": "cuBLASLt, autotuned (fill-bench lt)", "gemm": "served GEMM (fill-bench gemm)",
          "gemm_d": "served GEMM of D (fill-bench gemm_d)", "wgl": "wgmma.linear port (fill-bench wgl)",
          "ovl": "served call, overlapped (fill-bench ovl)", "int_mm": "cuBLASLt, torch._int_mm"}
VARIANT_ARMS = ("gemm", "gemm_d", "ovl")


def label(arm: str, variant: str | None) -> str:
    if arm.startswith("wgmma:"):
        return f"s8_linear wgmma.linear {arm[6:]}"
    base = LABELS.get(arm, arm)
    return f"{base} [{variant}]" if variant and arm in VARIANT_ARMS else base


def span(a: list, b: list) -> dict:
    """Two stamps, each [(globaltimer ns, clock64, sm), ...] one a CTA: every SM's clocks between them summed, the SMs'
    mean span and their count; sms 0 unless both have one CTA on each of the same SMs and none timed out (sm ~0)."""
    ia, ib = {int(s): (int(t), int(c)) for t, c, s in a}, {int(s): (int(t), int(c)) for t, c, s in b}
    bad = (1 << 64) - 1
    if len(ia) != len(a) or set(ia) != set(ib) or bad in ia or -1 in ia:
        return {"sm_clocks": 0, "span_ns": 0, "sms": 0}
    clocks = sum(ib[s][1] - ia[s][1] for s in ia)
    ns = sum(ib[s][0] - ia[s][0] for s in ia) / len(ia)
    return {"sm_clocks": clocks, "span_ns": ns, "sms": len(ia)}


def point(w: dict, m: int, n: int, k: int, theta: float = THETA) -> dict:
    """One window's numbers: calls of m n k MACs over its SM clocks and span."""
    macs = w["calls"] * m * n * k
    s = w["span_ns"] * 1e-9
    of = macs / w["sm_clocks"] / PEAK
    return {"macs": macs, "s": s, "of_4096": of, "idle": 1 - of, "theta": macs / s / theta,
            "sm_mhz": w["sm_clocks"] / w["sms"] / w["span_ns"] * 1e3, "ms_call": 1e3 * s / w["calls"],
            "macs_call": m * n * k, "clocks_call": w["sm_clocks"] / w["calls"], "s_call": s / w["calls"]}


def windows(out: Path) -> list[dict]:
    """Every window under out, as {shape, m, n, k, arm, variant, round, calls, ms, sm_clocks, span_ns, sms, cmhz,
    nvml_mhz, reasons}."""
    rows = []
    for f in sorted(out.glob("fb-*.jsonl")):
        variant = f.stem[3:]
        for line in f.read_text().splitlines():
            if not line.startswith("{"):
                continue
            fb = json.loads(line)
            for w in fb.get("windows", []):
                rows.append({"shape": fb["shape"], "m": fb["m"], "n": fb["n"], "k": fb["k"], "variant": variant, **w})
    t = out / "torch.jsonl"
    for line in t.read_text().splitlines() if t.exists() else []:
        w = json.loads(line)
        if w.get("kind") == "window":
            rows.append({"variant": None, "cmhz": None, **w})
    return rows


def table(rows: list[dict], theta: float = THETA) -> tuple[list[dict], list[dict]]:
    """One row a (shape, arm, variant): the median of its windows' numbers and its of-4096 spread; and the windows
    refused (a stamp without one CTA on every SM)."""
    groups: dict = {}
    refused = [r for r in rows if not r.get("sms")]
    for r in rows:
        if r.get("sms"):
            key = (r["shape"], r["arm"], r["variant"] if r["arm"] in VARIANT_ARMS else None)
            groups.setdefault(key, []).append(r)
    res = []
    for (shp, arm, variant), ws in groups.items():
        pts = [point(w, w["m"], w["n"], w["k"], theta) for w in ws]
        med = {f: statistics.median(p[f] for p in pts)
               for f in ("of_4096", "idle", "theta", "sm_mhz", "ms_call", "clocks_call", "s_call")}
        cm = [w["cmhz"] for w in ws if w.get("cmhz")]
        nv = [w["nvml_mhz"] for w in ws if w.get("nvml_mhz")]
        reasons = 0
        for w in ws:
            reasons |= int(w.get("reasons") or 0)
        res.append({"shape": shp, "m": ws[0]["m"], "n": ws[0]["n"], "k": ws[0]["k"], "arm": arm, "variant": variant,
                    "label": label(arm, variant), "rounds": len(ws), **med, "macs_call": pts[0]["macs_call"],
                    "spread": max(p["of_4096"] for p in pts) - min(p["of_4096"] for p in pts),
                    "window_s": statistics.median(p["s"] for p in pts),
                    "cmhz": statistics.median(cm) if cm else None, "nvml_mhz": statistics.median(nv) if nv else None,
                    "reasons": reasons})
    lt = {r["shape"]: r["theta"] for r in res if r["arm"] == "lt"}
    for r in res:
        r["of_lt"] = r["theta"] / lt[r["shape"]] if r["shape"] in lt else None
    shapes = list(dict.fromkeys(r["shape"] for r in res))
    return sorted(res, key=lambda r: (shapes.index(r["shape"]), -r["of_4096"])), refused


def steps(res: list[dict], theta: float = THETA) -> list[dict]:
    """Each (arm, variant) on every shape: MACs a call summed over the shapes over SM clocks a call summed the same way."""
    shapes = set(r["shape"] for r in res)
    by: dict = {}
    for r in res:
        by.setdefault((r["arm"], r["variant"]), []).append(r)
    out = []
    for (arm, variant), rs in by.items():
        if len(shapes) < 2 or {r["shape"] for r in rs} != shapes:
            continue
        macs, clocks, s = (sum(r[f] for r in rs) for f in ("macs_call", "clocks_call", "s_call"))
        out.append({"arm": arm, "variant": variant, "label": label(arm, variant), "shapes": len(rs),
                    "of_4096": macs / clocks / PEAK, "idle": 1 - macs / clocks / PEAK, "theta": macs / s / theta,
                    "ms_step": 1e3 * s})
    return sorted(out, key=lambda r: -r["of_4096"])


def fmt(x, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def markdown(res: list[dict], st: list[dict], refused: list[dict], theta: float) -> str:
    out = ["of 4096 = MACs / SM clocks / 4096, the SM clocks summed over every SM between two stamps of all of them "
           f"(clock64); tensor-idle = 1 - of 4096; theta against {theta / 1e12:g} T MAC/s over the stamps' span "
           "(globaltimer). SM MHz from the stamps; clock64 MHz on CTA 0 of each wg GEMM launch and NVML's SM clock "
           "(halfway through each window) beside it. x cuBLASLt: theta over fill-bench lt's on the shape.", "",
           "| shape | arm | rounds | ms a call | of 4096 | spread | tensor-idle | theta | x cuBLASLt | SM MHz "
           "| clock64 MHz | NVML MHz | reasons |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        out.append(f"| {r['shape']} | {r['label']} | {r['rounds']} | {r['ms_call']:.3f} | {r['of_4096']:.4f} | "
                   f"{r['spread']:.4f} | {r['idle']:.4f} | {r['theta']:.4f} | {fmt(r['of_lt'], '.3f')} | "
                   f"{r['sm_mhz']:.0f} | {fmt(r['cmhz'], '.0f')} | {fmt(r['nvml_mhz'], '.0f')} | {r['reasons']:#x} |")
    if st:
        out += ["", "The step (every shape, MACs and SM clocks a call summed):", "",
                "| arm | shapes | ms a step | of 4096 | tensor-idle | theta |", "|---|---|---|---|---|---|"]
        out += [f"| {r['label']} | {r['shapes']} | {r['ms_step']:.3f} | {r['of_4096']:.4f} | {r['idle']:.4f} | "
                f"{r['theta']:.4f} |" for r in st]
    if refused:
        out += ["", f"Refused {len(refused)} window(s) whose stamps weren't one CTA on every SM: "
                + ", ".join(f"{w['shape']} {w['arm']} round {w.get('round')}" for w in refused)]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res, refused = table(windows(a.out), a.theta)
    if not res:
        print(f"no windows under {a.out}", file=sys.stderr)
        return 1
    st = steps(res, a.theta)
    if a.json:
        a.json.write_text(json.dumps({"rows": res, "steps": st, "refused": refused}, indent=1) + "\n")
    print(markdown(res, st, refused, a.theta), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
