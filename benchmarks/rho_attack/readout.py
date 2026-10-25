"""The readout's two absolute tables from a `rho_attack.py` run (matched c, eta = 1), GRPO and SFT as a range:

    python3 readout.py RUN.json --out readout.md

- full training: the bound with its covering count at beta_v and at beta_gx, the cheapest recipe-kept attack, the
  cheapest attack overall, and the latter over each bound;
- the cliff just above k: the smallest S label above k, and S = 2k (inputs interpolated in log S between the labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIGS = {"deepseek-v4-pro": "DeepSeek-V4-Pro", "kimi-k2": "Kimi K2"}
PRECISIONS = ("fp8", "nvfp4")
KS = ("10kB", "1MB", "2.5MB")
WORKLOADS = ("grpo", "sft")


def sig(x: float) -> str:
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.2e}"


def span(xs: list[float | None]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "infeasible"
    lo, hi = sig(min(xs)), sig(max(xs))
    return lo if lo == hi else f"{lo}-{hi}"


def names(xs: list[str | None]) -> str:
    return " / ".join(dict.fromkeys(x or "infeasible" for x in xs))


def attack(rows: list[dict], key: str) -> str:
    return (f"{span([r[key]['rho'] if r[key] else None for r in rows])} "
            f"({names([r[key]['strategy'] if r[key] else None for r in rows])})")


def ratio(rows: list[dict], bound: str) -> str:
    return span([r["attack"]["rho"] / r["bound"][bound] if r["attack"] else None for r in rows])


def tables(doc: dict) -> str:
    order = lambda x: WORKLOADS.index(x.get("workload", "sft")) if x.get("workload", "sft") in WORKLOADS else 9  # noqa: E731
    rows = sorted((r for r in doc["rows"] if r["c_mode"] == "eta1"), key=order)
    curves = sorted((c for c in doc["curves"] if c["c_mode"] == "eta1"), key=order)
    out = ["Full training, matched c (eta = 1), absolute rho; GRPO and SFT as a range (low-high), strategies GRPO / SFT "
           "where they differ.", "",
           "| config | precision | k | bound (beta_v count) | bound (beta_gx count) | cheapest recipe-kept attack "
           "(strategy) | cheapest attack overall (strategy) | overall / bound (beta_v count) | "
           "overall / bound (beta_gx count) |", "|" + "---|" * 9]
    for cfg, name in CONFIGS.items():
        for prec in PRECISIONS:
            for k in KS:
                at = [r for r in rows if (r["config"], r["precision"], r["k_label"], r["S_label"]) == (cfg, prec, k, "full")
                      and r["workload"] in WORKLOADS]
                if not at:
                    continue
                out.append(f"| {name} | {prec} | {k} | {span([r['bound']['rho'] for r in at])} | "
                           f"{span([r['bound']['rho_gx'] for r in at])} | {attack(at, 'attack_exact')} | "
                           f"{attack(at, 'attack')} | {ratio(at, 'rho')} | {ratio(at, 'rho_gx')} |")
    out += ["", "The cliff just above k, matched c (eta = 1), absolute rho; GRPO and SFT as a range. The bound there is the "
            "cliff formula, the same under either count.", "",
            "| config | precision | k | smallest S label above k (bits) | bound | cheapest recipe-kept (strategy) | "
            "cheapest overall (strategy) | S = 2k: bound | S = 2k: cheapest recipe-kept (strategy) | "
            "S = 2k: cheapest overall (strategy) |", "|" + "---|" * 10]
    for cfg, name in CONFIGS.items():
        for prec in PRECISIONS:
            for k in KS:
                mine = [r for r in rows if (r["config"], r["precision"], r["k_label"]) == (cfg, prec, k)
                        and r["workload"] in WORKLOADS and r["S_label"] != "full"]
                above = [r for r in mine if r["S_bits"] > r["k_bits"]]
                cv = [c for c in curves if (c["config"], c["precision"], c["k_label"]) == (cfg, prec, k)
                      and c.get("workload") in WORKLOADS and "cliff_2k" in c]
                if not above or not cv:
                    continue
                s_min = min(r["S_bits"] for r in above)
                lab = [r for r in above if r["S_bits"] == s_min]
                c2 = [c["cliff_2k"] for c in cv]
                out.append(f"| {name} | {prec} | {k} | {lab[0]['S_label']} ({sig(s_min)}) | "
                           f"{span([r['bound']['rho'] for r in lab])} | {attack(lab, 'attack_exact')} | "
                           f"{attack(lab, 'attack')} | {span([c['bound'] for c in c2])} | "
                           f"{span([c['attack_exact'] for c in c2])} ({names([c['strategy_exact'] for c in c2])}) | "
                           f"{span([c['attack'] for c in c2])} ({names([c['strategy'] for c in c2])}) |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    text = tables(json.loads(Path(args.run).read_text()))
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
