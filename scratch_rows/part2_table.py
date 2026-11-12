"""Scratch, never lands: part2.py's merged rows as the note's tables and factors (Markdown on stdout)."""
import json
import sys
import tomllib
from pathlib import Path

rows = [r for r in json.loads(Path(sys.argv[1]).read_text()) if "missing" not in r]
held = set(tomllib.loads(Path("benchmarks/training_slowdown/training_slowdown/of_record.toml").read_text())["held"])
BF16 = 1480.125 / 492.0


def g(x):
    return "-" if x is None else f"{x:.4g}"


def ratio(a, b):
    return a / b if a and b else None


print("| entry | held | |C| gates | |C| AND | |C| rows | c gates | c AND | c rows | ρ gates | ρ AND | ρ rows | run gates | "
      "run AND (main prints) | run AND | run rows | binds (gates) |")
print("|" + "---|" * 16)
for r in rows:
    print(f"| {r['entry']} | {'yes' if r['entry'] in held else 'honest'} | {g(r['C0'])} | {g(r['C1'])} | {g(r['C2'])} | "
          f"{g(r['c0'])} | {g(r['c1'])} | {g(r['c2'])} | {g(r['rho0'])} | {g(r['rho1'])} | {g(r['rho2'])} | "
          f"{g(r['run0'])} | {g(r['printed_run_and0'])} | {g(r['run1'])} | {g(r['run2'])} | "
          f"{(r['binds']['0'] or '-').split(',')[0]} |")

figs = {"|C|": "C", "c": "c", "ρ (the bound of record)": "rho", "run's gates at ρ": "run"}
print("\nFactors, gates over AND and gates over rows (least, largest, over the entries with a value):\n")
print("| figure | gates / AND: least | largest | gates / rows: least | largest |")
print("|---|---:|---:|---:|---:|")
facts = {}
for name, k in figs.items():
    fa = [ratio(r[f"{k}0"], r[f"{k}1"]) for r in rows]
    fr = [ratio(r[f"{k}0"], r[f"{k}2"]) for r in rows]
    fa, fr = [x for x in fa if x], [x for x in fr if x]
    facts[k] = (fa, fr)
    if fa:
        print(f"| {name} | {min(fa):.4f} | {max(fa):.4f} | {min(fr):.4f} | {max(fr):.4f} |")
und = [(1 - r["printed_run_and0"] / r["run1"], r["entry"]) for r in rows if r["printed_run_and0"] and r["run1"]]
if und:
    print(f"\nper_period's printed AND gates against the run's AND gates: {min(und)[0]:+.2%} ({min(und)[1]}) to "
          f"{max(und)[0]:+.2%} ({max(und)[1]})")
mv = sorted((ratio(r["rho1"], r["rho0"]), r["entry"], ratio(r["C0"], r["C1"]), r["priced_in"]) for r in rows
            if ratio(r["rho1"], r["rho0"]))
print("\nρ's factor in AND gates (ρ AND / ρ gates), most moved first:")
for f, e, fc, p in mv[:3]:
    print(f"- {e}: ρ × {f:.4f}; its |C| at {fc:.4f} gates an AND; products priced in {p}")
print("least moved:", ", ".join(f"{e} × {f:.4f}" for f, e, *_ in mv[-3:]))
hon = {r["entry"]: r for r in rows}
h = hon.get("s-t20-1.1x-ref-optim")
if h:
    for r in rows:
        if r["start"] == "s-t20" and r["entry"] != h["entry"] and r["C0"] and h["C0"]:
            r["p_factor_vs_ref_optim"] = ratio(h["C0"], h["C1"]) / ratio(r["C0"], r["C1"])
    pf = [r["p_factor_vs_ref_optim"] for r in rows if r.get("p_factor_vs_ref_optim")]
    print(f"\np against s-t20-1.1x-ref-optim's step (C/H in AND gates over C/H in gates), s-t20 entries: "
          f"{min(pf):.4f} to {max(pf):.4f}; S = p·ρ moves by H's gates an AND over 3.008: "
          f"{ratio(h['C0'], h['C1']) / BF16:.4f}")
fc = [ratio(r["C0"], r["C1"]) for r in rows if ratio(r["C0"], r["C1"])]
print(f"\n|C|'s gates an AND over the entries: {min(fc):.4f} to {max(fc):.4f}; the bf16 GEMM's {BF16:.4f}")
