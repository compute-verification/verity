"""Scratch, never lands: part2.py's merged rows as the note's part 2 (Markdown on stdout)."""
import json
import sys
from pathlib import Path

rows = [r for r in json.loads(Path(sys.argv[1]).read_text()) if "missing" not in r]
BF16_A, BF16_R = 23682 / 7872, 23682 / 8065
U = ("gates", "AND", "rows alone")


def g(x):
    return "—" if x is None else f"{x:.4g}"


def q(a, b):
    return a / b if a and b else None


ok = [r for r in rows if r["C0"]]
coded = [r["entry"] for r in rows if not r["C0"]]
print("| entry | |C| gates | |C| AND | |C| rows | ρ gates | ρ AND | ρ rows | run's gates at S | run's AND | run's rows |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['entry']} | " + " | ".join(g(r[f"{k}{u}"]) for k in ("C", "rho", "run") for u in "012") + " |")

print("\nc by start (every entry of a start has its start's console c):\n")
print("| start | c gates | c AND | c rows |")
print("|---|---:|---:|---:|")
seen = {}
for r in rows:
    seen.setdefault(r["start"], r)
for s, r in sorted(seen.items()):
    print(f"| {s} | {g(r['c0'])} | {g(r['c1'])} | {g(r['c2'])} |")

print("\nFactors, each figure in AND gates and in rows alone over the same figure in gates (least, largest):\n")
print("| figure | AND / gates: least | largest | rows / gates: least | largest |")
print("|---|---:|---:|---:|---:|")
spans = {}
for name, k in (("|C|", "C"), ("c", "c"), ("ρ, the bound of record", "rho"), ("run's gates at S", "run")):
    fa = sorted(q(r[f"{k}1"], r[f"{k}0"]) for r in ok)
    fr = sorted(q(r[f"{k}2"], r[f"{k}0"]) for r in ok)
    spans[k] = (fa, fr)
    inv = f" (÷ {1 / fa[-1]:.4f} to {1 / fa[0]:.4f}; ÷ {1 / fr[-1]:.4f} to {1 / fr[0]:.4f})" if k != "rho" else ""
    print(f"| {name}{inv} | {fa[0]:.4f} | {fa[-1]:.4f} | {fr[0]:.4f} | {fr[-1]:.4f} |")
h = next(r for r in ok if r["entry"] == "s-t20-1.1x-ref-optim")
fh = (q(h["C1"], h["C0"]), q(h["C2"], h["C0"]))
ps = [(q(r["C1"], r["C0"]) / fh[0], q(r["C2"], r["C0"]) / fh[1], r["entry"]) for r in ok if r["start"] == "s-t20"]
print(f"| p = |C|/H, s-t20 entries against s-t20-1.1x-ref-optim | {min(p[0] for p in ps):.4f} | "
      f"{max(p[0] for p in ps):.4f} | {min(p[1] for p in ps):.4f} | {max(p[1] for p in ps):.4f} |")
sa, sr = 1 / (BF16_A * fh[0]), 1 / (BF16_R * fh[1])
print(f"| S = p·ρ = m·c/H, the same entries | {sa:.4f} | {sa:.4f} | {sr:.4f} | {sr:.4f} |")

print("\nMost moved (ρ in AND over ρ in gates, lowest first), with the kinds that mix in |C|:\n")
mv = sorted(ok, key=lambda r: q(r["rho1"], r["rho0"]))
for r in mv[:3]:
    print(f"- {r['entry']}: ρ × {q(r['rho1'], r['rho0']):.4f} in AND, × {q(r['rho2'], r['rho0']):.4f} in rows; "
          f"|C| at {q(r['C0'], r['C1']):.4f} gates an AND; products priced "
          + ", ".join(f"{k} {v:.1%}" for k, v in sorted(r["priced_in"].items())))
print("\nLeast moved:\n")
for r in mv[-3:][::-1]:
    print(f"- {r['entry']}: ρ × {q(r['rho1'], r['rho0']):.4f} in AND, × {q(r['rho2'], r['rho0']):.4f} in rows; "
          f"|C| at {q(r['C0'], r['C1']):.4f} gates an AND; products priced "
          + ", ".join(f"{k} {v:.1%}" for k, v in sorted(r["priced_in"].items())))

print("\nItem 6 numbers: per_period's printed AND gates over steps·|C|_AND·ρ_main (the circuit's AND count at main's ρ):\n")
gap = sorted((1 - r["printed_run_and0"] / (r["steps"] * r["C1"] * r["rho0"]), r["entry"]) for r in ok)
print(f"- least {gap[0][0]:.2%} ({gap[0][1]}), largest {gap[-1][0]:.2%} ({gap[-1][1]})")
eq = max(abs(1 - r["printed_run_and0"] / r["run1"]) for r in ok)
print(f"- against the run's AND gates in the AND unit (steps·|C|_AND·ρ_AND): at most {eq:.1e} apart")
print(f"\nNo figure in the library (`CODED`, needs a record): {', '.join(coded)}")
