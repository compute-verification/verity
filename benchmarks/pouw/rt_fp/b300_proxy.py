"""A B300 proxy for `overcredit.py`: are the C1 and FP-unit rows over-credited where int8 is slow?

B300 has no sheet in the machine model. Its dense int8 is about 1/30 of its FP8 (NVIDIA's Blackwell Ultra datasheet),
and sm_103a exposes no tcgen05 `kind::i8`. The proxy is B200's data sheet vector DS-100 with every UTCIMMA form's
DS_tensor cost times 30 (`SLOW`), so the cheater's int8 tensor route is B300's. Its scalar routes (lanes, baseline,
uniform) are B200's. Each variant's credited floor ♭ stays B200's, the FP route at B200's FP8 rate, which B300 matches.

It reads `overcredit.py`'s OUT (the variants and their cheaters in OUT/data/sm_100, and overcredit.json's sheet rows) and
prices each pair at q* ½ (Lean's floor) and 1.22 (Theorem C) with credit ♭ and operands free. The bound is
1 − T_cheat / ♭_V per committed word, as in `overcredit.py`. A cheater that leaves the tensor route writes each of the
variant's words at q* lane instructions plus the fold, which needs no extraction.

Run: `uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/rt_fp/b300_proxy.py OVERCREDIT_OUT OUT`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SLOW = 30
VECTOR = "DS-100"
QS = (0.5, 1.22)


def slow_int8(data: Path) -> int:
    p = data / "census.csv"
    n, lines = 0, []
    for line in p.read_text().splitlines():
        m = re.match(r"(UTCIMMA_[^,]+),DS_tensor,,,([0-9.]+),(.*)$", line)
        if m:
            line, n = f"{m[1]},DS_tensor,,,{float(m[2]) * SLOW:g},{m[3]}", n + 1
        lines.append(line)
    p.write_text("\n".join(lines) + "\n")
    return n


def priced(data: Path, design: str, q: float) -> dict:
    cmd = ["machine-model", "--data", str(data), "eps", design, "--credit", "floor", "--operands", "free", "--q", f"{q:g}",
           "--vector", VECTOR, "--json"]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def main(oc: Path, out: Path) -> int:
    b200 = oc / "data" / "sm_100"
    proxy = out / "data" / "b300-proxy"
    shutil.rmtree(proxy, ignore_errors=True)
    shutil.copytree(b200, proxy)
    if not slow_int8(proxy):
        raise SystemExit(f"{proxy}: no UTCIMMA DS_tensor line to slow")
    pairs = sorted({(r["variant"], r["cheater"]) for r in json.loads((oc / "overcredit.json").read_text())["rows"]
                    if r["device"] == "sm_100" and r["reading"] == "sheet"})
    rows = []
    for q in QS:
        for var, cheat in pairs:
            v, c = priced(b200, var, q), priced(proxy, cheat, q)
            credit = v["baseline"]["cycles_per_mma"]
            best = min(("cheat", "baseline"), key=lambda k: c[k]["cycles_per_mma"])
            t = c[best]["cycles_per_mma"] * c["L"] / v["L"]
            words = {k: round(x, 3) for k, x in c["cheat"]["words"].items() if x} if best == "cheat" else {"tensor": 1.0}
            rows.append({"q": q, "variant": var, "credit": credit, "model_core": v["core"], "cheater": cheat,
                         "route": best, "words": words, "cheat": t, "binds": c[best]["binding"], "bound": 1 - t / credit})
    (out / "b300_proxy.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [f"B300 proxy: {VECTOR} with UTCIMMA's DS_tensor ×{SLOW}; ♭ at B200's FP rate; credit ♭, operands free.", "",
             "| q* | variant | ♭ | model core ε (B200) | cheater | its words | T_cheat (binds) | ε_machine ≥ |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['q']:g} | {r['variant']} | {r['credit']:.1f} | {100 * r['model_core']:.3f}% | {r['cheater']} "
              f"({r['route']}) | {r['words']} | {r['cheat']:.1f} ({', '.join(r['binds'])}) | **{100 * r['bound']:.1f}%** |"
              for r in rows]
    (out / "b300_proxy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
