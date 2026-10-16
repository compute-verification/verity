"""The cost table of mgemm.cu's bench: each arm's time over its format's plain GEMM (cuBLASLt), 1 + gamma, per shape and as
a geometric mean over the Llama-3.1-8B linears, with the gate's verdict for the arm.

python3 mtable.py BENCH.jsonl GATE.jsonl > table.md
"""

import json
import math
import re
import sys

LLAMA = ["qkv", "o", "gate_up", "down"]


def describe(name):
    """(construction, format, r, receipt, forming, GEMM) of an arm name."""
    m = re.fullmatch(r"(?:(cg|cduo32|cduo64|ct4c32|ct4c64)_)?(plain|full|fullform|low|lowform|lowsep|lowtc|lowtcform)(\d*)_(i8|e4)", name)
    rc, kind, r, fmt = m.group(1), m.group(2), m.group(3), m.group(4)
    receipt = {None: "-", "cg": "-", "cduo32": "duo 32", "cduo64": "duo 64", "ct4c32": "TB4C 32", "ct4c64": "TB4C 64"}[rc]
    gemm = "custom" if rc else "cuBLASLt"
    if kind == "plain":
        return "plain", fmt, "-", receipt, "-", gemm
    if kind.startswith("full"):
        return "full rank", fmt, "-", receipt, "CUDA cores", "none (forming alone)" if kind == "fullform" else gemm
    forming = "tensor cores" if kind.startswith("lowtc") else "CUDA cores"
    if kind.endswith("form"):
        gemm = "none (forming alone)"
    elif kind == "lowsep":
        gemm = "cuBLASLt, peel a second GEMM"
    return "low rank", fmt, r, receipt, forming, gemm


def gate_verdicts(path):
    checks = {}
    for line in open(path):
        j = json.loads(line)
        for key, v in j.items():
            if isinstance(v, bool) and key not in ("ok", "receipts"):
                checks.setdefault(key, []).append(v)
    return checks


def arm_gate(name, checks):
    keys = [key for key in checks if re.fullmatch(rf"(U|R|XA|XB)_{re.escape(name)}(_nonzero)?", key)]
    _, fmt, r, *_ = describe(name)
    if name.startswith(("full", "cg_full", "cduo32_full", "cduo64_full", "ct4c32_full", "ct4c64_full")):
        keys += [f"full_{fmt}_peel", f"full_{fmt}_grid"]
    elif r != "-":
        keys += [f"low{r}_{fmt}_peel", f"low{r}_{fmt}_grid", f"low{r}_{fmt}_digits_fit"]
    keys += ["digit_overflow_flag"]
    if not all(key in checks for key in keys) or not any(key.startswith(("U_", "XA_")) for key in keys):
        return "unchecked"
    return "pass" if all(all(checks[key]) for key in keys) else "FAIL"


def geo(by, shapes, name, fmt, which):
    xs = [by[s]["arms"][name]["ms"] / by[s]["arms"][f"plain_{fmt}"]["ms"] for s in shapes
          if s in which and name in by[s]["arms"] and f"plain_{fmt}" in by[s]["arms"]]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def compact(by, shapes, checks):
    """1 + gamma, geometric mean over the Llama linears (and 4096^3 in brackets), one row a construction, format and r:
    no receipts (the best of cuBLASLt over the stacked operands and the custom mainloop), and each receipt on the custom
    mainloop; both operands formed on CUDA cores on every call."""
    print("| construction | format | r | no receipts | duo L = 32 | duo L = 64 | TB4C L = 32 | TB4C L = 64 | gate |")
    print("|---|---|---|---|---|---|---|---|---|")
    for fmt in ("i8", "e4"):
        for tag, r in [("full", "-")] + [(f"low{r}", str(r)) for r in (8, 16, 32, 64, 128)]:
            plain = [f"{'full' if tag == 'full' else tag}_{fmt}", f"cg_{tag}_{fmt}"]
            rc = [f"{c}_{tag}_{fmt}" for c in ("cduo32", "cduo64", "ct4c32", "ct4c64")]
            cells = []
            vals = {n: (geo(by, shapes, n, fmt, LLAMA), geo(by, shapes, n, fmt, ["sq4096"])) for n in plain + rc}
            best = min(plain, key=lambda n: vals[n][0])
            for n in [best] + rc:
                g, s = vals[n]
                cells.append(f"{g:.2f} ({s:.2f})")
            verdicts = {arm_gate(n, checks) for n in plain + rc}
            v = "pass" if verdicts == {"pass"} else "/".join(sorted(verdicts))
            print(f"| {'full rank' if tag == 'full' else 'low rank'} | {fmt} | {r} | " + " | ".join(cells) + f" | {v} |")


def main(bench, gate):
    rows = [json.loads(line) for line in open(bench)]
    checks = gate_verdicts(gate)
    shapes = [r["shape"] for r in rows]
    by = {r["shape"]: r for r in rows}
    names = list(rows[0]["arms"])
    low = rows[0]["low"]
    print("| construction | format | r | digits l (llama/4096) | receipts (L) | forming | GEMM | "
          + " | ".join(shapes) + " | geomean llama | gate |")
    print("|" + "---|" * (8 + len(shapes)))
    for name in names:
        cons, fmt, r, receipt, forming, gemm = describe(name)
        ratios = []
        for s in shapes:
            a = by[s]["arms"]
            ratios.append(a[name]["ms"] / a[f"plain_{fmt}"]["ms"] if name in a and f"plain_{fmt}" in a else float("nan"))
        ll = [x for s, x in zip(shapes, ratios) if s in LLAMA and not math.isnan(x)]
        geo = math.exp(sum(map(math.log, ll)) / len(ll)) if ll else float("nan")
        fi = 0 if fmt == "i8" else 1
        digits = "-"
        if r != "-":
            digits = "/".join(str(by[s]["low"][r]["L"][fi]) for s in ("o", "down") if s in by)
        print(f"| {cons} | {fmt} | {r} | {digits} | {receipt} | {forming} | {gemm} | "
              + " | ".join(f"{x:.3f}" for x in ratios) + f" | {geo:.3f} | {arm_gate(name, checks)} |")
    print()
    compact(by, shapes, checks)
    print()
    print("plain GEMM ms (cuBLASLt): " + ", ".join(
        f"{s} i8 {by[s]['arms']['plain_i8']['ms']:.4f} e4 {by[s]['arms']['plain_e4']['ms']:.4f}"
        for s in shapes if "plain_i8" in by[s]["arms"] and "plain_e4" in by[s]["arms"]))
    print("SM MHz (median, plain_i8): " + ", ".join(f"{s} {by[s]['arms']['plain_i8']['mhz']:.0f}" for s in shapes if "plain_i8" in by[s]["arms"]))
    print("digits l, per r, [i8, e4] at k = 4096 and 14336: " + json.dumps(
        {r: [by[s]["low"][r]["L"] for s in ("o", "down") if s in by] for r in low}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
