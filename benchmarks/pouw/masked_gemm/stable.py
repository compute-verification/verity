"""The table of sgemm.cu's bench: gamma_gen (the SHA-512 generation of both operands' noise alone) and the cheapest
1 + gamma of the edited GEMM with that generation, per shape, over cuBLASLt's plain int8 GEMM (and its E4M3 GEMM), with
no receipts and with duo at L = 32, and the gate's verdict for each arm.

python3 stable.py BENCH.jsonl GATE.jsonl > table.md
"""

import json
import math
import re
import sys

ALS = ["pm1", "u17", "u17w", "u17p", "u17wp"]
NONE = ["slt_{a}_i8", "slt_{a}_e4", "fus_{a}_i8", "fov_{a}_i8"]
DUO = ["sduo_{a}_i8", "sduo_{a}_e4", "fusd_{a}_i8", "fovd_{a}_i8"]
GROUPS = {"Llama linears": ["qkv", "o", "gate_up", "down"], "4096^3": ["sq4096"],
          "attention QK^T": ["qk2k", "qk8k"], "attention PV": ["pv2k", "pv8k"]}


def gate_verdicts(path):
    checks = {}
    for line in open(path):
        for key, v in json.loads(line).items():
            if isinstance(v, bool) and key not in ("ok", "receipts"):
                checks.setdefault(key, []).append(v)
    return checks


def arm_gate(name, checks):
    keys = [key for key in checks if re.fullmatch(rf"(U|R|XA|XB|G)_{re.escape(name)}", key)]
    if not keys:
        return "unchecked"
    parts = name.split("_")
    if parts[0] != "plain":
        al = parts[1]
        keys += ["sigma", f"noise_{al}_in_S", f"noise_{al}_uniform"]
        if parts[0] != "gen":
            keys += [f"AeBe_{al}_{parts[2]}_range", f"AeBe_{al}_{parts[2]}_exact_range"]
    keys += ["every_arm_checked"] + [key for key in checks if key.startswith("AB_")]
    return "pass" if all(key in checks and all(checks[key]) for key in keys) else "FAIL"


def cheapest(arms, pats, al, base, verdict):
    best = None
    for pat in pats:
        name = pat.format(a=al)
        if name in arms and verdict.get(name) == "pass":
            x = arms[name]["ms"] / arms[base]["ms"]
            if best is None or x < best[0]:
                best = (x, name)
    return best


def geo(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main(bench, gate):
    rows = [json.loads(line) for line in open(bench) if line.strip()]
    checks = gate_verdicts(gate)
    names = sorted({a for j in rows for a in j["arms"]})
    verdict = {a: arm_gate(a, checks) for a in names}
    out = ["# SHA-512 noise generation: gamma_gen and the cheapest 1 + gamma on sm_120", ""]
    if rows:
        out.append(f"GPU {rows[0]['gpu']}, {rows[0]['sms']} SMs; cuBLASLt algorithms (plain int8, plain E4M3) {rows[0]['lt_algo']}.")
    out.append("Gate (sref.py against hashlib, every arm bit-exact at four small shapes): "
               + ("every timed arm passed" if all(v == "pass" for v in verdict.values())
                  else ", ".join(f"{a} {v}" for a, v in verdict.items() if v != "pass")) + ".")
    for base, bname in [("plain_i8", "cuBLASLt int8 (s8 -> s32)"), ("plain_e4", "cuBLASLt E4M3 (-> f32)")]:
        al = ", ".join(ALS)
        out += ["", f"## Against {bname} (alphabets {al} in each cell)", "",
                "| shape | plain ms | gamma_gen | gen + edit pass | cheapest, no receipts | cheapest, duo 32 |", "|---|---|---|---|---|---|"]
        acc = {}
        for j in rows:
            arms = j["arms"]
            if base not in arms:
                continue
            b = arms[base]["ms"]
            g = [arms[f"gen_{a}"]["ms"] / b if f"gen_{a}" in arms else float("nan") for a in ALS]
            f = [arms[f"sform_{a}_i8"]["ms"] / b if f"sform_{a}_i8" in arms else float("nan") for a in ALS]
            cells = []
            for pats in (NONE, DUO):
                cs = []
                for a in ALS:
                    c = cheapest(arms, pats, a, base, verdict)
                    cs.append(f"{c[0]:.2f} ({c[1]})" if c else "–")
                    if c:
                        acc.setdefault((tuple(pats), a), {})[j["shape"]] = c[0]
                cells.append(" / ".join(cs))
            for i, a in enumerate(ALS):
                acc.setdefault(("gen", a), {})[j["shape"]] = g[i]
            out.append(f"| {j['shape']} | {b:.4f} | {' / '.join(f'{x:.3f}' for x in g)} | {' / '.join(f'{x:.3f}' for x in f)} | "
                       + " | ".join(cells) + " |")
        out += ["", f"Geomeans by group (gamma_gen; cheapest no receipts; cheapest duo 32; each {al}):", ""]
        for group, shapes in GROUPS.items():
            parts = []
            for kind in ["gen", tuple(NONE), tuple(DUO)]:
                vals = []
                for a in ALS:
                    d = acc.get((kind, a), {})
                    xs = [d[s] for s in shapes if s in d]
                    vals.append(geo(xs) if len(xs) == len(shapes) else float("nan"))
                parts.append(", ".join(f"{v:.3f}" for v in vals))
            out.append(f"- {group}: gamma_gen {parts[0]}; no receipts {parts[1]}; duo 32 {parts[2]}")
    out += ["", "## Every arm, ms (median of interleaved repetitions) and 1 + gamma against int8 / E4M3 cuBLASLt", "",
            "| shape | m | n | k | batch | arm | ms | / i8 | / e4 | MHz | gate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in rows:
        arms = j["arms"]
        for a in sorted(arms):
            ri = arms[a]["ms"] / arms["plain_i8"]["ms"] if "plain_i8" in arms else float("nan")
            re4 = arms[a]["ms"] / arms["plain_e4"]["ms"] if "plain_e4" in arms else float("nan")
            out.append(f"| {j['shape']} | {j['m']} | {j['n']} | {j['k']} | {j['batch']} | {a} | {arms[a]['ms']:.4f} | {ri:.3f} | "
                       f"{re4:.3f} | {arms[a]['mhz']:.0f} | {verdict[a]} |")
        if j.get("skipped"):
            out.append(f"| {j['shape']} | | | | | skipped: {', '.join(j['skipped'])} | | | | | |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
