"""The cost table of egemm.cu's bench: each arm's time over cuBLASLt's plain int8 GEMM and over its plain E4M3 GEMM (the
faster honest 8-bit baseline), 1 + gamma, per shape and as geometric means over the Llama-3.1-8B linears and over the
attention products, with the gate's verdict for the arm.

python3 etable.py BENCH.jsonl GATE.jsonl > table.md
"""

import json
import math
import re
import sys

GROUPS = {"Llama linears": ["qkv", "o", "gate_up", "down"], "4096^3": ["sq4096"],
          "attention QK^T": ["qk2k", "qk8k"], "attention PV": ["pv2k", "pv8k"]}
RCS = [("cg", "none"), ("cduo32", "duo 32"), ("cduo64", "duo 64"), ("ct4c32", "TB4C 32")]
ROWS = [("no edit, custom mainloop", "{rc}"), ("edit, separate pass", "sep_{rc}"), ("edit, fused", "fus_{rc}")]


def gate_verdicts(path):
    checks = {}
    for line in open(path):
        for key, v in json.loads(line).items():
            if isinstance(v, bool) and key not in ("ok", "receipts"):
                checks.setdefault(key, []).append(v)
    return checks


def arm_gate(name, checks):
    fmt = name.rsplit("_", 1)[1]
    keys = [key for key in checks if re.fullmatch(rf"(U|R|XA|XB)_{re.escape(name)}(_nonzero)?", key)]
    if not keys:
        return "unchecked"
    if name.startswith("sep") or name.startswith("fus"):
        keys += ["noise_keys", "noise_uniform_on_S", f"AeBe_{fmt}_range", f"AeBe_{fmt}_exact_range"]
    keys += [f"AB_{fmt}_range", "every_arm_checked"]
    return "pass" if all(key in checks and all(checks[key]) for key in keys) else "FAIL"


def geo(by, shapes, name, base):
    xs = [by[s]["arms"][name]["ms"] / by[s]["arms"][base]["ms"] for s in shapes
          if s in by and name in by[s]["arms"] and base in by[s]["arms"]]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if len(xs) == len(shapes) else float("nan")


def main(bench, gate):
    rows = [json.loads(line) for line in open(bench) if line.strip()]
    by = {j["shape"]: j for j in rows}
    checks = gate_verdicts(gate)
    names = sorted({a for j in rows for a in j["arms"]})
    verdict = {a: arm_gate(a, checks) for a in names}
    out = ["# The edited GEMM: 1 + gamma on sm_120", ""]
    if rows:
        j = rows[0]
        out.append(f"GPU {j['gpu']}, {j['sms']} SMs; cuBLASLt algorithms (plain int8, plain E4M3) {j['lt_algo']}.")
    out.append("Gate (eref.py, every arm bit-exact at four small shapes): "
               + ("every timed arm passed" if all(v == "pass" for v in verdict.values())
                  else ", ".join(f"{a} {v}" for a, v in verdict.items() if v != "pass")) + ".")
    for base, bname in [("plain_i8", "cuBLASLt int8 (s8 -> s32)"), ("plain_e4", "cuBLASLt E4M3 (-> f32)")]:
        for group, shapes in GROUPS.items():
            out += ["", f"## {group} ({', '.join(shapes)}), against {bname}" + (", geomean" if len(shapes) > 1 else ""), "",
                    "| construction | format | " + " | ".join(r for _, r in RCS) + " |",
                    "|---|---|" + "---|" * len(RCS)]
            for fmt in ["i8", "e4"]:
                out.append(f"| plain cuBLASLt | {fmt} | {geo(by, shapes, f'plain_{fmt}', base):.2f} |" + " – |" * (len(RCS) - 1))
                out.append(f"| edit, separate pass + cuBLASLt | {fmt} | {geo(by, shapes, f'sep_lt_{fmt}', base):.2f} |"
                           + " – |" * (len(RCS) - 1))
                for label, pat in ROWS:
                    cells = []
                    for rc, _ in RCS:
                        name = pat.format(rc=rc) + f"_{fmt}"
                        x = geo(by, shapes, name, base)
                        cells.append(f"{x:.2f}" + ("" if verdict.get(name) == "pass" else f" ({verdict.get(name, 'untimed')})"))
                    out.append(f"| {label} | {fmt} | " + " | ".join(cells) + " |")
                out.append(f"| (the edit pass alone) | {fmt} | {geo(by, shapes, f'sepform_{fmt}', base):.2f} |"
                           + " – |" * (len(RCS) - 1))
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
