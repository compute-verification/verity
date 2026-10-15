"""The table of pvgemm.cu's bench: P V's 1 + gamma over cuBLASLt's plain int8 P V, per sequence length and alphabet, for
the shared P noise (separate pass, fused) and the noise added where P is produced (shared or unshared; the route's time
less the noise-free producer's), with no receipts and with duo at L = 32, and gamma_gen; the gate's verdict for each arm.

python3 pvtable.py BENCH.jsonl GATE.jsonl > table.md
"""

import json
import re
import sys

ALS = ["pm1", "u17", "u17w", "u17p", "u17wp"]
# (label, arms summed with sign); prodN subtracted from the producer's arms (attention writes P~ anyway)
ROWS = [("gamma_gen, shared (E + V's noise)", [("genP", 1), ("genV", 1)]),
        ("shared, separate pass + cuBLASLt", [("sepS", 1)]),
        ("shared, fused (serialized)", [("fusS", 1)]),
        ("shared, fused (overlapped)", [("fovS", 1)]),
        ("producer, shared E + cuBLASLt", [("prodSlt", 1)]),
        ("producer, unshared SHA + cuBLASLt", [("prodUlt", 1)]),
        ("shared, separate pass + duo 32", [("sepSd", 1)]),
        ("shared, fused + duo 32", [("fusSd", 1)]),
        ("producer, shared E + duo 32", [("prodSd", 1)]),
        ("producer, unshared SHA + duo 32", [("prodUd", 1)]),
        ("(producer's increment, shared: prodS - prodN)", [("prodS", 1)]),
        ("(producer's increment, unshared: prodU - prodN)", [("prodU", 1)]),
        ("(V's edit pass alone)", [("editV", 1)])]
NONE = ["sepS", "fusS", "fovS", "prodSlt", "prodUlt"]
DUO = ["sepSd", "fusSd", "prodSd", "prodUd"]


def gate_verdicts(path):
    checks = {}
    for line in open(path):
        for key, v in json.loads(line).items():
            if isinstance(v, bool) and key not in ("ok",):
                checks.setdefault(key, []).append(v)
    return checks


def arm_gate(name, checks):
    keys = [k for k in checks if re.fullmatch(rf"(U|R|XA|XB|E|G)_{re.escape(name)}", k)]
    if name.startswith("prod") and name not in ("prodN", "prodmax"):
        keys += [f"XA_prod{name[4]}_{name.rsplit('_', 1)[1]}"]
    if name in ("prodN", "prodmax") or name.startswith("prod"):
        keys += ["Pt_range", "Pt_softmax_within_1", "mx_setup"] + (["mx_prodmax"] if name == "prodmax" else [])
    if not keys:
        return "unchecked"
    if "_" in name:
        al = name.rsplit("_", 1)[1]
        keys += [f"noise_{al}_in_S", f"Ps_{al}_range"]
    keys += ["sigma", "inputs_range", "every_arm_checked"]
    return "pass" if all(k in checks and all(checks[k]) for k in keys) else "FAIL"


def value(arms, parts, al):
    t = 0.0
    for a, sign in parts:
        name = f"{a}_{al}"
        if name not in arms:
            return None
        t += sign * arms[name]["ms"]
    if parts[0][0].startswith("prod"):
        t -= arms["prodN"]["ms"]
    return t / arms["plain"]["ms"]


def main(bench, gate):
    rows = [json.loads(line) for line in open(bench) if line.strip()]
    checks = gate_verdicts(gate)
    names = sorted({a for j in rows for a in j["arms"]})
    verdict = {a: arm_gate(a, checks) for a in names}
    out = ["# P V's noise cost: 1 + gamma on sm_120 against cuBLASLt int8 P V", ""]
    if rows:
        out.append(f"GPU {rows[0]['gpu']}, {rows[0]['sms']} SMs, H = {rows[0]['H']} heads, head dimension 128; cuBLASLt algorithms "
                   f"{rows[0]['lt_algo']}.")
    out.append("Gate (pvref.py against hashlib, every arm bit-exact at three small shapes): "
               + ("every timed arm passed" if all(v == "pass" for v in verdict.values())
                  else ", ".join(f"{a} {v}" for a, v in verdict.items() if v != "pass")) + ".")
    for j in rows:
        arms = j["arms"]
        out += ["", f"## s = {j['s']} (plain P V {arms['plain']['ms']:.4f} ms; noise-free producer prodN {arms['prodN']['ms']:.4f} ms "
                f"= {arms['prodN']['ms'] / arms['plain']['ms']:.2f}, its row-max pass {arms['prodmax']['ms'] / arms['plain']['ms']:.2f}; "
                f"duo 32 alone {arms['cduo']['ms'] / arms['plain']['ms']:.2f})", "",
                "| method | " + " | ".join(ALS) + " |", "|---|" + "---|" * len(ALS)]
        for label, parts in ROWS:
            cells = []
            for al in ALS:
                v = value(arms, parts, al)
                names_ = [f"{p}_{al}" for p, _ in parts]
                bad = [n for n in names_ if verdict.get(n) != "pass"]
                cells.append("–" if v is None else f"{v:.3f}" + (f" ({', '.join(verdict.get(n, 'untimed') for n in bad)})" if bad else ""))
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        for lab, pats in [("cheapest, no receipts", NONE), ("cheapest, duo 32", DUO)]:
            cells = []
            for al in ALS:
                best = None
                for p in pats:
                    if verdict.get(f"{p}_{al}") != "pass":
                        continue
                    v = value(arms, [(p, 1)], al)
                    if v is not None and (best is None or v < best[0]):
                        best = (v, p)
                cells.append(f"{best[0]:.2f} ({best[1]})" if best else "–")
            out.append(f"| **{lab}** | " + " | ".join(cells) + " |")
    out += ["", "## Every arm, ms (median of interleaved repetitions) and over plain P V", "",
            "| s | arm | ms | / plain | MHz | gate |", "|---|---|---|---|---|---|"]
    for j in rows:
        arms = j["arms"]
        for a in sorted(arms):
            out.append(f"| {j['s']} | {a} | {arms[a]['ms']:.4f} | {arms[a]['ms'] / arms['plain']['ms']:.3f} | {arms[a]['mhz']:.0f} | "
                       f"{verdict[a]} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
