"""S14's packed slots against ε_machine's own lanes route, and premise (M) against F1.

The packed-slot model (`isa_escape_joint.py`, not in this tree) prices a packed slot, w committed words through one
slot's output, each extracted at e lane instructions, against the credited route alone: a word's load is
max(step/w, the MMA's issue share/w + (fold + e)/32) on the sheet's rows, and on sm_120 also VF3e's joint rows
a·MMA + b·lanes. It calls sm_120 open for decodes between A6's 1/2 and the break-even. ε_machine's LP
(`machine_model/verity_machine_model/eps.py`) also has the lanes route, q* lane instructions a word folded on lanes,
which is the model's `per_word` at w → ∞ with e = q*. A packed word's load is 1/w of a credited word's MMA part plus
(fold + e) lanes. One word made 1/w on the credited route and (1 − 1/w) on the lanes route loads 1/w of the MMA part plus
(fold + (1 − 1/w)·q*) lanes. So on every row linear in (MMAs, slots, lane instructions) the packed route is dominated
when e ≥ (1 − 1/w)·q*, and A6's e ≥ 1/2 gives that at q* = 1/2 for every w.

This checks the domination by exact LP on the model's rows (every device, fold, reading and q*), decodes a radix-64
two-row packing bit-exactly (a realization of the model's credited-row window on words of spread under 64), and puts
F1's measured sm_120 rates on the shallow forms the model grades "M unmeasured". The model's constants (`MMA_ISSUE`,
`JI3E`, the steps) are copied here. Stdlib only.
Run: `python benchmarks/pouw/rt2_fp/slot_pack.py [--json OUT]`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from fractions import Fraction as Q

STEP = {"sm_120": Q(1, 8), "sm_90a": Q(1, 32), "sm_100a": Q(1, 64)}
MMA_ISSUE = {"sm_120": Q(1, 128), "sm_90a": Q(4, 64 * 256), "sm_100a": Q(1, 128 * 256)}
JI3E = {"JI3e_0": (Q("8.115163161"), Q(1)), "JI3e_1": (Q("4510.17215313632") / Q("300.08"), Q("0.317063107")),
        "JI3e_2": (Q("4662.85212194032") / Q("300.08"), Q("0.171692378"))}
A6 = Q(1, 2)
QSTAR = {"A6, proved (q* 1/2)": Q(1, 2), "Theorem C (q* 1.22)": Q("1.22"), "I3 (q 2.03)": Q("2.03")}
FOLDS = {"0": Q(0), "1/2": Q(1, 2), "FOLD-OPT 3": Q(3), "7/2": Q(7, 2), "fx1 4": Q(4)}
WS = (Q(3, 2), Q(2), Q(3), Q(4), Q(8), Q(32))
ES = (Q(1, 2), Q(2, 3), Q(3, 4), Q(1), Q(5, 3), Q(2), Q(3))

# F1 (sm_120 at locked 2100 MHz): MACs an SM-clock, and a slot's physical
# positions (a 2:4 shape's half its logical k; F1 counts a sparse form's nonzero products: sp s8 k64 = dense s8 k32).
F1 = {
    ("s8", False, "m16n8k32"): ("s8_k32", "1022.6734254939101", 32),
    ("f16", False, "m16n8k8"): ("f16_k8", "255.65350107194774", 8),
    ("bf16", False, "m16n8k8"): ("bf16_k8", "255.651836936945", 8),
    ("f16", False, "m8n8k4"): ("f16_m8n8k4", "30.480977600602444", 4),
    ("e4m3", False, "m16n8k16"): ("fp8_e4m3_k16", "511.30709853713523", 16),
    ("e5m2", False, "m16n8k16"): ("fp8_e5m2_k16", "511.30350766204197", 16),
    ("f16", True, "m16n8k16"): ("sp_f16_k16", "255.65202004615134", 8),
    ("tf32", True, "m16n8k8"): ("sp_tf32_k8", "127.83088718135443", 4),
    ("f64", False, "m8n8k4"): ("f64_m8n8k4", "1.7533600873576562", 4),
    ("f64", False, "m16n8k4"): ("f64_k4", "1.7533219070750512", 4),
    ("f64", False, "m16n8k8"): ("f64_k8", "1.7533409626646774", 8),
    ("f64", False, "m16n8k16"): ("f64_k16", "1.753369211892809", 16),
}
UNMEASURED_2755 = [("bf16", False, "m16n8k8"), ("bf16", True, "m16n8k16"), ("e4m3", False, "m16n8k16"),
                   ("e5m2", False, "m16n8k16"), ("f16", False, "m16n8k8"), ("f16", False, "m8n8k4"),
                   ("f16", True, "m16n8k16"), ("tf32", True, "m16n8k8")]


def per_word(dev: str, w: Q | None, e: Q, f: Q, joint: bool) -> dict[str, Q]:
    """The model's `per_word`: a word's load on each row, SMSP-cycles; w None is the lanes route (w → ∞)."""
    mma = Q(0) if w is None else MMA_ISSUE[dev] / w
    out = {"tensor": Q(0) if w is None else STEP[dev] / w, "issue": mma + (f + e) / 32}
    if joint:
        out |= {name: a * mma + b * (f + e) / 32 for name, (a, b) in JI3E.items()}
    return out


def credited_gain(dev: str, w: Q, e: Q, f: Q, joint: bool) -> Q:
    """The model's `packing_gain`: the credited route's load a word over the packed route's."""
    return max(per_word(dev, Q(1), Q(0), f, joint).values()) / max(per_word(dev, w, e, f, joint).values())


def solve(a: list[list[Q]], b: list[Q]) -> list[Q] | None:
    n = len(a)
    m = [row[:] + [v] for row, v in zip(a, b)]
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                k = m[r][c] / m[c][c]
                m[r] = [x - k * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def words_a_cycle(routes: list[dict[str, Q]]) -> Q:
    """max Σ n_r subject to Σ_r n_r·load_r ≤ 1 on every row and n ≥ 0: the LP's words a SMSP-cycle, exactly, by
    enumerating its vertices."""
    rows = sorted(set().union(*routes))
    k = len(routes)
    cons = [([r.get(row, Q(0)) for r in routes], Q(1)) for row in rows]
    cons += [([Q(-1) if i == j else Q(0) for j in range(k)], Q(0)) for i in range(k)]
    best = Q(0)
    for idx in itertools.combinations(range(len(cons)), k):
        x = solve([cons[i][0] for i in idx], [cons[i][1] for i in idx])
        if x is not None and all(sum(c * v for c, v in zip(row, x)) <= rhs for row, rhs in cons):
            best = max(best, sum(x))
    return best


def domination() -> dict:
    """Every (device, reading, fold, q*, w, e ≥ A6): the packed route's gain inside the LP {credited, lanes at q*}."""
    cases, violations, gains = 0, [], []
    lanes_gain = {}
    for dev in STEP:
        for joint in ((False, True) if dev == "sm_120" else (False,)):
            reading = f"{dev} {'VF3e' if joint else 'sheet'}"
            for fname, f in FOLDS.items():
                cred = per_word(dev, Q(1), Q(0), f, joint)
                alone = words_a_cycle([cred])
                for qname, q in QSTAR.items():
                    lanes = per_word(dev, None, q, f, joint)
                    base = words_a_cycle([cred, lanes])
                    lanes_gain[f"{reading}, fold {fname}, {qname}"] = float(base / alone)
                    for w, e in itertools.product(WS, ES):
                        g = words_a_cycle([cred, lanes, per_word(dev, w, e, f, joint)]) / base
                        cases += 1
                        dominated = e >= (1 - 1 / w) * q
                        row = {"reading": reading, "fold": fname, "q": qname, "w": str(w), "e": str(e),
                               "gain": float(g), "dominated": dominated}
                        if dominated and g != 1:
                            violations.append(row)
                        if g > 1:
                            gains.append(row)
    proved = [r for r in gains if r["q"].startswith("A6")]
    return {"cases": cases, "dominated_but_gains": violations, "gains": gains, "gains_at_A6": proved,
            "every_gain_undominated": all(not r["dominated"] for r in gains), "lanes_route_gain": lanes_gain,
            "largest_gain": max((r["gain"] for r in gains), default=1.0)}


def two_row(rng: random.Random, trials: int, L: int = 32) -> dict:
    """Two rows of bits packed as A = b + 64·b' against a bit column: one IMMA slot s = y + 64·y', y, y' in [0, L],
    decoded by LOP3.LUT (s & 63) and SHF.R.U32.HI (s >> 6), one lane instruction a word at w = 2."""
    bad, ops, top = 0, set(), 0
    for t in range(trials):
        if t < 8:
            b1, b2, c = ([(t >> i) & 1] * L for i in range(3))
        else:
            b1, b2, c = ([rng.getrandbits(1) for _ in range(L)] for _ in range(3))
        a = [x + 64 * y for x, y in zip(b1, b2)]
        ops |= set(a)
        s = sum(p * q for p, q in zip(a, c))
        top = max(top, s)
        bad += (s & 63, s >> 6) != (sum(p * q for p, q in zip(b1, c)), sum(p * q for p, q in zip(b2, c)))
    return {"trials": trials, "L": L, "mismatches": bad, "a_operands": sorted(ops),
            "a_in_s8": all(-128 <= v <= 127 for v in ops), "slot_max": top, "words_spread": L + 1}


def f1_slots() -> dict:
    """F1's time a slot, 4·P/R SMSP-cycles, against sm_120's step 1/8, on the forms the model grades "M unmeasured"."""
    out = {}
    for key in UNMEASURED_2755 + [k for k in F1 if k[0] in ("s8", "f64")]:
        name = f"{key[0]}{'.sp' if key[1] else ''} {key[2]}"
        if key not in F1:
            out[name] = {"F1": None}
            continue
        f1, rate, pos = F1[key]
        t = 4 * pos / Q(rate)
        out[name] = {"F1": f1, "macs_sm_clk": float(Q(rate)), "slot": float(t), "steps_a_slot": float(t / STEP["sm_120"]),
                     "at_least_a_step": t >= STEP["sm_120"]}
    return out


def dmma_beside_imma() -> dict:
    """(M)'s sum on a run with IMMA.16832 saturated and DMMA.884 at its alone rate beside it, as a fraction of the
    run's cycles, if the two run concurrently: (M) charges IMMA a step a slot and DMMA 64 slots' steps."""
    imma = 4 * 32 / Q(F1[("s8", False, "m16n8k32")][1])
    dmma = 64 * 4 * 4 / Q(F1[("f64", False, "m8n8k4")][1])
    m_sum = STEP["sm_120"] / imma + 64 * STEP["sm_120"] / dmma
    return {"imma_slot": float(imma), "dmma_mma": float(dmma), "M_charge_dmma": float(64 * STEP["sm_120"]),
            "M_sum_over_cycles_if_concurrent": float(m_sum)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    out = {"domination": domination(), "two_row": two_row(random.Random(20261019), 3000)}
    d = out["domination"]
    print(f"domination: {d['cases']} LPs; dominated but gains: {len(d['dominated_but_gains'])}; gains > 1: "
          f"{len(d['gains'])} (all undominated: {d['every_gain_undominated']}), at q* = 1/2: {len(d['gains_at_A6'])}; "
          f"largest {d['largest_gain']:.4f}")
    for r in sorted(d["gains"], key=lambda r: -r["gain"])[:6]:
        print("  gain", r)
    for k in ("sm_120 sheet, fold FOLD-OPT 3", "sm_120 VF3e, fold FOLD-OPT 3", "sm_120 VF3e, fold fx1 4",
              "sm_90a sheet, fold FOLD-OPT 3", "sm_100a sheet, fold FOLD-OPT 3"):
        print(f"  lanes route over credited alone, {k}: " + ", ".join(
            f"{q.split(' (')[0]} {v:.4f}" for kk, v in d["lanes_route_gain"].items() if kk.startswith(k)
            for q in [kk.split(', ')[-1]]))
    witness = {}
    for (wname, w, e), (fname, f), joint in itertools.product(
            (("two-row radix 64", Q(2), Q(1)), ("two-row + one plain slot", Q(3, 2), Q(2, 3)),
             ("#2755 chained", Q(3, 2), Q(5, 3))), (("FOLD-OPT 3", Q(3)), ("7/2", Q(7, 2)), ("fx1 4", Q(4))),
            (False, True)):
        key = f"{wname}, fold {fname}, {'VF3e' if joint else 'sheet'}"
        witness[key] = float(credited_gain("sm_120", w, e, f, joint))
    out["credited_row_gains_sm_120"] = witness
    for k, v in witness.items():
        print(f"  over the credited row (#2755's packing_gain), sm_120, {k}: {v:.4f}")
    print("two-row decode:", out["two_row"])
    out["f1"] = f1_slots()
    for k, v in out["f1"].items():
        print(f"F1 sm_120 {k}: {v}")
    out["dmma"] = dmma_beside_imma()
    print("DMMA beside IMMA:", out["dmma"])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
