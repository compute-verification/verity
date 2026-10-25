"""REDTEAM phase 19, target 3 (#3987 LEAK-LEAN): the program's port of the whole-window charge
(`NetworkCertifier.WindowCharge`, proved equal to the Security definitions by `CodeChargesWhole`) against
`window_charge.reference_window` on the corners the PR's 4,000 random instances don't reach, run through the PR's own
`certifier-window-difftest` exe; and `ChargeCount_v2`'s circuit against the reference on the same corners at small statics.

    PYTHONPATH=<worktree> python rt_charge.py <worktree> OUT

Families (each an instance in the difftest's form, compared by the exe):
  A  counts at the circuit's statics cap (N, H, O up to 2^16 - 1, every count's top bin and the bin edges 2^k - 1, 2^k,
     2^k + 1) and past it (2^17 + 3, two uniform steps; the Lean statement is over every N)
  B  the finish: windows whose charge lands on 2^20 - 3 … 2^20 + 1, at a step and at the finish
  C  the carry's last order entry: pos at 2^32 - 2, 2^32 - 1 and past 2^32; rank 0 and INDEX_CAP + 1 against r 0 and
     INDEX_CAP (the zigzag's two ends, ZIG_CAP); chains that push pos to the top
  D  empty windows: every (N, H, O) in {0, 1, 5}^3 with no slot holding a value, under steep, certain, random books and
     books with a bad table where a window reads it and where it doesn't
  E  the circuit (`evaluate_windows`) against the reference at {1,1,1}, {2,2,2}: disabled slots holding values past
     every cap, the carry's extremes and empty windows
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT / "verity/core/protocols/certifier/lean/scripts"))

import window_vectors as V  # noqa: E402
from verity.core.protocols.certifier import arith_encode as AE  # noqa: E402
from verity.core.protocols.certifier import window_charge as WC  # noqa: E402
from verity.core.protocols.certifier.arith_encode import MAX_TOTAL, RUNS  # noqa: E402

EXE = ROOT / "verity/core/protocols/certifier/lean/.lake/build/bin/certifier-window-difftest"
HIT_CAP, GAP_CAP, INDEX_CAP, ZIG_CAP = WC.HIT_CAP, WC.GAP_CAP, WC.INDEX_CAP, WC.ZIG_CAP
CW = AE.CW


def book(rng, N, H, O, kind="random", bad=None):
    bins = WC._bins(N, H, O)
    return WC.WindowBook(tuple(V._table(rng, b, "bad" if i == bad else kind) for i, (_n, b) in enumerate(bins)))


def steep_book(N, H, O):
    """Every bin frequency 1 at total MAX_TOTAL: each bin symbol takes 16 scalings."""
    return WC.WindowBook(tuple((tuple([1] * b), MAX_TOTAL) for _n, b in WC._bins(N, H, O)))


def slots(N, H, O, en=(), bit=(), hen=(), short=(), oen=(), p=(), r=()):
    def fill(v, n, d):
        v = list(v)
        return v + [d] * (n - len(v))
    return (fill(en, N, 0), fill(bit, N, 0), fill(hen, H, 0), fill(short, H, 0), fill(oen, O, 0), fill(p, O, 0),
            fill(r, O, 0))


def inst(N, H, O, bk, carry, windows, priors=None):
    return {"N": N, "H": H, "O": O, "book": bk, "priors": priors or carry.counts, "carry": carry,
            "windows": list(windows)}


def first(priors=((0, 0),) * RUNS):
    return WC.WindowCarry(tuple(priors))


# --- A: counts at and past the cap ------------------------------------------------------------------------------------

def family_a(rng):
    out = []
    edges = sorted({e for k in range(1, 17) for e in ((1 << k) - 1, 1 << k, (1 << k) + 1) if e < 1 << 16} | {65535})
    for S in edges:
        for n in sorted({S, max(S - 1, 0), (1 << (S.bit_length() - 1)), 0}):
            bk = book(rng, S, 0, 0)
            bit = [rng.randrange(2) for _ in range(S)]
            out.append(("A-N", inst(S, 0, 0, bk, first(), [slots(S, 0, 0, en=[1] * n, bit=bit)])))
    for S in [1, 2, 3, 4, 7, 8, 9, 255, 256, 257, 4095, 4096, 32767, 32768, 65535]:
        for n in sorted({S, max(S - 1, 0), 1 << (S.bit_length() - 1)}):
            bk = book(rng, 0, S, 0)
            vals = [rng.choice([0, 0, 1, 2, rng.randrange(64)]) if rng.random() < 0.98 else
                    rng.choice([rng.randrange(1 << 16), HIT_CAP]) for _ in range(S)]
            out.append(("A-H", inst(0, S, 0, bk, first(), [slots(0, S, 0, hen=[1] * n, short=vals)])))
            bk = book(rng, 0, 0, S)
            p, r, pos, rank = [], [], 0, 0
            for _ in range(S):
                pos += rng.choice([0, 1, 7])
                p.append(pos)
                r.append(rank if rng.random() < 0.95 else rng.choice([0, rng.randrange(1 << 24), INDEX_CAP]))
                pos, rank = pos + 1, r[-1] + 1
            out.append(("A-O", inst(0, 0, S, bk, first(), [slots(0, 0, S, oen=[1] * n, p=p, r=r)])))
    for S in [(1 << 17) + 3, (1 << 17) - 1, 1 << 17]:
        bk = book(rng, S, 0, 0)
        out.append(("A-N-past-statics", inst(S, 0, 0, bk, first(), [slots(S, 0, 0, en=[1] * S,
                                                                            bit=[rng.randrange(2) for _ in range(S)])])))
    return out


# --- B: the finish at 2^20 --------------------------------------------------------------------------------------------

def _pre_finish_bits(N, H, O, bk, carry, s):
    seen = {}
    real = AE.reference_count_finish

    def spy(cw, st, c):
        seen["bits"], seen["used"], seen["ok"] = st.bits, st.used, st.ok
        return real(cw, st, c)
    WC.AE.reference_count_finish = spy
    try:
        ok, length, _ = WC.reference_window(N, H, O, bk, carry, *s)
    finally:
        WC.AE.reference_count_finish = real
    return ok, length, seen


def family_b(rng):
    """H = 65535 hit slots under a steep hit table (each hit at v = 0, bin 0, about 16 bits) and up to 63 decisions
    at 1 : 1 priors as the fine step, calibrated then scanned to land the charge at 2^20 - 3 … 2^20 + 1, at the finish
    (the steps accepted) and at a step (a step's scalings past 2^20)."""
    out, H, N = [], 65535, 63
    bk = steep_book(N, H, 0)
    tables = list(bk.tables)
    tables[WC.SYMBOLS.index("n_arrived")] = ((1,) * len(tables[0][0]), len(tables[0][0]))
    bk = WC.WindowBook(tuple(tables))

    def run(k, d):
        s = slots(N, H, 0, en=[1] * d, bit=[int((i * 7) % 3 == 0) for i in range(d)], hen=[1] * k)
        _ok, _length, seen = _pre_finish_bits(N, H, 0, bk, first(), s)
        return s, seen["bits"] + 2 * seen["used"], seen["ok"]

    _s, total, ok = run(H - 200, 0)
    assert ok, "calibrate below the boundary"
    k0 = H - 200 + ((1 << CW) - 40 - total) // 16
    found = {}
    for k in range(k0, min(k0 + 8, H) + 1):
        for d in range(0, N + 1):
            s, total, ok = run(k, d)
            off = total - (1 << CW)
            tag = ("finish" if ok else "step", off if ok else 0)
            if -3 <= off <= 1 and ok and tag not in found:
                found[tag] = (k, d)
                out.append((f"B-finish{off:+d}", inst(N, H, 0, bk, first(), [s])))
            elif not ok and tag not in found:
                found[tag] = (k, d)
                out.append(("B-step-refused", inst(N, H, 0, bk, first(), [s])))
            if len(found) >= 6:
                return out, found
    return out, found


# --- C: the carry's last order entry -----------------------------------------------------------------------------------

def family_c(rng):
    out = []
    N, H, O = 1, 1, 3
    for pos in [(1 << 32) - 2, (1 << 32) - 1, 1 << 32, (1 << 40) + 5]:
        for rank in [0, 1, INDEX_CAP, INDEX_CAP + 1]:
            carry = WC.WindowCarry(((0, 0),) * RUNS, 0, pos, rank)
            for p in [pos, pos + 1, (1 << 32) - 2, (1 << 32) - 1]:
                for r in [0, INDEX_CAP, rank if rank <= INDEX_CAP else INDEX_CAP]:
                    bk = book(rng, N, H, O)
                    out.append(("C-carry", inst(N, H, O, bk, carry, [slots(N, H, O, oen=[1], p=[p], r=[r]),
                                                                      slots(N, H, O),
                                                                      slots(N, H, O, oen=[1], p=[p + 1], r=[0])])))
    for rank in [0, INDEX_CAP + 1]:
        for r in [0, INDEX_CAP]:
            carry = WC.WindowCarry(((0, 0),) * RUNS, 0, 5, rank)
            bk = book(rng, N, H, O)
            out.append(("C-zigzag-end", inst(N, H, O, bk, carry, [slots(N, H, O, oen=[1, 1, 1], p=[5, 6, 7],
                                                                         r=[r, 0, INDEX_CAP])])))
    for _ in range(40):
        bk = book(rng, N, H, O)
        start = WC.WindowCarry(((0, 0),) * RUNS, 0, (1 << 32) - 2 - 5 * O * GAP_CAP // 2, INDEX_CAP + 1)
        ws, pos = [], start.pos
        for w in range(5):
            p, r = [], []
            for _e in range(O):
                pos = min(pos + rng.choice([GAP_CAP, GAP_CAP, 0]), (1 << 32) - 2)
                p.append(pos)
                r.append(rng.choice([0, INDEX_CAP]))
                pos += 1
            ws.append(slots(N, H, O, oen=[1] * O, p=p, r=r))
        out.append(("C-chain-top", inst(N, H, O, bk, start, ws)))
    return out


# --- D: empty windows --------------------------------------------------------------------------------------------------

def family_d(rng):
    out = []
    for N in (0, 1, 5):
        for H in (0, 1, 5):
            for O in (0, 1, 5):
                for kind in ("steep", "certain", "random"):
                    bk = book(rng, N, H, O, kind)
                    out.append((f"D-{kind}", inst(N, H, O, bk, first(), [slots(N, H, O)] * 3)))
                for bad in range(8):
                    bk = book(rng, N, H, O, "random", bad)
                    out.append(("D-bad", inst(N, H, O, bk, first(), [slots(N, H, O)] * 2)))
                for run in range(4):
                    carry = WC.WindowCarry(((0, 0),) * RUNS, run, 0, INDEX_CAP + (run % 3))
                    out.append(("D-carry", inst(N, H, O, book(rng, N, H, O), carry, [slots(N, H, O)])))
    return out


# --- run the exe -------------------------------------------------------------------------------------------------------

def run_exe(cases):
    d = OUT / "vectors"
    d.mkdir(parents=True, exist_ok=True)
    fi, fp = d / "instances.jsonl", d / "python.jsonl"
    results = []
    with open(fi, "w") as a, open(fp, "w") as b:
        for _tag, x in cases:
            results.append(V.python_results(x))
            a.write(V.line(V.lean_input(x)))
            b.write(V.line(results[-1]))
    res = subprocess.run([str(EXE), str(fi), str(fp), str(len(cases))], capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip(), results


def tally(cases, results):
    acc = {}
    for (tag, _x), r in zip(cases, results):
        a = acc.setdefault(tag, [0, 0, 0])
        a[0] += 1
        a[1] += sum(1 for s in r["steps"] if s is not None)
        a[2] += sum(1 for s in r["steps"] if s is None)
    return acc


# --- E: the circuit against the reference -------------------------------------------------------------------------------

def family_e(rng):
    """(statics, case) pairs for `evaluate_windows`, each (book, carry, *slots) with disabled slots past every cap."""
    out = []
    for (N, H, O) in [(1, 1, 1), (2, 2, 2)]:
        for _ in range(60):
            bk = book(rng, N, H, O, rng.choice(["random", "steep", "certain"]))
            pos = rng.choice([0, 5, (1 << 32) - 2, (1 << 32) - 1])
            rank = rng.choice([0, INDEX_CAP, INDEX_CAP + 1])
            counts = tuple((min(a, (1 << 15) - 1), min(b, (1 << 15) - 1)) for a, b in V._counts(rng))
            carry = WC.WindowCarry(counts, rng.choice([0, 1, 2]), pos, rank)
            en = [rng.randrange(2) for _ in range(N)]
            hen = [rng.randrange(2) for _ in range(H)]
            oen = [rng.randrange(2) for _ in range(O)]
            short = [rng.choice([HIT_CAP + 1, (1 << 17) - 1, 3]) if not e else rng.choice([0, HIT_CAP])
                     for e in hen]
            p = [rng.choice([0, (1 << 32) - 1, pos - 1 if pos else 0]) if not e else
                 rng.choice([pos, min(pos + GAP_CAP, (1 << 32) - 2)]) for e in oen]
            r = [rng.choice([(1 << 25) - 1, INDEX_CAP + 1]) if not e else rng.choice([0, INDEX_CAP]) for e in oen]
            bit = [rng.randrange(2) for _ in range(N)]
            out.append(((N, H, O), (bk, carry, en, bit, hen, short, oen, p, r)))
    return out


def run_circuit(cases):
    by = {}
    for S, c in cases:
        by.setdefault(S, []).append(c)
    diffs, n, refused = [], 0, 0
    for (N, H, O), cs in by.items():
        got = WC.evaluate_windows(N, H, O, cs)
        for c, g in zip(cs, got):
            ok, length, nxt = WC.reference_window(N, H, O, *c)
            n += 1
            refused += not ok
            if g[0] != ok or (ok and (g[1], g[2]) != (length, nxt)):
                diffs.append(((N, H, O), c, g, (ok, length, nxt)))
    return n, refused, diffs


def main():
    rng = random.Random(19_3987)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    a, c, d = family_a(rng), family_c(rng), family_d(rng)
    b, found = family_b(rng)
    report["B_found"] = {f"{k[0]}{k[1]:+d}": v for k, v in found.items()}
    cases = a + b + c + d
    code, so, se, results = run_exe(cases)
    report["exe"] = {"returncode": code, "stdout": so, "stderr": se[:4000]}
    report["families"] = {k: {"instances": v[0], "windows accepted": v[1], "windows refused": v[2]}
                          for k, v in sorted(tally(cases, results).items())}
    n, refused, diffs = run_circuit(family_e(rng))
    report["circuit"] = {"cases": n, "refused": refused, "differences": len(diffs),
                         "first": [repr(x) for x in diffs[:3]]}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
