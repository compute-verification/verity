"""The cheapest bit-exact receipt fold at each device's certificate, sm_120 (fx1), the H200 (li) and the B200
(li heavy), and whether `FoldOpt`'s stated cost holds there.

`fold_routes.py` takes the six candidate routes one and two instructions deep. What decides `FoldOpt` is how a
certificate prices the pipes. Every certificate but one prices an ALU, an IMAD, an FFMA and a SHFL instruction alike on
its binding rows: DS-120 and DS-90 on DS_issue, VF3r-16 on JI3s_0, DC-90-ri on JI90_0 and JI90_1, DC-100-ri on issue.
There a cheaper fold needs fewer instructions, and none has them (`fold_routes.py`'s superoptimizer, `fold_opt_fx.py`).

The exception is the B200's sheet. Its cheater certificate is π* = DS_issue 0.0246 + DS_alu 0.9754, so a LOP3 costs
1.975 and an IMAD 0.0246, and any ALU work with an IMAD-pipe program of under 80 instructions prices under its LOP3.
The keyed rounds' XOR has one. For a constant K,

    y ⊕ K = (1 − 2K₀)·y + K + Σ_{b ∈ B} τ_b·2^{b+1}·(y >> b)   (mod 2^32),

where B is the set of bit positions 1..30 at which K's bit changes, τ_b = −1 where a run of ones starts and +1 where one
ends (bit 31's term is 0 mod 2^32). y >> b is IMAD.HI(y, 2^{32−b}, 0), so a keyed round (y ⊕ K)·a + c is

    acc = IMAD(y, (1 − 2K₀)·a, K·a + c);   for b in B:  acc = IMAD(IMAD.HI(y, 2^{32−b}, 0), τ_b·a·2^{b+1}, acc)

That is 1 + 2|B| IMAD-pipe instructions and no ALU one: 25 to 41 at li's keys (LK), 61 for any key with every
coefficient in a register (the key-oblivious form). The XORs of two words (li's own, the checkpoint's) have no such
program that pays: the bit-serial one is 6 instructions a bit.

At π* the IMAD-pipe round costs 25 × 0.0246 = 0.61 against the LOP3 round's 2.0, so `FoldOpt` as instantiated (the
design's single `[fold]` mix) is false at the B200 sheet's certificate. The cheater LP, re-solved with that route open
(`eps.fastest` with the fold blended at the best fraction x of keyed rounds and the uniform route kept on the honest
fold), moves well under 1% of the rounds, since issue binds again beside the ALU at once. The headline's rise depends
on the keyed rounds' keys, which the design doesn't fix: li's own keys (LK) take it from 0.472% to 0.490%, a fresh
warp-uniform key a round (the cheater takes the rounds with the fewest bit changes) to 0.515%, and a per-lane key
(the key-oblivious 61) to 0.473%. These rest on 1 + 2|B| being the least IMAD-pipe program, which is not shown. What is
shown: the fold's ALU work is what holds the sheet's cheater above its issue floor (the honest fold has the least
instruction count, `fold_routes.py`), so no pipe move reaches past that floor, 0.837%. The measured reading
(DC-100-ri, π* issue) and every other device are unchanged.

Run: `python fold_min.py [--mutations] [--verdict PATH]`. Stdlib, exact_kit, `segment`, `fold_routes` (the reference
steps) and the machine model.
"""
from __future__ import annotations

import random
import sys
from dataclasses import replace
from math import comb

from exact_kit.checks import Checks
from verity_machine_model import eps as E
from verity_machine_model.model import device

from fold_routes import keyed_round, u32
from verity.ml.protocols.pouw.circuit import segment as SG

SEED = 0xF01D
Q = 0.5                  # the sheet reading's q*
KEYED = 0.375            # li heavy's keyed rounds a word (3/8)
HEADLINE = {"sm120": ("dither-c-peel-fx1", ("DS-120", "VF3r-16")),
            "sm90": ("dither-2s-licpw-redux-l32", ("DS-90", "DC-90-ri")),
            "sm100": ("dither-2s-licpw-redux-heavy-l32", ("DS-100", "DC-100-ri"))}
LANE_CLASSES = ("ALU", "IMAD", "FFMA", "SHFL")


# ---- the IMAD-pipe programs ------------------------------------------------------------------------------------------

def imad(a: int, b: int, c: int) -> int:
    return u32(a * b + c)


def imad_hi(a: int, b: int, c: int) -> int:
    return u32((u32(a) * u32(b) >> 32) + c)


def boundaries(K: int) -> list[int]:
    """The bit positions b in 1..30 at which K's bit differs from bit b − 1 (bit 31's term vanishes mod 2^32)."""
    return [b for b in range(1, 31) if (K >> b & 1) != (K >> (b - 1) & 1)]


def keyed_program(K: int, a: int, c: int, oblivious: bool = False) -> list[tuple]:
    """(y ⊕ K)·a + c as IMAD-pipe instructions: ("IMAD", src, mul, add) and ("IMAD.HI", src, mul, add), src "y" or
    "s" (the last IMAD.HI's value), add "acc" or a constant. `oblivious`: every b in 1..30, a zero coefficient where K's
    bit doesn't change, so one program shape serves every key."""
    k0 = K & 1
    prog = [("IMAD", "y", u32(a * (1 - 2 * k0)), u32(K * a + c))]
    for b in range(1, 31):
        change = (K >> b & 1) != (K >> (b - 1) & 1)
        if not change and not oblivious:
            continue
        tau = (-1 if K >> b & 1 else 1) if change else 0
        prog += [("IMAD.HI", "y", 1 << (32 - b), 0), ("IMAD", "s", u32(tau * a << (b + 1)), "acc")]
    return prog


def run(prog: list[tuple], y: int) -> int:
    acc = s = 0
    for op, src, mul, add in prog:
        x = y if src == "y" else s
        c = acc if add == "acc" else add
        if op == "IMAD.HI":
            s = imad_hi(x, mul, c)
        else:
            acc = imad(x, mul, c)
    return acc


def xor2_program(x: int, w: int) -> tuple[int, int]:
    """x ⊕ w = x + w − 2·Σ_b 2^b·x_b·w_b on the IMAD pipe, bit-serially: x_b = IMAD.HI(IMAD(x, 2^{31−b}, 0), 2, 0),
    w_b likewise, p = IMAD(x_b, w_b, 0), acc = IMAD(p, −2^{b+1}, acc). Returns (value, instructions)."""
    acc, n = imad(x, 1, w), 1
    for b in range(32):
        xb = imad_hi(imad(x, 1 << (31 - b), 0), 2, 0)
        wb = imad_hi(imad(w, 1 << (31 - b), 0), 2, 0)
        p = imad(xb, wb, 0)
        acc = imad(p, u32(-(1 << (b + 1))), acc)
        n += 6
    return acc, n


# ---- the certificates and the B200 re-solve --------------------------------------------------------------------------

def certificate(dev: str, vector: str):
    data = device(dev)
    d = E.design(HEADLINE[dev][0], data)
    pi = E.machine(vector, data)
    s = E.setup(d, pi, data, "free", "floor")
    q = Q if vector.startswith("DS") else 1.22
    return data, d, pi, s, s.cheat(q)


def class_price(pi, lam: dict[str, float], k: str) -> float:
    return sum(w * pi.c[k].get(r, 0.0) for r, w in lam.items())


def blend(d, moves: list[tuple[float, int]], split: bool = False):
    """li heavy with the fraction x of its keyed rounds on the k-instruction IMAD-pipe route, for each (x, k): each such
    round's LOP3 + IMAD becomes k IMAD-pipe instructions ((k − 1)/2 of them IMAD.HI, class IMADHI when `split`)."""
    f = dict(d.fold)
    for x, k in moves:
        f["ALU"] -= KEYED * x
        if split:
            f["IMAD"] += KEYED * x * ((k + 1) / 2 - 1)
            f["IMADHI"] = f.get("IMADHI", 0.0) + KEYED * x * (k - 1) / 2
        else:
            f["IMAD"] += KEYED * x * (k - 1)
    return replace(d, name=f"{d.name}+imadxor", fold=f)


def resolve(s, menu, split: bool = False, luck: bool = False, hi: float = 0.02, steps: int = 48):
    """The cheater's least T over the share X of keyed rounds moved, filled cheapest key first from `menu` (k, share
    of rounds), the uniform route kept on the honest fold; `luck`, the lucky cheater's (Theorem S's shortfall open).
    Returns (X, the Mix)."""
    d = s.design

    def moves(X: float) -> list[tuple[float, int]]:
        out, left = [], X
        for k, share in menu:
            take = min(left, share)
            if take > 0:
                out.append((take, k))
            left -= take
        return out

    def at(X: float):
        dd = blend(d, moves(X), split)
        qu = Q + d.fold_cost() - dd.fold_cost()
        return replace(s, design=dd).cheat(Q, q_uniform=qu, luck=luck)

    lo, top = 0.0, min(hi, sum(sh for _, sh in menu))
    for _ in range(steps):
        m1, m2 = lo + (top - lo) / 3, top - (top - lo) / 3
        if at(m1).T <= at(m2).T:
            top = m2
        else:
            lo = m1
    X = (lo + top) / 2
    return X, at(X)


def main() -> int:
    ck = Checks()
    rng = random.Random(SEED)

    # -- 1. the IMAD-pipe keyed round is bit-exact -----------------------------------------------------------------------
    ok = True
    for _ in range(2000):
        K, a, c = rng.getrandbits(32), rng.getrandbits(32) | 1, rng.getrandbits(32)
        p, po = keyed_program(K, a, c), keyed_program(K, a, c, oblivious=True)
        for _ in range(16):
            y = rng.getrandbits(32)
            ok &= run(p, y) == run(po, y) == keyed_round(y, K, a, c)
    edge = (0, 0xFFFFFFFF, 1, 1 << 31, 0x80000001, 0x7FFFFFFE, 0x55555555, 0xAAAAAAAA, 0x0000FFFF, 0xFFFF0000)
    for K in edge:
        p = keyed_program(K, 0x9E3779B1, 0x7F4A7C15)
        ok &= all(run(p, y) == keyed_round(y, K, 0x9E3779B1, 0x7F4A7C15)
                  for y in (0, 0xFFFFFFFF, 1 << 31, 1, *(rng.getrandbits(32) for _ in range(64))))
    ck.check("the IMAD-pipe keyed round (1 IMAD, then IMAD.HI + IMAD a bit change of K in 1..30) is (y ^ K) a + c, and "
             "so is its key-oblivious form (2000 random keys x 16 states, and 10 edge keys)", ok)
    lk = {(ch, h): len(keyed_program(SG.LK[ch][h], SG.LA[ch][h], 0)) for ch in range(SG.NCH) for h in range(2)}
    ok_lk = all(run(keyed_program(SG.LK[ch][h], SG.LA[ch][h], c), y) == keyed_round(y, SG.LK[ch][h], SG.LA[ch][h], c)
                for (ch, h) in lk for y, c in ((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(200)))
    kmin = min(lk.values())
    ck.check("at li's own keys (segment.LK, a keyed round being li's half step at the pair (0, c)) it is bit-exact, "
             "25 to 41 IMAD-pipe instructions", ok_lk and kmin == 25 and max(lk.values()) == 41,
             f"k = 1 + 2|B| by (chain, half): {lk}")
    ck.check("its count is 1 + 2|B| (B: K's bit changes in 1..30), 61 for the key-oblivious form, and no ALU "
             "instruction", all(len(keyed_program(K, 3, 0)) == 1 + 2 * len(boundaries(K))
                                and len(keyed_program(K, 3, 0, oblivious=True)) == 61
                                and {op for op, *_ in keyed_program(K, 3, 0)} <= {"IMAD", "IMAD.HI"} for K in edge))
    ok2, n2 = True, 0
    for _ in range(3000):
        x, w = rng.getrandbits(32), rng.getrandbits(32)
        v, n2 = xor2_program(x, w)
        ok2 &= v == x ^ w
    ck.check("an XOR of two words (li's y ^ d0 ^ K, the checkpoint's) has a bit-serial IMAD-pipe program, 6 a bit",
             ok2 and n2 == 193, f"{n2} instructions")

    # -- 2. the certificates: only DS-100 prices the pipes apart ----------------------------------------------------------
    cert = {}
    for dev, (_, vectors) in HEADLINE.items():
        for v in vectors:
            data, d, pi, s, c = certificate(dev, v)
            prices = {k: class_price(pi, c.pi, k) for k in LANE_CLASSES if k in pi.c}
            cert[(dev, v)] = (pi, s, c, prices)
    alike = [(dev, v) for (dev, v), (_, _, _, pr) in cert.items() if max(pr.values()) - min(pr.values()) < 1e-6]
    ck.check("at every certificate but the B200 sheet's (DS-120, VF3r-16, DS-90, DC-90-ri, DC-100-ri) every lane "
             "class prices alike, so a pipe move can't undercut and a cheaper fold needs fewer instructions",
             sorted(alike) == sorted(k for k in cert if k != ("sm100", "DS-100")),
             {f"{dev} {v}": {k: round(x, 4) for k, x in pr.items()} for (dev, v), (_, _, _, pr) in cert.items()})
    pi100, s100, c100, pr100 = cert[("sm100", "DS-100")]
    ck.check("DS-100's cheater certificate is DS_issue 0.0246 + DS_alu 0.9754: an ALU instruction 1.975, an IMAD 0.0246",
             abs(pr100["ALU"] - 1.97543) < 1e-4 and abs(pr100["IMAD"] - 0.02457) < 1e-4, pr100)
    honest_round = pr100["ALU"] + pr100["IMAD"]
    imad_round = kmin * pr100["IMAD"]
    ck.check("at π* the cheapest LK key's IMAD-pipe round (25 IMAD) undercuts the LOP3 + IMAD round, and so does every "
             "constant key (61 at most): FOLD-OPT as instantiated is false at the B200 sheet's certificate",
             imad_round < honest_round and 61 * pr100["IMAD"] < honest_round,
             f"{imad_round:.4f} and {61 * pr100['IMAD']:.4f} against {honest_round:.4f}; break-even at "
             f"{honest_round / pr100['IMAD']:.1f} instructions")
    ck.check("an XOR of two words doesn't pay there: 193 IMAD-pipe instructions price over one LOP3",
             n2 * pr100["IMAD"] > pr100["ALU"], f"{n2 * pr100['IMAD']:.3f} against {pr100['ALU']:.3f}")

    # -- 3. the B200 sheet re-solved with the route open -------------------------------------------------------------------
    T_base = s100.baseline.T
    data100 = device("sm100")
    e0 = E.eps(s100.design, pi100, Q, data100, operands="free", credit="floor")
    rest = e0.skip + e0.mand
    menus = {
        "LK's cheapest key (k 25)": [(kmin, 1 / 16)],
        "fresh uniform keys a round, cheapest first": [(1 + 2 * b, comb(30, b) / 2 ** 30) for b in range(31)],
        "any constant key, key-oblivious (k 61)": [(61, 1.0)],
    }
    rows = {}
    for name, menu in menus.items():
        X, mix = resolve(s100, menu)
        _, lucky = resolve(s100, menu, luck=True)
        rows[name] = (X, 1 - mix.T / T_base, max(0.0, (mix.T - lucky.T) / T_base), mix)
    pi_hi = replace(pi100, c={**pi100.c, "IMADHI": {"DS_issue": 1.0, "DS_imad": 4.0}})
    X4, mix4 = resolve(replace(s100, machine=pi_hi), [(kmin, 1 / 16)], split=True)
    T4 = mix4.T
    lines = {n: f"x {X:.5f} of keyed rounds, ε_machine {m:.4%}, ε_luck {lu:.4%}, headline {rest + m + lu:.3%}"
             for n, (X, m, lu, _) in rows.items()}
    m25 = rows["LK's cheapest key (k 25)"][1]
    ck.check("re-solved, the cheater moves a fraction of the keyed rounds and ε_machine rises above the design's",
             all(m > e0.machine_eps for _, m, _, _ in rows.values()), {"design": f"ε_machine {e0.machine_eps:.4%}, "
                                                                     f"headline {e0.core():.3%}", **lines})
    ck.check("the rise is small, under 0.03 pp at LK's cheapest key: issue binds beside the ALU at once",
             0 < m25 - e0.machine_eps < 3e-4 and rows["LK's cheapest key (k 25)"][3].binding == ("DS_issue", "DS_alu"),
             f"+{(m25 - e0.machine_eps) * 100:.4f} pp")
    ck.check("pricing IMAD.HI at 4 on DS_imad (measured at 4.19 SMSP-cycles) leaves it unchanged: DS_imad has slack",
             abs((1 - T4 / T_base) - m25) < 1e-9, f"x {X4:.5f}, ε_machine {1 - T4 / T_base:.4%}")
    ck.check("the credited floor program ♭ is the honest fold's (the baseline binds DS_issue alone), so the bound moves "
             "only through the cheater", s100.baseline.pi == {"DS_issue": 1.0}, s100.baseline.pi)
    count_only = replace(s100, design=replace(s100.design, fold={"FFMA": s100.design.fold_cost()})).cheat(Q)
    cap = 1 - count_only.T / T_base
    _, s_ri, c_ri, _ = cert[("sm100", "DC-100-ri")]
    ck.check("the cap: with the fold priced at its instruction count alone (any pipe; DS-100's FFMA is issue only), the "
             "cheater is at its issue floor, the most any pipe move can reach, over every key model's re-solve",
             count_only.binding == ("DS_issue",) and all(m < cap for _, m, _, _ in rows.values()),
             f"ε_machine {cap:.4%}, headline {rest + cap:.3%}; the measured reading's ε_machine "
             f"{1 - c_ri.T / s_ri.baseline.T:.4%}")
    ck.check("the B200 measured reading (DC-100-ri, π* issue) is unchanged: every IMAD-pipe round has more instructions",
             c_ri.pi == {"issue": 1.0} and kmin > 2)

    print("\nB200 sheet (DS-100), the cheater with the IMAD-pipe keyed round open:")
    print(f"  design: ε_machine {e0.machine_eps:.4%}, headline {e0.core():.3%}")
    for n, line in lines.items():
        print(f"  {n}: {line}")
    print(f"  cap, the fold at its instruction count on any pipe: ε_machine {cap:.4%}, headline {rest + cap:.3%}")

    if ck.mutations:
        p = keyed_program(0x0F0F0F0F, 5, 7)
        bad = [(op, src, u32(mul + (1 << 20)) if i == 3 else mul, add) for i, (op, src, mul, add) in enumerate(p)]
        ck.mutation("a coefficient off by 2^20 breaks the keyed program (the bit-exact check isn't vacuous)",
                    any(run(bad, y) != keyed_round(y, 0x0F0F0F0F, 5, 7) for y in (rng.getrandbits(32) for _ in range(64))))
        flipped = [(op, src, u32(-mul) if op == "IMAD" and src == "s" else mul, add)
                   for op, src, mul, add in keyed_program(0x0F0F0F0F, 1, 0)]
        ck.mutation("the run signs τ are load-bearing (flipping them breaks it)",
                    any(run(flipped, y) != y ^ 0x0F0F0F0F for y in (rng.getrandbits(32) for _ in range(64))))
        X_over, mix_over = resolve(s100, [(400, 1.0)], steps=24)
        ck.mutation("the re-solve finds no gain in a route over break-even (400 instructions a round)",
                    X_over < 1e-5 and mix_over.T >= c100.T * (1 - 1e-12),
                    f"x {X_over:.2e}, T {mix_over.T:.10f} against {c100.T:.10f}")
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
