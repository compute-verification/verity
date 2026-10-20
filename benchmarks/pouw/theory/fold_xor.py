"""FOLD-XOR: the cheapest bit-exact IMAD- and FP-pipe program for z = x ⊕ y on the B200 sheet (DS-100), against the
narrowed FOLD-OPT premise the design of record states (note:20261019T0740Z-draft-rt-fold, FOLD-XOR).

The B200's design of record is `dither-2s-licpw-redux-ldtm0-l32` (#2706): plain li, whose half step is
y ← (y ⊕ d0 ⊕ k)·a + d1, so its LOP3 is an XOR of two data words. DS-100 prices a LOP3 on DS_issue and DS_alu and an
IMAD or FFMA on DS_issue alone at the cheater's certificate, so FOLD-OPT there narrows to a pipe premise. The design's
`[mand.fold]` states it as "li's XOR of two words has no IMAD- or FP-pipe program under 93 instructions (the bit-serial
one is 193)".

The menu. `facet_break.loads.class_of` routes IMAD and every form of it but .WIDE (.HI, .SHL, .MOV, .IADD, .X), IDP.4A
and IDP.2A to class IMAD (DS_issue 1, DS_imad 2), and FFMA, FADD, FMUL and the packed FFMA2, FADD2, FMUL2 to class FFMA
(DS_issue 1). IMAD.WIDE is IMADW and the HFMA2 family HFMA2, neither of which DS-100 has a row for, so neither is on
the menu and no program here uses them.

The program: 78 instructions, IMAD and IMAD.HI only (33 and 45), bit-exact, for x ⊕ y and for li's x ⊕ y ⊕ k at any
constant k. It works on 2-bit chunks. XOR mod 4 is the polynomial A + B − 2AB, so with A = x >> 2j and B = y >> 2j
(their high bits don't matter mod 4) t_j = 2^30·(A + B − 2AB) mod 2^32 is chunk j of z at the top of a word. The chunks
accumulate as Q ← (Q >> 2) + t_j, and Q = z after the 16th. Two rewritings save an instruction a chunk. With B' = B + 1
the IMAD.HI's addend makes S' = A + B + 1 in one instruction. Then E = 2^30 + 2^31·B (from B') gives
S'·E ≡ t_j + 2^30, and the 2^30 goes into the accumulator's shift, H = IMAD.HI(Q, 2^30, −2^30). A chunk is then B', S',
E, H and Q, 5 instructions, and chunk 0 is 3, which makes 3 + 15·5 = 78. XOR with a constant 2-bit κ is affine mod 4
(u ⊕ κ = σu + c with σ = ±1), so σ and c fold into E's and H's constants: k costs nothing.

The certificates (ldtm0 rebuilt from the heavy design, its fold li's and its LDTM floor 0, which reproduces #2706's
certificate bit for bit; checked against the real design with its `[fold_routes]` off when it is present. With them on,
the real design's certificate is DS_issue ⅔ + DS_alu ⅓, the Lean form's π*, where the break-even is 2 instructions):
- The plain cheater's π* is DS_issue 0.02135 + DS_alu 0.97865, so a LOP3 is 1.9786 and an IMAD 0.02135. The
  break-even is 92.66 instructions, and 78 is under it: the design's sentence is false.
- The lucky cheater's (Theorem S's shortfall open) is DS_issue 0.03147 + DS_alu 0.96853, break-even 62.54. INT-HB's
  "Lean form's π*" (0.0314 + 0.9686) has this structure. 78 is over it.

The headline. ε = skip + mand + ε_machine + ε_luck, and with ε_luck > 0 that's skip + mand + 1 − T_lucky/T₀: the lucky
cheater holds it. Re-solved with the route open (the cheater moves the share x of li's XORs that minimizes its T), the
plain cheater moves 0.017% of them and ε_machine rises from 0.0137% to 0.0163%. ε_luck falls by as much, the lucky
cheater doesn't move, and the headline stays 0.7052%. So the result changes a statement (the premise), not a headline ε
or a constant. Against the program's length n, the headline is 0.7052% for n ≥ 63, 0.7053% at 62, 0.716% at 40 and
0.746% at 20. It passes 1% only at n ≤ 3 (1.007%), and at n = 1 it is the count-only cap, 1.329%. So "under 1%" needs
only "no XOR program of 3 or fewer instructions", and 0.705% needs "none of 62 or fewer", which this script neither
finds nor rules out.

What is shown, and how:
- A construction: 78, with a bit-exact check over random words and edge cases, for x ⊕ y and x ⊕ y ⊕ k.
- The same 78 as a SASS loop body with its registers allocated (`sass_body`), checked on the host. SASS's IMAD.HI.U32
  adds a 64-bit register pair, hi32(a·b + {R[c+1]:R[c]}), so an IMAD.HI with addend c reads a pair {0, c}. The body
  holds those pairs, and its constants, in registers it never writes, and B' is written into the odd half of a pair
  whose even half stays 0, so it needs no move: 78 a XOR in steady state, every one class IMAD.
- On the device (`fold_xor.sh` on vy-dc-b200, r20261019-114028-7a16 and r20261019-114747-10bd): every kernel is
  bit-exact over 1,048,720 points, for k = 0 and one key, and ptxas's SASS has exactly the program's IMAD 33 and
  IMAD.HI.U32 45 a XOR. ptxas also rebuilds the pair {0, c} at every use: 47 moves a XOR unrolled (MOV 31.5, HFMA2
  15.5) and 50 plus 5 constant reloads in the loop. Neither MOV nor HFMA2 is on the menu, and as IMAD.MOV they'd make
  125, over 93. So the 78 needs the allocation above, which is checked on the host and was not run as hand-written SASS
  (no sm_100 assembler here).
- A proof: no program of only IMAD's low-word forms (polynomials mod 2^32) computes x ⊕ y at any length, because
  XOR mod 16 isn't a polynomial function (its Newton coefficient at (2, 4) is 8, which gcd(16, 2!·4!) = 16 doesn't
  divide). Every program needs an IMAD.HI, an IDP or an FP instruction. XOR mod 8 is a polynomial, so 3-bit chunks are
  the widest a low-word combine can take.
- A search, exhaustive in a restricted model (values mod 8 as unit or 2^29-scaled, 8 + 8 constants, the inputs offset
  as the extraction offsets them, the last instruction a scaled product plus the accumulator): no 3-bit combine has 2
  or 3 instructions, while the same search finds the 2-instruction 2-bit one. With a chunk costing 2 extractions, its
  combine c and a shift, 3-bit chunks total about 30 + 11c, so they'd need c ≤ 2 to get under 63. This rules out
  only that family. It is no bound on all programs.
- Not shown: any bound N ≥ 63, or N ≥ 4, over all programs. `fold_xor_search.py` searches all programs of k
  instructions on this menu (the forms `class_of` routes to IMAD and FFMA) for x ⊕ y ⊕ K, with every constant symbolic.

Run: `python fold_xor.py [--mutations] [--verdict PATH]`. Stdlib, numpy, exact_kit, `facet_break.loads` (the class
map) and the machine model. The device check is `fold_xor.sh` (fold_xor.cu on sm_100a), and
`python fold_xor.py --sass FILE` counts one XOR's SASS in its build by census class.
"""
from __future__ import annotations

import itertools
import json
import random
import re
import sys
from collections import Counter
from dataclasses import replace
from math import comb, factorial, gcd
from pathlib import Path

import numpy as np
from exact_kit.checks import Checks
from verity_machine_model import eps as E
from verity_machine_model.model import device

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facet_break.loads import class_of  # noqa: E402

SEED = 0xF0D5
M32 = (1 << 32) - 1
Q = 0.5                         # the sheet reading's q*
XOR_SHARE = 0.5                 # li's XOR LOP3s a word: two half steps a chain, a LOP3 and an IMAD each
HEAVY = "dither-2s-licpw-redux-heavy-l32"
LDTM0 = "dither-2s-licpw-redux-ldtm0-l32"
LI_FOLD = {"ALU": 0.53125, "IMAD": 0.5, "SHFL": 0.0234375}
#: ldtm0's [skip.served] at DS-100 (#2706, RT-HB-HEAD r20261019-083808-1f18); its mand is 0
LDTM0_SKIP_DS100 = 0.006849
SASS = {"IMAD": "IMAD", "IMAD.HI": "IMAD.HI.U32"}
#: u ⊕ κ on a 2-bit u is σ·u + c (mod 4)
AFFINE = {0: (1, 0), 1: (-1, 1), 2: (1, 2), 3: (-1, 3)}
EDGE = (0, 1, M32, M32 - 1, 1 << 31, (1 << 31) - 1, 0x55555555, 0xAAAAAAAA, 0x0F0F0F0F, 0xF0F0F0F0, 0x80000001,
        0x7FFFFFFE)


def u32(v: int) -> int:
    return v & M32


# ---- the program -----------------------------------------------------------------------------------------------------

def xor_program(K: int = 0) -> list[tuple]:
    """x ⊕ y ⊕ K as (form, dst, a, b, c): IMAD dst = a·b + c, IMAD.HI dst = hi32(a·b) + c, mod 2^32; an operand is a
    register name ("x", "y" or an earlier dst) or a constant. Q holds z at the end."""
    s, c = AFFINE[K & 3]
    prog = [("IMAD", "F", "y", u32(-s << 31), u32(s << 30)),
            ("IMAD", "G", "y", u32(s << 30), u32(c << 30)),
            ("IMAD", "Q", "x", "F", "G")]
    for j in range(1, 16):
        s, c = AFFINE[K >> 2 * j & 3]
        sh = 1 << (32 - 2 * j)
        prog += [("IMAD.HI", "Bp", "y", sh, 1),
                 ("IMAD.HI", "Sp", "x", sh, "Bp"),
                 ("IMAD", "E", "Bp", 1 << 31, u32(s * (3 << 30))),
                 ("IMAD.HI", "H", "Q", 1 << 30, u32((c - s) << 30)),
                 ("IMAD", "Q", "Sp", "E", "H")]
    return prog


def run(prog: list[tuple], x: int, y: int) -> int:
    r = {"x": x, "y": y}
    val = lambda o: r[o] if isinstance(o, str) else o
    for form, dst, a, b, c in prog:
        p = val(a) * val(b)
        r[dst] = u32((p >> 32 if form == "IMAD.HI" else p) + val(c))
    return r["Q"]


def bit_exact(prog: list[tuple], K: int, rng: random.Random, n: int) -> tuple[int, int]:
    pts = [(a, b) for a in EDGE for b in EDGE] + [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(n)]
    return len(pts), sum(run(prog, a, b) != a ^ b ^ K for a, b in pts)


#: the data registers a loop body rotates over, three XORs a body: s[n+2] = s[n] ^ s[n+1] ^ K with no move
DATA = (20, 21, 22)


def sass_body(K: int = 0) -> tuple[dict[int, int], list[tuple]]:
    """The 78 as a SASS loop body with its registers allocated: (invariants, body). The invariants are set once before
    the loop and the body writes none of them. An instruction is (op, d, a, b, c) on registers R0..R22: IMAD
    d = a·b + c, IMAD.HI.U32 d = hi32(a·b + {R[c+1]:R[c]}) with c an even register (SASS's 64-bit addend pair). b, or
    an IMAD's c, may be a constant ("c", v) (a constant bank or uniform register), never both."""
    C = lambda v: ("c", u32(v))
    s, c = AFFINE[K & 3]
    inv = {0: 0, 1: 1, 10: 0, 12: 1 << 31, 13: u32(s << 30), 14: u32(c << 30)}
    pairs, chunks = {}, []
    for j in range(1, 16):
        sj, cj = AFFINE[K >> 2 * j & 3]
        ha = u32((cj - sj) << 30)
        if ha not in pairs:
            pairs[ha] = 2 + 2 * len(pairs)
            inv |= {pairs[ha]: 0, pairs[ha] + 1: ha}
        chunks.append((1 << (32 - 2 * j), u32(sj * (3 << 30)), pairs[ha]))
    assert max(pairs.values()) <= 8

    def one(x, y, q):
        out = [("IMAD", 18, y, C(-s << 31), 13), ("IMAD", 19, y, C(s << 30), 14), ("IMAD", q, x, 18, 19)]
        for sh, ea, hp in chunks:
            out += [("IMAD.HI.U32", 11, y, C(sh), 0),      # B' into the odd half of {R10 = 0, R11}
                    ("IMAD.HI.U32", 15, x, C(sh), 10),     # S' = hi(x·sh) + B'
                    ("IMAD", 16, 11, 12, C(ea)),           # E
                    ("IMAD.HI.U32", 17, q, C(1 << 30), hp),  # H
                    ("IMAD", q, 15, 16, 17)]               # Q
        return out

    a, b, q = DATA
    return inv, one(a, b, q) + one(b, q, a) + one(q, a, b)


def run_sass(inv: dict[int, int], body: list[tuple], reps: int, x, y):
    """`reps` passes of `body` from R[DATA[0]], R[DATA[1]] = x, y (numpy uint64 arrays of words); returns R."""
    R = {r: np.full(x.shape, v, dtype=np.uint64) for r, v in inv.items()}
    R[DATA[0]], R[DATA[1]] = x.copy(), y.copy()
    val = lambda o: np.uint64(o[1]) if isinstance(o, tuple) else R[o]
    for _ in range(reps):
        for op, d, a, b, c in body:
            if op == "IMAD":
                R[d] = (R[a] * val(b) + val(c)) & np.uint64(M32)
            else:
                R[d] = (R[a] * val(b) + (R[c + 1] << np.uint64(32) | R[c])) >> np.uint64(32)
    return R


def sass_exact(K: int, rng: random.Random, n: int, reps: int = 3, inv: dict | None = None) -> tuple[int, int]:
    """`sass_body(K)` over `reps` passes (3·reps XORs in a chain) against the chain on words, edge words crossed and n
    random pairs; (points, mismatches)."""
    base, body = sass_body(K)
    pts = [(a, b) for a in EDGE for b in EDGE] + [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(n)]
    x, y = (np.array(v, dtype=np.uint64) for v in zip(*pts))
    R = run_sass(base if inv is None else inv, body, reps, x, y)
    seq = [x, y]
    for _ in range(3 * reps):
        seq.append(seq[-2] ^ seq[-1] ^ np.uint64(K))
    return len(pts), int(np.count_nonzero(R[DATA[1]] != seq[-1]))


def sass_legal(inv: dict[int, int], body: list[tuple]) -> list[str]:
    """What breaks SASS's operand rules or the hoisting: an empty list is a legal steady-state body."""
    bad = []
    for i, (op, d, a, b, c) in enumerate(body):
        if op not in ("IMAD", "IMAD.HI.U32") or not isinstance(a, int) or not isinstance(d, int):
            bad.append(f"{i}: {op} form")
        elif op == "IMAD.HI.U32" and not (isinstance(c, int) and c % 2 == 0):
            bad.append(f"{i}: addend pair at R{c}")
        elif op == "IMAD" and isinstance(b, tuple) and isinstance(c, tuple):
            bad.append(f"{i}: two constant operands")
        if d in inv:
            bad.append(f"{i}: writes invariant R{d}")
    return bad


# ---- the certificates and the re-solve -------------------------------------------------------------------------------

def ldtm0(data):
    """The design of record, rebuilt from the heavy design: li's fold and no tcgen05.ld in the operand floor."""
    d = E.design(HEAVY, data)
    return replace(d, name=LDTM0, fold=dict(LI_FOLD), fold_name="li", floor={**d.floor, "LDTM_32X32B_X128": 0.0})


def price(pi, lam: dict[str, float], k: str) -> float:
    return sum(w * pi.c[k].get(r, 0.0) for r, w in lam.items())


def routed(s, x: float, n: int, luck: bool = False):
    """The cheater with the share x of li's XORs on an n-instruction IMAD-pipe route (each LOP3 becomes n IMAD), the
    uniform route kept on the honest fold."""
    d = s.design
    f = dict(d.fold)
    f["ALU"] -= XOR_SHARE * x
    f["IMAD"] += XOR_SHARE * x * n
    dd = replace(d, name=f"{d.name}+xor{n}", fold=f)
    return replace(s, design=dd).cheat(Q, q_uniform=Q + d.fold_cost() - dd.fold_cost(), luck=luck)


def least(s, n: int, luck: bool = False, hi: float = 0.02, steps: int = 60) -> tuple[float, object]:
    lo, top = 0.0, hi
    for _ in range(steps):
        m1, m2 = lo + (top - lo) / 3, top - (top - lo) / 3
        if routed(s, m1, n, luck).T <= routed(s, m2, n, luck).T:
            top = m2
        else:
            lo = m1
    x = (lo + top) / 2
    return x, routed(s, x, n, luck)


def headline(s, rest: float, n: int) -> tuple[float, float, float, float, object, object]:
    T0 = s.baseline.T
    x, mix = least(s, n)
    xl, lucky = least(s, n, luck=True)
    m, lu = 1 - mix.T / T0, max(0.0, (mix.T - lucky.T) / T0)
    return rest + m + lu, m, lu, x, mix, lucky


# ---- polynomiality and the combine search ----------------------------------------------------------------------------

def newton_failures(w: int, N: int = 24) -> list[tuple[int, int, int, int]]:
    """(i, j, b, g) where XOR mod 2^w's Newton coefficient b_ij isn't divisible by g = gcd(2^w, i!·j!): XOR mod 2^w is
    an integer-polynomial function iff there are none."""
    m = 1 << w
    f = lambda a, b: (a % m) ^ (b % m)
    bad = []
    for i in range(N):
        for j in range(N):
            b = sum((-1) ** (i - p + j - q) * comb(i, p) * comb(j, q) * f(p, q)
                    for p in range(i + 1) for q in range(j + 1)) % m
            g = gcd(m, factorial(i) * factorial(j))
            if b % g:
                bad.append((i, j, b, g))
    return bad


def combine_search(W: int, depth: int) -> int:
    """Combines of `depth` instructions for a W-bit chunk in the restricted model: the number found. A value is
    (scaled, info over the 2^{2W} chunk pairs mod 2^W): unit (its value mod 2^W) or scaled (2^{32−W}·info mod 2^32). A
    product of two scaled values is 0, a unit times a scaled is scaled, a sum of a unit and a scaled keeps the unit's
    info. The inputs are the extracted chunks with the extraction's offsets, (X, Y) = (x + c1, y + c2) or
    (S, Y) = (x + y + c2, y + c2), so the target is XOR of the de-offset chunks up to an additive constant (the
    accumulator's). The last instruction is a scaled product."""
    m = 1 << W
    P, R = (a.ravel() for a in np.meshgrid(np.arange(m), np.arange(m), indexing="ij"))
    norm = lambda a: tuple(((a - a[0]) % m).tolist())
    targets = {norm(((P - c1) % m) ^ ((R - c2) % m)) for c1 in range(m) for c2 in range(m)}
    targets |= {norm(((P - R) % m) ^ ((R - c2) % m)) for c2 in range(m)}

    def mul(a, b):
        if a[0] and b[0]:
            return (True, np.zeros(m * m, dtype=np.int64))
        return (a[0] or b[0], (a[1] * b[1]) % m)

    def add(a, b):
        if a[0] == b[0]:
            return (a[0], (a[1] + b[1]) % m)
        return (False, a[1] if not a[0] else b[1])

    consts = [(sc, np.full(m * m, c, dtype=np.int64)) for sc in (False, True) for c in range(m)]
    key = lambda v: (v[0], *v[1].tolist())

    def step(regs):
        pool, out = regs + consts, {}
        for u, v in itertools.combinations_with_replacement(range(len(pool)), 2):
            pr = mul(pool[u], pool[v])
            for w in range(len(pool)):
                r = add(pr, pool[w])
                out.setdefault(key(r), r)
        return list(out.values())

    def finals(regs, need):
        pool = regs + consts
        return sum(1 for u, v in itertools.combinations_with_replacement(range(len(pool)), 2)
                   if need in (u, v) and (pr := mul(pool[u], pool[v]))[0] and norm(pr[1]) in targets)

    base = [(False, P.copy()), (False, R.copy())]
    if depth == 2:
        return sum(finals(base + [r1], 2) for r1 in step(base))
    return sum(finals(base + [r1, r2], 3) for r1 in step(base) for r2 in step(base + [r1]))


# ---- the device build's SASS ---------------------------------------------------------------------------------------

CHAINS = {8: "_Z5chainILi8EEvPKjS1_Pji", 16: "_Z5chainILi16EEvPKjS1_Pji"}
LOOP = "_Z4loopPKjS0_Pjii"
_FUNC = re.compile(r"Function : (\S+)")
_INSN = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*)")
_TARGET = re.compile(r"0x([0-9a-f]+)")


def by_class(ops: dict) -> dict:
    classes = Counter()
    for op, n in ops.items():
        classes[class_of(op) or "unclassed"] += n
    return dict(classes)


def sass_mix(text: str) -> dict:
    """One XOR's SASS mix in `fold_xor.sh`'s build, by census class and by opcode:
    - unrolled: chain<16>'s opcodes less chain<8>'s, over 8 (the chains' loads, store and exit cancel);
    - loop: the body of `loop`'s one backward branch, one XOR an iteration, the loop's own count, compare and branch
      in it."""
    per, func = {}, None
    for line in text.splitlines():
        if (m := _FUNC.search(line)):
            func = m.group(1)
            per[func] = []
        elif func and (m := _INSN.match(line)):
            per[func].append((int(m.group(1), 16), m.group(2), m.group(3)))
    count = lambda insns: Counter(op for _, op, _ in insns)
    lo, hi = count(per[CHAINS[8]]), count(per[CHAINS[16]])
    ops = {op: (hi[op] - lo[op]) / 8 for op in sorted(set(lo) | set(hi)) if hi[op] != lo[op]}
    out = {"per_xor_by_class": by_class(ops), "per_xor_by_opcode": ops, "per_xor": sum(ops.values())}
    if LOOP in per:
        back = [(int(t.group(1), 16), at) for at, op, rest in per[LOOP]
                if op.split(".")[0] == "BRA" and (t := _TARGET.search(rest)) and int(t.group(1), 16) < at]
        assert len(back) == 1, f"loop has {len(back)} backward branches, not 1"
        start, end = back[0]
        body = dict(sorted(count(i for i in per[LOOP] if start <= i[0] <= end).items()))
        out["loop"] = {"per_xor_by_class": by_class(body), "per_xor_by_opcode": body, "per_xor": sum(body.values())}
    return out


# ---- the checks ------------------------------------------------------------------------------------------------------

def main() -> int:
    if "--sass" in sys.argv:
        print(json.dumps(sass_mix(Path(sys.argv[sys.argv.index("--sass") + 1]).read_text()), indent=1))
        return 0
    ck = Checks()
    rng = random.Random(SEED)

    # -- 1. the program ------------------------------------------------------------------------------------------------
    prog = xor_program()
    npts, bad = bit_exact(prog, 0, rng, 200_000)
    ck.check("the 78-instruction program is x ^ y, bit-exact (the 12 edge words crossed, and 200,000 random pairs)",
             bad == 0 and len(prog) == 78, f"{npts} points, {bad} mismatches")
    keyed = [0xFFFFFFFF, 0x80000001, 0x0F0F0F0F, 0xAAAAAAAA, *(rng.getrandbits(32) for _ in range(60))]
    kbad = sum(bit_exact(xor_program(K), K, rng, 2000)[1] for K in keyed)
    ck.check("with li's constant k folded into its constants it is x ^ y ^ k at the same 78 (64 keys, the edge "
             "words and 2,000 random pairs each)", kbad == 0 and all(len(xor_program(K)) == 78 for K in keyed),
             f"{kbad} mismatches")
    inv, body = sass_body()
    skeys = [0, 0x9E13B91C, 0xFFFFFFFF, *(rng.getrandbits(32) for _ in range(5))]
    sbad = [sass_exact(K, rng, 20_000)[1] for K in skeys]
    sops = Counter(op for op, *_ in body)
    ck.check("as a SASS loop body with its registers allocated it is 78 a XOR (IMAD 33, IMAD.HI.U32 45, class IMAD) "
             "and needs no move: three XORs a body over three rotating registers, IMAD.HI.U32's addend pairs {0, c} "
             "and the constants in registers the body never writes, within SASS's operand rules; bit-exact under the "
             "pair semantics over 3 passes (9 chained XORs) for 8 keys",
             len(body) == 3 * 78 and sops == {"IMAD": 99, "IMAD.HI.U32": 135} and not sass_legal(inv, body)
             and all(sass_body(K)[1] and not sass_legal(*sass_body(K)) for K in skeys)
             and {class_of(op) for op in sops} == {"IMAD"} and sum(sbad) == 0,
             {"ops": dict(sops), "invariant registers": len(inv), "mismatches": sbad, "illegal": sass_legal(inv, body)})
    forms = {f: sum(1 for p in prog if p[0] == f) for f in SASS}
    kinds = {f: class_of(SASS[f]) for f in SASS}
    ck.check("its forms are IMAD (33) and IMAD.HI (45), each class IMAD at DS-100 (facet_break.loads.class_of); no "
             "IMAD.WIDE, IDP, FP or ALU instruction", forms == {"IMAD": 33, "IMAD.HI": 45}
             and set(kinds.values()) == {"IMAD"}, {"forms": forms, "classes": kinds})
    menu = {op: class_of(op) for op in ("IMAD.WIDE.U32", "HFMA2", "HADD2", "IDP.4A.U8.U8", "IDP.2A.LO.U16.U8", "FFMA2",
                                        "FADD2", "FMUL2", "FFMA")}
    data = device("sm100")
    pi = E.machine("DS-100", data)
    ck.check("the menu: DS-100 prices class IMAD (IDP.4A and IDP.2A included) and FFMA (FFMA2, FADD2, FMUL2 "
             "included); IMAD.WIDE (IMADW) and HFMA2 have no DS-100 row, so no program prices them there",
             menu["IMAD.WIDE.U32"] == "IMADW" and menu["HFMA2"] == menu["HADD2"] == "HFMA2"
             and {menu["IDP.4A.U8.U8"], menu["IDP.2A.LO.U16.U8"]} == {"IMAD"}
             and {menu[k] for k in ("FFMA2", "FADD2", "FMUL2", "FFMA")} == {"FFMA"}
             and "IMADW" not in pi.c and "HFMA2" not in pi.c and {"IMAD", "FFMA"} <= set(pi.c), menu)

    # -- 2. the certificates -------------------------------------------------------------------------------------------
    d = ldtm0(data)
    s = E.setup(d, pi, data, "free", "floor")
    plain, lucky = s.cheat(Q), s.cheat(Q, luck=True)
    pa, pm = price(pi, plain.pi, "ALU"), price(pi, plain.pi, "IMAD")
    la, lm = price(pi, lucky.pi, "ALU"), price(pi, lucky.pi, "IMAD")
    ck.check("ldtm0's plain cheater certificate is DS_issue 0.02135 + DS_alu 0.97865: a LOP3 1.9786, an IMAD 0.02135, "
             "break-even 92.66 IMAD-pipe instructions a LOP3", abs(plain.pi["DS_issue"] - 0.0213538) < 1e-6
             and abs(pa / pm - 92.6599) < 1e-3, {"pi": plain.pi, "breakeven": round(pa / pm, 4)})
    ck.check("its lucky cheater's is DS_issue 0.03147 + DS_alu 0.96853, break-even 62.54 (INT-HB's Lean-form π*, "
             "0.0314 + 0.9686, is this structure)", abs(lucky.pi["DS_issue"] - 0.0314742) < 1e-6
             and abs(la / lm - 62.545) < 1e-2, {"pi": lucky.pi, "breakeven": round(la / lm, 4)})
    real = E.design(LDTM0, data) if LDTM0 in E.designs(data) else None   # #2706's, once it lands
    if real is not None:
        bare = replace(real, fold_routes={}, fold_routes_source="")
        rp, r0 = E.setup(real, pi, data, "free", "floor").cheat(Q), E.setup(bare, pi, data, "free", "floor").cheat(Q)
        ra, rm = price(pi, rp.pi, "ALU"), price(pi, rp.pi, "IMAD")
        ck.check("the rebuild's certificate is the real design's with its fold routes off (#2706's menu), bit for bit; "
                 "with them on it is DS_issue 2/3 + DS_alu 1/3, break-even 2", r0.pi == plain.pi
                 and real.fold == d.fold and abs(rp.pi["DS_issue"] - 2 / 3) < 1e-9 and abs(ra / rm - 2) < 1e-9,
                 {"routes off": r0.pi, "routes on": rp.pi, "breakeven": round(ra / rm, 4)})
    ck.check("78 is under the plain break-even (the design's 'no IMAD- or FP-pipe program under 93' is false) and over "
             "the lucky one", 78 * pm < pa and 78 * lm > la,
             f"plain: 78 x {pm:.5f} = {78 * pm:.4f} < {pa:.4f}; lucky: 78 x {lm:.5f} = {78 * lm:.4f} > {la:.4f}")

    # -- 3. the headline -----------------------------------------------------------------------------------------------
    T0 = s.baseline.T
    lu0 = max(0.0, (plain.T - lucky.T) / T0)
    base = 1 - plain.T / T0
    rest = LDTM0_SKIP_DS100
    if real is not None:
        e0 = E.eps(bare, pi, Q, data, operands="free", credit="floor")
        ck.check("the real design's mand is 0 and, with its fold routes off, its headline is its skip + ε_machine + "
                 "ε_luck (its [skip.served], re-measured with the routes, is not #2706's constant)", e0.mand == 0.0
                 and abs(e0.core() - (e0.skip + base + lu0)) < 1e-9,
                 f"skip {e0.skip:.4%} (#2706's {rest:.4%}), core {e0.core():.5%}")
    h78, m78, l78, x78, mix78, lk78 = headline(s, rest, 78)
    ck.check("re-solved with the 78 route open, the plain cheater moves a share of li's XORs and ε_machine rises; "
             "ε_luck falls by as much and the lucky cheater doesn't move, so ε_machine + ε_luck is unchanged",
             x78 > 1e-5 and m78 > base and abs((m78 + l78) - (base + lu0)) < 1e-9 and abs(lk78.T - lucky.T) < 1e-12,
             f"x {x78:.6f}; ε_machine {base:.5%} -> {m78:.5%}, ε_luck {lu0:.5%} -> {l78:.5%}")
    hd = {n: headline(s, rest, n)[0] for n in (1, 3, 4, 20, 40, 62, 63, 78, 93)}
    ck.check("against the route's length n the headline is unchanged at 0.7052% for n >= 63 (and at 93, which doesn't "
             "pay), rises at 62 and below, passes 1% only at n <= 3, and at n = 1 is the count-only cap 1.329%",
             all(abs(hd[n] - (rest + base + lu0)) < 1e-9 for n in (63, 78, 93))
             and abs(rest + base + lu0 - 0.007052) < 5e-7 and hd[62] > hd[63]
             and hd[1] > hd[3] > 0.01 > hd[4] > hd[20] > hd[40] > hd[62] and abs(hd[1] - 0.013288) < 5e-6,
             {n: f"{v:.4%}" for n, v in hd.items()})

    # -- 4. what's proven, and the family search -----------------------------------------------------------------------
    fails = {w: newton_failures(w) for w in (2, 3, 4)}
    ck.check("XOR mod 4 and mod 8 are polynomial functions and XOR mod 16 isn't (Newton coefficient (2, 4) is 8, not "
             "divisible by 16): no program of IMAD's low-word forms alone computes x ^ y at any length",
             not fails[2] and not fails[3] and (2, 4, 8, 16) in fails[4], fails[4][:3])
    ck.check("XOR mod 4 = A + B - 2AB on every 2-bit pair (the program's combine)",
             all((a + b - 2 * a * b) % 4 == a ^ b for a in range(4) for b in range(4)))
    found = {(W, c): combine_search(W, c) for W, c in ((2, 2), (3, 2), (3, 3))}
    ck.check("the restricted mod-8 search: no 3-bit combine of 2 or 3 instructions, while the same search finds the "
             "2-instruction 2-bit one (so it isn't vacuous); 3-bit chunks would need c <= 2 to total under 63",
             found[(2, 2)] > 0 and found[(3, 2)] == 0 and found[(3, 3)] == 0, found)

    print("\nB200 sheet (DS-100), ldtm0, li's XOR on the IMAD pipe:")
    print(f"  plain break-even {pa / pm:.2f}, lucky {la / lm:.2f}; the program is 78")
    for n, v in hd.items():
        print(f"  n = {n:3d}: headline {v:.4%}")

    if ck.mutations:
        assert prog[3][:2] == ("IMAD.HI", "Bp") and isinstance(prog[3][3], int)
        bad1 = [(f, dst, a, u32(b + (1 << 20)) if i == 3 else b, c) for i, (f, dst, a, b, c) in enumerate(prog)]
        ck.mutation("chunk 1's extraction shift off by 2^20 breaks the program (the bit-exact check isn't vacuous)",
                    bit_exact(bad1, 0, rng, 256)[1] > 0)
        bad2 = [(f, dst, a, b, u32(c - (1 << 30)) if dst == "H" and isinstance(c, int) else c)
                for f, dst, a, b, c in prog]
        ck.mutation("the -2^30 folded into H's addend is load-bearing", bit_exact(bad2, 0, rng, 256)[1] > 0)
        ck.mutation("x ^ y's program is wrong for a key with a nonzero chunk (k does change the constants)",
                    bit_exact(prog, 0x0000000C, rng, 256)[1] > 0)
        ck.mutation("the SASS body's B' pair with a nonzero low half breaks it (the interpreter reads the pair)",
                    sass_exact(0, rng, 256, inv={**inv, 10: M32})[1] > 0)
        x93, _ = least(s, 93)
        ck.mutation("a route over the plain break-even (93) doesn't pay: the re-solve moves nothing", x93 < 1e-9,
                    f"x {x93:.2e}")
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
