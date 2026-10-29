"""FOLD/DYNAMIC: can the fold be drawn per call? Fold families whose structure (op order, rotations) and constants are
live coins, drawn after the words are committed; for each, whether a random member has a cheaper program than its
honest one.

A family F gives each step t a member θ_t of b coin bits and a step s' = step(θ_t, s, w). The families (fold_dyn.c's,
at width W; at 32 bits each multiplier is odd and each rotation in 1..31):

- rotadd       s' = rotl(s, r) + w                      honest 1 (LEA.HI, r an immediate), 2 (SHF.L.W + IADD3, r a register)
- rotxoradd    s' = rotl(s ⊕ x, r) + w                  2 / 3
- rotmuladd    s' = rotl(s, r)·k + w                    2 / 2
- mulrotadd    s' = rotl(s·k, r) + w                    2 / 3
- xormul       s' = (s ⊕ w)·k                           2   (SPARSE's (c4))
- xorrot       s' = rotl(s ⊕ w, r)                      2
- lik          s' = (s ⊕ w ⊕ x)·k + y                   2   (LOP3 + IMAD: li's half step with live keys)
- arxm         s' = (rotl(s ⊕ x, r) + w)·k + y          3 / 4 (LOP3 + LEA.HI + IMAD; SHF.L.W + IADD3 for the LEA.HI)
- rand         s' = g2(g1(s) (+|⊕) w), g1 ∈ {rot, mul, xor, add}, g2 ∈ {rot, mul}, each with its constant: 2–3

The honest counts are the lane instructions `fold_dyn_sass.sh` reads off sm_90a's and sm_120's SASS, beside the
word's own IADD3.

What this script checks (CPU):

1. **The superoptimizer battery** (`fold_dyn`, built here with gcc): for random members, single and multi-step, at
   width W ≤ 8 with every constant free in every operand, the least program of at most 2 instructions over the
   modelled ISA that computes the member's state on every input (or on every zero-data input, `-z`). Controls: rotadd's
   step is found at 1 and two of its steps at 2; the 2-instruction families' steps are found at 2; degenerate members
   (k = 1, x = 0) are found shorter; leaving LEA.HI out of the ISA loses rotadd's 1-instruction program (a mutation).
2. **decomp** (COUNTER's attack): at zero data the words are x + R_t(i) + C_t(j), so a fold that decomposes over the
   sum is cheap. For a member over T steps, the 2^W × 2^W matrix of each state bit over (row part, column part) has a
   GF(2) rank, the number of AND terms a bit-sliced decomposition needs; and the additive residue (how often
   f(a1,b1) - f(a1,b2) - f(a2,b1) + f(a2,b2) ≠ 0) is 0 exactly when the state is row plus column. A fold resists
   decomp when every bit is near full rank and the residue near 1 - 2^-W after a few steps.
3. **The counting theorem** (paper, below): the least per-output program for a random member over T steps has at least
   (b'T - s)/d instructions except with probability 2^-s, b' the coin bits a step less those its symmetries lose
   (`LOSS`). The script computes d, the bound a step, and its margin over credit B's 0.866 eq a step, for each family
   at T = 74 and 293 (the lead's window shapes), and counts the distinct functions of every one-step member (widths 4
   and 6) and every two-step pair (width 3).

The counting theorem. Let P(L) be the straight-line programs of L instructions over a form set of size F (opcode,
modifiers, immediate shift or LUT, predicate guard), each operand an input (n of them), an earlier result (two a
two-output instruction) or a 32-bit constant, at least one operand not a constant (a minimal program has none: fold the
constant into its readers), and the result that is the output one of n + 2L. |P(L)| ≤ (n + 2L) · (F · 3(n + 2L) ·
(n + 2L + 2^32)^2)^L = (n + 2L) · 2^(d(L)·L), and `Pouw.FoldDyn.card_computed_by_short_le` (Security/Proofs/Pouw/
FoldDyn.lean) proves the count of members some such program computes, under any semantics. If θ is uniform on 2^(bT)
members and each function is at most 2^(T(b - b')) members' (`LOSS`), then Pr_θ[some program in P(L) computes f_θ]
≤ |P(L)| · 2^(-b'T). So with
probability ≥ 1 - 2^-s, every program computing f_θ has L ≥ (b'T - s)/d(L). It counts per-output SIMT programs
whose inputs are the words (not functions of θ): a program shared across outputs (packing, bit-slicing, per-row or
per-column precomputation, tables) escapes it, and that is the premise's part. Each lane instruction costs at least
the dispatch floor, 32 W1 units = 1 eq (4 warp instructions an SM-clock), whatever it writes: under W1's write count
alone predicate-only instructions are free and the bound says nothing.

Run: `python fold_dynamic.py [--tier quick|full] [--mutations] [--verdict PATH]`. Stdlib, exact_kit and gcc. `quick`
(the suite's, about 2 minutes on 4 cores) runs width 6 and the width-8 single steps that finish fast; `full` (CPU
minutes on a node) adds every member at width 8, width-8 two-step members and three steps of rotadd.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from exact_kit.checks import Checks

HERE = Path(__file__).resolve().parent

#: per step at 32 bits: coin bits b, honest lane instructions with r an immediate and with r a register
FAMILIES = {
    "rotadd": (math.log2(31), 1, 2),
    "rotxoradd": (32 + math.log2(31), 2, 3),
    "rotmuladd": (31 + math.log2(31), 2, 2),
    "mulrotadd": (31 + math.log2(31), 2, 3),
    "xormul": (31.0, 2, 2),
    "xorrot": (math.log2(31), 2, 2),
    "lik": (32 + 31 + 32.0, 2, 2),
    "arxm": (32 + math.log2(31) + 31 + 32, 3, 4),
}
#: coin bits a step lost to symmetries: (within one step, a step of a long fold). arxm: (x, y) ~ (x ⊕ 2^(31-r),
#: y + 2^31). lik: (x, y) ~ (x ⊕ 2^31, y + 2^31) and (x, k, y) ~ (~x, -k, y - k). Across steps y_t's top bit moves
#: into x_{t+1}'s (adding 2^31 is xoring it), 2 and 3 a step at 32 bits; every pair of steps at width 3 loses 4 and 7, so
#: a long fold's figure is the width-3 one's half, the larger.
LOSS = {"arxm": (1, 2), "lik": (2, 3.5), "rotmuladd": (0, 0), "xormul": (0, 0), "rotxoradd": (0, 0)}
#: forms: census forms (sm_90's 963) x 256 (LUT, shift or sign set) x 16 (predicate guard)
FORMS = 963 * 256 * 16
#: credit B: the adversary's zero-data cost must be >= 0.99 of stock, 28/32 eq a k32 step
CREDIT_B = 0.99 * 28 / 32
#: the lead's window shapes (k32 steps a window)
STEPS = (74, 293)
INPUTS = 64
SECURITY = 40


def d_bits(L: int, n: int = INPUTS) -> float:
    """log2 of the programs of one more instruction: form, one variable operand of 3, two operands of any kind."""
    m = n + 2 * L
    return math.log2(FORMS) + math.log2(3 * m) + 2 * math.log2(m + 2 ** 32)


def counting_bound(b: float, T: int, s: int = SECURITY, n: int = INPUTS) -> int:
    """The least L with log2 |Prog(L)| = log2(n + 2L) + d(L)·L ≥ bT - s (`Pouw.FoldDyn.card_prog`, the output's
    result chosen among n + 2L): below it, at most 2^-s of the members have a program of L instructions."""
    need = b * T - s
    L = 0
    while math.log2(n + 2 * (L + 1)) + d_bits(L + 1, n) * (L + 1) < need:
        L += 1
    return L + 1


# ---- the superoptimizer ------------------------------------------------------------------------------------------------

def build(tmp: Path) -> Path:
    exe = tmp / "fold_dyn"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(HERE / "fold_dyn.c")], check=True)
    return exe


def search(exe: Path, W: int, fam: str, S: int, D: int, members: int, seed: int, *flags: str, NP: int = 32) -> dict:
    """Run fold_dyn; {'n1', 'members': [(theta, honest, found, program)]}, found an int (0: an input) or None."""
    out = subprocess.run([str(exe), str(W), str(NP), fam, str(S), str(D), str(members), str(seed), *flags],
                         capture_output=True, text=True, check=True).stdout
    res = {"n1": 0, "members": []}
    for line in out.splitlines():
        if line.startswith("N1 "):
            res["n1"] = int(line.split()[1])
        elif line.startswith("M "):
            parts = line.split(maxsplit=5)
            theta, honest, found = parts[1], int(parts[3]), parts[5] if len(parts) > 5 else "-"
            f = found.split(maxsplit=1)
            if f[0] == "-":
                res["members"].append((theta, honest, None, ""))
            elif f[0] == "0":
                res["members"].append((theta, honest, 0, "input"))
            else:
                res["members"].append((theta, honest, int(f[0]), f[1] if len(f) > 1 else ""))
    return res


def summary(res: dict) -> dict:
    ms = res["members"]
    short = [m for m in ms if m[2] is not None and m[2] < m[1]]
    found_at = {}
    for m in ms:
        found_at[str(m[2])] = found_at.get(str(m[2]), 0) + 1
    return {"members": len(ms), "n1": res["n1"], "found_at": found_at, "shorter_than_honest": len(short),
            "examples": [f"{m[0]}: {m[3]}" for m in short[:3]]}


def degenerate(fam: str, theta: str, W: int) -> bool:
    """A member with a step that collapses: arxm's key x with rotl(x, r) in {0, 2^(W-1)} (the xor is then an add, so
    the step is LEA.HI and IMAD), or a multiplier k = 1. At 32 bits each has probability about 2^-31 a step."""
    top = 1 << (W - 1)
    for st in theta.split(","):
        f = dict((p[0], int(p[1:])) for p in st.split("."))
        if fam == "arxm" and (((f["x"] << f["r"]) | (f["x"] >> (W - f["r"]))) & ((1 << W) - 1)) in (0, top):
            return True
        if fam in ("rotmuladd", "mulrotadd", "xormul") and f.get("k") == 1:
            return True
    return False


# ---- decomp: the zero-data state over (row part, column part) ---------------------------------------------------------

def step32(fam: str, th: dict, s: int, w: int, W: int) -> int:
    M = (1 << W) - 1

    def rotl(x, r):
        r %= W
        return ((x << r) | (x >> (W - r))) & M if r else x

    k, x, y, r = th["k"], th["x"], th["y"], th["r"]
    if fam == "rotadd":
        return (rotl(s, r) + w) & M
    if fam == "rotxoradd":
        return (rotl(s ^ x, r) + w) & M
    if fam == "rotmuladd":
        return (rotl(s, r) * k + w) & M
    if fam == "mulrotadd":
        return (rotl((s * k) & M, r) + w) & M
    if fam == "xormul":
        return ((s ^ w) * k) & M
    if fam == "xorrot":
        return rotl(s ^ w, r)
    if fam == "lik":
        return ((s ^ w ^ x) * k + y) & M
    if fam == "arxm":
        return ((rotl(s ^ x, r) + w) * k + y) & M
    if fam == "xor":
        return s ^ w
    raise ValueError(fam)


def draw(rng: random.Random, W: int) -> dict:
    M = (1 << W) - 1
    return {"k": rng.randrange(1 << W) | 1, "x": rng.randrange(1 << W), "y": rng.randrange(1 << W),
            "r": rng.randrange(1, W)}


def gf2_rank(rows: list[int]) -> int:
    rank, rows = 0, [r for r in rows if r]
    while rows:
        p = max(rows)
        top = p.bit_length() - 1
        rows = [r ^ p if (r >> top) & 1 else r for r in rows if r != p]
        rows = [r for r in rows if r]
        rank += 1
    return rank


def decomp(fam: str, W: int, T: int, seed: int) -> dict:
    """The state after T zero-data steps over 2^W rows and columns: words w_t = x_t + R_t(i) + C_t(j) with R_t, C_t
    random walks (R_t = R_{t-1} + ρ_t(i)); each state bit's GF(2) rank over (i, j), and the additive residue."""
    rng = random.Random(seed)
    M, n = (1 << W) - 1, 1 << W
    th = [draw(rng, W) for _ in range(T)]
    x0 = [rng.randrange(n) for _ in range(T)]
    R = [[0] * T for _ in range(n)]
    C = [[0] * T for _ in range(n)]
    for i in range(n):
        a = c = 0
        for t in range(T):
            a = (a + rng.randrange(n)) & M
            c = (c + rng.randrange(n)) & M
            R[i][t], C[i][t] = a, c
    s0 = rng.randrange(n)
    F = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = s0
            for t in range(T):
                s = step32(fam, th[t], s, (x0[t] + R[i][t] + C[j][t]) & M, W)
            F[i][j] = s
    ranks = []
    for b in range(W):
        rows = []
        for i in range(n):
            v = 0
            for j in range(n):
                v |= ((F[i][j] >> b) & 1) << j
            rows.append(v)
        ranks.append(gf2_rank(rows))
    bad = tot = 0
    for _ in range(4096):
        a1, a2, b1, b2 = (rng.randrange(n) for _ in range(4))
        tot += 1
        bad += ((F[a1][b1] - F[a1][b2] - F[a2][b1] + F[a2][b2]) & M) != 0
    return {"ranks": ranks, "residue": bad / tot}


def fold32(fam: str, T: int, N: int, seed: int):
    """The 32-bit state after T zero-data steps of a random member over an N x N block of outputs: words
    w_t = x_t + R_t(i) + C_t(j), with R_t and C_t walks of 14-bit increments (about 20 bits at T = 74, as the lead's)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    u32 = np.uint32
    s = np.full((N, N), rng.integers(0, 2 ** 32, dtype=np.uint64).astype(u32), dtype=u32)
    R = np.zeros(N, dtype=u32)
    C = np.zeros(N, dtype=u32)
    with np.errstate(over="ignore"):
        for _ in range(T):
            R += rng.integers(0, 2 ** 14, N, dtype=np.uint64).astype(u32)
            C += rng.integers(0, 2 ** 14, N, dtype=np.uint64).astype(u32)
            x0, x, y = (u32(int(v)) for v in rng.integers(0, 2 ** 32, 3, dtype=np.uint64))
            k = u32(int(rng.integers(0, 2 ** 32, dtype=np.uint64)) | 1)
            r = u32(int(rng.integers(1, 32)))
            w = x0 + R[:, None] + C[None, :]
            if fam == "arxm":
                u = s ^ x
                s = ((u << r) | (u >> (u32(32) - r))) + w
                s = s * k + y
            elif fam == "lik":
                s = (s ^ w ^ x) * k + y
            elif fam == "xormul":
                s = (s ^ w) * k
            elif fam == "xor":
                s = s ^ w
            else:
                raise ValueError(fam)
    return s


def gf2_rank_np(bits) -> int:
    """GF(2) rank of an (N, N) 0/1 matrix, N a multiple of 64, by elimination on packed rows."""
    import numpy as np

    M = np.packbits(bits.astype(np.uint8), axis=1, bitorder="little").view(np.uint64).copy()
    rows, n = M.shape[0], bits.shape[1]
    rank = 0
    for col in range(n):
        wd, b = divmod(col, 64)
        bit = np.uint64(1 << b)
        cand = np.nonzero(M[rank:, wd] & bit)[0]
        if cand.size == 0:
            continue
        p = rank + int(cand[0])
        if p != rank:
            M[[rank, p]] = M[[p, rank]]
        hit = (M[:, wd] & bit) != 0
        hit[rank] = False
        M[hit] ^= M[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def decomp32(fam: str, T: int, N: int, seed: int, bits=(0, 1, 2, 3, 8, 16, 31)) -> dict:
    """decomp at full width: each listed state bit's GF(2) rank over an N x N block. A bit-sliced rank-r decomposition
    costs about r/4 lane instructions an output, so a rank near N with N >= 12T is no cheaper than honest arxm."""
    import numpy as np

    F = fold32(fam, T, N, seed)
    return {str(b): gf2_rank_np(((F >> np.uint32(b)) & np.uint32(1)).astype(np.uint8)) for b in bits}


# ---- injectivity of one step ----------------------------------------------------------------------------------------

def one_step_tables(fam: str, W: int):
    """Every member of one step at width W (k odd, r in 1..W-1, x and y free where the family has them), as an
    (members, 2^W, 2^W) array of s' over (s, w)."""
    import numpy as np

    n = 1 << W
    uses = {"rotadd": "r", "rotxoradd": "xr", "rotmuladd": "kr", "mulrotadd": "kr", "xormul": "k", "xorrot": "r",
            "lik": "kxy", "arxm": "kxyr"}[fam]
    ks = range(1, n, 2) if "k" in uses else [1]
    xs = range(n) if "x" in uses else [0]
    ys = range(n) if "y" in uses else [0]
    rs = range(1, W) if "r" in uses else [1]
    tabs = [[[step32(fam, {"k": k, "x": x, "y": y, "r": r}, s, w, W) for w in range(n)] for s in range(n)]
            for k in ks for x in xs for y in ys for r in rs]
    return np.array(tabs, dtype=np.uint8)


def classes(tabs) -> int:
    import numpy as np

    flat = np.ascontiguousarray(tabs.reshape(tabs.shape[0], -1))
    return len(np.unique(flat.view(np.dtype((np.void, flat.shape[1])))))


def two_step_classes(fam: str, W: int) -> tuple[int, int]:
    """Members of two steps at width W, every pair, and the distinct functions of (s, w1, w2) among them."""
    import numpy as np

    T = one_step_tables(fam, W)
    m, n = T.shape[0], 1 << W
    F = np.empty((m, m, n, n, n), dtype=np.uint8)
    for a in range(m):
        F[a] = T[:, T[a]]  # F[a, b, s, w1, w2] = T[b, T[a, s, w1], w2]
    return m * m, classes(F.reshape(m * m, n, n * n))


def main() -> int:
    ck = Checks()
    tier = "quick"
    if "--tier" in ck.args:
        tier = ck.args[ck.args.index("--tier") + 1]
    full = tier == "full"
    out_dir = Path(os.environ.get("OUT", "."))
    report: dict = {"tier": tier, "search": {}, "decomp": {}, "counting": {}, "injective": {}}

    # 3. the counting theorem, first: it needs no search
    for fam, (b, h_imm, h_reg) in FAMILIES.items():
        row = {}
        b_eff = b - LOSS.get(fam, (0, 0))[1]
        for T in STEPS:
            L = counting_bound(b_eff, T, n=2 * T + 4)
            row[str(T)] = {"L": L, "per_step": L / T, "d": d_bits(L, 2 * T + 4), "margin_over_credit_b": L / T - CREDIT_B}
        report["counting"][fam] = {"b": b, "b_eff": b_eff, "honest_imm": h_imm, "honest_reg": h_reg, **row}
        print(f"counting {fam:10s} b={b:6.2f} b_eff={b_eff:6.2f}  " + "  ".join(
            f"T={T}: L>={row[str(T)]['L']} ({row[str(T)]['per_step']:.3f}/step, d {row[str(T)]['d']:.1f})"
            for T in STEPS))
    for fam in ("arxm", "lik"):
        ck.check(f"counting: {fam} per-output programs need over credit B's {CREDIT_B:.3f} eq a step",
                 all(report["counting"][fam][str(T)]["per_step"] > CREDIT_B for T in STEPS),
                 ", ".join(f"T {T}: {report['counting'][fam][str(T)]['per_step']:.3f}" for T in STEPS))
    ck.check("counting: rotadd's 5 coin bits a step buy under 0.1 eq a step (structure alone is cheap to describe)",
             all(report["counting"]["rotadd"][str(T)]["per_step"] < 0.1 for T in STEPS))

    for fam, W in (("arxm", 4), ("lik", 4), ("rotmuladd", 6), ("xormul", 6), ("rotxoradd", 6)):
        tabs = one_step_tables(fam, W)
        n, distinct = tabs.shape[0], classes(tabs)
        lost = math.log2(n / distinct)
        report["injective"][fam] = {"width": W, "members": n, "distinct": distinct, "bits_lost": lost}
        ck.check(f"injective: one step of {fam} loses exactly its stated symmetries' {LOSS[fam][0]} bit(s) (width {W})",
                 abs(lost - LOSS[fam][0]) < 1e-9, f"{distinct} functions of {n} members")
    for fam in ("arxm", "lik"):
        n, distinct = two_step_classes(fam, 3)
        lost = math.log2(n / distinct)
        report["injective"][f"{fam}.two_steps"] = {"width": 3, "members": n, "distinct": distinct, "bits_lost": lost}
        ck.check(f"injective: two steps of {fam} lose at most 2 x {LOSS[fam][1]} bits (width 3, every pair)",
                 lost <= 2 * LOSS[fam][1] + 1e-9, f"{distinct} functions of {n} members, {lost:.2f} bits")

    # 2. decomp
    W_dec = 6 if not full else 7
    for fam in ("xor", "xormul", "lik", "rotxoradd", "arxm"):
        for T in (1, 2, 4):
            res = decomp(fam, W_dec, T, 7 + T)
            report["decomp"][f"{fam}.T{T}"] = res
            print(f"decomp {fam:10s} W={W_dec} T={T}: ranks {res['ranks']}  residue {res['residue']:.3f}", flush=True)
    full_rank = (1 << W_dec)

    def resists(res: dict) -> bool:
        return min(res["ranks"][1:]) > 0.75 * full_rank and res["residue"] > 0.9

    ck.check("decomp control: the unkeyed XOR fold's low bit has GF(2) rank <= 2 at every T",
             all(report["decomp"][f"xor.T{T}"]["ranks"][0] <= 2 for T in (1, 2, 4)))
    ck.check("decomp control: arxm's first step is row plus column (additive residue 0)",
             report["decomp"]["arxm.T1"]["residue"] == 0.0)
    ck.check("decomp: arxm at 4 steps has every bit but the lowest above 3/4 of full rank and residue above 0.9",
             resists(report["decomp"]["arxm.T4"]),
             f"ranks {report['decomp']['arxm.T4']['ranks']}, residue {report['decomp']['arxm.T4']['residue']:.3f}")
    if ck.mutations:
        ck.mutation("the unkeyed XOR fold in arxm's place fails the decomp check",
                    not resists(report["decomp"]["xor.T4"]), f"ranks {report['decomp']['xor.T4']['ranks']}")
    ck.check("decomp: xormul's low bit stays low rank (a T-function: bit 0 reads bits 0 only)",
             report["decomp"]["xormul.T4"]["ranks"][0] <= 2, f"ranks {report['decomp']['xormul.T4']['ranks']}")

    # 2b. decomp at 32 bits: each state bit's GF(2) rank over an N x N block of zero-data outputs
    N32, Ts32 = (2048, (1, 4, 74, 293)) if full else (512, (1, 4, 16, 74))
    report["decomp32"] = {"N": N32}
    for fam in ("xor", "xormul", "lik", "arxm"):
        for T in Ts32:
            r = decomp32(fam, T, N32, 11 + T)
            report["decomp32"][f"{fam}.T{T}"] = r
            print(f"decomp32 {fam:7s} N={N32} T={T}: ranks by bit {r}", flush=True)
    d32 = report["decomp32"]
    ck.check("decomp32 control: the unkeyed XOR fold's bit 0 has rank <= 2 and its bit 31 <= 1 at every T",
             all(d32[f"xor.T{T}"]["0"] <= 2 and d32[f"xor.T{T}"]["31"] <= 1 for T in Ts32))
    ck.check("decomp32 control: arxm's first step's bit 0 is row xor column (rank <= 2)", d32["arxm.T1"]["0"] <= 2)
    ck.check(f"decomp32: from 4 steps every tested bit of arxm has rank >= N - 8 on an N = {N32} block (32 bits)",
             all(v >= N32 - 8 for T in Ts32 if T >= 4 for v in d32[f"arxm.T{T}"].values()),
             "; ".join(f"T {T}: min {min(d32[f'arxm.T{T}'].values())}" for T in Ts32 if T >= 4))
    ck.check("decomp32: (c4) and lik keep bit 0 row xor column at every T (the T-function's free low bit)",
             all(d32[f"{f_}.T{T}"]["0"] <= 2 for f_ in ("xormul", "lik") for T in Ts32))
    if ck.mutations:
        ck.mutation("the unkeyed XOR fold in arxm's place fails the 32-bit rank check",
                    not all(v >= N32 - 8 for T in Ts32 if T >= 4 for v in d32[f"xor.T{T}"].values()))
    if full:
        r = decomp32("arxm", 293, 4096, 304)
        d32["arxm.T293.N4096"] = r
        ck.check("decomp32: arxm at 293 steps is full rank on a 4096 block (a rank route costs >= 1024 an output, "
                 "honest 879)", all(v >= 4096 - 8 for v in r.values()), str(r))

    # 1. the battery
    with tempfile.TemporaryDirectory() as tmp:
        exe = build(Path(tmp))
        runs = []
        # single steps, every member, D 1 at width 8 (D 2 where it is quick)
        for fam in ("rotadd", "rotmuladd", "xormul", "rotxoradd", "mulrotadd", "xorrot", "lik"):
            runs.append((f"{fam}.W8.S1.all", 8, fam, 1, 1, 0, 1, ("-a",)))
        runs.append(("arxm.W8.S1.all.D1", 8, "arxm", 1, 1, 0, 1, ("-a",)))
        # single steps at depth 2, width 6
        for fam in ("rotmuladd", "xormul", "rotxoradd", "mulrotadd", "xorrot", "lik", "arxm", "rand"):
            runs.append((f"{fam}.W6.S1.D2", 6, fam, 1, 2, 24 if not full else 96, 2, ()))
        # two steps at depth 2, width 6: rotadd's control, then the 2- and 3-instruction families
        runs.append(("rotadd.W6.S2.D2", 6, "rotadd", 2, 2, 8 if not full else 64, 3, ()))
        for fam in ("rotmuladd", "xormul", "lik", "arxm", "rand"):
            runs.append((f"{fam}.W6.S2.D2", 6, fam, 2, 2, 4 if not full else 64, 4, ()))
        # zero data
        runs.append(("rotadd.W6.S2.D2.z", 6, "rotadd", 2, 2, 4 if not full else 32, 5, ("-z",)))
        runs.append(("arxm.W6.S2.D2.z", 6, "arxm", 2, 2, 2 if not full else 32, 5, ("-z",)))
        if full:
            for fam in ("rotmuladd", "xormul", "lik", "arxm", "rand"):
                runs.append((f"{fam}.W8.S1.D2", 8, fam, 1, 2, 48, 6, ()))
            runs.append(("rotadd.W6.S3.D2", 6, "rotadd", 3, 2, 16, 7, ()))
            for fam in ("arxm", "lik", "rand"):
                runs.append((f"{fam}.W8.S2.D2", 8, fam, 2, 2, 8, 8, ()))
        for name, W, fam, S, D, m, seed, flags in runs:
            res = search(exe, W, fam, S, D, m, seed, *flags)
            report["search"][name] = summary(res)
            report["search"][name]["raw"] = res["members"]
            sm = report["search"][name]
            print(f"search {name:24s} N1 {sm['n1']:9d}  members {sm['members']:5d}  found {sm['found_at']}  "
                  f"shorter than honest {sm['shorter_than_honest']}  {sm['examples'][:1]}", flush=True)

        S_ = report["search"]
        raw = lambda n: S_[n]["raw"]
        ck.check("control: every member of rotadd's step is one LEA.HI (width 8)",
                 all(m[2] == 1 and m[3].startswith("LEA.HI") for m in raw("rotadd.W8.S1.all")))
        ck.check("control: two rotadd steps are found at 2 (width 6)",
                 all(m[2] is not None and m[2] <= 2 for m in raw("rotadd.W6.S2.D2")))
        for fam in ("rotmuladd", "xormul", "lik"):
            ck.check(f"control: {fam}'s step is found at <= 2 for every sampled member (width 6)",
                     all(m[2] is not None and m[2] <= 2 for m in raw(f"{fam}.W6.S1.D2")))
        ck.check("control: rotmuladd at k = 1 is one instruction, shorter than honest",
                 all(m[2] == 1 for m in raw("rotmuladd.W8.S1.all") if m[0].startswith("k1.")))
        ck.check("control: xormul at k = 1 is one instruction",
                 all(m[2] == 1 for m in raw("xormul.W8.S1.all") if m[0] == "k1"))
        ck.check("rotmuladd at k != 1 has no 1-instruction step (width 8, every member)",
                 all(m[2] is None for m in raw("rotmuladd.W8.S1.all") if not m[0].startswith("k1.")))
        ck.check("lik: no member's step is one instruction (width 8, every k)",
                 all(m[2] is None for m in raw("lik.W8.S1.all")))
        ck.check("arxm: no member's step is one instruction (width 8, every k and r)",
                 all(m[2] is None for m in raw("arxm.W8.S1.all.D1")))
        for name in ("arxm.W6.S1.D2",) + (("arxm.W8.S1.D2",) if full else ()):
            Wn = int(name.split(".")[1][1:])
            ck.check(f"arxm: every sampled member whose step is two instructions is degenerate, rotl(x, r) in "
                     f"{{0, 2^(W-1)}} ({name})",
                     all(m[2] is None or degenerate("arxm", m[0], Wn) for m in raw(name)),
                     f"{sum(m[2] is not None for m in raw(name))} of {len(raw(name))} found")
        for fam in ("rotmuladd", "xormul", "lik", "arxm"):
            ck.check(f"{fam}: every sampled two-step member shorter than honest has a degenerate step (width 6)",
                     all(m[2] is None or m[2] >= m[1] or degenerate(fam, m[0], 6) for m in raw(f"{fam}.W6.S2.D2")))
        ck.check("arxm at zero data: no sampled two-step member has a 2-instruction program (width 6)",
                 all(m[2] is None for m in raw("arxm.W6.S2.D2.z")))
        if full:
            for fam in ("arxm", "lik", "rand"):
                ck.check(f"{fam}: no sampled two-step member is shorter than honest at width 8",
                         all(m[2] is None or m[2] >= m[1] for m in raw(f"{fam}.W8.S2.D2")))

        if ck.mutations:
            res = search(exe, 8, "rotadd", 1, 1, 0, 1, "-a", "-x", "LEA.HI")
            ck.mutation("rotadd without LEA.HI is no longer one instruction",
                        all(m[2] != 1 or not m[3].startswith("LEA.HI") for m in res["members"])
                        and any(m[2] is None for m in res["members"]))

    for v in report["search"].values():
        v.pop("raw", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fold_dynamic.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
