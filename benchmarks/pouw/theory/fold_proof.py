"""FOLD-PROOF: the sm_120 sheet headline's `FoldOptFoldedAt` premise at the issue facet on DS-120, turned from a
conjecture into a per-word instruction lower bound checked over a stated program class, plus the argument and the named
residual premises for the class the search doesn't reach.

The headline `Accounting.SecurityProofs.TS.Sm120SheetHeadline` (#2482/#2791) rests on `FoldOptFoldedAt Sm120.model
(run x) ((1 - share)·Nw) lam` at its certificate π* (`Sm120.piStar`: the facet `.issue`, or `.issue` with a weight
moved onto `.smem`, A5b's shared-memory row). `FoldOptFoldedAt Z run folded lam` is
`(1 - Z.muF)·folded·Z.phiStar lam ≤ Z.G.price lam (nfold run)` (#2791 `Definitions/Accounting/Assumptions/TS.lean`). On the
data-sheet machine every facet row but the pipes is simple, and FOLD-OPT at π* follows from FOLD-OPT at `.issue`:

- at π* = w_issue·`.issue` + w_smem·`.smem` a class k costs w_issue + w_smem·smem(k)/32 ≥ w_issue a warp instruction,
  and the menu's classes (ALU, IMAD, UNI) hold no shared memory, so `phiStar π* = w_issue·phiStar .issue` and the
  premise at π* is the one at `.issue` scaled by w_issue;
- `Sheet.machine`'s `.issue` row is 1 for every one of the 122 classes (A2's catch-all), so
  `price (onFacet .issue) n = G.load n .issue = Σ_k 1·n(k) = Σ_k n(k)`: the fold's price at issue is the total count of
  fold warp instructions a unit, every class -- uniform, warp-wide, memory, bulk -- one issue slot.
- `phiStar lam = ⨅_φ demandPrice lam (G.load (counts folds[φ]))`; at `.issue` each route's demand is `Σ_k counts(k)`.
  `Sm120.folds = [[(0,1,16),(24,1,16)], [(105,4,1)]]`: route 0 (ALU 1/16 + IMAD 1/16, fx1 on 32 lanes) is 1/8 a word,
  route 1 (UNI 4) is 4 a word, so `phiStar .issue = 1/8`.
- `muF = 0` (`Encoding.DesignData.model`), `Nw = 128`.

So the premise is `folded·(1/8) ≤ Σ_k nfold(k)`: at full `folded = Nw = 128`, the fold issues **≥ 16 warp instructions a
unit**, which `flatFold = [(0,8,1),(24,8,1)]` (8 ALU + 8 IMAD) meets with equality. One warp instruction advances 32
lanes a word, so 16 warp instructions over 128 words is **4 lane-instructions a word** (`phiStar = 1/8 = 4/32`). fx1 is
exactly four: `d = p·m_q` (IMAD), `u = y ⊕ d ⊕ kx` (LOP3), `y' = u·a + z` (IMAD), `z' = rotl(u, r) + y'` (LEA.HI).

**The class statement.** "FOLD-OPT at the issue facet on DS-120" is exactly: no bit-exact fold of W words into the receipt
runs in fewer than W/8 warp instructions a unit -- equivalently, in fewer than 4 lane-instructions a word on the modelled
ISA. §1 checks this chain of equalities against `Sm120.lean`'s own data.

**The bound, and how the search extends it (the restriction lemma).** Fix every word's inputs but word i's. The
restriction of a straight-line fold program to word i computes `(y'_i, z'_i)` from `(y_i, z_i, w_i)`, every instruction
that doesn't read word i becoming a constant. If

- **(WIDTH-32)** every fold warp instruction's output depends on ≤ 32 folded words (one word a lane), and
- **(NO-MIX)** no fold lane-instruction depends on the words of two chains (so a 64-bit lane-instruction cannot advance
  two chains, and the restriction's constants are the keys alone, never another word's values),

then the restriction is a one-word program over word i's inputs and the keys, FOLD-MIN-4 (#2751,
r20261019-094858-36d3: fx1 needs ≥ 4 instructions at 32 bits, over fold_wide3.c's modelled ISA with the ten inputs as
its only operands) gives it ≥ 4 lane-instruction slots, and those slots don't overlap across words: ≥ 4W lane-slots a
unit = ≥ 4W/32 = W/8 warp instructions. Moves (SHFL, LDG/LDS, STG/STS, PRMT, MOV, SEL) relocate word-values and only
add slots. The class inherits FOLD-MIN-4's operands: an immediate other than its ten (a key product, say) is outside it.

§2 checks NO-MIX's cheapest counterexamples mechanically: no single two-word instruction of any form (IMAD.WIDE,
IADD.64, IMNMX.64; in-test, width 4), and no program of at most 3 instructions over both chains' inputs and the keys
ending in one (`--wide3`, width 4 and 32 bits), writes outputs of two chains (the cross pairs of
{y'_1,z'_1,y'_2,z'_2}). NO-MIX past three instructions is a premise; WIDTH-32 is the other, and §3 states what rests on
it and models the form that breaks it.

**What §3 argues (no search).**
- **Lane-shared** (SHFL, REDUX): SHFL is a move (it carries a lane's word to another lane, computing nothing). REDUX's
  per-lane effect is a reduction of one value with a constant (x ⊕ c, x + c, min/max), width-1 arithmetic that is not a
  keyed fold step; it also writes a warp-uniform result, so it can't carry 32 lanes' distinct outputs.
- **Uniform datapath** (UIMAD, ULOP3): a uniform instruction writes one warp-uniform word, width 1, and the fold's
  inputs vary by lane, so a uniform chain can't carry per-lane data (RT3-120-2: ptxas folds uniform-alone chains away).
  The menu already prices it: route 1 (UNI 4) is 4 warp instructions a word, 32 times route 0's 1/8, since a lane
  instruction folds 32 words an issue slot and a uniform one folds 1. It never undercuts.
- **Whole-block / bulk** (cp.reduce.async.bulk.xor = UBLKRED): violates WIDTH-32, so the mechanical bound doesn't cover
  it, and an argument doesn't either. The built route (RT3-120's mode 10, 8 chains a thread, a 16-warp block) is 21.18
  a unit on DS-120, above 16. But its per-word part is 3.75 lane-instructions a word (2 IMAD, LEA.HI, and a quarter
  each of STS.128, STG.E.128, LDG.E.128: the XOR is in the memory system, the moves are 4 words wide), under fx1's 4,
  and the 43 fixed instructions a set-warp (barriers, fences, branches) and thread 0's bodies amortize over the chains
  a thread. §3's model, mode 10's own counts with n chains a thread (n/8 logical lanes) and W warps a block, is
  `15 + (43 + bodies/W)/n` a unit. With mode 10's four static kk slots, shared memory keeps it above 16; with one kk
  slot reloaded each trip it is about 15.79 (n = 112, W = 3), under the floor. That route is not built. So FOLD-OPT at
  issue on DS-120 is checked for register-resident folds only, and WIDTH-32 is a load-bearing premise.

Run: `python fold_proof.py [--mutations] [--wide3 [--w32] | --merge VERDICT...] [--verdict PATH]`. Stdlib +
exact_kit + `fold_opt_fx` (and, with `--wide3`, `fold_wide3.c` built with gcc). `--wide3` (width 4) and
`--wide3 --w32` are CPU-hours and go to a CPU node (`research run`), each writing its verdict; `--merge` takes those two
verdicts and adds them to the fast battery, so the merged verdict is the whole result. The default in-test battery is
width 4, pure Python, about a second.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))

import fold_opt_fx as fx  # noqa: E402
from exact_kit.checks import SCHEMA, Checks  # noqa: E402

M32 = 0xFFFFFFFF
LANES = 32                       # a warp instruction advances 32 lanes, one word a lane
SM120 = "Security/Definitions/Accounting/Headline/Sm120.lean"
SHEET = "Security/Definitions/Accounting/TS/Sheet.lean"


def repo_root() -> Path:
    """The repository root (theory/ -> pouw/ -> benchmarks/ -> repo)."""
    return Path(__file__).resolve().parents[3]


# ---- §1: the class statement, parsed from Sm120.lean ---------------------------------------------------------------------

def _balanced(text: str, start: int) -> str:
    """The `[...]` list beginning at the first `[` at or after `start`, brackets balanced."""
    i = text.index("[", start)
    depth, j = 0, i
    while j < len(text):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
        j += 1
    raise ValueError("unbalanced list")


def _tuples(block: str) -> list[tuple[int, int, int]]:
    """Every `(a, b, c)` triple in `block`, in order."""
    return [(int(a), int(b), int(c)) for a, b, c in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", block)]


_PISTAR_TERM = re.compile(r"(\d+)\s*/\s*(\d+)\s*\*\s*onFacet\s*\(\s*SheetRow\.(\w+)\s*:\s*SheetRow\s+Pipe\s*\)\s*\w+")


def parse_pistar(text: str) -> dict[str, Fraction]:
    """`Sm120.piStar`'s weight on each sheet row: a facet (`def piStar : SheetRow Pipe := .issue`, weight 1) or a sum
    of `p / q * onFacet (SheetRow.r : SheetRow Pipe) r` terms (`def piStar : SheetRow Pipe → ℝ := fun r => …`)."""
    if m := re.search(r"def\s+piStar\s*:\s*SheetRow\s+Pipe\s*:=\s*\.(\w+)", text):
        return {m.group(1): Fraction(1)}
    m = re.search(r"def\s+piStar\s*:\s*SheetRow\s+Pipe\s*→\s*ℝ\s*:=\s*fun\s+\w+\s*=>(.*?)\n\s*\n", text, re.S)
    if m is None or _PISTAR_TERM.sub("", m.group(1)).replace("+", "").strip():
        raise ValueError("Sm120.piStar is neither a facet nor a sum of weighted facets")
    w: dict[str, Fraction] = {}
    for p, q, row in _PISTAR_TERM.findall(m.group(1)):
        w[row] = w.get(row, Fraction(0)) + Fraction(int(p), int(q))
    return w


def parse_sm120() -> dict:
    """`Sm120.folds` (as a list of routes, each a list of (class, num, den)), `Nw`, `piStar` (its weight on each sheet
    row), `flatFold`, `checks`, the classes, each class's shared-memory bytes (`smem`), and the `.issue` coefficient the
    machine prices every class at, read from the Lean sources."""
    text = (repo_root() / SM120).read_text(encoding="utf-8")
    inner = _balanced(text, text.index("folds :="))[1:-1]
    routes, depth, buf = [], 0, ""
    for ch in inner:
        if ch == "[":
            depth += 1
            if depth == 1:
                buf = ""
                continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                routes.append(_tuples(buf))
                continue
        if depth >= 1:
            buf += ch
    nw = int(re.search(r"Nw\s*:=\s*(\d+)", text).group(1))
    checks = int(re.search(r"checks\s*:\s*ℕ\s*:=\s*(\d+)", text).group(1))
    flatfold = _tuples(_balanced(text, re.search(r"def\s+flatFold\s*:[^=]*:=", text).end()))
    classes = re.findall(r'"(\w+)"', _balanced(text, re.search(r"def\s+classes\s*:[^=]*:=", text).end()))
    smem = [int(x) for x in re.findall(r"\d+", _balanced(text, re.search(r"\bsmem\s*:=", text).end()))]
    sheet = (repo_root() / SHEET).read_text(encoding="utf-8")
    issue_coeff = int(re.search(r"\|\s*\.issue\s*=>\s*(\d+)", sheet).group(1))
    smem_per = int(re.search(r"\|\s*\.smem\s*=>\s*S\.smem\s+k\s*/\s*(\d+)", sheet).group(1))
    return {"folds": routes, "Nw": nw, "checks": checks, "piStar": parse_pistar(text), "flatFold": flatfold,
            "classes": classes, "smem": smem, "issue_coeff": issue_coeff, "smem_per": smem_per}


def demand_at_issue(route: list[tuple[int, int, int]], issue_coeff: int) -> Fraction:
    """`demandPrice (onFacet .issue) (G.load (counts route))` = Σ_k issue_coeff·(num/den) over the route's classes."""
    return sum((Fraction(num, den) * issue_coeff for _k, num, den in route), Fraction(0))


def demand_at_pistar(route: list[tuple[int, int, int]], d: dict) -> Fraction:
    """The route's price at π*: Σ_k (num/den)·(w_issue·issue_coeff + w_smem·smem(k)/32) over its classes."""
    w = d["piStar"]
    return sum((Fraction(num, den) * (w.get("issue", 0) * d["issue_coeff"]
                                      + w.get("smem", 0) * Fraction(d["smem"][k], d["smem_per"]))
                for k, num, den in route), Fraction(0))


def pistar_ok(w: dict[str, Fraction]) -> bool:
    """π* weights `.issue`, and at most `.smem` besides, each weight ≥ 0: the rows the reduction to `.issue` covers."""
    return w.get("issue", 0) > 0 and set(w) <= {"issue", "smem"} and min(w.values()) >= 0


def reduces_to_issue(d: dict) -> bool:
    """Every class holds ≥ 0 bytes of shared memory and the menu's classes none, so at π* every class costs at least
    w_issue a warp instruction and each route exactly w_issue times its demand at `.issue`."""
    menu = {k for route in d["folds"] for k, _num, _den in route}
    return (len(d["smem"]) == len(d["classes"]) and min(d["smem"]) >= 0 and all(d["smem"][k] == 0 for k in menu)
            and d["smem_per"] > 0)


A5B_PISTAR = """noncomputable def piStar : SheetRow Pipe → ℝ := fun r =>
    3564 / 3761 * onFacet (SheetRow.issue : SheetRow Pipe) r +
    197 / 3761 * onFacet (SheetRow.smem : SheetRow Pipe) r

"""


def class_statement(ck: Checks, d: dict) -> Fraction:
    """§1: the class statement, as a chain of equalities against Sm120.lean's own data. Returns φ*(issue)."""
    ck.check("Sm120.folds is the menu [[ALU 1/16, IMAD 1/16], [UNI 4]] (route 0 is fx1 on 32 lanes, route 1 the "
             "uniform fold)", d["folds"] == [[(0, 1, 16), (24, 1, 16)], [(105, 4, 1)]], d["folds"])
    ck.check("Sm120.Nw = 128 words a unit", d["Nw"] == 128, d["Nw"])
    w = d["piStar"]
    ck.check("Sm120.piStar weights .issue and at most .smem besides, each weight ≥ 0 (a facet is .issue alone)",
             pistar_ok(w), {r: str(v) for r, v in w.items()})
    ck.check("piStar is read as a facet or term by term as a weighting (A5b's 3564/3761 on .issue, 197/3761 on .smem)",
             parse_pistar("def piStar : SheetRow Pipe := .issue\n") == {"issue": 1}
             and parse_pistar(A5B_PISTAR) == {"issue": Fraction(3564, 3761), "smem": Fraction(197, 3761)})
    ck.check("Sheet.machine prices every class 1 slot at .issue (A2's catch-all), so price (onFacet .issue) n = Σ_k n(k)",
             d["issue_coeff"] == 1, d["issue_coeff"])
    menu = sorted({k for route in d["folds"] for k, _num, _den in route})
    ck.check("at π* a class k costs w_issue + w_smem·smem(k)/32 ≥ w_issue a warp instruction (smem ≥ 0 for each of the "
             "sheet's classes), and the menu's classes (ALU, IMAD, UNI) hold no shared memory, so price π* n ≥ "
             "w_issue·Σ_k n(k) and phiStar π* = w_issue·phiStar .issue: FOLD-OPT at .issue gives it at π*",
             reduces_to_issue(d), {d["classes"][k]: d["smem"][k] for k in menu})
    if ck.mutations:
        ck.mutation("a π* with weight on a pipe row is refused: the reduction to .issue doesn't cover it",
                    not pistar_ok({**w, "lop": Fraction(1, 2)}))
        uni = d["folds"][1][0][0]
        ck.mutation("a menu class holding shared memory (UNI at 16 B) breaks the reduction to .issue",
                    not reduces_to_issue({**d, "smem": [*d["smem"][:uni], 16, *d["smem"][uni + 1:]]}))
        try:
            parse_pistar(A5B_PISTAR.replace("+", "-", 1))
            refused = False
        except ValueError:
            refused = True
        ck.mutation("a π* that is no sum of weighted facets is refused, not read in part", refused)

    demands = [demand_at_issue(r, d["issue_coeff"]) for r in d["folds"]]
    ck.check("route 0 (ALU 1/16 + IMAD 1/16) costs 1/8 warp instruction a word at issue", demands[0] == Fraction(1, 8),
             demands[0])
    ck.check("route 1 (UNI 4) costs 4 a word at issue", demands[1] == Fraction(4), demands[1])
    phi = min(demands)
    ck.check("phiStar(.issue) = min over routes = 1/8 warp instruction a word", phi == Fraction(1, 8), phi)
    phi_pi = min(demand_at_pistar(r, d) for r in d["folds"])
    ck.check("phiStar π* = w_issue/8, and flatFold (8 ALU + 8 IMAD, no shared memory) costs Nw·phiStar π* = 16·w_issue",
             phi_pi == w["issue"] * phi and demand_at_pistar(d["flatFold"], d) == d["Nw"] * phi_pi, str(phi_pi))

    folded = d["Nw"]                                        # full folded = Nw (share -> 0)
    rhs_floor = folded * phi                                # (1 - muF)·folded·phiStar, muF = 0
    ck.check("muF = 0, so FoldOptFoldedAt at full folded = Nw is Nw·phiStar = 16 warp instructions a unit ≤ Σ_k nfold(k)",
             rhs_floor == Fraction(16), rhs_floor)
    flat_total = sum(Fraction(num, den) for _k, num, den in d["flatFold"])
    ck.check("flatFold (8 ALU + 8 IMAD) is 16 warp instructions a unit: ♭'s fold meets the floor with equality",
             flat_total == Fraction(16) and d["flatFold"] == [(0, 8, 1), (24, 8, 1)], d["flatFold"])

    lane = rhs_floor * LANES / folded                       # warp instructions · 32 lanes / words
    ck.check("the floor is 4 lane-instructions a word (16 warp instructions · 32 lanes / 128 words); phiStar = 1/8 = 4/32",
             lane == Fraction(4) and phi == Fraction(4, LANES), lane)
    return phi


# ---- §2: NO-MIX, the two-chain / wide-form search ------------------------------------------------------------------------

def two_chain_case(width: int, npts: int, seed: int = 0xF0CA):
    """fx1 for two independent words sharing one position's keys, at `width` bits over `npts` points: the operand pool
    (both chains' inputs y1,z1,w1,y2,z2,w2 and the shared constants kx, a, m1 and inverses) and the four outputs
    y1',z1',y2',z2'. The cheater knows the keys; the two words are distinct commitments."""
    mask = (1 << width) - 1
    # random.Random, not fold_opt_fx.lcg: an LCG mod 2^32 has period 2^w in its low w bits, so at small width its
    # consecutive columns alias (chain 2's inputs would equal chain 1's).
    rng = random.Random(seed)
    cols = {}
    for nm in ("y1", "z1", "w1", "y2", "z2", "w2"):
        cols[nm] = [rng.getrandbits(width) for _ in range(npts)]
    a = (fx.key_mul(0, 0) & mask) | 1
    kx = fx.key_xor(0, 0) & mask
    m1 = (fx.key_mul(1, 0) & mask) | 1
    r = 1 if width <= 8 else fx.rot_of(0)
    for nm, cv in (("kx", kx), ("a", a), ("m1", m1), ("0", 0), ("1", 1)):
        cols[nm] = [cv] * npts

    def rot(x, n):
        n %= width
        return ((x << n) | (x >> (width - n))) & mask if n else x & mask

    def word(y, z, w):
        dp = (w * m1) & mask
        u = (y ^ dp ^ kx) & mask
        yp = (u * a + z) & mask
        return yp, (rot(u, r) + yp) & mask

    out = {}
    for ch in (1, 2):
        yp, zp = [], []
        for i in range(npts):
            a1, b1 = word(cols[f"y{ch}"][i], cols[f"z{ch}"][i], cols[f"w{ch}"][i])
            yp.append(a1)
            zp.append(b1)
        out[f"y{ch}"] = yp
        out[f"z{ch}"] = zp
    return cols, out, r


CROSS = (("y1", "y2"), ("y1", "z2"), ("z1", "y2"), ("z1", "z2"))


def no_mix_pure(ck: Checks, width: int = 4, npts: int = 8) -> None:
    """§2 in-test: NO-MIX at `width` bits, pure Python. For each cross pair of outputs from distinct chains, neither
    output is a single instruction of the two-chain pool, and no single two-word instruction of any form (IMAD.WIDE and
    .U32, IADD.64 and IMNMX.64 in every sign/variant) writes the pair as its two words, either way round. Width 4 is the
    small model (fx1 there rotates by 1); a program reaching the last two-word instruction through up to two more, and
    32 bits, go to `--wide3` (fold_wide3.c, a CPU node)."""
    cols, out, r = two_chain_case(width, npts)
    pool = {k: v for k, v in cols.items()}
    labels = {"y1": out["y1"], "z1": out["z1"], "y2": out["y2"], "z2": out["z2"]}

    # positive control: an honest 8-instruction program (fx1 twice) reproduces all four outputs.
    ck.check("two chains: honest 8-instruction program reproduces all four outputs (y'_1, z'_1, y'_2, z'_2)",
             all(labels[k] == out[k] for k in labels))
    # non-vacuity: the two chains are independent here (six distinct input columns, four distinct outputs), so a cross
    # pair can't be met by one chain's value standing in for the other's.
    ins = [tuple(cols[k]) for k in ("y1", "z1", "w1", "y2", "z2", "w2")]
    outs = [tuple(out[k]) for k in ("y1", "z1", "y2", "z2")]
    ck.check(f"two chains, width {width}: the six input columns and the four outputs are pairwise distinct",
             len(set(ins)) == 6 and len(set(outs)) == 4)

    pairs = fx._two_word_pairs(width, pool)
    for t1n, t2n in CROSS:
        t1, t2 = tuple(out[t1n]), tuple(out[t2n])
        s1 = fx.width_single_instruction(width, list(t1), pool)
        s2 = fx.width_single_instruction(width, list(t2), pool)
        both = pairs.get((t1, t2)) or pairs.get((t2, t1))
        ck.check(f"two chains, width {width}: no single two-word instruction (IMAD.WIDE, IADD.64, IMNMX.64, every form) "
                 f"writes the cross pair ({t1n}', {t2n}') of distinct chains, and neither is a single instruction",
                 s1 is None and s2 is None and both is None, (s1, s2, both))

    if ck.mutations:
        # the pair lookup is not vacuous: a pair one two-word instruction of the pool really writes is found.
        lh = [fx.wide(cols["y1"][i], cols["a"][i], cols["z2"][i], cols["w2"][i], True, width) for i in range(npts)]
        lo, hi = tuple(x[0] for x in lh), tuple(x[1] for x in lh)
        ck.mutation("the two-word pair lookup is not vacuous: an IMAD.WIDE of the pool's own words is found",
                    pairs.get((lo, hi)) is not None, pairs.get((lo, hi)))
        a64 = [fx.add64(cols["y1"][i], cols["y2"][i], cols["z1"][i], cols["z2"][i], 0, width) for i in range(npts)]
        lo, hi = tuple(x[0] for x in a64), tuple(x[1] for x in a64)
        ck.mutation("and so is an IADD.64 straddling the two chains (y2:y1) + (z2:z1), its carry coupling them",
                    pairs.get((lo, hi)) is not None, pairs.get((lo, hi)))
        # a pair of two single-instruction values (not a fold) is caught as a single instruction each.
        mask = (1 << width) - 1
        easy1 = [(cols["y1"][i] ^ cols["kx"][i]) & mask for i in range(npts)]
        easy2 = [(cols["y2"][i] * cols["a"][i] + cols["z2"][i]) & mask for i in range(npts)]
        ck.mutation("a cross pair of single-instruction values is caught (y1^kx, y2*a+z2 are each single instructions)",
                    fx.width_single_instruction(width, easy1, pool) is not None
                    and fx.width_single_instruction(width, easy2, pool) is not None)


def no_mix_wide3(ck: Checks, width: int) -> None:
    """§2 full menu via `fold_wide3.c` (IMAD.WIDE, IADD.64, IMNMX.64, SEL.64 writing a pair): for each cross pair, no
    program of at most 3 instructions over the nine inputs (both chains' words and the shared keys) writes it as a last
    two-word instruction (`found`), nor a last SEL.64 (`sel`), and neither word is a word of a program of at most 2
    (`le2`), with `fold_opt_fx`'s controls found. At `width = 32` this is the 32-bit NO-MIX; it is CPU-hours (a CPU
    node)."""
    cols, out, r = two_chain_case(width, 2 if width >= 16 else 12)
    pool = {k: cols[k] for k in ("y1", "z1", "w1", "y2", "z2", "w2", "kx", "a", "m1")}
    src = repo_root() / "benchmarks/pouw/theory/fold_wide3.c"
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "fold_wide3"
        subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(src)], check=True)

        # non-vacuity: fold_wide3.c's first-instruction enumeration over this pool is fold_opt_fx's own.
        three = {k: pool[k] for k in ("y1", "z1", "a")}
        got = fx.width_wide3(exe, width, three, {"c": (pool["y1"], pool["z1"])})["c"]
        ck.check(f"width {width}: fold_wide3.c's distinct first instructions over three inputs are fold_opt_fx.py's",
                 got["firsts"] == fx._new_word_sets(width, three), (got["firsts"], fx._new_word_sets(width, three)))

        # every two-word form and every way to a last one, over chain 1's six words, is found (fold_wide32's controls
        # at 32 bits, over six inputs as there: fold_wide3.c takes at most 16 jobs a pass).
        six = {"y": pool["y1"], "z": pool["z1"], "w": pool["w1"], "kx": pool["kx"], "a": pool["a"], "m1": pool["m1"]}
        if width == 32:
            import fold_wide32 as w32
            ctl = w32.controls(six)
        else:
            ctl = dict(fx.wide3_controls(width, six))
        kind = {"ctl": "found", "sel": "sel", "le2": "le2"}
        got = fx.width_wide3(exe, width, six, ctl)
        for lab in ctl:
            k = kind[lab.split("_")[0]]
            wit = [x for x in got[lab]["witnesses"] if x[0] == {"found": "F", "sel": "S", "le2": "L"}[k]]
            ck.check(f"width {width}: fold_wide3.c finds its control {lab} ({k}) over chain 1's six inputs",
                     got[lab][k] > 0, wit[:1])

        # the nine-input pass finds a pair a program mixing the two chains writes (LOP3 0x96(y1, w2, kx), IMAD with z2,
        # a last IMAD.WIDE), so the cross pairs' zeros below are the search's, not a blind spot for two chains.
        mixed = {"y": pool["y1"], "z": pool["z2"], "w": pool["w2"], "kx": pool["kx"], "a": pool["a"], "m1": pool["m1"]}
        mix = {"ctl_mix_lop3_imad": fx.wide3_controls(width, mixed)["ctl_lop3_imad"]}
        jobs = {**{f"x_{t1n}_{t2n}": (out[t1n], out[t2n]) for t1n, t2n in CROSS}, **mix}
        res = fx.width_wide3(exe, width, pool, jobs)
        ck.check(f"width {width}, two-chain pool (nine inputs): fold_wide3.c finds a two-chain program's last IMAD.WIDE "
                 "pair (LOP3 over y1, w2; IMAD with z2)", res["ctl_mix_lop3_imad"]["found"] > 0,
                 res["ctl_mix_lop3_imad"]["witnesses"][:1])
        for t1n, t2n in CROSS:
            hit = res[f"x_{t1n}_{t2n}"]
            seen = {k: hit[k] for k in ("found", "sel", "le2", "firsts", "prefixes")}
            ck.check(f"two chains, width {width}: {CROSS_TAG} ({t1n}', {t2n}') as a last two-word instruction "
                     "(IMAD.WIDE, IADD.64, IMNMX.64) or a last SEL.64, and neither is a word of a program of at most 2 "
                     "(chain 2's words as operands too)",
                     hit["found"] == 0 and hit["sel"] == 0 and hit["le2"] == 0, seen)


# ---- §2: the records' merge ----------------------------------------------------------------------------------------------

CROSS_TAG = "no program of at most 3 instructions writes the cross pair"
CTL_TAG = "fold_wide3.c finds"
CONTROLS = 14                # wide3's 13 over chain 1's six inputs and the two-chain one over the nine


def record_ok(doc: dict, width: int) -> tuple[bool, dict]:
    """Whether `doc` is a passing `--wide3` verdict of this script at `width`: its four cross-pair checks and its
    controls all passed. The detail names the script digest it ran at."""
    checks, args = doc.get("checks", []), doc.get("args", [])
    cross = [c for c in checks if CROSS_TAG in c["name"] and f"width {width}:" in c["name"]]
    ctl = [c for c in checks if CTL_TAG in c["name"] and f"width {width}" in c["name"]]
    ok = (doc.get("schema") == SCHEMA and doc.get("script", {}).get("name") == Path(__file__).name
          and doc.get("ok") is True and "--wide3" in args and ("--w32" in args) == (width == 32)
          and len(cross) == len(CROSS) and all(c["ok"] for c in cross)
          and len(ctl) >= CONTROLS and all(c["ok"] for c in ctl))
    return ok, {"sha256": doc.get("script", {}).get("sha256", "")[:12], "cross": len(cross), "controls": len(ctl),
                "counts": doc.get("counts")}


def _synthetic_record(width: int) -> dict:
    checks = [{"name": f"two chains, width {width}: {CROSS_TAG} ({a}', {b}')", "ok": True, "detail": None}
              for a, b in CROSS]
    checks += [{"name": f"width {width}: {CTL_TAG} its control c{i}", "ok": True, "detail": None}
               for i in range(CONTROLS)]
    return {"schema": SCHEMA, "script": {"name": Path(__file__).name, "sha256": "0" * 64}, "ok": True,
            "args": ["--wide3", *(["--w32"] if width == 32 else [])], "checks": checks, "counts": {}}


def merge_selftest(ck: Checks) -> None:
    """The merge's predicate, on records of the shape `--wide3` writes: it accepts a passing one at its own width only,
    and refuses a failed cross pair or a missing control."""
    ck.check("the records' merge accepts a passing --wide3 record at its own width (4 and 32)",
             record_ok(_synthetic_record(4), 4)[0] and record_ok(_synthetic_record(32), 32)[0])
    if ck.mutations:
        bad = _synthetic_record(32)
        bad["checks"][1]["ok"] = False
        ck.mutation("the merge refuses a record with a cross pair found", not record_ok(bad, 32)[0])
        ck.mutation("the merge refuses a width-4 record offered as the 32-bit one", not record_ok(_synthetic_record(4), 32)[0])
        short = _synthetic_record(4)
        short["checks"].pop()
        ck.mutation("the merge refuses a record missing a control", not record_ok(short, 4)[0])


def merge_records(ck: Checks, paths: list[str]) -> None:
    """The records' merge: the NO-MIX runs' `--wide3` verdicts, one at width 4 and one at 32 bits, each `record_ok` at
    its width. With §1 and §3's fast battery, the merged verdict is the whole result."""
    docs = [(Path(p), json.loads(Path(p).read_text(encoding="utf-8"))) for p in paths]
    for width in (4, 32):
        mine = [(p, d) for p, d in docs if ("--w32" in d.get("args", [])) == (width == 32)]
        ok, detail = record_ok(mine[0][1], width) if len(mine) == 1 else (False, {})
        ck.check(f"NO-MIX at width {width}, from its run's record: every check passed, none of the four cross pairs is "
                 "written by a program of at most 3 instructions over the nine inputs, every control found", ok,
                 {"records": [p.name for p, _ in mine], **detail})


# ---- §3: the per-unit bound (restriction-lemma counting) and the residual premises ---------------------------------------

def per_unit_bound(ck: Checks, phi: Fraction) -> None:
    """§3: the counting from FOLD-MIN-4's ≥ 4 a word to ≥ W/8 warp instructions a unit, and what mixing would cost. The
    numbers are checked; WIDTH-32 and NO-MIX are the premises the argument rests on (NO-MIX checked in §2 up to three
    instructions)."""
    W, per_word = 128, 4                                    # FOLD-MIN-4: ≥ 4 lane-instruction slots a word at 32 bits
    slots = W * per_word                                    # lane-slots a unit, no overlap across words
    warp_instrs = Fraction(slots, LANES)
    ck.check("counting: 4 lane-slots a word · 128 words / 32 lanes = 16 warp instructions a unit = Nw·phiStar",
             warp_instrs == W * phi == Fraction(16), warp_instrs)
    # what a violation would cost: one 64-bit lane-instruction advancing two chains halves the count (the two-chain
    # break NO-MIX rules out); a bulk op folding a whole block breaks WIDTH-32 (§3's model).
    mixed = Fraction(slots, LANES * 2)
    ck.check("if NO-MIX failed (a 64-bit lane-instruction advanced two chains), the count would halve to 8 a unit, below "
             "the floor -- so NO-MIX is load-bearing; §2 checks it up to three instructions", mixed == Fraction(8), mixed)
    # the margin: the sheet's bound breaks only below 16.99993 issue slots a unit (RT2-120, cpw_floor.py), and an
    # accepted unit is IMMA 1 + the fold + CP-W's checkpoint, whose egress floor is 0.25 a unit.
    unit = 1 + W * phi + Fraction(1, 4)
    ck.check("the unit floor: IMMA 1 + the fold's 16 + RT2-120's 0.25 egress floor = 17.25, 0.25 slots a unit above the "
             "break line 16.99993, so a fold under 15.74993 a unit would break the headline's number",
             unit == Fraction(69, 4) and unit - Fraction(1699993, 100000) == Fraction(25007, 100000), unit)


def residuals(ck: Checks, d: dict) -> None:
    """§3: the lane-shared, uniform and whole-block arguments, as structural checks of the fold's shape (no search)."""
    # SHFL moves a lane's word, computing nothing; a move cannot be a fold step (it has no arithmetic). fx1 is a
    # bijection of the word, so a move that lost information couldn't stand in for it either.
    a, kx, r = fx.key_mul(0, 0), fx.key_xor(0, 0), fx.rot_of(0)
    g = fx.lcg(0xA11CE)
    ws = [next(g) for _ in range(256)]
    ck.check("lane-shared: fx1 is a bijection of the word (256 distinct words, 256 distinct d'): SHFL and the moves "
             "relocate words, they do not fold", len(set(ws)) == 256 and len({fx.fx_chain(w, 1) for w in ws}) == 256)
    # the fold's per-word output varies by lane, so the uniform datapath (one warp-uniform word an instruction) and a
    # REDUX (one warp-uniform reduction) cannot carry 32 lanes' distinct outputs.
    pts = [(next(g), next(g), next(g)) for _ in range(LANES)]
    outs = {fx.fx_word(y, z, w, 1, a, kx, r)[0] for (y, z, w) in pts}
    ck.check("uniform/REDUX: the fold's y' varies by lane (32 distinct outputs over 32 lanes), so a warp-uniform "
             "instruction, which writes one word, cannot write them", len(outs) == LANES, len(outs))
    # a uniform instruction folds one warp-uniform word, so route 1 (UNI 4) is 4 warp instructions a word: 32 times route
    # 0's 1/8, because a lane instruction folds 32 words an issue slot and a uniform one folds 1.
    r0, r1 = (demand_at_issue(rt, d["issue_coeff"]) for rt in d["folds"])
    ck.check("uniform: route 1 (UNI 4 a word) is 32 times route 0 (1/8 a word) at issue -- width 1 against 32 -- so the "
             "uniform datapath never undercuts the lane fold", r1 == LANES * r0, (r1, r0))



# ---- §3: whole-block folds (bulk reduce-xor), an instruction-count model ------------------------------------------------

# cpw_routes mode 10 (`cpwx<10>`, r20261019-192718-5d46, RT3-120): a set-warp is 4 trips, each one step of a thread's
# n = 8 chains. Its SASS a set-warp at DS-120's issue row (every warp instruction one slot):
MODE10_SCALING = {"IMAD": 64, "LEA.HI": 32, "STS.128": 8, "STG.E.128": 8, "LDG.E.128": 8}   # 15 a chain a set-warp
MODE10_FIXED = {"MOV": 7, "LEA": 1, "uniform": 5, "BAR.SYNC": 8, "MEMBAR": 4, "FENCE": 4, "BSSY": 4, "BSYNC": 4,
                "branch": 5, "LDC": 1}                                                         # per set-warp, any n
MODE10_BODIES = 103          # thread 0's bodies a set a block (UBLKRED, UTMACMDFLUSH, ELECT loops), from 21.18·8·16 − 163·16
MODE10_HEADLINE = Fraction(2118, 100)                  # RT3-120's 21.18 a unit on DS-120, the bodies included
REGS, SMEM = 255, 99 * 1024                            # sm_120: registers a thread, dynamic shared memory a block
REG_BASE = 25                # registers beside the 2n chain words (one trip's MMA words and products, addresses): estimate
RELOAD_BODIES = 136          # bodies a set a block with the kk slot reloaded each trip (103 + a bulk copy and its
                             # mbarrier wait, about 8 a trip): an estimate, and the check below says how far it may be off


def bulk_fold(n: int, warps: int, bodies: int) -> Fraction:
    """The model: mode 10 with n chains a thread (n/8 logical lanes a thread) in a block of `warps` warps. A set-warp
    folds n units; it issues 15 a chain, 43 fixed, and its share of thread 0's bodies, so a unit costs
    15 + (43 + bodies/warps)/n warp instructions at issue."""
    per_chain = Fraction(sum(MODE10_SCALING.values()), 8)
    return per_chain + (sum(MODE10_FIXED.values()) + Fraction(bodies, warps)) / n


def bulk_min(slots: int, bodies: int) -> tuple[Fraction, int, int]:
    """The model's cheapest feasible point: n a multiple of 8 with 2n + REG_BASE registers, and `slots` shared slots of
    n words a thread (the p·fk slot, and one or four kk slots) in a block's shared memory."""
    best = None
    for n in range(8, REGS, 8):
        if 2 * n + REG_BASE > REGS:
            break
        for warps in range(1, 33):
            if slots * n * 4 * LANES * warps > SMEM:
                break
            c = bulk_fold(n, warps, bodies)
            if best is None or c < best[0]:
                best = (c, n, warps)
    return best


def whole_block(ck: Checks) -> None:
    """§3: whole-block folds break WIDTH-32 (one UBLKRED XORs a whole block's words), so §2's bound doesn't reach them.
    The model below is mode 10's own SASS counts with the chains a thread and the block size as parameters. It is a
    model, not a built program: it says WIDTH-32 is load-bearing, not that the sheet is broken."""
    scaling, fixed = sum(MODE10_SCALING.values()), sum(MODE10_FIXED.values())
    ck.check("whole-block: mode 10's 163 warp instructions a set-warp are 120 that scale with the chains (64 IMAD, 32 "
             "LEA.HI, 8 each of STS.128, STG.E.128, LDG.E.128: 3.75 a word, under fx1's 4) and 43 that don't",
             (scaling, fixed) == (120, 43), (scaling, fixed))
    ck.check("whole-block: the built route (mode 10, 8 chains a thread, 16 warps) stays above the floor: 163/8 = 20.4 a "
             "unit every warp, and the model gives back RT3-120's 21.18 with its bodies",
             Fraction(scaling + fixed, 8) > 16 and round(bulk_fold(8, 16, MODE10_BODIES), 2) == MODE10_HEADLINE,
             (Fraction(scaling + fixed, 8), float(bulk_fold(8, 16, MODE10_BODIES))))
    st = bulk_min(5, MODE10_BODIES)
    ck.check("whole-block model, mode 10's layout (four static kk slots): its cheapest feasible point stays above 16 "
             "(shared memory caps the chains a block)", st[0] > 16, (float(st[0]), f"n={st[1]}", f"warps={st[2]}"))
    rl = bulk_min(2, RELOAD_BODIES)
    ck.check("whole-block MODEL, unbuilt: with the kk slot reloaded each trip (one slot, so 2n words a thread of shared "
             "memory), the cheapest feasible point is below 16 a unit -- FOLD-OPT at issue on DS-120 is not checked for "
             "whole-block folds, and WIDTH-32 is load-bearing", rl[0] < 16,
             (float(rl[0]), f"n={rl[1]}", f"warps={rl[2]}"))
    slack = rl[2] * (rl[1] - fixed)
    ck.check("whole-block model: at that point the route stays below 16 for bodies up to warps·(n − 43) a set a block, "
             "twice mode 10's 103, so the estimate of the reload's cost isn't what decides it",
             bulk_fold(rl[1], rl[2], slack - 1) < 16 <= bulk_fold(rl[1], rl[2], slack) and slack >= 2 * MODE10_BODIES,
             slack)
    unit = 1 + rl[0] + Fraction(1, 4)
    ck.check("whole-block model, the headline: IMMA 1 + the modelled fold + RT2-120's 0.25 egress floor is above the break "
             "line 16.99993 by under 0.05 slots a unit (the 0.25 margin shrinks to 0.04; the egress floor's own argument "
             "isn't re-examined for a route whose y words already leave registers every trip)",
             Fraction(1699993, 100000) < unit < Fraction(1699993, 100000) + Fraction(5, 100), float(unit))


def main() -> int:
    ck = Checks()
    d = parse_sm120()
    phi = class_statement(ck, d)
    if "--wide3" in ck.args:
        no_mix_wide3(ck, 32 if "--w32" in ck.args else 4)
    else:
        no_mix_pure(ck, width=4)
        merge_selftest(ck)
        if "--merge" in ck.args:
            merge_records(ck, [a for a in ck.args if a != "--merge"])
    per_unit_bound(ck, phi)
    residuals(ck, d)
    whole_block(ck)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
