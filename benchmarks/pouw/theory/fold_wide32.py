"""FOLD-OPT's two-word search at the full 32-bit word, for the forms width 4 doesn't carry.

`fold_opt_fx.py --wide3` runs `fold_wide3.c` at width 4. Reducing mod 2^4 commutes with LOP3, IADD3, IMAD and LEA, so
there it covers 32 bits for programs of those forms; it doesn't commute with IMAD.HI, LEA.HI's rotate, IMAD.WIDE's high
word, IADD.64's carry into the high word or IMNMX.64's comparison, nor with fx1's own rotation, and there the 32-bit
claim was an argument by hand. Here the same search runs at W = 32: each value is the full word at two random points
(W*NP <= 64), over `fold_opt_fx.py`'s ten inputs (y, z, w, kx, a, m1, a^-1, m1^-1, 0, 1) with the design's keys (fgemm.cu's
key_mul(0,0), key_xor(0,0), key_mul(1,0), r = rot_of(0) = 6). A program that writes fx1's (y', z') for every input writes
it at the two points, so none found means none in the modelled ISA at 32 bits; a hit would only be a candidate.

Checks:
1. Width 32 enumerates what `fold_opt_fx.py` does: `fold_wide3.c`'s distinct first instructions over three inputs are
   `fold_opt_fx._new_word_sets`'s.
2. The search finds what it must at width 32, over six inputs (y, z, w, kx, a, m1): every control of
   `fold_opt_fx.wide3_controls` (each two-word form first and last, each way a last instruction is reached, a word of a
   second instruction), and a SEL.64 control built here, since `wide3_controls`' own needs a pattern no IMNMX.64 form
   picks, which two points don't always have.
3. fx1 at 32 bits: neither y' nor z' is a word of a program of at most 2 instructions, no program of at most 3 writes
   (y', z') as its last instruction's two words (IMAD.WIDE, IADD.64, IMNMX.64), nor a last SEL.64 under any predicate.
   With 2, fx1 needs at least 4 instructions in the modelled ISA at 32 bits (`fold_opt_fx.py`'s counting argument).

Mutations (`--mutations`): built without each part of the search (`-DDROP=k`), `fold_wide3.c` misses the width-32
controls only that part reaches.

The modelled ISA is `fold_wide3.c`'s (its header): no SHF, PRMT, IDP, POPC, FLO, BREV or BMSK form, no predicate but a
last SEL.64's, and only the inputs as operands (no other immediate). Run: `python fold_wide32.py [--quick] [--mutations]
[--verdict PATH]`. The controls take about 40 CPU-minutes; fx1 over ten inputs about 4 CPU-hours (`--quick`: over
the six, 3 CPU-minutes), so a full run goes to a CPU node (`research run`). Builds `fold_wide3.c` with gcc; OpenMP takes
OMP_NUM_THREADS.
"""
from __future__ import annotations

import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))

import fold_opt_fx as fx  # noqa: E402
from exact_kit.checks import Checks  # noqa: E402

W = 32
NPT = 2
MASK = (1 << W) - 1
COUNT = {"ctl": "found", "sel": "sel", "le2": "le2"}
PARTS = ((1, "a last IADD.64", ("ctl_add64_pre", "ctl_add64_live", "ctl_mnmx_first")),
         (2, "a last IMNMX.64 or SEL.64 over a second-instruction word", ("ctl_mnmx_last", "sel_pair")),
         (3, "a last IMAD.WIDE with p or q a second-instruction word", ("ctl_lop3_imad", "ctl_wide_leahi", "ctl_lea_maj")),
         (4, "a last IMAD.WIDE with p and q before it", ("ctl_pre_chi", "ctl_pre_clo")),
         (5, "a second instruction's words in le2", ("le2_add64_hi",)),
         (6, "IADD.64 and IMNMX.64 before the last instruction", ("ctl_add64_first", "ctl_mnmx_first")))


def rotl(x: int, r: int) -> int:
    r %= W
    return ((x << r) | (x >> (W - r))) & MASK if r else x


def points(seed: int):
    g = fx.lcg(seed)
    return [[next(g) for _ in range(NPT)] for _ in range(3)]


def fx1_case(seed: int = 0xBEEF):
    """fx1's ten inputs at two points and its (y', z') there."""
    y, z, w = points(seed)
    a, kx, r, m1 = fx.key_mul(0, 0), fx.key_xor(0, 0), fx.rot_of(0), fx.key_mul(1, 0)
    out = [fx.fx_word(y[i], z[i], w[i], 1, a, kx, r) for i in range(NPT)]
    yp, zp = [o[0] for o in out], [o[1] for o in out]
    const = {"kx": kx, "a": a, "m1": m1, "ainv": fx.modinv(a), "m1inv": fx.modinv(m1), "0": 0, "1": 1}
    ins = {"y": y, "z": z, "w": w, **{k: [v] * NPT for k, v in const.items()}}
    return ins, yp, zp, r


def sel_pair(cols):
    """A SEL.64 target over two points: two pairs of words, one reading a second instruction's word (v2 = IMAD(v1, m1, z)
    after v1 = LOP3 0x96(y, w, kx)), taking one pair at point 0 and the other at point 1, where no IMNMX.64 form does."""
    y, z, w, kx, a, m1 = (cols[k] for k in ("y", "z", "w", "kx", "a", "m1"))
    v1 = [y[i] ^ w[i] ^ kx[i] for i in range(NPT)]
    v2 = [(v1[i] * m1[i] + z[i]) & MASK for i in range(NPT)]
    words = {"y": y, "z": z, "w": w, "kx": kx, "a": a, "m1": m1, "v1": v1, "v2": v2}
    pairs = [(lo, hi) for lo in words for hi in words]
    for pa in pairs:
        if "v2" not in pa:
            continue
        for pb in pairs:
            A = [(words[pa[0]][i], words[pa[1]][i]) for i in range(NPT)]
            B = [(words[pb[0]][i], words[pb[1]][i]) for i in range(NPT)]
            t = [A[0], B[1]]
            if A[0] == B[0] or A[1] == B[1]:
                continue
            if not any(all(fx.mnmx64(*A[i], *B[i], v, W) == t[i] for i in range(NPT)) for v in range(4)):
                return [x[0] for x in t], [x[1] for x in t]
    raise AssertionError("no SEL.64 control over these words")


def mnmx_first(cols):
    """`wide3_controls`' ctl_mnmx_first (IMNMX.64 first, a LEA of its low word, IADD.64 last), over two pairs of inputs
    the IMNMX.64 picks differently at the two points: at two points the original's can pick one pair at both, and then
    its words are inputs and no IMNMX.64 is needed."""
    have = {tuple(v) for v in cols.values()}
    for al, ah, bl, bh in itertools.product(("y", "z", "w", "kx"), repeat=4):
        for v in range(4):
            x1 = [fx.mnmx64(cols[al][i], cols[ah][i], cols[bl][i], cols[bh][i], v, W) for i in range(NPT)]
            lo, hi = tuple(x[0] for x in x1), tuple(x[1] for x in x1)
            if lo in have or hi in have:
                continue
            v2 = [((lo[i] << 2) + cols["a"][i]) & MASK for i in range(NPT)]
            t = [fx.add64(v2[i], hi[i], cols["m1"][i], cols["y"][i], 0, W) for i in range(NPT)]
            return [x[0] for x in t], [x[1] for x in t]
    raise AssertionError("no IMNMX.64 over these inputs picks differently at the two points")


def controls(cols):
    jobs = {k: v for k, v in fx.wide3_controls(W, cols).items() if k != "sel_free"}
    jobs["sel_pair"] = sel_pair(cols)
    jobs["ctl_mnmx_first"] = mnmx_first(cols)
    return jobs


def kind(label: str) -> str:
    return COUNT[label.split("_")[0]]


def main() -> int:
    ck = Checks()
    quick = "--quick" in ck.args
    src = Path(__file__).resolve().parent / "fold_wide3.c"
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "fold_wide3"
        subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(src)], check=True)

        ins, yp, zp, r = fx1_case()
        three = {k: ins[k] for k in ("y", "z", "a")}
        got = fx.width_wide3(exe, W, three, {"count": (ins["y"], ins["z"])})["count"]
        want = fx._new_word_sets(W, three)
        ck.check("width 32: fold_wide3.c's distinct first instructions over three inputs are fold_opt_fx.py's",
                 got["firsts"] == want, (got["firsts"], want))

        six = {k: ins[k] for k in ("y", "z", "w", "kx", "a", "m1")}
        jobs = controls(six)
        res = fx.width_wide3(exe, W, six, jobs)
        for lab, hit in res.items():
            wit = [x for x in hit["witnesses"] if x[0] == {"found": "F", "sel": "S", "le2": "L"}[kind(lab)]]
            ck.check(f"width 32: fold_wide3.c finds its control {lab} ({kind(lab)})", hit[kind(lab)] > 0, wit[:1])

        if quick:
            ins = six
        got = fx.width_wide3(exe, W, ins, {"fx1": (yp, zp)})["fx1"]
        seen = {k: got[k] for k in ("found", "sel", "le2", "firsts", "prefixes", "witnesses")}
        n = f"{len(ins)} inputs, r = {r}"
        ck.check(f"32 bits ({n}): neither of fx1's outputs is a word of a program of at most 2 instructions",
                 got["le2"] == 0, seen)
        ck.check(f"32 bits ({n}): no program of at most 3 instructions writes fx1's (y', z') as its last instruction's "
                 "two words (IMAD.WIDE, IADD.64, IMNMX.64)", got["found"] == 0, seen)
        ck.check(f"32 bits ({n}): nor as a last SEL.64's under any predicate", got["sel"] == 0, seen)

        if ck.mutations:
            for drop, what, labs in PARTS:
                exe_d = Path(tmp) / f"fold_wide3_drop{drop}"
                subprocess.run(["gcc", "-O2", "-fopenmp", f"-DDROP={drop}", "-o", str(exe_d), str(src)], check=True)
                left = {lab: v[kind(lab)] for lab, v in
                        fx.width_wide3(exe_d, W, six, {lab: jobs[lab] for lab in labs}).items()}
                ck.mutation(f"width 32: fold_wide3.c without {what} misses {', '.join(labs)}", not any(left.values()),
                            left)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
