"""FOLD-SKIP: zero-data provers that never form a drawn fold's words, priced additively against stock.

At zero data a k32 word is w_t(i, j) = x_t + R_t(i) + C_t(class(i), j): a data term, the row walk and the column walk
(N-W1's `fold_dynamic.fold32` model, one row class, walks of 14-bit increments; with two classes a warp takes rows of
one class). Honest writes each word once (the MMA, one write: the write that credits stock) and then runs the fold
step. A step that adds its word before a multiply distributes over the word's parts. For arxm,
s' = (rotl(s ^ x, r) + w)·k + y, the prover runs

    v = LOP3(s ^ x);  u = LEA.HI(v, x_t + C_t(j), v, r);  s' = IMAD(u, k, R_t(i)·k + y)

with a column term in LEA.HI's addend and a row term in IMAD's: the same three instructions as honest's step, and no
word. With r a register (SHF.L.W and IADD3 for the LEA.HI) both sides pay one more, so the gap is the same. From a start
state shared by every output and fresh each call (`fold32`'s s0), step 1 is row plus column too:
s_1 = R_1(i)·k_1 + ((rotl(s0 ^ x_1, r_1) + x_1' + C_1(j))·k_1 + y_1), one IADD3. A state carried across calls has no
such edge, and the word skip alone breaks §2a.

Families (N-W1's table, honest instructions beyond the word):
- `arxm` (rotl(s ^ x, r) + w)·k + y [3]: the skip saves the word's write every step.
- `rotmuladd` rotl(s, r)·k + w [2] and `rotxoradd` rotl(s ^ x, r) + w [2]: the word is added last, so skipping it costs
  one IADD3 for the second part: a tie.
- `lik` (s ^ w ^ x)·k + y [2] and `xormul` (s ^ w)·k [2]: the word enters by XOR and must be formed.

Prices (W1): one instruction writes one register, one eq (32 W1); the MMA writes one word an eq. A thread holds `rows`
rows by `cols` columns of outputs and computes its rows' terms (the walk's value and R·k + y, 2 a row-step) and its
columns' terms (the walk's value and x + C, 2 a column-step), the column terms once a warp when every thread of the
warp holds the same columns (uniform registers: 63 a warp, so at most 32 columns beside the coins). The walks'
increments are k32 dot products of the noise's row and column sides, `WALK` eq each (8 IDP4As), once over the whole
`SHAPE` matrix for each row-step and each class's column-step. The gap is Daniel's §2a: cost(H) − cost(adv) as a share
of stock, (32 − q)/32 eq a word. N-W1's DynFoldXO premise asks c = 0.866 eq an output-step (credit B); §2a needs the
adversary within 1% of stock of honest's whole cost, word included.

Run: `uv run --no-project --with numpy --with-editable ../exact_kit python -B fold_skip.py [--mutations] [--json=OUT]`
(from this directory).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
from exact_kit.checks import Checks

U32 = np.uint32
M32 = (1 << 32) - 1
FOLD = {"arxm": 3, "rotmuladd": 2, "rotxoradd": 2, "lik": 2, "xormul": 2}
Q = 4
STOCK = (32 - Q) / 32
CREDIT_B = 0.99 * STOCK
STEPS = (74, 293)
LAYOUTS = (("4x32, uniform columns", 4, 32, True), ("1x32, uniform columns", 1, 32, True),
           ("8x8, per thread", 8, 8, False))
WALK = 8
SHAPE = (512, 2048)
CLASSES = 2


def rotl(v, r):
    r = U32(r)
    return (v << r) | (v >> (U32(32) - r))


def instance(seed: int, n: int, steps: int, shared_s0: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    big = lambda size=None: rng.integers(0, 2**32, size, dtype=np.uint64).astype(U32)  # noqa: E731
    R = np.cumsum(rng.integers(0, 2**14, (steps, n), dtype=np.uint64), axis=0).astype(np.uint64) & M32
    C = np.cumsum(rng.integers(0, 2**14, (steps, n), dtype=np.uint64), axis=0).astype(np.uint64) & M32
    coins = [{"x": big(), "y": big(), "k": U32(int(big()) | 1), "r": int(rng.integers(1, 32)), "d": big()}
             for _ in range(steps)]
    s0 = np.full((n, n), big(), dtype=U32) if shared_s0 else big((n, n))
    return {"R": R.astype(U32), "C": C.astype(U32), "coins": coins, "s0": s0, "n": n, "steps": steps}


def step(fam: str, s, w, c):
    x, y, k, r = c["x"], c["y"], c["k"], c["r"]
    if fam == "arxm":
        return (rotl(s ^ x, r) + w) * k + y
    if fam == "rotmuladd":
        return rotl(s, r) * k + w
    if fam == "rotxoradd":
        return rotl(s ^ x, r) + w
    if fam == "lik":
        return (s ^ w ^ x) * k + y
    if fam == "xormul":
        return (s ^ w) * k
    raise ValueError(fam)


def reference(fam: str, inst: dict, cells) -> list[int]:
    """One output at a time in Python ints, the words formed: the fold's definition."""
    out = []
    for i, j in cells:
        s = int(inst["s0"][i, j])
        for t, c in enumerate(inst["coins"]):
            w = (int(c["d"]) + int(inst["R"][t, i]) + int(inst["C"][t, j])) & M32
            x, y, k, r = int(c["x"]), int(c["y"]), int(c["k"]), c["r"]
            rot = lambda v: ((v << r) | (v >> (32 - r))) & M32  # noqa: E731
            s = {"arxm": ((rot(s ^ x) + w) * k + y), "rotmuladd": rot(s) * k + w, "rotxoradd": rot(s ^ x) + w,
                 "lik": (s ^ w ^ x) * k + y, "xormul": (s ^ w) * k}[fam] & M32
        out.append(s)
    return out


def honest(fam: str, inst: dict):
    """The served program: the MMA writes each word, then the step. Returns (state, writes an output-step)."""
    s = inst["s0"].copy()
    with np.errstate(over="ignore"):
        for t, c in enumerate(inst["coins"]):
            w = c["d"] + inst["R"][t][:, None] + inst["C"][t][None, :]
            s = step(fam, s, w, c)
    return s, 1 + FOLD[fam]


def skip(fam: str, inst: dict, edge: bool = True, mutate: str | None = None):
    """The word-skipping prover. Returns (state, instructions an output-step beyond precomputation, the step-1 saving
    an output, row ops a row-step, column ops a column-step)."""
    s = inst["s0"].copy()
    R, C = inst["R"], inst["C"]
    first = 0
    with np.errstate(over="ignore"):
        for t, c in enumerate(inst["coins"]):
            x, y, k, r = c["x"], c["y"], c["k"], c["r"]
            colp = c["d"] + C[t]
            if fam == "arxm":
                if t == 0 and edge:
                    base = rotl(inst["s0"][0, 0] ^ (x if mutate != "edge x" else x + U32(1)), r)
                    s = (R[t] * k)[:, None] + ((base + colp) * k + y)[None, :]
                    first = FOLD[fam] - 1
                    continue
                rowp = R[t] * k + (y if mutate != "row y" else U32(0))
                u = rotl(s ^ x, r) + (colp if mutate != "col d" else C[t])[None, :]
                s = u * k + rowp[:, None]
            elif fam == "rotmuladd":
                s = rotl(s, r) * k + colp[None, :] + R[t][:, None]
            elif fam == "rotxoradd":
                s = rotl(s ^ x, r) + colp[None, :] + R[t][:, None]
            else:
                s = step(fam, s, colp[None, :], c) + R[t][:, None]
    body = {"arxm": 3, "rotmuladd": 3, "rotxoradd": 3}.get(fam, FOLD[fam] + 1)
    return s, body, first, 2, 2


def overhead(row_ops: int, col_ops: int, rows: int, cols: int, uniform: bool) -> float:
    """Precomputation an output-step: the thread's row and column terms, and the walks' increments over SHAPE."""
    n, m = SHAPE
    return row_ops / cols + col_ops / (rows * (32 if uniform else 1)) + WALK / m + WALK * CLASSES / n


def main() -> int:
    checks = Checks()
    out_path = next((a.split("=", 1)[1] for a in checks.args if a.startswith("--json=")), None)
    report = {"stock_eq_a_word": STOCK, "credit_b_eq": CREDIT_B, "walk_eq": WALK, "shape": SHAPE,
              "classes": CLASSES, "families": {}}
    cells = [(0, 0), (1, 2), (5, 17), (31, 7), (63, 63)]
    for fam in FOLD:
        rows = report["families"][fam] = {}
        for T in STEPS:
            for seed in (0, 1):
                inst = instance(seed, 64, T)
                hs, hc = honest(fam, inst)
                ref = reference(fam, inst, cells)
                checks.check(f"{fam} T {T} seed {seed}: honest equals the reference on sampled outputs",
                             [int(hs[i, j]) for i, j in cells] == ref)
                ks, body, first, ro, co = skip(fam, inst)
                exact = bool(np.array_equal(ks, hs))
                if fam in ("lik", "xormul"):
                    checks.check(f"{fam} T {T} seed {seed}: the word enters by XOR, so leaving it unformed is wrong",
                                 not exact)
                    rows[f"T{T}.s{seed}"] = {"honest": hc, "skip_exact": exact, "cheapest": hc}
                    continue
                checks.check(f"{fam} T {T} seed {seed}: the word-skipping prover equals honest on all 4,096 outputs",
                             exact)
                per = {}
                for name, r_, c_, uni in LAYOUTS:
                    adv = body + overhead(ro, co, r_, c_, uni) - first / T
                    no_edge = body + overhead(ro, co, r_, c_, uni)
                    per[name] = {"adversary": round(adv, 4), "gap_eq": round(hc - adv, 4),
                                 "gap_share_of_stock": round((hc - adv) / STOCK, 4),
                                 "gap_share_no_edge": round((hc - no_edge) / STOCK, 4)}
                rows[f"T{T}.s{seed}"] = {"honest": hc, "skip_exact": exact, "step_body": body,
                                         "edge_saving_an_output": first, "layouts": per}
        if fam == "arxm":
            live = instance(2, 64, 74, shared_s0=False)
            checks.check("arxm: the step-1 edge needs a start state shared by the outputs (wrong from a live one)",
                         not np.array_equal(skip(fam, live)[0], honest(fam, live)[0]))
            checks.check("arxm: from a live start the word skip alone still equals honest",
                         np.array_equal(skip(fam, live, edge=False)[0], honest(fam, live)[0]))
    a = report["families"]["arxm"]
    lay = [v for row in a.values() for v in row["layouts"].values()]
    checks.check("arxm: the skip saves the word's write every step; under §2a the gap exceeds 1% of stock at every "
                 "layout and both T, with the step-1 edge and without it",
                 all(v["gap_share_no_edge"] > 0.01 for v in lay),
                 {k: {n: (v["gap_share_of_stock"], v["gap_share_no_edge"]) for n, v in row["layouts"].items()}
                  for k, row in a.items()})
    checks.check("arxm: the skip still pays more than credit B's 0.866 eq an output-step, so it doesn't falsify "
                 "DynFoldXO at N-W1's c; it falsifies the c §2a needs, honest's 4 eq less 1% of stock",
                 all(CREDIT_B < v["adversary"] < 4 - 0.01 * STOCK for v in lay),
                 f"adversary {min(v['adversary'] for v in lay):.3f}–{max(v['adversary'] for v in lay):.3f} eq")
    for fam in ("rotmuladd", "rotxoradd"):
        checks.check(f"{fam}: the skip ties honest before precomputation (the word added last costs one IADD3)",
                     all(row["step_body"] == row["honest"] for row in report["families"][fam].values()))
    if checks.mutations:
        inst = instance(3, 64, 74)
        hs = honest("arxm", inst)[0]
        for m, what in (("row y", "the row term without y"), ("col d", "the column term without the data term"),
                        ("edge x", "the step-1 edge with x off by one")):
            checks.mutation(f"arxm skip with {what}", not np.array_equal(skip("arxm", inst, mutate=m)[0], hs))
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=1) + "\n")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
