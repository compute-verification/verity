"""FOLD-FAMILY: how many of a fold family's members have a program cheaper than honest, over 1–3-step windows.

A family is a grammar for a fold step N ← op(N, D_t), its op kind and constants drawn per step by live coins after the
words are committed. A member is a window's s drawn steps. The prover sees the member before it computes, so the
cheapest prover compiles per member, and a family's saving is the mean over members of honest − cheapest. The premise
"a random member has no shortcut on average" is falsified by a positive share of members with one.

For each sampled member the search asks for a straight-line program of honest − 1 instructions over N-FLOOR's
`fold_window.Program` (every LOP3, IADD3 with any operand negated, IMAD and LEA form, each a T-function: output bit i
reads input bits ≤ i) that maps the window's inputs to N_s on every example, at width W. `unsat` at width W proves no
32-bit program of that length exists over that set, since a program's low W bits are a W-bit program of the same
length. `sat` gives a program that fits the examples, rechecked on a larger grid. Instructions that move high bits down
(PRMT, shifts right, IMAD.WIDE's high word) are outside the set; families with rotations aren't T-functions and aren't
searched here.

Targets:
- `zero`: D_t = R_t + C_t (zero data), with R_t, C_t, the member's constants and free terms given: s arbitrary
  functions of the constants, s of a row's terms and s of a column's terms. Honest pays the word's write and the op's
  instructions a step.
- `words`: D_t given (data the prover can't choose, the MMA writes each word); honest pays the op's instructions.

Op kinds (honest instructions beyond the word), k an odd constant, c a constant, g a two-input function of (D, c)
that depends on D, so N ↦ op(N, D) is a bijection:
- `mulxor` (N ⊕ D)·k [2], `xormul` N·k ⊕ D [2], `addmul` (N + D)·k [2], `muladd` N·k + D [1],
- `xorg` N ⊕ g(D, c) [1], `xorgmul` (N ⊕ g(D, c))·k [2], `sqxor` (N ⊕ D)² + c [2].
A bitwise g that depends on D at every bit is D ⊕ h(c), so the g kinds are XOR with a drawn constant.
A round fold's word entry, one half of U = h + W + K_t, is `addk` N + D + k [1], and with the word XORed by a
per-position constant first, `xoradd` N + (D ⊕ c) + k [2].

Families: `mulxor` (FOLD-OPT's step, the keys drawn), `arith` (the first four kinds uniform), `lop` (the two g kinds),
`all` (the seven fold kinds uniform), `round` (`xoradd`).

`--sweep=KIND` replaces the drawn members by KIND's one-step members at every w-bit c (k drawn odd). An unsat at c
then holds for every 32-bit pair of constants whose low w bits are c and that k. `--sweep-c=C` fixes c at C and sweeps
every w-bit k instead, even ones included (a round's K_t is any constant), or only the k that `--ks=K,K` lists.

`--free=F` gives the zero target F free terms of each kind instead of s. An unsat holds only for programs with at most
F of each: at c = 1, (R + C) ⊕ 1 = (R − 2r_0) + (C − 2c_0 + 1) + 4·r_0·c_0 (r_0, c_0 the low bits), one IMAD of two
free terms, so with F = 2 the 32-bit word entry has a two-instruction program at every k, and with F = 1 only where
4k ≡ 0 mod 2^w. A w-bit sweep at a c whose low w bits are 1 certifies nothing a prover with two free terms a side
can't beat.

`--grids=AxBxC,AxBxC` sets the zero target's two grids (rows × columns × entry states). Rows and columns are drawn with
repeats, the free terms are per drawn row and column, and each cell draws its own entry states, so a sat at any grid
is a fit to be rechecked and an unsat at any grid holds.

Run: `uv run --no-project --with z3-solver python -B fold_family.py OUT.json --window=PATH/fold_window.py [--width=8]
[--members=12] [--steps=1,2] [--families=...] [--targets=zero,words] [--timeout=300] [--jobs=4] [--sweep=KIND]
[--sweep-c=C] [--ks=K,K] [--free=F] [--grids=3x3x4,4x4x6]` (from this directory). PATH is `benchmarks/pouw/hash_sparse/` on N-FLOOR's branch.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import random
import sys
import time
import zlib

import z3

KINDS = {"mulxor": 2, "xormul": 2, "addmul": 2, "muladd": 1, "xorg": 1, "xorgmul": 2, "sqxor": 2, "addk": 1,
         "xoradd": 2}
FAMILIES = {"mulxor": ["mulxor"], "arith": ["mulxor", "xormul", "addmul", "muladd"], "lop": ["xorg", "xorgmul"],
            "all": list(KINDS)[:7], "round": ["xoradd"]}
G_TABLES = [t for t in range(16) if all(((t >> c) & 1) != ((t >> (2 + c)) & 1) for c in (0, 1))]
WINDOW = None


def load_window(path):
    spec = importlib.util.spec_from_file_location("fold_window", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def g_of(t, d, c, mask):
    """The two-input table t over (d, c) bitwise: bit 2·b + c of t is g(b, c)."""
    out = 0
    for b in (0, 1):
        for cb in (0, 1):
            if (t >> (2 * b + cb)) & 1:
                out |= (d if b else ~d) & (c if cb else ~c)
    return out & mask


def step(kind, n, d, p, mask):
    k, c, t = p["k"], p["c"], p["t"]
    if kind == "mulxor":
        return ((n ^ d) * k) & mask
    if kind == "xormul":
        return ((n * k) ^ d) & mask
    if kind == "addmul":
        return ((n + d) * k) & mask
    if kind == "muladd":
        return (n * k + d) & mask
    if kind == "xorg":
        return n ^ g_of(t, d, c, mask)
    if kind == "xorgmul":
        return ((n ^ g_of(t, d, c, mask)) * k) & mask
    if kind == "sqxor":
        return ((n ^ d) * (n ^ d) + c) & mask
    if kind == "addk":
        return (n + d + k) & mask
    if kind == "xoradd":
        return (n + (d ^ c) + k) & mask
    raise ValueError(kind)


def honest(target, member):
    return sum(KINDS[kind] + (target == "zero") for kind, _ in member)


def draw_member(rng, family, s, w):
    return [(rng.choice(FAMILIES[family]), {"k": rng.randrange(0, 1 << w) | 1, "c": rng.randrange(0, 1 << w),
                                            "t": rng.choice(G_TABLES)}) for _ in range(s)]


def instance(spec):
    """The solver for one member at length L: examples over rows × columns × entry states (zero) or states × words."""
    w, member, target, L = spec["width"], spec["member"], spec["target"], spec["L"]
    s, mask, F = len(member), (1 << spec["width"]) - 1, spec.get("free") or len(member)
    nr, nc, nn = spec["grid"]
    rng = random.Random(spec["seed"])
    consts = [x for _, p in member for x in (p["k"], p["c"])]
    names = ["N"] + ([f"R{t + 1}" for t in range(s)] + [f"C{t + 1}" for t in range(s)] if target == "zero"
                     else [f"D{t + 1}" for t in range(s)])
    names += [f"{x}{t + 1}" for t in range(s) for x in "kc"] + [f"keyfun{f}" for f in range(F)]
    if target == "zero":
        names += [f"rowfun{f}" for f in range(F)] + [f"colfun{f}" for f in range(F)]
    kf = [z3.BitVec(f"kf{f}", w) for f in range(F)]
    rf = [[z3.BitVec(f"rf{f}_{r}", w) for r in range(nr)] for f in range(F)]
    cf = [[z3.BitVec(f"cf{f}_{c}", w) for c in range(nc)] for f in range(F)]
    sol = z3.SolverFor("QF_BV")
    prog = WINDOW.Program("p", L, len(names), w)
    sol.add(prog.constraints())
    V = lambda x: z3.BitVecVal(x & mask, w)  # noqa: E731
    rows = [[rng.randrange(0, 1 << w) for _ in range(s)] for _ in range(nr)]
    cols = [[rng.randrange(0, 1 << w) for _ in range(s)] for _ in range(nc)]
    for r in range(nr):
        for c in range(nc):
            for _ in range(nn):
                n = rng.randrange(0, 1 << w)
                ds = ([(rows[r][t] + cols[c][t]) & mask for t in range(s)] if target == "zero"
                      else [rng.randrange(0, 1 << w) for _ in range(s)])
                want = n
                for (kind, p), d in zip(member, ds):
                    want = step(kind, want, d, p, mask)
                srcs = [V(n)] + ([V(x) for x in rows[r]] + [V(x) for x in cols[c]] if target == "zero"
                                 else [V(x) for x in ds])
                srcs += [V(x) for x in consts] + kf
                if target == "zero":
                    srcs += [rf[f][r] for f in range(F)] + [cf[f][c] for f in range(F)]
                sol.add(prog.run(srcs)[-1] == V(want))
    return sol, prog, names


def solve(spec):
    t0 = time.time()
    sol, prog, names = instance(spec)
    sol.set("timeout", int(spec["timeout"] * 1000))
    r = sol.check()
    out = dict(spec, result=str(r), seconds=round(time.time() - t0, 2), sources=names)
    if r == z3.sat:
        out["program"] = prog.decode(sol.model())
    return out


def specs(opt):
    w, members = int(opt.get("--width", 8)), int(opt.get("--members", 12))
    steps = [int(x) for x in str(opt.get("--steps", "1,2")).split(",")]
    fams = str(opt.get("--families", ",".join(FAMILIES))).split(",")
    targets = str(opt.get("--targets", "zero,words")).split(",")
    timeout = float(opt.get("--timeout", 300))
    zgrids = [tuple(map(int, g.split("x"))) for g in str(opt.get("--grids", "3x3x4,4x4x6")).split(",")]
    sweep = opt.get("--sweep")
    if sweep:
        rng = random.Random(f"sweep-{sweep}")
        fams, steps = [f"sweep-{sweep}"], [1]
        drawn = [[(sweep, {"k": rng.randrange(0, 1 << w) | 1, "c": c, "t": 3})] for c in range(1 << w)]
        if "--sweep-c" in opt:
            c = int(opt["--sweep-c"], 0)
            fams = [f"sweep-{sweep}-c{c}"]
            ks = [int(k, 0) for k in str(opt["--ks"]).split(",")] if "--ks" in opt else range(1 << w)
            drawn = [[(sweep, {"k": k, "c": c, "t": 3})] for k in ks]
    free = int(opt["--free"]) if "--free" in opt else None
    out = []
    for fam in fams:
        for s in steps:
            rng = random.Random(f"{fam}-{s}")
            for i in range(len(drawn) if sweep else members):
                member = drawn[i] if sweep else draw_member(rng, fam, s, w)
                for target in targets:
                    h = honest(target, member)
                    for L, grid in ((h - 1, zgrids[0] if target == "zero" else (1, 1, 24)),
                                    (h - 1, zgrids[1] if target == "zero" else (1, 1, 48)), (h, (2, 2, 3))):
                        if L == h and i:
                            continue
                        out.append({"name": f"{fam}-s{s}-m{i}-{target}-L{L}-g{'x'.join(map(str, grid))}"
                                            + (f"-F{free}" if free else ""),
                                    "family": fam, "s": s, "index": i, "target": target, "member": member,
                                    "honest": h, "L": L, "grid": grid, "width": w, "free": free,
                                    "timeout": timeout * (3 if L == h else 1),
                                    "seed": zlib.crc32(f"{fam}-{s}-{i}-{target}-{L}-{grid}".encode())})
    return out


def summarize(results):
    """A member has a shortcut when both grids at honest − 1 are sat, and none when either is unsat."""
    by = {}
    for x in results:
        if x["L"] != x["honest"]:
            by.setdefault((x["family"], x["s"], x["target"]), {}).setdefault(x["index"], []).append(x["result"])
    rows = []
    for (fam, s, target), cell in sorted(by.items()):
        vals = ["unsat" if "unsat" in v else "sat" if v == ["sat", "sat"] else "open" for v in cell.values()]
        rows.append({"family": fam, "s": s, "target": target, "members": len(vals),
                     "shortcut": vals.count("sat"), "none": vals.count("unsat"),
                     "open": len(vals) - vals.count("sat") - vals.count("unsat")})
    return rows


def main():
    global WINDOW
    args = sys.argv[1:]
    out_path = args[0]
    opt = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in args[1:]}
    WINDOW = load_window(opt["--window"])
    todo = specs(opt)
    todo.sort(key=lambda x: (x["s"], x["L"], x["grid"]))
    results = []
    ctx = mp.get_context("fork")
    with ctx.Pool(int(opt.get("--jobs", mp.cpu_count()))) as pool:
        for res in pool.imap_unordered(solve, todo):
            results.append(res)
            print(json.dumps({k: res[k] for k in ("name", "result", "seconds")}), flush=True)
            with open(out_path, "w") as f:
                json.dump({"results": sorted(results, key=lambda x: x["name"]), "summary": summarize(results)}, f,
                          indent=1)
    sane = [x for x in results if x["L"] == x["honest"]]
    bad = [x["name"] for x in sane if x["result"] != "sat"]
    for row in summarize(results):
        print(row)
    print(f"honest-length sanity: {len(sane) - len(bad)}/{len(sane)} sat" + (f"; not sat: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
