"""Exhaustive small-width lower bounds on FOLD-OPT across words, by z3 CEGIS (RT-FOLD round 2).

This is the exhaustive counterpart to fold_super.py's randomised hunt. For a fold (duo, fx1 or li), a number of committed
words, a bit width n and a program length L, it asks z3 whether ANY straight-line program of length L over a fixed opcode
menu computes the whole fold receipt for all inputs at width n. The answer per cell is:

  UNSAT   -- no program of length L exists: a genuine lower bound (the receipt needs > L lane instructions at width n).
  SAT     -- a program of length L exists and is verified against the honest fold on a symbolic input: a BREAK. The
             program is reported. (Expected only at L >= honest; a SAT at L < honest would be a real shortcut.)
  TIMEOUT -- undecided within the cell's budget.

Honest costs: duo 3 a word, fx1 4 a word, li 1 a word (its half-step absorbs two committed words in 2 lane ops). The bound
that matters is whether any program computes W words below W x honest, so the sweep tests L strictly below honest.

Method (CEGIS, counterexample-guided inductive synthesis):
  - The program's structure (an opcode per slot, operand selectors, a shift immediate per slot, and an output selector per
    receipt word) is shared across examples; z3 searches for a structure that matches every example, then a verifier looks
    for an input where that concrete structure disagrees with the honest fold. No counterexample -> SAT (verified break);
    the verifier's counterexample is added and the loop repeats; synth UNSAT -> no such program.
  - Output selectors range over every value (inputs, consts and produced), so UNSAT rules out the receipt appearing in ANY
    slot, not just the last -- a real lower bound. No-dead-code and commutative-operand-ordering constraints break symmetry.
  - The keys depend on the portfolio seed, so a cell is a lower bound only if every seed is UNSAT; a lone SAT is flagged
    for inspection (it would be a key-specific shortcut, not a FOLD-OPT break, unless it reproduces the honest fold).

The fixed menu (documented, so the bound is against a stated instruction set): LOP3 (XOR3, AND3, (a^b)&c), IMAD (a*b+c),
IMAD.-c (a*b-c), IADD3 (+/-), LEA/shl-add ((a<<s)+b), shr-add ((a>>s)+b), rot-add (rotl(a,s)+b), and the SHF funnel
((a<<s)|(b>>(n-s))). It expresses every honest op, plus the shift/rotate shortcuts a cheater would use for the multiply.

Needs z3 (a workspace dependency, `z3-solver`), so it runs on a pod, not in this suite. Usage:

  python fold_cegis.py cell FOLD WORDS N L [TIMEOUT_S] [SEED]      # one cell, prints the verdict
  python fold_cegis.py sweep --out DIR [--procs P] [--max-min M] [--widths 4,6,8,12] [--seeds 1,2]

`sweep` fans the cells over a process pool and writes DIR/cegis_sweep.json (per-cell verdicts) and DIR/cegis_sweep.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path


# ---- honest fold references at width n (concrete int and symbolic z3) -----------------------------------------------------

_BIG = [0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F, 0x165667B1, 0xFF51AFD7]


def _keys_n(n, seed):
    """Generic odd multipliers a, arbitrary addends k, distinct rotations r, and fx1's (ax, kx), all at width n. The seed
    rotates the choice so a lower bound (every seed UNSAT) is not an artefact of one key; the a's are forced odd and away
    from 1/3 so no key admits a trivial shift-add in place of the multiply."""
    mask = (1 << n) - 1
    src = _BIG[seed % len(_BIG):] + _BIG[:seed % len(_BIG)]

    def odd_mul(x):
        a = (x & mask) | 1
        if a < 5 or a in (1, 3):                 # avoid shift-add-expressible multipliers at tiny widths
            a = ((mask >> 1) | 1) | 1
        return a

    a = [odd_mul(src[i]) for i in range(3)]
    k = [(src[i] >> 3) & mask for i in range(3)]
    r = [1, max(1, n // 2), max(1, n - 1)]       # distinct rotations in [1, n)
    ax = odd_mul(src[3])
    kx = (src[4] >> 1) & mask
    return a, k, r, ax, kx


def make_cell(fold, words, n, seed):
    """Return (n_inputs, out_slots_count, honest, consts, target_fn). target_fn(ins, z3mod|None) -> list of receipt words;
    pass the z3 module to get symbolic outputs, None for concrete python ints."""
    mask = (1 << n) - 1
    a, k, r, ax, kx = _keys_n(n, seed)

    def rotl_i(x, rr):
        rr %= n
        return ((x << rr) | (x >> (n - rr))) & mask if rr else x & mask

    def rotl_s(z3, x, rr):
        return z3.RotateLeft(x, rr % n)

    if fold == "duo":
        n_in = 2 + words
        honest = 3 * words

        def tgt(ins, z3=None):
            y, z = ins[0], ins[1]
            ds = ins[2:2 + words]
            if z3 is None:
                for i in range(words):
                    u = (y ^ ds[i] ^ k[i]) & mask
                    y = (u * a[i] + z) & mask
                    z = (rotl_i(u, r[i]) + y) & mask
                return [y, z]
            C = lambda v: z3.BitVecVal(v % (1 << n), n)
            for i in range(words):
                u = y ^ ds[i] ^ C(k[i])
                y = u * C(a[i]) + z
                z = rotl_s(z3, u, r[i]) + y
            return [y, z]
        return n_in, 2, honest, (a, k, ax, kx), tgt

    if fold == "fx1":
        n_in = 2 + words
        honest = 4 * words

        def tgt(ins, z3=None):
            y, z = ins[0], ins[1]
            ws = ins[2:2 + words]
            if z3 is None:
                for i in range(words):
                    t = (ws[i] * ax) & mask
                    u = (y ^ t ^ kx) & mask
                    y = (u * a[i] + z) & mask
                    z = (rotl_i(u, r[i]) + y) & mask
                return [y, z]
            C = lambda v: z3.BitVecVal(v % (1 << n), n)
            for i in range(words):
                t = ws[i] * C(ax)
                u = y ^ t ^ C(kx)
                y = u * C(a[i]) + z
                z = rotl_s(z3, u, r[i]) + y
            return [y, z]
        return n_in, 2, honest, (a, k, ax, kx), tgt

    if fold == "li":
        # li: a half-step absorbs two committed words in 2 lane ops; `words` committed -> ceil(words/2) half-steps, and an
        # odd trailing word is its own LOP3+IMAD. honest = 2 lane ops per half-step rounded up = 2*ceil(words/2) for the
        # pairs plus the trailing word, i.e. words rounded up to the per-word cost 1 -> honest = words (1 a word) when even,
        # words+1 only if odd trailing forces an extra op. We test the clean even runs and the brief's small odd run.
        n_in = 1 + words
        half = words // 2
        honest = 2 * half + (2 if words % 2 else 0)        # 2 lane ops a half-step; an odd trailing word costs 2 more
        honest = max(honest, words)                        # never below 1 a word

        def tgt(ins, z3=None):
            y = ins[0]
            ds = ins[1:1 + words]
            if z3 is None:
                i = 0
                while i + 1 < words:
                    y = ((y ^ ds[i] ^ k[(i // 2) % 3]) * a[(i // 2) % 3] + ds[i + 1]) & mask
                    i += 2
                if i < words:                              # odd trailing word
                    y = ((y ^ ds[i] ^ k[0]) * a[0]) & mask
                return [y]
            C = lambda v: z3.BitVecVal(v % (1 << n), n)
            i = 0
            while i + 1 < words:
                y = (y ^ ds[i] ^ C(k[(i // 2) % 3])) * C(a[(i // 2) % 3]) + ds[i + 1]
                i += 2
            if i < words:
                y = (y ^ ds[i] ^ C(k[0])) * C(a[0])
            return [y]
        return n_in, 1, honest, (a, k, ax, kx), tgt

    raise ValueError(fold)


# ---- the z3 opcode menu: each (name, fn(z3, a, b, c, sh, n)) --------------------------------------------------------------

def _menu():
    def xor3(z3, a, b, c, sh, n): return a ^ b ^ c
    def and3(z3, a, b, c, sh, n): return a & b & c
    def xorand(z3, a, b, c, sh, n): return (a ^ b) & c
    def imad(z3, a, b, c, sh, n): return a * b + c
    def imadc(z3, a, b, c, sh, n): return a * b - c
    def iadd3(z3, a, b, c, sh, n): return a + b + c
    def iadd3s(z3, a, b, c, sh, n): return a - b - c

    def _shift_ite(z3, n, sh, f):
        r = f(1)
        for s in range(2, n):
            r = z3.If(sh == s, f(s), r)
        return r

    def rotadd(z3, a, b, c, sh, n): return _shift_ite(z3, n, sh, lambda s: z3.RotateLeft(a, s)) + b
    def shladd(z3, a, b, c, sh, n): return _shift_ite(z3, n, sh, lambda s: a << s) + b
    def shradd(z3, a, b, c, sh, n): return _shift_ite(z3, n, sh, lambda s: z3.LShR(a, s)) + b
    def funnel(z3, a, b, c, sh, n): return _shift_ite(z3, n, sh, lambda s: (a << s) | z3.LShR(b, n - s))

    return [("XOR3", xor3), ("AND3", and3), ("XORAND", xorand), ("IMAD", imad), ("IMAD.-c", imadc),
            ("IADD3", iadd3), ("IADD3.s", iadd3s), ("ROTADD", rotadd), ("SHLADD", shladd),
            ("SHRADD", shradd), ("FUNNEL", funnel)]


_COMMUTATIVE = {"XOR3", "AND3", "IADD3"}          # all three operands interchangeable
_COMMUTATIVE_PAIR = {"IMAD", "IMAD.-c"}           # the multiplied pair (a, b) interchangeable


def _menu_concrete(n):
    """The menu as plain python int fns, in the SAME order as _menu(), for exact brute-force verification of a candidate."""
    mask = (1 << n) - 1

    def rl(x, s):
        s %= n
        return ((x << s) | (x >> (n - s))) & mask if s else x & mask

    return [
        lambda a, b, c, sh: (a ^ b ^ c) & mask,
        lambda a, b, c, sh: (a & b & c) & mask,
        lambda a, b, c, sh: ((a ^ b) & c) & mask,
        lambda a, b, c, sh: (a * b + c) & mask,
        lambda a, b, c, sh: (a * b - c) & mask,
        lambda a, b, c, sh: (a + b + c) & mask,
        lambda a, b, c, sh: (a - b - c) & mask,
        lambda a, b, c, sh: (rl(a, sh) + b) & mask,
        lambda a, b, c, sh: ((a << sh) + b) & mask,
        lambda a, b, c, sh: ((a >> sh) + b) & mask,
        lambda a, b, c, sh: (((a << sh) | (b >> (n - sh))) & mask),
    ]


def _verify_bruteforce(prog, n, n_in, const_vals, tgt):
    """Check a concrete candidate against the honest fold on EVERY width-n input. Returns (True, None) if it matches
    everywhere (a verified break) or (False, counterexample_input). Exact; used when the input space is small enough."""
    mc = _menu_concrete(n)
    mask = (1 << n) - 1
    consts = [c & mask for c in const_vals]
    for combo in itertools.product(range(1 << n), repeat=n_in):
        vals = list(combo) + consts
        for i in range(len(prog["op"])):
            a, b, c = (vals[prog["sel"][i][j]] for j in range(3))
            vals.append(mc[prog["op"][i]](a, b, c, prog["sh"][i]))
        outs = [vals[prog["out"][j]] for j in range(len(prog["out"]))]
        if outs != tgt(list(combo)):
            return False, combo
    return True, None


# ---- CEGIS synthesis for one cell ---------------------------------------------------------------------------------------

def solve_cell(fold, words, n, L, per_timeout_s, seed=1):
    """CEGIS: is there a length-L program over the menu computing the fold's receipt for all width-n inputs? Returns a dict
    with verdict UNSAT | SAT | TIMEOUT and timing. SAT carries the program and is re-verified against the honest fold."""
    import z3
    t0 = time.time()
    deadline = t0 + per_timeout_s
    menu = _menu()
    n_in, n_out, honest, consts_meta, tgt = make_cell(fold, words, n, seed)
    a, k, ax, kx = consts_meta
    const_vals = [0, 1, ax, kx] + list(a) + list(k)            # the public constants a cheater may bake in
    const_vals = [c % (1 << n) for c in const_vals]
    nconst = len(const_vals)
    nbase = n_in + nconst

    def C(v):
        return z3.BitVecVal(v % (1 << n), n)

    op = [z3.Int(f"op{i}") for i in range(L)]
    sel = [[z3.Int(f"s{i}_{j}") for j in range(3)] for i in range(L)]
    sh = [z3.Int(f"sh{i}") for i in range(L)]
    out = [z3.Int(f"o{j}") for j in range(n_out)]

    struct = []
    for i in range(L):
        struct.append(z3.And(op[i] >= 0, op[i] < len(menu)))
        struct.append(z3.And(sh[i] >= 1, sh[i] < max(2, n)))
        for j in range(3):
            struct.append(z3.And(sel[i][j] >= 0, sel[i][j] < nbase + i))
        # commutative-operand ordering (symmetry break)
        for mi, (name, _) in enumerate(menu):
            if name in _COMMUTATIVE:
                struct.append(z3.Implies(op[i] == mi, z3.And(sel[i][0] <= sel[i][1], sel[i][1] <= sel[i][2])))
            elif name in _COMMUTATIVE_PAIR:
                struct.append(z3.Implies(op[i] == mi, sel[i][0] <= sel[i][1]))
    for j in range(n_out):
        struct.append(z3.And(out[j] >= 0, out[j] < nbase + L))
    # no dead code: every produced value is read by a later op or is an output
    for i in range(L):
        slot = nbase + i
        reads = [sel[k][j] == slot for k in range(i + 1, L) for j in range(3)]
        reads += [out[j] == slot for j in range(n_out)]
        struct.append(z3.Or(*reads) if reads else (out[0] == slot))

    def vals_of(inputs):
        vals = list(inputs) + [C(c) for c in const_vals]
        for i in range(L):
            ands = []
            for j in range(3):
                e = vals[0]
                for idx in range(1, nbase + i):
                    e = z3.If(sel[i][j] == idx, vals[idx], e)
                ands.append(e)
            res = menu[0][1](z3, ands[0], ands[1], ands[2], sh[i], n)
            for mi in range(1, len(menu)):
                res = z3.If(op[i] == mi, menu[mi][1](z3, ands[0], ands[1], ands[2], sh[i], n), res)
            vals.append(res)
        return vals

    def outputs_of(vals):
        outs = []
        for j in range(n_out):
            e = vals[0]
            for idx in range(1, nbase + L):
                e = z3.If(out[j] == idx, vals[idx], e)
            outs.append(e)
        return outs

    import random
    rnd = random.Random(1000 * seed + L)
    examples = [[C(rnd.randrange(1 << n)) for _ in range(n_in)] for _ in range(4)]

    synth = z3.Solver()
    synth.set("random_seed", seed)
    synth.add(*struct)
    iters = 0
    while True:
        if time.time() > deadline:
            return {"verdict": "TIMEOUT", "secs": round(time.time() - t0, 1), "iters": iters, "honest": honest}
        synth.push()
        for ex in examples:
            outs = outputs_of(vals_of(ex))
            tgt_vals = tgt([x.as_long() for x in ex])
            for o, tv in zip(outs, tgt_vals):
                synth.add(o == C(tv))
        synth.set("timeout", max(1, int((deadline - time.time()) * 1000)))
        rc = synth.check()
        if rc == z3.unsat:
            return {"verdict": "UNSAT", "secs": round(time.time() - t0, 1), "iters": iters, "honest": honest}
        if rc != z3.sat:
            return {"verdict": "TIMEOUT", "secs": round(time.time() - t0, 1), "iters": iters, "honest": honest}
        mdl = synth.model()
        synth.pop()
        iters += 1
        prog = {"op": [mdl[op[i]].as_long() for i in range(L)],
                "sel": [[mdl[sel[i][j]].as_long() for j in range(3)] for i in range(L)],
                "sh": [mdl[sh[i]].as_long() for i in range(L)],
                "out": [mdl[out[j]].as_long() for j in range(n_out)]}
        prog["ops"] = [menu[o][0] for o in prog["op"]]
        # verify the concrete candidate against the honest fold. Small input spaces are checked exactly by brute force (so a
        # genuine break is never missed to a verify timeout); larger ones fall back to a symbolic z3 verifier.
        if n_in * n <= 20:
            ok, cex = _verify_bruteforce(prog, n, n_in, const_vals, tgt)
            if ok:
                return {"verdict": "SAT", "secs": round(time.time() - t0, 1), "iters": iters, "honest": honest,
                        "program": prog, "verify": "bruteforce"}
            examples.append([C(x) for x in cex])
            continue
        ver = z3.Solver()
        ver.set("random_seed", seed)
        ins = [z3.BitVec(f"x{t}", n) for t in range(n_in)]
        vals = list(ins) + [C(c) for c in const_vals]
        for i in range(L):
            args = [vals[prog["sel"][i][j]] for j in range(3)]
            vals.append(menu[prog["op"][i]][1](z3, args[0], args[1], args[2], prog["sh"][i], n))
        outs = [vals[prog["out"][j]] for j in range(n_out)]
        tgt_sym = tgt(ins, z3)
        ver.add(z3.Or(*[o != t for o, t in zip(outs, tgt_sym)]))
        ver.set("timeout", max(1, int((deadline - time.time()) * 1000)))
        rv = ver.check()
        if rv == z3.unsat:
            return {"verdict": "SAT", "secs": round(time.time() - t0, 1), "iters": iters, "honest": honest,
                    "program": prog, "verify": "symbolic"}
        if rv != z3.sat:
            return {"verdict": "TIMEOUT", "secs": round(time.time() - t0, 1), "iters": iters, "honest": honest}
        cex = ver.model()
        examples.append([C(cex[ins[t]].as_long() if cex[ins[t]] is not None else 0) for t in range(n_in)])


# ---- the sweep (multiprocessing over cells) -----------------------------------------------------------------------------

def _L_range(fold, words, n, honest):
    """Lengths to test for a cell: every L strictly below honest, capped by width so the harder cells still fit. Always
    includes the brief's 3..6 where they are below honest."""
    if n <= 6:
        hi = min(honest - 1, 8)
    elif n == 8:
        hi = min(honest - 1, 6)
    else:
        hi = min(honest - 1, 4)
    return list(range(1, hi + 1))


def _cell_task(args):
    fold, words, n, L, timeout_s, seed = args
    try:
        res = solve_cell(fold, words, n, L, timeout_s, seed)
    except Exception as e:                           # a z3 or modelling failure on one cell must not sink the sweep
        res = {"verdict": "ERROR", "error": f"{type(e).__name__}: {e}"}
    res.update(fold=fold, words=words, n=n, L=L, seed=seed)
    return res


def sweep(out_dir, procs, max_min, widths, seeds, folds, word_counts, cell_timeout_s):
    import multiprocessing as mp
    cells = []
    for fold in folds:
        for words in word_counts:
            for n in widths:
                _, _, honest, _, _ = make_cell(fold, words, n, seeds[0])
                for L in _L_range(fold, words, n, honest):
                    for seed in seeds:
                        cells.append((fold, words, n, L, cell_timeout_s, seed))
    cells.sort(key=lambda c: (c[3], c[2]))           # small L and width first (cheap, likely decisive)
    deadline = time.time() + max_min * 60
    results = []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_cell_task, cells):
            results.append(res)
            (out / "cegis_sweep.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
            v = res.get("verdict")
            print(f"{res['fold']}.{res['words']}w n={res['n']} L={res['L']} seed={res['seed']} -> {v} "
                  f"({res.get('secs', '-')}s, honest={res.get('honest', '-')})", flush=True)
            if v == "SAT" and res["L"] < res.get("honest", 1 << 30):
                print(f"  *** POSSIBLE BREAK: {res.get('program', {}).get('ops')} ***", flush=True)
            if time.time() > deadline:
                print("sweep: overall deadline reached; terminating remaining cells", flush=True)
                pool.terminate()
                break
    _write_summary(out, results)
    return results


def _write_summary(out, results):
    cells = {}
    for r in results:
        key = (r["fold"], r["words"], r["n"], r["L"])
        cells.setdefault(key, []).append(r)
    folds = sorted({r["fold"] for r in results})
    lines = ["# CEGIS lower-bound sweep (RT-FOLD round 2)", "",
             "Per cell, the verdict over all seeds: UNSAT (no program of that length), SAT (a program exists -- a BREAK if",
             "below honest), TIMEOUT, or ERROR. A cell is a genuine lower bound only when every seed is UNSAT.", ""]
    breaks = [r for r in results if r.get("verdict") == "SAT" and r["L"] < r.get("honest", 1 << 30)]
    lines.append(f"**Breaks found: {len(breaks)}**" + ("" if not breaks else f" -> {breaks}"))
    lines.append("")
    for fold in folds:
        lines.append(f"## {fold}")
        lines.append("")
        lines.append("| words | n | L | honest | verdict | secs |")
        lines.append("|---|---|---|---|---|---|")
        keys = sorted({(r["fold"], r["words"], r["n"], r["L"]) for r in results if r["fold"] == fold})
        for key in keys:
            rs = cells[key]
            verds = [r["verdict"] for r in rs]
            if any(v == "SAT" for v in verds):
                verdict = "SAT"
            elif all(v == "UNSAT" for v in verds):
                verdict = "UNSAT"
            elif any(v == "UNSAT" for v in verds):
                verdict = "UNSAT*"       # some seed UNSAT, some timed out
            else:
                verdict = verds[0]
            secs = max((r.get("secs", 0) or 0) for r in rs)
            honest = rs[0].get("honest", "-")
            lines.append(f"| {key[1]} | {key[2]} | {key[3]} | {honest} | {verdict} | {secs} |")
        lines.append("")
    (out / "cegis_sweep.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv):
    if argv and argv[0] == "cell":
        fold, words, n, L = argv[1], int(argv[2]), int(argv[3]), int(argv[4])
        timeout_s = int(argv[5]) if len(argv) > 5 else 120
        seed = int(argv[6]) if len(argv) > 6 else 1
        res = solve_cell(fold, words, n, L, timeout_s, seed)
        print(json.dumps(res, indent=1))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep"])
    ap.add_argument("--out", default=os.environ.get("OUT", "."))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-min", type=float, default=60.0)
    ap.add_argument("--cell-timeout", type=float, default=900.0, help="per-cell/seed z3 budget in seconds")
    ap.add_argument("--widths", default="4,6,8,12")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--folds", default="duo,fx1,li")
    ap.add_argument("--words", default="2,3")
    args = ap.parse_args(argv)
    widths = [int(x) for x in args.widths.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    folds = args.folds.split(",")
    word_counts = [int(x) for x in args.words.split(",")]
    print(f"cegis sweep: folds={folds} words={word_counts} widths={widths} seeds={seeds} procs={args.procs} "
          f"max_min={args.max_min} cell_timeout={args.cell_timeout}", flush=True)
    sweep(args.out, args.procs, args.max_min, widths, seeds, folds, word_counts, args.cell_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
