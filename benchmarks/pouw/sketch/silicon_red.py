"""SI-RED against the silicon proposals: each attack runs on the proposal's own reference, checked bit for bit.

- **Closure bypasses**: an instruction that advances the transcript for fewer issue cycles than a full-K MMA of the
  credited form. `ISA` lists the variants that share the credited m16n8k32 or its format, with their issue cycles
  relative to it (data-sheet ratios: 2:4 sparsity doubles a format's MMA rate, and int4 runs at twice int8 where a part
  has it). Each proposal's trigger rule and tag are encoded as worded (`RULES`). SI-THEORY's runs as its own
  `closure_ref.Unit`, whose advance is keyed on the shape and whose tag is `shape_tag(shape)` in s8's format: a
  C-injected issue on the variant, checked with `closure_ref.verify`.
- **Copy across tiles**: at zero data with no operand noise every word is zero, so a transcript started from the call's
  coins is the same chain on every tile. One chain a call, its state sealed under each tile's salt. Run on
  `closure_ref` (tile statements sharing the coins) and on SI-FOLD's `_keccak_transcript`. The fix derives each tile's
  start from (coins, tile index).
- **Double duty on SI-NOISE's sketch**, σ ← rotl(σ, 1) ⊕ (acc + n) with b-bit noise n. On hidden words ≡ 0 mod 2^b
  the add carries nothing, so σ(acc^H + n) ⊕ σ(acc^H) = σ(n), the zero-data claim's σ, without n. The hidden work is
  a GEMM whose operands are scaled so every product is ≡ 0 mod 2^b (int7 × int8 at b = 1, int4 × int4 at b = 8),
  recovered exactly. It is priced as the software fold of the hidden words: with the row's finalization linear (XOR of
  the row's cells, 32 bits folded to 16), and with each cell's σ read. The fix absorbs acc + n·φ, φ = 2³¹ − 1.

Prices are W1 writes (a 3-input LOP3 one, a rotate one), and stock is one a word (every lane carries data), as in
`red.py`. The harness's own designs are tied to the references here: `red.d_closure` to `closure_ref.F` and
`red.d_keccak` to `_keccak_transcript`.

Run (from this directory, with DIR holding closure_ref.py, silicon_fold.py and rho.py, and silicon_noise.py with its
hash_counter.py): `uv run --no-project --with numpy --with-editable ../exact_kit python -B silicon_red.py OUT.json
--lib=DIR [--mutations]`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

import red
from exact_kit.checks import Checks

M32 = (1 << 32) - 1
ROW = 64
PHI = (1 << 31) - 1
CREDITED = ("s8s8s32", False, (16, 8, 32))
ISA = (
    ("mma s8 m16n8k32", "s8s8s32", False, (16, 8, 32), 1.0),
    ("mma u8 m16n8k32", "u8u8s32", False, (16, 8, 32), 1.0),
    ("mma.sp s8 m16n8k32", "s8s8s32", True, (16, 8, 32), 0.5),
    ("mma s4 m16n8k32", "s4s4s32", False, (16, 8, 32), 0.5),
    ("mma s8 m16n8k16", "s8s8s32", False, (16, 8, 16), 0.5),
    ("mma.sp s8 m16n8k64", "s8s8s32", True, (16, 8, 64), 1.0),
)
RULES = {
    "SI-THEORY (closure_ref: the credited shape advances, absorbing shape_tag(shape))":
        (lambda f, sp, sh: sh == CREDITED[2], lambda f, sp, sh: sh),
    "SI-FOLD (a full k-step of the credited format and shape, absorbing its format and shape tag)":
        (lambda f, sp, sh: (f, sh) == (CREDITED[0], CREDITED[2]), lambda f, sp, sh: (f, sh)),
    "SI-NOISE P5 (the credited format with full k)":
        (lambda f, sp, sh: (f, sh) == (CREDITED[0], CREDITED[2]), lambda f, sp, sh: (f, sh)),
    "fixed (the whole descriptor advances and is the tag: format, sparsity, shape)":
        (lambda f, sp, sh: (f, sp, sh) == CREDITED, lambda f, sp, sh: (f, sp, sh)),
}


def rule_table():
    out = {}
    for rule, (adv, tag) in RULES.items():
        out[rule] = []
        for name, f, sp, sh, cyc in ISA:
            passes = adv(f, sp, sh) and tag(f, sp, sh) == tag(*CREDITED)
            out[rule].append({"variant": name, "advances": adv(f, sp, sh), "passes as credited": passes,
                              "cycles": cyc, "t1": round(1 - cyc, 3) if passes else 0.0})
    return out


def isa_unit(cr, fixed=False):
    class IsaUnit(cr.Unit):
        """closure_ref's Unit issuing an ISA variant: the reference's fold rule (or the fixed one), the variant's
        cycles. A shape the reference has no cycles for is outside `advance`, so it computes acc and leaves the fold."""

        def issue(self, variant, A, B, C):
            _, f, sp, shape, cyc = variant
            before = self.cycles
            if (fixed and (f, sp, shape) != CREDITED) or shape not in cr.ISSUE_CYCLES:
                acc = ((A.astype(np.int64) @ B.astype(np.int64) + C.astype(np.int64)) & M32).astype(np.uint32)
                acc = acc.view(np.int32)
            else:
                acc = self.mma(shape, A, B, C)
            self.cycles = before + cyc
            return acc
    return IsaUnit()


def c_inject(cr, st, variant, unit):
    unit.fold_init(st.coins)
    m, n, k = variant[3]
    for acc in cr.honest_words(st):
        unit.issue(variant, np.zeros((m, k), np.int8), np.zeros((k, n), np.int8), acc)
    return cr.seal(unit.state, len(st.A), st.salt), unit.cycles


def closure_section(checks, cr, T, rng):
    st = cr.statement(rng, T)
    h = cr.prove_honest(st)
    rows = []
    for v in ISA:
        for fixed in (False, True):
            sl, cyc = c_inject(cr, st, v, isa_unit(cr, fixed))
            ok = cr.verify(st, sl, T)[0]
            rows.append({"variant": v[0], "unit": "fixed" if fixed else "closure_ref", "verifies": ok,
                         "cycles_vs_honest": round(cyc / h.cycles, 3)})
    get = {(r["variant"], r["unit"]): r for r in rows}
    for name in ("mma.sp s8 m16n8k32", "mma s4 m16n8k32"):
        r = get[(name, "closure_ref")]
        checks.check(f"closure_ref: C-injection through {name} verifies at half the cycles (T {T})",
                     r["verifies"] and r["cycles_vs_honest"] == 0.5, r)
        checks.check(f"the fixed unit refuses {name}", not get[(name, "fixed")]["verifies"])
    checks.check("both units refuse k16 and accept the credited issue at full cycles",
                 not get[("mma s8 m16n8k16", "closure_ref")]["verifies"]
                 and not get[("mma s8 m16n8k16", "fixed")]["verifies"]
                 and all(get[("mma s8 m16n8k32", u)]["verifies"] and get[("mma s8 m16n8k32", u)]["cycles_vs_honest"]
                         == 1.0 for u in ("closure_ref", "fixed")))
    return rows


def closure_copy(checks, cr, T, N, rng, served=False, per_tile=False):
    """N tile statements under one call's coins; one chain run on tile 0, its state sealed under every tile's salt."""
    coins = rng.bytes(32)
    base = cr.statement(rng, T, zero=not served)
    tiles = [cr.Statement(base.A, base.B, coins + (i.to_bytes(4, "big") if per_tile else b""), rng.bytes(64))
             for i in range(N)]
    if served:
        tiles = [cr.Statement(cr.statement(rng, T).A, cr.statement(rng, T).B, t.coins, t.salt) for t in tiles]
    u = cr.Unit()
    u.fold_init(tiles[0].coins)
    m, n, _ = cr.SHAPE_CREDITED
    acc = np.zeros((m, n), np.int32)
    for a, b in zip(tiles[0].A, tiles[0].B):
        acc = u.mma(cr.SHAPE_CREDITED, a, b, acc)
    ok = [cr.verify(t, cr.seal(u.state, T, t.salt), T)[0] for t in tiles]
    return {"tiles": N, "verified": sum(ok), "issued": u.issues, "claimed": N * T,
            "t1": round(1 - u.issues / (N * T), 4) if all(ok) else 0.0}


def keccak_copy(sf, T, N, rng, per_tile=False, lead=False):
    coin = int(rng.integers(0, 1 << 63))
    outs = []
    for i in range(N):
        iv = int.from_bytes(hashlib.sha512(coin.to_bytes(8, "big") + i.to_bytes(4, "big")).digest()[:8], "big") \
            if per_tile else coin
        if lead:
            w, _, _ = sf._zero_data_words(np.random.default_rng(100 + i), T, 4, 8)
            steps = [[int(w[t, a, j]) for a in range(4) for j in range(8)] for t in range(T)]
        else:
            steps = [[0] * 32 for _ in range(T)]
        outs.append(tuple(sf._keccak_transcript(steps, iv)))
    return {"tiles": N, "distinct transcripts": len(set(outs)), "t1": round(1 - len(set(outs)) / N, 4)}


def ties(checks, cr, sf, rng):
    """The harness's designs equal the references they stand for."""
    cells, R = 6, 3
    W = rng.integers(0, 1 << 32, (cells, 32 * R), dtype=np.uint64)
    coins = rng.bytes(32)
    v = cr.fold_init(coins).v
    k = {f"h{i}": red.Val(np.uint64(v[i]), red.M64) for i in range(8)}
    got = red.d_closure([red.Val(W[:, t], red.M32) for t in range(W.shape[1])], k).a
    want = []
    for c in range(cells):
        s = None
        for wd in [*cr.shape_tag(cr.SHAPE_CREDITED), *(int(x) for x in W[c])]:
            s = cr.F(s, wd, coins)
        want.append(s.v[0])
    checks.check("red.d_closure equals closure_ref's F over [tag, words] (a's word, 6 cells, 48 rounds)",
                 [int(x) for x in got] == want and red.CLOSURE_TAG == cr.shape_tag(cr.SHAPE_CREDITED))
    iv = int(rng.integers(0, 1 << 63))
    got = red.d_keccak([red.Val(W[:, t], red.M32) for t in range(W.shape[1])],
                       {"iv": red.Val(np.uint64(iv), red.M64)}).a
    want = []
    for c in range(cells):
        lanes = sf._keccak_transcript([[int(x) for x in W[c, 32 * r:32 * r + 32]] for r in range(R)], iv)
        x = 0
        for ln in lanes:
            x ^= ln
        want.append(x)
    checks.check("red.d_keccak equals SI-FOLD's _keccak_transcript (XOR of the 25 lanes, 6 cells, 3 rounds)",
                 [int(x) for x in got] == want)


def hidden_gemm(rng, rows, cols, T, b):
    """acc_t = Σ_{s ≤ t} A_s·B_s over k32 steps, A = 2^sa·A', B = 2^sb·B', sa + sb = b, every operand in s8."""
    sa, sb = (b + 1) // 2, b // 2
    A = rng.integers(-(1 << (7 - sa)), 1 << (7 - sa), (T, rows, 32)) << sa
    B = rng.integers(-(1 << (7 - sb)), 1 << (7 - sb), (T, 32, cols)) << sb
    inc = np.einsum("tik,tkj->ijt", A, B)
    useful = np.einsum("tik,tkj->ij", A >> sa, B >> sb)
    acc = np.cumsum(inc, axis=2)
    exact = bool(np.array_equal(acc[:, :, -1] >> b, useful)) and bool(np.all(acc & ((1 << b) - 1) == 0))
    return (acc & M32).reshape(rows * cols, T).astype(np.int64), exact, (int(A.min()), int(A.max()), int(B.min()),
                                                                         int(B.max()))


def fin(sig, rows):
    x = np.bitwise_xor.reduce(sig.reshape(rows, -1), axis=1)
    return (x ^ (x >> 16)) & 0xFFFF


def row_program(acc, rows, rot=1):
    """The cheater's fold of a row's hidden words: XOR the row's cells a step with the rotated state, one rotate and
    ⌈cells/2⌉ LOP3 a step. Returns the row's σ_x and its writes."""
    A = acc.reshape(rows, -1, acc.shape[1])
    cells, T = A.shape[1], A.shape[2]
    v = np.zeros(rows, np.int64)
    for t in range(T):
        v = red_rotl(v, rot) ^ np.bitwise_xor.reduce(A[:, :, t], axis=1)
    return v, T * (1 + math.ceil(cells / 2))


def red_rotl(x, k=1):
    return ((x << k) | (x >> (32 - k))) & M32


def cell_program(acc):
    """Each cell's σ_x read alone: the words grouped by rotation (T − 1 − t) mod 32, each group XORed, rotated once
    and the groups XORed. Returns σ_x per cell and its writes a cell."""
    cells, T = acc.shape
    groups = {}
    for t in range(T):
        groups.setdefault((T - 1 - t) % 32, []).append(t)
    out = np.zeros(cells, np.int64)
    writes = 0
    for r, ts in groups.items():
        g = np.bitwise_xor.reduce(acc[:, ts], axis=1)
        writes += math.ceil((len(ts) - 1) / 2) + (r != 0)
        out ^= red_rotl(g, r) if r else g
    writes += math.ceil((len(groups) - 1) / 2)
    return out, writes


def separation(sn, rng, T, b, rows=4, phi=1, hidden=None, rot=1):
    """SI-NOISE's xorrot sketch on a hidden GEMM's words under its PRF noise, against the zero-data claim's σ."""
    cols = ROW
    acc, useful, rng_ab = hidden if hidden is not None else hidden_gemm(rng, rows, cols, T, b)
    n = sn.prf_words(int(rng.integers(0, 1 << 62)), (rows * cols, T), b)
    z = (n * phi) & M32
    r = np.zeros(T, np.int64)
    sH = sn.sketch("xorrot", acc, z, r)
    sD = sn.sketch("xorrot", np.zeros_like(acc), z, r)
    sx = sn.sketch("xorrot", acc, np.zeros_like(z), r)
    vrow, wrow = row_program(acc, rows, rot)
    vcell, wcell = cell_program(acc)
    return {"T": T, "b": b, "phi": phi, "useful_exact": useful, "operand_range": rng_ab,
            "cells_exact": float(np.mean(sD == (sH ^ sx))),
            "rows_exact": float(np.mean(fin(sD, rows) == (fin(sH, rows) ^ (vrow ^ (vrow >> 16)) & 0xFFFF))),
            "row_program_exact": bool(np.array_equal(vrow, np.bitwise_xor.reduce(sx.reshape(rows, -1), axis=1))),
            "cell_program_exact": bool(np.array_equal(vcell, sx)),
            "writes_a_word_row": round(wrow / (cols * T), 4), "writes_a_word_cell": round(wcell / T, 4),
            "zero_data_distinct_rows": int(len(set(fin(sD, rows).tolist())))}


def gap(writes_a_word):
    return round(max(0.0, 1 - writes_a_word), 4)


def noise_section(checks, sn, rng, Ts=(16, 64, 256), bs=(1, 8)):
    rows = []
    for b in bs:
        for T in Ts:
            s = separation(sn, rng, T, b)
            s["gap_row_linear"], s["gap_cell"] = gap(s["writes_a_word_row"]), gap(s["writes_a_word_cell"])
            f = separation(sn, rng, T, b, phi=PHI)
            s["fixed"] = {k: f[k] for k in ("cells_exact", "rows_exact", "zero_data_distinct_rows")}
            rows.append(s)
    checks.check("the hidden GEMM (operands ≡ 0 mod 2^b in s8) is recovered exactly from its accumulators",
                 all(s["useful_exact"] for s in rows))
    checks.check("σ(acc^H + n) ⊕ σ_x(acc^H) is the zero-data claim's σ(n) on every cell, at b 1 and 8, T 16–256",
                 all(s["cells_exact"] == 1.0 and s["rows_exact"] == 1.0 for s in rows))
    checks.check("the cheater's row and cell programs compute σ_x of the hidden words exactly",
                 all(s["row_program_exact"] and s["cell_program_exact"] for s in rows))
    checks.check("absorbing acc + n·(2³¹ − 1) stops the separation on the same hidden words (no cell exact)",
                 all(s["fixed"]["cells_exact"] < 0.01 for s in rows), [s["fixed"]["cells_exact"] for s in rows])
    z = np.zeros((8, 64), np.int64)
    n = sn.prf_words(5, z.shape, 8)
    checks.check("copy is refused on SI-NOISE: at zero data the cells' σ differ (the noise is indexed by cell)",
                 len(set(sn.sketch("xorrot", z, n, np.zeros(64, np.int64)).tolist())) == 8)
    T, cells = 64, 1024
    distinct = {}
    for b, phi in ((1, 1), (8, 1), (1, PHI), (8, PHI), (8, 0x01010101)):
        n = sn.prf_words(11 + b, (cells, T), b)
        sd = sn.sketch("xorrot", np.zeros((cells, T), np.int64), (n * phi) & M32, np.zeros(T, np.int64))
        distinct[(b, phi)] = len(set(sd.tolist()))
    checks.check(f"the zero-data σ keeps its entropy under φ = 2³¹ − 1 ({cells} cells, T {T}: as many distinct σ as "
                 "with φ = 1)", all(distinct[(b, PHI)] >= distinct[(b, 1)] - 1 >= cells - 2 for b in (1, 8)),
                 {f"b{b} φ {p:#x}": v for (b, p), v in distinct.items()})
    checks.check("replicating an 8-bit n across the word (φ = 0x01010101, even weight) leaves the zero-data σ at most "
                 "2^8 values", distinct[(8, 0x01010101)] <= 256, distinct[(8, 0x01010101)])
    for phi in (1, PHI):
        T, b = 24, 8
        acc = rng.integers(0, 1 << 32, (16, T), dtype=np.int64)
        base = sn.prf_words(7, (16, T), b)
        seen = set()
        for v in range(1 << b):
            nn = base.copy()
            nn[:, T // 2] = v
            seen.add(tuple(sn.sketch("xorrot", acc, (nn * phi) & M32, np.zeros(T, np.int64)).tolist()))
        checks.check(f"with φ = {phi:#x} a skipped step's {b}-bit noise is still guessed at 2^-{b} (256 distinct σ)",
                     len(seen) == 1 << b)
    return rows


def main() -> int:
    checks = Checks()
    opts = dict(a[2:].split("=", 1) for a in checks.args if a.startswith("--") and "=" in a)
    args = [a for a in checks.args if not a.startswith("--")]
    out_path = args[0] if args else "silicon_red.json"
    sys.path.insert(0, os.path.abspath(opts.get("lib", ".")))
    import closure_ref as cr
    import silicon_fold as sf
    import silicon_noise as sn

    rng = np.random.default_rng(int(opts.get("seed", 0)))
    res = {"schema": "si-red-silicon/v1", "rules": rule_table()}
    ties(checks, cr, sf, rng)
    t = res["rules"]
    checks.check("as worded, every proposal's rule passes the sparse m16n8k32 issue as credited, and SI-THEORY's the "
                 "int4 one too; the fixed rule passes only the credited issue",
                 all(any(r["variant"] == "mma.sp s8 m16n8k32" and r["passes as credited"] for r in t[k])
                     for k in list(RULES)[:3])
                 and any(r["variant"] == "mma s4 m16n8k32" and r["passes as credited"] for r in t[list(RULES)[0]])
                 and [r["variant"] for r in t[list(RULES)[3]] if r["passes as credited"]] == ["mma s8 m16n8k32"])
    res["closure"] = {T: closure_section(checks, cr, T, rng) for T in (8, 64)}

    res["copy"] = {"closure_ref": {}, "ft": {}}
    for N in (4, 16):
        c = res["copy"]["closure_ref"]
        c[f"N{N} zero, call coins"] = closure_copy(checks, cr, 8, N, rng)
        c[f"N{N} zero, coins ‖ tile"] = closure_copy(checks, cr, 8, N, rng, per_tile=True)
        c[f"N{N} served, call coins"] = closure_copy(checks, cr, 8, N, rng, served=True)
        f = res["copy"]["ft"]
        f[f"N{N} zero, call coin"] = keccak_copy(sf, 8, N, rng)
        f[f"N{N} zero, coin ‖ tile"] = keccak_copy(sf, 8, N, rng, per_tile=True)
        f[f"N{N} lead's noise, call coin"] = keccak_copy(sf, 8, N, rng, lead=True)
    c, f = res["copy"]["closure_ref"], res["copy"]["ft"]
    checks.check("closure_ref at zero data: one chain's state seals every tile under the call's coins (T1 1 − 1/N)",
                 all(c[f"N{N} zero, call coins"]["verified"] == N and c[f"N{N} zero, call coins"]["issued"] == 8
                     for N in (4, 16)), {k: v["t1"] for k, v in c.items()})
    checks.check("closure_ref with coins ‖ tile index: the copied state verifies only tile 0's seal",
                 all(c[f"N{N} zero, coins ‖ tile"]["verified"] == 1 for N in (4, 16)))
    checks.check("closure_ref on served operands: copy verifies only tile 0's seal",
                 all(c[f"N{N} served, call coins"]["verified"] == 1 for N in (4, 16)))
    checks.check("SI-FOLD's FT at zero data from the call's coin: every tile's transcript is the same chain",
                 all(f[f"N{N} zero, call coin"]["distinct transcripts"] == 1 for N in (4, 16)))
    checks.check("FT from H(coin ‖ tile), or on the lead's noise words: every tile's transcript differs",
                 all(f[f"N{N} zero, coin ‖ tile"]["distinct transcripts"] == N
                     and f[f"N{N} lead's noise, call coin"]["distinct transcripts"] == N for N in (4, 16)))

    res["noise"] = noise_section(checks, sn, rng)

    if checks.mutations:
        st = cr.statement(rng, 8)
        sl, _ = c_inject(cr, st, ISA[2], isa_unit(cr))
        bad = cr.Statement(st.A, st.B, st.coins, st.salt[::-1])
        checks.mutation("the sparse C-injected seal under another tile's salt", not cr.verify(bad, sl, 8)[0])
        u = cr.Unit()
        u.fold_init(st.coins)
        m, n, _ = cr.SHAPE_CREDITED
        for acc in cr.honest_words(st)[1:]:
            u.mma(cr.SHAPE_CREDITED, np.zeros((m, 32), np.int8), np.zeros((32, n), np.int8), acc)
        checks.mutation("C-injection one issue short", not cr.verify(st, cr.seal(u.state, 8, st.salt), 8)[0])
        acc, useful, rab = hidden_gemm(rng, 4, ROW, 16, 8)
        s = separation(sn, rng, 16, 8, hidden=((acc | 1) & M32, useful, rab))
        checks.mutation("hidden words with bit 0 set (a carry into the noise bits)", s["cells_exact"] < 1.0,
                        s["cells_exact"])
        s = separation(sn, rng, 16, 8, hidden=(acc, useful, rab), rot=2)
        checks.mutation("the cheater's row program rotating by 2", s["rows_exact"] < 1.0, s["rows_exact"])
        with checks.mutant("FT with the tile index in the start state"):
            r = keccak_copy(sf, 8, 4, rng, per_tile=True)
            checks.check("every tile's transcript is the same chain", r["distinct transcripts"] == 1)

    json.dump(res, open(out_path, "w"), indent=1, default=str)
    for rule, rs in res["rules"].items():
        print(f"{rule}: " + ", ".join(f"{r['variant']} T1 {r['t1']}" for r in rs if r["passes as credited"]))
    for s in res["noise"]:
        print(f"SI-NOISE b {s['b']} T {s['T']}: row-linear {s['writes_a_word_row']} a word (gap {s['gap_row_linear']}),"
              f" cell {s['writes_a_word_cell']} (gap {s['gap_cell']}), fixed exact {s['fixed']['cells_exact']}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
