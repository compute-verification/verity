#!/usr/bin/env python3
"""F3 (option 4, pure recursion): what V* pays per nonzero of a unit's matrices when it reads them as data, in ANDs,
measured on the repository's own gadgets. Nothing here is estimated but the composition, which is written out below.

    python benchmarks/private_circuit/nonzero_cost.py [--out DIR] [--units rope,gemm,silu,f32mul,rmsnorm] [--reps 2]

**What V* computes.** The lincheck's circuit fold, Lean's `Flock.Stmt.foldedPartial`: for the unit's slot type (2^sl
columns), each nonzero (i, c) of A_0 adds ``α · lw[i mod 64] · eq(ρ_lo, i >> 6) · eq(r_lo, c >> 6)`` into ``comb[c mod 64]``
(B_0's without α), at the rep's coins. The coins are known before V* is fixed, so they are circuit constants
(`rec_algebra`): the weight is a constant function of the entry's hidden index bits, and the entries' values are 1 (GF(2)).
Per nonzero and rep V* pays

* ``weight``: the index bits (2 sl - 6) cut into chunks, each chunk a constant table read by its hidden bits through the
  monomial basis (2^k - k - 1 ANDs, shared by the 128 output bits; `fold_weight`), and the chunks multiplied (one
  ``Gf128Mul_v1`` per extra chunk). The chunking minimises the ANDs; ``lw`` keeps the row's six low bits in one chunk.
* ``demux``: the product into one of 64 accumulators by the hidden c mod 64 (a 6-bit one-hot decode, then 64 x 128 ANDs).
* the accumulation itself is XORs. Padding to a nonzero bound is free: an entry repeated twice cancels in characteristic 2.

Per nonzero and session (once, both reps read the same data) V* also authenticates the entry against the registration:
its ``2 sl`` index bits are row bits of the registered value, so it pays their hashing. Read whole, a registered value is
rehashed leaf by leaf (row digest, ``Hm96Sha512Leaf_v1``, ``FrameV3Leaf_v1``, and one ``FrameV3Node_v1`` per leaf for the
tree's n - 1 nodes), never a Merkle path per leaf; `registration` measures it per row width, and the per-row hidden read
with a path of depth D beside it, which is what a read of single entries at hidden positions would pay.

**Real units.** C-Flock's own lowerings (``ir_lower.layout``): the RoPE pair, the tensor-core step that a GEMM coordinate and
an attention head share, the SiLU·mul element, the RMSNorm row and the Boolean ``F32MulFtz_v3``, each in its own process
(two lowerings register kernels that collide). ``counts`` gives ``nnz_a`` and ``nnz_b``; the rows give the slot balance
(the busiest of the 64 column slots against the mean) and the packed count (distinct (row, c >> 6)).

ANDs are the traced Boolean gates' (`evaluation.bits.compile`, `trace.trace`, or a `forms.Circuit` with no commits); C-Flock's
rows add the copies a `forms.Circuit` commits (a SHA-512 compression: 58,120 ANDs, 81,402 rows).

Writes ``nonzero_cost.json`` to ``--out`` or ``$RESEARCH_RUN_DIR/out`` when either is set; prints the tables either way.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

UNITS = ("rope", "gemm", "silu", "f32mul", "rmsnorm")


# --- gadgets --------------------------------------------------------------------------------------------------------------

def _monomials(C, bits):
    """Every monomial of ``bits`` (subset S -> AND of its bits), index S as a bitmask; 2^k - k - 1 ANDs."""
    from verity.ml.circuits.boolean.forms import ONE

    m = [ONE]
    for b in bits:
        m = m + [C.AND(x, b) for x in m]
    return m


def _moebius(table: list[int]) -> list[int]:
    """The ANF coefficients of each output bit, as ints over the table's words: f(x) = XOR over S <= x of coef[S]."""
    a = list(table)
    k = len(a).bit_length() - 1
    for j in range(k):
        for s in range(len(a)):
            if s >> j & 1:
                a[s] ^= a[s ^ (1 << j)]
    return a


def const_read(C, bits, table: list[int], width: int = 128):
    """``table[bits]`` for a constant table: its monomials, then each output bit an XOR of them (no ANDs)."""
    from verity.ml.circuits.boolean.forms import ZERO

    m = _monomials(C, bits)
    coef = _moebius(table)
    out = []
    for j in range(width):
        acc = ZERO
        for s, c in enumerate(coef):
            if c >> j & 1:
                acc = acc ^ m[s]
        out.append(acc)
    return out


def _eq_table(point: list[int], n: int) -> list[int]:
    from verity.ml.circuits.ops.boolean import gf2k as G

    out = [1]
    for r in point[:n]:
        out = [G.mul(e, 1 ^ r) for e in out] + [G.mul(e, r) for e in out]
    return out


def _chunks(n_bits: int, first_min: int, mul_ands: int) -> list[int]:
    """Contiguous chunk sizes of ``n_bits`` minimising sum(2^k - k - 1) + (len - 1) * mul_ands, the first at least
    ``first_min`` bits."""
    best = {0: (0, [])}
    for end in range(1, n_bits + 1):
        cands = []
        for k in range(1, min(end, 16) + 1):
            start = end - k
            if start == 0 and k < first_min or start not in best:
                continue
            cost, ch = best[start]
            cands.append((cost + (1 << k) - k - 1 + (mul_ands if ch else 0), ch + [k]))
        if cands:
            best[end] = min(cands)
    return best[n_bits][1]


def fold_weight(sl: int, seed: int = 0, check: int = 64) -> dict:
    """The per-entry weight ``lw[i & 63] · eq(ρ, i >> 6) · eq(r, c >> 6)`` (α a constant factor, free) as a circuit of the
    entry's hidden index bits, at random constant coins, checked on ``check`` random entries against the field reference."""
    from verity.ml.circuits.boolean.forms import Circuit
    from verity.ml.circuits.ops.boolean import gf2k as G

    rnd = random.Random(seed)
    F = lambda: rnd.getrandbits(128)
    lw, rho, r = [F() for _ in range(64)], [F() for _ in range(sl - 6)], [F() for _ in range(sl - 6)]
    n = 2 * sl - 6
    sizes = _chunks(n, 6, 2187)
    # bit t of the index: 0..5 the row's low bits (lw), then ρ's (row bits 6..sl-1), then r's (column bits 6..sl-1)
    C = Circuit()
    i_bits, q_bits = C.input("i", sl), C.input("q", sl - 6)
    order = i_bits + q_bits
    terms, at = [], 0
    for k in sizes:
        idx = list(range(at, at + k))
        tab = []
        for v in range(1 << k):
            x = 1
            lo = 0
            for j, t in enumerate(idx):
                b = v >> j & 1
                if t < 6:
                    lo |= b << t
                elif t < sl:
                    x = G.mul(x, rho[t - 6] if b else 1 ^ rho[t - 6])
                else:
                    x = G.mul(x, r[t - sl] if b else 1 ^ r[t - sl])
            if idx[0] == 0:
                x = G.mul(x, lw[lo])
            tab.append(x)
        terms.append(const_read(C, [order[t] for t in idx], tab))
        at += k
    w = terms[0]
    for t in terms[1:]:
        w = G.mul_forms(C, w, t)
    C.output("w", w)
    ands = C.counts()["ands"]
    # check: bit-sliced, one vector per entry
    ents = [(rnd.randrange(1 << sl), rnd.randrange(1 << (sl - 6))) for _ in range(check)]
    vals = {"i": [sum(((e[0] >> b) & 1) << v for v, e in enumerate(ents)) for b in range(sl)],
            "q": [sum(((e[1] >> b) & 1) << v for v, e in enumerate(ents)) for b in range(sl - 6)]}
    outs, _ = C.evaluate(vals, check)
    erho, er = _eq_table(rho, sl - 6), _eq_table(r, sl - 6)
    for v, (i, q) in enumerate(ents):
        got = sum(((outs["w"][j] >> v) & 1) << j for j in range(128))
        want = G.mul(G.mul(lw[i & 63], erho[i >> 6]), er[q])
        assert got == want, (sl, i, q)
    return {"sl": sl, "index_bits": n, "chunks": sizes, "ands": ands, "checked": check}


def demux() -> dict:
    """``w`` into one of 64 accumulators by a hidden 6-bit slot: the one-hot decode and the 64 x 128 gated copies."""
    from verity.ml.circuits.boolean.forms import ONE, Circuit

    C = Circuit()
    s, w = C.input("s", 6), C.input("w", 128)
    hot = [ONE]
    for b in s:
        hot = [C.AND(h, b.inv()) for h in hot] + [C.AND(h, b) for h in hot]
    for l, h in enumerate(hot):
        C.output(f"acc{l}", [C.AND(h, x) for x in w])
    return {"ands": C.counts()["ands"]}


def gadgets() -> dict:
    """The registered Definitions' ANDs, through `evaluation.bits.compile` (`.traced` for the frame-v3 parts' statics)."""
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.boolean import trace as TR
    from verity.ml.circuits.boolean.forms import Circuit
    from verity.ml.circuits.evaluation import bits as EB
    from verity.ml.circuits.ops.boolean import gf2k as G
    from verity.ml.circuits.types import Array
    from verity.ml.commitments.gates import frame_v3 as FV
    from verity.ml.commitments.gates import hm96 as H
    from verity.ml.commitments.gates import rowleaf as RL
    from verity.ml.commitments.gates.sha512 import compress_gadget

    word = Array(64, BL.BIT)
    comp = TR.trace(lambda C, *e: compress_gadget(C, e[:8], e[8:]), (Array(8, word), Array(16, word)), Array(8, word))
    C = Circuit()
    cv, msg = C.input("cv", 512), C.input("msg", 1024)
    C.output("out", [b for w in compress_gadget(C, [cv[64 * i:64 * i + 64] for i in range(8)],
                                                 [msg[64 * i:64 * i + 64] for i in range(16)]) for b in w])
    row = "hm96-sha512/row/v2"
    seg = {m: RL.Sha512RowSegSegment.traced(m).counts["and"] for m in (1024, 8192)}
    return {
        "sha512_compression": comp.counts["and"],
        "sha512_compression_rows": C.counts()["ands"],
        "Gf128Mul_v1": EB.compile(G.Gf128Mul).counts["and"],
        "Hm96Sha512Leaf_v1": EB.compile(H.Hm96Sha512Leaf).counts["and"],
        "FrameV3Leaf_v1{1,row/v2}": FV.FrameV3Leaf.traced(1, row).counts["and"],
        "FrameV3Node_v1{1}": FV.FrameV3Node.traced(1).counts["and"],
        "FrameV3Node_v1{2}": FV.FrameV3Node.traced(2).counts["and"],
        "LmsEqual_v1": 511,
        "Sha512RowSegSegment_v1{1024}": seg[1024],
        "Sha512RowSegSegment_v1{8192}": seg[8192],
    }


def registration(g: dict, row_bits: int, depth: int) -> dict:
    """ANDs per registered row bit: read whole (digest, hm96 leaf, frame leaf, one node per leaf) and read alone at a hidden
    position (the same plus a path of ``depth`` nodes and the root comparison). A ``sha512/row/v2`` row of b bits hashes
    ceil((ceil(b / 8) + 17) / 128) blocks after its constant prefix block."""
    comps = -(-(-(-row_bits // 8) + 17) // 128)
    digest = comps * g["sha512_compression"]
    leaf = g["Hm96Sha512Leaf_v1"] + g["FrameV3Leaf_v1{1,row/v2}"]
    whole = digest + leaf + g["FrameV3Node_v1{2}"]
    alone = digest + leaf + depth * g["FrameV3Node_v1{2}"] + g["LmsEqual_v1"]
    return {"row_bits": row_bits, "digest_compressions": comps, "whole_per_row": whole, "whole_per_bit": whole / row_bits,
            "hidden_read_per_row": alone, "depth": depth}


# --- real units -----------------------------------------------------------------------------------------------------------

def _lowering(name: str):
    from verity_numerical.bench import lowerings
    from verity_numerical.bench import templates as TM

    reg = lowerings.registry("C-Flock")
    if name == "f32mul":
        from verity_flock import ir_lower as IL

        from verity.ml.circuits.boolean.forms import Circuit
        from verity.ml.circuits.ops.boolean import attention as AT

        C = Circuit()
        a, b = C.input("in0", 32), C.input("in1", 32)
        C.output("out0", AT.F32MulFtz.build(C, a, b))
        return "F32MulFtz_v3", IL.layout(C, ["in0", "in1"], ["out0"])
    tid, params, fn = {"rope": ("rope-head", {"D": 64}, "lowering"),
                       "gemm": ("gemm-coordinate", {"K": 2048, "semantics": "sm80.mma.m16n8k16.bf16"}, "circuit_lowering"),
                       "silu": ("silu-mul", {"I": 8192}, "lowering"),
                       "rmsnorm": ("rmsnorm-fused-cuda", {"N": 2048, "EPS": 1e-05}, "lowering")}[name]
    sub = TM.get(tid).bind(**params)
    low = getattr(reg.module(tid), fn)(sub)
    return f"{sub.id} ({low.name})", low.layout


def unit_stats(name: str) -> dict:
    label, lay = _lowering(name)
    c = lay.counts
    sl = max(7, lay.const.bit_length())
    out = {"unit": label, "rows": c["rows"], "ands": c["ands"], "nnz_a": c["nnz_a"], "nnz_b": c["nnz_b"], "sl": sl}
    for m, rows in (("a", lay.rows_a), ("b", lay.rows_b)):
        slots = [0] * 64
        packed = 0
        for r in rows:
            cols = [int(x) for x in r if x != "C"]
            cols += [lay.const] * (len(r) - len(cols))
            for x in cols:
                slots[x & 63] += 1
            packed += len({x >> 6 for x in cols})
        n = sum(slots)
        out[f"slot_max_over_mean_{m}"] = max(slots) / (n / 64) if n else 0.0
        out[f"packed_{m}"] = packed
    out["nnz"] = out["nnz_a"] + out["nnz_b"]
    out["nnz_per_row"] = out["nnz"] / max(1, c["rows"])
    return out


def _unit_subprocess(name: str) -> dict:
    p = subprocess.run([sys.executable, __file__, "--one-unit", name], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


# --- the model ------------------------------------------------------------------------------------------------------------

def per_nonzero(u: dict, g: dict, w: dict, dm: dict, reg: dict, reps: int) -> dict:
    """ANDs per nonzero of the unit, per session: the fold (``reps`` times) and the registration read (once)."""
    bits = 2 * u["sl"]
    fold = reps * (w["ands"] + dm["ands"])
    regist = bits * reg["whole_per_bit"]
    # levers: per-slot streams (no demux, each of the 64 streams padded to the busiest), packed by column word
    slot_pad = max(u["slot_max_over_mean_a"], u["slot_max_over_mean_b"])
    streams = reps * w["ands"] * slot_pad + (bits - 6) * reg["whole_per_bit"] * slot_pad
    packed = u["packed_a"] + u["packed_b"]
    packed_cost = (reps * (w["ands"] + dm["ands"]) + (bits - 6 + 64) * reg["whole_per_bit"]) * packed / u["nnz"]
    return {"index_bits": bits, "fold": fold, "registration": regist, "total": fold + regist,
            "streams_total": streams, "streams_pad": slot_pad, "packed_total": packed_cost, "packed_ratio": u["nnz"] / packed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--units", default=",".join(UNITS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--row-bits", type=int, nargs="*", default=[1024, 8192, 65536])
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--one-unit")
    a = ap.parse_args()
    if a.one_unit:
        print(json.dumps(unit_stats(a.one_unit)))
        return
    g = gadgets()
    dm = demux()
    units = [_unit_subprocess(n) for n in a.units.split(",") if n]
    weights = {sl: fold_weight(sl) for sl in sorted({u["sl"] for u in units} | {13, 14, 16, 18, 20})}
    regs = {b: registration(g, b, a.depth) for b in a.row_bits}
    reg = regs[max(a.row_bits)]
    rows = []
    for u in units:
        rows.append(u | {"cost": per_nonzero(u, g, weights[u["sl"]], dm, reg, a.reps)})
    res = {"gadgets": g, "demux": dm, "weights": weights, "registration": regs, "reps": a.reps, "units": rows}

    print("gadget ANDs:", json.dumps(g))
    print(f"demux (one-hot of c mod 64, 64 x 128 copies): {dm['ands']}")
    print("weight per entry and rep (sl: chunks -> ANDs):", {sl: (w["chunks"], w["ands"]) for sl, w in weights.items()})
    for b, r in regs.items():
        print(f"registration, rows of {b} bits: whole {r['whole_per_row']:,} per row = {r['whole_per_bit']:.1f} per bit; "
              f"hidden read at D={r['depth']}: {r['hidden_read_per_row']:,} per row")
    print(f"\nper nonzero and session, {a.reps} reps, rows of {max(a.row_bits)} bits read whole:")
    print(f"{'unit':58} {'nnz':>10} {'sl':>3} {'nz/row':>6} {'fold':>7} {'reg':>6} {'total':>7} {'V* per session':>15} "
          f"{'streams':>8} {'pad':>5} {'packed':>7} {'nz/pk':>5}")
    for u in rows:
        c = u["cost"]
        print(f"{u['unit'][:58]:58} {u['nnz']:>10,} {u['sl']:>3} {u['nnz_per_row']:>6.1f} {c['fold']:>7,.0f} "
              f"{c['registration']:>6,.0f} {c['total']:>7,.0f} {c['total'] * u['nnz']:>15.3e} {c['streams_total']:>8,.0f} "
              f"{c['streams_pad']:>5.2f} {c['packed_total']:>7,.0f} {c['packed_ratio']:>5.2f}")
    print("\nV* per session at a shape's nonzero bound N (padding by cancelling pairs is free), and the instances per session at "
          "which V* equals the inner proof's own ANDs (instances x unit ANDs):")
    bounds = (16, 18, 20, 23)
    print(f"{'sl':>3} {'per nz':>7} " + " ".join(f"{'N=2^' + str(b):>10}" for b in bounds))
    for sl, w in weights.items():
        per = a.reps * (w["ands"] + dm["ands"]) + 2 * sl * reg["whole_per_bit"]
        print(f"{sl:>3} {per:>7,.0f} " + " ".join(f"{per * 2 ** b:>10.2e}" for b in bounds))
    res["bounds"] = {sl: a.reps * (w["ands"] + dm["ands"]) + 2 * sl * reg["whole_per_bit"] for sl, w in weights.items()}
    for u in rows:
        u["even_instances"] = u["cost"]["total"] * u["nnz"] / u["ands"]
        print(f"  {u['unit'][:56]:56} unit ANDs {u['ands']:>9,}  V* = inner at {u['even_instances']:>12,.0f} instances")
    out = a.out or (Path(os.environ["RESEARCH_RUN_DIR"]) / "out" if os.environ.get("RESEARCH_RUN_DIR") else None)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "nonzero_cost.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
