#!/usr/bin/env python3
"""F3 (option 4, pure recursion): what V* pays per nonzero of a unit's matrices when it reads them as data, in ANDs,
measured on the repository's own gadgets. Nothing here is estimated but the composition, which is written out below.

    python benchmarks/private_circuit/nonzero_cost.py [--out DIR] [--units rope,gemm,silu,f32mul,rmsnorm] [--reps 2]
    python benchmarks/private_circuit/nonzero_cost.py --buckets prover.json [--out DIR]

``--buckets`` takes compute-accounting's F6 prover table (``verity/f6-prover/v1``) and gives V* per session at each unsplit
bucket instead (`bucket_cost`): ANDs unpacked and packed, D under C-Flock's layout, and F6's prove-time and RSS law.
``--candidate3 DIR`` (proofs' F4 K2 tree, art:5d4b3d5f…) adds the frozen design's V* beside it (`candidate3`), which runs the
slot sumcheck's verifier and opens the registered bits instead of reading the matrix. ``--proofs-c3 buckets.json`` (with
``--buckets`` for the law) lays proofs' own candidate-3 table (art:4e9ff89a…) out as multi-block statements (`proofs_c3`).

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
import math
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


def _eq(point: list[int], x: int) -> int:
    from verity.ml.circuits.ops.boolean import gf2k as G

    out = 1
    for j, r in enumerate(point):
        out = G.mul(out, r if x >> j & 1 else 1 ^ r)
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
    for v, (i, q) in enumerate(ents):
        got = sum(((outs["w"][j] >> v) & 1) << j for j in range(128))
        want = G.mul(G.mul(lw[i & 63], _eq(rho, i >> 6)), _eq(r, q))
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


# --- F6's buckets ---------------------------------------------------------------------------------------------------------

# nonzeros per packed (row, c >> 6) entry of the unit measured above that a bucket's label family lowers to; a bucket takes the
# smallest over its labels (F6 gives no per-label mix), and a family not measured takes the smallest measured (the tc step's)
PACKED_RATIO = {"rope": 4.64, "gemm": 3.67, "silu": 23.94, "f32mul": 4.48, "rmsnorm": 4.24}
FAMILY = (("RoPE", "rope"), ("SiluMul", "silu"), ("RMSNorm", "rmsnorm"), ("Gemm", "gemm"), ("ScaledMm", "gemm"),
          ("Attention", "gemm"))
RHO = 26.5  # nonzeros per row of V*'s own lowering, F6's default


def _family(label: str) -> str | None:
    return next((f for p, f in FAMILY if p in label), None)


def bucket_cost(b: dict, part: str, g: dict, weights: dict, dm: dict, reg: dict, reps: int, fit: dict, glue: float,
                bytes_per_nnz: float) -> dict:
    """V* for one session of an F6 bucket: each of its ``slots`` slots an independently hidden type with the bucket's nonzero
    bound Z (F6 counts nnz_a + nnz_b + rows; the identity C's rows are not read, so V* reads Z (1 - 1/ρ)) at slot size
    sl = k_log, and its type chosen by a hidden path into the registry."""
    s = b["session"]
    sl, slots = s["k_log"], b["slots"]
    nnz = slots * b["Z"] * (1 - 1 / s["rho"])
    w = weights[sl]["ands"]
    per_bit = reg["whole_per_bit"]
    fold, regist = reps * (w + dm["ands"]), 2 * sl * per_bit
    fams = {_family(x) for x in b["labels"]}
    ratio = min(PACKED_RATIO[f] if f else min(PACKED_RATIO.values()) for f in fams)
    packed_fold, packed_reg = fold * nnz / ratio, (2 * sl - 6 + 64) * per_bit * nnz / ratio
    path = g["FrameV3Node_v1{2}"]
    sel_fixed = slots * (20 * path + g["LmsEqual_v1"])
    sel_types = slots * (max(1, (b["types"] - 1).bit_length()) * path + g["LmsEqual_v1"])
    copy = g["sha512_compression_rows"] / g["sha512_compression"]

    def law(fold_ands: float, hash_ands: float) -> dict:
        return {"ands": fold_ands + hash_ands} | layout(fold_ands + copy * hash_ands, fit, glue, bytes_per_nnz)

    return {"part": part, "Z": b["Z"], "slots": slots, "types": b["types"], "labels": b["labels"],
            "sessions_public": b["sessions_public"], "sl": sl, "nnz_read": nnz, "packed_ratio": ratio,
            "per_nonzero": {"weight": reps * w, "demux": reps * dm["ands"], "registration": regist, "total": fold + regist},
            "selection_fixed_depth20": sel_fixed, "selection_by_types": sel_types,
            "inner": {"D": s["D"], "rss_gib": s["rss_gib"], "prove_s": s["prove_s"]},
            "unpacked": law(fold * nnz, regist * nnz + sel_fixed), "packed": law(packed_fold, packed_reg + sel_fixed)}


NODE_GIB = 1716
K_LOG_MAX = 27
SLOTS = 16


def _law(fit: dict, key: str, D: float, cb: float, k: int) -> float:
    x = (1.0, D / 2 ** 30, cb / 2 ** 30, 2.0 ** k / 2 ** 30)  # F6 v2: (1, D, text, block); v1 has no block term
    return sum(a * v for a, v in zip(fit["coef"][key], x))


def layout(rows: float, fit: dict, glue: float, bytes_per_nnz: float) -> dict:
    """C-Flock's layout of a circuit of ``rows`` rows under the proved range (``Stmt.InRange``: a block is at most 2^27
    bits, ``IN_RANGE_K_LOG``; a block of r rows is 2^max(22, ceil(log2 r) + 2) bits, so at most 2^25 rows), in sessions of
    ``SLOTS`` blocks (F6's cap session: 16 slots at k_log 27, Z <= 2^29 nonzeros a slot), its text at ``RHO`` nonzeros a
    row (V*'s netlist is its own, shared by no slot), and F6's fitted prove time (summed over the sessions) and peak RSS
    (one session's). ``fits``: one block (``blocks``) or one cap session (``sessions``)."""
    one = max(22, math.ceil(rows).bit_length() + 2)
    blocks = 1 if one <= K_LOG_MAX else math.ceil(rows / 2 ** (K_LOG_MAX - 2))
    k = min(one, K_LOG_MAX)
    sessions = math.ceil(blocks / SLOTS)
    per = min(blocks, SLOTS)
    rows_per = rows / blocks * per
    D, cb = per * 2.0 ** k, rows_per * RHO * bytes_per_nnz + glue
    rss = _law(fit, "rss_gib", D, cb, k)
    prove = rows / rows_per * _law(fit, "prove_s", D, cb, k)
    return {"rows": rows, "k_log": k, "blocks": blocks, "sessions": sessions, "D_session": D, "D_total": blocks * 2.0 ** k,
            "circuit_bytes_session": cb, "rss_gib": rss, "prove_s": prove, "fits_block": blocks == 1,
            "fits_session": sessions == 1, "fits_node": rss <= NODE_GIB}


def _k2(path: Path):
    """proofs' F4 K2 count (art:5d4b3d5f…: ``k2.py`` and its ``units.json``, main's ``rec_outer.lowering`` rows per RecOpen)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("f4_k2", path / "k2.py")
    k2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k2)
    return k2, json.loads((path / "units.json").read_text())


def candidate3(k2, units: dict, Z: float, slots: int, sl: int, types: int, reps: int, fit: dict, glue: float,
               bytes_per_nnz: float) -> dict:
    """Candidate 3's V* rows for one session, from proofs' K2 count (note:proofs/20261010T0250Z-draft-f4-bitsumcheck §5.3):
    each slot's sparse phase over 2^nu entries (nu = ceil(log2 Z), its verifier's GF(2^128) products at 2,187 each, per rep),
    its firewall commitments (70 hm96 leaves per two reps) and its root_tau read (1e6), and one registered opening (both
    reps' claims at once, ``k2.opening`` at m = max(22, nu + ceil(log2(2 sl + 2))), plus 1,400 products of its own algebra)
    per slot, or per distinct type when a type's slots batch into one opening (§1.4). The rows are K2's (C-Flock rows of
    the RecOpen lowering); ``ands`` takes the hashing's rows back to ANDs at the compression's rows per AND."""
    nu = math.ceil(math.log2(Z))
    m = max(22, nu + math.ceil(math.log2(2 * sl + 2)))
    open_rows, grade, levels = k2.opening(units, m, True)
    open_alg = 1400 * k2.ROWS_PER_PRODUCT
    sparse = reps / 2 * k2.sparse_mults(nu, sl) * 2187
    firewall = reps / 2 * 70 * 142_282
    per_slot = sparse + firewall + 1e6
    copy = 81_402 / 58_120

    def total(openings: int) -> dict:
        rows = openings * (open_rows + open_alg) + slots * per_slot
        ands = openings * (open_rows / copy + open_alg) + slots * (sparse + (firewall + 1e6) / copy)
        return {"openings": openings, "ands": ands} | layout(rows, fit, glue, bytes_per_nnz)

    return {"nu": nu, "m": m, "grade": grade, "opening_rows": open_rows, "opening_levels": levels,
            "opening_algebra_rows": open_alg, "sparse_rows_per_slot": sparse, "firewall_rows_per_slot": firewall,
            "root_read_rows_per_slot": 1e6, "per_slot": total(1) if slots > 1 else None, "unbatched": total(slots),
            "batched": total(min(slots, types))}


SLOT_LOG = 18  # circuit.py's hm96 and sha512x3 slots (HM_LOG, SHA_LOG)
LINK_PORTS = ("acc", "acc_out", "out")  # RecOpen's rows another statement writes or reads: the chain and the top
STATE_BITS = 256  # what one acc row carries across: two GF(2^128) running sums (its words 0..15)


def _comps(words: int) -> int:
    return -(-(2 * words + 17) // 128) - 1


def recopen_block(L: int, H: int, useful: int, links: bool = True, n: int | None = None) -> dict:
    """``circuit.compose``'s ranges for a statement of ``n`` ``RecOpen{L, H}`` instances (`rec_outer`): per VU the unit's slot,
    one hm96 slot per row (rounded to a power of two) and its compressions three to a sha512x3 slot, placed by
    ``circuit._ranges``; ``G`` the most VUs (a power of two) a block of k_log <= 27 holds, ``k_log`` the block's at
    ``min(G, n)`` VUs. ``links=False`` drops the three registered rows (`LINK_PORTS`), the price of the cross-statement wires
    being the difference."""
    from verity_flock import circuit as FC
    from verity_flock import rec_open as R

    words = {"row": R.row_words(L), "aux": R.aux_words(H), "acc": R.ACC_WORDS, "coef": 16 * L, "dirs": R.DIRS_WORDS,
             "out": R.OUT_WORDS, "acc_out": R.ACC_WORDS}
    ports = [p for p in words if links or p not in LINK_PORTS]
    n_sha, n_hm, ul = -(-sum(_comps(words[p]) for p in ports) // 3), FC._pow2(len(ports)), FC._slot_log(useful)

    def k_log(G: int) -> int:
        return FC._ranges([("unit", SLOT_LOG, G * n_sha, {"net": "sha512x3", "packed": True}),
                           ("unit", SLOT_LOG, G * n_hm, {"net": "hm96"}), ("unit", ul, G, {"net": "unit"}),
                           ("mask", 14, 1, {"net": "mask"})])[1]

    G = 1
    while k_log(2 * G) <= K_LOG_MAX:
        G *= 2
    return {"unit_log": ul, "sha512x3": n_sha, "hm96": n_hm, "vu": 2 ** ul + (n_sha + n_hm) * 2 ** SLOT_LOG, "k_log_1": k_log(1),
            "G": G, "k_log": k_log(min(G, FC._pow2(n)) if n else G)}


_LOWERED: dict = {}


def _unit_rows(k2, units: dict, L: int, H: int) -> tuple[int, str, bool]:
    """``RecOpen{L, H}``'s unit rows, its grade and whether ``rec_open`` refuses H: K2's lowered count (``units.json``) where
    it has one at 1 <= H <= 16, else this checkout's own ``rec_outer.lowering(L, H)`` (H 0 to ``rec_open.DIRS_BITS``); a
    height ``rec_open`` refuses keeps K2's extrapolation, marked refused."""
    from verity_flock import rec_open as R
    from verity_flock import rec_outer as O

    if 1 <= H <= 16 and f"{L},{H}" in units:
        return units[f"{L},{H}"]["rows"], "C", False
    if not 0 <= H <= R.DIRS_BITS:
        return int(k2.unit_rows(units, L, H)[0]), "E", True
    if (L, H) not in _LOWERED:
        _LOWERED[L, H] = O.lowering(L, H).layout.useful
        print(f"  lowered RecOpen{{{L},{H}}}: {_LOWERED[L, H]:,} unit rows", flush=True)
    return _LOWERED[L, H], "C (lowered here)", False


def _session_law(blocks: int, k: int, text_rows: float, fit: dict, glue: float, bpn: float) -> dict:
    """F6's law over a statement of ``blocks`` blocks of 2^k, in cap sessions of ``SLOTS``; its text one netlist."""
    sessions, cb = math.ceil(blocks / SLOTS), text_rows * RHO * bpn + glue
    full, last = divmod(blocks, SLOTS)
    prove = full * _law(fit, "prove_s", SLOTS * 2.0 ** k, cb, k) + (last and _law(fit, "prove_s", last * 2.0 ** k, cb, k))
    return {"blocks": blocks, "k_log": k, "sessions": sessions, "D_total": blocks * 2.0 ** k, "prove_s": prove,
            "rss_gib": _law(fit, "rss_gib", min(blocks, SLOTS) * 2.0 ** k, cb, k)}


def proofs_c3(path: Path, prover: Path) -> dict:
    """V*'s outer proof for candidate 3 as multi-block statements, on proofs' bucket table (``buckets.json`` of art:4e9ff89a…,
    ``k2.py`` and ``units.json`` beside it) and F6's law (``prover.json``, v2).

    ``rows_rule``: proofs' rows (V* alone, with T types, and with the inner table's openings) laid out by `layout`.
    ``split``: the statements C-Flock would prove, rec-v0's (`verity_flock.rec_vstage`): each Ligerito level of the registered
    opening and of the inner table's openings one statement of identical ``RecOpen{L, H}`` instances (`recopen_block`, from
    the circuit's own ranges, row commitments included), the opening's algebra one, and the sparse phase (its algebra, the
    firewall commitments of its messages, the type selects) one; the last two by `layout`'s rule, not being lowered.
    Statements are bound only through registered values (PROTOCOL.md §16.10 "Registered reads"): ``boundaries`` lists each
    crossing, its rows and the state it carries; ``link_rows`` and ``link_D`` what those rows cost in the blocks."""
    p, rows = json.loads(prover.read_text()), json.loads(path.read_text())
    fit, glue, bpn = p["fit"], p["glue_bytes"], p["bytes_per_nnz"]
    k2, units = _k2(path.parent)
    msg = 2 ** SLOT_LOG + 2 ** SLOT_LOG // 3  # a 64-word registered row: its hm96 slot and a third of a sha512x3 slot
    cache: dict = {}

    def recopen(L, H):
        if (L, H) not in cache:
            useful, grade, refused = _unit_rows(k2, units, L, H)
            on, off = recopen_block(L, H, int(useful)), recopen_block(L, H, int(useful), links=False)
            cache[L, H] = {"L": L, "H": H, "unit_rows": int(useful), "grade": grade, "refused": refused,
                           "link_D_per_instance": on["vu"] - off["vu"]} | on
        return cache[L, H]

    out = []
    for r in rows:
        T = r["types"] if r["templates"] > 1 else 1
        sts, refused, link_rows, link_D = [], [], 0, 0.0
        m = r["m_if_T_types"] if T > 1 else r["m"]
        for tag, mm, reg in (("open", m, True), ("inner", r["inner_m"], False)):
            for lv, (L, H, n) in enumerate(k2.statements(mm, reg)):
                f = recopen(L, H)
                blocks, k = math.ceil(n / f["G"]), recopen_block(L, H, f["unit_rows"], n=n)["k_log"]
                sts.append({"statement": f"{tag} L{lv}", "circuit": f"RecOpen{{{L},{H}}}", "instances": n, "G": f["G"],
                            "same_circuit": True} | _session_law(blocks, k, f["unit_rows"], fit, glue, bpn))
                link_rows += 3 * n
                link_D += n * f["link_D_per_instance"]
                if f["refused"]:
                    refused.append(f"{tag} L{lv}: RecOpen{{{L},{H}}}")
        sparse = r["terms"]["sparse algebra"] + r["terms"]["firewall commitments"] + r["terms"]["type selects"]
        for name, rws in (("opening algebra", r["terms"]["opening algebra"]), ("sparse phase", sparse)):
            lay = layout(rws, fit, glue, bpn)
            sts.append({"statement": name, "circuit": name, "rows": rws, "same_circuit": lay["blocks"] == 1}
                       | _session_law(lay["blocks"], lay["k_log"], rws / lay["blocks"], fit, glue, bpn))
        msgs = r["terms"]["firewall commitments"] / 142_282  # vstar_c3.MSG: rows of one firewall commitment
        boundaries = [
            {"from": "firewall (the coin server's records)", "to": "sparse phase", "rows": round(msgs), "state_bits": None,
             "what": "each new inner message, a registered row whose b||c the coin server recorded (rec-c<i>)"},
            {"from": "root_tau", "to": "sparse phase", "rows": int(T > 1), "state_bits": SLOTS * math.ceil(math.log2(T)) if T > 1 else 0,
             "what": "the slots' hidden types, one registered row (none with one template)"},
            {"from": "sparse phase", "to": "opening algebra", "rows": 1, "state_bits": STATE_BITS,
             "what": "the sparse sumcheck's final claim per rep at the opening's point (the point is coins: public)"},
            {"from": "block", "to": "next block (any RecOpen statement)", "rows": 1, "state_bits": STATE_BITS,
             "what": "the rec-acc chain: two running sums; every instance reads acc and writes acc_out"},
            {"from": "RecOpen statements", "to": "opening algebra", "rows": 4, "state_bits": 4 * STATE_BITS,
             "what": "each rep's first and last rec-acc rows"}]
        link_rows += round(msgs) + int(T > 1) + 2 + 4
        link_D += (round(msgs) + int(T > 1) + 2 + 4) * msg
        D = sum(s["D_total"] for s in sts)
        rule = {k: layout(r[f], fit, glue, bpn) for k, f in (("c3", "vstar_rows"), ("c3_T", "vstar_rows_if_T_types"),
                                                             ("total", "total_rows"))}
        out.append({"workload": r["workload"], "part": r["part"], "Z": r["Z"], "m": m, "inner_m": r["inner_m"], "T": T,
                    "rows_rule": rule, "split": sts, "boundaries": boundaries, "refused": refused,
                    "blocks": sum(s["blocks"] for s in sts), "sessions": sum(s["sessions"] for s in sts),
                    "distinct_circuits": len({s["circuit"] for s in sts}), "D_total": D, "link_rows": link_rows,
                    "link_D": link_D, "link_share": link_D / D, "prove_s": sum(s["prove_s"] for s in sts),
                    "rss_gib": max(s["rss_gib"] for s in sts)})

    print("RecOpen statements as composed (circuit._ranges, row commitments included): per VU, G VUs a k_log-27 block")
    print(f"{'circuit':>16} {'unit rows':>10} {'VU':>10} {'xunit':>5} {'k_log(1)':>8} {'G':>3} {'links/VU':>9} refused")
    for f in sorted(cache.values(), key=lambda f: (f["L"], f["H"])):
        print(f"{f'RecOpen{{{f["L"]},{f["H"]}}}':>16} {f['unit_rows']:>10,} {f['vu']:>10,} {f['vu'] / f['unit_rows']:>5.2f} "
              f"{f['k_log_1']:>8} {f['G']:>3} {f['link_D_per_instance']:>9,} {'H outside rec_open' if f['refused'] else ''}")
    print("\nper bucket: proofs' rows by layout's rule (total = V* + inner openings) vs the split's statements")
    print(f"{'workload':24} {'part':4} {'m':>3} {'im':>3} {'rule blk':>8} {'split blk':>9} {'stmts':>5} {'ses':>4} "
          f"{'prove s':>8} {'RSS':>6} {'link rows':>9} {'link D':>7} refused")
    for o in out:
        wl = o["workload"].removeprefix("vllm/").split("__")
        print(f"{wl[0] + '/' + wl[4] + '/' + wl[5] if len(wl) > 5 else o['workload'][:24]:24} {o['part']:4} {o['m']:>3} "
              f"{o['inner_m']:>3} {o['rows_rule']['total']['blocks']:>8} {o['blocks']:>9} {len(o['split']):>5} "
              f"{o['sessions']:>4} {o['prove_s']:>8.3g} {o['rss_gib']:>6.0f} {o['link_rows']:>9,} {o['link_share']:>7.1%} "
              f"{'; '.join(o['refused'])}")
    rng = lambda key, f=lambda o, k: o[k]: (min(f(o, key) for o in out), max(f(o, key) for o in out))
    summary = {"rule_blocks": {k: rng(k, lambda o, k: o["rows_rule"][k]["blocks"]) for k in ("c3", "c3_T", "total")},
               "rule_sessions": {k: rng(k, lambda o, k: o["rows_rule"][k]["sessions"]) for k in ("c3", "c3_T", "total")},
               "split_blocks": rng("blocks"), "split_sessions": rng("sessions"), "statements": rng("distinct_circuits"),
               "prove_s": rng("prove_s"), "rss_gib": rng("rss_gib"), "link_rows": rng("link_rows"),
               "link_share": rng("link_share"), "refused_rows": sum(bool(o["refused"]) for o in out), "rows": len(out)}
    print("\nsummary:", json.dumps(summary))
    return {"source": str(path), "prover": str(prover), "format": p["format"], "recopen": list(cache.values()),
            "summary": summary, "buckets": out}


AUDIT_S = (1, 4, 16, 64, None)  # inner sessions one outer proof covers; None: all of the bucket's


def _rows_template(rows: float, s: int) -> tuple[int, int]:
    """``s`` instances of an unlowered circuit of ``rows`` rows as one template statement, by `layout`'s rule: an instance of
    k_log k1 = max(22, ceil(log2 rows) + 2), 2^(27 - k1) of them a block (or, past 27, each its own blocks of 2^25 rows)."""
    k1 = max(22, math.ceil(rows).bit_length() + 2)
    if k1 > K_LOG_MAX:
        return s * math.ceil(rows / 2 ** (K_LOG_MAX - 2)), K_LOG_MAX
    G = 2 ** (K_LOG_MAX - k1)
    return math.ceil(s / G), k1 + (min(G, 1 << (s - 1).bit_length()).bit_length() - 1)


def per_audit(path: Path, prover: Path) -> dict:
    """Outer prover-hours per audit when one outer proof covers S inner sessions of one bucket (`AUDIT_S`), on proofs'
    candidate-3 rows and F6's per-bucket session counts (``prover.json``'s unsplit buckets, mean and public, the rows proofs'
    table is keyed by) and law.

    One outer proof over s sessions of a bucket proves: the registered opening once (its RecOpen statements; the s sessions'
    2s claims enter its Ligerito part as one ring-switch round each, ``rec_live.rep_rounds``, which no query or H depends on),
    and per session the opening algebra (proofs' whole-opening size, an upper bound on the per-claim part), the sparse phase
    and the inner table's openings (fresh commitments each session). Equal circuits merge into one statement: the s sessions'
    instances of each inner level's ``RecOpen{L, H}`` (`recopen_block`, row commitments included), and the algebra and
    sparse phase as template statements (`_rows_template`). Each statement's cap sessions cost F6's law (`_session_law`)."""
    p, rows = json.loads(prover.read_text()), json.loads(path.read_text())
    fit, glue, bpn = p["fit"], p["glue_bytes"], p["bytes_per_nnz"]
    k2, units = _k2(path.parent)
    by = {(r["workload"], r["part"], r["Z"]): r for r in rows}
    blk: dict = {}

    def recopen(L, H, n):
        if (L, H, n) not in blk:
            u = int(_unit_rows(k2, units, L, H)[0])
            f = recopen_block(L, H, u, n=n)
            blk[L, H, n] = _session_law(math.ceil(n / f["G"]), f["k_log"], u, fit, glue, bpn)
        return blk[L, H, n]

    def outer(r: dict, s: int) -> dict:
        """One outer proof over ``s`` sessions of bucket ``r``: seconds by part, the largest statement's RSS."""
        m = r["m_if_T_types"] if r["templates"] > 1 else r["m"]
        reg = [recopen(L, H, n) for L, H, n in k2.statements(m, True)]
        inner = [recopen(L, H, s * n) for L, H, n in k2.statements(r["inner_m"], False)]
        sparse = r["terms"]["sparse algebra"] + r["terms"]["firewall commitments"] + r["terms"]["type selects"]
        rest = [_session_law(*_rows_template(x, s), x, fit, glue, bpn) for x in (r["terms"]["opening algebra"], sparse)]
        return {"registered": sum(x["prove_s"] for x in reg), "per_session": sum(x["prove_s"] for x in rest),
                "inner": sum(x["prove_s"] for x in inner), "rss_gib": max(x["rss_gib"] for x in reg + inner + rest),
                "blocks": sum(x["blocks"] for x in reg + inner + rest)}

    def bucket(r: dict, N: float, S: int | None, public: bool) -> dict:
        """Expected seconds by part for N sessions in outer proofs of at most S (public: N whole, split as evenly as can be;
        mean: N/S proofs of S, or one of N interpolated between its whole neighbours)."""
        S = S or math.ceil(N)
        add = lambda acc, o, w: {k: acc.get(k, 0) + w * v for k, v in o.items() if k != "rss_gib"} | {
            "rss_gib": max(acc.get("rss_gib", 0), o["rss_gib"])}
        acc: dict = {}
        if public:
            P = math.ceil(N / S)
            for i in range(P):
                acc = add(acc, outer(r, int(N) // P + (i < int(N) % P)), 1)
        elif N > S:
            acc = add(acc, outer(r, S), N / S)
        elif N < 1:
            acc = add(acc, outer(r, 1), N)
        else:
            lo, f = math.floor(N), N - math.floor(N)
            acc = add(add(acc, outer(r, lo), 1 - f), outer(r, lo + 1), f) if f else add(acc, outer(r, lo), 1)
        return acc

    out = []
    for w in p["workloads"]:
        res = {"workload": w["workload"], "buckets": [], "hours": {}}
        bks = [(part, b, by[w["workload"], part, b["Z"]]) for part in ("body", "tail") for b in w["unsplit"][part]]
        for part, b, r in bks:
            res["buckets"].append({"part": part, "Z": b["Z"], "types": b["types"], "templates": r["templates"],
                                   "sessions_mean": b["sessions_mean"], "sessions_public": b["sessions_public"],
                                   "m": r["m_if_T_types"] if r["templates"] > 1 else r["m"], "inner_m": r["inner_m"],
                                   "one_session": outer(r, 1)})
        for cnt in ("mean", "public"):
            for S in AUDIT_S:
                tot: dict = {}
                for part, b, r in bks:
                    x = bucket(r, b[f"sessions_{cnt}"], S, cnt == "public")
                    tot = {k: (max if k == "rss_gib" else lambda a, c: a + c)(tot.get(k, 0), v) for k, v in x.items()}
                h = {k: v / 3600 for k, v in tot.items() if k not in ("rss_gib", "blocks")}
                res["hours"][f"{cnt}/S={S or 'all'}"] = {"vstar": h["registered"] + h["per_session"],
                                                       "with_inner": h["registered"] + h["per_session"] + h["inner"],
                                                       **h, "blocks": tot["blocks"], "rss_gib": tot["rss_gib"]}
            res["hours"][f"{cnt}/floor"] = {
                "vstar": sum(b[f"sessions_{cnt}"] * (outer(r, 64)["per_session"] / 64) for _, b, r in bks) / 3600,
                "with_inner": sum(b[f"sessions_{cnt}"] * ((outer(r, 64)["per_session"] + outer(r, 64)["inner"]) / 64)
                                  for _, b, r in bks) / 3600}
        out.append(res)

    print("\nper audit, outer prover-hours (composed RecOpen with row commitments; F6 unsplit sessions; one outer proof over at "
          "most S sessions of one bucket). V* alone / with the inner table's openings")
    cols = [f"S={S or 'all'}" for S in AUDIT_S] + ["floor"]
    print(f"{'workload':34} {'cnt':6} {'sessions':>8} " + " ".join(f"{c:>13}" for c in cols))
    for res in out:
        wl = res["workload"].removeprefix("vllm/").split("__")
        for cnt in ("mean", "public"):
            n = sum(b[f"sessions_{cnt}"] for b in res["buckets"])
            print(f"{wl[0] + '/' + wl[4] + '/' + wl[5] + ('/' + wl[7] if wl[7] != 'mixed' else ''):34} {cnt:6} {n:>8.0f} "
                  + " ".join(f"{res['hours'][f'{cnt}/{c}']['vstar']:>5.1f}/{res['hours'][f'{cnt}/{c}']['with_inner']:<7.1f}"
                             for c in cols))
    rng = lambda key, part: (min(o["hours"][key][part] for o in out), max(o["hours"][key][part] for o in out))
    summary = {f"{key}/{part}": rng(key, part) for key in out[0]["hours"] for part in ("vstar", "with_inner")}
    summary["rss_gib_max"] = max(o["hours"][k]["rss_gib"] for o in out for k in o["hours"] if "rss_gib" in o["hours"][k])
    print("\nsummary:", json.dumps(summary))
    return {"source": str(path), "prover": str(prover), "format": p["format"], "S": [S or "all" for S in AUDIT_S],
            "summary": summary, "workloads": out}


def buckets(path: Path, reps: int, depth: int, c3: Path | None = None) -> dict:
    p = json.loads(path.read_text())
    g, dm = gadgets(), demux()
    reg = registration(g, 65536, depth)
    rows = [(w["workload"], part, b) for w in p["workloads"] for part in ("body", "tail") for b in w["unsplit"][part]]
    weights = {sl: fold_weight(sl) for sl in sorted({b["session"]["k_log"] for _, _, b in rows})}
    out = [{"workload": wl} | bucket_cost(b, part, g, weights, dm, reg, reps, p["fit"], p["glue_bytes"], p["bytes_per_nnz"])
           for wl, part, b in rows]
    if c3:
        k2, units = _k2(c3)
        for r, (_, _, b) in zip(out, rows):
            r["candidate3"] = candidate3(k2, units, b["Z"], b["slots"], b["session"]["k_log"], b["types"], reps, p["fit"],
                                         p["glue_bytes"], p["bytes_per_nnz"])
    print("weight per entry and rep (sl: chunks -> ANDs):", {sl: (w["chunks"], w["ands"]) for sl, w in weights.items()})
    print(f"demux {dm['ands']}; registration, 64 KiB rows read whole: {reg['whole_per_bit']:.1f} ANDs per bit; "
          f"selection path per slot at depth 20: {20 * g['FrameV3Node_v1{2}'] + g['LmsEqual_v1']:,}")
    print(f"\nV* per session, {reps} reps; rows = fold ANDs + {g['sha512_compression_rows'] / g['sha512_compression']:.2f} x "
          f"hash ANDs, in blocks of k_log <= {K_LOG_MAX} (2^{K_LOG_MAX - 2} rows) and cap sessions of {SLOTS} blocks; "
          f"F6's law ({p['format']}) per cap session, text = rows x {RHO} x {p['bytes_per_nnz']} B; prove summed")
    print(f"{'workload':24} {'part':4} {'Z/slot':>8} {'sl':>3} {'slots':>5} {'types':>5} {'nz/nz':>7} {'unpacked':>9} "
          f"{'blocks':>8} {'nz/pk':>5} {'packed':>9} {'blocks':>8} {'sessions':>8} {'prove s':>9} {'RSS GiB':>8} {'sel/sess':>9}")
    for r in out:
        u, k = r["unpacked"], r["packed"]
        wl = r["workload"].removeprefix("vllm/").split("__")
        print(f"{wl[0] + '/' + wl[4] + '/' + wl[5]:24} {r['part']:4} {r['Z']:>8.3g} {r['sl']:>3} {r['slots']:>5} {r['types']:>5} "
              f"{r['per_nonzero']['total']:>7,.0f} {u['ands']:>9.2e} {u['blocks']:>8.2e} {r['packed_ratio']:>5.2f} "
              f"{k['ands']:>9.2e} {k['blocks']:>8.2e} {k['sessions']:>8.2e} {k['prove_s']:>9.3g} {k['rss_gib']:>8.3g} "
              f"{r['selection_by_types']:>9.2e}")
    if c3:
        print(f"\ncandidate 3 (proofs' K2 count), V* per session, {reps} reps: one registered opening per slot (unbatched) or "
              f"per distinct type (batched), and one inner slot alone; same layout and law. blk: k_log-{K_LOG_MAX} blocks; "
              f"ses: cap sessions of {SLOTS} blocks; 'vs opt4': option 4 packed rows / unbatched")
        print(f"{'workload':24} {'part':4} {'Z/slot':>8} {'slots':>5} {'types':>5} {'m':>3} {'ANDs':>9} {'rows':>9} {'blk':>4} "
              f"{'ses':>3} {'prove s':>8} {'RSS GiB':>8} | {'batched':>9} {'blk':>4} {'ses':>3} {'prove s':>8} | "
              f"{'1 slot':>9} {'blk':>4} {'prove s':>8} | {'vs opt4':>7}")
        for r in out:
            c = r["candidate3"]
            u, bt, one = c["unbatched"], c["batched"], c["per_slot"] or c["unbatched"]
            wl = r["workload"].removeprefix("vllm/").split("__")
            print(f"{wl[0] + '/' + wl[4] + '/' + wl[5]:24} {r['part']:4} {r['Z']:>8.3g} {r['slots']:>5} {r['types']:>5} "
                  f"{c['m']:>3} {u['ands']:>9.2e} {u['rows']:>9.2e} {u['blocks']:>4} {u['sessions']:>3} {u['prove_s']:>8.3g} "
                  f"{u['rss_gib']:>8.3g} | {bt['rows']:>9.2e} {bt['blocks']:>4} {bt['sessions']:>3} {bt['prove_s']:>8.3g} | "
                  f"{one['rows']:>9.2e} {one['blocks']:>4} {one['prove_s']:>8.3g} | {r['packed']['rows'] / u['rows']:>7.0f}")
    return {"source": str(path), "reps": reps, "gadgets": g, "demux": dm, "weights": weights, "registration": reg,
            "rho": RHO, "node_gib": NODE_GIB, "packed_ratio": PACKED_RATIO, "candidate3_source": str(c3) if c3 else None,
            "buckets": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--buckets", type=Path, help="F6's prover.json: V* per session at each unsplit bucket instead")
    ap.add_argument("--candidate3", type=Path, help="with --buckets: proofs' F4 K2 tree (k2.py, units.json), candidate 3 beside")
    ap.add_argument("--proofs-c3", type=Path, help="with --buckets (for F6's law): proofs' candidate-3 buckets.json (k2.py and "
                                                  "units.json beside it), V*'s outer proof as multi-block statements alone")
    ap.add_argument("--per-audit", action="store_true", help="with --proofs-c3: outer prover-hours per audit over S sessions")
    ap.add_argument("--units", default=",".join(UNITS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--row-bits", type=int, nargs="*", default=[1024, 8192, 65536])
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--one-unit")
    a = ap.parse_args()
    if a.one_unit:
        print(json.dumps(unit_stats(a.one_unit)))
        return
    out = a.out or (Path(os.environ["RESEARCH_RUN_DIR"]) / "out" if os.environ.get("RESEARCH_RUN_DIR") else None)
    if a.proofs_c3:
        res = proofs_c3(a.proofs_c3, a.buckets)
        if out:
            out.mkdir(parents=True, exist_ok=True)
            (out / "proofs_c3_multiblock.json").write_text(json.dumps(res, indent=1))
        if a.per_audit:
            aud = per_audit(a.proofs_c3, a.buckets)
            if out:
                (out / "per_audit.json").write_text(json.dumps(aud, indent=1))
        return
    if a.buckets:
        res = buckets(a.buckets, a.reps, a.depth, a.candidate3)
        if out:
            out.mkdir(parents=True, exist_ok=True)
            (out / "bucket_cost.json").write_text(json.dumps(res, indent=1))
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
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "nonzero_cost.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
