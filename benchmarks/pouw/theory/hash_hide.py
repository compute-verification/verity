"""Hiding and zero knowledge for the proposals that reduce PoUW's hashing, priced: what each costs to hide, as a share of
h (honest hashing over stock in W1) and of T1's budget, and what it leaks if it is published.

    uv run python benchmarks/pouw/theory/hash_hide.py OUT_JSON

The parts:

- ``ands``: the in-circuit AND gates a drawn unit spends a committed 32-bit word, counted by compiling the tree's own
  Boolean Definitions: one SHA-512 compression (32 words) and the keyed folds li, duo, fx1 and xin (32 words each, the
  compute protocol's receipts, the one-hash rule's named Duo exception). Beside them, by the textbook counts (an estimate),
  a keyed 32-bit multiply, a carry-less 64-bit multiply over GF(2^64), a multiply mod 2^61 - 1 and a random byte lookup
  (one-hot decoder; its output XORs are free).
- ``per_value``: the hashing a value costs when it is hidden on its own (its own published digest) rather than inside the
  forward's words tree under one hm96 instance (layout L1): hm96 with the forced salt derivation (a), the pad
  b = x xor SHA-512(ks || salt || pos), the keyed digest, and the keyed leaf (SHA-512 started from the window's secret
  midstate S), each with the value's own frame leaf where the mechanism needs one. As a share of the lead's hashing a
  useful MAC, one hidden value every M useful MACs costs 888 c / M for c compressions, since the lead hashes 4 bytes a
  word and a word carries 27.75 useful MACs (the HASH card's basis lead).
- ``sketch``: h over the lead's h for a b-byte sketch committed in place of every s words, inside the words tree (no
  hiding cost beyond its bytes), and with each sketch hidden on its own under each mechanism of ``per_value``; and h in
  W1 a useful MAC for a 4-byte keyed-fold receipt every s words, with the fold's own work (its registers written a word,
  counted from its word form: an estimate) beside the receipt's hashing.
- ``rounds``: chained rounds, R commit-then-coin rounds a call. Each round publishes one hiding root before its coin: under
  L1 + (a) one hm96 instance, so the hiding term (T1's honest-only part) is R times L1's term at one instance a call (the
  inputs are L1's term at each point); under the keyed leaf every root is a keyed digest and costs nothing.
- ``rounds_keyed``: the same rounds under the keyed leaf on Llama-3.2-1B's linears at the five points. Hiding costs 0;
  what a round adds is chunks, since each round's words close their own last chunk: `chunk_head` registers a chunk in
  T1 (the card's honest-only first-block excess) and, credited, a padding block, a part-filled block and a node in h.
  Beside them, the round root's wait on the critical path, priced as the card prices A's root (its latencies), as a
  multiple of that per-call wait.
- ``grinding``: tries a window at a Fiat-Shamir seed, for each re-roll knob: an hm96 leaf's blind (an unforced mask,
  the seed reading the public leaf), the data (the seed reading the public leaf under a forced mask, or a private root
  or the keyed digest under any mask) and the floor any knob pays (a fresh root and the seed's last block), each
  priced in compressions at ShaWrites' cheapest form against one H200's, a datacenter's and the world's compute a
  window. The window's live salt bounds a grind to its window. From T: the noised coordinates m a steer needs at
  mu = 1/2 a coordinate for 2^-40 a window and pooled over a week, month and year, and the premium ln(T/delta_g)
  against the T_q = 2^128 the grinding premium reads (and 2^80, its other reading).
- ``fold_seal``: the XOR fold of a cell's words sealed with the cell's final word O' (one sigma a cell or a row of 8
  cells): h in the words tree and with each seal hidden on its own, counting sigma alone and sigma with its O's; a drawn
  unit's committed output bits and size class; and, a chunk size at a time, the drawn unit's in-circuit opening against
  T1's `chunk_head`.
- ``xor_m``: one u32 for m consecutive words (their XOR; at m = 2 also trunc-16's layout, and at m 1,024 and 4,096 a
  sketch value): the same output bits, opening and `chunk_head` rows, h with each value hidden on its own, and at the
  five points hiding's T1 and a call's `chunk_head` on Llama-3.2-1B's linears.
- ``per_output``: one or two u32 values an output in place of its words (a multiply-xor fold's state, alone or with the
  last word, and a quadratic sketch): hiding's T1 under the keyed leaf and under L1 + (a), h with each output hidden on
  its own, a SHA-512 keystream pad over the values on a link, a drawn unit's output bits, size class and opening, and
  `chunk_head` on Llama-3.2-1B's linears a chunk size at a time.
- ``periods``: the rows above that a report gives per monitoring period, in AND gates a window, week, month and year
  for one H200, a datacenter and the world, scaled from each row's share of stock.
- ``leak``: a published sketch. A linear sketch s = sum_t r_t w_t (mod p) with public coins r is a public linear function
  of the running words; S sketches with independent r over an S-step chain recover every word, hence every step's
  partial product (checked exactly here on random int8 data), and its bits count in leakage as a share of a unit's
  262,144 output bits.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

R_WORDS = 3515.38                    # ShaWrites' registers a compression for the words' form
H_LEAD = {"basis": 127.5, "low": 119.5, "high": 233.0}   # the lead's h, W1 a useful MAC: the card's basis and two readings
WORD_BYTES, MACS_WORD, BLOCK = 4, 27.75, 128
LEAD_C_MAC = WORD_BYTES / BLOCK / MACS_WORD   # compressions a useful MAC in the lead: 1/888
REG = 32
# registers written a word absorbed, from each fold's word form: duo u = y^p^k, y = u a + z, z = rotl(u, r) + y; xin one
# rotl, xor and multiply-add a half step of two words; li one xor and multiply-add a half step of two words
FOLD_REGS = {"duo": 4.0, "xin": 1.5, "li": 1.0}
PRF_SHARE = 53_568 / 58_120          # a compression from round 8 over a whole one (hm96's PRF)
MASK_REGS = 16 * (12 * (20 + 10 + 3) + 37 + 32)   # M_key·y in the H200 byte-table form, registers written a salt
PAD_XOR_REGS = 16
FRAME_LEAF_PREFIX = 33 + 4 + 4 + (8 + 64) + 2 * (8 + 3) + (8 + 22) + 8
NODE = 2                             # a binary frame-v3 node's compressions past its midstate
UNIT_WORDS = 262_144 // 32           # a drawn unit's words at the largest public size class
ANDS_MAC = 2079 / 16
# L1 + (a)'s hiding term at one instance a call, % of stock (W1 floor / alone): the inputs to `rounds`
L1_TERM = {"decode, 1 row": (1.6070, 3.2200), "decode, 16 rows": (0.1004, 0.2012), "T 512, 64 rows": (0.0251, 0.0503),
           "T 512, 512 rows": (0.003139, 0.006289), "T 8,192": (0.0002, 0.00039)}
LEAD_T1 = {"decode, 1 row": None, "decode, 16 rows": None, "T 512, 64 rows": 0.066, "T 512, 512 rows": 0.0436,
           "T 8,192": 0.0407}
CALLS = 64
TARGET, BUDGET = 0.1, 1.0            # % of stock
P61 = (1 << 61) - 1

LLAMA = [("qkv", 2048, 3072), ("o", 2048, 2048), ("gate_up", 2048, 16384), ("down", 8192, 2048)]   # (name, K, N)
LAYERS = 16
POINT_ROWS = {"T 512, 64 rows": 64, "T 512, 512 rows": 512, "T 8,192": 8192, "decode, 16 rows": 16, "decode, 1 row": 1}
CHUNK = 262_144                      # the lead's words chunk, bytes
CHUNK_HEAD_REGS = 23                 # honest's first-block excess a chunk (the card's chunk_head): T1
PAD_REGS, CHAIN_REGS, FIRST_REGS = 2254, 3532, 3510   # ShaWrites a padding block, a chained and a first compression
COMPRESSION_S, LAUNCH_S = 4.23e-6, 2.7e-6             # the card's latencies: a compression on one thread, a launch
A_LEAF = 1024                        # A's leaf bytes
FOLD_UNITS_WORD = 16                 # half a three-input LOP3 a word
H200_MACS_S = 9.89e14 * 0.4
DC_GPUS = 100_000
WORLD_MACS_YEAR = 4.68e30
WINDOW_S = 600
PERIOD_S = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
TQ_LOG2, DELTA_LOG2, DELTA_G_LOG2 = 128, 40, 40 + 7   # the premium's T_q; 2^-40 the target; 2^-40/(2·64 rounds)
KEYED, HM96A = "keyed leaf (the value's leaf from S)", "hm96 + (a) over its own frame leaf"


def lead_steps(k: int) -> int:
    """The lead's k32 words a dot product: step 1 holds 20 data lanes, each later step 28."""
    return 1 + math.ceil((k - 20) / 28)


def size_class(bits: float) -> int:
    c = 16
    while c < bits:
        c *= 2
    return c


def blocks(n: int) -> int:
    return (n + 17 + 127) // 128


def ands() -> dict:
    from verity.core.protocols.pouw.circuit import segment as SG
    from verity.ml.circuits.evaluation import bits as BS
    from verity.ml.commitments.gates import lms

    sha = BS.compile(lms.Sha512Compress).counts["and"]
    out = {"SHA-512 compression": {"ands_32_words": sha, "ands_a_word": sha / 32, "label": "checked"}}
    for fold in ("Li", "Duo", "Fx1", "Xin"):
        n = BS.compile(getattr(SG, f"Pouw{fold}Absorb")).counts["and"]
        out[f"{fold.lower()} fold"] = {"ands_32_words": n, "ands_a_word": n / 32, "label": "checked"}
    est = {"keyed u32 multiply (both operands private)": 32 * 33 // 2 + 31 * 31 // 2 + 31,
           "GF(2^64) carry-less multiply, per 32-bit word": 64 * 64 // 2,
           "multiply mod 2^61 - 1 (60 data bits a coefficient), per 32-bit word": (61 * 61 + 60 * 61 + 61) * 32 // 60,
           "random byte lookup x4, per 32-bit word": 4 * (2 * (2 * 4 + 16) + 256)}
    out.update({k: {"ands_a_word": v, "label": "estimate"} for k, v in est.items()})
    for v in out.values():
        v["over_sha"] = v["ands_a_word"] / (sha / 32)
    unit = {"words recomputed (32 MACs a word)": UNIT_WORDS * 32 * ANDS_MAC,
            "SHA-512 over the unit's words": UNIT_WORDS * sha / 32}
    unit.update({f"{k} over the unit's words": UNIT_WORDS * v["ands_a_word"] for k, v in out.items() if "fold" in k})
    unit.update({f"a {c // 1024} KiB chunk opened whole": blocks(c) * sha for c in (16384, 65536, 262144)})
    return {"a_word": out, "a_drawn_unit": unit}


def per_value(n_bytes: int = 4) -> dict:
    """Compressions (register-equivalent at R_WORDS) a value hidden on its own: total honest hashing, and the honest-only
    part that no premise credits."""
    leaf = blocks(FRAME_LEAF_PREFIX + n_bytes)
    mask = MASK_REGS / R_WORDS
    hm = 3 * PRF_SHARE + 2
    mech = {
        HM96A: {"total": leaf + hm + mask, "honest_only": mask},
        "hm96 unforced over its own frame leaf": {"total": leaf + hm + mask, "honest_only": leaf + hm + mask},
        "pad (b = x xor PRF), x <= 64 bytes": {"total": 1 + PAD_XOR_REGS / R_WORDS, "honest_only": PAD_XOR_REGS / R_WORDS},
        "keyed digest f(S, pos || x)": {"total": 1.0, "honest_only": 0.0},
        KEYED: {"total": float(blocks(FRAME_LEAF_PREFIX + n_bytes) - 1), "honest_only": 0.0},
        "own frame leaf, unhidden, in a binary tree": {"total": float(leaf + NODE), "honest_only": 0.0},
    }
    for v in mech.values():
        v["macs_for_1pct_of_lead_h"] = 100 * v["total"] / LEAD_C_MAC
    return {"value_bytes": n_bytes, "lead_compressions_a_mac": LEAD_C_MAC, "mechanisms": mech}


def sketch(pv: dict) -> dict:
    out = {}
    for b in (1, 4, 8):
        for s in (1, 2, 4, 8, 10, 16, 32, 64, 256):
            base = b / (WORD_BYTES * s)
            row = {"in the words tree (L1)": base}
            for name, v in pv["mechanisms"].items():
                if name.startswith("own frame leaf") or name.startswith("hm96 unforced"):
                    continue
                row[f"each hidden on its own: {name}"] = base + v["total"] / (s * MACS_WORD) / LEAD_C_MAC
            out[f"{b} B a sketch, one sketch every {s} words"] = {
                "h_over_lead": row, "h": {k: {r: x * h for r, h in H_LEAD.items()} for k, x in row.items()}}
    folds = {}
    for f, regs in FOLD_REGS.items():
        work = regs * REG / MACS_WORD
        folds[f] = {"fold_w1_a_mac": work, "by_s": {
            s: {"h": H_LEAD["basis"] / s + work, "cut": H_LEAD["basis"] / (H_LEAD["basis"] / s + work)}
            for s in (4, 8, 16, 32, 64, 256)},
            "s_for_10x": H_LEAD["basis"] / (H_LEAD["basis"] / 10 - work) if work < H_LEAD["basis"] / 10 else None}
    out["keyed-fold receipt, 4 B every s words, in the words tree"] = folds
    return out


def rounds() -> dict:
    out = {}
    for point, (fl, al) in L1_TERM.items():
        per_r = {R: {"floor": fl * (R * CALLS + 1) / (CALLS + 1), "alone": al * (R * CALLS + 1) / (CALLS + 1)}
                 for R in (1, 2, 4, 8, 16, 64)}
        lead = LEAD_T1[point]
        room = None if lead is None else BUDGET - lead
        out[point] = {"L1 + (a), % of stock": per_r, "keyed leaf, % of stock": 0.0,
                      "max R at 0.1% (floor / alone)": [max(0, int((TARGET * (CALLS + 1) / x - 1) / CALLS)) for x in (fl, al)],
                      "max R in T1's room beside the lead (floor / alone)":
                          None if room is None else [max(0, int((room * (CALLS + 1) / x - 1) / CALLS)) for x in (fl, al)]}
    return out


def _a_tree(rows: int, k: int) -> int:
    leaves = math.ceil(rows * k / A_LEAF)
    return math.ceil(math.log2(leaves)) if leaves > 1 else 0


def rounds_keyed() -> dict:
    out = {}
    for point, rows in POINT_ROWS.items():
        stock = rows * LAYERS * sum(k * n for _, k, n in LLAMA)
        words_b = [rows * n * lead_steps(k) * WORD_BYTES for _, k, n in LLAMA]
        a_wait = [(math.ceil(A_LEAF / BLOCK) + 1 + 2 + 3 * _a_tree(rows, k) + 1) * COMPRESSION_S
                  + (2 + _a_tree(rows, k)) * LAUNCH_S for _, k, _ in LLAMA]

        def chunks(r: int) -> int:
            return LAYERS * sum(r * math.ceil(b / (r * CHUNK)) for b in words_b)

        per_r = {}
        for r in (1, 2, 4, 8, 16, 32, 64):
            extra = chunks(r) - chunks(1)
            trees = [math.ceil(math.log2(max(1, math.ceil(b / (r * CHUNK))))) for b in words_b]
            # a round's keyed root after its last word, the hashing keeping pace with the words: the last block and the
            # padding block, three compressions a level of the round's subtree, then the digest; a launch a level
            waits = [(2 + 3 * t + 1) * COMPRESSION_S + (2 + t) * LAUNCH_S for t in trees]
            h_extra = extra * (PAD_REGS + CHAIN_REGS * (1 + NODE)) * REG / stock
            per_r[r] = {"extra_chunks_a_forward": extra,
                        "t1_chunk_head_pct": 100 * extra * CHUNK_HEAD_REGS * REG / stock,
                        "h_credited_w1_a_mac": h_extra, "h_credited_over_lead": h_extra / H_LEAD["basis"],
                        "round_wait_us_by_linear": {name: 1e6 * w for (name, _, _), w in zip(LLAMA, waits)},
                        "added_wait_over_lead_a_root_wait": (r - 1) * sum(waits) / sum(a_wait),
                        "added_wait_ms_a_forward": 1e3 * (r - 1) * LAYERS * sum(waits),
                        "added_coin_round_trips_a_forward": (r - 1) * CALLS}
        fits = [r for r, v in per_r.items() if v["t1_chunk_head_pct"] <= TARGET]
        out[point] = {"stock_macs_a_forward": stock, "chunks_a_forward_at_R1": chunks(1), "hiding_t1_pct": 0.0,
                      "lead_a_root_wait_us_by_linear": {name: 1e6 * w for (name, _, _), w in zip(LLAMA, a_wait)},
                      "lead_a_root_wait_ms_a_forward": 1e3 * LAYERS * sum(a_wait),
                      "by_R": per_r, "max_R_listed_within_0.1pct": max(fits) if fits else 0}
    return out


def grinding() -> dict:
    """Tries a window at a seed for each re-roll knob, and what T then asks of a steer and of the premium."""
    compression_w1 = FIRST_REGS * REG
    window_macs = H200_MACS_S * WINDOW_S
    scopes = {"one H200": window_macs, "datacenter": window_macs * DC_GPUS,
              "world": WORLD_MACS_YEAR * WINDOW_S / PERIOD_S["year"]}
    windows = {p: s / WINDOW_S for p, s in PERIOD_S.items()}

    def at(log2_t: float) -> dict:
        return {"log2_tries_a_window": log2_t,
                "m_a_window": math.ceil(log2_t + DELTA_LOG2),
                "m_pooled": {p: math.ceil(log2_t + math.log2(w) + DELTA_LOG2) for p, w in windows.items()},
                "premium_over_tq": (log2_t + DELTA_G_LOG2) / (TQ_LOG2 + DELTA_G_LOG2)}

    out = {"budget_macs_a_window": scopes, "windows_a_period": windows, "compression_w1": compression_w1,
           "at T_q = 2^128": at(TQ_LOG2), "at T_q = 2^80": at(80), "points": {}}
    for point, rows in POINT_ROWS.items():
        d = min(_a_tree(rows, k) for _, k, _ in LLAMA)
        # an hm96 leaf's blind, when the seed reads the public leaf: c = H(sp || y) one compression past sp's constant
        # block (y's last 64 bytes), the leaf H(lp || b || c) two past lp's constant block (b || c fills one block, then
        # the padding block, so b's move by M_key's image of the change costs nothing more), the seed's two blocks that
        # hold the roots two. The data, when the seed reads a private root (L1 + (a)) or the keyed digest: the last
        # leaf's last block, three compressions a level of A's tree, the seed's two; read through the public leaf, the
        # leaf over the new root too. Any try needs a fresh root, a SHA-512 output, and the seed's last block: the floor
        blind, data, floor = 1 + 2 + 2, 1 + 3 * d + 2, 2
        knobs = {"hm96 blind, the seed reading the public leaf (unforced mask)": blind,
                 "data, the seed reading the public leaf (forced mask)": data + 2,
                 "data, the seed reading a private root or the keyed digest (any mask)": data,
                 "any knob, the floor": floor}
        row = {"cheapest_a_tree_levels": d, "knobs": {},
               "bits_forcing_the_mask_removes_when_the_seed_reads_the_public_leaf": math.log2((data + 2) / blind),
               "bits_the_keyed_leaf_or_a_private_root_removes": math.log2(data / blind),
               "bits_any_knob_adds_over_the_blind": math.log2(blind / floor)}
        for knob, c in knobs.items():
            row["knobs"][knob] = {"compressions_a_try": c, "by_scope": {
                s: at(math.log2(macs / (c * compression_w1))) for s, macs in scopes.items()}}
        out["points"][point] = row
    return out


def fold_seal(sha: int) -> dict:
    """The XOR fold of a cell's words, sealed with its final word O' under the forward's words root."""
    recompute = UNIT_WORDS * 32 * ANDS_MAC
    keep = (KEYED, HM96A)
    out = {}
    for k in (2048, 8192):
        wc = lead_steps(k)
        fold = FOLD_UNITS_WORD * wc / k
        for cells in (1, 8):
            row = {"words_a_cell": wc, "useful_macs_a_seal": cells * k, "fold_w1_a_mac": fold}
            for name, words in (("sigma alone", 1), ("sigma and its cells' O'", 1 + cells)):
                seal = H_LEAD["basis"] * words / (cells * wc)
                mech = per_value(WORD_BYTES * words)["mechanisms"]
                row[name] = {"h_seal_in_tree": seal, "h_in_tree": seal + fold,
                             "h_each_seal_hidden": {m: seal + fold + H_LEAD["basis"] * mech[m]["total"] / (cells * k)
                                                    / LEAD_C_MAC for m in keep}}
            cells_unit = UNIT_WORDS / wc
            bits = 32 * cells_unit * (1 + 1 / cells)
            seal_b = bits / 8
            unit_macs = cells_unit * k
            row["unit"] = {"cells": cells_unit, "committed_output_bits": bits, "size_class": size_class(bits),
                           "class_share_of_lead": size_class(bits) / 262_144, "seal_bytes": seal_b}
            row["by_chunk_kib"] = {}
            for c in (1024, 4096, 16384, 65536, 262144):
                opening = max(1, math.ceil(seal_b / c)) * blocks(c) * sha
                row["by_chunk_kib"][c // 1024] = {
                    "opening_ands": opening, "opening_over_recompute": opening / recompute,
                    "t1_chunk_head_pct": 100 * CHUNK_HEAD_REGS * REG * seal_b / (c * unit_macs)}
            out[f"K {k:,}, a seal {'a cell' if cells == 1 else 'a row of 8 cells'}"] = row
    return out


def xor_m(sha: int) -> dict:
    """One u32 committed every m consecutive words in their place: m = 1 is the lead, m = 2 also trunc-16's layout (two
    words' low halves a u32), and m in the thousands a sketch value (nh2, nh or xor at G = m). h divides the lead's
    whole h by m, A's rows included, so at large m it under-reads by A's rows. ``by_point`` takes a call's words as one
    stream and counts its last chunk part-filled."""
    recompute = UNIT_WORDS * 32 * ANDS_MAC
    mech = per_value()["mechanisms"]
    keep = (KEYED, HM96A)
    chunks = (1024, 4096, 16384, 65536, 262144)
    out = {}
    for m in (1, 2, 8, 11, 16, 31, 1024, 4096):
        bits = 32 * math.ceil(UNIT_WORDS / m)
        unit_b = bits / 8
        row = {"committed_output_bits": bits, "size_class": size_class(bits),
               "class_share_of_lead": size_class(bits) / 262_144, "h_hashing_in_tree": H_LEAD["basis"] / m,
               "h_each_hidden": {x: H_LEAD["basis"] / m + H_LEAD["basis"] * mech[x]["total"] / (m * MACS_WORD)
                                 / LEAD_C_MAC for x in keep}, "by_chunk_kib": {}, "by_point": {}}
        for c in chunks:
            opening = max(1, math.ceil(unit_b / c)) * blocks(c) * sha
            row["by_chunk_kib"][c // 1024] = {
                "opening_ands": opening, "opening_over_recompute": opening / recompute,
                "t1_chunk_head_pct": 100 * CHUNK_HEAD_REGS * REG / (c / WORD_BYTES * m * MACS_WORD)}
        for point, rows in POINT_ROWS.items():
            stock = rows * LAYERS * sum(k * n for _, k, n in LLAMA)
            row["by_point"][point] = {
                "hiding_t1_pct": {KEYED: 0.0, "L1 + (a), floor / alone": L1_TERM[point]},
                "t1_chunk_head_pct_by_chunk_kib": {ch // 1024: 100 * LAYERS * CHUNK_HEAD_REGS * REG * sum(
                    math.ceil(rows * n * lead_steps(k) * WORD_BYTES / m / ch) for _, k, n in LLAMA) / stock
                    for ch in chunks}}
        name = {1: "lead (every word)", 2: "xor-2 or trunc-16"}.get(m, f"xor-{m}")
        out[name if m < 1024 else f"sketch, a u32 every {m:,} words"] = row
    return out


def per_output(sha: int) -> dict:
    """u32 values committed an output in place of its words: the multiply-xor fold's state N_T (with the last word D_T,
    or alone with D_T bound by the output's own commitment) and the quadratic sketch sum_t w_t^2 mod 2^32."""
    recompute = UNIT_WORDS * 32 * ANDS_MAC
    keep = (KEYED, HM96A)
    out = {}
    for name, c in (("fold, N_T and D_T", 2), ("fold, N_T alone", 1), ("quadratic sketch", 1)):
        mech = per_value(WORD_BYTES * c)["mechanisms"]
        row = {"u32_an_output": c, "by_K": {}, "by_point": {}}
        for k in (2048, 8192):
            tree = H_LEAD["basis"] * c / lead_steps(k)
            outs = UNIT_WORDS / lead_steps(k)
            unit_b = WORD_BYTES * c * outs
            row["by_K"][k] = {
                "h_hashing_in_tree": tree,
                "h_each_output_hidden": {x: tree + H_LEAD["basis"] * mech[x]["total"] / k / LEAD_C_MAC for x in keep},
                "h_sha512_pad_on_a_link": tree * BLOCK / 64,
                "unit_outputs": outs, "committed_output_bits": 8 * unit_b, "size_class": size_class(8 * unit_b),
                "class_share_of_lead": size_class(8 * unit_b) / 262_144,
                "opening_by_chunk_kib": {ch // 1024: {
                    "ands": max(1, math.ceil(unit_b / ch)) * blocks(ch) * sha,
                    "over_recompute": max(1, math.ceil(unit_b / ch)) * blocks(ch) * sha / recompute}
                    for ch in (1024, 4096, 16384)}}
        row["h_mac_weighted"] = {
            "in_tree": mac_weighted({k: v["h_hashing_in_tree"] for k, v in row["by_K"].items()}),
            "sha512_pad_on_a_link": mac_weighted({k: v["h_sha512_pad_on_a_link"] for k, v in row["by_K"].items()})}
        for point, rows in POINT_ROWS.items():
            stock = rows * LAYERS * sum(k * n for _, k, n in LLAMA)
            row["by_point"][point] = {
                "hiding_t1_pct": {KEYED: 0.0, "L1 + (a), floor / alone": L1_TERM[point]},
                "t1_chunk_head_pct_by_chunk_kib": {ch // 1024: 100 * LAYERS * CHUNK_HEAD_REGS * REG * sum(
                    math.ceil(rows * n * WORD_BYTES * c / ch) for _, _, n in LLAMA) / stock
                    for ch in (1024, 4096, 16384, 65536, 262144)}}
        out[name] = row
    return out


def mac_weighted(by_k: dict) -> float:
    macs = {k: sum(kk * n for _, kk, n in LLAMA if kk == k) for k in by_k}
    return sum(macs[k] * v for k, v in by_k.items()) / sum(macs.values())


def periods(rows: dict[str, float]) -> dict:
    """Each row's share of stock (h as a multiple of it, a T1 term as a fraction) in AND gates a window, week, month
    and year, for one H200, the datacenter and the world: scaled from the share, at stock's ANDs a MAC."""
    h200 = H200_MACS_S * ANDS_MAC
    rate = {"one H200": h200, "datacenter": h200 * DC_GPUS, "world": WORLD_MACS_YEAR * ANDS_MAC / PERIOD_S["year"]}
    spans = {"window": WINDOW_S, **PERIOD_S}
    stock = {s: {p: r * t for p, t in spans.items()} for s, r in rate.items()}
    return {"stock_ands": stock,
            "rows": {name: {"share_of_stock": x, "ands": {s: {p: x * v for p, v in by.items()} for s, by in stock.items()}}
                     for name, x in rows.items()}}


def _solve_mod(m: list[list[int]], y: list[int], p: int) -> list[int] | None:
    n = len(m)
    a = [row[:] + [y[i]] for i, row in enumerate(m)]
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] % p), None)
        if piv is None:
            return None
        a[c], a[piv] = a[piv], a[c]
        inv = pow(a[c][c], p - 2, p)
        a[c] = [x * inv % p for x in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [(x - f * z) % p for x, z in zip(a[r], a[c])]
    return [a[i][n] for i in range(n)]


def leak(seed: int = 7, trials: int = 8) -> dict:
    """Recover an S-step chain's running words, and its partials, from S published linear sketches with public coins."""
    rng = random.Random(seed)
    res = []
    for t in range(trials):
        S = (8, 16, 32, 64)[t % 4]
        a = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(S)]
        b = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(S)]
        kappa = [sum(x * y for x, y in zip(a[i], b[i])) for i in range(S)]
        w, acc = [], 0
        for k in kappa:
            acc += k
            w.append(acc)
        r = [[rng.randrange(P61) for _ in range(S)] for _ in range(S)]
        sk = [sum(ri * wi for ri, wi in zip(row, w)) % P61 for row in r]
        got = _solve_mod(r, sk, P61)
        signed = None if got is None else [g - P61 if g > P61 // 2 else g for g in got]
        rec = None if signed is None else [signed[0]] + [signed[i] - signed[i - 1] for i in range(1, S)]
        res.append({"steps": S, "words_recovered": signed == w, "partials_recovered": rec == kappa})
    per_unit = {f"{b} B a sketch every {s} words": {"bits_a_unit": UNIT_WORDS // s * 8 * b,
                                                    "share_of_unit_output_bits": (UNIT_WORDS // s * 8 * b) / 262_144}
                for b in (4, 8) for s in (1, 8, 32, 256)}
    return {"recovery": res, "all_recovered": all(x["words_recovered"] and x["partials_recovered"] for x in res),
            "published_bits": per_unit}


def main() -> None:
    out_path = Path(sys.argv[1])
    pv = per_value()
    a = ands()
    sha = a["a_word"]["SHA-512 compression"]["ands_32_words"]
    out = {"ands": a, "per_value": pv, "sketch": sketch(pv), "rounds": rounds(), "rounds_keyed": rounds_keyed(),
           "grinding": grinding(), "fold_seal": fold_seal(sha), "xor_m": xor_m(sha), "per_output": per_output(sha),
           "leak": leak()}
    fs, po, rk = out["fold_seal"], out["per_output"], out["rounds_keyed"]
    o_seal = "sigma and its cells' O'"
    out["periods"] = periods({
        "the lead, h": H_LEAD["basis"],
        "fold seal, K 2,048, a cell, in the tree, h": fs["K 2,048, a seal a cell"][o_seal]["h_in_tree"],
        "fold seal, K 2,048, a cell, a keyed leaf a seal, h": fs["K 2,048, a seal a cell"][o_seal]["h_each_seal_hidden"][KEYED],
        "fold seal, K 8,192, a row of 8, in the tree, h": fs["K 8,192, a seal a row of 8 cells"][o_seal]["h_in_tree"],
        "xor-16, in the tree, h": out["xor_m"]["xor-16"]["h_hashing_in_tree"],
        "sketch a u32 every 1,024 words, in the tree, h (the u32s' hashing only)":
            out["xor_m"]["sketch, a u32 every 1,024 words"]["h_hashing_in_tree"],
        "sketch a u32 every 1,024 words, a keyed leaf each, h (the u32s' hashing only)":
            out["xor_m"]["sketch, a u32 every 1,024 words"]["h_each_hidden"][KEYED],
        "fold N_T and D_T, in the tree, h (hashing only)": po["fold, N_T and D_T"]["h_mac_weighted"]["in_tree"],
        "fold N_T and D_T, a SHA-512 pad on a shared link, h": po["fold, N_T and D_T"]["h_mac_weighted"]["sha512_pad_on_a_link"],
        "quadratic sketch, in the tree, h (hashing only)": po["quadratic sketch"]["h_mac_weighted"]["in_tree"],
        "T1's allowance": BUDGET / 100,
        "keyed rounds, T 512, 64 rows, R 16: T1 chunk_head": rk["T 512, 64 rows"]["by_R"][16]["t1_chunk_head_pct"] / 100,
        "keyed rounds, decode, 1 row, R 16: T1 chunk_head": rk["decode, 1 row"]["by_R"][16]["t1_chunk_head_pct"] / 100,
        "L1 + (a) rounds, decode, 1 row, R 1: T1 hiding (floor)": L1_TERM["decode, 1 row"][0] / 100})
    out_path.write_text(json.dumps(out, indent=1, sort_keys=False))
    a = out["ands"]["a_word"]
    print("in-circuit ANDs a 32-bit word (x SHA-512):")
    for k, v in a.items():
        print(f"  {k:48s} {v['ands_a_word']:9.1f}  {v['over_sha']:.3f}  {v['label']}")
    print("drawn unit (8,192 words), ANDs:")
    for k, v in out["ands"]["a_drawn_unit"].items():
        print(f"  {k:48s} {v:14,.0f}")
    print("a 4-byte value hidden on its own, compressions (total / honest-only) and MACs a value for 1% of the lead's h:")
    for k, v in pv["mechanisms"].items():
        print(f"  {k:48s} {v['total']:6.3f} / {v['honest_only']:6.3f}  {v['macs_for_1pct_of_lead_h']:12,.0f}")
    print("sketch h over the lead's (4 B a sketch):")
    for k, v in out["sketch"].items():
        if k.startswith("4 B"):
            print(f"  {k}: " + ", ".join(f"{n.split(': ')[-1][:18]} {x:.3f}" for n, x in v["h_over_lead"].items()))
    for f, v in out["sketch"]["keyed-fold receipt, 4 B every s words, in the words tree"].items():
        print(f"  {f} receipt: fold {v['fold_w1_a_mac']:.2f} W1 a MAC, s for 10x {v['s_for_10x']}; "
              + ", ".join(f"s {s}: h {x['h']:.2f} ({x['cut']:.1f}x)" for s, x in v["by_s"].items()))
    print("chained rounds:", json.dumps(out["rounds"], indent=None)[:2000])
    print("chained rounds under the keyed leaf (T1 chunk_head %, h over the lead's, added wait over the lead's A-root wait):")
    for p, v in out["rounds_keyed"].items():
        print(f"  {p:16s} max R within 0.1%: {v['max_R_listed_within_0.1pct']}; lead's A-root wait "
              f"{v['lead_a_root_wait_ms_a_forward']:.2f} ms a forward; " + ", ".join(
                  f"R {r}: {x['t1_chunk_head_pct']:.4f} / {x['h_credited_over_lead']:.5f} / "
                  f"{x['added_wait_over_lead_a_root_wait']:.2f} ({x['added_wait_ms_a_forward']:.1f} ms)"
                  for r, x in v["by_R"].items()))
    g = out["grinding"]
    print(f"grinding: at T_q = 2^128, m {g['at T_q = 2^128']['m_a_window']} a window; at 2^80, m "
          f"{g['at T_q = 2^80']['m_a_window']}, premium x{g['at T_q = 2^80']['premium_over_tq']:.2f}")
    for p, v in g["points"].items():
        for knob, x in v["knobs"].items():
            print(f"  {p:16s} {knob[:30]:30s} {x['compressions_a_try']:3d} compressions: " + ", ".join(
                f"{s} log2 T {y['log2_tries_a_window']:.1f}, m {y['m_a_window']}/yr {y['m_pooled']['year']}, "
                f"premium x{y['premium_over_tq']:.2f}" for s, y in x["by_scope"].items()))
    print("fold seals (h in the tree, sigma alone / with O'; keyed leaf a seal; class; 16 KiB opening / T1):")
    for p, v in out["fold_seal"].items():
        s, o = v["sigma alone"], v["sigma and its cells' O'"]
        c16 = v["by_chunk_kib"][16]
        print(f"  {p:28s} {s['h_in_tree']:.3f} / {o['h_in_tree']:.3f}; keyed {o['h_each_seal_hidden'][KEYED]:.2f}; "
              f"class {v['unit']['size_class']}; {c16['opening_ands']:,.0f} ANDs / {c16['t1_chunk_head_pct']:.5f}%")
    print("a u32 every m words (class; h in tree; keyed leaf / hm96 + (a) each; 16 KiB opening / T1; a call's T1 "
          "chunk_head at 16 KiB, T 512 64 rows / decode 1 row):")
    for p, v in out["xor_m"].items():
        c16, bp = v["by_chunk_kib"][16], v["by_point"]
        print(f"  {p:32s} class {v['size_class']:7d}; {v['h_hashing_in_tree']:.4f}; "
              f"{v['h_each_hidden'][KEYED]:.2f} / {v['h_each_hidden'][HM96A]:.1f}; "
              f"{c16['opening_ands']:,.0f} / {c16['t1_chunk_head_pct']:.5f}%; "
              f"{bp['T 512, 64 rows']['t1_chunk_head_pct_by_chunk_kib'][16]:.5f}% / "
              f"{bp['decode, 1 row']['t1_chunk_head_pct_by_chunk_kib'][16]:.5f}%")
    print("u32s an output (h in tree, keyed leaf each, pad; class; 1 KiB opening; T1 chunk_head 1 / 16 / 256 KiB):")
    for p, v in out["per_output"].items():
        for k, x in v["by_K"].items():
            print(f"  {p:18s} K {k:5d} {x['h_hashing_in_tree']:.3f}, {x['h_each_output_hidden'][KEYED]:.2f}, "
                  f"{x['h_sha512_pad_on_a_link']:.3f}; class {x['size_class']}; "
                  f"{x['opening_by_chunk_kib'][1]['ands']:,.0f}")
        print("    " + ", ".join(f"{q}: " + " / ".join(f"{y:.4f}" for c, y in z["t1_chunk_head_pct_by_chunk_kib"].items()
                                                         if c in (1, 16, 256)) for q, z in v["by_point"].items()))
    print("leak: all recovered =", out["leak"]["all_recovered"])
    print("per monitoring period, AND gates (window / week / month / year):")
    for name, v in out["periods"]["rows"].items():
        print(f"  {name[:56]:56s} {v['share_of_stock']:10.4g}  " + "  ".join(
            f"{s}: " + " / ".join(f"{x:.2e}" for x in by.values()) for s, by in v["ands"].items()))


if __name__ == "__main__":
    main()
