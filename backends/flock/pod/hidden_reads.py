"""Reads at a private, data-dependent index (value hiding, item 4): the constructions as gate circuits, their exact counts, and
``flock-circuit`` statements of two of them for C-Flock's CPU prover, M0 and ``--zk`` (``75-hidden-reads.sh``).

    python hidden_reads.py check                    each gadget against its native reference (hashlib, hm96, a plain read)
    python hidden_reads.py counts --out FILE [--heavy]
                                                    every construction's counts; --heavy adds today's read of the 2^27 tanh_rn table
    python hidden_reads.py stage --unit merkle|mux --n N [N ...] --out DIR [--depth D | --tab-log T --reads R]
                                                    circuit.txt, inst-N.bin, pub-N.bin (verity_flock.circuit's composite)

A count is the live cone of a circuit built with ``verity.ml.boolean.forms``: ``ands`` are products of two non-constant forms,
``rows`` every committed bit (``ands``, the SHA-512 gadget's committed copies, inputs, assertion rows): what Flock commits.

* (a) ``merkle``: an in-circuit opening of a hiding tree at a private position. The leaf is ``hm96-sha512/v1`` over a
  ``sha512/row/v1`` row of one block (64 u16 or 32 u32 entries; six compressions: the row block and its padding, the salt's two,
  ``tree_leaf``'s two), the entry picked inside the leaf by a mux tree, then per level a conditional swap (512 ANDs) and a node:
  ``c2`` is SHA-512 of the two digests (two compressions), ``c1`` the compression function alone from a tree constant IV.
* (b) ``mux``: a mux tree over committed contents, ``w (2^n - 1)`` ANDs for 2^n entries of w bits.
* (d) ``sort``: a batch of N reads of one committed table of T entries, by a Beneš network over the T table records, the N read
  records and padding (switch settings are witness), then per adjacent pair ``tag_k AND (x_{k-1} XOR x_k) = 0`` for every
  bit of (index, pad, value). Every read equals its predecessor, so by induction a table record of its index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

from verity.commitments import hm96
from verity.ml.boolean.forms import NOT, ONE, ZERO, Circuit, and_reduce, const_bits

from verity_flock import sha512_circuit as S

H5 = hm96.SHA512
KEY = H5.default_key
#: the one-compression node's IV: the chaining value after a constant tag block (a tree constant, never a witness)
C1_TAG = b"verity/hidden-reads/node-c1/v0".ljust(128, b"\0")
LEAF_WORDS = 64          # one SHA-512 block of u16 entries
ROW_PREFIX = S.sha512_row_prefix(LEAF_WORDS)


# --- counting ----------------------------------------------------------------------------------------------------------------

def counts(C: Circuit) -> dict:
    live = C.live()
    ands = copies = 0
    for v in live:
        if C.kind[v] == "and":
            a, b = C.data[v]
            if (a.is_const() and a.c) or (b.is_const() and b.c):
                copies += 1
            else:
                ands += 1
    ins = sum(1 for v in live if C.kind[v] == "in")
    return {"ands": ands, "copies": copies, "inputs": ins, "asserts": len(C.asserts), "rows": ands + copies + ins + len(C.asserts)}


def delta(f, *args) -> dict:
    """The counts ``f(C, *args)`` adds to a circuit whose inputs it creates itself (inputs excluded from rows)."""
    C = Circuit()
    outs = f(C, *args)
    C.output("o", [b for w in outs for b in (w if isinstance(w, list) else [w])])
    c = counts(C)
    c["rows"] -= c["inputs"]
    return c


# --- bits and words ----------------------------------------------------------------------------------------------------------

def cwords(vals) -> list[list]:
    return [const_bits(v, 64) for v in vals]


def to_words(bits: list) -> list[list]:
    """Byte-string-order bits (hm96's order; a u16 LE row's word i bit b is bit 16 i + b) as 64-bit message words."""
    assert len(bits) % 64 == 0
    w = [[None] * 64 for _ in range(len(bits) // 64)]
    for t, b in enumerate(bits):
        p = S.word_bit(t)
        w[p >> 6][p & 63] = b
    return w


def to_bits(words: list[list]) -> list:
    flat = [b for w in words for b in w]
    return [flat[S.word_bit(t)] for t in range(len(flat))]


# --- gadgets -----------------------------------------------------------------------------------------------------------------

def leaf_hm96(C, row: list, salt: list) -> list[list]:
    """``hm96.SHA512.tree_leaf(key, commit_string(key, SHA-512(row prefix || row), y))`` for a one-block row (1024 bits) and a
    192-byte salt (1536 bits), both in byte-string order: six compressions."""
    x = S.compress_gadget(C, cwords(S.midstate(ROW_PREFIX)), to_words(row))
    x = S.compress_gadget(C, x, cwords(S.words_be(S.pad_tail(len(ROW_PREFIX) + 2 * LEAF_WORDS))))
    yw = to_words(salt)
    c = S.compress_gadget(C, cwords(S.midstate(H5.salt_prefix)), yw[:16])
    tail = S.words_be(bytes(64) + S.pad_tail(128 + 192)[:64])[8:]
    c = S.compress_gadget(C, c, yw[16:] + cwords(tail))
    xs, rows = to_bits(x), H5.rows_of(KEY)
    b = []
    for i in range(512):
        acc, r = xs[i], rows[i]
        while r:
            j = (r & -r).bit_length() - 1
            acc = acc ^ salt[j]
            r &= r - 1
        b.append(acc)
    bc = b + to_bits(c)
    leaf = S.compress_gadget(C, cwords(S.midstate(H5.leaf_prefix(KEY))), to_words(bc))
    return S.compress_gadget(C, leaf, cwords(S.words_be(S.pad_tail(256))))


def node_c2(C, left: list[list], right: list[list]) -> list[list]:
    st = S.compress_gadget(C, cwords(S.IV), left + right)
    return S.compress_gadget(C, st, cwords(S.words_be(S.pad_tail(128))))


def node_c1(C, left: list[list], right: list[list]) -> list[list]:
    return S.compress_gadget(C, cwords(S.midstate(C1_TAG)), left + right)


NODES = {"c1": node_c1, "c2": node_c2}


def swap(C, s, x: list, y: list) -> tuple[list, list]:
    """``(x, y)`` if s is 0, else ``(y, x)``: one AND per bit."""
    d = [C.AND(s, a ^ b) for a, b in zip(x, y)]
    return [a ^ e for a, e in zip(x, d)], [b ^ e for b, e in zip(y, d)]


def merkle_root(C, leaf: list[list], sibs: list[list[list]], idx: list, node) -> list[list]:
    """The root over ``leaf`` at the private position ``idx`` (LSB first: bit k is the side at level k)."""
    cur = leaf
    for sib, b in zip(sibs, idx):
        lo, hi = swap(C, b, [x for w in cur for x in w], [x for w in sib for x in w])
        cur = node(C, [lo[64 * i:64 * i + 64] for i in range(8)], [hi[64 * i:64 * i + 64] for i in range(8)])
    return cur


def mux_tree(C, idx: list, entries: list[list]) -> list:
    """``entries[idx]``: fold one index bit per level, ``w (2^n - 1)`` ANDs."""
    level = list(entries)
    assert len(level) == 1 << len(idx)
    for b in idx:
        level = [[e0 ^ C.AND(b, e0 ^ e1) for e0, e1 in zip(level[2 * i], level[2 * i + 1])] for i in range(len(level) // 2)]
    return level[0]


def band_rows(lo: int = 118, hi: tuple[int, int] = (128, 129)) -> list[tuple[int, int]]:
    """``elementwise.bf16_band``'s rows (sign, even exponent field e0), 256 entries each."""
    return [(s, e0) for s in (0, 1) for e0 in range(lo & ~1, hi[s] + 1, 2)]


def band_hidden(C, g: list, table: list[list[list]]) -> list:
    """``bf16_band``'s 15 magnitude bits with the band's contents committed (``table[row][k]``, 15 bits): per row the AND of
    its minterm (sign and e[1:8]) with a mux tree over the 256 low minterms (mantissa and e[0])."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    out = [ZERO] * 15
    for (sign, e0), ent in zip(band_rows(), table):
        lit = [e[1 + i] if (e0 >> 1) >> i & 1 else NOT(e[1 + i]) for i in range(7)] + [s if sign else NOT(s)]
        row = and_reduce(C, lit)
        v = mux_tree(C, m + [e[0]], ent)
        out = [o ^ C.AND(row, x) for o, x in zip(out, v)]
    return out


def benes(C, recs: list[list], sw: list) -> list[list]:
    """A Beneš network over 2^k records: the first layer's switches, the top half, the bottom half, the last layer's (``sw`` is
    consumed in that order; M log M - M / 2 switches)."""
    M = len(recs)
    if M == 2:
        return list(swap(C, sw.pop(0), recs[0], recs[1]))
    top, bot = [], []
    for j in range(M // 2):
        a, b = swap(C, sw.pop(0), recs[2 * j], recs[2 * j + 1])
        top.append(a)
        bot.append(b)
    top, bot = benes(C, top, sw), benes(C, bot, sw)
    out = []
    for j in range(M // 2):
        out.extend(swap(C, sw.pop(0), top[j], bot[j]))
    return out


def benes_route(perm: list[int]) -> list[int]:
    """Switch settings in :func:`benes`'s order so that input i reaches output ``perm[i]`` (the looping algorithm)."""
    M = len(perm)
    if M == 2:
        return [0 if perm[0] == 0 else 1]
    inv = [0] * M
    for i, o in enumerate(perm):
        inv[o] = i
    side: list = [None] * M                # per input: 0 top, 1 bottom
    for start in range(0, M, 2):
        if side[start] is not None:
            continue
        i, side[start] = start, 0
        while True:
            p = i ^ 1                      # an input switch's two inputs take different halves
            side[p] = 1 - side[i]
            q = inv[perm[p] ^ 1]           # and so do the two inputs of an output switch
            if side[q] is not None:
                break
            side[q], i = side[i], q
    first = [0 if side[2 * j] == 0 else 1 for j in range(M // 2)]
    tp, bp = [0] * (M // 2), [0] * (M // 2)
    for i in range(M):
        (tp if side[i] == 0 else bp)[i // 2] = perm[i] // 2
    last = [0] * (M // 2)
    for i in range(M):
        if side[i] == 0:
            last[perm[i] // 2] = 0 if perm[i] % 2 == 0 else 1
    return first + benes_route(tp) + benes_route(bp) + last


def sorted_read(C, table: list[list], reads: list[tuple[list, list]], n: int, M: int, sw: list) -> None:
    """Assert every read ``(idx, val)`` is ``table[idx]``: records ``idx | pad | val | tag`` (table 0, read 1, padding pad = 1)
    through :func:`benes` (M a power of two), the first record not a read, each read equal to its predecessor."""
    w = len(table[0])
    recs = [const_bits(i, n) + [ZERO] + list(t) + [ZERO] for i, t in enumerate(table)]
    recs += [list(i) + [ZERO] + list(v) + [ONE] for i, v in reads]
    recs += [[ZERO] * n + [ONE] + [ZERO] * w + [ZERO]] * (M - len(recs))
    out = benes(C, recs, sw)
    C.assert_zero("first", out[0][-1])
    for k in range(1, M):
        t = out[k][-1]
        for a, b in zip(out[k - 1][:-1], out[k][:-1]):
            C.assert_zero(f"r{k}", C.AND(t, a ^ b))


def sort_witness(table_n: int, idx: list[int], M: int) -> list[int]:
    """The honest routing for :func:`sorted_read`: records sorted by (pad, index, tag)."""
    keys = [(0, i, 0, i) for i in range(table_n)] + [(0, x, 1, table_n + r) for r, x in enumerate(idx)]
    keys += [(1, 0, 0, table_n + len(idx) + p) for p in range(M - table_n - len(idx))]
    order = sorted(keys)
    perm = [0] * M
    for pos, k in enumerate(order):
        perm[k[3]] = pos
    return benes_route(perm)


# --- counts ------------------------------------------------------------------------------------------------------------------

def _merkle_parts() -> dict:
    def leaf(C):
        return leaf_hm96(C, C.input("row", 1024), C.input("salt", 1536))

    def level(C, node):
        a, b, s = C.input("a", 512), C.input("b", 512), C.input("s", 1)[0]
        lo, hi = swap(C, s, a, b)
        return node(C, [lo[64 * i:64 * i + 64] for i in range(8)], [hi[64 * i:64 * i + 64] for i in range(8)])

    def pick(C, n, w):
        return mux_tree(C, C.input("i", n), [C.input(f"e{k}", w) for k in range(1 << n)])

    comp = delta(lambda C: S.compress_gadget(C, [C.input(f"cv{i}", 64) for i in range(8)], [C.input(f"m{i}", 64) for i in range(16)]))
    return {"compression": comp, "leaf_hm96": delta(leaf), "level_c1": delta(level, node_c1), "level_c2": delta(level, node_c2),
            "pick_64x16": delta(pick, 6, 16), "pick_32x31": delta(pick, 5, 31)}


def _add(*cs, k=None) -> dict:
    out = {"ands": 0, "rows": 0}
    for c, n in cs:
        out["ands"] += n * c["ands"]
        out["rows"] += n * c["rows"]
    return out


def mux_count(n: int, w: int) -> dict:
    """(b) for 2^n entries of w bits: ``w (2^n - 1)`` ANDs (checked by building at small n)."""
    a = w * ((1 << n) - 1)
    return {"ands": a, "rows": a}


def sort_count(T: int, N: int, n: int, w: int) -> dict:
    """(d) for N reads of a T-entry table: M = the next power of two of T + N records of r = n + w + 2 bits."""
    M = 1 << max(1, (T + N - 1).bit_length())
    k = M.bit_length() - 1
    sw = M * k - M // 2
    r = n + w + 2
    ands = sw * r + (M - 1) * (r - 1)
    return {"M": M, "switches": sw, "record_bits": r, "ands": ands, "rows": ands + sw + 1 + (M - 1) * (r - 1),
            "ands_per_read": ands / N, "rows_per_read": (ands + sw + 1 + (M - 1) * (r - 1)) / N}


def binding_count(bits: int, P: dict) -> dict:
    """Binding a whole table of ``bits`` to one hiding row commitment in the circuit: its blocks, the padding block and hm96's
    salt (``sha512/row/v1`` + ``hm96-sha512/v1``; the leaf itself is the verifier's)."""
    blocks = -(-bits // 1024)
    return _add((P["compression"], blocks + 3)) | {"blocks": blocks}


def open_count(P: dict, depth: int, node: str, pick: str | None) -> dict:
    c = _add((P["leaf_hm96"], 1), (P[f"level_{node}"], depth), *(((P[pick], 1),) if pick else ()))
    return c | {"depth": depth, "node": node, "compressions": 6 + depth * (1 if node == "c1" else 2)}


def all_counts(heavy: bool) -> dict:
    P = _merkle_parts()
    out: dict = {"parts": P}
    # (a) in-circuit openings: table cases (packed leaves), embeddings (the row is the leaf; its hashing is the read's anyway), MoE
    cases = {}
    for name, n, w in (("gelu_tanh_bf16", 16, 16), ("gelu_band", 12, 15), ("tanh_rn", 27, 31)):
        per_leaf = 64 if w <= 16 else 32
        T = 3072 if name == "gelu_band" else 1 << n
        depth_packed = math.ceil(math.log2(T / per_leaf))
        pick = "pick_64x16" if per_leaf == 64 else "pick_32x31"
        cases[name] = {"entries": T, "value_bits": w,
                       "a_one_per_leaf_c2": open_count(P, math.ceil(math.log2(T)), "c2", None),
                       "a_packed_c2": open_count(P, depth_packed, "c2", pick),
                       "a_packed_c1": open_count(P, depth_packed, "c1", pick),
                       "b_mux": mux_count(math.ceil(math.log2(T)), w),
                       "binding_per_unit": binding_count(T * w, P)}
    cases["gelu_band"]["b_mux"] = delta(lambda C: band_hidden(C, C.input("g", 16), [[C.input(f"t{r}_{k}", 15) for k in range(256)] for r in range(12)]))
    for name, n, w in (("gelu_tanh_bf16", 16, 16), ("gelu_band", 12, 15)):
        T = cases[name]["entries"]
        cases[name]["d_sort"] = {str(N): sort_count(T, N, n, w) for N in (256, 1024, 4096, 16384, 65536)}
    out["tables"] = cases
    out["today"] = today_counts(heavy)
    emb = {}
    for model, V, Hd in (("gemma2-2b", 256000, 2304), ("llama32-1b", 128256, 2048), ("mistral-7b", 32768, 4096), ("olmoe-1b-7b", 50304, 2048),
                         ("qwen25-15b", 151936, 1536), ("qwen3-4b", 151936, 2560)):
        d = math.ceil(math.log2(V))
        row = binding_count(Hd * 16, P)
        emb[model] = {"V": V, "H": Hd, "depth": d, "row_hashing_public_position": row,
                      "a_path_c2": _add((P["level_c2"], d)), "a_path_c1": _add((P["level_c1"], d)),
                      "b_mux": {"ands": V * Hd * 16}}
    out["embedding"] = emb
    moe = {}
    for fam, E, K in (("MoeExpertGemm_v1 (up, W13)", 64, 2048), ("MoeExpertGemmW_v1 (down, W2)", 64, 1024)):
        d = int(math.log2(E))
        moe[fam] = {"E": E, "K": K, "row_hashing_public_position": binding_count(K * 16, P),
                    "a_expert_minor_c2": _add((P["level_c2"], d)), "a_expert_minor_c1": _add((P["level_c1"], d)),
                    "b_mux_over_experts": {"ands": E * K * 16}, "padded_all_experts": {"compute_factor": E / 8}}
    out["moe"] = moe
    return out


def today_counts(heavy: bool) -> dict:
    """Today's reads with public contents: ``boolean_export.table_gates`` (the prover's inline read) and the Boolean IR's band."""
    from verity_flock import boolean_export as BE
    from verity.ml.boolean.elementwise import bf16_band
    from verity_vllm.program.registry import boolean_dense as BD
    from verity_vllm.program.registry import dense as D

    out = {"gelu_tanh_bf16_table_gates": BE.table_gates("gelu_tanh_bf16")}
    out["gelu_band_public"] = delta(lambda C: bf16_band(C, C.input("g", 16), D.gelu_tanh_bf16_bits, 118, (128, 129)))
    out["gelu_tanh_bf16_boolean_ir"] = delta(lambda C: BD.gelu_tanh_bf16(C, C.input("g", 16)))
    for t in ("ex2", "rcp") + (("tanh_rn", "tanh_mufu") if heavy else ()):
        try:
            out[f"{t}_table_gates"] = BE.table_gates(t)
        except Exception as e:  # a table the host can't load (no MUFU fixtures): say so
            out[f"{t}_table_gates"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


# --- references ---------------------------------------------------------------------------------------------------------------

def _lanes(bits: list[int]) -> list[int]:
    return [b & 1 for b in bits]


def _bytes_bits(b: bytes) -> list[int]:
    return [(b[t >> 3] >> (t & 7)) & 1 for t in range(8 * len(b))]


def _bits_bytes(bits: list[int]) -> bytes:
    return bytes(sum(bits[8 * i + j] << j for j in range(8)) for i in range(len(bits) // 8))


def check(seed: int = 1) -> dict:
    """Each gadget on random inputs against its native reference."""
    rng = random.Random(seed)
    res = {}
    # leaf and a c2 / c1 opening of depth 3
    for node, ref in (("c2", lambda l, r: hashlib.sha512(l + r).digest()),
                      ("c1", lambda l, r: b"".join(x.to_bytes(8, "big") for x in S.compress_ref(S.midstate(C1_TAG), S.words_be(l + r))))):
        row, salt = rng.randbytes(128), rng.randbytes(192)
        sibs = [rng.randbytes(64) for _ in range(3)]
        idx = rng.randrange(8)
        C = Circuit()
        r_, s_ = C.input("row", 1024), C.input("salt", 1536)
        sb = [C.input(f"s{k}", 512) for k in range(3)]
        ib = C.input("i", 3)
        root = merkle_root(C, leaf_hm96(C, r_, s_), [to_words(x) for x in sb], ib, NODES[node])
        C.output("root", to_bits(root))
        vals = {"row": _bytes_bits(row), "salt": _bytes_bits(salt), "i": [(idx >> k) & 1 for k in range(3)]}
        vals.update({f"s{k}": _bytes_bits(sibs[k]) for k in range(3)})
        outs, fail = C.evaluate(vals, 1)
        x = S.row_digest_ref(ROW_PREFIX, row)
        cur = H5.tree_leaf(KEY, H5.commit_string(KEY, x, salt))
        for k in range(3):
            cur = ref(sibs[k], cur) if (idx >> k) & 1 else ref(cur, sibs[k])
        res[f"merkle_{node}"] = _bits_bytes(_lanes(outs["root"])) == cur and not fail
    # mux tree and the hidden band against a plain read
    C = Circuit()
    ib = C.input("i", 5)
    ents = [C.input(f"e{k}", 7) for k in range(32)]
    C.output("v", mux_tree(C, ib, ents))
    tab = [rng.randrange(128) for _ in range(32)]
    ok = True
    for i in range(32):
        vals = {"i": [(i >> k) & 1 for k in range(5)], **{f"e{k}": [(tab[k] >> j) & 1 for j in range(7)] for k in range(32)}}
        o, _ = C.evaluate(vals, 1)
        ok &= sum(b << j for j, b in enumerate(o["v"])) == tab[i]
    res["mux_tree"] = ok
    from verity_vllm.program.registry import dense as D
    from verity.ml.boolean.elementwise import bf16_band
    C = Circuit()
    g = C.input("g", 16)
    tb = [[C.input(f"t{r}_{k}", 15) for k in range(256)] for r in range(12)]
    C.output("h", band_hidden(C, g, tb))
    C.output("p", bf16_band(C, g, D.gelu_tanh_bf16_bits, 118, (128, 129)))
    words = [[(D.gelu_tanh_bf16_bits(s << 15 | (e0 | k >> 7) << 7 | k & 0x7F) & 0x7FFF if 118 <= (e0 | k >> 7) <= (128, 129)[s] else 0)
              for k in range(256)] for s, e0 in band_rows()]
    T = 64
    gs = [rng.randrange(1 << 16) for _ in range(T)]
    vals = {"g": [sum(((x >> j) & 1) << lane for lane, x in enumerate(gs)) for j in range(16)]}
    for r in range(12):
        for k in range(256):
            for j in range(15):
                vals.setdefault(f"t{r}_{k}", []).append(((1 << T) - 1) if (words[r][k] >> j) & 1 else 0)
    o, _ = C.evaluate(vals, T)
    res["band_hidden_equals_public_band"] = o["h"] == o["p"]
    # the sorted batch read: honest witness passes, a wrong value fails
    n, w, Tn, N = 4, 5, 16, 8
    M = 1 << (Tn + N - 1).bit_length()
    tabv = [rng.randrange(1 << w) for _ in range(Tn)]
    idx = [rng.randrange(Tn) for _ in range(N)]
    sw_bits = sort_witness(Tn, idx, M)
    for label, bad in (("sorted_read_honest", None), ("sorted_read_forged_rejected", 3)):
        C = Circuit()
        tbl = [C.input(f"t{k}", w) for k in range(Tn)]
        rd = [(C.input(f"ri{r}", n), C.input(f"rv{r}", w)) for r in range(N)]
        swi = C.input("sw", len(sw_bits))
        sorted_read(C, tbl, rd, n, M, list(swi))
        vals = {f"t{k}": [(tabv[k] >> j) & 1 for j in range(w)] for k in range(Tn)}
        for r in range(N):
            v = tabv[idx[r]] ^ (bad if r == 0 and bad else 0)
            vals[f"ri{r}"] = [(idx[r] >> j) & 1 for j in range(n)]
            vals[f"rv{r}"] = [(v >> j) & 1 for j in range(w)]
        vals["sw"] = sw_bits
        _, fail = C.evaluate(vals, 1)
        res[label] = (fail == 0) if bad is None else (fail != 0)
        if bad is None:
            c = counts(C)
            want = sort_count(Tn, N, n, w)
            res["sort_count_formula_bounds_built"] = 0.8 * want["ands"] <= c["ands"] <= want["ands"]
    return res


# --- flock-circuit statements ---------------------------------------------------------------------------------------------------

#: the proved shapes: a depth-9 c2 opening of a packed hm96 leaf with its entry picked (24 compressions, a 2^21 unit slot), and
#: 120 mux-tree reads of a committed 1024-entry u16 table (a 2^21 unit slot)
MERKLE_DEPTH = 9
MUX_N, MUX_READS = 10, 120


class _Port:
    def __init__(self, name):
        self.name, self.dtype = name, "u16"


class _Sub:
    def __init__(self, unit, ins, outs):
        self.id = f"hidden-reads/{unit}"
        self.inputs, self.outputs = [_Port(p) for p, _ in ins], [_Port(p) for p, _ in outs]

        class _T:
            id = f"hidden-reads-{unit}"
        self.template = _T()


class _Set:
    def __init__(self, sub):
        self.subcircuit, self.name = sub, sub.id
        self.content_digest = hashlib.sha256(sub.id.encode()).hexdigest()


def _unit_circuit(unit: str, depth: int = MERKLE_DEPTH, tab_log: int = MUX_N, reads: int = MUX_READS):
    """The unit circuit (inputs ``in{j}``, one u16 word each, outputs ``out{j}``), its ports and an input generator."""
    C = Circuit()
    if unit == "merkle":
        assert depth <= 10
        ins = [("leaf", LEAF_WORDS), ("aux", 448)]
        w = [C.input(f"in{j}", 16) for j in range(LEAF_WORDS + 448)]
        row = [b for x in w[:LEAF_WORDS] for b in x]
        aux = [b for x in w[LEAF_WORDS:] for b in x]
        salt = aux[:1536]
        sibs = [to_words(aux[1536 + 512 * k:1536 + 512 * (k + 1)]) for k in range(depth)]
        ib = aux[1536 + 512 * depth:1536 + 512 * depth + 16]
        root = merkle_root(C, leaf_hm96(C, row, salt), sibs, ib[6:6 + depth], node_c2)
        entry = mux_tree(C, ib[:6], [row[16 * k:16 * k + 16] for k in range(LEAF_WORDS)])
        outb = to_bits(root) + entry
        outs = [("out", 33)]

        def gen(rng, n):
            a = rng.integers(0, 1 << 16, size=(n, LEAF_WORDS + 448), dtype=np.uint64)
            a[:, LEAF_WORDS + 96 + 32 * depth + 1:] = 0
            return a
    else:
        T = 1 << tab_log
        assert T % 64 == 0 and reads <= 128
        ins = [("table", T), ("aux", 128)]
        w = [C.input(f"in{j}", 16) for j in range(T + 128)]
        outb = []
        for r in range(reads):
            outb += mux_tree(C, w[T + r][:tab_log], w[:T])
        outs = [("out", reads)]

        def gen(rng, n):
            a = rng.integers(0, 1 << 16, size=(n, T + 128), dtype=np.uint64)
            a[:, T:T + reads] &= T - 1
            a[:, T + reads:] = 0
            return a
    for j in range(len(outb) // 16):
        C.output(f"out{j}", outb[16 * j:16 * j + 16])
    return C, ins, outs, gen


def lowering(unit: str, **shape):
    from verity_flock import ir_lower as IL

    C, ins, outs, gen = _unit_circuit(unit, **shape)
    n_in, n_out = sum(w for _, w in ins), sum(w for _, w in outs)
    U = IL.Units(None, [16] * n_in, [16] * n_out, [[("leaf", p) for p in range(n_in)]], [[("ret", p) for p in range(n_out)]],
                 (), [], [], [], builder=lambda: C)
    lay = IL.layout(C, [f"in{j}" for j in range(n_in)], [f"out{j}" for j in range(n_out)])
    leaves = {"in": [list(range(n_in))], "out": [list(range(n_out))], "in_ports": [[w, 16] for _, w in ins],
              "out_ports": [[w, 16] for _, w in outs]}

    class Low(IL.Lowering):
        @property
        def leaves(self):
            return leaves

    return Low(f"hidden-reads-{unit}", None, U, C, lay), _Sub(unit, ins, outs), ins, outs, gen


def stage(unit: str, ns: list[int], out: Path, **shape) -> dict:
    from verity_flock import circuit as FC

    low, sub, ins, outs, gen = lowering(unit, **shape)
    FC.program_digests = lambda s, statics=None: {s.id: hashlib.sha512(s.id.encode()).hexdigest()}
    prog = hashlib.sha512(sub.id.encode()).hexdigest()
    text = FC.compose(low, sub, program=prog)
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit.txt").write_text(text)
    meta = json.loads(text.split("\n", 2)[1][5:])
    s = _Set(sub)
    rng = np.random.default_rng(20261001)
    info = {"unit": unit, "shape": shape, "unit_counts": counts(low.circuit), "layout": low.layout.counts, "unit_log": meta["unit_log"],
            "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"], "per_vu": meta["per_vu"], "pin": FC.digest(text), "n": {}}
    for n in ns:
        flat = gen(rng, n)
        res, _, ok = low.evaluate(flat)
        assert int(ok) == (1 << n) - 1, "every instance satisfies the unit rows"
        rows, at = {}, 0
        for nm, w in ins:
            rows[nm] = flat[:, at:at + w].astype(np.uint16)
            at += w
        o, at = {}, 0
        for nm, w in outs:
            o[nm] = res[:, at:at + w].astype(np.uint16)
            at += w
        FC.write(text, s, 0, n, out / f"inst-{n}.bin", out / f"pub-{n}.bin", rows=rows, outs=o)
        blocks = -(-n // meta["vus_per_block"])
        info["n"][str(n)] = {"blocks": blocks, "m": meta["k_log"] + max(0, (blocks - 1).bit_length())}
    return info


# --- prover seconds -----------------------------------------------------------------------------------------------------------

def _live(run: Path) -> list[dict]:
    out = []
    for ln in (run / "live.tsv").read_text().splitlines():
        unit, n, mode, js = ln.split("\t", 3)
        v = json.loads(js)
        wit = (v["witness_prebuilt_s"] or 0) + v["witness_s"]
        out.append({"unit": unit, "n": int(n), "mode": mode, "warm": v["warm"], "accepted": v["accepted"], "m": v["dense_m"],
                    "k_log": v["k_log"], "witness_s": wit, "proof_s": v["prove_total_s"], "prove_s": wit + v["prove_total_s"], "e2e_s": v["e2e_s"]})
    return out


def rates(runs: list[Path]) -> dict:
    """Per unit and mode: the timed proofs' median prover seconds at each size (``prove_s``: the witness, built before the
    session, and both reps' proofs), and the marginal seconds per committed bit between the two largest sizes (what a
    construction's rows cost once fixed costs are paid)."""
    import statistics

    timed: dict = {}
    for p in (p for r in runs for p in _live(r)):
        if not p["accepted"]:
            raise ValueError(f"a rejected proof: {p}")
        if not p["warm"]:
            timed.setdefault((p["unit"], p["mode"], p["n"]), []).append(p)
    out: dict = {}
    for (u, md, n), ps in sorted(timed.items()):
        out.setdefault(f"{u}/{md}", {"points": []})["points"].append(
            {"n": n, "m": ps[0]["m"], "k_log": ps[0]["k_log"], "proofs": len(ps), "prove_s": statistics.median(x["prove_s"] for x in ps),
             **{k: statistics.median(x[k] for x in ps) for k in ("witness_s", "proof_s", "e2e_s")}})
    for c in out.values():
        a, b = c["points"][-2], c["points"][-1]
        c["s_per_bit"] = (b["prove_s"] - a["prove_s"]) / (2 ** b["m"] - 2 ** a["m"])
        c["s_per_instance"] = (b["prove_s"] - a["prove_s"]) / (b["n"] - a["n"])
        c["ns_per_bit"] = round(1e9 * c["s_per_bit"], 3)
        c["between"] = [a["m"], b["m"]]
    return out


def table(counts: dict, rt: dict, reads_per_block: tuple[int, ...] = (1024, 4096, 16384)) -> list[dict]:
    """Case x construction: ANDs and committed rows per read (a table's binding amortized over a block's reads where the
    construction needs the whole table in the block), and prover seconds per read at the measured rates (SHA-512-bound rows at
    the merkle unit's, mux and sort rows at the mux unit's; rows packed densely, so a power-of-two slot can cost up to 2x)."""
    lim = 1 << K_MAX_BLOCK
    rows: list[dict] = []

    def add(case, cons, ands, rws, kind, note="", feasible=True):
        r = {"case": case, "construction": cons, "ands": int(ands), "rows": int(rws), "feasible": feasible, "note": note}
        if feasible:
            for md in ("m0", "zk"):
                r[f"s_{md}"] = rws * rt[f"{kind}/{md}"]["s_per_bit"]
        rows.append(r)

    T, today = counts["tables"], counts["today"]
    g, b, t = T["gelu_tanh_bf16"], T["gelu_band"], T["tanh_rn"]
    add("gelu_tanh_bf16 (2^16 x 16)", "c today (public contents)", today["gelu_tanh_bf16_table_gates"]["and"],
        today["gelu_tanh_bf16_table_gates"]["and"], "mux", "the rule forbids it: a program table's contents are a value")
    add("gelu_tanh_bf16 (2^16 x 16)", "a merkle, packed leaf, c2", g["a_packed_c2"]["ands"], g["a_packed_c2"]["rows"], "merkle")
    add("gelu_tanh_bf16 (2^16 x 16)", "a merkle, packed leaf, c1", g["a_packed_c1"]["ands"], g["a_packed_c1"]["rows"], "merkle")
    add("gelu_tanh_bf16 (2^16 x 16)", "b mux over committed table", g["b_mux"]["ands"], g["b_mux"]["rows"] + g["binding_per_unit"]["rows"],
        "mux", f"binding the table is {g['binding_per_unit']['rows']} rows per block (> 2^{K_MAX_BLOCK})", feasible=g["binding_per_unit"]["rows"] < lim)
    d = g["d_sort"]["65536"]
    add("gelu_tanh_bf16 (2^16 x 16)", "d sorted batch, 2^16 reads", d["ands"] / 65536, (d["rows"] + g["binding_per_unit"]["rows"]) / 65536, "mux",
        f"{d['rows'] + g['binding_per_unit']['rows']} rows in one block (> 2^{K_MAX_BLOCK})", feasible=False)
    gb = today["gelu_band_public"]
    add("gelu band (3072 x 15, the Boolean IR's)", "c today (public band)", gb["ands"], gb["rows"], "mux",
        f"the whole gelu_tanh_bf16 element is {today['gelu_tanh_bf16_boolean_ir']['ands']} ANDs")
    add("gelu band (3072 x 15, the Boolean IR's)", "a merkle, packed leaf, c2", b["a_packed_c2"]["ands"], b["a_packed_c2"]["rows"], "merkle")
    add("gelu band (3072 x 15, the Boolean IR's)", "a merkle, packed leaf, c1", b["a_packed_c1"]["ands"], b["a_packed_c1"]["rows"], "merkle")
    bind = b["binding_per_unit"]["rows"]
    for R in reads_per_block:
        add("gelu band (3072 x 15, the Boolean IR's)", f"b mux over committed band, {R} reads/block", b["b_mux"]["ands"], b["b_mux"]["rows"] + bind / R,
            "mux", f"binding {bind} rows per block", feasible=R * b["b_mux"]["rows"] + bind < lim)
    for R in reads_per_block:
        d = b["d_sort"][str(R)]
        add("gelu band (3072 x 15, the Boolean IR's)", f"d sorted batch, {R} reads/block", d["ands"] / R, (d["rows"] + bind) / R, "mux",
            f"M = {d['M']} records; a block-level slot, as today's MUFU lookup slots", feasible=d["rows"] + bind < lim)
    tt = today["tanh_rn_table_gates"]
    if "and" in tt:
        add("tanh_rn (2^27 x 31)", "c today (public contents)", tt["and"], tt["and"], "mux", f"{tt['xor']} XOR terms in its linear rows")
    add("tanh_rn (2^27 x 31)", "a merkle, packed leaf, c2", t["a_packed_c2"]["ands"], t["a_packed_c2"]["rows"], "merkle")
    add("tanh_rn (2^27 x 31)", "a merkle, packed leaf, c1", t["a_packed_c1"]["ands"], t["a_packed_c1"]["rows"], "merkle")
    add("tanh_rn (2^27 x 31)", "b mux / d sort", t["b_mux"]["ands"], t["b_mux"]["rows"], "mux",
        f"binding the table is {t['binding_per_unit']['rows']} rows", feasible=False)
    for model, e in counts["embedding"].items():
        case = f"embedding {model} (V {e['V']} x H {e['H']})"
        add(case, "row hashing alone (paid at a public position too)", e["row_hashing_public_position"]["ands"], e["row_hashing_public_position"]["rows"], "merkle")
        add(case, f"a merkle path over the row, c2 (depth {e['depth']})", e["a_path_c2"]["ands"], e["a_path_c2"]["rows"], "merkle", "on top of the row hashing")
        add(case, f"a merkle path over the row, c1 (depth {e['depth']})", e["a_path_c1"]["ands"], e["a_path_c1"]["rows"], "merkle", "on top of the row hashing")
        add(case, "b mux over the table", e["b_mux"]["ands"], e["b_mux"]["ands"], "mux", feasible=False)
    for fam, e in counts["moe"].items():
        case = f"MoE {fam} (E {e['E']}, K {e['K']}), per weight row"
        add(case, "row hashing alone (paid at a public position too)", e["row_hashing_public_position"]["ands"], e["row_hashing_public_position"]["rows"], "merkle")
        add(case, "a expert-minor subtree, c2 (6 levels)", e["a_expert_minor_c2"]["ands"], e["a_expert_minor_c2"]["rows"], "merkle", "on top of the row hashing")
        add(case, "a expert-minor subtree, c1 (6 levels)", e["a_expert_minor_c1"]["ands"], e["a_expert_minor_c1"]["rows"], "merkle", "on top of the row hashing")
        add(case, "b mux over the experts' rows", e["b_mux_over_experts"]["ands"], e["b_mux_over_experts"]["ands"] + e["E"] * e["row_hashing_public_position"]["rows"],
            "mux", "every expert's row bound in the block", feasible=e["E"] * e["row_hashing_public_position"]["rows"] < lim)
    return rows


#: the largest block a flock-circuit statement takes (verity_flock.circuit.K_MAX)
K_MAX_BLOCK = 26


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "counts", "stage", "summary"])
    ap.add_argument("--runs", nargs="+", default=[], help="summary: run dirs whose out/ holds live.tsv (and one counts.json)")
    ap.add_argument("--unit", choices=["merkle", "mux"])
    ap.add_argument("--n", type=int, nargs="+", default=[])
    ap.add_argument("--out")
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--depth", type=int, default=MERKLE_DEPTH, help="merkle: the opening's levels in the circuit")
    ap.add_argument("--tab-log", type=int, default=MUX_N, help="mux: log2 of the table's u16 entries")
    ap.add_argument("--reads", type=int, default=MUX_READS, help="mux: reads per unit")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        r = check()
        print(json.dumps(r))
        return 0 if all(r.values()) else 1
    if a.cmd == "counts":
        r = all_counts(a.heavy)
        Path(a.out).write_text(json.dumps(r, indent=1, sort_keys=True))
        print(json.dumps({k: r["tables"][k]["a_packed_c2"] for k in r["tables"]}))
        return 0
    if a.cmd == "summary":
        outs = [Path(r) / "out" for r in a.runs]
        counts_path = next(o / "counts.json" for o in outs if (o / "counts.json").exists() and "tanh_rn_table_gates" in
                           json.loads((o / "counts.json").read_text())["today"])
        rt = rates(outs)
        r = {"runs": [Path(x).name for x in a.runs], "counts_from": counts_path.parent.parent.name, "rates": rt,
             "table": table(json.loads(counts_path.read_text()), rt)}
        Path(a.out).write_text(json.dumps(r, indent=1))
        for row in r["table"]:
            s = "  ".join(f"{row[k]:.4g}" if k in row else "-" for k in ("s_m0", "s_zk"))
            print(f"{row['case'][:44]:44}  {row['construction'][:52]:52}  {row['ands']:>14,}  {row['rows']:>14,}  {s}  {row['note'][:70]}")
        return 0
    shape = {"depth": a.depth} if a.unit == "merkle" else {"tab_log": a.tab_log, "reads": a.reads}
    info = stage(a.unit, a.n, Path(a.out), **shape)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
