"""The GPU's no-fold call (v31lib's Prover over libv31.so) checked against the reference itself,
`verity/core/protocols/pouw/circuit/dither.py`, rather than v31.py's mirror of it, on filler calls (the pad,
`dither.Shape` with `o` False: T_1 .. T_S a tile), served calls (an operand A, `o` True: T_1 .. T_{S-1} and then O a
tile, O at 32 bits or at 16, `ow`) and gate-up calls (`act`: O and then, after each tile-row's tiles, its rows of A′). On
each case: the call's seeds (`dither.fill_seeds` for a filler, `dither.seeds` for a served call) as the device's seed
buffer holds them (v31's `fill_seeds_bytes`), read back stream by stream against dither's byte for byte, the device's pad
(`dither.pad_rows`), every committed value of sampled cells (`dither.cell_values`: the T_s the layout commits, O at its
width, and a gate cell's A′ by `dither.act_word` on its own and its up cell's 32-bit O under the amax of the GPU's row
record), sampled
leaves rebuilt from only the cells they hold (`dither.leaf_row`) and opened against the GPU's root on the GPU's own paths
(`dither.opens`, salted `dither.salts(Ks, off)` at the call's offset), the verifier's check of sampled cells' last blocks
on the GPU's openings (`dither.block_inputs`, `dither.check_last`; a gate cell's reads its up row, the map's scalars,
amax from its row's opened record and A′), and, on the small cases, the whole call (`dither.values`, `dither.acts`,
`dither.values_buffer`, `dither.commit`) against the GPU's values and root. A gate-up call also: two whole rows of A′ and
their records against dither's map (`dither.act_codes` over the row's `dither.silu_mul`: amax, its column and the
codes) on O's rows, which numpy takes on v31's coins (d_B for every column) and `dither.cell_values` ties at sampled
columns, those rows' A′ leaves (codes and record) opened against the root, both amax checks on each (the consumer's,
`dither.check_amax`, h at the record's column recomputed from A and B, and every gate cell's bound) passing the GPU's
row and refusing a forged amax (the row's second-largest |h|, its codes formed under it and the record naming its
cell), every row with a column at amax, and the device's A′ buffer and records against what it committed; on the small
ones a consumer call whose A is
the device's A′ where it lies (`Prover.act_ptr`), its seeds drawn on the producer's root, against dither on A′'s rows as
opened from that root (`dither.act_row`). The device runs on the seeds read back from its buffer and the reference on
dither's. A cheating call must fail them.

    PYTHONPATH=. python3 benchmarks/pouw/accel_v31/fill_dither.py --lib LIB [--cells 64] [--leaves 8] [--blocks 8] [--out DIR]

dither.py is pure Python, so only the small cases rebuild the whole call. One line a case, then the verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31
import v31lib

from verity.core.protocols.pouw.circuit import dither

# NAME M N K LP L ROW KIND CHEAT FULL: KIND fill (the pad, from fill seeds), served (an operand A, from served seeds, O at
# 32 bits), o16 (O at 16), act (a gate-up call: O at 16 and A′) or act32 (A′, O at 32); a served kind's operands are in
# -127 .. 126, as a served root refuses a row of A + d_A or B + d_B spanning -128 .. 128. FULL 1: the whole call through
# dither.py (and a gate-up call's consumer)
CASES = """
f-small 128 256 768 256 512 1024 fill none 1
f-qkv-L4096 256 6144 4096 512 4096 1024 fill none 0
f-o-L1024-row16k 256 4096 4096 128 1024 16384 fill none 0
f-up-L4096 128 28672 8192 512 4096 1024 fill none 0
s-small 256 128 1024 128 512 2048 served none 1
s-small-S1 128 256 768 256 768 1024 served none 1
s-qkv-L4096-row16k 256 6144 4096 512 4096 16384 served none 0
s-o-L1024-row16k 256 4096 4096 128 1024 16384 served none 0
s-down-L4096 128 384 14336 512 4096 1024 served none 0
s-up-L8192 128 28672 4096 512 8192 1024 served none 0
f-qkv-L4096-row8k 256 6144 4096 512 4096 8192 fill none 0
s-down-L4096-row8k 128 384 14336 512 4096 8192 served none 0
f-small-row32k 128 256 1024 256 512 32768 fill none 1
s-small-row32k 256 128 1024 128 512 32768 served none 1
f-o-L4096-row32k 128 512 4096 512 4096 32768 fill none 0
s-down-L4096-row32k 128 384 14336 512 4096 32768 served none 0
o-small 256 128 1024 128 512 2048 o16 none 1
o-small-S1 128 256 768 256 768 1024 o16 none 1
o-qkv-L4096-row16k 256 6144 4096 512 4096 16384 o16 none 0
o-down-L4096-row16k 128 4096 14336 512 4096 16384 o16 none 0
a-small 128 256 768 256 768 1024 act none 1
a-small-S2 128 256 1024 128 512 1024 act none 1
a-small32 128 256 1024 256 1024 2048 act32 none 1
a-mid 256 2560 2048 256 2048 1024 act none 0
a-up-L4096-row16k 128 28672 4096 512 4096 16384 act none 0
a-up65536-L4096-row16k 128 65536 4096 512 4096 16384 act none 0
c-skip 128 256 2048 256 512 1024 fill skip:3 1
c-forge 256 256 4096 256 1024 1024 served forge:2:0.5 0
c-forgeO 256 256 4096 256 1024 1024 served forgeO:0.5 0
c-forgeO-o16 256 256 4096 256 1024 1024 o16 forgeO:0.5 0
c-forgeO-act 128 512 2048 256 2048 1024 act forgeO:0.5 0
c-forgeA 128 512 2048 256 2048 1024 act forgeA:0.5 0
c-forgeH 128 512 2048 256 2048 1024 act forgeH:0.5 0
"""

#: KIND: (served, O's width, A′)
KINDS = {"fill": (False, 32, False), "served": (True, 32, False), "o16": (True, 16, False), "act": (True, 16, True),
         "act32": (True, 32, True)}

#: an A′ leaf is rebuilt by dither.leaf_row when its gate and up dots take at most this many products (else from O's rows)
LEAF_ROW_PRODUCTS = 1 << 23


def case_seeds(served: bool, rng: np.random.Generator, receipt: bytes | None = None) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """(the seeds read back from the device's seed buffer, dither's) for one call."""
    if served:
        ref = dither.seeds(os.urandom(64), os.urandom(64) if receipt is None else receipt, int(rng.integers(1 << 62)))
    else:
        ref = dither.fill_seeds(os.urandom(64), os.urandom(64))
    return v31.fill_seeds_from(v31.fill_seeds_bytes(ref), served), ref


def committed(sh: dither.Shape, buf: bytes, i: int, j: int) -> list[int]:
    """Cell (i, j)'s committed values in `buf`: its T_s (s <= W), then O when the call commits it (at `ow` bits), then a
    gate cell's A′ byte."""
    def at(pos: int, off: int, n: int) -> int:
        a = sh.leaf_span(pos)[0] + off
        return int.from_bytes(buf[a:a + n], "little")

    out = [at(*sh.word_leaf(i, j, s), 2) for s in range(1, sh.W + 1)]
    if sh.o:
        out.append(at(*sh.o_leaf(i, j), sh.ow // 8))
    if sh.gate(j):
        out.append(at(*sh.act_leaf(i, j), 1))
    return out


def o_rows(sh: dither.Shape, seeds: dict[str, bytes], A: np.ndarray, B: np.ndarray, rows: list[int]) -> dict[int, np.ndarray]:
    """O's rows `rows` (n s32 words each, mod 2^32), (A + d_A)(B + d_B)^T in numpy on v31's coins, B's dither a block of
    rows at a time (exact in float64: |sum| < 2^26)."""
    ad = (A[rows].astype(np.float64) + v31.bits_rows(seeds["dA"], "dA", rows, sh.k)).T
    out = np.empty((len(rows), sh.n), np.int64)
    for r0 in range(0, sh.n, 4096):
        r1 = min(r0 + 4096, sh.n)
        bd = B[r0:r1].astype(np.float64) + v31.bits_rows(seeds["dB"], "dB", range(r0, r1), sh.k)
        out[:, r0:r1] = np.rint(bd @ ad).T.astype(np.int64)
    return {i: out[x] & 0xFFFFFFFF for x, i in enumerate(rows)}


def act_rows(sh: dither.Shape, O: dict[int, np.ndarray], sa: list[int], sw: list[int]
             ) -> dict[int, tuple[list[int], bytes, dither.ActRecord, list[int]]]:
    """Each row's (h, A′ row bytes, record, codes) by dither's map on its whole O row (`dither.act_codes`): the row's I
    codes and then its record."""
    out = {}
    for i, o in O.items():
        hs = [dither.silu_mul(dither.dequant(int(o[j]), sa[i], sw[j]), dither.dequant(int(o[j + sh.I]), sa[i], sw[j + sh.I]))
              for j in range(sh.I)]
        codes, rec = dither.act_codes(hs)
        out[i] = hs, bytes(x & 0xFF for x in codes) + rec.pack(), rec, codes
    return out


def records(recs: np.ndarray) -> list[bytes]:
    """The device's records (m x dither.REC bytes) a row each."""
    return [bytes(r) for r in recs]


def check_act(rep: dict, sh: dither.Shape, ref: dict[str, bytes], Aref: np.ndarray, B: np.ndarray, p: v31lib.Prover,
              c: int, root: bytes, buf: bytes, Ap: np.ndarray, recs: np.ndarray, over: int, picks: list[int],
              rng: np.random.Generator) -> dict[int, bytes]:
    """A gate-up call's A′: the device's A′ buffer and records against what it committed, two whole rows (and the rows of
    every sampled A′ leaf) against dither's map on O's rows, tied to dither.cell_values at two columns a row, the rows' A′
    leaves opened against the root, and both amax checks on each whole row (`dither.check_amax` on the GPU's record with
    every gate cell's bound, then `v31.row_amax_ok` on a forged amax). Returns the reference rows of the sampled A′
    leaves."""
    tl, nb = sh.n // dither.TILE * sh.lpt, dither.TILE * sh.arow
    starts = [sh.leaf_span(tm * sh.stride + tl)[0] for tm in range(sh.m // dither.TILE)]
    rows_b = np.concatenate([Ap.view(np.uint8), recs], axis=1)
    rep["act_buffer_equal"] = all(buf[a:a + nb] == rows_b[x * dither.TILE:(x + 1) * dither.TILE].tobytes()
                                  for x, a in enumerate(starts))
    rep["act_over"] = over
    whole = sorted({int(rng.integers(dither.TILE)), sh.m - dither.TILE + int(rng.integers(dither.TILE))})
    if len(whole) == 1:
        whole.append((whole[0] + 1) % sh.m)
    act_picks = [q for q in picks if sh.leaf_of(q)[0] == dither.ACT]
    rows = sorted(set(whole) | {sh.leaf_of(q)[1][0][0] for q in act_picks})
    O = o_rows(sh, ref, Aref, B, rows)
    tied = 0
    for i in whole:
        j = int(rng.integers(sh.I))
        tied += all(dither.cell_values(sh, ref, Aref[i], B[jj], i, jj)[1] == int(O[i][jj]) for jj in (j, j + sh.I))
    ar = act_rows(sh, O, list(map(int, p.sa)), list(map(int, p.sw)))
    rec_b = records(recs)
    rep["act_rows"], rep["act_rows_tied"] = len(whole), tied
    rep["act_rows_amax_equal"] = sum(ar[i][2].amax == int.from_bytes(rec_b[i][:2], "little") for i in whole)
    rep["act_rows_equal"] = sum(ar[i][1] == Ap[i].tobytes() + rec_b[i] for i in whole)
    sc = dither.ActScales(list(map(int, p.sa)), list(map(int, p.sw)))
    rep["act_amax_ok"] = sum(dither.check_amax(sh, ref, Aref[i], B, i, rec_b[i], sc) == 1
                             and not v31.gate_refusals(ar[i][0], Ap[i].tolist(), int.from_bytes(rec_b[i][:2], "little"))
                             for i in whole)
    forged = [(i, v31.forged_amax(ar[i][0])) for i in whole]
    forged = [(i, x) for i, x in forged if x is not None]
    rep["act_forged"] = len(forged)
    rep["act_forged_refused"] = sum(not v31.row_amax_ok(ar[i][0], fc, fr.pack()) for i, (fc, fr) in forged)
    dom, salt = sh.domain(), dither.salts(p.ks, p.calls[c]["off"])
    lv = [q for i in whole for q in sh.act_leaves(i)]
    rows_open = 0
    for q, (g_row, _, g_path) in zip(lv, p.open(c, lv)):
        (i, j0), n = sh.leaf_of(q)[1][0], sh.leaf_span(q)[1]
        rows_open += g_row == ar[i][1][j0:j0 + n] and dither.opens(dom, root, sh.leaves, q, g_row, salt(q), g_path)
    rep["act_leaves"], rep["act_leaves_open"] = len(lv), rows_open
    out = {}
    for q in act_picks:
        (i, j0), n = sh.leaf_of(q)[1][0], sh.leaf_span(q)[1]
        small = 2 * n * sh.k <= LEAF_ROW_PRODUCTS
        out[q] = dither.leaf_row(sh, ref, Aref, B, q, sc, ar[i][2]) if small else ar[i][1][j0:j0 + n]
    return out


def check_blocks(sh: dither.Shape, ref: dict[str, bytes], Aref: np.ndarray, B: np.ndarray, p: v31lib.Prover, c: int,
                 root: bytes, sc: dither.ActScales | None, blocks: int, rng: np.random.Generator) -> tuple[int, list]:
    """The verifier on `blocks` sampled cells' last blocks (half of them gate cells on a gate-up call), from the GPU's
    openings: every opened row opens the root, and `dither.check_last` passes on `dither.block_inputs`."""
    dom, salt = sh.domain(), dither.salts(p.ks, p.calls[c]["off"])
    good, bad = 0, []
    for b in range(blocks):
        i = int(rng.integers(sh.m))
        j = int(rng.integers(sh.I if sh.act and b % 2 == 0 else sh.n))
        pos = sh.opened(i, j, sh.S)
        got = p.open(c, pos)
        rows = {q: r for q, (r, _, _) in zip(pos, got)}
        opened = all(dither.opens(dom, root, sh.leaves, q, r, salt(q), path) for q, (r, _, path) in zip(pos, got))
        kw = (dict(up=B[j + sh.I], sc=sc, amax=int.from_bytes(dither.rec_at(sh, rows, i)[:2], "little"),
                   ap=dither.act_at(sh, rows, i, j)) if sh.gate(j) else {})
        ins = dither.block_inputs(sh, ref, Aref[i], B[j], i, j, sh.S, t0=dither.word_at(sh, rows, i, j, sh.S - 1),
                                  o=dither.out_at(sh, rows, i, j), **kw)
        if opened and dither.check_last(sh, ref, i, j, ins) == 1:
            good += 1
        else:
            bad.append((i, j, "opening" if not opened else "check_last"))
    return good, bad


def run_consumer(lib: v31lib.Lib, sh: dither.Shape, p: v31lib.Prover, c: int, root: bytes, Ap_ref: list[list[int]],
                 rng: np.random.Generator) -> dict:
    """A consumer of the gate-up call (m x 128 at K = I, O at 16 bits, the producer's row: one ROW a process) whose A is the
    producer's A′ where it lies on the device, its seeds drawn on the producer's root: A′'s rows opened from the
    producer's root (`dither.act_row`) against the reference's A′, and the consumer's whole call through dither on them
    against the GPU's values and root."""
    k2 = sh.I
    sh2 = dither.Shape(sh.m, dither.TILE, k2, 128, min(k2, 1024), sh.row, o=True, ow=16)
    lv = [q for i in range(sh.m) for q in sh.act_leaves(i)]
    dom, salt = sh.domain(), dither.salts(p.ks, p.calls[c]["off"])
    got = p.open(c, lv)
    rows = {q: r for q, (r, _, _) in zip(lv, got)}
    opened = sum(dither.opens(dom, root, sh.leaves, q, r, salt(q), path) for q, (r, _, path) in zip(lv, got))
    A2 = [dither.act_row(sh, rows, i) for i in range(sh.m)]
    B2 = rng.integers(-127, 127, size=(sh2.n, k2), dtype=np.int8)
    seeds, ref = case_seeds(True, rng, receipt=root)
    with v31lib.Prover(sh2.m, sh2.n, k2, sh2.lp, sh2.L, sh2.row, served=True, ow=16, lib=lib) as q:
        q.set_weights(B2)
        c2 = q.prove(seeds, p.act_ptr(c))
        root2, buf2 = q.root(c2), q.values(c2)
        T, O = dither.values(sh2, ref, A2, B2)
        want = dither.values_buffer(sh2, T, O)
        out = {"shape": sh2.label(), "a_leaves": len(lv), "a_leaves_open": opened, "a_is_ref": A2 == Ap_ref,
               "values_equal": want == buf2,
               "root_equal": dither.commit(sh2, want, dither.salts(q.ks, q.calls[c2]["off"])).root == root2}
    out["ok"] = (out["a_leaves_open"] == out["a_leaves"] and out["a_is_ref"] and out["values_equal"] and out["root_equal"])
    return out


def run_case(lib: v31lib.Lib, line: str, cells: int, leaves: int, blocks: int, rng: np.random.Generator) -> dict:
    name, m, n, k, lp, L, row, kind, cheat, full = line.split()
    m, n, k, lp, L, row = map(int, (m, n, k, lp, L, row))
    served, ow, act = KINDS[kind]
    sh = dither.Shape(m, n, k, lp, L, row, o=served, ow=ow, act=act)
    lo, hi = (-127, 127) if served else (-128, 128)
    B = rng.integers(lo, hi, size=(n, k), dtype=np.int8)
    A = rng.integers(lo, hi, size=(m, k), dtype=np.int8) if served else None
    seeds, ref = case_seeds(served, rng)
    rep: dict = {"name": name, "shape": sh.label(), "L": L, "S": sh.S, "W": sh.W, "kind": kind, "cheat": cheat,
                 "seeds_equal": seeds == ref}
    Aref = A if served else np.array(dither.pad_rows(ref, k, range(m)), dtype=np.int8)
    with v31lib.Prover(m, n, k, lp, L, row, served=served, ow=ow, act=act, lib=lib) as p:
        p.set_weights(B)
        c = p.prove(seeds, p.load_a(A) if served else None, cheat)
        root, off = p.root(c), p.calls[c]["off"]
        buf = p.values(c)
        Ap, recs, over = p.act(c) if act else (None, None, 0)
        amax = recs[:, :2].copy().view("<u2").reshape(-1) if act else None
        sa, sw = (list(map(int, p.sa)), list(map(int, p.sw))) if act else (None, None)
        if not served:
            rep["pad_equal"] = p.pad() == Aref.tobytes()
        bad_cells = []
        for _ in range(cells):
            i, j = int(rng.integers(m)), int(rng.integers(n))
            T, O = dither.cell_values(sh, ref, Aref[i], B[j], i, j)
            want = T[:sh.W] + ([O & sh.omask] if served else [])
            if sh.gate(j):
                ou = dither.cell_values(sh, ref, Aref[i], B[j + sh.I], i, j + sh.I)[1]
                want.append(dither.act_word(O, ou, sa[i], sw[j], sw[j + sh.I], int(amax[i])))
            got = committed(sh, buf, i, j)
            if got != want:
                bad_cells.append((i, j, next(x for x in range(len(want)) if got[x] != want[x])))
        rep["cells"], rep["cells_bad"] = cells, bad_cells[:8]
        rep["cells_equal"] = cells - len(bad_cells)
        picks = [int(x) for x in rng.choice(sh.leaves, size=min(leaves, sh.leaves), replace=False)]
        act_ref = check_act(rep, sh, ref, Aref, B, p, c, root, buf, Ap, recs, over, picks, rng) if act else {}
        rows_equal = opens = 0
        dom, salt = sh.domain(), dither.salts(p.ks, off)
        for pos, (g_row, _, g_path) in zip(picks, p.open(c, picks)):
            ref_row = act_ref[pos] if pos in act_ref else dither.leaf_row(sh, ref, Aref, B, pos)
            rows_equal += g_row == ref_row
            opens += dither.opens(dom, root, sh.leaves, pos, ref_row, salt(pos), g_path)
        rep["leaves"], rep["rows_equal"], rep["leaves_open"] = len(picks), rows_equal, opens
        rep["act_leaves_sampled"] = len(act_ref)
        if served:
            sc = dither.ActScales(sa, sw) if act else None
            rep["blocks"] = blocks
            rep["blocks_pass"], rep["blocks_bad"] = check_blocks(sh, ref, Aref, B, p, c, root, sc, blocks, rng)
            rep["blocks_bad"] = rep["blocks_bad"][:8]
        if full == "1":
            T, O = dither.values(sh, ref, Aref, B)
            Ar = None
            Rr = None
            if act:
                Ar, Rr = dither.acts(sh, O, dither.act_scales(sh, sa, sw))
                rep["rec_equal"] = [r.pack() for r in Rr] == records(recs)
                rep["act_equal"] = Ar == Ap.tolist()
            want = dither.values_buffer(sh, T, O, Ar, Rr)
            rep["values_equal"] = want == buf
            rep["root_equal"] = dither.commit(sh, want, salt).root == root
            if act and cheat == "none":
                rep["consumer"] = run_consumer(lib, sh, p, c, root, Ar, rng)
        rep["root"] = root.hex()[:16]
    honest = (rep["seeds_equal"] and rep.get("pad_equal", True) and rep["cells_equal"] == cells
              and rep["rows_equal"] == rep["leaves"] and rep["leaves_open"] == rep["leaves"]
              and rep.get("blocks_pass", 0) == rep.get("blocks", 0)
              and rep.get("act_buffer_equal", True) and rep.get("act_rows_tied", 0) == rep.get("act_rows", 0)
              and rep.get("act_rows_amax_equal", 0) == rep.get("act_rows", 0)
              and rep.get("act_rows_equal", 0) == rep.get("act_rows", 0)
              and rep.get("act_leaves_open", 0) == rep.get("act_leaves", 0)
              and rep.get("act_amax_ok", 0) == rep.get("act_rows", 0) and not rep.get("act_over", 0)
              and rep.get("values_equal", True) and rep.get("root_equal", True)
              and rep.get("rec_equal", True) and rep.get("act_equal", True) and rep.get("consumer", {}).get("ok", True))
    rep["accept"] = bool(honest)
    rep["good"] = rep["accept"] == (cheat == "none") and rep.get("act_forged_refused", 0) == rep.get("act_forged", 0)
    return rep


def describe(r: dict) -> str:
    sh = r["shape"]
    kind = r["kind"] + (f" ow {sh['ow']}" if "ow" in sh else "") + (f" I {sh['act']}" if "act" in sh else "")
    parts = [f"{kind} {sh['m']}x{sh['n']}x{sh['k']} L' {sh['lp']} L {r['L']} row {sh['row']} S {r['S']} W {r['W']}",
             f"seeds {'equal' if r['seeds_equal'] else 'DIFFER'}"]
    if "pad_equal" in r:
        parts.append(f"pad {'equal' if r['pad_equal'] else 'DIFFERS'}")
    parts.append(f"cells {r['cells_equal']}/{r['cells']} equal"
                 + (f" (first off: (i, j, value) {tuple(r['cells_bad'][0])})" if r["cells_bad"] else ""))
    parts.append(f"leaves {r['rows_equal']}/{r['leaves']} rows equal ({r['act_leaves_sampled']} of A′), "
                 f"{r['leaves_open']}/{r['leaves']} open the GPU's root")
    if "act_rows" in r:
        parts.append(f"A′ buffer and records {'equal' if r['act_buffer_equal'] else 'DIFFER'}, {r['act_over']} rows with "
                     f"no column at amax; whole rows: O tied {r['act_rows_tied']}/{r['act_rows']}, amax "
                     f"{r['act_rows_amax_equal']}/{r['act_rows']}, A′ and record {r['act_rows_equal']}/{r['act_rows']} "
                     f"equal, their A′ leaves {r['act_leaves_open']}/{r['act_leaves']} equal and open, the amax checks "
                     f"(the consumer's at the record's column, every gate cell's bound) {r['act_amax_ok']}/{r['act_rows']} "
                     f"pass, a forged amax (the second-largest |h|) {r['act_forged_refused']}/{r['act_forged']} refused")
    if "blocks" in r:
        parts.append(f"last blocks {r['blocks_pass']}/{r['blocks']} pass the verifier"
                     + (f" (first off: {tuple(r['blocks_bad'][0])})" if r["blocks_bad"] else ""))
    if "values_equal" in r:
        parts.append(f"whole call: values {'equal' if r['values_equal'] else 'DIFFER'}, root "
                     f"{'equal' if r['root_equal'] else 'DIFFERS'}"
                     + (f", records {'equal' if r['rec_equal'] else 'DIFFER'}, A′ {'equal' if r['act_equal'] else 'DIFFERS'}"
                        if "rec_equal" in r else ""))
    if "consumer" in r:
        q = r["consumer"]
        parts.append(f"consumer {q['shape']['m']}x{q['shape']['n']}x{q['shape']['k']} on the device's A′: its A′ leaves "
                     f"{q['a_leaves_open']}/{q['a_leaves']} open, A {'is' if q['a_is_ref'] else 'ISN’T'} the reference's "
                     f"A′, values {'equal' if q['values_equal'] else 'DIFFER'}, root "
                     f"{'equal' if q['root_equal'] else 'DIFFERS'}")
    return (f"{r['name']}: {'ACCEPT' if r['accept'] else 'REFUSE'} ({r['cheat']}) " + "; ".join(parts)
            + f"; root {r['root']}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cells", type=int, default=64)
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default="", help="the cases of these kinds only (comma-separated)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv[1:])
    lib, rng = v31lib.Lib(a.lib), np.random.default_rng(a.seed)
    only = set(a.only.split(",")) if a.only else None
    reps = []
    for line in CASES.strip().splitlines():
        if only and line.split()[7] not in only:
            continue
        r = run_case(lib, line, a.cells, a.leaves, a.blocks, rng)
        reps.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "fill-dither.json").write_text(json.dumps(reps, indent=1))
    honest = [r for r in reps if r["cheat"] == "none"]
    cheats = [r for r in reps if r["cheat"] != "none"]
    good = all(r["good"] for r in reps)
    print(f"dither.py recheck: {'ACCEPT' if good else 'REFUSE'}: {sum(r['accept'] for r in honest)}/{len(honest)} honest "
          f"calls exact against dither.py ({sum(r['kind'] != 'fill' for r in honest)} served, "
          f"{sum(r['kind'] in ('o16', 'act') for r in honest)} with O at 16 bits, {sum('act' in r['kind'] for r in honest)} "
          f"with A′, {sum('consumer' in r for r in honest)} consumers on the device's A′), "
          f"{sum(not r['accept'] for r in cheats)}/{len(cheats)} cheating calls refused", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
