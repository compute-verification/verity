"""The engine against the CPU reference (``verity.ml.commitments``), bit for bit: row digests on random rows at edge sizes
(every variant), hm96 leaves under both salt forms and fused with their units' frame leaves, frame leaves, unit frame
tables, levels (a front child, strided trees), pads and roots, and on a CUDA device the frozen H200
layout's rows (``bench.h200_rows``) on sampled slots. ``python check.py --out DIR`` runs it on the current CUDA device (or
``--device host``) and writes check.json; the tests call the same functions on the host twin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as E

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import (
    SCHEMA_HM96_SHA512_ROW_V2,
    sha512_row_v2_digest,
    sha512_row_v2_prefix,
)
from verity.ml.crypto.chacha20 import keystream

INST = hm96.SHA512
KEY = INST.default_key
SALT_KEY = bytes(range(32, 64))
ROOTS_SCHEMA = "frame-v3-sha512/root/v1"
EDGE_BYTES = (0, 1, 13, 110, 111, 112, 127, 128, 129, 239, 240, 1023, 1024, 1025, 2048)
LEAF_MINB = {False: (1, 2), True: (0,)}   # the leaf kernels checked on a device, and on the host twin


def domain(tag: str, n: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(f"verity/sha512-rows/check/{tag}", {"n": n}), -1, RangeIndexedDomain(0, n),
                            hash="sha512")


def row_digest(row: bytes) -> bytes:
    """sha512/row/v2's digest of a row of whole bytes."""
    d = hashlib.sha512(sha512_row_v2_prefix(8 * len(row)) + row).digest()
    if row:
        assert d == sha512_row_v2_digest(row, 8 * len(row))
    return d


def salt(key: bytes, nonce: int, r: int) -> bytes:
    return keystream(key, nonce, 3 * r, 3)


class Dev:
    """Inputs on an engine's device: numpy for the host twin, torch tensors on CUDA."""

    def __init__(self, engine: E.Engine) -> None:
        self.e = engine

    def put(self, a: np.ndarray) -> Any:
        if self.e.host:
            return np.ascontiguousarray(a)
        return self.e.torch.from_numpy(np.ascontiguousarray(a)).to(self.e.device)


def random_tensors(rng: np.random.Generator, slots: int) -> list[np.ndarray]:
    """Rows of widths that exercise the fast path (16-byte aligned rows and strides), the word path and the byte path."""
    return [rng.integers(0, 256, (slots, n), dtype=np.uint8) for n in (2048, 1100, 4, 2304)]


def random_rows() -> list[list[tuple[int, int, int]]]:
    rows: list[list[tuple[int, int, int]]] = [[(0, 0, n)] for n in EDGE_BYTES]
    rows += [[(1, 3, n)] for n in (0, 1, 111, 112, 129, 1000)]                    # unaligned starts, an unaligned stride
    rows += [[(3, 16, n)] for n in (127, 128, 256, 2288)]                         # aligned, offset by a 16-byte word
    rows += [[(0, 0, 1024), (2, 0, 4)], [(0, 0, 2048), (2, 1, 3)], [(1, 5, 100), (3, 3, 200)], [(2, 0, 4), (0, 0, 2048)],
             [(3, 0, 2048), (0, 0, 2048)], [(0, 0, 0), (1, 0, 129)]]
    return rows


def ref_rows(tensors: Sequence[np.ndarray], rows: Sequence[Sequence[tuple[int, int, int]]], slot: int) -> list[bytes]:
    return [b"".join(tensors[t][slot, o:o + n].tobytes() for t, o, n in parts) for parts in rows]


def check_rows(engine: E.Engine, slots: int = 3, seed: int = 1, variants: Sequence[str] | None = None) -> dict[str, Any]:
    """Every variant's row digests against the reference on random rows: {variant: mismatches}."""
    rng = np.random.default_rng(seed)
    tensors, rows = random_tensors(rng, slots), random_rows()
    dev = Dev(engine)
    table = engine.rows(rows, [dev.put(t) for t in tensors])
    want = [row_digest(r) for s in range(slots) for r in ref_rows(tensors, rows, s)]
    want_gs = [want[s * len(rows) + g] for g in range(len(rows)) for s in range(slots)]
    out = {}
    for v in variants or engine.variants:
        got = E.digests(engine.row_digests(table, v))
        out[v] = sum(a != b for a, b in zip(got, want_gs, strict=True))
    return {"rows": len(rows), "slots": slots, "sizes": sorted({sum(n for _, _, n in p) for p in rows}), "mismatches": out}


FUSED_DEST = (0, 5, 1, 6, 7, 2, 3, 8, 9, 10, 11, 4)   # check_leaves' rows: unit d < 5 is row g alone, else the rest's d - 5
FUSED_RANK0 = (1 << 24) - 2                            # the proof blocks' ranks: a slot-tree leaf is one chunk below 2^24, else two


def check_leaves(engine: E.Engine, slots: int = 3, seed: int = 2) -> dict[str, Any]:
    """hm96 leaves under a salt key and under given salts, two domains, ranks and positions of 1 to 8 bytes (a frame leaf's
    rest of prefix and value one chunk up to 6 bytes each, q <= 47, else two); and fused: five rows each a proof block alone, whose
    slot-tree leaves (ranks FUSED_RANK0 + d, from ``frame_table``) the leaf kernel writes, the other seven rows' leaves
    compacted."""
    rng = np.random.default_rng(seed)
    tensors, rows = random_tensors(rng, slots), random_rows()[:12]
    dev = Dev(engine)
    G = len(rows)
    doms = [domain("unit-a", 1 << 17), domain("unit-b", 1 << 17)]
    did_of_row = [g % 2 for g in range(G)]
    rank = [(0, 1, 2, 255, 256, 65535, 65536, 70000, 1 << 40, 1 << 48, (1 << 40) - 1, (1 << 64) - 15)[g] for g in range(G)]
    position = [r + 7 * (g % 3) for g, r in enumerate(rank)]
    salt_row = [1000 + 3 * g for g in range(G)]
    table = engine.rows(rows, [dev.put(t) for t in tensors])
    lt = engine.leaf_table([d.domain_id for d in doms], did_of_row, rank, position, salt_row, SCHEMA_HM96_SHA512_ROW_V2)
    x = engine.row_digests(table)
    nonce0 = 5
    ys = np.zeros((G, slots, 192), dtype=np.uint8)
    for g in range(G):
        for s in range(slots):
            ys[g, s] = np.frombuffer(rng.bytes(192), dtype=np.uint8)
    units, n_rest = 5, G - 5
    slot_dom = domain("slots", 1 << 25)
    frames = engine.frame_table([slot_dom.domain_id], FUSED_RANK0, units)
    dest = engine.u32s(np.asarray(FUSED_DEST, dtype=np.uint32))
    bad_key = bad_given = bad_fused = 0
    for minb in LEAF_MINB[engine.host]:
        got_key = E.digests(engine.leaves(lt, x, E.SaltKey(SALT_KEY, nonce0), minb=minb))
        got_given = E.digests(engine.leaves(lt, x, dev.put(ys), minb=minb))
        run = engine._words(slots, units)
        rest = E.digests(engine.leaves(lt, x, E.SaltKey(SALT_KEY, nonce0), engine._words(n_rest, slots), minb, frames, dest,
                                       run))
        run_d = E.digests(run)
        for g in range(G):
            for s in range(slots):
                xr = row_digest(ref_rows(tensors, rows, s)[g])
                for y, got in ((salt(SALT_KEY, nonce0 + s, salt_row[g]), got_key), (ys[g, s].tobytes(), got_given)):
                    h = INST.tree_leaf(KEY, INST.commit_string(KEY, xr, y))
                    want = doms[did_of_row[g]].leaf(rank[g], position[g], SCHEMA_HM96_SHA512_ROW_V2, h)
                    bad = got[g * slots + s] != want
                    if got is got_key:
                        bad_key += bad
                        d = FUSED_DEST[g]
                        if d < units:
                            u = FUSED_RANK0 + d
                            bad_fused += run_d[s * units + d] != slot_dom.leaf(u, u, E.ROOT_SCHEMA, want)
                        else:
                            bad_fused += rest[(d - units) * slots + s] != want
                    else:
                        bad_given += bad
    return {"rows": G, "slots": slots, "leaf_minb": list(LEAF_MINB[engine.host]),
            "mismatches": {"salt_key": bad_key, "salts": bad_given, "fused": bad_fused}}


def check_trees(engine: E.Engine, seed: int = 3) -> dict[str, Any]:
    """Frame leaves, pads, levels and roots against tree_levels, for widths 1 to 13, one domain or one a tree."""
    rng = np.random.default_rng(seed)
    bad: dict[str, int] = {}
    for n in (1, 2, 3, 5, 8, 13):
        T = 3
        doms = [domain(f"tree-{n}-{t}", n) for t in range(T)]
        values = [[rng.bytes(64) for _ in range(n)] for _ in range(T)]
        vw = np.stack([E.words(v) for v in values])
        dev = Dev(engine)
        for shared in (True, False):
            ds = doms[:1] * T if shared else doms
            dids = [doms[0].domain_id] if shared else [d.domain_id for d in doms]
            leaves = engine.frame_leaves(dev.put(vw), dids, 0, ROOTS_SCHEMA)
            want_leaves = [ds[t].leaf(j, j, ROOTS_SCHEMA, values[t][j]) for t in range(T) for j in range(n)]
            k = f"n{n}-{'shared' if shared else 'own'}"
            bad[f"{k}-leaves"] = sum(a != b for a, b in zip(E.digests(leaves), want_leaves, strict=True))
            roots = E.digests(engine.root(leaves, dids))
            want_roots = [tree_levels(ds[t], want_leaves[t * n:(t + 1) * n])[-1][0] for t in range(T)]
            bad[f"{k}-roots"] = sum(a != b for a, b in zip(roots, want_roots, strict=True))
    big = domain("index", 1 << 20)
    kids = np.stack([E.words([rng.bytes(64) for _ in range(2 * 300)])])
    up = E.digests(engine.level(Dev(engine).put(kids), [big.domain_id], 9))
    kb = E.digests(kids)
    bad["level9-index-to-299"] = sum(up[i] != big.node(9, i, kb[2 * i], kb[2 * i + 1]) for i in range(300))
    first = (1 << 40) - 7
    up = E.digests(engine.level(Dev(engine).put(kids), [big.domain_id], 9, first, engine.node_mids([big.domain_id], 9)))
    bad["level9-index-from-2^40-7"] = sum(up[i] != big.node(9, first + i, kb[2 * i], kb[2 * i + 1]) for i in range(300))
    # a front child and each tree's children a slice of a wider array, read in place (the slot trees' fold)
    first = (1 << 56) + 3
    wide = np.stack([E.words([rng.bytes(64) for _ in range(9)]) for _ in range(3)])
    wd = Dev(engine).put(wide)
    up = E.digests(engine.level(wd[:, 2:7], [big.domain_id], 4, first, None, wd[:, 0]))
    want = []
    for t in range(3):
        c = E.digests(wide[t, [0, 2, 3, 4, 5, 6]])
        want += [big.node(4, first + i, c[2 * i], c[2 * i + 1]) for i in range(3)]
    bad["level4-front-child-strided"] = sum(a != b for a, b in zip(up, want, strict=True))
    # each proof block's slot-tree leaf from its frame table, one chunk below rank 2^24 and two from there, against frame_leaves
    for rank0 in (0, (1 << 24) - 3, (1 << 56) - 2):
        n, S = 5, 2
        ft = engine.frame_table([big.domain_id], rank0, n)
        vals = np.stack([E.words([rng.bytes(64) for _ in range(S)]) for _ in range(n)])
        units = Dev(engine).put(np.asarray([3, 0, 4, 1, 2], dtype=np.int64))
        run = engine.frame_units(ft, units, Dev(engine).put(vals), engine._words(S, n))
        ref = engine.frame_leaves(Dev(engine).put(np.ascontiguousarray(vals[[1, 3, 4, 0, 2]].transpose(1, 0, 2))),
                                  [big.domain_id], rank0)
        rd = E.digests(run)
        bad[f"frame-units-from-{rank0}"] = sum(a != b for a, b in zip(rd, E.digests(ref), strict=True))
        bad[f"frame-units-from-{rank0}-reference"] = sum(
            rd[s * n + d] != big.leaf(rank0 + d, rank0 + d, ROOTS_SCHEMA, E.digests(vals[[1, 3, 4, 0, 2]][d, s])[0])
            for s in range(S) for d in range(n))
    pads = E.digests(engine.pads([big.domain_id], 1, 65530, 10))
    bad["pads-65530"] = sum(pads[j] != big.pad(65530 + j) for j in range(10))
    return {"mismatches": bad}


def check_layout(engine: E.Engine, slots: int, sample: Sequence[int], seed: int = 4) -> dict[str, Any]:
    """The H200 layout's rows (one layer) at ``slots`` slots: every row's digest and leaf on the sampled slots."""
    import bench
    torch = engine.torch
    gen = torch.Generator(device=engine.device).manual_seed(seed)
    sources, rows = bench.h200_rows()
    dev_tensors = [torch.randint(0, 256, (slots, n), dtype=torch.uint8, device=engine.device, generator=gen) for n in sources]
    table = engine.rows(rows, dev_tensors)
    G = len(rows)
    dom = domain("layout-unit", 1)
    lt = engine.leaf_table([dom.domain_id], [0] * G, [0] * G, None, list(range(G)))
    x = engine.row_digests(table)
    xs = E.digests(x)
    ls = [E.digests(engine.leaves(lt, x, E.SaltKey(SALT_KEY), minb=m)) for m in LEAF_MINB[engine.host]]
    bad_x = bad_l = 0
    for s in sample:
        host = [t[s:s + 1].cpu().numpy() for t in dev_tensors]
        for g, row in enumerate(ref_rows(host, rows, 0)):
            xr = row_digest(row)
            bad_x += xs[g * slots + s] != xr
            h = INST.tree_leaf(KEY, INST.commit_string(KEY, xr, salt(SALT_KEY, s, g)))
            want = dom.leaf(0, 0, SCHEMA_HM96_SHA512_ROW_V2, h)
            bad_l += sum(got[g * slots + s] != want for got in ls)
    return {"slots": slots, "rows": G, "sampled_slots": list(sample), "leaf_minb": list(LEAF_MINB[engine.host]),
            "mismatches": {"x": bad_x, "leaves": bad_l}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layout-slots", type=int, nargs="*", default=[4096, 16384])
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    engine = E.Engine(a.device)
    t0 = time.time()
    doc: dict[str, Any] = {"device": str(engine.device), "default": engine.default, "variants": engine.variants,
                           "lib": str(engine.lib.path)}
    doc["rows"] = check_rows(engine)
    doc["rows_many_slots"] = check_rows(engine, slots=97, seed=11, variants=[engine.default, *engine.variants[:2]])
    doc["leaves"] = check_leaves(engine)
    doc["trees"] = check_trees(engine)
    if not engine.host:
        for slots in a.layout_slots:
            doc[f"layout_{slots}"] = check_layout(engine, slots, sorted({0, 1, 31, 32, slots // 2, slots - 1}))
    doc["seconds"] = time.time() - t0

    def total(d: Any) -> int:
        if isinstance(d, dict):
            return sum(total(v) for k, v in d.items() if k != "sizes")
        return int(d) if isinstance(d, (int, np.integer)) and not isinstance(d, bool) else 0

    doc["all_mismatches"] = sum(total(v["mismatches"]) for v in doc.values() if isinstance(v, dict) and "mismatches" in v)
    (a.out / "check.json").write_text(json.dumps(doc, indent=1, default=int) + "\n")
    print(json.dumps({"all_mismatches": doc["all_mismatches"], "seconds": round(doc["seconds"], 1)}))
    return 0 if doc["all_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
