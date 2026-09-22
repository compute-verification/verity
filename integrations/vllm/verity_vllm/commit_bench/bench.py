"""CPERF 2026-09-17 §2-§4: waterfall of the current commitment path + controlled ablations on the same logical bytes.

    PYTHONPATH=. python -m verity_vllm.commit_bench.bench --store <commit dir> [--store-v2 <commit-v2 dir>] \
        --out <json> [--reps 3] [--procs 4] [--skip-weights] [--skip-emit]

Everything runs in ONE process on ONE thread unless the row says otherwise (``procs`` rows).  The RU trees (the
q-scaled second-stage volume) are measured over ALL trees of the store; the weights tree (v1 whole-tensor and
v2 row-leaf) once each.  Rows:

  waterfall.*   the production functions, one stage at a time (index parse, read, ids, framing, leaf hash, fold, record)
  e2e.*         ``build_commitments`` as replay.py times it, with the real hash and with the no-op hash (§3.2)
  emit.*        ``CaptureWriter.add`` replayed from the stored bytes (the capture-time copy + validation + leaf hash + write)
  ablate.*      the same bytes under: production / no-op hash / contiguous SHA-256 / packed identities / zero-copy /
                trusted index (no double hashing) / BLAKE3 / process-parallel trees
  layout.*      leaf granularity sweep over each RU's bytes (row leaves as-is, 1/4/16/64 KiB chunks, per-position,
                per-layer, one leaf per RU) with opening sizes
  weights.*     the same for the weights tree (v1 / v2 r=1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from verity_vllm import hashing, identity
from verity_vllm.capture import CaptureReader, CaptureWriter, RUReader
from verity_vllm.commit import MerkleTree, build_commitments, weights_tree

from . import fasttree as ft
from .prov import GpuSampler, provenance, timed

_REAL_H = hashing.H
ZERO32 = bytes(32)


def H_noop(tag: str, *parts: bytes) -> bytes:
    """Framing work of :func:`verity_vllm.hashing.H` without the SHA-256 arithmetic (the §3.2 ablation)."""
    t = tag.encode("utf-8")
    n = len(len(t).to_bytes(4, "big")) + len(t)
    for p in parts:
        n += len(len(p).to_bytes(8, "big")) + len(p)
    return ZERO32


class patched_noop:
    def __enter__(self) -> None:
        hashing.H = H_noop

    def __exit__(self, *exc: object) -> None:
        hashing.H = _REAL_H


# ------------------------------------------------------------------------------------------------------ RU material


class RUMaterial:
    """One RU's bytes and pre-resolved ids, loaded once so every ablation sees identical inputs."""

    def __init__(self, reader: CaptureReader, rid: str, step: int) -> None:
        ru = reader.ru(rid, step)
        self.tree_id = ru.tree_id
        self.rid, self.step, self.phase = rid, step, ru.index.phase
        self.positions = tuple(ru.index.positions)
        self.entries = list(ru.index.entries)
        self.values_path = ru.dir / "values.bin"
        self.buf = self.values_path.read_bytes()  # one contiguous copy of values.bin (bytes)
        self.nbytes = len(self.buf)
        self.ids = [e.canonical_id for e in self.entries]
        self.sorted_ids = identity.tree_sorted(self.ids)
        by_id = {e.canonical_id: e for e in self.entries}
        self.sorted_entries = [by_id[c] for c in self.sorted_ids]
        self.headers = [ft.leaf_header(e.canonical_id, e.dtype, e.shape) for e in self.sorted_entries]
        self.values = [self.buf[e.offset : e.offset + e.nbytes] for e in self.sorted_entries]
        mv = memoryview(self.buf)
        self.views = [mv[e.offset : e.offset + e.nbytes] for e in self.sorted_entries]
        self.index_leaves = {e.canonical_id: bytes.fromhex(e.leaf) for e in self.entries}
        self.dir = ru.dir


def load_material(reader: CaptureReader) -> list[RUMaterial]:
    return [RUMaterial(reader, r, s) for r, s in reader.ru_keys()]


# ------------------------------------------------------------------------------------------------------ waterfall


def waterfall(store: Path, mats: list[RUMaterial], reps: int) -> dict[str, Any]:
    reader = CaptureReader(store)
    keys = reader.ru_keys()
    rows: dict[str, Any] = {}

    def s_index():
        fresh = CaptureReader(store)
        for r, s in keys:
            fresh.ru(r, s)
        return {"n": len(keys)}
    rows["index_parse"] = timed(s_index, reps, what="CaptureReader + RUReader(index.json) for every RU")

    def s_read():
        n = 0
        for r, s in keys:
            ru = RUReader(store / "ru" / r / f"{s:06d}", r, s)
            for e in ru.index.entries:
                n += len(ru.read(e.canonical_id))
        return {"bytes": n}
    rows["read_memmap_to_bytes"] = timed(s_read, reps, what="RUReader.read: memmap slice -> bytes copy, every leaf (includes the index parse)")

    def s_ids():
        n = 0
        for m in mats:
            for c in m.ids:
                identity.parse(c)
            identity.tree_sorted(m.ids)
            hashing.ids_digest(m.sorted_ids)
            n += len(m.ids)
        return {"ids": n}
    rows["ids_parse_sort_digest"] = timed(s_ids, reps, what="identity.parse every id + tree_sorted + ids_digest (what MerkleTree/record do)")

    def s_frame():
        with patched_noop():
            for m in mats:
                for e, v in zip(m.sorted_entries, m.values, strict=True):
                    hashing.leaf_hash(e.canonical_id, e.dtype, e.shape, v)
    rows["leaf_framing_noop"] = timed(s_frame, reps, what="hashing.leaf_hash with H = framing only (no SHA): Python framing cost")

    def s_leaf():
        for m in mats:
            for e, v in zip(m.sorted_entries, m.values, strict=True):
                hashing.leaf_hash(e.canonical_id, e.dtype, e.shape, v)
    rows["leaf_hash"] = timed(s_leaf, reps, what="hashing.leaf_hash (production) over every leaf, bytes already in memory")

    leaves_by_tree = [{e.canonical_id: ft.leaf_digest(h, v) for e, h, v in zip(m.sorted_entries, m.headers, m.values, strict=True)} for m in mats]

    def s_fold():
        for m, lv in zip(mats, leaves_by_tree, strict=True):
            MerkleTree(lv, m.tree_id)
    rows["merkle_fold_production"] = timed(s_fold, reps, what="MerkleTree(leaves): tree_sorted + identity.parse per id + node/lift SHA-256 (production)")

    def s_fold_noop():
        with patched_noop():
            for m, lv in zip(mats, leaves_by_tree, strict=True):
                MerkleTree(lv, m.tree_id)
    rows["merkle_fold_noop"] = timed(s_fold_noop, reps, what="MerkleTree with H = framing only: the fold's parse/sort/Python cost without node SHA")

    def s_fold_packed():
        for m, lv in zip(mats, leaves_by_tree, strict=True):
            ft.fold([lv[c] for c in m.sorted_ids])
    rows["merkle_fold_packed"] = timed(s_fold_packed, reps, what="fasttree.fold over pre-sorted leaf digests (no parse, no dict, direct hashlib)")

    def s_record():
        trees = {}
        for m, lv in zip(mats, leaves_by_tree, strict=True):
            t = MerkleTree(lv, m.tree_id)
            trees[m.tree_id] = t.record("ru", request_id=m.rid, step=m.step, phase=m.phase, positions=list(m.positions))
        return {"json_bytes": len(json.dumps({"trees": trees}, indent=1, sort_keys=True))}
    rows["fold_plus_record"] = timed(s_record, reps, what="MerkleTree + tree.record (ids_digest) + json dump: fold row above plus finalisation")
    return rows


def e2e(store: Path, reps: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for recompute in (True, False):
        def f(recompute=recompute):
            fresh = CaptureReader(store)
            with GpuSampler() as g:
                doc = build_commitments(fresh, recompute_leaves=recompute, write=False)
            return {"trees": len(doc["trees"]), "gpu": g.summary()}
        rows[f"build_commitments_recompute={recompute}"] = timed(f, reps, what="replay.py's commitments_seconds cell (write=False)")
    for recompute in (True, False):
        def g(recompute=recompute):
            fresh = CaptureReader(store)
            with patched_noop():
                doc = build_commitments(fresh, recompute_leaves=recompute, write=False)
            return {"trees": len(doc["trees"])}
        rows[f"build_commitments_NOOP_HASH_recompute={recompute}"] = timed(g, reps, what="same call, H = framing only: pipeline intact, zero hash arithmetic (§3.2)")
    return rows


def emit_sim(mats: list[RUMaterial], profile_id: str, reps: int, tmp_root: Path) -> dict[str, Any]:
    """Replay ``CaptureWriter.add`` for every stored value into a scratch dir (the capture-time leaf hashing + copy + write)."""
    rows: dict[str, Any] = {}

    def run(noop: bool):
        d = Path(tempfile.mkdtemp(prefix="cperf-emit-", dir=tmp_root))
        try:
            ctx = patched_noop() if noop else _null()
            with ctx:
                n = 0
                for m in mats:
                    with CaptureWriter(d, m.rid, m.step, phase=m.phase, positions=m.positions, profile_id=profile_id, on_duplicate="flag") as w:
                        for e, v in zip(m.sorted_entries, m.values, strict=True):
                            w.add(e.canonical_id, e.dtype, e.shape, v)
                            n += 1
            return {"leaves": n}
        finally:
            shutil.rmtree(d, ignore_errors=True)
    rows["capture_writer_add_real"] = timed(lambda: run(False), reps, what="CaptureWriter.add + finish for every stored value (make_value validation, identity.parse, leaf SHA, file write, index.json)")
    rows["capture_writer_add_noop"] = timed(lambda: run(True), reps, what="same with H = framing only")
    return rows


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


# ------------------------------------------------------------------------------------------------------ ablations


def _root_of(store_commitments: dict, tree_id: str) -> str:
    return store_commitments["trees"][tree_id]["root"]


def ablations(mats: list[RUMaterial], commitments: dict, reps: int, procs: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    roots = {m.tree_id: _root_of(commitments, m.tree_id) for m in mats}
    total_bytes = sum(m.nbytes for m in mats)
    total_leaves = sum(len(m.ids) for m in mats)

    def check(name, fn):
        ok = all(fn(m).hex() == roots[m.tree_id] for m in mats)
        return ok

    # (1) production: read bytes already in memory -> leaf_hash -> MerkleTree
    def prod(m: RUMaterial) -> bytes:
        return MerkleTree({e.canonical_id: hashing.leaf_hash(e.canonical_id, e.dtype, e.shape, v) for e, v in zip(m.sorted_entries, m.values, strict=True)}, m.tree_id).root
    rows["1_production"] = timed(lambda: [prod(m) for m in mats], reps, roots_match=check("prod", prod))

    # (2) no-op hash, pipeline intact
    def prod_noop():
        with patched_noop():
            for m in mats:
                prod(m)
    rows["2_noop_hash_pipeline_intact"] = timed(prod_noop, reps, what="(1) with H = framing only: ceiling on any hash-function change")

    # (3) contiguous SHA-256 over each RU's values.bin, one update; and raw per-leaf sha256 with no framing / no Python H
    rows["3a_contiguous_sha256_per_ru"] = timed(lambda: [hashlib.sha256(m.buf).digest() for m in mats], reps, bytes=total_bytes,
                                                what="hashlib.sha256(values.bin) per RU: the streaming asymptote on these bytes")
    rows["3b_raw_sha256_per_leaf"] = timed(lambda: [[hashlib.sha256(v).digest() for v in m.values] for m in mats], reps, leaves=total_leaves,
                                           what="hashlib.sha256(value) per leaf, no framing, no id: SHA init/final + C-level per-call cost")
    rows["3c_contiguous_sha256_all"] = timed(lambda: hashlib.sha256(b"".join(m.buf for m in mats)).digest(), reps, bytes=total_bytes,
                                             what="one SHA-256 over all RU bytes concatenated (includes the join copy)")

    # (4) packed identities: headers precomputed, leaves in tree order, direct hashlib, fasttree fold
    def packed(m: RUMaterial) -> bytes:
        return ft.tree_root(m.headers, m.values)
    rows["4_packed_identity_sha256"] = timed(lambda: [packed(m) for m in mats], reps, roots_match=check("packed", packed),
                                             what="fasttree: pre-resolved ids/headers, tree order known, hashlib direct, same preimages")
    rows["4b_packed_header_build"] = timed(lambda: [[ft.leaf_header(e.canonical_id, e.dtype, e.shape) for e in m.sorted_entries] for m in mats], reps,
                                           what="the one-time cost of building the packed headers (paid at description/plan time, not per commit)")
    rows["4c_packed_noop"] = timed(lambda: [ft.tree_root(m.headers, m.values, ft._Noop) for m in mats], reps,
                                   what="fasttree with the no-op backend: Python floor of the packed pipeline")

    # (5) copy removal
    def packed_zero_copy(m: RUMaterial) -> bytes:
        return ft.tree_root(m.headers, m.views)
    rows["5a_packed_zero_copy_memoryview"] = timed(lambda: [packed_zero_copy(m) for m in mats], reps, roots_match=check("zc", packed_zero_copy),
                                                   what="(4) hashing memoryview slices instead of bytes copies")

    def packed_from_memmap(m: RUMaterial) -> bytes:
        import numpy as np
        mm = np.memmap(m.values_path, dtype=np.uint8, mode="r")
        mv = memoryview(mm)
        return ft.tree_root(m.headers, [mv[e.offset : e.offset + e.nbytes] for e in m.sorted_entries])
    rows["5b_packed_from_memmap_zero_copy"] = timed(lambda: [packed_from_memmap(m) for m in mats], reps, roots_match=check("mm", packed_from_memmap),
                                                    what="(4) straight from a fresh memmap of values.bin (page cache warm), no bytes() copies at all")

    def trusted(m: RUMaterial) -> bytes:
        return MerkleTree(m.index_leaves, m.tree_id).root
    rows["5c_trusted_index_no_rehash_production"] = timed(lambda: [trusted(m) for m in mats], reps, roots_match=check("tr", trusted),
                                                          what="production fold over the leaf hashes CaptureWriter already computed (removes the double leaf hashing)")
    rows["5d_trusted_index_packed_fold"] = timed(lambda: [ft.fold([m.index_leaves[c] for c in m.sorted_ids]) for m in mats], reps,
                                                 what="fasttree fold over the stored leaf hashes: the incremental cost of a commit when leaves are hashed once at capture")

    # (7) BLAKE3 (same framing, same tree; different digest -> roots differ by construction)
    try:
        b3 = ft.backend("blake3")
        rows["7a_packed_blake3_1thread"] = timed(lambda: [ft.tree_root(m.headers, m.values, b3) for m in mats], reps, what="(4) with blake3 (max_threads=1) as the leaf/node hash")
        import blake3 as _b3mod
        rows["7b_contiguous_blake3_per_ru_1thread"] = timed(lambda: [_b3mod.blake3(m.buf, max_threads=1).digest() for m in mats], reps, bytes=total_bytes)
        rows["7c_contiguous_blake3_all_auto_threads"] = timed(lambda: _b3mod.blake3(b"".join(m.buf for m in mats), max_threads=_b3mod.blake3.AUTO).digest(), reps, bytes=total_bytes,
                                                              what="blake3 rayon multi-thread over all RU bytes (NOT the commitment semantics; throughput reference only)")
    except Exception as e:  # noqa: BLE001
        rows["7_blake3"] = {"error": repr(e)}

    # (8) process-parallel trees (fork), packed sha256: independent trees are embarrassingly parallel
    if procs > 1:
        def par():
            with mp.get_context("fork").Pool(procs) as pool:
                out = pool.map(_packed_root_from_dir, [(str(m.dir), m.rid, m.step) for m in mats], chunksize=2)
            return {"roots_match": all(r == roots[t] for t, r in out)}
        rows[f"8_packed_sha256_{procs}procs_from_disk"] = timed(par, reps, what=f"{procs} forked workers, each: RUReader index parse + memmap + packed sha256 tree (wall includes pool start)")
    return rows


def _packed_root_from_dir(args: tuple[str, str, int]) -> tuple[str, str]:
    import numpy as np
    d, rid, step = args
    ru = RUReader(Path(d), rid, step)
    ids = identity.tree_sorted([e.canonical_id for e in ru.index.entries])
    by = {e.canonical_id: e for e in ru.index.entries}
    mm = memoryview(np.memmap(Path(d) / "values.bin", dtype=np.uint8, mode="r"))
    headers = [ft.leaf_header(c, by[c].dtype, by[c].shape) for c in ids]
    values = [mm[by[c].offset : by[c].offset + by[c].nbytes] for c in ids]
    return ru.tree_id, ft.tree_root(headers, values).hex()


# ------------------------------------------------------------------------------------------------------ layouts


def layouts(mats: list[RUMaterial], reps: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    total_bytes = sum(m.nbytes for m in mats)
    row_leaf = 1152  # the dominant activation row (bf16 x 576)

    def sweep(name: str, build):
        lay = [build(m) for m in mats]
        leaves = sum(len(h) for h, _ in lay)
        rows[name] = timed(lambda: [ft.tree_root(h, v) for h, v in lay], reps, leaves=leaves, bytes=total_bytes,
                           opening_one_row=ft.opening_bytes(max(1, leaves // len(mats)), max(len(v[0]) for _, v in lay if v), 0))
        rows[name]["leaves_decode_ru"] = len(lay[-1][0]) if mats[-1].phase == "decode" else None

    sweep("row_leaves_as_is", lambda m: (m.headers, m.values))
    for kib in (1, 4, 16, 64):
        sweep(f"chunk_{kib}KiB", lambda m, k=kib: ft.chunk_layout(m.tree_id, m.buf, k * 1024))
    sweep("one_leaf_per_ru", lambda m: ft.chunk_layout(m.tree_id, m.buf, max(1, m.nbytes)))

    def per_position(m: RUMaterial):
        groups: dict[int, list[bytes]] = defaultdict(list)
        for e, v in zip(m.sorted_entries, m.values, strict=True):
            groups[identity.parse(e.canonical_id).t or 0].append(v)
        hs, vs = [], []
        for t in sorted(groups):
            b = b"".join(groups[t])
            hs.append(ft.leaf_header(f"{m.tree_id}/t={t}", "uint8", (len(b),)))
            vs.append(b)
        return hs, vs
    sweep("per_position_concat", per_position)

    def per_layer_position(m: RUMaterial):
        groups: dict[tuple, list[bytes]] = defaultdict(list)
        for e, v in zip(m.sorted_entries, m.values, strict=True):
            c = identity.parse(e.canonical_id)
            groups[(c.t or 0, c.layer or "")].append(v)
        hs, vs = [], []
        for key in sorted(groups):
            b = b"".join(groups[key])
            hs.append(ft.leaf_header(f"{m.tree_id}/t={key[0]}/l={key[1]}", "uint8", (len(b),)))
            vs.append(b)
        return hs, vs
    sweep("per_layer_per_position_concat", per_layer_position)
    rows["_note"] = ("chunk/concat layouts hash the same bytes under the same framing with synthetic ids; the id parse of the concat "
                     "layouts is a one-time grouping cost outside the timed region (packed identities). opening_one_row = bytes to "
                     f"authenticate ONE {row_leaf}-B activation row: whole leaf + 32 B per level.")
    return rows


# ------------------------------------------------------------------------------------------------------ weights


def weights(store: Path, label: str) -> dict[str, Any]:
    reader = CaptureReader(store)
    w = reader.weights
    idx = w.index
    rows: dict[str, Any] = {"format_version": idx.format_version, "tensors": len(idx.entries), "leaves": idx.leaf_count()}
    raws = {e.canonical_id: w.read(e.canonical_id) for e in idx.entries}
    nbytes = sum(len(r) for r in raws.values())
    rows["bytes"] = nbytes
    rows["read_s"] = timed(lambda: [w.read(e.canonical_id) for e in idx.entries], 1)["first_s"]
    rows["production_weights_tree_recompute"] = timed(lambda: {"root": weights_tree(CaptureReader(store), recompute=True).root.hex()}, 1)
    rows["production_weights_tree_trusted_index"] = timed(lambda: {"root": weights_tree(CaptureReader(store), recompute=False).root.hex()}, 1)
    # packed
    headers, values = [], []
    ids = []
    for e in idx.entries:
        raw = raws[e.canonical_id]
        if e.layout.kind == "whole":
            ids.append(e.canonical_id)
            headers.append(ft.leaf_header(e.canonical_id, e.dtype, e.shape))
            values.append(raw)
        else:
            rb = e.row_nbytes
            mv = memoryview(raw)
            for start in e.block_starts():
                end = min(start + e.layout.rows_per_leaf, e.rows)
                cid = e.block_id(start)
                ids.append(cid)
                headers.append(ft.leaf_header(cid, e.dtype, e.block_shape(start, end)))
                values.append(mv[start * rb : end * rb])
    pos = {c: i for i, c in enumerate(ids)}
    order = [pos[c] for c in identity.tree_sorted(ids)]
    hs = [headers[i] for i in order]
    vs = [values[i] for i in order]
    rows["packed_sha256"] = timed(lambda: {"root": ft.tree_root(hs, vs).hex()}, 1)
    rows["packed_sha256"]["roots_match"] = rows["packed_sha256"]["root"] == rows["production_weights_tree_recompute"]["root"]
    rows["packed_noop"] = timed(lambda: ft.tree_root(hs, vs, ft._Noop), 1)
    rows["contiguous_sha256"] = timed(lambda: hashlib.sha256(b"".join(raws.values())).digest(), 1, bytes=nbytes)
    try:
        import blake3
        b3 = ft.backend("blake3")
        rows["packed_blake3_1thread"] = timed(lambda: ft.tree_root(hs, vs, b3), 1)
        rows["contiguous_blake3_auto_threads"] = timed(lambda: blake3.blake3(b"".join(raws.values()), max_threads=blake3.blake3.AUTO).digest(), 1, bytes=nbytes)
        rows["contiguous_blake3_1thread"] = timed(lambda: blake3.blake3(b"".join(raws.values()), max_threads=1).digest(), 1, bytes=nbytes)
    except Exception as e:  # noqa: BLE001
        rows["blake3"] = {"error": repr(e)}
    rows["opening_one_row"] = ft.opening_bytes(idx.leaf_count(), 1152)
    return rows


# ------------------------------------------------------------------------------------------------------ main


def summarize(mats: list[RUMaterial]) -> dict[str, Any]:
    by_phase = defaultdict(lambda: {"trees": 0, "leaves": 0, "bytes": 0})
    for m in mats:
        p = by_phase[m.phase]
        p["trees"] += 1
        p["leaves"] += len(m.ids)
        p["bytes"] += m.nbytes
    return {"trees": len(mats), "leaves": sum(len(m.ids) for m in mats), "bytes": sum(m.nbytes for m in mats), "by_phase": dict(by_phase)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="commit-mode store (v1 weights)")
    ap.add_argument("--store-v2", default=None, help="the weights_v2 --rows-per-leaf 1 copy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--skip-emit", action="store_true")
    ap.add_argument("--tmp", default="/tmp")
    args = ap.parse_args()
    store = Path(args.store)
    t0 = time.perf_counter()
    prov = provenance()
    reader = CaptureReader(store)
    commitments = reader.commitments
    mats = load_material(reader)
    doc: dict[str, Any] = {"schema": "veritor/cperf/bench/v1", "store": str(store), "profile_id": reader.profile_id, "reps": args.reps,
                           "provenance": prov, "ru_material": summarize(mats), "single_thread": True}
    Path(args.tmp).mkdir(parents=True, exist_ok=True)

    def section(name, fn):
        try:
            doc[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            doc[name] = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
        print(f"[bench] {name} done at {time.perf_counter() - t0:.1f}s", flush=True)

    section("waterfall", lambda: waterfall(store, mats, args.reps))
    section("e2e", lambda: e2e(store, args.reps))
    if not args.skip_emit:
        section("emit", lambda: emit_sim(mats, reader.profile_id, max(1, args.reps - 1), Path(args.tmp)))
    section("ablate", lambda: ablations(mats, commitments, args.reps, args.procs))
    section("layout", lambda: layouts(mats, args.reps))
    if not args.skip_weights:
        section("weights_v1", lambda: weights(store, "v1"))
        if args.store_v2:
            section("weights_v2", lambda: weights(Path(args.store_v2), "v2"))
    doc["seconds_total"] = time.perf_counter() - t0
    doc["provenance_end"] = {"load_avg_1m": os.getloadavg()[0], "gpu": prov["gpu"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")

    def compact(rows: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in rows.items():
            if isinstance(v, dict) and "median_s" in v:
                out[k] = {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("all_s", "what")}
            else:
                out[k] = v
        return out
    small = {k: (compact(doc[k]) if isinstance(doc.get(k), dict) else doc.get(k)) for k in ("provenance", "ru_material", "waterfall", "e2e", "emit", "ablate", "layout", "weights_v1", "weights_v2", "seconds_total") if k in doc}
    print("CPERF-BENCH-JSON " + json.dumps(small, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
