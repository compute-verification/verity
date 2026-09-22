#!/usr/bin/env python3
"""C0 on a commit store: derive every leaf position from static context, build the position-identified root, cross-check
openings against the production scheme, time rank/unrank and the two leaf hashings.

    PYTHONPATH=. python scripts/cprof_c0.py --store <commit dir> [--reps 3]

Prints ``CPERF-C0-JSON {...}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

import numpy as np  # noqa: E402

from verity_vllm import identity  # noqa: E402
from verity_vllm.capture import CaptureReader  # noqa: E402
from verity_vllm.commit import MerkleTree, Value, verify_opening  # noqa: E402
from verity_vllm.commit_bench import fasttree as ft  # noqa: E402
from verity_vllm.commit_bench.prov import provenance  # noqa: E402
from verity_vllm.commit_bench.semantic import POS_LEAF_TAG, Layout, RUContext, StaticContext, Template, pos_leaf, verify_position_opening  # noqa: E402
from verity_vllm.commit_bench.stream import StreamingMerkle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--openings", type=int, default=64)
    a = ap.parse_args()
    reader = CaptureReader(a.store)
    keys = reader.ru_keys()
    rng = random.Random(23)
    # ---- static context -------------------------------------------------------------------------------------------
    # template from ONE decode RU (stands in for the Program layout); prompt length from the record (here: the prefill RU's
    # position count, which is what the record's request table pins)
    first_decode = next(k for k in keys if k[1] == 1)
    template = Template.from_decode_index(reader.ru(*first_decode).index.entries)
    prompt_len = {rid: len({identity.parse(e.canonical_id).t for e in reader.ru(rid, 0).index.entries}) for rid, s in keys if s == 0}
    profile = identity.parse(reader.ru(*first_decode).index.entries[0].canonical_id).profile_id
    program_digest = hashlib.sha256(b"program:" + template.digest()).digest()  # placeholder until the Program correspondence is wired
    query_id = hashlib.sha256(b"query:ru-module-outputs/v0").digest()
    trees = reader.commitments["trees"]

    # ---- 1. bijection: every stored id has an arithmetic position; every position names a stored id -------------------
    bij = {"trees": 0, "ok": 0, "ids_total": 0, "mismatch_examples": []}
    t_rank, t_unrank, t_cid = [], [], []
    per_tree = {}
    for rid, step in keys:
        ru = reader.ru(rid, step)
        ctx = RUContext(profile, rid, prompt_len[rid], step)
        lay = Layout(template, ctx)
        ents = ru.index.entries
        by = {e.canonical_id: e for e in ents}
        ok = lay.count == len(ents)
        t0 = time.perf_counter()
        derived = [lay.canonical_id(i) for i in range(lay.count)]
        t_cid.append((time.perf_counter() - t0) / lay.count)
        ok = ok and set(derived) == set(by)
        if ok:
            for i, c in enumerate(derived):
                _t, row = lay.unrank(i)
                e = by[c]
                if (e.dtype, tuple(e.shape)) != (row.dtype, row.shape):
                    ok = False
                    bij["mismatch_examples"].append([c, e.dtype, list(e.shape), row.dtype, list(row.shape)])
                    break
            t0 = time.perf_counter()
            ranks = [lay.rank(c) for c in derived]
            t_rank.append((time.perf_counter() - t0) / lay.count)
            ok = ok and ranks == list(range(lay.count))
            t0 = time.perf_counter()
            for i in range(lay.count):
                lay.unrank(i)
            t_unrank.append((time.perf_counter() - t0) / lay.count)
        else:
            bij["mismatch_examples"].append([ru.tree_id, lay.count, len(ents), sorted(set(derived) ^ set(by))[:3]])
        bij["trees"] += 1
        bij["ok"] += ok
        bij["ids_total"] += len(ents)
        per_tree[ru.tree_id] = (lay, ctx, ents, by, ru.dir / "values.bin")
    bij["mismatch_examples"] = bij["mismatch_examples"][:5]

    # ---- 2. position-identified root vs today's id-carrying leaves: cost --------------------------------------------
    # precompute the position order once (static: it is the emit order under the semantic layout) so the timed loop hashes only
    pos_order = {tid: [by[lay.canonical_id(i)] for i in range(lay.count)] for tid, (lay, ctx, ents, by, vpath) in per_tree.items()}

    def build_pos_timed() -> tuple[dict[str, bytes], float]:
        roots = {}
        t0 = time.perf_counter()
        for tid, (lay, ctx, ents, by, vpath) in per_tree.items():
            mm = memoryview(np.memmap(vpath, dtype=np.uint8, mode="r"))
            digests = [pos_leaf(mm[e.offset : e.offset + e.nbytes]) for e in pos_order[tid]]
            roots[tid] = ft.fold(digests)
        return roots, time.perf_counter() - t0

    headers = {tid: [(ft.leaf_header(c, by[c].dtype, by[c].shape), by[c]) for c in identity.tree_sorted(list(by))] for tid, (lay, ctx, ents, by, vpath) in per_tree.items()}

    def build_packed_timed() -> tuple[dict[str, bytes], float]:
        roots = {}
        t0 = time.perf_counter()
        for tid, (lay, ctx, ents, by, vpath) in per_tree.items():
            mm = memoryview(np.memmap(vpath, dtype=np.uint8, mode="r"))
            digests = [ft.leaf_digest(h, mm[e.offset : e.offset + e.nbytes]) for h, e in headers[tid]]
            roots[tid] = ft.fold(digests)
        return roots, time.perf_counter() - t0

    leaves = sum(len(v) for v in pos_order.values())
    tp, tk = [], []
    for _ in range(a.reps):
        pos_roots, t = build_pos_timed()
        tp.append(t)
        packed_roots, t = build_packed_timed()
        tk.append(t)
    packed_match = sum(1 for tid, r in packed_roots.items() if trees[tid]["root"] == r.hex())
    header_bytes_today = statistics.median(len(h) for hs in headers.values() for h, _ in hs)
    val_bytes = statistics.median(e.nbytes for es in pos_order.values() for e in es)

    def blocks(n: int) -> int:
        return (n + 9 + 63) // 64

    # ---- 3. openings: position + static context authenticates the same bytes as the production id-carrying opening ----
    op_ok = op_same = 0
    t_open_pos, t_open_prod, t_verify_pos = [], [], []
    for _ in range(a.openings):
        tid = rng.choice(list(per_tree))
        lay, ctx, ents, by, vpath = per_tree[tid]
        i = rng.randrange(lay.count)
        mm = memoryview(np.memmap(vpath, dtype=np.uint8, mode="r"))
        sc = StaticContext(program_digest, query_id, template, ctx, 0, lay.count)
        R = sc.root(pos_roots[tid])
        # opener: unrank -> which value to fetch (or regenerate), then the path from a re-streamed tree
        t0 = time.perf_counter()
        cid = lay.canonical_id(i)
        b = StreamingMerkle(track=i)
        for e in pos_order[tid]:
            b.add(POS_LEAF_TAG, mm[e.offset : e.offset + e.nbytes])
        mroot = b.finish()
        e = by[cid]
        value = bytes(mm[e.offset : e.offset + e.nbytes])
        t_open_pos.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ok, named = verify_position_opening(R, sc, i, value, b.path)
        t_verify_pos.append(time.perf_counter() - t0)
        # production: id-carrying leaf opened in the production tree and verified with the production verifier
        t0 = time.perf_counter()
        prod = MerkleTree({c: bytes.fromhex(by[c].leaf) for c in by}, tid)
        popen = prod.open(Value(cid, e.dtype, tuple(e.shape), value, "captured"))
        t_open_prod.append(time.perf_counter() - t0)
        pok = verify_opening(bytes.fromhex(trees[tid]["root"]), trees[tid]["leaf_count"], popen)
        op_ok += ok and pok and mroot == pos_roots[tid]
        op_same += (named == cid) and (popen.value == value)
        # tamper: a different value at the same position must fail
        assert not verify_position_opening(R, sc, i, bytes(len(value)), b.path)[0] or value == bytes(len(value))

    out = {
        "schema": "veritor/cperf/c0/v1", "store": a.store, "provenance": {k: provenance()[k] for k in ("host", "cpu", "load_avg_1m", "python")},
        "static_context": {
            "template_rows_per_position": len(template.per_position), "template_rows_per_ru": len(template.per_ru),
            "template_digest": template.digest().hex(), "template_json_bytes": len(json.dumps([[r.layer, r.op, r.port, r.dtype, list(r.shape)] for r in template.per_position + template.per_ru])),
            "profile": profile, "requests": len(prompt_len), "prompt_lens": sorted(set(prompt_len.values())),
            "per_ru_context_bytes": "request id + u32 prompt_len + u32 step (+ epoch) ~ 48 B", "program_digest": program_digest.hex()[:16] + "... (placeholder: sha256 of template until the Program correspondence is wired)",
        },
        "bijection": bij,
        "rank_unrank_us_per_leaf": {"unrank": statistics.median(t_unrank) * 1e6, "canonical_id_format": statistics.median(t_cid) * 1e6, "rank_from_id_string_with_parse": statistics.median(t_rank) * 1e6},
        "leaf_hash": {
            "today_preimage_bytes": int(header_bytes_today + 8 + val_bytes), "today_sha_blocks": blocks(int(header_bytes_today + 8 + val_bytes)),
            "position_preimage_bytes": int(len(POS_LEAF_TAG) + 8 + val_bytes), "position_sha_blocks": blocks(int(len(POS_LEAF_TAG) + 8 + val_bytes)),
            "ids_digest_preimage_bytes_today_total": sum(len(c) for _, (lay, ctx, ents, by, vpath) in per_tree.items() for c in by),
        },
        "build": {"leaves": leaves, "trees": len(per_tree), "packed_ids_s_median": statistics.median(tk), "position_s_median": statistics.median(tp),
                  "packed_us_per_leaf": statistics.median(tk) / leaves * 1e6, "position_us_per_leaf": statistics.median(tp) / leaves * 1e6,
                  "packed_roots_match_commitments_json": f"{packed_match}/{len(per_tree)}"},
        "openings": {"n": a.openings, "position_opening_verified_and_production_verified": op_ok, "same_value_and_id": op_same,
                     "open_pos_restream_s_median": statistics.median(t_open_pos), "verify_pos_us_median": statistics.median(t_verify_pos) * 1e6,
                     "open_prod_rebuild_tree_s_median": statistics.median(t_open_prod)},
    }
    print("CPERF-C0-JSON " + json.dumps(out, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
