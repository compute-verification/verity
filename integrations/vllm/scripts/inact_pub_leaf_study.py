"""INACT-PUB: leaf-format study for public-length inactive regions.

Read-only against a committed bundle (default: the INT B0 archive); everything else is
synthetic and in-process.  No commitment-store code is modified; this script *uses*
``verity_vllm.hashing`` / ``verity_vllm.commit`` to establish facts about the actual v2
row-leaf format and to measure candidate commitment treatments of inactive (post-EOS
padded) decode steps.

Facts established (printed and written to JSON):

  F1  The v1 leaf frame hashes the canonical id FIRST:
      leaf = H("verity-vllm/leaf/v1", canonical_id, dtype, shape, value).
      Two pad rows with byte-identical values but ids differing only in ``t=`` get
      DISTINCT leaves -> no leaf identity, no subtree sharing, no midstate sharing
      under the current format.  (There is no random salt; the id is the "salting".)
  F2  A real B0 decode step commits R leaves / V bytes (census from the bundle).
  F3  Candidate treatments, measured per inactive step (synthetic B0-shaped step):
        current   v2 leaf frame, full value hashed per leaf (what INACT-BASE does)
        id-last   proposed v3 frame H(tag, dtype, shape, value, id): one SHA-256
                  midstate per row class, shared across ALL inactive steps and
                  requests (hashlib.copy()); per leaf only the id suffix is absorbed
        id-free   proposed v3 frame H(tag, dtype, shape, value) with identity bound
                  by (tree, index) + ids_digest: the whole per-step leaf sequence is
                  step-invariant -> the step tree root is computed ONCE for all m
                  inactive steps; per step only ids_digest remains
        run-dedup a request-level tree where an inactive region is a contiguous run
                  of identical leaves: distinct hashes per level = O(log m); root
                  validated against a naive fold over the same leaves
  F4  Open-and-compare cost: authenticate one committed inactive row (verify_opening
      on a real tree built with the real code) + memcmp against the canonical
      constant.  This is the whole per-sample verifier cost of construction A.

Usage (from the repo root; CPU only, RSS well under 1 GB):

    .venv/bin/python scripts/inact_pub_leaf_study.py \
        [--bundle ~/veritor-artifacts/int/INT/INT-B0-20260908T0235Z/adapter/commit-v2] \
        [--out docs/data/inactive-public/leaf_study.json] [--steps 20]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # vllm-poc/ -> verity_vllm importable

from verity_vllm import hashing  # noqa: E402
from verity_vllm.commit import MerkleTree, verify_opening  # noqa: E402

# Fallback census (measured from INT-B0-20260908T0235Z, request 472ba20e step 000005)
# when the archive is not mounted: (op class, leaf count, total bytes).
FALLBACK_CENSUS = [
    ("layers.*.self_attn.rotary_emb.rope", 60, 46080),
    ("layers.*.post_attention_layernorm.rmsnorm", 60, 69120),
    ("layers.*.input_layernorm.rmsnorm", 59, 67968),
    ("layers.*.self_attn.qkv_proj.gemm", 30, 57600),
    ("layers.*.self_attn.attn.attn", 30, 34560),
    ("layers.*.self_attn.o_proj.gemm", 30, 34560),
    ("layers.*.mlp.gate_up_proj.gemm", 30, 184320),
    ("layers.*.mlp.act_fn.act_mul", 30, 92160),
    ("layers.*.mlp.down_proj.gemm", 30, 34560),
    ("norm.rmsnorm", 2, 2304),
    ("embed_tokens.embedding", 1, 1152),
    ("lm_head.gemm", 1, 98304),
    ("sampler.select", 1, 4),
]

ID_LAST_TAG = "verity-vllm/leaf/v3-idlast-CANDIDATE"  # not registered; study only
ID_FREE_TAG = "verity-vllm/leaf/v3-idfree-CANDIDATE"  # not registered; study only


def _u64be(n: int) -> bytes:
    return n.to_bytes(8, "big")


def _frame_prefix(tag: str) -> bytes:
    t = tag.encode()
    return len(t).to_bytes(4, "big") + t


def load_step_entries(bundle: Path | None) -> tuple[list[dict], str]:
    """(entries, source): each entry {canonical_id, dtype, shape, nbytes}."""
    if bundle is not None:
        ru_root = bundle / "ru"
        for req_dir in sorted(ru_root.iterdir()):
            for step_dir in sorted(req_dir.iterdir()):
                if not step_dir.name.isdigit():
                    continue
                idx = json.loads((step_dir / "index.json").read_text())
                entries = idx["entries"] if isinstance(idx, dict) else idx
                phase = idx.get("phase") if isinstance(idx, dict) else None
                if phase == "decode" or (phase is None and "ph=decode" in entries[0]["canonical_id"]):
                    out = [
                        {"canonical_id": e["canonical_id"], "dtype": e["dtype"],
                         "shape": e["shape"], "nbytes": e["nbytes"]}
                        for e in entries
                    ]
                    return out, str(step_dir)
        raise SystemExit(f"no decode step found under {ru_root}")
    # synthesize from the fallback census
    out = []
    profile, req = "d7afb8da14c4", "req-synth"
    special_l = {"norm.rmsnorm": "norm", "embed_tokens.embedding": "embed",
                 "lm_head.gemm": "lm_head", "sampler.select": "sampler"}
    for op, count, total in FALLBACK_CENSUS:
        per = total // count
        for i in range(count):
            if op.startswith("layers"):
                layer, lfield, port = op.replace("layers.*.", f"layers.{i % 30}."), f"l={i % 30}", i // 30
            else:
                layer, lfield, port = op, f"l={special_l[op]}", i
            cid = f"p={profile}/r={req}/t=36/ph=decode/{lfield}/op={layer}/port={port}"
            out.append({"canonical_id": cid, "dtype": "bfloat16", "shape": [per // 2], "nbytes": per})
    return out, "fallback census (embedded)"


def retarget_step(entries: list[dict], t: int) -> list[dict]:
    """The same step template at another decode position (ids differ only in t=)."""
    out = []
    for e in entries:
        cid = re.sub(r"/t=\d+/", f"/t={t}/", e["canonical_id"])
        out.append({**e, "canonical_id": cid})
    return out


def pad_value(nbytes: int) -> bytes:
    """Canonical absorbing fill for the study: zeros (the flag row would be 0x01)."""
    return b"\x00" * nbytes


def _sha_bytes_leaf_current(e: dict) -> int:
    """Bytes absorbed by SHA-256 for one v2 leaf (frame + parts)."""
    tag = len(hashing.LEAF_TAG) + 4
    parts = [e["canonical_id"].encode(), e["dtype"].encode(),
             hashing.shape_text(e["shape"]).encode(), b"\x00" * 0]
    return tag + sum(8 + len(p) for p in parts[:-1]) + 8 + e["nbytes"]


def _sha_bytes_nodes(leaf_count: int) -> int:
    """Bytes absorbed by the node/lift hashes of one tree build."""
    tag_node = len(hashing.NODE_TAG) + 4
    total, width = 0, leaf_count
    while width > 1:
        pairs, odd = width // 2, width % 2
        total += pairs * (tag_node + 2 * (8 + 32)) + odd * (tag_node + 8 + 32)
        width = pairs + odd
    return total


def _sha_bytes_ids(entries: list[dict]) -> int:
    return len(hashing.IDS_TAG) + 4 + sum(8 + len(e["canonical_id"].encode()) for e in entries)


def bench_current(entries: list[dict], steps: int) -> dict:
    """v2 frame, id-first: every leaf of every inactive step is a full-value hash."""
    t0 = time.perf_counter()
    n_hashes = 0
    for s in range(steps):
        step = retarget_step(entries, 1000 + s)
        leaves = {}
        for e in step:
            leaves[e["canonical_id"]] = hashing.leaf_hash(
                e["canonical_id"], e["dtype"], e["shape"], pad_value(e["nbytes"]))
            n_hashes += 1
        MerkleTree(leaves)  # nodes + ids_digest cost included in the wall time
    dt = time.perf_counter() - t0
    sha_bytes = (sum(_sha_bytes_leaf_current(e) for e in entries)
                 + _sha_bytes_nodes(len(entries)) + _sha_bytes_ids(entries))
    return {"seconds_per_step": dt / steps, "leaf_hashes_per_step": n_hashes // steps,
            "sha_bytes_per_step": sha_bytes}


def bench_id_last(entries: list[dict], steps: int) -> dict:
    """Candidate v3, id-last: midstate per row class shared across steps/requests."""
    # one midstate per distinct (dtype, shape, value) row class
    midstates: dict[tuple, "hashing.hashlib.sha256"] = {}
    import hashlib
    t0 = time.perf_counter()
    for e in entries:  # one-time per template
        key = (e["dtype"], tuple(e["shape"]), e["nbytes"])
        if key not in midstates:
            h = hashlib.sha256()
            h.update(_frame_prefix(ID_LAST_TAG))
            for part in (e["dtype"].encode(), hashing.shape_text(e["shape"]).encode(),
                         pad_value(e["nbytes"])):
                h.update(_u64be(len(part)) + part)
            midstates[key] = h
    setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in range(steps):
        step = retarget_step(entries, 1000 + s)
        leaves = {}
        for e in step:
            h = midstates[(e["dtype"], tuple(e["shape"]), e["nbytes"])].copy()
            cid = e["canonical_id"].encode()
            h.update(_u64be(len(cid)) + cid)
            leaves[e["canonical_id"]] = h.digest()
        MerkleTree(leaves)
    dt = time.perf_counter() - t0
    # per leaf: the id suffix is absorbed on a copied midstate (+ <=2 finalize blocks)
    sha_bytes = (sum(8 + len(e["canonical_id"].encode()) + 128 for e in entries)
                 + _sha_bytes_nodes(len(entries)) + _sha_bytes_ids(entries))
    return {"seconds_per_step": dt / steps, "midstate_setup_seconds": setup,
            "row_classes": len(midstates), "sha_bytes_per_step": sha_bytes}


def bench_id_free(entries: list[dict], steps: int) -> dict:
    """Candidate v3, id-free: the leaf sequence is step-invariant, so the step root is
    computed once; per step only ids_digest (id list still binds via ids_digest)."""
    t0 = time.perf_counter()
    ordered = sorted(entries, key=lambda e: e["canonical_id"])
    level = [hashing.H(ID_FREE_TAG, e["dtype"].encode(),
                       hashing.shape_text(e["shape"]).encode(), pad_value(e["nbytes"]))
             for e in ordered]
    while len(level) > 1:
        nxt = [hashing.node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(hashing.lift_hash(level[-1]))
        level = nxt
    once = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in range(steps):
        step = retarget_step(entries, 1000 + s)
        hashing.ids_digest(sorted(e["canonical_id"] for e in step))
    per_step = (time.perf_counter() - t0) / steps
    return {"seconds_per_step": per_step, "one_time_tree_seconds": once,
            "sha_bytes_per_step": _sha_bytes_ids(entries),
            "note": "per-step cost is ids_digest only; the root is a per-template constant "
                    "(zeros-absorbing) or a per-request constant (forwarded-state)"}


def _fold_naive(leaves: list[bytes]) -> tuple[bytes, int]:
    """The exact node/lift rule of ``verity_vllm.commit.MerkleTree``, counting hash calls."""
    calls = 0
    level = list(leaves)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashing.node_hash(level[i], level[i + 1])); calls += 1
        if len(level) % 2 == 1:
            nxt.append(hashing.lift_hash(level[-1])); calls += 1
        level = nxt
    return level[0], calls


def _fold_uniform(leaf: bytes, m: int) -> tuple[bytes, int]:
    """Fold a level of m identical leaves with O(log m) hash calls.

    State: the level is (k copies of u) followed by a short list of distinct tail
    nodes.  Per level one node(u,u) covers the whole uniform run; the tail (grown by
    at most one leftover u per level) is folded literally.  Same node/lift rule as
    the naive fold; root equality is asserted by the caller for every m.
    """
    calls = 0
    u, k = leaf, m
    tail: list[bytes] = []
    while k + len(tail) > 1:
        width = k + len(tail)
        if k >= 2:
            paired = hashing.node_hash(u, u); calls += 1
        else:
            paired = None
        nxt_tail: list[bytes] = []
        rest = ([u] if k % 2 == 1 else []) + tail
        i = 0
        while i + 1 < len(rest):
            nxt_tail.append(hashing.node_hash(rest[i], rest[i + 1])); calls += 1
            i += 2
        if i < len(rest):  # odd level width -> lift the trailing node
            assert width % 2 == 1
            nxt_tail.append(hashing.lift_hash(rest[i])); calls += 1
        if paired is None:
            # uniform run exhausted; finish naively over the (short) tail
            root, extra = _fold_naive(nxt_tail) if len(nxt_tail) > 1 else (nxt_tail[0], 0)
            return root, calls + extra
        u, k, tail = paired, k // 2, nxt_tail
    return (u if k == 1 else tail[0]), calls


def bench_run_dedup(m_values: list[int]) -> dict:
    """A request-level tree where the inactive region is a contiguous run of m identical
    (id-free) leaves: distinct hash invocations = O(log m) instead of m-1.  The dedup
    root is asserted equal to the naive fold's root for every m."""
    leaf = hashing.H(ID_FREE_TAG, b"bfloat16", b"[576]", pad_value(1152))
    out = {}
    for m in m_values:
        root_naive, calls_naive = _fold_naive([leaf] * m)
        root_dedup, calls_dedup = _fold_uniform(leaf, m)
        assert root_dedup == root_naive, f"dedup root mismatch at m={m}"
        out[str(m)] = {"m": m, "hash_calls_naive": calls_naive,
                       "hash_calls_dedup": calls_dedup, "roots_equal": True}
    return out


def bench_open_and_compare(entries: list[dict]) -> dict:
    """Construction A per-sample verifier cost with the REAL tree code: verify_opening
    of one committed pad row + byte-compare against the canonical constant."""
    leaves, values = {}, {}
    for e in entries:
        v = pad_value(e["nbytes"])
        values[e["canonical_id"]] = (e, v)
        leaves[e["canonical_id"]] = hashing.leaf_hash(e["canonical_id"], e["dtype"], e["shape"], v)
    tree = MerkleTree(leaves, "ru/req-synth/000040")
    # the biggest row (lm_head logits) is the worst case
    cid = max(entries, key=lambda e: e["nbytes"])["canonical_id"]
    e, v = values[cid]
    from verity_vllm.capture import Value
    opening = tree.open(Value(cid, e["dtype"], tuple(e["shape"]), v, "derived"))
    reps = 200
    t0 = time.perf_counter()
    ok = True
    for _ in range(reps):
        ok &= verify_opening(tree.root, tree.leaf_count, opening)
        ok &= opening.value == pad_value(e["nbytes"])
    dt = (time.perf_counter() - t0) / reps
    assert ok
    path_bytes = sum(32 for p in opening.path if p is not None)
    return {"seconds_per_sample": dt, "row_nbytes": e["nbytes"],
            "path_bytes": path_bytes, "depth": len(opening.path),
            "opened_bytes_total": e["nbytes"] + path_bytes,
            "verified": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_bundle = Path.home() / "veritor-artifacts/int/INT/INT-B0-20260908T0235Z/adapter/commit-v2"
    ap.add_argument("--bundle", default=str(default_bundle))
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    bundle = Path(args.bundle).expanduser()
    entries, source = load_step_entries(bundle if bundle.exists() else None)
    total_bytes = sum(e["nbytes"] for e in entries)

    # F1: identical value bytes, ids differing only in t= -> distinct leaves
    a, b = entries[0], retarget_step([entries[0]], 999_999)[0]
    val = pad_value(a["nbytes"])
    la = hashing.leaf_hash(a["canonical_id"], a["dtype"], a["shape"], val)
    lb = hashing.leaf_hash(b["canonical_id"], b["dtype"], b["shape"], val)
    f1 = {"id_a": a["canonical_id"], "id_b": b["canonical_id"],
          "value_bytes_equal": True, "leaves_equal": la == lb,
          "leaf_frame": "H(tag, canonical_id, dtype, shape, value) - id FIRST",
          "consequence": "no leaf identity across positions; no midstate sharing; "
                         "no constant-subtree dedup under the current v2 format"}
    assert la != lb

    results = {
        "source_step": source,
        "census": {"leaves_per_decode_step": len(entries), "bytes_per_decode_step": total_bytes},
        "F1_leaf_identity": f1,
        "F3_current_format": bench_current(entries, args.steps),
        "F3_id_last_midstate": bench_id_last(entries, args.steps),
        "F3_id_free": bench_id_free(entries, args.steps),
        "F3_run_dedup": bench_run_dedup([16, 64, 1024, 4096]),
        "F4_open_and_compare": bench_open_and_compare(entries),
    }
    cur = results["F3_current_format"]["seconds_per_step"]
    results["speedups_vs_current"] = {
        "id_last": cur / results["F3_id_last_midstate"]["seconds_per_step"],
        "id_free_per_step": cur / results["F3_id_free"]["seconds_per_step"],
    }

    print(json.dumps(results, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwritten: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
