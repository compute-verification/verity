"""Join the per-rank capture directories of a tensor-parallel replay into one run (lane TP).

Each rank's ``<out>/rank<r>/`` is an ordinary capture-format run whose ids all carry ``/s=<r>``. The
joined run ``<out>/`` holds, per replay unit, the union of the ranks' leaves (distinct ids, so no
conflicts), one boundary per request (the ranks' boundaries must agree byte for byte: the same
scheduler drove both), the union of the ranks' weight shards (``w=<param>/s=<r>``) and a description
whose nodes record the ranks observed. Leaves are re-added through the writer so every leaf hash is
recomputed from the bytes; nothing is trusted from the per-rank indices.

The join is bookkeeping, not computation: the two ranks' capture streams are two authentication
locations of one program's values. Which logical value each ``/s=<r>`` leaf is a copy of is the
profile's binding (``profiles/b1-tp2/capture_mapping.py``), not this module's concern.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from verity_vllm import identity
from verity_vllm.capture import (
    BOUNDARY_FILE,
    INDEX_FILE,
    RU_DIR,
    WEIGHTS_DIR,
    Boundary,
    RunWriter,
    RUReader,
    WeightsReader,
)


def _rank_dirs(out: Path) -> list[Path]:
    dirs = sorted((p for p in out.glob("rank*") if p.is_dir() and p.name[4:].isdigit()), key=lambda p: int(p.name[4:]))
    if len(dirs) < 2:
        raise SystemExit(f"{out}: expected >= 2 rank directories, found {[d.name for d in dirs]}")
    return dirs


def _ru_dirs(rank_dir: Path) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    ru_root = rank_dir / RU_DIR
    if not ru_root.exists():
        return out
    for req in ru_root.iterdir():
        if not req.is_dir():
            continue
        for step in req.iterdir():
            if step.is_dir() and step.name.isdigit():
                out[(req.name, int(step.name))] = step
    return out


def merge_ranks(out: Path, *, store_values: bool, log=print) -> dict[str, Any]:
    out = Path(out)
    ranks = _rank_dirs(out)
    t0 = time.perf_counter()
    manifest = json.loads((ranks[0] / "manifest.json").read_text()) if (ranks[0] / "manifest.json").exists() else None
    per_rank_rus = [_ru_dirs(r) for r in ranks]
    keys = set(per_rank_rus[0])
    for k in per_rank_rus[1:]:
        if set(k) != keys:
            only = sorted(set(k) ^ keys)[:5]
            raise SystemExit(f"rank replay-unit sets differ (same scheduler must drive every rank): e.g. {only}")
    stats: dict[str, Any] = {"ranks": len(ranks), "replay_units": len(keys), "leaves": 0, "bytes": 0, "per_rank_leaves": [0] * len(ranks),
                             "boundary_disagreements": [], "rus_with_position_disagreement": []}

    if store_values:
        if manifest is None:
            raise SystemExit("stored per-rank runs must carry manifest.json")
        writer = RunWriter(out, manifest)
        for rid, step in sorted(keys):
            readers = [RUReader(pr[(rid, step)], rid, step) for pr in per_rank_rus]
            phase = readers[0].index.phase
            positions = tuple(readers[0].index.positions)
            for r in readers[1:]:
                if r.index.phase != phase or tuple(r.index.positions) != positions:
                    stats["rus_with_position_disagreement"].append({"request_id": rid, "step": step})
            ru = writer.ru(rid, step, phase=phase, positions=positions, on_duplicate="error")
            for i, r in enumerate(readers):
                for e in r.index.entries:
                    if e.derived is not None:
                        continue  # (no slot families in the dense TP row)
                    data = r.read(e.canonical_id)
                    ru.add(e.canonical_id, e.dtype, list(e.shape), data, origin=e.origin)
                    stats["leaves"] += 1
                    stats["bytes"] += len(data)
                    stats["per_rank_leaves"][i] += 1
            ru.finish()
        # boundaries: one per request, the ranks must agree
        for rid in sorted({rid for rid, _ in keys}):
            docs = []
            for r in ranks:
                p = r / RU_DIR / rid / BOUNDARY_FILE
                docs.append(p.read_bytes() if p.exists() else None)
            if docs[0] is None:
                continue
            if any(d != docs[0] for d in docs[1:]):
                stats["boundary_disagreements"].append(rid)
            writer.write_boundary(Boundary.from_json(json.loads(docs[0])))
        # weights: every rank's shards, re-added (bytes re-hashed)
        wreaders = [WeightsReader(r / WEIGHTS_DIR) for r in ranks if (r / WEIGHTS_DIR / INDEX_FILE).exists()]
        if wreaders:
            ww = writer.weights(store_values=all(w.index.store_values for w in wreaders),
                                source_sha256=wreaders[0].index.source_sha256)
            n = 0
            for w in wreaders:
                for e in w.index.entries:
                    if w.index.store_values and e.offset is not None:
                        raw = w.read(e.canonical_id)
                        ww.add(e.param, e.dtype, list(e.shape), raw, shard=e.shard, source=e.source)
                    else:
                        ww.add(e.param, e.dtype, list(e.shape), None, shard=e.shard, sha256=e.sha256, leaf=e.leaf, source=e.source)
                    n += 1
            ww.finish()
            stats["weights"] = n
        writer.write_description(merge_descriptions([json.loads((r / "description.json").read_text()) for r in ranks]))
        writer.finish()
    else:
        # hash-only reference runs: leaves.json sidecars, union per replay unit
        for rid, step in sorted(keys):
            docs = [json.loads((pr[(rid, step)] / "leaves.json").read_text()) for pr in per_rank_rus]
            merged = dict(docs[0])
            merged["leaves"] = {}
            for i, d in enumerate(docs):
                if d["phase"] != docs[0]["phase"] or d["positions"] != docs[0]["positions"]:
                    stats["rus_with_position_disagreement"].append({"request_id": rid, "step": step})
                merged["leaves"].update(d["leaves"])
                stats["per_rank_leaves"][i] += len(d["leaves"])
            stats["leaves"] += len(merged["leaves"])
            d = out / RU_DIR / rid / f"{step:06d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "leaves.json").write_text(json.dumps(merged))
        # the compiled-fx capture writes no description in reference mode (sidecars only); the eager one does
        if all((r / "description.json").exists() for r in ranks):
            (out / "description.json").write_text(json.dumps(merge_descriptions([json.loads((r / "description.json").read_text()) for r in ranks]), indent=1, sort_keys=True))
        if manifest is not None:
            shutil.copy(ranks[0] / "manifest.json", out / "manifest.json")
    _merge_compiled_artifacts(out, ranks, stats)
    stats["seconds"] = time.perf_counter() - t0
    log(f"tp_merge: {len(ranks)} ranks -> {stats['replay_units']} RUs, {stats['leaves']} leaves, {stats['bytes'] / 1e9:.2f} GB in {stats['seconds']:.1f}s")
    return stats


def _merge_compiled_artifacts(out: Path, ranks: list[Path], stats: dict[str, Any]) -> None:
    """[COMP, CX1] a compiled-fx per-rank capture also leaves ``kernels/<hash>.py`` (the Inductor kernel sources the
    description's relations name, keyed by normalised-source hash: identical across ranks when the ranks compiled the same
    bodies) and ``fx_ids.json`` (the rank's record-id table). Union the kernels into ``<out>/kernels/`` (a hash present on one
    rank only is recorded), keep every rank's id table as ``fx_ids.rank<r>.json``."""
    kdirs = [r / "kernels" for r in ranks if (r / "kernels" / "index.json").exists()]
    if kdirs:
        dst = out / "kernels"
        dst.mkdir(exist_ok=True)
        union: dict[str, str] = {}
        per_rank: list[set[str]] = []
        for kd in kdirs:
            idx = json.loads((kd / "index.json").read_text())
            names = idx.get("kernels", {})
            per_rank.append(set(names))
            for h, name in names.items():
                src = kd / f"{h}.py"
                if src.exists() and not (dst / f"{h}.py").exists():
                    shutil.copy(src, dst / f"{h}.py")
                union.setdefault(h, name)
        common = set.intersection(*per_rank) if per_rank else set()
        (dst / "index.json").write_text(json.dumps({"kernels": union, "hash": "sha256[:16] of the name-normalised source (compiled_fx.kernels.normalise_kernel)",
                                                    "ranks": len(kdirs), "kernels_common_to_all_ranks": len(common),
                                                    "kernels_on_some_ranks_only": sorted(set(union) - common)}, indent=1) + "\n")
        stats["kernels"] = {"union": len(union), "common_to_all_ranks": len(common), "some_ranks_only": len(set(union) - common)}
    for r in ranks:
        f = r / "fx_ids.json"
        if f.exists():
            shutil.copy(f, out / f"fx_ids.{r.name}.json")


def merge_descriptions(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """One description for the joined run. Every rank sees the same node set with the same per-row
    shapes (a symmetric split); the joined node records the ranks that materialised it. Nodes seen on
    one rank only (none expected) are kept and flagged."""
    base = dict(docs[0])
    base.pop("shard", None)
    ranks = [int(d.get("shard")) for d in docs]
    world = len(docs)
    by_op: dict[str, dict[str, Any]] = {}
    for d in docs:
        rank = int(d.get("shard"))
        for n in d["nodes"]:
            cur = by_op.get(n["op"])
            if cur is None:
                cur = dict(n)
                cur["ranks_observed"] = []
                cur["shard_shapes"] = {}
                cur["consts_by_shard"] = {}
                by_op[n["op"]] = cur
            cur["ranks_observed"].append(rank)
            cur["shard_shapes"][str(rank)] = n.get("shape")
            cur["consts_by_shard"][str(rank)] = n.get("consts") or {}
            if n.get("shape") != cur.get("shape"):
                cur["shard_shape_disagreement"] = True
    nodes = list(by_op.values())
    for n in nodes:
        # schema: `shards` is the world size (instances carry s=<rank>); every node is materialised
        # on every rank in a symmetric split, a node seen on fewer ranks is flagged
        n["shards"] = world
        n["ranks_missing"] = sorted(set(ranks) - set(n["ranks_observed"]))
        if all(c == n["consts_by_shard"][str(ranks[0])] for c in n["consts_by_shard"].values()):
            n.pop("consts_by_shard")
        # legacy op_type strings from per-rank descriptions written before the schema mapping
        n["op_type"] = {"all_reduce": "allreduce", "all_gather": "other"}.get(n["op_type"], n["op_type"])
        # the collectives are functions of every rank's operand: name them explicitly so one relation
        # checks each rank's replica (all_reduce: p0 + p1; all_gather: p0 ++ p1)
        if n["op"].endswith(".all_reduce") or n["op"].endswith(".all_gather"):
            src = next((a for a in n.get("args", []) if a.get("source") == "node"), None)
            if src is not None:
                n["args"] = [{"name": f"p{r}", "source": "node", "op": src["op"], "port": src.get("port", 0), "shard": r}
                             for r in ranks]
                n["relation"] = "allreduce.bf16.v1" if n["op"].endswith(".all_reduce") else "allgather.copy.v1"
        if n["op"].endswith(".embedding") and n.get("runtime_op") == "_C.vocab_parallel_embedding":
            n["relation"] = "embedding.vocab_parallel.v1"
    base["nodes"] = nodes
    base["tensor_parallel"] = {"ranks": ranks, "world_size": world,
                               "note": "ids carry /s=<rank>; a node's 'shards' is the world size, 'ranks_observed' the ranks that materialised it"}
    return base


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="run dir containing rank0/, rank1/, ...")
    ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--description-only", action="store_true",
                    help="re-join only description.json from the rank dirs and re-derive required.json (the values/ru/weights stay)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.description_only:
        from verity_vllm import required as vrequired
        from verity_vllm.description import Description

        ranks = sorted(p for p in out.glob("rank*") if p.is_dir())
        docs = [json.loads((r / "description.json").read_text()) for r in ranks]
        merged = merge_descriptions(docs)
        (out / "description.json").write_text(json.dumps(merged, indent=1, sort_keys=True))
        desc_obj = Description(merged)
        partition = {"partition_id": "row-default", "decomposition": {"default": "row"}}
        if (out / "required.json").exists():
            (out / "required.json").unlink()
        req = vrequired.freeze_required(out / "required.json", desc_obj, partition)
        print(json.dumps({"nodes": len(desc_obj.nodes), "ranks": len(ranks), "required_interfaces": len(req["interfaces"])}, indent=1))
        return
    stats = merge_ranks(out, store_values=not a.hash_only)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    _main()
