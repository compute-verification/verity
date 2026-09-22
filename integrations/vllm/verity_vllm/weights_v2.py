"""Convert a bundle's ``weights/`` from format v1 (one leaf per tensor) to v2 (one leaf per block of
rows; capture-format.md section 11).

The tensor bytes are unchanged (``values.bin`` is byte-identical, so the manifest's
``weights_sha256`` still binds it); only the leaf layout, ``leaves.bin`` and the weights tree root
change.  Every other tree of the bundle is untouched, so ``commitments.json`` is copied with the
weights record replaced and ``weights/conversion.json`` records the v1 root it replaced.

    PYTHONPATH=. python -m verity_vllm.weights_v2 <src_run_dir> <dst_run_dir> [--rows-per-leaf 1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .capture import (
    COMMITMENTS_FILE,
    INDEX_FILE,
    LEAVES_FILE,
    WEIGHTS_DIR,
    WEIGHTS_TREE_ID,
    CaptureError,
    CaptureReader,
    WeightsWriter,
)
from .commit import weights_tree

CONVERSION_FILE = "conversion.json"


def row_eligible(shape: tuple[int, ...]) -> bool:
    """Tensors committed per row: rank >= 2 with more than one row.  Vectors (norm weights,
    biases) stay whole-tensor leaves; opening one costs a few KB."""
    return len(shape) >= 2 and shape[0] > 1


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_tree(src: Path, dst: Path, *, skip: set[Path]) -> int:
    """Hard-link (fallback copy) every file of ``src`` into ``dst`` except ``skip`` (relative)."""
    n = 0
    for root, _dirs, files in os.walk(src):
        for name in files:
            p = Path(root) / name
            rel = p.relative_to(src)
            if rel in skip:
                continue
            _link_or_copy(p, dst / rel)
            n += 1
    return n


def convert_bundle(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    rows_per_leaf: int = 1,
    link: bool = True,
) -> dict[str, Any]:
    """Write a v2-weights copy of the bundle at ``src`` into ``dst`` (must not exist)."""
    src_p, dst_p = Path(src), Path(dst)
    if dst_p.exists():
        raise CaptureError(f"{dst_p} exists")
    reader = CaptureReader(src_p)
    if not reader.has_weights():
        raise CaptureError("bundle has no weights index")
    if not reader.weights.index.store_values:
        raise CaptureError("v2 conversion needs the stored weight bytes (weights-values store)")
    old_index = reader.weights.index
    t0 = time.perf_counter()
    skip = {
        Path(WEIGHTS_DIR) / INDEX_FILE,
        Path(WEIGHTS_DIR) / LEAVES_FILE,
        Path(WEIGHTS_DIR) / CONVERSION_FILE,
        Path(COMMITMENTS_FILE),
    }
    if link:
        linked = link_tree(src_p, dst_p, skip=skip)
    else:
        shutil.copytree(src_p, dst_p, ignore=lambda d, names: [n for n in names if (Path(d) / n).relative_to(src_p) in skip])
        linked = -1
    # values.bin is re-emitted by the writer in the same order -> byte-identical; drop the link first.
    values_dst = dst_p / WEIGHTS_DIR / "values.bin"
    if values_dst.exists():
        values_dst.unlink()
    writer = WeightsWriter(
        dst_p,
        store_values=True,
        source_sha256=dict(old_index.source_sha256),
        format_version=2,
    )
    rows_leaves = 0
    for e in old_index.entries:
        raw = reader.weights.read(e.canonical_id)
        rpl = rows_per_leaf if row_eligible(e.shape) else None
        new = writer.add(
            e.param,
            e.dtype,
            e.shape,
            raw,
            shard=e.shard,
            sha256=e.sha256,
            leaf=e.leaf,
            source=e.source,
            rows_per_leaf=rpl,
        )
        if new.layout.kind == "rows":
            rows_leaves += new.layout.leaf_count
    new_index = writer.finish()
    t_write = time.perf_counter() - t0
    new_reader = CaptureReader(dst_p)
    t1 = time.perf_counter()
    tree = weights_tree(new_reader)
    t_tree = time.perf_counter() - t1
    old_commitments = reader.commitments
    old_record = None
    doc: dict[str, Any]
    if old_commitments is not None:
        doc = json.loads(json.dumps(old_commitments))
        old_record = doc.get("trees", {}).get(WEIGHTS_TREE_ID)
        doc.setdefault("trees", {})[WEIGHTS_TREE_ID] = tree.record(
            "weights", weights_sha256=dict(new_index.source_sha256)
        )
        (dst_p / COMMITMENTS_FILE).write_text(
            json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    info = {
        "schema": "verity-vllm/weights-conversion/v1",
        "source_run_dir": str(src_p),
        "rows_per_leaf": rows_per_leaf,
        "from": {
            "format_version": old_index.format_version,
            "leaf_count": old_index.leaf_count(),
            "root": old_record.get("root") if old_record else None,
        },
        "to": {
            "format_version": 2,
            "leaf_count": tree.leaf_count,
            "row_leaves": rows_leaves,
            "whole_leaves": tree.leaf_count - rows_leaves,
            "root": tree.root.hex(),
            "leaves_nbytes": new_index.leaves_nbytes,
        },
        "values_nbytes": new_index.values_nbytes,
        "values_sha256": _file_sha256(values_dst),
        "values_sha256_unchanged": _file_sha256(values_dst)
        == _file_sha256(src_p / WEIGHTS_DIR / "values.bin"),
        "linked_files": linked,
        "seconds": {"write": round(t_write, 3), "tree": round(t_tree, 3)},
    }
    (dst_p / WEIGHTS_DIR / CONVERSION_FILE).write_text(
        json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rows-per-leaf", type=int, default=1)
    ap.add_argument("--copy", action="store_true", help="copy instead of hard-linking unchanged files")
    args = ap.parse_args(argv)
    info = convert_bundle(args.src, args.dst, rows_per_leaf=args.rows_per_leaf, link=not args.copy)
    json.dump(info, sys.stdout, indent=1, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
