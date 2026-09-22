"""Batch invariance of selective replay: the selected requests' leaves vs a full-replay control.

    PYTHONPATH=. python -m verity_vllm_adapter.compare_selected --selective <run> --control <commit run> --out <json>

The selective run replayed only ``J`` (its own batch); the control replayed every production request
(another batch, W3c's ``replay.py --mode commit``).  Value ids do not name the batch or the step, so
the same canonical id on both sides is the same logical value; equal leaf hashes (SHA-256 over id,
dtype, shape, bytes) mean bit-identical bytes.  Every id of the selective run must be present in the
control and hash-equal; mismatching leaves are diffed scalar by scalar.  Restricted to the RU leaves
(the boundary and weights trees are bound to the production record and the checkpoint separately).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import dtypes, identity
from verity_vllm_adapter.compare import load_leaves, numel, read_value


def compare_selected(selective: Path, control: Path) -> dict[str, Any]:
    s_leaves, _ = load_leaves(selective)
    c_leaves, _ = load_leaves(control)
    requests = sorted({cid.split("/")[1] for cid in s_leaves if cid.startswith("v=")} | {cid.split("r=")[1].split("/")[0] for cid in s_leaves if "r=" in cid})
    both = sorted(set(s_leaves) & set(c_leaves))
    only_s = sorted(set(s_leaves) - set(c_leaves))
    compared = sum(numel(s_leaves[c]["shape"]) for c in both)
    mismatched = [c for c in both if s_leaves[c]["leaf"] != c_leaves[c]["leaf"]]
    mismatched_scalars = 0
    examples = []
    for cid in mismatched:
        a, b = read_value(s_leaves[cid]), read_value(c_leaves[cid])
        if a is not None and b is not None and len(a) == len(b):
            va = np.frombuffer(a, dtype=dtypes.view_dtype(s_leaves[cid]["dtype"]))
            vb = np.frombuffer(b, dtype=dtypes.view_dtype(c_leaves[cid]["dtype"]))
            diff = int((va != vb).sum())
            mismatched_scalars += diff
            if len(examples) < 5:
                examples.append({"canonical_id": cid, "differing_scalars": diff, "first_index": int(np.argmax(va != vb))})
        else:
            mismatched_scalars += numel(s_leaves[cid]["shape"])
    # per interface family (op_type of the description node, and the port): leaves compared / equal
    try:
        desc = json.loads((selective / "description.json").read_text())
        op_type = {nd["op"]: nd["op_type"] for nd in desc["nodes"]}
    except Exception:  # noqa: BLE001
        op_type = {}
    fam: dict[str, dict[str, int]] = {}
    bad = set(mismatched)
    for cid in both:
        try:
            p = identity.parse(cid)
            key = f"{op_type.get(p.op or '', 'other')}/p={p.port}"
        except Exception:  # noqa: BLE001
            key = "other"
        f = fam.setdefault(key, {"leaves": 0, "equal": 0, "scalars": 0})
        f["leaves"] += 1
        f["equal"] += cid not in bad
        f["scalars"] += numel(s_leaves[cid]["shape"])
    return {
        "per_family": fam,
        "selective": str(selective), "control": str(control),
        "selective_leaves": len(s_leaves), "control_leaves": len(c_leaves), "leaves_both": len(both),
        "leaves_only_selective": len(only_s), "only_selective_examples": only_s[:5],
        "compared_scalars": compared, "mismatched_leaves": len(mismatched), "mismatched_scalars": mismatched_scalars,
        "first_divergent_ids": sorted(mismatched)[:5], "examples": examples,
        "bit_identical": not mismatched and not only_s,
        "method": "per-leaf SHA-256 leaf hashes (contract section 5) matched by canonical id between the selective run's RU leaves "
                  "(J replayed alone) and the full-replay control's (all N requests in one replay); scalar diff of mismatching leaves",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selective", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc = compare_selected(Path(args.selective), Path(args.control))
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("bit_identical", "leaves_both", "leaves_only_selective", "compared_scalars", "mismatched_leaves", "mismatched_scalars")}))


if __name__ == "__main__":
    main()
