"""Boundary noninterference (contract §3/§9): compare captures leaf by leaf.

``python -m verity_vllm_adapter.compare --commit <run> --ref <run> [--ref <run>] --out compare.json``

Reference runs are replay-reference captures (hash-only sidecars ``ru/<rid>/<step>/leaves.json``
or full W2 RUs); the commit run is the replay+commit capture. Leaves are matched by canonical id;
equal leaf hashes (SHA-256 over id, dtype, shape, bytes) mean bit-identical values. Where both
sides stored values, mismatching leaves are diffed scalar by scalar; otherwise every scalar of a
mismatching leaf counts as mismatched (upper bound). ``compared_scalars`` counts every scalar of
every leaf present on both sides.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import dtypes


def strip_profile(cid: str) -> str:
    """The id without its p=<profile> prefix: two runs of the same requests under different
    profiles (M2 vs M1: another batch composition) are matched on the rest of the id."""
    return cid.split("/", 1)[1] if cid.startswith("p=") else cid


def load_leaves(run: Path, ignore_profile: bool = False) -> tuple[dict[str, dict[str, Any]], bool]:
    """canonical_id -> {dtype, shape, leaf, [ru_dir, offset, nbytes]}; bool = values available.
    Captured entries only: derived (inactive MoE slot) leaves are implied by the captured active
    slots' ids, so a routing difference shows up as ids only on one side."""
    leaves: dict[str, dict[str, Any]] = {}
    have_values = False
    key = strip_profile if ignore_profile else (lambda c: c)
    for idx in sorted(run.glob("ru/*/*/index.json")):
        doc = json.loads(idx.read_text())
        have_values = True
        for e in doc["entries"]:
            leaves[key(e["canonical_id"])] = {"dtype": e["dtype"], "shape": e["shape"], "leaf": e["leaf"], "ru_dir": idx.parent, "offset": e["offset"], "nbytes": e["nbytes"]}
    for lf in sorted(run.glob("ru/*/*/leaves.json")):
        doc = json.loads(lf.read_text())
        for cid, e in doc["leaves"].items():
            leaves[key(cid)] = {"dtype": e["dtype"], "shape": e["shape"], "leaf": e["leaf"]}
    return leaves, have_values


def read_value(e: dict[str, Any]) -> bytes | None:
    if "ru_dir" not in e:
        return None
    with (e["ru_dir"] / "values.bin").open("rb") as f:
        f.seek(e["offset"])
        return f.read(e["nbytes"])


def numel(shape: list[int]) -> int:
    return int(math.prod(shape)) if shape else 1


def compare(commit: Path, refs: list[Path], ignore_profile: bool = False) -> dict[str, Any]:
    c_leaves, c_vals = load_leaves(commit, ignore_profile)
    out: dict[str, Any] = {"commit": str(commit), "references": [str(r) for r in refs], "pairs": [], "ignore_profile": ignore_profile}
    identical_all = True
    compared_total = 0
    mismatched_total = 0
    for ref in refs:
        r_leaves, r_vals = load_leaves(ref, ignore_profile)
        both = set(c_leaves) & set(r_leaves)
        only_c = len(set(c_leaves) - set(r_leaves))
        only_r = len(set(r_leaves) - set(c_leaves))
        compared = sum(numel(c_leaves[c]["shape"]) for c in both)
        mismatched_leaves = [c for c in both if c_leaves[c]["leaf"] != r_leaves[c]["leaf"]]
        mismatched_scalars = 0
        localized = 0
        examples = []
        for cid in mismatched_leaves:
            a, b = read_value(c_leaves[cid]), read_value(r_leaves[cid])
            if a is not None and b is not None and len(a) == len(b):
                va = np.frombuffer(a, dtype=dtypes.view_dtype(c_leaves[cid]["dtype"]))
                vb = np.frombuffer(b, dtype=dtypes.view_dtype(r_leaves[cid]["dtype"]))
                diff = int((va != vb).sum())
                mismatched_scalars += diff
                localized += 1
                if len(examples) < 5:
                    examples.append({"canonical_id": cid, "differing_scalars": diff, "first_index": int(np.argmax(va != vb))})
            else:
                mismatched_scalars += numel(c_leaves[cid]["shape"])
                if len(examples) < 5:
                    examples.append({"canonical_id": cid, "differing_scalars": "unlocalized (hash-only)"})
        pair = {
            "reference": str(ref), "leaves_both": len(both), "leaves_only_commit": only_c, "leaves_only_reference": only_r,
            "compared_scalars": compared, "mismatched_leaves": len(mismatched_leaves), "mismatched_scalars": mismatched_scalars,
            "mismatches_localized": localized, "identical": not mismatched_leaves and only_c == 0 and only_r == 0,
            "first_divergent_ids": sorted(mismatched_leaves)[:5], "examples": examples,
            "reference_values_stored": r_vals, "commit_values_stored": c_vals,
        }
        out["pairs"].append(pair)
        identical_all = identical_all and pair["identical"]
        compared_total += compared
        mismatched_total += mismatched_scalars
    # reference vs reference (replay stability, §15.5 step 2)
    if len(refs) >= 2:
        a, _ = load_leaves(refs[0], ignore_profile)
        b, _ = load_leaves(refs[1], ignore_profile)
        both = set(a) & set(b)
        mm = [c for c in both if a[c]["leaf"] != b[c]["leaf"]]
        out["reference_vs_reference"] = {"leaves_both": len(both), "mismatched_leaves": len(mm), "identical": not mm and len(a) == len(b),
                                         "first_divergent_ids": sorted(mm)[:5]}
    out.update({
        "identical": identical_all,
        "compared_scalars": compared_total,
        "mismatched_scalars": mismatched_total,
        "reference_method": "fresh-process replay-reference captures (trace replay, no commitment writer) vs the replay+commit capture; "
                            "per-leaf SHA-256 leaf hashes (contract §5) matched by canonical id, scalar diff where values are stored",
    })
    for r in refs:
        rj = r / "replay.json"
        if rj.exists():
            out["reference_run_s"] = json.loads(rj.read_text()).get("timing", {}).get("run_s")
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ignore-profile", action="store_true", help="match ids without the p= prefix (cross-profile diagnostic, e.g. M2 vs M1)")
    args = ap.parse_args()
    doc = compare(Path(args.commit), [Path(r) for r in args.ref], ignore_profile=args.ignore_profile)
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("identical", "compared_scalars", "mismatched_scalars")}))
    for p in doc["pairs"]:
        print(json.dumps({k: p[k] for k in ("reference", "leaves_both", "leaves_only_commit", "leaves_only_reference", "mismatched_leaves", "mismatched_scalars")}))
    if "reference_vs_reference" in doc:
        print("ref vs ref:", json.dumps(doc["reference_vs_reference"]))


if __name__ == "__main__":
    main()
