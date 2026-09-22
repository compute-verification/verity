#!/usr/bin/env python
"""[REC 2026-09-09, S5 "unconditional weight binding"] The AUDITOR's authorized weight commitment, computed
BEFORE the audit from the auditor's own pinned reference -- never copied from a prover's manifest.

What it reads:   manifests/checkpoints.json (frozen repo / revision / safetensors sha256 / config sha256), the
                 local checkpoint files (re-hashed against those pins), the semantic profile's weight rules
                 (which tensors the program reads, in which dtype, fused / padded how) and config.json.
What it computes: every weight leaf exactly as capture format v2 defines it (one leaf per parameter row for
                 rank >= 2 tensors, whole-tensor leaves for vectors / scalars; derived buffers from config.json;
                 runtime constants from the profile) and the Merkle root over them (``MerkleTree`` sorted order).
What it emits:   weights_authorized.json = {checkpoint identity, expected weights root, leaf_count, ids_digest,
                 layout (format 2, rows_per_leaf 1, rules digest), leaf table (param -> canonical id, dtype, shape,
                 leaf ids), covers (the exact leaf-id set + digest), MEASURED price (seconds, bytes)}.

The record's verifier (selective_verify.json#bindings.model) later rebuilds the same tree from the checkpoint and
compares it with the prover's committed root; e2e.instance_check compares BOTH with this pre-challenge value.
Run in the vLLM venv (torch + safetensors):   python scripts/rec_weights_authorize.py --case B0 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
sys.path.insert(0, str(POC))

from verity_vllm import hashing, identity  # noqa: E402
from verity_vllm.capture import WEIGHTS_TREE_ID, LeafLayout, WeightEntry, row_block_leaves  # noqa: E402
from verity_vllm.commit import MerkleTree  # noqa: E402
from verity_vllm.weights_v2 import row_eligible  # noqa: E402
from verity_vllm_adapter import binding as vb  # noqa: E402
from verity_vllm_adapter import profile as prof  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="B0")
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--semantic-profile", default=str(POC / "manifests" / "semantic-profiles" / "llama-eager-batch-invariant-v1.json"))
    ap.add_argument("--rows-per-leaf", type=int, default=1)
    ap.add_argument("--rope-dtype", default="bfloat16", help="dtype of the rotary cos/sin cache buffer as the runtime holds it (bf16 for a bf16 model)")
    ap.add_argument("--rope-device", default="cuda", help="device on which the runtime computed the rotary cache (vLLM builds it on the model device; cos/sin rounding differs cpu vs cuda)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t_all = time.perf_counter()
    cp = prof.checkpoint_for(a.case, a.checkpoints)
    local = Path(cp["local_path"])
    # 1. the reference: files re-hashed against the frozen pins (the auditor's own copy, the auditor's own pins)
    file_shas: dict[str, str] = {}
    for fname, sha in cp["safetensors_sha256"].items():
        h = hashlib.sha256()
        with (local / fname).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        file_shas[fname] = h.hexdigest()
        if h.hexdigest() != sha:
            raise SystemExit(f"{fname}: local sha256 {h.hexdigest()[:16]} != frozen {sha[:16]} -- refusing to authorize")
    config_sha = hashlib.sha256((local / "config.json").read_bytes()).hexdigest()
    if cp.get("config_sha256") and config_sha != cp["config_sha256"]:
        raise SystemExit("config.json sha256 != frozen pin -- refusing to authorize")
    config = json.loads((local / "config.json").read_text())
    profile = vb.load_semantic_profile(a.semantic_profile)
    profile_sha = hashlib.sha256(Path(a.semantic_profile).read_bytes()).hexdigest()
    rules = profile["weights"]["rules"]
    wdtype = profile["weights"]["dtype"]
    sym = vb._symbols(profile, config)
    # 2. the expected parameter set from the rules alone (no prover index)
    params: list[tuple[str, dict, dict]] = []
    for rule in rules:
        if rule.get("optional_when") and config.get(rule["optional_when"]):
            continue
        if "{i}" in rule["param"]:
            for i in range(int(sym["L"])):
                env = {**sym, "i": i}
                params.append((vb._ev(rule["param"], env), rule, env))
        else:
            params.append((rule["param"], rule, dict(sym)))
    ckpt = vb._Checkpoint(local, sorted(cp["safetensors_sha256"]))
    leaves: dict[str, bytes] = {}
    table: list[dict] = []
    derived: dict[str, str] = {}
    bytes_total = 0
    t0 = time.perf_counter()
    import torch

    for param, rule, env in params:
        cid = str(identity.weight_id(param, None))
        if rule.get("derived") == "constant":
            dtype, shape = rule["dtype"], tuple(int(x) for x in rule.get("shape", []))
        elif rule.get("derived") == "rope_cos_sin_cache":
            dtype = a.rope_dtype
            shape = None  # from the recomputed bytes below
        else:
            dtype, shape = wdtype, None
        # a stand-in entry for the recomputation (shape / sha are filled from the bytes; for rope the dtype decides the bytes)
        probe = WeightEntry(canonical_id=cid, param=param, shard=None, dtype=dtype, shape=shape or (), nbytes=0, sha256="", leaf="", offset=None)
        if rule.get("derived") == "rope_cos_sin_cache":
            raw = vb.rope_cos_sin_cache(int(vb._ev(rule["rotary_dim"], env)), float(vb._ev(rule["base"], env)), int(vb._ev(rule["max_position"], env)), dtype, a.rope_device)
            n_el = len(raw) // torch.tensor([], dtype=getattr(torch, dtype)).element_size()
            shape = (int(vb._ev(rule["max_position"], env)), n_el // int(vb._ev(rule["max_position"], env)))
            derived[param] = f"rope cos/sin cache recomputed from config.json ({dtype}, shape {list(shape)}, computed on {a.rope_device})"
        else:
            raw = vb._recompute_entry(probe, rule, env, ckpt, wdtype, derived)
            if rule.get("derived") == "constant":  # the adapter's note compares against a committed sha; here there is none (auditor side)
                derived[param] = f"constant {rule['value']} ({rule['dtype']}, shape {list(rule.get('shape', []))}) fixed by the profile; recomputed here (no prover bytes compared)"
            if shape is None:
                parts = [ckpt.tensor(vb._ev(name, env)) for name in rule["from"]]
                t = torch.cat(parts, dim=int(rule.get("cat_dim", 0))) if len(parts) > 1 else parts[0]
                mult = rule.get("pad_rows_to_multiple")
                n0 = t.shape[0] + ((int(mult) - t.shape[0] % int(mult)) if mult and t.shape[0] % int(mult) else 0)
                shape = (int(n0), *[int(d) for d in t.shape[1:]])
        bytes_total += len(raw)
        rpl = a.rows_per_leaf if row_eligible(shape) else None
        layout = LeafLayout("rows", rpl, (shape[0] + rpl - 1) // rpl, None) if rpl else LeafLayout()
        entry = WeightEntry(canonical_id=cid, param=param, shard=None, dtype=dtype, shape=shape, nbytes=len(raw), sha256=hashing.sha256_hex(raw),
                            leaf=hashing.leaf_hash(cid, dtype, shape, raw).hex(), offset=None, layout=layout)
        if layout.kind == "rows":
            ids = []
            for start, block_leaf in zip(entry.block_starts(), row_block_leaves(entry, raw), strict=True):
                leaves[entry.block_id(start)] = block_leaf
                ids.append(entry.block_id(start))
            leaf_ids = {"first": ids[0], "last": ids[-1], "count": len(ids)}
        else:
            leaves[cid] = bytes.fromhex(entry.leaf)
            leaf_ids = {"first": cid, "last": cid, "count": 1}
        table.append({"param": param, "canonical_id": cid, "dtype": dtype, "shape": list(shape), "nbytes": len(raw), "layout": layout.kind,
                      "rows_per_leaf": layout.rows_per_leaf or None, "leaf_ids": leaf_ids, "tensor_sha256": entry.sha256,
                      "source": rule.get("from") or f"derived:{rule.get('derived')}"})
    tree = MerkleTree(leaves, WEIGHTS_TREE_ID)
    t_tree = time.perf_counter() - t0
    ids_sorted = identity.tree_sorted(list(leaves))
    covers_digest = hashlib.sha256("\n".join(ids_sorted).encode()).hexdigest()
    table_digest = hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()
    doc = {
        "schema": "verity-vllm/weights-authorized/v1", "utc": prof.utc_stamp(), "case": a.case,
        "claim": "every weight value used by the verified computation is bound to THIS authorized weight object (GUIDANCE §57.10 S5)",
        "authorized_independently_of_prover": True,
        "checkpoint": {"repo": cp["repo"], "revision": cp["revision"], "safetensors_sha256": dict(cp["safetensors_sha256"]), "files_rehashed": file_shas,
                       "config_sha256": config_sha, "pinned_in": str(Path(a.checkpoints)), "local_path": str(local)},
        "layout": {"capture_format": 2, "rows_per_leaf": a.rows_per_leaf, "row_rule": "rank >= 2 and shape[0] > 1 -> one leaf per row block; else whole-tensor leaf",
                   "weights_dtype": wdtype, "rope_cache_dtype": a.rope_dtype, "rope_cache_device": a.rope_device, "semantic_profile": str(Path(a.semantic_profile)), "semantic_profile_sha256": profile_sha,
                   "rules": len(rules), "params": len(params), "leaf_table_digest": table_digest},
        "expected_root": tree.root.hex(), "leaf_count": tree.leaf_count, "ids_digest": tree.ids_digest.hex(), "tree_id": WEIGHTS_TREE_ID,
        "leaf_table": table, "derived_buffers": derived,
        "covers": {"kind": "weights-tree-leaf-ids/v1", "units": [{"canonical_id": r["canonical_id"], "leaf_ids": r["leaf_ids"]} for r in table],
                   "leaf_count": tree.leaf_count, "digest": covers_digest, "digest_rule": "sha256 of the tree-sorted leaf ids joined by newline",
                   "verification": "e2e.instance_check compares this root with selective_verify.json#bindings.model (leaves rebuilt from the pinned checkpoint at audit "
                                   "time, root_matches) and with the prover's committed weights root; any disagreement -> the check did not bind these units"},
        "no_other_path": {"assertion": "the program's weight reads are exactly the leaves of this table: every weight Input class of the semantic profile's rules is a "
                                       "leaf (or row block) here and the verifier refuses a committed index with a missing or unexpected parameter "
                                       "(bindings.model missing_params / unexpected_params); a weight value reaching the computation outside this table has no "
                                       "commitment and fails the InputEq opening of any sampled unit that reads it",
                          "checked_by": "selective_verify.json#bindings.model.{missing_params, unexpected_params, root_matches} == {[], [], true}",
                          "program_side_map": "program weight Input gates -> (param, row) is the derived correspondence (DERIVE-CORR L2/L3); not re-derived here"},
        "price": {"seconds_total": round(time.perf_counter() - t_all, 3), "seconds_leaves_and_tree": round(t_tree, 3), "bytes_read": bytes_total, "label": "MEASURED",
                  "note": "auditor-side, once per checkpoint; the per-audit price is the verifier's bindings.model.seconds (selective_verify.json)"},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights_authorized.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"[rec_weights_authorize] root {tree.root.hex()} leaves {tree.leaf_count} ids_digest {tree.ids_digest.hex()[:16]} params {len(params)} "
          f"bytes {bytes_total} in {doc['price']['seconds_total']} s -> {out / 'weights_authorized.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
