"""red-team-standard-hash: recompute agkr-bound's pinned instance-set roots (`backends/gkr/verifier/src/commitments.rs`
PINS) from the frozen sets with the core only, so the roots A-GKR's tree check compares against are anchored to the set.

frame-v3 pins (`R+sha256`, `R+blake3`): `verity.commitments` `sha256_row_digest` / `blake3_row_digest` row digests,
`MerkleTree` over `CommitmentDomain(binding, owner, RangeIndexedDomain(0, N))`, binding the `hashauth.binding_digest`
convention (`identity_digest("verity/ligero-b/auth-binding/v2h", {dataset, tier, manifest_sha256, lo, hi, K, tree, schema})`),
owners a -1 / b -2 / y -1, y leaves the word big-endian at the u16 / u32 schema width.
vllm-v1 pins (`R+vllm-v1`): core `vllm_v1.pos_leaf` over the rows' little-endian bytes (y: the word's y_width
little-endian bytes), core `fold`, core `StepDomain(...).bind`, with the program / ctx / geo / layout digests of the
verifier's `operand_domain` (`vllm_v1.rs`) rebuilt here with the core `identity_digest`.

    python -m backends.direct.ligero.redteam.rtsh_agkr_pins --pins PATH/TO/commitments.rs --out OUT.json [--only REL]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

PROCS = int(os.environ.get("VY_CPU_THREADS", os.cpu_count() or 1))
BINDING_TAG = "verity/ligero-b/auth-binding/v2h"
MAP_TAG = "verity/gkr-commit/vllm-v1"
OWNER = {"a": -1, "b": -2, "y": -1}


def parse_pins(text: str) -> list[dict]:
    out = []
    for body in re.findall(r"Pin \{(.*?)\n\s*\},", text, re.S):
        p = dict(re.findall(r'(\w+): "([^"]*)"', body))
        p |= {k: int(v) for k, v in re.findall(r"(\w+): (\d+)\b(?!\")", body) if k not in p}
        if "relation" in p:
            out.append(p)
    return out


def _row(args):
    kind, words, bits, role = args
    from verity.commitments import rowleaf
    from verity.commitments import vllm_v1 as V
    if kind == "sha256":
        return rowleaf.sha256_row_digest(words, bits, role)
    if kind == "blake3":
        return rowleaf.blake3_row_digest(words, bits, role)
    return V.pos_leaf(rowleaf.row_bytes(words, bits))


def recompute(p: dict) -> dict:
    import numpy as np
    from verity.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain, identity_digest
    from verity.commitments import rowleaf
    from verity.commitments import vllm_v1 as V

    from .. import relchain
    from ..relations import relation

    base, leaf = p["relation"].split("+", 1)
    rel = relation(base)
    lo, n = int(p["lo"]), int(p["vus"])
    assert lo == 0, "only [0, N) pins"
    msha = relchain.instances_digest(rel, n)
    data = relchain.instances(rel, n, procs=PROCS)
    xb, wb, yw, k = int(p["x_bits"]), int(p["w_bits"]), int(p["y_width"]), int(p["k"])
    x_rows = [[int(w) for w in np.asarray(v[0]).reshape(-1)] for v in data]
    w_cols = [[int(w) for w in np.asarray(v[1]).reshape(-1)] for v in data]
    y = [int(rel.y_public(int(v[3]))) for v in data]
    res = {"relation": p["relation"], "dataset": [p["dataset"], rel.instances_dataset], "tier": [p["tier"], rel.instances_tier],
           "manifest_sha256": [p["manifest_sha256"], msha], "rows_k": [k, len(x_rows[0]), len(w_cols[0])]}
    with Pool(PROCS) as pool:
        if leaf in ("sha256", "blake3"):
            schema = rowleaf.SCHEMA_SHA256_ROW if leaf == "sha256" else rowleaf.SCHEMA_BLAKE3_ROW
            ysch = "u16" if yw == 2 else "u32"
            roots = {}
            for t, rows, bits, role in (("a", x_rows, xb, rowleaf.ROLE_X), ("b", w_cols, wb, rowleaf.ROLE_W), ("y", None, None, None)):
                sch = schema if t != "y" else ysch
                binding = bytes.fromhex(identity_digest(BINDING_TAG, {"dataset": rel.instances_dataset, "tier": rel.instances_tier,
                                                                      "manifest_sha256": msha, "lo": 0, "hi": n, "K": k, "tree": t, "schema": sch}))
                vals = ([w.to_bytes(yw, "big") for w in y] if t == "y"
                        else pool.map(_row, [(leaf, r, bits, role) for r in rows], chunksize=32))
                tree = MerkleTree(CommitmentDomain(binding, OWNER[t], RangeIndexedDomain(0, n)), dict(enumerate(vals)), lambda _p, s=sch: s)
                roots[t] = tree.commitment.root.hex()
        else:
            hi = lo + n
            program = bytes.fromhex(identity_digest(f"{MAP_TAG}/program/v0", {"relation": base}))
            geo = bytes.fromhex(identity_digest(f"{MAP_TAG}/geo/v0", {"vus": n, "K": k, "n_y": int(p["n_y"])}))
            roots = {}
            for t, rows, bits, value, words, count in (("a", x_rows, xb, "x-row", k, n), ("b", w_cols, wb, "w-column", k, n),
                                                       ("y", None, 8 * yw, "y-word", 1, n * int(p["n_y"]))):
                ctx = bytes.fromhex(identity_digest(f"{MAP_TAG}/ctx/v0", {"dataset": rel.instances_dataset, "tier": rel.instances_tier,
                                                                          "manifest_sha256": msha, "lo": lo, "hi": hi, "port": t}))
                layout = bytes.fromhex(identity_digest(f"{MAP_TAG}/layout/v0", {"port": t, "value": value, "word_bits": bits, "words": words,
                                                                                "byte_order": "little"}))
                dom = V.StepDomain(program, ctx, geo, layout, count)
                leaves = ([V.pos_leaf(w.to_bytes(yw, "little")) for w in y] if t == "y"
                          else pool.map(_row, [("pos", r, bits, None) for r in rows], chunksize=32))
                roots[t] = dom.bind(V.fold(leaves)).hex()
    res["roots"] = {t: [p[t], roots[t]] for t in "aby"}
    res["match"] = {"set": res["manifest_sha256"][0] == msha and res["dataset"][0] == res["dataset"][1] and res["tier"][0] == res["tier"][1],
                    **{t: p[t] == roots[t] for t in "aby"}}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    pins = parse_pins(Path(args.pins).read_text())
    out = []
    for p in pins:
        if args.only and p["relation"] != args.only:
            continue
        try:
            r = recompute(p)
        except Exception as e:  # noqa: BLE001
            r = {"relation": p["relation"], "error": f"{type(e).__name__}: {e}"}
        print(json.dumps({"relation": r["relation"], "match": r.get("match"), "error": r.get("error")}), flush=True)
        out.append(r)
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0 if out and all(r.get("match") and all(r["match"].values()) for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
