"""[R10 owner 7] REVIEW-BUILT sidecar (owner 6's `veritor/r10/e2e-witness/v1` shape) from owner 6's committed stage-1 outputs
(`roots.json`, `steps/step_<s>.bin`) plus the HF safetensors file for the weight rows — used for (a) an independent check that the
weights root binds the real stored weights (non-fused tensors: o_proj, norms, embed), (b) the binding negatives of BRIEF §6 run
against every available verifier.  Trees are built with `review_l1_verify.py` (hashlib only); safetensors/numpy are used only to READ
the weight bytes.  Selection: owner 6's challenge rule restricted by rejection sampling to o_proj coordinates (labelled — NOT the
protocol's dispersed sample; the W rows of qkv/gate_up are vLLM-fused tensors that need the live model).

Usage: review_build_sidecar.py <run dir> <safetensors path> --B 8 --out sidecar.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

from verity_capture.bench import review_l1_verify as rv

CH = 256
K = 576
# challenge derivation: challenge-v1 (verity_protocol.challenge); the R10 tag is retired
QUERY_ID = "17593554089537b1"
MODULES = (("self_attn.qkv_proj", 960, "input_layernorm"), ("self_attn.o_proj", 576, "self_attn.attn"), ("mlp.gate_up_proj", 3072, "post_attention_layernorm"))
N_SUM = sum(n for _, n, _ in MODULES)


def path_json(p):
    return [{"kind": "lift"} if s is None else {"kind": "sibling", "digest": s.hex()} for s in p]


def load_safetensor(path: Path, name: str) -> bytes:
    """Raw little-endian bytes of one tensor from a .safetensors file (header parse; no torch)."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        lo, hi = meta["data_offsets"]
        f.seek(8 + n + lo)
        return f.read(hi - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("safetensors")
    ap.add_argument("--B", type=int, default=8); ap.add_argument("--out", required=True)
    ap.add_argument("--rho", required=True, help="challenge-v1 verifier randomness (64 hex chars = 32 raw bytes)")
    a = ap.parse_args()
    d = Path(a.run_dir)
    rj = json.loads((d / "roots.json").read_text())
    pd = bytes.fromhex(rj["program_digest"]); geo = bytes.fromhex(rj["geo_digest"]); run_id = rj["run_id"]
    steps = rj["steps"]
    L = 30
    tokens = [next(m for m in s["layout"] if m[1] == "model.layers.0.self_attn.qkv_proj/0")[3][0] for s in steps]
    per_step = [t * L * N_SUM for t in tokens]
    population = sum(per_step)

    def decode(idx):
        s = 0
        while idx >= per_step[s]:
            idx -= per_step[s]; s += 1
        t, rem = divmod(idx, L * N_SUM)
        layer, jj = divmod(rem, N_SUM)
        for mod, n, xmod in MODULES:
            if jj < n:
                return {"step": s, "layer": layer, "module": f"model.layers.{layer}.{mod}", "x_module": f"model.layers.{layer}.{xmod}", "token": t, "j": jj, "N": n}
            jj -= n

    t_c = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # challenge-v1 seed (docs/protocol/challenge-v1.md) from --rho; the o_proj rejection sampling below is a LABELLED review restriction
    from verity_protocol import challenge as chal_v1

    rho = bytes.fromhex(a.rho)
    instance = chal_v1.l1_instance(pd, bytes.fromhex(QUERY_ID), population, a.B, bytes.fromhex(rj["weights_root"]), bytes.fromhex(rj["run_root"]))
    c = chal_v1.challenge_seed(rho, instance)
    picks, coords, k = [], [], 0
    while len(coords) < a.B:                                   # rejection sampling to o_proj (labelled; NOT the protocol sample)
        idx = chal_v1.derive_indices(c, population, k + 1)[k]
        co = decode(idx); k += 1
        if co["module"].endswith("o_proj"):
            picks.append(idx); coords.append(co)
    # trees
    stream_cache, level_cache, w_cache = {}, {}, {}

    def stream(s):
        if s not in stream_cache:
            stream_cache[s] = (d / "steps" / f"step_{s}.bin").read_bytes()
            level_cache[s] = rv.fold_levels(rv.l1_leaves(stream_cache[s], s))
            assert level_cache[s][-1][0].hex() == steps[s]["merkle_root"], "stream does not reproduce the committed merkle root"
        return stream_cache[s], level_cache[s]

    wt_by_name = {t["name"]: t for t in rj["weights"]["tensors"]}

    def weight(name):
        if name not in w_cache:
            raw = load_safetensor(Path(a.safetensors), name)
            lv = rv.fold_levels(rv.weight_subtree_leaves(raw))
            w_cache[name] = (raw, lv, lv[-1][0].hex() == wt_by_name[name]["subtree_root"])
        return w_cache[name]

    trees, obls = {}, []
    for kk, co in enumerate(coords):
        s = co["step"]; sd = steps[s]
        lay = {m[1]: m for m in sd["layout"]}
        xt = lay.get(co["x_module"] + "/0") or lay[co["x_module"]]
        yt = lay.get(co["module"] + "/0") or lay[co["module"]]
        sb, lv = stream(s)
        x_off = xt[8] + 2 * co["token"] * K
        y_off = yt[8] + 2 * (co["token"] * co["N"] + co["j"])
        wname = co["module"] + ".weight"
        wraw, wlv, w_match = weight(wname)
        w_off = 2 * co["j"] * K
        x_words = struct.unpack("<%dH" % K, sb[x_off:x_off + 2 * K]); y_word = struct.unpack("<H", sb[y_off:y_off + 2])[0]
        tkey = f"step:{s}"
        tree = trees.setdefault(tkey, {"id": tkey, "framing": f"cmt-chunk-fa2h-v7.s{s}", "root": sd["merkle_root"], "count": sd["n_leaves"], "owner_plus_two": 3,
                                        "domain_id": sd["layout_digest"], "bound_root": sd["root"], "leaves": {}})
        wkey = f"weights:{wname}"
        wtree = trees.setdefault(wkey, {"id": wkey, "framing": "cmt-posleaf-v0", "root": wt_by_name[wname]["subtree_root"], "count": wt_by_name[wname]["n_leaves"],
                                         "owner_plus_two": 2, "domain_id": hashlib.sha256(b"verity/r10/weights-tensor/v1" + wname.encode()).hexdigest(),
                                         "subtree_root_recomputed_from_safetensors_equal": w_match, "leaves": {}})
        x_ranks = sorted({(x_off + 2 * i) // CH for i in range(K)}); y_rank = y_off // CH
        w_ranks = sorted({(w_off + 2 * i) // CH for i in range(K)})
        for r in sorted(set(x_ranks) | {y_rank}):
            tree["leaves"].setdefault(r, {"rank": r, "bytes_hex": sb[r * CH:(r + 1) * CH].hex(), "path": path_json(rv.path_in_levels(lv, r))})
        for r in w_ranks:
            wtree["leaves"].setdefault(r, {"rank": r, "bytes_hex": wraw[r * CH:(r + 1) * CH].hex(), "path": path_json(rv.path_in_levels(wlv, r))})
        obls.append({"k": kk, "logical": co,
                     "x": {"tree": tkey, "tensor": xt[1], "shape": xt[3], "stream_off": xt[8], "flat_word0": co["token"] * K, "ranks": x_ranks, "word_index0": (x_off % CH) // 2},
                     "W": {"tree": wkey, "tensor": wname, "shape": wt_by_name[wname]["shape"], "flat_word0": co["j"] * K, "ranks": w_ranks, "word_index0": (w_off % CH) // 2},
                     "y": {"tree": tkey, "tensor": yt[1], "shape": yt[3], "stream_off": yt[8], "flat_word": co["token"] * co["N"] + co["j"], "rank": y_rank,
                           "word_index": (y_off % CH) // 2, "y_word": int(y_word)},
                     "expected_y": int(y_word)})
    for t in trees.values():
        t["leaves"] = [t["leaves"][r] for r in sorted(t["leaves"])]
    side = {"schema": "veritor/r10/e2e-witness/v1", "layout_id": "L1-chunk256", "built_by": "owner 7 review_build_sidecar.py (o_proj-only rejection sample; NOT the protocol sample)",
            "context": {"program_digest": rj["program_digest"], "query_id": QUERY_ID, "run_id": run_id, "geo_digest": rj["geo_digest"], "committer": rj["committer"]},
            "roots": {"run_root": rj["run_root"], "weights_root": rj["weights_root"], "roots_written_utc": rj["written_utc"], "roots_file": "roots.json"},
            "challenge": {"rule": "challenge-v1 seed + NON-PROTOCOL o_proj rejection sampling (review)", "spec": "docs/protocol/challenge-v1.md",
                          "rho_hex": rho.hex(), "seed_hex": c.hex(), "instance": chal_v1.instance_to_json(instance), "B": a.B, "derived_utc": t_c,
                          "roots_written_utc": rj["written_utc"], "population": population, "indices": None, "picked_indices": picks, "coordinates": coords,
                          "selection": "review: derive_indices(seed) restricted by rejection to o_proj coordinates (labelled; not the protocol sample)"},
            "relation": {"id": "R-GEMM-COORD-K576", "K": K},
            "obligations": obls, "trees": list(trees.values()),
            "weights_subtree_roots_recomputed_from_safetensors": {n: v[2] for n, v in w_cache.items()},
            "guest": {"statement": None, "witness": None, "mode": None, "approved": None}}
    Path(a.out).write_text(json.dumps(side, indent=1) + "\n")
    print(json.dumps({"B": a.B, "coords": coords[:3], "trees": len(trees), "leaves": sum(len(t["leaves"]) for t in trees.values()),
                      "weights_subtree_roots_recomputed_equal": side["weights_subtree_roots_recomputed_from_safetensors"], "out": a.out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
