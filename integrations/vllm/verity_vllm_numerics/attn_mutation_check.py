"""Mutation tests for the FA2 model (CONTRACT §11 / Table 2 "mutations" column), CPU only.

Reads ``<stress>/mutations.json`` written by ``attn_stress.py`` (one-bit flips of one K / V / Q word
and the production output on the mutated input) and, for each mutation, answers two questions:

  rejected     would a verifier holding the *original* authenticated operands reject the mutated
               output?  = model(original) != production(mutated).  A flip that leaves the production
               output bit-identical (a low bit of a key nobody attends to) is *not* a fault in the
               output and cannot be detected by any output check; it is counted separately as
               ``no_output_change`` and never as a pass.
  tracks       does the model on the *mutated* operands reproduce the production output on the
               mutated operands bit-for-bit?  (the model follows the input, it is not a lookup)

    PYTHONPATH=. python -m verity_vllm_numerics.attn_mutation_check STRESS_DIR OUT.json [--tables DIR] [--natural DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import fa2_model as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stress")
    ap.add_argument("out")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--natural", default=None, help="natural fixture dir for the mutations whose base_fixture points there")
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()
    sx = Path(args.stress)
    idx = json.loads((sx / "index.json").read_text())
    consts = dict(idx["consts"])
    consts.setdefault("gpu", idx.get("gpu"))
    H, KVH, D = consts["num_heads"], consts["num_kv_heads"], consts["head_size"]
    scale = float(consts["softmax_scale"])
    tables = M.MufuTables.load(args.tables or (sx / "mufu"))
    muts = json.loads((sx / "mutations.json").read_text())
    by_seq = {c["seq"]: c for c in idx["calls"]}
    nat_idx = json.loads((Path(args.natural) / "index.json").read_text()) if args.natural else None
    nat_by_seq = {c["seq"]: c for c in nat_idx["calls"]} if nat_idx else {}

    res = []
    counts = {"attempted": 0, "rejected": 0, "no_output_change": 0, "undetected_with_output_change": 0, "tracks_exact": 0, "tracks_mismatch": 0, "unresolved": 0}
    cache: dict[str, np.ndarray] = {}

    def load(base: Path, f: str) -> np.ndarray:
        key = str(base / f)
        if key not in cache:
            cache[key] = np.load(base / f)
        return cache[key]

    for m in muts:
        if m.get("base_fixture"):
            if not args.natural:
                counts["unresolved"] += 1
                res.append({**m, "status": "skipped: natural fixture not given"})
                continue
            base = Path(args.natural)
            c = nat_by_seq[m["seq"]]
        else:
            base = sx
            c = by_seq[m["seq"]]
        Lq, Lk = c["Lq"], c["Lk"]
        rows = np.asarray(m["rows"], np.int64)
        q_saved = load(base, c["q_file"])  # (n_saved_rows, H, D) -- the saved rows (all rows for stress cases)
        k = load(base, c["k_file"])[:Lk]
        v = load(base, c["v_file"])[:Lk]
        reason = M.coverage(Lq, Lk, H, KVH, D, consts, arch=args.arch)
        if reason:
            counts["unresolved"] += 1
            res.append({**m, "status": f"unresolved: {reason}"})
            continue
        # full (Lq, H, D) query with the saved rows in place (rows not saved are zero and never compared)
        q = np.zeros((Lq, H, D), np.uint16)
        saved_rows = np.asarray(c["rows"], np.int64)
        q[saved_rows] = q_saved
        prod_mut = np.load(sx / m["production_out_mutated_file"])  # (len(rows), H, D)
        model_orig = M.attention_call(q[rows], k, v, rows, Lq, Lk, scale, tables).out
        which, idx_ = m["operand"], tuple(m["index"])
        qm, km, vm = q.copy(), k.copy(), v.copy()
        {"q": qm, "k": km, "v": vm}[which][idx_] = m["new_word"]
        rm = M.attention_call(qm[rows], km, vm, rows, Lq, Lk, scale, tables)
        model_mut = rm.out
        prod_changed = int(m["production_output_words_changed"])
        rejected = bool((model_orig != prod_mut).any())
        mut_nonfinite = M.nonfinite_words(qm[rows], km, vm) > 0 or bool(rm.nonfinite.any())
        tracks = bool((model_mut == prod_mut).all())
        counts["attempted"] += 1
        if prod_changed == 0:
            counts["no_output_change"] += 1
        elif rejected:
            counts["rejected"] += 1
        else:
            counts["undetected_with_output_change"] += 1
        if mut_nonfinite:
            counts["tracks_unresolved_nonfinite"] = counts.get("tracks_unresolved_nonfinite", 0) + 1
            tracks = None  # the model declines the mutated operands (inf/NaN): neither exact nor mismatch
        else:
            counts["tracks_exact" if tracks else "tracks_mismatch"] += 1
        res.append({**m, "status": "ok", "rejected": rejected, "model_tracks_mutation": tracks, "mutated_operands_nonfinite": mut_nonfinite,
                    "model_vs_production_mutated_mismatch_words": int((model_mut != prod_mut).sum()),
                    "model_orig_vs_production_mutated_diff_words": int((model_orig != prod_mut).sum())})
    by_operand = {}
    for r in res:
        if r.get("status") != "ok":
            continue
        b = by_operand.setdefault(r["operand"], {"attempted": 0, "rejected": 0, "no_output_change": 0, "tracks_exact": 0})
        b["attempted"] += 1
        b["rejected"] += int(r["rejected"] and r["production_output_words_changed"] > 0)
        b["no_output_change"] += int(r["production_output_words_changed"] == 0)
        b["tracks_exact"] += int(bool(r["model_tracks_mutation"]))
        b["tracks_unresolved_nonfinite"] = b.get("tracks_unresolved_nonfinite", 0) + int(r["model_tracks_mutation"] is None)
    out = {"stress_fixture": str(sx), "natural_fixture": args.natural, "counts": counts, "by_operand": by_operand, "mutations": res}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"counts": counts, "by_operand": by_operand}, indent=1))


if __name__ == "__main__":
    main()
