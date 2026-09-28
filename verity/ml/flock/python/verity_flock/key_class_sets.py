"""Synthetic input sets for a key-count class cell of ``attention-head``: for each T in [lo, hi], the first ``per_t`` admitted
draws of the spine generator at that T (``verity_numerical.bench.generate.generate(attention_head(D, BN, T="t:t"), per_t, seed)``),
merged in T order into one set whose provenance lists every per-T set's recipe and content digest.

    python -m verity_flock.key_class_sets --lo 1 --hi 128 --per-t 16 --seed 101 --out DIR
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def build(lo: int, hi: int, per_t: int, seed: int, out, D: int = 64, BN: int = 128, procs: int = 4) -> dict:
    from verity_numerical.bench import input_sets as IS
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate

    ports: dict[str, list] = {}
    rows, parts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for T in range(lo, hi + 1):
            sub = TM.attention_head(D, BN, T=f"{T}:{T}")
            r = generate(sub, per_t, seed, Path(tmp) / f"t{T}", procs=procs)
            s = IS.InputSet.open(r["path"])
            for k, v in s.instances().items():
                ports.setdefault(k, []).extend(list(v) if not isinstance(v, list) else v)
            for row in s.index()[:per_t] if isinstance(s.index(), list) else s.index()["rows"]:
                rows.append([len(rows), *row[1:]])
            parts.append({"T": T, "content_digest": s.content_digest, "recipe": s.manifest["provenance"]["recipe"]})
        sub = TM.attention_head(D, BN, T=f"{lo}:{hi}")
        prov = {k: None for k in IS.PROVENANCE_FIELDS}
        prov.update({"source": "synthetic", "selection": f"per T in [{lo}, {hi}]: the first {per_t} admitted draws at that T, in T order",
                     "recipe": {"generator": "verity_flock.key_class_sets", "lo": lo, "hi": hi, "per_t": per_t, "seed": seed,
                                "per_T": parts}})
        return IS.write(out, f"{sub.set_name}-t{lo}-{hi}", sub.relation, ports, rows, prov, schema=IS.SCHEMA) | {
            "path": str(Path(out) / f"{sub.set_name}-t{lo}-{hi}")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verity_flock.key_class_sets")
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--per-t", type=int, default=16)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.lo, a.hi, a.per_t, a.seed, a.out, procs=a.procs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
