"""The first recursive session's inner side (hiding model architectures, lane rec-v0; ``80-rec-inner.sh``): the inner
statement staged as M0's ``verity/flock-circuit`` over RoPE d64, proved by the unmodified prover against
``verity_flock.rec_live``'s proxy and coin server.

    python rec_inner.py stage --n N --sets DIR --out DIR      circuit.txt, inst-N.bin, pub-N.bin; prints the statement's shape
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verity_flock import circuit as FC

SEED = 20261004


def stage(n: int, sets: Path, out: Path) -> dict:
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    s = InputSet.open(generate(TM.rope_head(64), n, SEED, sets)["path"])
    low = FC.lowering_for_set(s)
    FC.stage(low, s, 0, n, out)
    text = (out / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    G = meta["vus_per_block"]
    blocks = -(-n // G)
    return {"subcircuit": s.subcircuit.id, "unit": low.name, "n": n, "vus_per_block": G, "k_log": meta["k_log"],
            "units_per_vu": meta["units_per_vu"], "unit_log": meta["unit_log"], "unit_rows": low.layout.useful,
            "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "blocks": blocks,
            "m": meta["k_log"] + max(0, (blocks - 1).bit_length()), "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    st = sp.add_parser("stage")
    st.add_argument("--n", type=int, required=True)
    st.add_argument("--sets", required=True)
    st.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = stage(a.n, Path(a.sets), Path(a.out))
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
