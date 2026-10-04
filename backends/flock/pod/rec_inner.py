"""The first recursive session's inner side (hiding model architectures, lane rec-v0; ``80-rec-inner.sh``): the inner
statement staged as M0's ``verity/flock-circuit`` over RoPE d64, proved by the unmodified prover against
``verity_flock.rec_live``'s proxy and coin server.

    python rec_inner.py stage --n N --sets DIR --out DIR      circuit.txt, inst-N.bin, pub-N.bin; prints the statement's shape
    python rec_inner.py forge --what challenge|query --inner DIR
                                                             flips one coin of rep 0 in DIR's session.json and public.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verity_flock import circuit as FC
from verity_flock import rec_vstar as V

SEED = 20261004
STREAM = "circuit/rep0"


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
            "m": meta["k_log"] + max(3, (blocks - 1).bit_length()), "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text)}


def forge(what: str, inner: Path) -> dict:
    """Flip the low bit of one coin of rep 0, in the session record and in the coin server's public record alike: the first
    coin after the binding round (``challenge``) or the first query coin of level 0 (``query``)."""
    pub, _, proofs = V.load(inner)
    if what == "challenge":
        k = next(r["k"] for r in V._stream_rounds(pub, STREAM) if r["k"] > 0 and r["n"])
    else:
        k = V.query_rounds(pub, STREAM, [len(lv.rows) for lv in V.decode(proofs[0])["levels"]])[0]["k"]
    sess = json.loads((inner / "session" / "session.json").read_text())
    for rd in (next(r for r in next(s for s in sess["streams"] if s["stream"] == STREAM)["rounds"] if r["k"] == k),
               next(r for r in pub["rounds"] if r["stream"] == STREAM and r["k"] == k)):
        c = rd["coins"][0]
        rd["coins"][0] = c[:-1] + format(int(c[-1], 16) ^ 1, "x")
    (inner / "session" / "session.json").write_text(json.dumps(sess))
    (inner / "public.json").write_text(json.dumps(pub))
    return {"forged": what, "stream": STREAM, "k": k, "coin": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    st = sp.add_parser("stage")
    st.add_argument("--n", type=int, required=True)
    st.add_argument("--sets", required=True)
    st.add_argument("--out", required=True)
    fg = sp.add_parser("forge")
    fg.add_argument("--what", choices=("challenge", "query"), required=True)
    fg.add_argument("--inner", required=True)
    a = ap.parse_args(argv)
    res = stage(a.n, Path(a.sets), Path(a.out)) if a.cmd == "stage" else forge(a.what, Path(a.inner))
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
