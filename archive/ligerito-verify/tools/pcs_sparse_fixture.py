"""Dump ligerito-proto commit-first PCS proofs with sparse claims (LGSC0004's Libra mask claims) for archive/ligerito-verify.

Needs the CUDA device path (``pcs.commit``) and a tree whose ``proto/pcs.py`` has ``open(..., sparse=)`` (lane/ligerito-relation-2
>= abd8f5e, lane/ligerito-pcs-fast).  From that tree's root on a GPU pod:

    python <this file> OUT.json

Shape of proto's ``fast_test.test_zk_padding_and_sparse_claims``: n = 16, Dims(16, [4, 3, 3], [24, 20, 16], 2), three evaluation
points, one sparse term inside round-1 column 3 (40 cells) and one spread over the vector (25 cells); t_pad 0 and 32.  Coins: the
relation's Fiat-Shamir prefix ``absorb("statement", b"relation statement")``, ``absorb("root", root_1)``, then the PCS's own.
Every case is verified by ``verify_open`` before it is written, with the negatives' Python verdicts recorded.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from archive.direct.ligerito.proto import ext
from archive.direct.ligerito.proto.dims import Dims
from archive.direct.ligerito.proto.ext import D, P
from archive.direct.ligerito.proto.pcs import Proof, commit, evaluate, open as pcs_open, verify_open
from archive.direct.ligerito.proto.transcript import FiatShamirCoins

RNG = np.random.default_rng(1729)


def coins(root: bytes) -> FiatShamirCoins:
    tr = FiatShamirCoins()
    tr.absorb(b"statement", b"relation statement")
    tr.absorb(b"root", root)
    return tr


def main(out: str) -> None:
    n = 16
    dims = Dims(n, [4, 3, 3], [24, 20, 16], 2)
    f = torch.as_tensor(RNG.integers(0, P, size=1 << n, dtype=np.int64).astype(np.int32), device="cuda")
    fv = f.to(torch.int64).cpu().numpy()
    zs = RNG.integers(0, P, size=(3, n, D), dtype=np.int64)
    k1 = dims.k[0]
    sp = []
    for cells in (3 * (1 << k1) + RNG.choice(1 << k1, size=40, replace=False), RNG.choice(1 << n, size=25, replace=False)):
        cells = cells.astype(np.int64)
        w = RNG.integers(0, P, size=(len(cells), D), dtype=np.int64)
        val = ext.sum_ext(ext.mul(w, fv[cells][:, None] * np.array([1, 0, 0, 0, 0, 0])), axis=0)
        sp.append((cells, w, val))
    cells, w, val = sp[0]
    negatives = {"value+1": [(cells, w, (val + 1) % P), sp[1]], "weights+1": [(cells, (w + 1) % P, val), sp[1]],
                 "cells reversed": [(cells[::-1].copy(), w, val), sp[1]], "term dropped": sp[:1], "no terms": []}
    cases = []
    for tp in (0, 32):
        com = commit(f, dims, t_pad=tp)
        values = evaluate(com, zs)
        raw = pcs_open(com, zs, values, coins(com.root), sparse=sp).to_bytes()
        pf = Proof.from_bytes(raw)
        ok, why = verify_open(pf, zs, values, coins(pf.roots[0]), t_pad=tp, sparse=sp)
        assert ok, why
        negs = {}
        for name, bad in negatives.items():
            nok, nwhy = verify_open(pf, zs, values, coins(pf.roots[0]), t_pad=tp, sparse=bad)
            assert not nok, name
            negs[name] = nwhy
        cases.append({"t_pad": tp, "proof_hex": raw.hex(), "values": np.asarray(values).tolist(), "python_negatives": negs})
    doc = {"what": "ligerito-proto commit-first PCS proofs (Proof.to_bytes) with sparse claims",
           "coins": "FiatShamirCoins: absorb(statement, b'relation statement'), absorb(root, root_1), then verify_open",
           "n": n, "kprime": dims.kprime, "queries": dims.queries, "rate_log2": 2,
           "points": zs.tolist(),
           "sparse": [{"cells": c.tolist(), "weights": ww.tolist(), "value": np.asarray(v).tolist()} for c, ww, v in sp],
           "negatives": {k: [{"cells": c.tolist(), "weights": ww.tolist(), "value": np.asarray(v).tolist()} for c, ww, v in bad]
                         for k, bad in negatives.items()},
           "cases": cases}
    with open(out, "w") as fh:
        json.dump(doc, fh)
    print(f"wrote {out}: {len(cases)} cases, {len(negatives)} negatives each")


if __name__ == "__main__":
    main(sys.argv[1])
