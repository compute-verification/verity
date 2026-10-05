"""ligerito-proto's ``verify_open`` on the commit-first PCS proof with sparse claims that ligerito-verify's reference prover
writes (``lgto::tests::pcs_sparse_proto_bytes``): Rust prover and Rust verifier against the Python verifier, on the sparse-claim
transcript (``b"sparse"`` absorbs, ``m + K`` batching coefficients, ``_Sparse``) that LGSC0004's Libra mask claims use.

``verify_open`` is numpy only; ``proto/pcs.py`` imports torch at module level without using it on this path, so a stub
stands in where torch is absent (the laptop).  From the root of a tree whose ``proto/pcs.py`` has ``open(..., sparse=)``
(lane/ligerito-relation-2 >= abd8f5e):

    PCS_SPARSE_PROTO_OUT=IN.json cargo test --release pcs_sparse_proto_bytes -- --ignored    # in archive/ligerito-verify
    PYTHONDONTWRITEBYTECODE=1 python <this file> IN.json [OUT.json]
"""
from __future__ import annotations

import json
import sys
from unittest import mock

try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = mock.MagicMock()

import numpy as np

from archive.direct.ligerito.proto.ext import P
from archive.direct.ligerito.proto.pcs import Proof, verify_open
from archive.direct.ligerito.proto.transcript import FiatShamirCoins


def coins(root: bytes) -> FiatShamirCoins:
    tr = FiatShamirCoins()
    tr.absorb(b"statement", b"relation statement")
    tr.absorb(b"root", root)
    return tr


def main(src: str, out: str | None) -> int:
    doc = json.load(open(src))
    pf = Proof.from_bytes(bytes.fromhex(doc["proof_hex"]))
    zs = np.asarray(doc["points"], dtype=np.int64)
    vals = np.asarray(doc["values"], dtype=np.int64)
    sp = [(np.asarray(t["cells"], dtype=np.int64), np.asarray(t["weights"], dtype=np.int64), np.asarray(t["value"], dtype=np.int64))
          for t in doc["sparse"]]
    (c0, w0, v0), t1 = sp[0], sp[1]
    cases = {
        "honest": (sp, vals, True),
        "value+1": ([(c0, w0, (v0 + 1) % P), t1], vals, False),
        "weights+1": ([(c0, (w0 + 1) % P, v0), t1], vals, False),
        "cells reversed": ([(c0[::-1].copy(), w0, v0), t1], vals, False),
        "term dropped": (sp[:1], vals, False),
        "no terms": ([], vals, False),
        "eval value+1": (sp, np.where(np.arange(vals.size).reshape(vals.shape) == 6, (vals + 1) % P, vals), False),
    }
    res, bad = {}, 0
    for name, (terms, v, want) in cases.items():
        ok, why = verify_open(pf, zs, v, coins(pf.roots[0]), sparse=terms)
        res[name] = {"accepted": bool(ok), "reason": why, "expected": want}
        bad += ok != want
        print(f"{name:16s} accepted={ok!s:5s} expected={want!s:5s} {why}")
    if out:
        json.dump({"input": src, "python_verify_open": res, "mismatches": bad}, open(out, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
