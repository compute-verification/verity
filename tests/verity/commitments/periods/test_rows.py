"""`rows`' bytes on a small case, pinned: `commit_ports`' roots and ``b || c`` strings over every port kind (rows of each
schema, given strings, words) under salts handed in as data, on every chunking and on the parallel path, so a change to the
frame or the hiding leaves that moves a byte fails here."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
from verity.commitments.periods import rows as SR

#: each port's root, and the SHA-512 of its rows' ``b || c`` strings
ROOTS = {
    "a": "e88758156d36d64a027199a5251f60d3861c8d32e4e63880b3edd6f7101c7b8b00cc12b6ccaccf04d2a703a9e8e4a4aacbbe7b9dfa3390c6b924433e762793d3",
    "b": "7b1b3ffc0b39ec6c5a366e493676bdddef2919d9c0c1c63c8c9ba22fca5770ba96c4df9738e0950e35e6249e25b70c48a0ea0daa5794ae747d58b8c044a5adf6",
    "c": "a0a913444728963213793bef5c9a36b88269a31209685ee0c7f11021a85152b84ffb4ee6c6953ad3b95b983aaa1656116c2e0bb077be1584f064cbb6c431b260",
    "d": "4015eafa82b4334898766bb7ab6b018ca762dc0b779d1b82735dc3761b4eb9d6f35c192d9b1d29b1c42bb7089e641fb67705a0bf5cfb3f32aab803944e926589",
    "s": "e3b1e26d5e37f90967416bd7b4ad530b9d688e95eb28a53caa2297251402a50c1b30b87f5baab0bf80ddcf247e1ea3ecba9cff1db7bec1646b855c077aed4669",
    "w": "978fb55b38a4427145253f0687be0f18efc790cfb4615667c4cec6d750f296acb3df4dae66ac67a30e08475cb28f9e7d93a2fc93a873c498622106159c38faa0",
}
BCS = {
    "a": "7dbc344c8da25e3ccb22c81bf253f0a1140ab7d57bf5a9d5efa46d4a3a9cf5f933a220d3f0de2024b327b905c6f254010e0e44698c797555f2c34a1b8f887901",
    "b": "ba37cd912f550a1dee3c6bb630fd80b3c2fe803c32cf4e71c6170f218196653d3b162fa4ea05171991ccdd982e4c9f25d5d3e4cb1f92bb035aa9929fa46dfaba",
    "c": "2c9cf956ce6e1931d135ea18a72a5faedcf1f5c3364724563016f8b4a53445c85450472b9810083ca0023270b2e8fa092bb49afc472e86499c27bfeffa00efb3",
    "d": "a4339777faf773dc83b19600e6c522e5ccd9cc51c5335f097b4df267319a51102cd7cbf606fd0b209d2b4c1182bb467954b910ce9f0cdd85d47726f0219a9438",
    "s": "7df050dde248c210c5ae71bbf3d53f13161abcda23808d0244ea04ba0f8b0c526bf36638dd80b19d92598fd0645439d9effcfd4b8b189b7311151f486c6ed357",
}
COUNTS = {"a": 1, "b": 5, "c": 8, "d": 13, "w": 7, "s": 6}
SCHEMAS = {"c": SCHEMA_HM96_SHA512_ROW_V2, "d": SCHEMA_HM96_SHA512_ROW_SEG}


def _sha512(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


def _ports() -> tuple[dict, dict, dict, dict, dict]:
    rng = np.random.default_rng(5)
    doms = {p: CommitmentDomain(hashlib.sha512(p.encode()).digest(), -1, RangeIndexedDomain(0, n), "sha512") for p, n in COUNTS.items()}
    rows = {"a": rng.integers(0, 1 << 16, (1, 64), dtype=np.uint16), "b": rng.integers(0, 1 << 16, (5, 64), dtype=np.uint16),
            "c": rng.integers(0, 1 << 16, (8, 40), dtype=np.uint16), "d": rng.integers(0, 1 << 16, (13, 600), dtype=np.uint16)}
    ys = {p: hashlib.shake_256(f"salts {p}".encode()).digest(COUNTS[p] * SR.SALT_BYTES) for p in rows}
    words = {"w": rng.integers(0, 1 << 16, 7, dtype=np.uint16)}
    strings = {"s": bytes(rng.integers(0, 256, 6 * SR.BC_BYTES, dtype=np.uint8))}
    return doms, rows, ys, words, strings


@pytest.mark.parametrize("chunk,workers", [(1, 1), (2, 1), (2, 2), (4, 1), (SR.CHUNK, 1)])
def test_commit_ports_is_pinned(chunk: int, workers: int) -> None:
    doms, rows, ys, words, strings = _ports()
    roots, bc = SR.commit_ports({p: doms[p] for p in (*rows, *words)}, rows, words, ys, workers=workers, chunk=chunk,
                                schemas=SCHEMAS)
    more, bc_s = SR.commit_ports({"s": doms["s"]}, {}, {}, {}, chunk=chunk, strings=strings,
                                 schemas={"s": SCHEMA_HM96_SHA512_ROW_V2})
    assert {p: r.hex() for p, r in {**roots, **more}.items()} == ROOTS
    assert {p: _sha512(b) for p, b in {**bc, **bc_s}.items()} == BCS
