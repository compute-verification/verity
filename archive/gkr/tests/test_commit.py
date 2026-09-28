"""The operands-committed statement's harness side (``archive/gkr/gpu/commit.py``): row digests are the core
frame-v3 reference's (``verity.ml.commitments.frame_v3/vectors.json``), the domains are B-Ligero's (``hashauth``: same
bindings, and the same roots for the BLAKE3 leaf it registers), and the statement extension adds exactly the 32
digest-limb columns to the epilogue and to the chain's public line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

GKR = Path(__file__).resolve().parents[1]
REPO = GKR.parents[1]
for p in (GKR, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpu import commit as CM  # noqa: E402

VECTORS = REPO / "verity" / "ml" / "commitments" / "frame_v3" / "vectors.json"
ISET = CM.InstanceSet("synthetic", "dev", "ab" * 32, 8, 16)


def _rows(seed: int, n: int, k: int, bits: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << bits, size=(n, k), dtype=np.int64)


def test_row_digests_are_the_core_vectors():
    v = json.loads(VECTORS.read_text())
    for c in v["row_leaves"]:
        raw = bytes.fromhex(c["row_bytes"])
        step = c["word_bits"] // 8
        words = np.array([[int.from_bytes(raw[i:i + step], "little") for i in range(0, len(raw), step)]])
        assert CM.row_digests(words, c["word_bits"], c["role"], c["hash"])[0].hex() == c["digest"], (c["hash"], c["n_words"])


def test_limbs_round_trip():
    d = bytes(range(32))
    assert CM.limbs(d) == [int.from_bytes(d[2 * i:2 * i + 2], "big") for i in range(16)]
    assert b"".join(x.to_bytes(2, "big") for x in CM.limbs(d)) == d


def test_bindings_are_hashauths():
    hashauth = pytest.importorskip("archive.direct.ligero.hashauth")
    for tree, schema in (("a", "sha256/row/v1"), ("b", "blake3-keyed/row/v2"), ("y", "u16")):
        assert CM.binding(ISET, 0, 5, tree, schema) == hashauth.binding_digest(
            dataset="synthetic", tier="dev", manifest_sha256="ab" * 32, lo=0, hi=5, K=8, tree=tree, schema=schema)


def test_blake3_roots_are_b_ligeros():
    """B-Ligero's BLAKE3 leaf frames rows of more than one 1024-byte chunk only: the bench rows (K = 1536)."""
    hashauth = pytest.importorskip("archive.direct.ligero.hashauth")
    iset = CM.InstanceSet("synthetic", "dev", "ab" * 32, 1536, 16)
    x, w = _rows(1, 5, 1536, 16), _rows(2, 5, 1536, 16)
    y = _rows(3, 5, 1, 16)
    c = CM.commit("blake3", "bf16-hopper", iset, 0, x, w, y, 2)
    bindings = {n: CM.binding(iset, 0, 5, n, "blake3-keyed/row/v2" if n != "y" else "u16") for n in "aby"}
    trees, dig = hashauth.build_trees(x_rows=x, w_cols=w, y_words=y.reshape(-1), word_bits=16, y_bits=16, bindings=bindings, leaf="blake3")
    for n in "aby":
        assert trees[n].ref.root == c.roots[n], n
    assert c.relation == "bf16-hopper+blake3"


def test_commitment_text_and_public_words():
    x, w, y = _rows(4, 3, 8, 16), _rows(5, 3, 8, 16), _rows(6, 3, 2, 16)
    c = CM.commit("sha256", "bf16-ampere", ISET, 0, x, w, y, 2)
    t = c.text()
    assert t.splitlines()[:3] == ["scheme frame-v3", "leaf sha256", "relation bf16-ampere+sha256"]
    assert "y u16 2" in t and "K 8" in t and "x 16" in t
    pub = c.public(y)
    assert pub.shape == (3, 2 + 32)
    assert (pub[:, :2] == y).all()
    for v in range(3):
        assert b"".join(int(l).to_bytes(2, "big") for l in pub[v, 2:18]) == c.digests["a"][v]
        assert b"".join(int(l).to_bytes(2, "big") for l in pub[v, 18:]) == c.digests["b"][v]
    with pytest.raises(ValueError):
        CM.commit("sha256", "fp8-hopper", CM.InstanceSet("s", "t", "ab" * 32, 8, 8), 0, x, w, y, 2)   # 16-bit words in an 8-bit row


def test_changed_operand_changes_its_root_only():
    x, w, y = _rows(7, 4, 8, 16), _rows(8, 4, 8, 16), _rows(9, 4, 1, 16)
    c = CM.commit("sha256", "bf16-ampere", ISET, 0, x, w, y, 2)
    x2 = x.copy()
    x2[2, 3] ^= 1
    c2 = CM.commit("sha256", "bf16-ampere", ISET, 0, x2, w, y, 2)
    assert c2.roots["a"] != c.roots["a"] and c2.roots["b"] == c.roots["b"] and c2.roots["y"] == c.roots["y"]


def test_vllm_v1_commitment_is_the_reference_schemes():
    from verity.ml.commitments.vllm_v1 import VllmV1, domain_digest, pos_leaf

    x, w, y = _rows(10, 5, 8, 16), _rows(11, 5, 8, 16), _rows(12, 5, 1, 16)
    c = CM.commit("vllm-v1", "bf16-ampere", ISET, 0, x, w, y, 2)
    t = c.text().splitlines()
    assert t[:3] == ["scheme vllm-v1", "leaf pos", "relation bf16-ampere+vllm-v1"]
    doms = {p: CM.vllm_domain("bf16-ampere", ISET, 0, 5, 1, 2, p) for p in CM.PORTS}
    assert t[-3:] == [f"domain_{p} {domain_digest(doms[p]).hex()}" for p in CM.PORTS]
    ref = VllmV1(doms, {"a": lambda _i: 16, "b": lambda _i: 16, "y": lambda _i: 2})
    for n, rows in (("a", x), ("b", w)):
        leaves = [ref.tree_leaf(n, i, ref.leaf_layout(n, i).digest(CM.row_bytes(rows[i:i + 1], 16)[0].tobytes())) for i in range(5)]
        assert leaves == c.digests[n] == [pos_leaf(CM.row_bytes(rows[i:i + 1], 16)[0].tobytes()) for i in range(5)]
        assert ref.root(n, leaves) == c.roots[n], n
    assert ref.root("y", [pos_leaf(int(v).to_bytes(2, "little")) for v in y.reshape(-1)]) == c.roots["y"]
    assert len({domain_digest(d) for d in doms.values()}) == 3
    x2 = x.copy()
    x2[4, 0] ^= 1
    c2 = CM.commit("vllm-v1", "bf16-ampere", ISET, 0, x2, w, y, 2)
    assert c2.roots["a"] != c.roots["a"] and c2.roots["b"] == c.roots["b"] and c2.roots["y"] == c.roots["y"]
    assert CM.commit("vllm-v1", "bf16-ampere", ISET, 1, x, w, y, 2).roots["a"] != c.roots["a"]   # the range is in the domain


EPI = "ncols 3\nnwires 4\none 0\ncol 0 one\ncol 1 y16\ncol 2 t\nprod 3 1 1 1 1 0 1 2 1 0\nassert a 1 3 1 0\n"


def test_extend_epilogue_and_chain():
    e = CM.extend_epilogue(EPI)
    lines = e.splitlines()
    assert "ncols 35" in lines and "nwires 36" in lines
    assert [l for l in lines if l.startswith("col ")][-32:] == [f"col {3 + j} {n}" for j, n in enumerate(CM.DIGEST_COLS)]
    assert "prod 3 1 1 1 1 0 1 2 1 0" in lines
    with pytest.raises(ValueError):
        CM.extend_epilogue(e)
    assert CM.extend_chain("link - a b 0 1\n").splitlines()[-1] == "public y16 " + " ".join(CM.DIGEST_COLS)
    assert CM.extend_chain("link - a b 0 1\npublic s t f\n").splitlines() == ["link - a b 0 1", "public s t f " + " ".join(CM.DIGEST_COLS)]
    circuit = pytest.importorskip("gpu.circuit")
    c = circuit.parse_circuit(e)
    assert c.ncols == 35 and c.col_names[3:] == CM.DIGEST_COLS
