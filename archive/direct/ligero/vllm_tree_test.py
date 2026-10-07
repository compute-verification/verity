"""``vllm_tree`` and the vllm-v1 branch of ``hashauth`` without torch: the tree rule and every opening path against the
core ``vllm-v1`` vectors, the statement's multiproofs, and the PROTOCOL.md section 9 negatives (a wrong domain, a bare tree
root, siblings mixed across trees, a wrong position, a wrong value length)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from verity.ml.commitments import vllm_v1 as core

from . import hashauth, vllm_tree
from .leaf import vllm_v1 as lv

VECTORS = json.loads((Path(core.__file__).parent / "vectors.json").read_text())
H = bytes.fromhex


def _tree(leaves: list[bytes], ctx: bytes = b"\1" * 32) -> vllm_tree.VllmTree:
    dom = vllm_tree.domain("a", ctx, len(leaves), K=1536, word_bits=8, y_bytes=4)
    return vllm_tree.build("a", ctx, np.frombuffer(b"".join(leaves), dtype=np.uint8).reshape(-1, 32), dom)


def test_tree_roots_and_every_opening_path_match_the_core_vectors() -> None:
    for c in VECTORS["trees"]:
        n = c["leaf_count"]
        if n == 0:
            continue
        leaves = [H(x) for x in c["leaves"]]
        t = _tree(leaves)
        assert t.tree_root.hex() == c["root"]
        for i, path in enumerate(c["paths"]):
            sibs = [H(x) for x in path if x is not None]
            assert vllm_tree.multiproof(t, [i]) == sibs, (n, i)
            assert vllm_tree.fold_multiproof(n, {i: leaves[i]}, sibs).hex() == c["root"]


def test_openings_bind_step_roots_and_domain_digests() -> None:
    for o in VECTORS["openings"]:
        if o["kind"] != "step" or o["domain"].get("chunk") is not None:
            continue
        d = o["domain"]
        dom = core.StepDomain(H(d["program_digest"]), H(d["ctx_digest"]), H(d["geo_digest"]), H(d["layout_digest"]), d["leaf_count"])
        leaf = core.pos_leaf(H(o["value"]))
        sibs = [H(x) for x in o["path"] if x is not None]
        try:
            ok = dom.bind(vllm_tree.fold_multiproof(dom.leaf_count, {o["index"]: leaf}, sibs)).hex() == o["root"]
        except ValueError:
            ok = False
        assert ok == o["accept"], o
        if o["accept"]:
            assert core.domain_digest(dom).hex() == o["domain_digest"]
            assert o["leaf_prefix"] == lv.prefix(len(H(o["value"]))).hex()


def test_core_rejections_are_rejected() -> None:
    for r in VECTORS["rejections"]:
        sibs = [H(x) for x in r["path"] if x is not None]
        try:
            ok = vllm_tree.fold_multiproof(r["leaf_count"], {r["index"]: H(r["leaf"])}, sibs).hex() == r["root"]
        except ValueError:
            ok = False
        assert not ok, r["why"]


@pytest.mark.parametrize("n", (1, 2, 3, 5, 8, 13, 64, 100))
def test_multiproofs_of_any_rank_set_fold_to_the_root(n: int) -> None:
    rng = np.random.default_rng(n)
    leaves = [bytes(rng.integers(0, 256, 32, dtype=np.uint8)) for _ in range(n)]
    t = _tree(leaves)
    assert t.tree_root == core.fold(leaves)
    for _ in range(12):
        ranks = sorted(set(rng.integers(0, n, size=int(rng.integers(1, n + 1))).tolist()))
        sibs = vllm_tree.multiproof(t, ranks)
        assert vllm_tree.fold_multiproof(n, {r: leaves[r] for r in ranks}, sibs) == t.tree_root
        if sibs:
            with pytest.raises(ValueError):
                vllm_tree.fold_multiproof(n, {r: leaves[r] for r in ranks}, sibs[:-1])
            bad = list(sibs)
            bad[0] = bytes(32)
            assert vllm_tree.fold_multiproof(n, {r: leaves[r] for r in ranks}, bad) != t.tree_root
        with pytest.raises(ValueError):
            vllm_tree.fold_multiproof(n, {r: leaves[r] for r in ranks}, sibs + [bytes(32)])


# --- the statement: hashauth over vllm-v1 ports ----------------------------------------------------------------------

N, K, WB, YB = 6, 256, 8, 4
VSHAPE = (K, WB, YB)


@pytest.fixture(scope="module")
def committed():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 256, size=(N, K))
    w = rng.integers(0, 256, size=(N, K))
    y = rng.integers(0, 1 << 22, size=N)
    scheme = lv.VllmV1Leaf()
    bindings = {n: vllm_tree.ctx_digest(dataset="d", tier="t", manifest_sha256="m", lo=0, hi=N, tree=n) for n in "aby"}
    trees, dig = hashauth.build_trees(x_rows=x, w_cols=w, y_words=y, word_bits=WB, y_bits=22, bindings=bindings, leaf=scheme)
    return x, w, y, scheme, trees, dig


def _statement(committed, vus=(1, 2, 3)):
    x, w, y, scheme, trees, dig = committed
    auth = hashauth.HashAuth.for_vus(trees, vus, vus, vus)
    sibs = hashauth.open_trees(trees, auth)
    digests = np.concatenate([dig["a"][list(vus)], dig["b"][list(vus)]], 1)
    return auth, sibs, digests, y[list(vus)]


def _verify(scheme, auth, sibs, digests, y_words, vshape=VSHAPE):
    return hashauth.verify_hash_auth(auth, sibs, digests=digests, y_words=y_words, y_bits=22, leaf=scheme, vshape=vshape)


def test_roots_are_the_core_scheme_over_the_verifier_domains(committed) -> None:
    x, w, y, scheme, trees, dig = committed
    for n, vals in (("a", [bytes(r.astype(np.uint8)) for r in x]), ("b", [bytes(r.astype(np.uint8)) for r in w]),
                    ("y", [int(v).to_bytes(YB, "little") for v in y])):
        dom = vllm_tree.domain(n, trees[n].ref.binding, N, K=K, word_bits=WB, y_bytes=YB)
        cfg = core.VllmV1({n: dom}, value_nbytes={n: lambda _i, vals=vals: len(vals[0])})
        assert trees[n].root == cfg.root(n, [core.pos_leaf(v) for v in vals])
        assert trees[n].root != trees[n].tree_root


def test_honest_statement_verifies(committed) -> None:
    ok, why = _verify(committed[3], *_statement(committed))
    assert ok, why


def test_wrong_domain_is_rejected(committed) -> None:
    scheme = committed[3]
    auth, sibs, digests, yw = _statement(committed)
    other = vllm_tree.ctx_digest(dataset="d", tier="t", manifest_sha256="m", lo=0, hi=N + 1, tree="a")
    moved = dataclasses.replace(auth, trees=(dataclasses.replace(auth.trees[0], binding=other), *auth.trees[1:]))
    assert not _verify(scheme, moved, sibs, digests, yw)[0]
    # another program / geometry of the verifier's own (a K the relation does not have) is another domain as well
    assert not _verify(scheme, auth, sibs, digests, yw, vshape=(K + 64, WB, YB))[0]
    # and the statement digest names the domain: two statements over the same roots with different domains differ
    assert hashauth.digest_bytes(auth, scheme, VSHAPE) != hashauth.digest_bytes(auth, scheme, (K + 64, WB, YB))


def test_bare_tree_root_is_rejected(committed) -> None:
    scheme, trees = committed[3], committed[4]
    auth, sibs, digests, yw = _statement(committed)
    for i, n in enumerate("aby"):
        refs = list(auth.trees)
        refs[i] = dataclasses.replace(refs[i], root=trees[n].tree_root)
        assert not _verify(scheme, dataclasses.replace(auth, trees=tuple(refs)), sibs, digests, yw)[0], n


def test_siblings_mixed_across_trees_are_rejected(committed) -> None:
    scheme = committed[3]
    auth, sibs, digests, yw = _statement(committed)
    assert not _verify(scheme, auth, [sibs[1], sibs[0], sibs[2]], digests, yw)[0]
    mixed = [list(sibs[0]), sibs[1], sibs[2]]
    mixed[0][0] = sibs[1][0]
    assert not _verify(scheme, auth, mixed, digests, yw)[0]
    # an x row's digest opened in the W tree (a subtree of one tree reused in another)
    swapped = np.concatenate([digests[:, lv.DIGEST_ELEMS:], digests[:, :lv.DIGEST_ELEMS]], 1)
    assert not _verify(scheme, auth, sibs, swapped, yw)[0]


def test_wrong_position_is_rejected(committed) -> None:
    scheme, trees = committed[3], committed[4]
    auth, sibs, digests, yw = _statement(committed)
    # a VU claiming another x row: refused by the layout (positions are the verifier's)
    moved = dataclasses.replace(auth, x_index=(2, 2, 3))
    assert not _verify(scheme, moved, sibs, digests, yw)[0]
    # the right digests, opened as if they sat one position further on (their own honest paths at those ranks)
    shifted = hashauth.HashAuth.for_vus(trees, (2, 3, 4), (2, 3, 4), (2, 3, 4))
    assert not _verify(scheme, shifted, hashauth.open_trees(trees, shifted), digests, yw)[0]


def test_wrong_value_length_is_rejected(committed) -> None:
    x, w, y, scheme, trees, dig = committed
    auth, sibs, digests, yw = _statement(committed)
    # an x digest over the same bytes with one byte fewer (another length prefix)
    short = np.asarray(lv.VllmV1Leaf().native(x[[1, 2, 3], :K - 1], word_bits=WB, role=1))
    bad = digests.copy()
    bad[:, :lv.DIGEST_ELEMS] = short
    assert not _verify(scheme, auth, sibs, bad, yw)[0]
    # the output words committed as 2-byte values when the relation's are 4
    assert not _verify(scheme, auth, sibs, digests, yw, vshape=(K, WB, 2))[0]


def test_device_trees_equal_the_host_trees(committed, monkeypatch) -> None:
    from . import frame_gpu
    if not frame_gpu.available():
        pytest.skip("no GPU committer")
    x, w, y, scheme, trees, dig = committed
    bindings = {n: trees[n].ref.binding for n in "aby"}
    dev, ddig = hashauth.build_trees(x_rows=x, w_cols=w, y_words=y, word_bits=WB, y_bits=22, bindings=bindings, leaf=scheme)
    monkeypatch.setattr(frame_gpu, "ENABLED", False)
    host, hdig = hashauth.build_trees(x_rows=x, w_cols=w, y_words=y, word_bits=WB, y_bits=22, bindings=bindings, leaf=scheme)
    for n in "aby":
        assert dev[n].ref == host[n].ref and all(np.array_equal(a, b) for a, b in zip(dev[n].levels, host[n].levels))
    for n in "ab":
        assert np.array_equal(ddig[n], hdig[n])
