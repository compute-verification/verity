"""``authentication = included`` for B-Ligero (``archive/direct/ligero/auth.py``, ``serialize.py`` v3/v2 formats).

Torch-free: the word trees, the canonical multiproofs, the statement block and the verifier's authentication check are
numpy + ``verity.ml.commitments``.  The pinned digests below are the SAME constants ``archive/ligero-verify/src/auth.rs``
pins in its unit tests, so the two verifiers are checked against one set of numbers (a divergence shows up on the
laptop, without a Rust toolchain).  The proof-file round trip needs ``protocol.Proof`` and is skipped without torch.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verity.ml.commitments.multiproof import multiproof, verify_multiproof
from verity.ml.commitments import MerkleTree

from archive.direct.ligero import auth

BINDING = bytes(range(32))
WORDS = np.array([0x3F80, 0x0000, 0xBF80, 0x4000, 0x7F7F], dtype=np.uint16)
ROOT_A = "7dc53a78722230c27619305152c2f8008ae19cb2cb8418ebc670f9bb8379e7b8"
LEAVES_A = [
    "53c94bb782bf50fbb1b994d1a456cf6a5e29f4d1bed63050f60e05d6082d213b",
    "84232b121e122a7b9a194f56bc56b91d14724db5d4035cdcc5376405a72a58f4",
    "bad8e6963d4f80157eaa33bd844fefbd60e2b8f549c870ada02b4b38a8396c23",
    "d67978e6fdfa4242be4f5065b7bb2733ce7b55b9baefa6b962181f04e60795b9",
    "f51e8e31d728520f0a4b96a431bbc4c60078ed518132071f2016c1238ffa4b5f",
]


def _three_trees():
    ta = auth.build_word_tree("a", BINDING, WORDS, procs=1)
    tb = auth.build_word_tree("b", bytes([1]) * 32, WORDS, procs=1)
    ty = auth.build_word_tree("y", bytes([2]) * 32, WORDS[:2], procs=1)
    return ta, tb, ty


def test_pinned_vectors_shared_with_the_rust_verifier() -> None:
    ta, tb, ty = _three_trees()
    dom = auth.domain_for(BINDING, -1, 5)
    assert dom.positions.identity_digest == "cac30b672f5392ae0261a286d659ca9ca0d9c0f4a71f47dfbde289a3d2d75164"
    assert dom.domain_id.hex() == "152b53bdcd4ff2e59540a24fe076d904ad95d15146b4ceb0cacf5e034fa9cc67"
    assert ta.ref.root.hex() == ROOT_A
    assert [bytes(x).hex() for x in ta.levels[0][:5]] == LEAVES_A
    assert bytes(ta.levels[0][5]).hex() == "8be67fa2ecf6b0a6886509e55a8332fb8b08436e4fa910e5deba8c8595bb6ce8"  # pad leaf
    assert bytes(ta.levels[1][0]).hex() == "516dbaaf3502018e6e7d0f4591fb0f7e2ace0ffff2cc23be3ce8a7f76512a169"
    assert [s.hex() for s in auth.multiproof_siblings(ta, np.array([1, 3]))] == [
        LEAVES_A[0], LEAVES_A[2], "1ece3166a87f4bc8ae47a0da8d594bf2abaa456973a3940ffadb5a940d2eae99"]
    assert tb.ref.root.hex() == "5ae6c7331b39619769df51b123bb9c7a334b6f23e3c67f03a9de201241a58256"
    assert ty.ref.root.hex() == "d5dcb571025cec7c4ca779bf71aa8030eaf52aaad344ff299d6c3dab2a9b5562"
    st = auth.AuthStatement.for_vus({"a": ta, "b": tb, "y": ty}, [0, 1])
    d = auth.digest_bytes(st)
    assert len(d) == 340
    assert hashlib.sha256(d).hexdigest() == "e79b8500b819c06a4cbc6bbdfc6e595f4796690fef0974020cfdb120f0f26cef"
    # the digest binds every public field of the block
    for changed in (auth.AuthStatement((0, 2), st.trees),
                    auth.AuthStatement(st.vu_index, (auth.TreeRef("a", BINDING, -1, 5, bytes(32)), tb.ref, ty.ref)),
                    auth.AuthStatement(st.vu_index, (auth.TreeRef("a", bytes(32), -1, 5, ta.ref.root), tb.ref, ty.ref)),
                    auth.AuthStatement(st.vu_index, (auth.TreeRef("a", BINDING, -1, 6, ta.ref.root), tb.ref, ty.ref))):
        assert auth.digest_bytes(changed) != d


def test_word_tree_and_multiproofs_agree_with_verity_commitments() -> None:
    rng = np.random.default_rng(7)
    words = rng.integers(0, 1 << 16, size=333, dtype=np.uint16)
    tree = auth.build_word_tree("b", bytes([9]) * 32, words, procs=1)
    dom = auth.domain_for(bytes([9]) * 32, -2, 333)
    ref = MerkleTree(dom, {i: int(w).to_bytes(2, "big") for i, w in enumerate(words.tolist())}, lambda _p: auth.SCHEMA)
    assert ref.commitment.root == tree.ref.root
    patterns = [[0], [332], [5, 6], [0, 1, 2, 3], [1, 2], [7], [6, 7], [3, 4, 5, 6, 7, 8, 9, 10, 11],   # run ends of both parities
                [0, 1, 2, 3, 6, 7],          # two runs whose parents touch and merge
                [2, 3, 4, 5, 8, 9, 10, 11],  # runs separated by exactly one unknown pair
                [10, 12, 14, 16],            # singletons
                list(range(100, 150)) + list(range(151, 200)) + [300, 331, 332],
                sorted(rng.choice(333, 40, replace=False).tolist()), list(range(333))]
    for _ in range(20):
        patterns.append(sorted(rng.choice(333, int(rng.integers(1, 80)), replace=False).tolist()))
    for ranks in patterns:
        sibs = auth.multiproof_siblings(tree, np.array(ranks))
        assert sibs == multiproof(ref, ranks), ranks
        leaves = {r: bytes(tree.levels[0][r]) for r in ranks}
        assert verify_multiproof(tree.ref.root, 333, dom.domain_id, leaves, sibs)
    assert auth.multiproof_siblings(tree, np.array([], dtype=np.int64)) == []
    with pytest.raises(ValueError):
        auth.multiproof_siblings(tree, np.array([333]))


def _instance(n_total_vus: int = 6, K: int = 32, vus=(1, 4)):
    """Tiny committed input set (K words per VU, steps = K / 16) and the honest statement words of ``vus``."""
    rng = np.random.default_rng(3)
    a_all = rng.integers(0, 1 << 16, size=n_total_vus * K, dtype=np.uint16)
    b_all = rng.integers(0, 1 << 16, size=n_total_vus * K, dtype=np.uint16)
    y_all = rng.integers(0, 1 << 16, size=n_total_vus, dtype=np.uint16)
    trees = {"a": auth.build_word_tree("a", bytes([1]) * 32, a_all, procs=1),
             "b": auth.build_word_tree("b", bytes([2]) * 32, b_all, procs=1),
             "y": auth.build_word_tree("y", bytes([3]) * 32, y_all, procs=1)}
    st = auth.AuthStatement.for_vus(trees, vus)
    steps = K // 16
    a = np.concatenate([a_all[v * K:(v + 1) * K] for v in vus]).reshape(-1, 16)  # column-major (n_cols, 16)
    b = np.concatenate([b_all[v * K:(v + 1) * K] for v in vus]).reshape(-1, 16)
    y = y_all[list(vus)]
    sibs = auth.open_trees(trees, st, K=K, steps=steps)
    return trees, st, sibs, a, b, y, K, steps


def test_verify_auth_accepts_honest_and_rejects_the_three_negatives() -> None:
    _trees, st, sibs, a, b, y, K, steps = _instance()
    ok, why = auth.verify_auth(st, sibs, a_words=a, b_words=b, y_words=y, K=K, steps=steps, procs=1)
    assert ok, why
    # (1) a word that is not under the root: swap one operand word
    a2 = a.copy()
    a2[3, 5] ^= 0x0001
    ok, why = auth.verify_auth(st, sibs, a_words=a2, b_words=b, y_words=y, K=K, steps=steps, procs=1)
    assert not ok and why == "auth: a: multiproof rejected"
    b2 = b.copy()
    b2[0, 0], b2[0, 1] = b2[0, 1], b2[0, 0]
    ok, why = auth.verify_auth(st, sibs, a_words=a, b_words=b2, y_words=y, K=K, steps=steps, procs=1)
    assert not ok and why == "auth: b: multiproof rejected"
    y2 = y.copy()
    y2[1] ^= 0x8000
    ok, why = auth.verify_auth(st, sibs, a_words=a, b_words=b, y_words=y2, K=K, steps=steps, procs=1)
    assert not ok and why == "auth: y: multiproof rejected"
    # (2) the right words claimed at the wrong index
    wrong_idx = auth.AuthStatement((1, 5), st.trees)
    ok, why = auth.verify_auth(wrong_idx, sibs, a_words=a, b_words=b, y_words=y, K=K, steps=steps, procs=1)
    assert not ok and why.startswith("auth: a: multiproof rejected")
    out_of_range = auth.AuthStatement((1, 6), st.trees)
    ok, why = auth.verify_auth(out_of_range, sibs, a_words=a, b_words=b, y_words=y, K=K, steps=steps, procs=1)
    assert not ok and why == "auth: a: leaf index out of range"
    # (3) the right words under a different root
    other_root = bytes(x ^ 1 for x in st.trees[1].root)
    diff = auth.AuthStatement(st.vu_index, (st.trees[0], auth.TreeRef("b", st.trees[1].binding, -2, st.trees[1].count,
                                                                     other_root), st.trees[2]))
    ok, why = auth.verify_auth(diff, sibs, a_words=a, b_words=b, y_words=y, K=K, steps=steps, procs=1)
    assert not ok and why == "auth: b: multiproof rejected"
    diff_binding = auth.AuthStatement(st.vu_index, (auth.TreeRef("a", bytes(32), -1, st.trees[0].count, st.trees[0].root),
                                                    st.trees[1], st.trees[2]))
    ok, _ = auth.verify_auth(diff_binding, sibs, a_words=a, b_words=b, y_words=y, K=K, steps=steps, procs=1)
    assert not ok
    # transcript-side: truncated / extended / swapped multiproofs
    for mutated in ([sibs[0][:-1], sibs[1], sibs[2]], [sibs[0], sibs[1] + [sibs[1][-1]], sibs[2]],
                    [sibs[1], sibs[0], sibs[2]], [sibs[0], sibs[1]]):
        ok, _ = auth.verify_auth(st, mutated, a_words=a, b_words=b, y_words=y, K=K, steps=steps, procs=1)
        assert not ok
    # the block itself refuses non-increasing indices and misordered trees
    with pytest.raises(ValueError):
        auth.AuthStatement((4, 1), st.trees)
    with pytest.raises(ValueError):
        auth.AuthStatement((1, 1), st.trees)
    with pytest.raises(ValueError):
        auth.AuthStatement((1, 4), (st.trees[1], st.trees[0], st.trees[2]))


def test_positions_are_column_major_and_contiguous_per_vu() -> None:
    _, st, _, _, _, _, K, steps = _instance()
    pos = auth.auth_positions(st, K=K, steps=steps)
    assert pos["a"].tolist() == list(range(1 * K, 2 * K)) + list(range(4 * K, 5 * K))
    assert pos["b"].tolist() == pos["a"].tolist()
    assert pos["y"].tolist() == [1, 4]
    with pytest.raises(ValueError):
        auth.auth_positions(st, K=K, steps=steps + 1)


def test_tree_cache_round_trip(tmp_path: Path) -> None:
    trees, *_ = _instance()
    for t in trees.values():
        t.save(tmp_path)
    loaded, hit = auth.load_or_build_trees(tmp_path, a_words=np.zeros(6 * 32, np.uint16), b_words=np.zeros(6 * 32, np.uint16),
                                           y_words=np.zeros(6, np.uint16),
                                           bindings={n: trees[n].ref.binding for n in auth.TREE_NAMES}, procs=1)
    assert hit and all(loaded[n].ref == trees[n].ref for n in auth.TREE_NAMES)
    # a different binding misses the cache (the trees are rebuilt over the given words)
    rebuilt, hit = auth.load_or_build_trees(tmp_path, a_words=np.zeros(6 * 32, np.uint16), b_words=np.zeros(6 * 32, np.uint16),
                                            y_words=np.zeros(6, np.uint16),
                                            bindings={"a": bytes(32), "b": bytes(32), "y": bytes(32)}, procs=1)
    assert not hit and rebuilt["a"].ref.root != trees["a"].ref.root


def test_statement_v3_round_trip_and_v2_compatibility() -> None:
    from archive.direct.ligero import serialize as ser

    _, st_auth, _, a, b, y, _K, steps = _instance()
    l, n_vus = 8, 2
    a_full = np.zeros((l, 16), np.uint16)
    b_full = np.zeros((l, 16), np.uint16)
    y16 = np.zeros(l, np.uint16)
    a_full[:n_vus * steps] = a
    b_full[:n_vus * steps] = b
    y16[steps - 1] = y[0]
    y16[2 * steps - 1] = y[1]
    st = ser.Statement(l, 64, 6, 20, -128.0, False, 0, "fiat-shamir", 1, n_vus, steps, a_full, b_full, y16, st_auth)
    raw = ser.statement_bytes(st)
    assert raw[:8] == ser.STMT_MAGIC_V3
    back = ser.read_statement(raw)
    assert back.auth == st_auth and np.array_equal(back.a, a_full) and back.steps == steps
    # without the block the file is the v2 layout, byte for byte
    st2 = ser.Statement(l, 64, 6, 20, -128.0, False, 0, "fiat-shamir", 1, n_vus, steps, a_full, b_full, y16, None)
    raw2 = ser.statement_bytes(st2)
    assert raw2[:8] == ser.STMT_MAGIC and raw2[8:] == raw[8:8 + len(raw2) - 8] and ser.read_statement(raw2).auth is None
    # the block is what the statement digest absorbs, so a flipped root changes the file and the digest bytes
    flipped = ser.read_statement(raw[:-1] + bytes([raw[-1] ^ 1]))
    assert flipped.auth.trees[2].root != st_auth.trees[2].root
    assert auth.digest_bytes(flipped.auth) != auth.digest_bytes(st_auth)
    with pytest.raises(ValueError):
        ser.read_statement(raw + b"\0")


def test_proof_v2_round_trip_needs_torch() -> None:
    pytest.importorskip("torch")
    from archive.direct.ligero import serialize as ser
    from archive.direct.ligero.protocol import Proof

    _, _, sibs, *_ = _instance()
    rng = np.random.default_rng(1)
    D, l, t, M, depth = 2, 4, 3, 5, 3
    pf = Proof(bytes(32), rng.integers(0, 100, (D, l), dtype=np.uint32), rng.integers(0, 100, (D, l - 1), dtype=np.uint32),
               [0, 3, 5], rng.integers(0, 100, (M, t), dtype=np.uint32), [[bytes([i]) * 32 for i in range(depth)] for _ in range(t)],
               "blake3", {}, rng.integers(0, 100, (D, 2 * l - 1), dtype=np.uint32), None, None, None, [list(s) for s in sibs])
    raw = ser.proof_bytes(pf)
    assert raw[:8] == ser.PROOF_MAGIC_V2
    back = ser.read_proof(raw)
    assert back.auth == [list(s) for s in sibs]
    assert pf.n_bytes()["auth_multiproofs"] == 32 * sum(len(s) for s in sibs)
    pf.auth = None
    assert ser.proof_bytes(pf)[:8] == ser.PROOF_MAGIC and ser.read_proof(ser.proof_bytes(pf)).auth is None
