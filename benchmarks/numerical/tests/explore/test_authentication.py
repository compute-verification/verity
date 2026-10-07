"""The authentication-layer costing (``explore.authentication``): the K=1536 fixture's leaf/opening structure as the
module states it, and the option table's pinned numbers at B=4096."""

from __future__ import annotations

import json

import pytest
from verity.ml.commitments.leaves import DIGEST_BYTES, leaf_hash, path_shape, tree_depth
from verity_sp1.proofs.packed import decode_packed_witness
from verity_numerical.explore import authentication as auth

OPENINGS = auth.read_openings()
REL = auth.relation_side(4096, 1536)


def test_fixture_leaf_and_opening_structure():
    """3 073 references into 2 roots resolve to exactly three whole-row leaves: x (1536 words), W[j,:] (1536 words),
    and the output row (2048 words, one word read)."""
    assert OPENINGS.K == 1536 and OPENINGS.references == 3073 and len(OPENINGS.leaves) == 3
    assert OPENINGS.roots == {"ru/307c6016-75d6-547f-9dfc-24a350a873dd/000009": 368, "weights": 928325}
    x, w, y = OPENINGS.leaf("x"), OPENINGS.leaf("w"), OPENINGS.leaf("y")
    assert (x.value_bytes, w.value_bytes, y.value_bytes) == (3072, 3072, 4096)
    assert (x.words_read, w.words_read, y.words_read) == (1536, 1536, 1)
    assert x.word_bits == w.word_bits == y.word_bits == 16
    assert x.dtype == w.dtype == y.dtype == "bfloat16"
    assert (x.shape_text, w.shape_text, y.shape_text) == ("[1536]", "[1,1536]", "[2048]")
    assert x.tree == y.tree != w.tree and w.tree == "weights"
    assert (x.depth, w.depth, y.depth) == (9, 20, 9) and (x.lifts, w.lifts, y.lifts) == (1, 0, 1)
    assert tree_depth(368) == 9 and tree_depth(928325) == 20
    assert path_shape(368, x.rank) == path_shape(368, y.rank)
    assert w.canonical_id == "w=layers.5.self_attn.qkv_proj.weight/c=1014" and w.rank == 834107


def test_fixture_witness_hashes_to_the_recorded_leaves():
    """The recorded witness carries the three leaves' bytes; their leaf hashes fold to the object's roots under the
    reference scheme -- what any in-circuit option (a)/(c) must reproduce."""
    from verity.ml.commitments.leaves import fold_path

    doc = json.loads(auth.FIXTURE.read_text())
    roots = {r["id"]: r for r in doc["object"]["roots"]}
    leaves = {v["id"]: v for v in doc["object"]["leaves"]}
    witness = decode_packed_witness(bytes.fromhex(doc["witness"]))
    seen = 0
    for tree in witness.trees:
        for leaf in tree:
            entry = leaves[leaf.canonical_id]
            root = roots[entry["root"]]
            digest = leaf_hash(leaf.canonical_id, leaf.dtype, leaf.shape_text, leaf.value)
            assert fold_path(digest, entry["leaf"], leaf.path) == bytes.fromhex(root["root"])
            assert len(leaf.path) == tree_depth(root["count"])
            seen += 1
    assert seen == 3


def test_sha256_block_counts_from_the_framing():
    """Preimage = u32 tag length + tag + (u64 length + part) per part; node = 103 B and lift = 63 B are 2 blocks each."""
    x, w, y = OPENINGS.leaf("x"), OPENINGS.leaf("w"), OPENINGS.leaf("y")
    assert x.preimage_bytes == 4 + 19 + (8 + 116) + (8 + 8) + (8 + 6) + (8 + 3072) == 3257
    assert w.preimage_bytes == 4 + 19 + (8 + 43) + (8 + 8) + (8 + 8) + (8 + 3072) == 3186
    assert y.preimage_bytes == 4 + 19 + (8 + 116) + (8 + 8) + (8 + 6) + (8 + 4096) == 4281
    assert (x.leaf_compressions, w.leaf_compressions, y.leaf_compressions) == (52, 50, 68)
    assert auth.NODE_COMPRESSIONS == auth.sha256_compressions(4 + 19 + 2 * (8 + DIGEST_BYTES)) == 2
    assert auth.LIFT_COMPRESSIONS == auth.sha256_compressions(4 + 19 + 8 + DIGEST_BYTES) == 2
    assert (x.path_compressions, w.path_compressions, y.path_compressions) == (18, 40, 18)
    assert auth.instance_compressions(OPENINGS, auth.Sharing("none", 1, 1, 1)) == 246
    assert (x.public_prefix_compressions, w.public_prefix_compressions, y.public_prefix_compressions) == (2, 1, 2)
    assert auth.sha256_compressions(55) == 1 and auth.sha256_compressions(56) == 2


def test_sp1_measured_cycles_per_compression():
    """138 164 merkle.auth cycles (r20260921-224405-22dc) over the fixture's 246 compressions."""
    assert round(138_164 / 246) == 562


def test_relation_side_matches_the_protocol_documents():
    assert REL.mul_gates_batch == 34_370 * 4096 == 140_779_520
    assert round(REL.committed_a_batch / 1e6, 1) == 181.4
    assert round(REL.committed_b_batch / 1e6, 1) == 779.6


def test_hash_circuit_constants():
    assert auth.SHA256.mul_gates == 26_170 and auth.SHA256.committed_b == 78_510
    assert auth.SHA256.committed_a == (64 * 4 + 8 + 48) * 32 + 512 == 10_496
    assert auth.POSEIDON2_SBOXES == 8 * 12 + 22 == 118 and auth.POSEIDON2_GOLDILOCKS.mul_gates == 472
    assert auth.POSEIDON2_GOLDILOCKS.committed_a == 0 and auth.POSEIDON2_GOLDILOCKS.committed_b == 1416


@pytest.mark.parametrize(
    ("option", "sharing", "inv", "frac_a_committed", "frac_a_mul", "frac_b_committed"),
    [
        ("a", "none", 246.0, 58.32, 187.31, 101.47),
        ("c", "none", 170.0, 40.30, 129.44, 70.12),
        ("d", "none", 252.0, 0.0, 3.46, 1.87),
        ("a", "bench24", 90.9, 21.55, 69.22, 37.50),
        ("c", "bench24", 50.7, 12.02, 38.61, 20.91),
        ("d", "bench24", 85.0, 0.0, 1.17, 0.63),
        ("a", "one_row", 90.0, 21.34, 68.56, 37.14),
        ("c", "one_row", 50.0, 11.86, 38.09, 20.64),
        ("d", "one_row", 84.0, 0.0, 1.15, 0.63),
    ],
)
def test_option_table_pinned(option, sharing, inv, frac_a_committed, frac_a_mul, frac_b_committed):
    r = auth.cost_option(option, OPENINGS, auth.SHARING[sharing], 4096, REL)
    assert r.compatible_with_private_values
    assert round(r.hash_invocations_per_instance, 1) == inv
    assert round(r.fraction_committed_a, 2) == frac_a_committed
    assert round(r.fraction_mul_gates_a, 2) == frac_a_mul
    assert round(r.fraction_committed_b, 2) == frac_b_committed


def test_option_b_is_free_in_circuit_and_incompatible_with_private_values():
    r = auth.cost_option("b", OPENINGS, auth.SHARING["none"], 4096, REL)
    assert not r.compatible_with_private_values
    assert r.mul_gates_per_instance == 0 and r.committed_a_per_instance == 0 and r.committed_b_per_instance == 0
    assert r.verifier_hash_invocations_per_instance == 246
    # x + W + output row bytes plus 38 sibling digests
    assert r.communication_bytes_per_instance == 3072 + 3072 + 4096 + (9 + 20 + 9) * 32
    assert r.communication_bytes_batch == 4096 * r.communication_bytes_per_instance


def test_option_c_discloses_digests_and_saves_the_path():
    a = auth.cost_option("a", OPENINGS, auth.SHARING["bench24"], 4096, REL)
    c = auth.cost_option("c", OPENINGS, auth.SHARING["bench24"], 4096, REL)
    assert c.hash_invocations_per_instance < a.hash_invocations_per_instance
    assert c.verifier_hash_invocations_per_instance > 0 and a.verifier_hash_invocations_per_instance == 0
    assert "digests disclosed" in c.hiding_assumption
    # the W row alone is 50 of the 50.7 amortised compressions: sharing x/y does not rescue SHA-256
    assert round(c.hash_invocations_by_type["w:leaf"], 1) == 50.0


def test_depth_for_candidate_a():
    """Merkle-Damgard chains a leaf's blocks: option (a) puts the W leaf's (50 + 40) x 192 = 17 280 layers on A's
    critical path unless the chaining states are committed as hints (then 384 layers, +256 committed bits per block)."""
    a = auth.cost_option("a", OPENINGS, auth.SHARING["none"], 4096, REL)
    assert a.depth_layers_chained == (50 + 40) * 192 == 17_280 and a.depth_layers_hinted == 2 * 192
    d = auth.cost_option("d", OPENINGS, auth.SHARING["none"], 4096, REL)
    assert d.depth_layers_chained == (86 + 9) * 90 == 8_550 and d.depth_layers_hinted == 180
    b = auth.cost_option("b", OPENINGS, auth.SHARING["none"], 4096, REL)
    assert b.depth_layers_chained == 0


def test_killing_question_and_report_shape():
    rep = auth.build_report(4096)
    assert rep["what_is_private"]["costed_first"] == "reading P"
    assert rep["killing_question"]["answer"].startswith("yes")
    assert {r["option"] for r in rep["options"]} == set(auth.OPTIONS)
    assert len(rep["options"]) == 12
    for r in rep["options"]:
        if r["option"] in ("a", "c"):
            assert r["fraction_committed_a"] > 10 and r["fraction_committed_b"] > 20
        if r["option"] == "d":
            assert r["fraction_committed_b"] < 2 and r["fraction_mul_gates_a"] < 4


def test_main_prints_table(capsys):
    auth.main(["64"])
    out = capsys.readouterr().out
    assert "3073 references into 3 leaves" in out and "option" in out and "bench24" in out
