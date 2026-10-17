import hashlib

from verity_proof_throughput import instances as I
from verity_proof_throughput.workloads import ladder, workload

W = workload("bf16-k6144")


def test_the_manifest_pins_its_ladder_from_its_lanes():
    m = I.load(W)
    assert (m["schema"], m["dataset"], m["tier"], m["n"]) == (I.SCHEMA, I.DATASET, W.name, W.n_max)
    assert len(m["lanes"]) == W.n_max and m["recipe"]["definitions"] == W.definitions()
    assert m["prefixes"] == {str(b): I.prefix_digest(m["lanes"], b) for b in ladder(W.n_max)}


def test_a_recomputed_prefix_matches_the_manifest():
    got = I.check_prefix(W, 4)
    assert got["ok"] and got["prefix"] == I.load(W)["prefixes"]["4"]
    assert got["manifest_sha256"] == hashlib.sha256(I.manifest_path(W).read_bytes()).hexdigest()


def test_lane_digests_cover_every_word_in_order():
    a = I.lane_digests([[1, 2], [3, 4]], [[5, 6]])
    assert a[0] == hashlib.sha256(I.LANE_TAG + bytes([1, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0])).hexdigest()
    assert a != I.lane_digests([[1, 2], [3, 4]], [[5, 7]])
    assert I.prefix_digest(a, 1) != I.prefix_digest(a, 2)


def test_a_result_names_a_prefix_of_the_tier():
    ref = I.instances_ref(W, 64)
    assert ref == {"dataset": I.DATASET, "tier": W.name, "range": [0, 64], "manifest_sha256": I.manifest_sha256(W)}
