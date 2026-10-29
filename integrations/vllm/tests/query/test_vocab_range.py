"""The vocabulary-range policy (`verity_vllm.query.vocab_range`) and its source: `EmbeddingShard_v1` commits exactly the shifted id and the
two range bits (no recompute); `values` is the circuit's arithmetic on every edge id; off, a TP rank's manifest is byte-identical; on, one
`vocab_range` identity per (step, embedding site) on the site's op path with the rank's own range; the plan, coverage and form (B) know the
family.  The source: tests/acquire/test_vocab_range_source.py."""
from __future__ import annotations

import json

import pytest

from tests.check.test_sampled_replay_v2_addresses import _tp_rank_program, _v2_manifest
from verity_vllm.query import vocab_range as VR
from verity_vllm.query.manifest import format as fmt

V, VS = 16, 8


@pytest.mark.parametrize("VS_,H,START", [(8, 32, 0), (8, 32, 8), (24576, 4, 24576)])
def test_the_shard_commits_the_shifted_id_and_both_range_bits(VS_, H, START):
    from verity.ml.definitions import word as W_

    G = W_.Graph(VR.shard_fn(VS_, H, START))
    cut = W_.units(G, 16, 32)["cut"]
    assert cut.ok and G.recomputed == [], cut.codes
    assert VR.slots(VS_, H, START) == (1, 3, 5)
    rule = W_.unit_rule(VR.shard_fn(VS_, H, START))
    assert rule["committed_interior_words"] == VR.WORDS and not rule["violations"]


def test_values_are_the_irs_on_every_edge_id():
    from verity_vllm.properties.vocab_range_exactness import edge_ids, ir_values

    for start in (0, VS):
        for tok in edge_ids(start, VS, V) + [-1, 2**31 - 1, 2**31, 2**32 - 1]:
            assert VR.values(tok, start, VS) == ir_values(VS, start, tok), (start, tok)
    assert VR.values(VS, VS, VS) == (0, 1, 1) and VR.values(VS - 1, VS, VS) == (-1, 0, 1) and VR.values(2 * VS, VS, VS) == (VS, 1, 0)


def _rank_manifest(rank: int, on: bool | None):
    """Rank `rank`'s manifest: under the policy (`on`), with no taps given (None), or with an empty tap list (False)."""
    res = {"program": {"digest": f"{rank}" * 64, "function": "Derived[vLLMRequest[Test]]"}, "wrapper": "request", "LP": 2, "T": 1,
           "target": {"declared": {"compute_capability": [8, 9]}}, "attention_impls_observed": {}, "tp": {"world": 2, "rank": rank}}
    return _v2_manifest(_tp_rank_program(rank), res, workload={"sweep": {"row_id": "tp2-df", "batch": 1, "arrive_steps": [0]}}, rank=rank, ranks=2,
                        **({} if on is None else {"taps": (VR.POLICY,) if on else ()}))


@pytest.mark.parametrize("rank", [0, 1])
def test_a_tp_rank_commits_its_range_values_at_the_embedding_site(rank):
    off, on = _rank_manifest(rank, False), _rank_manifest(rank, True)
    assert json.dumps(off, sort_keys=True) == json.dumps(_rank_manifest(rank, None), sort_keys=True), "off is byte-identical"
    rows = [x for x in on["identities"] if x["family"] == VR.FAMILY]
    assert [(x["step"], x["op_path"], x["output_member"], x["element_range"], x["rows"], x["width"], x["dtype"], x["start"], x["shard_rows"])
            for x in rows] == [(0, "model.embed_tokens", VR.MEMBER, [0, 6], 2, 3, "i32", rank * VS, VS),
                               (1, "model.embed_tokens", VR.MEMBER, [0, 3], 1, 3, "i32", rank * VS, VS)]
    sites = {x["op_path"] for x in on["identities"] if x["output_member"] in ("in", f"part_rank{rank}") and x["op_path"].endswith("embed_tokens")}
    assert sites == {"model.embed_tokens"}, "the values sit on the op path of the site's ids and partial"
    strip = lambda m: sorted(json.dumps(x, sort_keys=True) for x in m["identities"] if x["family"] != VR.FAMILY)  # noqa: E731
    assert strip(on) == strip(off) and on["complete"]
    assert VR.policy_of(on) and not VR.policy_of(off) and on["query"]["vocab_range"]["words"] == 9
    assert VR.manifest_sites(on) == {"model.embed_tokens": {"start": rank * VS, "shard_rows": VS}}


def test_manifest_sites_refuses_two_ranges_at_one_site():
    man = _rank_manifest(0, True)
    man["identities"].append(dict(next(x for x in man["identities"] if x["family"] == VR.FAMILY), start=VS, step=7))
    with pytest.raises(ValueError, match="two ranges"):
        VR.manifest_sites(man)


def test_the_family_is_in_every_vocabulary():
    from verity_vllm.acquire import plan
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay import coverage
    from verity.ml.definitions import word as W_

    assert VR.FAMILY in fmt.FAMILIES and plan.ROUTE_OF_FAMILY[VR.FAMILY] == "vocab_tap" and VR.FAMILY in coverage.QUERY_FAMILY_MECHANISM
    entry = {"identity": {"op_path": "model.embed_tokens", "output_member": VR.MEMBER}, "name": f"model.embed_tokens/{VR.MEMBER}"}
    assert OC._family_of(entry) == "hidden_stream"
    g = {"family": "EmbeddingShard_v1", "statics": {"VS": 8, "H": 32, "START": 8}, "definition": "EmbeddingShard_v1{VS=8,H=32,START=8}",
         "calls": 3, "where": "model.embed_tokens"}
    assert W_.check_calls([g], acquired=VR.TAP_WORDS)["acquired_interior_words"] == 9
